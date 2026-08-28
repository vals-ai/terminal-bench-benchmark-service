"""Grading in a separate, network-blocked sandbox.

For datasets whose tasks declare ``verifier.environment_mode = "separate"``:
the grader and its dependencies live in the task's verifier image, and the
grader must see neither the agent's environment nor the network. The agent is
root in its own sandbox for its whole phase, so anything it produces is treated
as hostile input on the way across.
"""

from __future__ import annotations

import hashlib
import shlex
from collections.abc import Sequence
from posixpath import dirname
from dataclasses import dataclass
from typing import Any

REWARD_PATH = "/logs/verifier/reward.txt"
GRADE_COMMAND = "bash /tests/test.sh"
CONVENTION_ARTIFACT_DIR = "/logs/artifacts"
VERIFIER_CREATE_TIMEOUT_SECONDS = 600
# Idleness is counted in sandbox events, not process liveness, so a grader that
# runs quietly must outlast its own timeout by this margin.
VERIFIER_AUTO_STOP_MARGIN_MINUTES = 30
# An artifact passes through this process's memory, and the container is small.
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
# Ratios above a thousand to one are easy to produce, so the packed bound alone
# would let a small archive fill the verifier's disk.
MAX_EXPANDED_ARTIFACT_BYTES = 1024 * 1024 * 1024
MAX_ARTIFACT_MEMBERS = 200_000
MAX_CONCURRENT_VERIFIER_CREATES = 4
# Creating the verifier and carrying artifacts into it, before the grader runs.
PREPARE_TIMEOUT_SECONDS = 1800.0
PRESENT = "PRESENT"
# tar pads a file that shrank while being read out to its listed length, so
# the archive holds a right-sized member with a fabricated tail.
SHRANK_MARKER = "File shrank"
# Concurrent artifact transfers per process. Nothing else bounds concurrent
# evaluations on this path, and each transfer holds roughly three times the
# packed size while the bytes pass from one sandbox to the other.
MAX_CONCURRENT_TRANSFERS = 2


class UnsupportedArtifactError(ValueError):
    """A declared artifact this runtime cannot honour faithfully."""


class VerifierEnvironmentError(RuntimeError):
    """The grading environment could not be built, run, or read back.

    Distinct from a failed submission: this is a grading fault and must not be
    recorded as a zero for the model.
    """


@dataclass(frozen=True)
class ArtifactSpec:
    """One declared artifact to carry from the agent environment to the verifier."""

    source: str
    # Compose-based tasks name the sidecar service that owns the path. This
    # runtime serves single-container tasks, so a service-scoped artifact means
    # the task needs the compose runtime, not that the file is missing.
    service: str | None = None

    @classmethod
    def parse(cls, entry: str | dict[str, Any]) -> ArtifactSpec:
        if isinstance(entry, str):
            return cls(source=entry)
        source = entry.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError(f"Artifact entry has no usable source: {entry!r}")
        # `destination` and `exclude` change what the verifier is supposed to
        # see. Carrying such an entry as if it were a plain source would grade
        # the opposite of what the task asked for, so refuse it instead.
        unsupported = sorted(set(entry) - {"source", "service"})
        if unsupported:
            raise UnsupportedArtifactError(
                f"Artifact {source} uses unsupported keys ({', '.join(unsupported)})"
            )
        service = entry.get("service")
        return cls(source=source, service=service if isinstance(service, str) else None)


def parse_artifacts(entries: Sequence[str | dict[str, Any]] | None) -> list[ArtifactSpec]:
    """Parse a task definition's ``artifacts`` list."""
    return [ArtifactSpec.parse(entry) for entry in entries or []]


def local_artifacts(artifacts: Sequence[ArtifactSpec]) -> list[ArtifactSpec]:
    """Artifacts that live in the agent's own container."""
    return [artifact for artifact in artifacts if artifact.service is None]


def sidecar_artifacts(artifacts: Sequence[ArtifactSpec]) -> list[ArtifactSpec]:
    """Artifacts owned by a compose sidecar, which this runtime cannot collect."""
    return [artifact for artifact in artifacts if artifact.service is not None]


def prepare_logs_command() -> str:
    """Create the directories graders write into, without relaxing them.

    Upstream graders assume the harness has already made ``/logs/verifier``;
    several write into it without creating it first. Some verifier images
    deliberately harden those directories to 700 because the grader runs the
    agent's code as an unprivileged user, so this only creates what is missing
    and never chmods a tree the image set up on purpose.

    The verifier directory is removed first, and this runs after the agent's
    artifacts have been unpacked: an archive that planted a file, a directory or
    a symlink there cannot survive to catch the grader's reward.
    """
    return (
        "mkdir -p /logs/verifier /logs/agent " + CONVENTION_ARTIFACT_DIR + " && "
        "find /logs/verifier -mindepth 1 -delete"
    )


def reject_irregular_members_command(stage: str) -> str:
    """Fail when a staged archive holds anything the submission has no business carrying.

    The archive is produced by tar running as root in the agent's sandbox, so
    the agent can replace tar itself and emit any member type or mode it likes.
    Three things are refused:

    One thing is refused: anything that is not a regular file or a directory. `pack_command` selects files and
    directories explicitly, so a symlink, device or socket in the archive means
    the archive was not written by that command -- the agent is root in its own
    sandbox and can replace tar. A symlink is the dangerous one: copied
    faithfully into the verifier it can point the grader's own input at the
    grader's answer key.

    Hard links are left alone. GNU tar sanitises link targets the way it
    sanitises member names, so a link pointing out of the archive is refused by
    tar itself; within one submission tree they alias each other and nothing
    else, and real build tooling produces them. Every verifier image in this
    dataset is Debian, Ubuntu or conda based -- if one ever ships BusyBox tar,
    that reasoning has to be revisited.

    `find`'s status is captured through an assignment rather than tested
    directly: `test` sees only the string a command substitution produced, so a
    `find` that failed before printing anything would otherwise pass the gate.
    """
    quoted = shlex.quote(stage)
    return f'irregular=$(find {quoted} ! -type f ! -type d -print -quit) && test -z "$irregular"'


def strip_privileged_bits_command(stage: str) -> str:
    """Clear set-user-ID and set-group-ID bits from a staged submission.

    Extraction runs as root, so a member arriving at mode 04755 would be owned
    by root. Seventeen of these graders deliberately drop to an unprivileged
    user before running the agent's code; a setuid binary in the submission
    would hand that code a way back to root, and with it the answer key. The
    bits are stripped rather than refused because the file itself is ordinary
    submission data.
    """
    return f"chmod -R a-s {shlex.quote(stage)}"


def read_reward_command() -> str:
    return f"cat {REWARD_PATH}"


def parse_reward(raw: str) -> float:
    """Read the verifier's reward, rejecting anything outside [0, 1].

    The value is meaned into a percentage downstream, so an out-of-range reward
    would silently inflate the whole board rather than fail.
    """
    value = float(raw.strip().splitlines()[-1])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"reward {value} is outside [0, 1]")
    return value


def verifier_sandbox_name(task_id: str, run_id: str, attempt: str) -> str:
    """A provider-safe name for one grading attempt.

    The attempt is part of the name because Daytona resolves a name conflict by
    returning the existing sandbox: without it, a second evaluation of the same
    agent sandbox would silently grade against the first attempt's artifacts.
    """
    parts = [_safe(task_id), _safe(run_id), _safe(attempt)]
    return f"tb-verifier-{'-'.join(parts)}"[:120]


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in value)


def auto_stop_minutes(verifier_timeout_seconds: float) -> int:
    """Idle shutdown that outlasts the task's own grading budget."""
    return int(verifier_timeout_seconds // 60) + VERIFIER_AUTO_STOP_MARGIN_MINUTES

def artifact_archive_name(source: str) -> str:
    """A collision-free temp archive name for one artifact path."""
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    return f"tb-artifact-{digest}.tar.gz"


def exists_command(source: str) -> str:
    """Report whether the agent produced the artifact.

    The verdict is the printed word, not the exit status: a non-zero exit means
    the check itself failed, which is a grading fault rather than a submission
    the agent never made.
    """
    quoted = shlex.quote(source)
    return f"if [ -e {quoted} ]; then echo PRESENT; else echo ABSENT; fi"


def pack_command(source: str, archive: str) -> str:
    """Archive one artifact in the agent's sandbox and print its packed size.

    Regular files, directories, and symlinks that resolve to a regular file --
    those are dereferenced, so a submission that links to its own output still
    carries it, while a dangling or non-regular link is left out rather than
    failing the pack. The member list is built before the archive so a failing
    `find` cannot yield a truncated archive that still reports success.

    A long-running agent can change the submission underneath the pack. A member
    that disappeared is skipped, one that grew is packed short and accepted, and
    one that shrank is padded with zeros to its listed length -- a fabricated
    tail the caller detects via `fabricated_content`, since tar reports it with
    the same exit 1 as growth. Diagnostics are folded into stdout so that check
    sees them.

    No size is reported here: the agent owns this sandbox's tooling, so the
    bounds are enforced on bytes the service holds and on the verifier's own tar.
    """
    quoted_source = shlex.quote(source)
    quoted_archive = shlex.quote(archive)
    members = shlex.quote(f"{archive}.members")
    return (
        f"set -e; find {quoted_source} \\( -type f -o -type d -o -xtype f \\) -print0 > {members}; "
        f"set +e; tar -czhf {quoted_archive} --null --no-recursion --ignore-failed-read "
        f"-T {members} 2>&1; status=$?; "
        f"rm -f {members}; "
        f'[ "$status" -le 1 ] || exit "$status"'
    )


def expanded_size_command(archive: str) -> str:
    """Print the archive's uncompressed byte total and its member count.

    Listing decompresses the stream without writing anything, so an archive that
    expands enormously is refused while it is still small.
    """
    quoted = shlex.quote(archive)
    return f"tar -tzvf {quoted} | awk '{{bytes += $3; members += 1}} END {{print bytes+0, members+0}}'"


def fabricated_content(pack_output: str) -> bool:
    """Whether packing padded a member with zeros to hide a shrinking file."""
    return SHRANK_MARKER in pack_output


def unpack_command(source: str, archive: str) -> str:
    """Place one artifact in the verifier at its original path.

    The archive is written by tar running in the agent's sandbox, where the
    agent is root, so its member list is entirely attacker-chosen. It is
    therefore never extracted at ``/``: it is unpacked into a staging directory
    and only the declared path is copied out of it, which discards any extra
    member aimed at the grader, its dependencies, or the reward file.

    The destination is removed first so nothing is written through a symlink
    left there by an earlier member, and ``cp -a`` keeps symlinks inside the
    submission as symlinks instead of following them.
    """
    quoted_source = shlex.quote(source)
    relative = shlex.quote(source.lstrip("/"))
    quoted_archive = shlex.quote(archive)
    stage = shlex.quote(f"{archive}.stage")
    parent = shlex.quote(dirname(source) or "/")
    return (
        f"rm -rf {stage} && mkdir -p {stage} && "
        f"tar -xzf {quoted_archive} -C {stage} --no-same-owner --no-same-permissions && "
        f"{reject_irregular_members_command(f'{archive}.stage')} && "
        f"{strip_privileged_bits_command(f'{archive}.stage')} && "
        f"test -e {stage}/{relative} && "
        f"mkdir -p {parent} && rm -rf {quoted_source} && "
        f"cp -a {stage}/{relative} {quoted_source} && "
        f"rm -rf {stage} {quoted_archive}"
    )

