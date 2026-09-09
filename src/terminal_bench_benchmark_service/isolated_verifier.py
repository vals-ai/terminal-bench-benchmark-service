"""Grading in a separate, network-blocked sandbox.

For datasets whose tasks declare ``verifier.environment_mode = "separate"``:
the grader and its dependencies live in the task's verifier image, and the
grader must see neither the agent's environment nor the network. The agent is
root in its own sandbox for its whole phase, so anything it produces is treated
as hostile input on the way across.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Sequence
from posixpath import dirname
from dataclasses import dataclass
from typing import Any, cast

REWARD_PATH = "/logs/verifier/reward.txt"
REWARD_JSON_PATH = "/logs/verifier/reward.json"
GRADE_COMMAND = "bash /tests/test.sh"
CONVENTION_ARTIFACT_DIR = "/logs/artifacts"
VERIFIER_CREATE_TIMEOUT_SECONDS = 600
# Idleness is counted in sandbox events, not process liveness, so a grader that
# runs quietly must outlast its own timeout by this margin.
VERIFIER_AUTO_STOP_MARGIN_MINUTES = 30
# An artifact passes through this process's memory. TBench4's checkpoint
# consolidation task produces a 186 MB safetensors file, so leave room for
# that valid artifact while keeping transfers bounded below the verifier's
# 1 GB expanded-artifact limit.
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
# Ratios above a thousand to one are easy to produce, so the packed bound alone
# would let a small archive fill the verifier's disk.
MAX_EXPANDED_ARTIFACT_BYTES = 1024 * 1024 * 1024
MAX_ARTIFACT_MEMBERS = 200_000
MAX_CONCURRENT_VERIFIER_CREATES = 4
# Creating the verifier and carrying artifacts into it, before the grader runs.
PREPARE_TIMEOUT_SECONDS = 1800.0
# All of a task's artifacts share one budget. One task declares fourteen, and
# per-artifact bounds alone would let preparation run for most of a day.
PREPARE_BUDGET_SECONDS = 5400.0
PRESENT = "PRESENT"
# tar pads a file that shrank while being read out to its listed length, so
# the archive holds a right-sized member with a fabricated tail.
SHRANK_MARKER = "File shrank"
# The SDK upload boundary accepts bytes, so one transfer necessarily holds one
# packed archive in process memory. Keep those bounded uploads serialized; the
# download side is spooled to disk before the single upload copy is made.
MAX_CONCURRENT_TRANSFERS = 1


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
    destination: str | None = None
    exclude: tuple[str, ...] = ()

    @classmethod
    def parse(cls, entry: str | dict[str, Any]) -> ArtifactSpec:
        if isinstance(entry, str):
            return cls(source=entry)
        source = entry.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError(f"Artifact entry has no usable source: {entry!r}")
        unsupported = sorted(set(entry) - {"source", "destination", "service", "exclude"})
        if unsupported:
            raise UnsupportedArtifactError(f"Artifact {source} uses unsupported keys ({', '.join(unsupported)})")
        service = entry.get("service")
        destination = entry.get("destination")
        if destination is not None and not isinstance(destination, str):
            raise ValueError(f"Artifact {source} has an invalid destination")
        exclude_value = entry.get("exclude", [])
        if not isinstance(exclude_value, list):
            raise ValueError(f"Artifact {source} has an invalid exclude list")
        exclude_values = cast(list[object], exclude_value)
        if not all(isinstance(pattern, str) for pattern in exclude_values):
            raise ValueError(f"Artifact {source} has an invalid exclude list")
        exclude_patterns = tuple(cast(str, pattern) for pattern in exclude_values)
        return cls(
            source=source,
            service=service if isinstance(service, str) else None,
            destination=destination,
            exclude=exclude_patterns,
        )


@dataclass(frozen=True)
class CollectSpec:
    """One verifier collect hook executed in the completed agent runtime."""

    command: str
    service: str = "main"
    timeout_sec: float = 60.0
    user: str | None = None


def parse_artifacts(entries: Sequence[str | dict[str, Any]] | None) -> list[ArtifactSpec]:
    """Parse a task definition's ``artifacts`` list."""
    return [ArtifactSpec.parse(entry) for entry in entries or []]


def parse_collect_hooks(entries: Sequence[dict[str, Any]] | None) -> list[CollectSpec]:
    """Parse Harbor verifier collect hooks without changing their commands."""
    hooks: list[CollectSpec] = []
    for entry in entries or []:
        unsupported = sorted(set(entry) - {"command", "service", "timeout_sec", "user"})
        if unsupported:
            raise ValueError(f"Collect hook uses unsupported keys ({', '.join(unsupported)})")
        command = entry.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError(f"Collect hook has no usable command: {entry!r}")
        service = entry.get("service", "main")
        if not isinstance(service, str) or not service:
            raise ValueError(f"Collect hook has an invalid service: {entry!r}")
        timeout = entry.get("timeout_sec", 60.0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"Collect hook has an invalid timeout_sec: {entry!r}")
        user = entry.get("user")
        if user is not None and (not isinstance(user, str) or not user):
            raise ValueError(f"Collect hook has an invalid user: {entry!r}")
        hooks.append(CollectSpec(command=command, service=service, timeout_sec=float(timeout), user=user))
    return hooks


def local_artifacts(artifacts: Sequence[ArtifactSpec]) -> list[ArtifactSpec]:
    """Artifacts that live in the agent's own container."""
    return [artifact for artifact in artifacts if artifact.service is None]


def sidecar_artifacts(artifacts: Sequence[ArtifactSpec]) -> list[ArtifactSpec]:
    """Artifacts owned by a compose sidecar."""
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
        "mkdir -p /logs/verifier /logs/agent " + CONVENTION_ARTIFACT_DIR + " && find /logs/verifier -mindepth 1 -delete"
    )


def reject_irregular_members_command(stage: str) -> str:
    """Fail when a staged archive holds anything the submission has no business carrying.

    The archive is produced by tar running as root in the agent's sandbox, so
    the agent can replace tar itself and emit any member type or mode it likes.
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
    """Print the reward, preferring ``reward.json`` over ``reward.txt`` as Harbor does."""
    return f"if [ -e {REWARD_JSON_PATH} ]; then cat {REWARD_JSON_PATH}; else cat {REWARD_PATH}; fi"


def parse_reward(raw: str) -> float:
    """Read the verifier's reward, rejecting anything outside [0, 1].

    ``reward.json`` holds ``{"reward": <number>}``; ``reward.txt`` holds a bare
    number on its last line. The value is meaned into a percentage downstream,
    so an out-of-range reward would silently inflate the whole board rather
    than fail.
    """
    text = raw.strip()
    if text.startswith("{"):
        rewards = cast(dict[str, object], json.loads(text))
        reward = rewards["reward"]
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise ValueError(f"reward {reward!r} is not a number")
        value = float(reward)
    else:
        value = float(text.splitlines()[-1])
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


def pack_command(source: str, archive: str, exclude: Sequence[str] = ()) -> str:
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

    ``exclude`` patterns are passed to tar, matching Harbor's artifact
    collection semantics. Patterns beginning with ``./`` are also passed
    without that prefix because the archive member names retain the declared
    absolute path after tar strips its leading slash.

    No size is reported here: the agent owns this sandbox's tooling, so the
    bounds are enforced on bytes the service holds and on the verifier's own tar.

    Compose sidecars (redis:alpine, kafka-native) ship BusyBox find and tar, so
    the member list is newline-separated and only GNU-specific flags that tar
    accepts are passed: BusyBox tar already skips a vanished member with exit 1.
    A name holding a newline would split into two missing members, so packing
    refuses it instead of shipping an archive without that file.
    """
    quoted_source = shlex.quote(source)
    quoted_archive = shlex.quote(archive)
    members = shlex.quote(f"{archive}.members")
    exclude_args = " ".join(
        f"--exclude={shlex.quote(pattern)}"
        for pattern in dict.fromkeys(pattern for value in exclude for pattern in (value, value.removeprefix("./")))
    )
    if exclude_args:
        exclude_args = f" {exclude_args}"
    return (
        f"set -e; "
        f'if [ -n "$(find {quoted_source} -name "$(printf \'*\\n*\')" -print -quit)" ]; then '
        'echo "an artifact file name contains a newline"; exit 2; fi; '
        f"find {quoted_source} \\( -type f -o -type d \\) -print > {members}; "
        f"find {quoted_source} -type l -exec test -f {{}} \\; -print >> {members}; "
        "set +e; ignore=$(tar --ignore-failed-read --version >/dev/null 2>&1 && echo --ignore-failed-read); "
        f"tar -czhf {quoted_archive}{exclude_args} --no-recursion $ignore "
        f"-T {members} 2>&1; status=$?; "
        f"rm -f {members}; "
        f'[ "$status" -le 1 ] || exit "$status"'
    )


def expanded_size_command(archive: str) -> str:
    """Print the archive's uncompressed byte total and member count.

    Measured from the decompressed stream, not from `tar -tv` columns: the
    owner and group fields in those come from the header the agent wrote, and a
    name containing a space shifts the size column out of position -- a 1.5 GiB
    member reads as zero. Reading stops one byte past the limit, so measuring a
    bomb costs no more than measuring a legitimate submission, and the members
    are only listed once the byte total is known to be sane.

    Written for POSIX sh. The verifier image is the task's, and `set -o
    pipefail` is a bashism that aborts dash outright, so tar's own status is
    taken directly rather than through a pipeline.
    """
    quoted = shlex.quote(archive)
    listing = shlex.quote(f"{archive}.members")
    return (
        f"bytes=$(gzip -dc {quoted} 2>/dev/null | head -c {MAX_EXPANDED_ARTIFACT_BYTES + 1} | wc -c); "
        f'if [ "$bytes" -gt {MAX_EXPANDED_ARTIFACT_BYTES} ]; then echo "$bytes 0"; exit 0; fi; '
        f"tar -tzf {quoted} > {listing} || exit 1; "
        f"members=$(wc -l < {listing}); rm -f {listing}; "
        'echo "$bytes $members"'
    )


def dir_symlink_command(source: str) -> str:
    """Find a symlinked directory in the artifact, which packing cannot follow.

    Its subtree would be left out of the archive with no error, and the grader
    would mark the model down for output it produced. ``test -d`` follows the
    link; BusyBox find has no ``-xtype``.
    """
    quoted = shlex.quote(source)
    return f'found=$(find {quoted} -type l -exec test -d {{}} \\; -print -quit) && test -z "$found"'


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
    left there by an earlier member, and ``cp -a`` copies modes and timestamps
    through unchanged; the staging gate has already rejected any symlink.
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
