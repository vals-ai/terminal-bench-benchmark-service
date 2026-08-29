"""Contract for grading in a separate, network-blocked sandbox."""

import pytest

from terminal_bench_benchmark_service import isolated_verifier


def test_prepare_creates_the_directories_graders_assume() -> None:
    command = isolated_verifier.prepare_logs_command()

    # Several upstream graders write straight into /logs/verifier without
    # creating it, because the harness is expected to have done so.
    assert "/logs/verifier" in command
    assert isolated_verifier.CONVENTION_ARTIFACT_DIR in command


def test_sandbox_name_is_provider_safe_and_bounded() -> None:
    assert isolated_verifier.verifier_sandbox_name("some/task.id", "run id!", "ab12") == (
        "tb-verifier-some-task-id-run-id--ab12"
    )
    assert len(isolated_verifier.verifier_sandbox_name("x" * 200, "y" * 200, "ab12")) <= 120


def test_each_attempt_gets_its_own_sandbox_name() -> None:
    """Daytona returns the existing sandbox on a name conflict, so names cannot repeat."""
    first = isolated_verifier.verifier_sandbox_name("3x2pt-inference", "run-1", "ab12")
    second = isolated_verifier.verifier_sandbox_name("3x2pt-inference", "run-1", "cd34")

    assert first != second


def test_auto_stop_outlasts_the_task_verifier_timeout() -> None:
    """A grader that runs quietly for hours must not be reaped as idle."""
    assert isolated_verifier.auto_stop_minutes(12000) > 12000 / 60
    assert isolated_verifier.auto_stop_minutes(120) >= isolated_verifier.VERIFIER_AUTO_STOP_MARGIN_MINUTES


def test_pack_carries_resolving_symlinks_but_not_dangling_ones() -> None:
    """A submission that links to its own output must still carry it."""
    command = isolated_verifier.pack_command("/app/out", "/tmp/a.tar.gz")

    assert "-type f -o -type d -o -xtype f" in command
    assert "-czhf" in command, "a listed symlink is stored as its target's content"


def test_pack_reports_no_size_of_its_own() -> None:
    """Bounds are enforced on bytes the service holds, not on the agent's word."""
    assert "wc -c" not in isolated_verifier.pack_command("/app/out", "/tmp/a.tar.gz")


def test_existence_is_reported_in_output_not_exit_status() -> None:
    """A failed check is a grading fault, not a submission the agent never made."""
    command = isolated_verifier.exists_command("/app/out")

    assert isolated_verifier.PRESENT in command
    assert "ABSENT" in command


def test_pack_tolerates_a_file_changing_while_it_is_read() -> None:
    """An agent process still writing during the pack must not fault the grade."""
    command = isolated_verifier.pack_command("/app/out", "/tmp/a.tar.gz")

    # tar exit 1 means a file changed; the archive is complete.
    assert '[ "$status" -le 1 ] || exit "$status"' in command


def test_pack_does_not_take_its_status_from_the_last_pipeline_stage() -> None:
    """A find that fails part-way must not yield a truncated archive reporting success."""
    command = isolated_verifier.pack_command("/app/out", "/tmp/a.tar.gz")

    assert command.startswith("set -e; find ")
    assert "| tar" not in command


def test_unsupported_artifact_keys_are_refused() -> None:
    """`destination` and `exclude` change what the verifier should see."""
    with pytest.raises(isolated_verifier.UnsupportedArtifactError, match="destination"):
        isolated_verifier.parse_artifacts([{"source": "/logs/artifacts", "destination": "elsewhere"}])


def test_parse_artifacts_splits_local_from_sidecar() -> None:
    artifacts = isolated_verifier.parse_artifacts(
        ["/app/result.npz", {"source": "/app/sim/probe.json", "service": "sim"}]
    )

    assert [a.source for a in isolated_verifier.local_artifacts(artifacts)] == ["/app/result.npz"]
    assert [(a.service, a.source) for a in isolated_verifier.sidecar_artifacts(artifacts)] == [
        ("sim", "/app/sim/probe.json")
    ]


def test_parse_artifacts_handles_a_task_that_declares_none() -> None:
    assert isolated_verifier.parse_artifacts(None) == []


def test_parse_artifacts_rejects_an_entry_with_no_source() -> None:
    with pytest.raises(ValueError, match="no usable source"):
        isolated_verifier.parse_artifacts([{"service": "sim"}])


def test_a_task_with_sidecar_artifacts_is_refused_rather_than_mis_graded() -> None:
    """A compose task's output lives where this runtime cannot reach it."""
    artifacts = isolated_verifier.parse_artifacts([{"source": "/app/out.json", "service": "sim"}])

    # Nothing is gradeable from the agent container alone, so the caller raises
    # rather than grading an empty submission as a zero.
    assert isolated_verifier.local_artifacts(artifacts) == []
    assert isolated_verifier.sidecar_artifacts(artifacts) == artifacts


def test_unpack_never_extracts_at_the_filesystem_root() -> None:
    """The archive is written by the agent, so its members are attacker-chosen."""
    command = isolated_verifier.unpack_command("/app/result.npz", "/tmp/a.tar.gz")

    assert " -C / " not in command
    assert "-C /tmp/a.tar.gz.stage" in command


def test_unpack_copies_only_the_declared_path_out_of_staging() -> None:
    command = isolated_verifier.unpack_command("/app/out", "/tmp/a.tar.gz")

    # Extra members land in staging and are discarded with it.
    assert "cp -a /tmp/a.tar.gz.stage/app/out /app/out" in command
    assert command.startswith("rm -rf /tmp/a.tar.gz.stage")
    assert "rm -rf /tmp/a.tar.gz.stage /tmp/a.tar.gz" in command


def test_unpack_clears_the_destination_before_writing_it() -> None:
    """A symlink left at the destination must not be written through."""
    command = isolated_verifier.unpack_command("/app/out", "/tmp/a.tar.gz")

    assert "rm -rf /app/out" in command
    assert command.index("rm -rf /app/out") < command.index("cp -a")


def test_pack_and_unpack_quote_hostile_paths() -> None:
    hostile = "/app/a b; rm -rf /"
    packed = isolated_verifier.pack_command(hostile, "/tmp/a.tar.gz")
    unpacked = isolated_verifier.unpack_command(hostile, "/tmp/a.tar.gz")

    for command in (packed, unpacked):
        assert "'/app/a b; rm -rf /'" in command or "'app/a b; rm -rf /'" in command
        assert "; rm -rf / &&" not in command


def test_prepare_logs_does_not_relax_hardened_directories() -> None:
    """Some images set /logs/verifier to 0700 on purpose; recreating it loses that."""
    command = isolated_verifier.prepare_logs_command()

    assert "chmod" not in command
    assert "rm -rf /logs/verifier" not in command
    assert "find /logs/verifier -mindepth 1 -delete" in command


def test_staged_archive_rejects_members_packing_could_not_produce() -> None:
    """A symlink member means the agent replaced tar; it can redirect the grader."""
    command = isolated_verifier.unpack_command("/app/out", "/tmp/a.tar.gz")

    assert "! -type f ! -type d" in command
    # The check must gate the copy, not follow it.
    assert command.index("! -type f ! -type d") < command.index("cp -a")


def test_find_failure_fails_the_gate() -> None:
    """`test` sees only the string a command substitution produced, not its status."""
    command = isolated_verifier.reject_irregular_members_command("/tmp/s")

    assert command.startswith("irregular=$(find ")
    assert command.endswith('&& test -z "$irregular"')


def test_setuid_bits_are_stripped_before_the_grader_runs() -> None:
    """Graders drop privileges to run the submission; a setuid file undoes that."""
    command = isolated_verifier.unpack_command("/app/out", "/tmp/a.tar.gz")

    assert "chmod -R a-s /tmp/a.tar.gz.stage" in command
    assert command.index("chmod -R a-s") < command.index("cp -a")
    assert "--no-same-permissions" in command


def test_reward_outside_the_unit_interval_is_refused() -> None:
    """A reward is meaned into a percentage, so 100 would inflate the board."""
    assert isolated_verifier.parse_reward("1.0\n") == 1.0
    assert isolated_verifier.parse_reward("0\n") == 0.0
    for bad in ("100", "-1", "1.5"):
        with pytest.raises(ValueError, match="outside"):
            isolated_verifier.parse_reward(bad)


def test_pack_skips_a_member_that_disappeared() -> None:
    """Writing to a temp name and renaming it removes a file between listing and packing."""
    assert "--ignore-failed-read" in isolated_verifier.pack_command("/app/out", "/tmp/a.tar.gz")


def test_a_shrinking_member_is_detected_as_fabricated() -> None:
    """tar pads it to its listed length, so the grader sees a right-sized file of zeros."""
    assert isolated_verifier.fabricated_content("tar: /app/out/f: File shrank by 40 bytes; padding with zeros")
    assert not isolated_verifier.fabricated_content("tar: /app/out/f: file changed as we read it")
    assert not isolated_verifier.fabricated_content("")


def test_expanded_size_is_measured_without_extracting() -> None:
    """A packed bound does not bound what unpacking writes: ratios beat it easily."""
    command = isolated_verifier.expanded_size_command("/tmp/a.tar.gz")

    # Not from `tar -tv` columns: its owner and group come from the header the
    # agent wrote, and a name holding a space shifts the size column away.
    assert "-tzvf" not in command
    assert "gzip -dc /tmp/a.tar.gz" in command
    assert f"head -c {isolated_verifier.MAX_EXPANDED_ARTIFACT_BYTES + 1}" in command
    assert "set -o pipefail" in command  # a truncated archive must not read as empty
    # Listing only: nothing is written while measuring.
    assert "-x" not in command


