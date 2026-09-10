"""Local Docker Compose contract tests for TBench4 runtime overlays."""

import asyncio
import json
import subprocess
from pathlib import Path

import pytest
from benchmark_service import ComposeSource, ImageSource
from benchmark_service.sandbox.types import ExecResult

from terminal_bench_benchmark_service.compose_runtime import (
    compose_runtime_source,
    runtime_compose_definition,
)


class FailingCleanupSandbox:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, command: str, **_kwargs: object) -> ExecResult:
        self.commands.append(command)
        if len(self.commands) == 1:
            return ExecResult(exit_code=1, output="compose failed")
        return ExecResult(exit_code=0, output="")


class StagingSandbox:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.uploads: dict[str, bytes] = {}

    async def exec(self, command: str, **_kwargs: object) -> ExecResult:
        self.commands.append(command)
        return ExecResult(exit_code=0, output="")

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        self.uploads[remote_path] = content


def test_compose_source_uses_a_provider_compatible_outer_image() -> None:
    source = compose_runtime_source("task/one", "example/main@sha256:" + "a" * 64, {"api": "example/api"})

    assert isinstance(source, ComposeSource)
    assert isinstance(source.outer, ImageSource)
    assert source.outer.image == (
        "docker:28.3.3-dind@sha256:a56b3bdde89315ed2cc0e4906e582b5033d93bf20d9cb9510c2cdd4e7f7690b1"
    )
    assert "task-one" in source.compose_command
    assert "runtime.json" in source.compose_command


def test_runtime_overlay_replaces_task_builds_with_pinned_images(tmp_path: Path) -> None:
    task_file = tmp_path / "docker-compose.yaml"
    runtime_file = tmp_path / "runtime.json"
    task_file.write_text(
        """
services:
  main:
    build:
      context: .
  api:
    build:
      context: ./api
""".strip()
        + "\n"
    )
    runtime_file.write_text(
        json.dumps(
            runtime_compose_definition(
                "example/main@sha256:" + "a" * 64,
                {"api": "example/api@sha256:" + "b" * 64},
                {"cpus": 2, "memory_mb": 4096},
            )
        )
    )

    result = subprocess.run(
        [
            "docker",
            "compose",
            "--project-name",
            "tbench4-overlay-test",
            "--project-directory",
            str(tmp_path),
            "--file",
            str(task_file),
            "--file",
            str(runtime_file),
            "config",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    config = json.loads(result.stdout)

    assert config["services"]["main"]["image"] == "example/main@sha256:" + "a" * 64
    assert config["services"]["api"]["image"] == "example/api@sha256:" + "b" * 64
    # Compose retains the source build metadata during merge, but the runtime
    # starts with --no-build so these pinned images are what actually run.
    assert config["services"]["main"]["build"]["context"] == str(tmp_path)
    assert config["services"]["api"]["build"]["context"] == str(tmp_path / "api")
    assert any(
        volume["source"] == "/bundle" and volume["target"] == "/bundle"
        for volume in config["services"]["main"]["volumes"]
    )
    assert "privileged" not in config["services"]["main"]


def test_runtime_overlay_passes_allocated_gpu_into_nested_task() -> None:
    runtime = runtime_compose_definition(
        "example/main@sha256:" + "a" * 64,
        {},
        {"cpus": 2, "memory_mb": 4096, "gpus": 1, "gpu_types": ["H100"]},
    )

    assert runtime["services"]["main"]["privileged"] is True


def test_staging_uses_minimal_compose_for_tasks_without_sidecars(tmp_path: Path) -> None:
    from terminal_bench_benchmark_service.compose_runtime import _stage_files

    sandbox = StagingSandbox()
    asyncio.run(
        _stage_files(
            tmp_path / "environment",
            "example/main@sha256:" + "a" * 64,
            {},
            {"cpus": 1, "memory_mb": 1024},
            sandbox,  # type: ignore[arg-type]
        )
    )

    assert sandbox.uploads["/terminal-bench/task.json"] == b'{"services":{"main":{}}}\n'


class UnmappableOwnerSandbox(StagingSandbox):
    """Nested dockerd that cannot apply the task image's layers but accepts everything else."""

    async def exec(self, command: str, **_kwargs: object) -> ExecResult:
        self.commands.append(command)
        if command.startswith("docker pull "):
            return ExecResult(
                exit_code=1,
                output='failed to register layer: failed to Lchown "/app/data" for UID 197609, GID 197121: '
                "lchown /app/data: invalid argument",
            )
        if command.startswith("crane config "):
            return ExecResult(exit_code=0, output=json.dumps({"config": {"Env": ["PATH=/x"], "WorkingDir": "/app"}}))
        if command.endswith("config --services"):
            return ExecResult(exit_code=0, output="main\n")
        return ExecResult(exit_code=0, output="")


def test_unmappable_image_owners_are_rewritten_and_imported_locally(tmp_path: Path) -> None:
    from terminal_bench_benchmark_service.compose_runtime import start_compose_runtime

    sandbox = UnmappableOwnerSandbox()
    asyncio.run(
        start_compose_runtime(
            tmp_path,
            "atrx-vep-crispr",
            "example/main@sha256:" + "a" * 64,
            {},
            {"cpus": 2, "memory_mb": 4096},
            sandbox,  # type: ignore[arg-type]
        )
    )

    (import_command,) = [command for command in sandbox.commands if "docker import" in command]
    assert "crane export example/main@sha256:" in import_command
    assert "-c 'ENV PATH=/x' -c 'WORKDIR /app'" in import_command
    runtime = json.loads(sandbox.uploads["/terminal-bench/runtime.json"])
    assert runtime["services"]["main"]["image"] == "terminal-bench/main:reowned"
    assert runtime["services"]["main"]["pull_policy"] == "never"


def test_other_pull_failures_are_not_reimported(tmp_path: Path) -> None:
    from terminal_bench_benchmark_service.compose_runtime import start_compose_runtime

    class ManifestUnknownSandbox(StagingSandbox):
        async def exec(self, command: str, **_kwargs: object) -> ExecResult:
            self.commands.append(command)
            if command.startswith("docker pull "):
                return ExecResult(exit_code=1, output="manifest unknown")
            return ExecResult(exit_code=0, output="")

    sandbox = ManifestUnknownSandbox()
    with pytest.raises(RuntimeError, match="manifest unknown"):
        asyncio.run(
            start_compose_runtime(
                tmp_path,
                "task",
                "example/main@sha256:" + "a" * 64,
                {},
                {"cpus": 1, "memory_mb": 1024},
                sandbox,  # type: ignore[arg-type]
            )
        )
    assert not any("docker import" in command for command in sandbox.commands)


def test_reown_script_rewrites_only_owners_outside_the_id_maps(tmp_path: Path) -> None:
    import io
    import subprocess
    import tarfile

    from terminal_bench_benchmark_service.compose_runtime import _REOWN_SCRIPT_SOURCE

    (tmp_path / "uid_map").write_text("0 886432 65536\n")
    (tmp_path / "gid_map").write_text("0 886432 65536\n")
    script = tmp_path / "reown.py"
    script.write_text(_REOWN_SCRIPT_SOURCE)
    source = io.BytesIO()
    with tarfile.open(fileobj=source, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, uid, gid in [("vep", 197609, 197121), ("huge", 3_000_000, 3_000_000), ("mine", 1000, 65535)]:
            info = tarfile.TarInfo(name)
            info.uid, info.gid, info.uname, info.gname = uid, gid, "juancmunoz", ""
            info.size = 5
            archive.addfile(info, io.BytesIO(b"hello"))
        link = tarfile.TarInfo("link")
        link.type, link.linkname, link.uid = tarfile.SYMTYPE, "vep", 197609
        archive.addfile(link)

    result = subprocess.run(
        ["python3", str(script), str(tmp_path / "uid_map"), str(tmp_path / "gid_map")],
        input=source.getvalue(),
        capture_output=True,
        check=True,
    )

    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as rewritten:
        members = {member.name: member for member in rewritten.getmembers()}
        assert (members["vep"].uid, members["vep"].gid, members["vep"].uname) == (0, 0, "")
        assert (members["huge"].uid, members["huge"].gid) == (0, 0)
        assert (members["mine"].uid, members["mine"].gid) == (1000, 65535)
        assert members["link"].issym() and members["link"].linkname == "vep" and members["link"].uid == 0
        assert rewritten.extractfile("vep").read() == b"hello"  # type: ignore[union-attr]


def test_cleanup_stops_dockerd_when_compose_down_fails() -> None:
    from terminal_bench_benchmark_service.compose_runtime import stop_compose_runtime

    sandbox = FailingCleanupSandbox()
    with pytest.raises(RuntimeError, match="compose failed"):
        asyncio.run(stop_compose_runtime("task/one", sandbox))  # type: ignore[arg-type]

    assert sandbox.commands[-1] == "pkill -TERM dockerd || true"
