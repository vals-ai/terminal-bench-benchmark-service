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


def test_cleanup_stops_dockerd_when_compose_down_fails() -> None:
    from terminal_bench_benchmark_service.compose_runtime import stop_compose_runtime

    sandbox = FailingCleanupSandbox()
    with pytest.raises(RuntimeError, match="compose failed"):
        asyncio.run(stop_compose_runtime("task/one", sandbox))  # type: ignore[arg-type]

    assert sandbox.commands[-1] == "pkill -TERM dockerd || true"
