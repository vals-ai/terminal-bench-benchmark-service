import asyncio
import os
import subprocess
import tomllib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from benchmark_service.context import tool_context_scope
from benchmark_service.sandbox import ExecResult
from benchmark_service.schemas import MCPToolContext, NativeToolContext, ToolContext

import terminal_bench_benchmark_service.benchmark_service as service_module
from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark
from tests.test_runtime_lifecycle import FakeSandbox


@pytest.mark.parametrize(
    ("task_id", "context"),
    [
        ("medical-claims-processing", MCPToolContext(mode="mcp", model="openai/test-model")),
        ("medical-claims-processing", NativeToolContext(mode="native")),
        ("html-js-filter", MCPToolContext(mode="mcp", model="openai/test-model")),
    ],
)
def test_codex_registers_the_medical_sidecar_and_preserves_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task_id: str, context: ToolContext
) -> None:
    config = tmp_path / ".codex/config.toml"
    config.parent.mkdir()
    initial = (
        'model = "aurora-alpha"\n[feedback]\nenabled = true\n[mcp_servers.existing]\nurl = "http://existing/mcp"\n'
    )
    config.write_text(initial)

    class ConfigSandbox(FakeSandbox):
        async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> ExecResult:
            if "mcp_servers.playwright" not in command:
                return await super().exec(command, cwd=cwd, timeout=timeout)
            result = subprocess.run(
                ["bash", "-c", command], env={**os.environ, "HOME": str(tmp_path)}, capture_output=True, text=True
            )
            return ExecResult(exit_code=result.returncode, output=result.stdout + result.stderr)

    monkeypatch.setattr(service_module, "start_compose_runtime", AsyncMock())
    monkeypatch.setattr(service_module, "ComposeSandbox", lambda sandbox, _source: sandbox)
    service = asyncio.run(TerminalBenchBenchmark.create())

    async def setup() -> None:
        with tool_context_scope(context):
            _ = [chunk async for chunk in service.setup_task(task_id, ConfigSandbox(), dataset="terminal-bench-4.0")]

    asyncio.run(setup())
    expected = tomllib.loads(initial)
    if task_id == "medical-claims-processing" and context.mode == "mcp":
        expected["mcp_servers"]["playwright"] = {"url": "http://playwright-mcp:3080/mcp", "required": True}
    assert tomllib.loads(config.read_text()) == expected
