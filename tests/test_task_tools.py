"""The service reports the tools of a task. It writes no agent configuration and adds no prompt note."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from benchmark_service import MCPServer
from benchmark_service.sandbox import ExecResult

from terminal_bench_benchmark_service import benchmark_service as service_module
from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark, task_mcp_servers

DATASET_PATHS = {
    "terminal-bench-2.0": "datasets/terminal-bench-2",
    "terminal-bench-2.1": "datasets/terminal-bench-2.1/tasks",
    "terminal-bench-4.0": "datasets/terminal-bench-4/tasks",
    "terminal-bench-4.1": "datasets/terminal-bench-4.1/tasks",
}
PLAYWRIGHT = MCPServer(name="playwright", transport="sse", url="http://playwright-mcp:3080/sse")


@pytest.fixture(scope="module")
def benchmark() -> TerminalBenchBenchmark:
    return asyncio.run(TerminalBenchBenchmark.create())


def test_stdio_and_http_servers_keep_every_field() -> None:
    servers = task_mcp_servers(
        "t",
        {
            "mcp_servers": [
                {"name": "fs", "transport": "stdio", "command": "npx", "args": ["-y", "srv"]},
                {"name": "docs", "transport": "http", "url": "http://docs:1/mcp", "headers": {"A": "${TOKEN}"}},
            ]
        },
    )
    assert [s.model_dump(mode="json", exclude_defaults=False) for s in servers] == [
        {"name": "fs", "transport": "stdio", "url": None, "command": "npx", "args": ["-y", "srv"]},
        {
            "name": "docs",
            "transport": "streamable-http",
            "url": "http://docs:1/mcp",
            "command": None,
            "args": [],
            "headers": {"A": "${TOKEN}"},
        },
    ]


def test_task_with_no_servers_gets_an_empty_list() -> None:
    assert task_mcp_servers("t", {}) == []
    assert task_mcp_servers("t", {"mcp_servers": []}) == []


def test_unsupported_server_key_is_an_error_not_a_dropped_tool() -> None:
    with pytest.raises(ValueError, match="unsupported keys: \\['env'\\]"):
        task_mcp_servers(
            "t", {"mcp_servers": [{"name": "x", "transport": "sse", "url": "http://x", "env": {"A": "b"}}]}
        )


@pytest.mark.parametrize("dataset", ["default", "terminal-bench-4.0", "terminal-bench-4.1"])
def test_medical_claims_processing_reports_its_playwright_server(
    benchmark: TerminalBenchBenchmark, dataset: str
) -> None:
    response = asyncio.run(benchmark.retrieve_task("medical-claims-processing", dataset=dataset))
    assert response.mcp_servers == [PLAYWRIGHT]
    assert response.skills_dir is None
    body = json.loads(response.model_dump_json())
    assert body["mcp_servers"] == [
        {"name": "playwright", "transport": "sse", "url": "http://playwright-mcp:3080/sse", "command": None, "args": []}
    ]
    assert "skills_dir" not in body


@pytest.mark.parametrize("dataset", ["default", "terminal-bench-4.0", "terminal-bench-4.1"])
def test_cumulative_layout_shift_reports_its_skills_dir(benchmark: TerminalBenchBenchmark, dataset: str) -> None:
    response = asyncio.run(benchmark.retrieve_task("cumulative-layout-shift", dataset=dataset))
    assert response.skills_dir == "/app/barber-shop-site/.agents/skills"
    assert response.mcp_servers == []
    assert "mcp_servers" not in json.loads(response.model_dump_json())


def test_task_without_tools_leaves_both_fields_out_of_the_response(benchmark: TerminalBenchBenchmark) -> None:
    for dataset in ("terminal-bench-2.0", "terminal-bench-4.0"):
        task_id = next(
            t for t in benchmark.datasets[dataset] if t not in ("medical-claims-processing", "cumulative-layout-shift")
        )
        body = json.loads(asyncio.run(benchmark.retrieve_task(task_id, dataset=dataset)).model_dump_json())
        assert "mcp_servers" not in body
        assert "skills_dir" not in body


def test_only_two_tasks_of_each_4_x_dataset_report_tools_and_none_of_2_x(benchmark: TerminalBenchBenchmark) -> None:
    for dataset, expected in {
        "terminal-bench-2.0": ([], []),
        "terminal-bench-2.1": ([], []),
        "terminal-bench-4.0": (["medical-claims-processing"], ["cumulative-layout-shift"]),
        "terminal-bench-4.1": (["medical-claims-processing"], ["cumulative-layout-shift"]),
    }.items():
        tasks = benchmark.datasets[dataset]
        environments = {t: task["task_definition"].get("environment", {}) for t, task in tasks.items()}
        assert [t for t, env in environments.items() if task_mcp_servers(t, env)] == expected[0], dataset
        assert [t for t, env in environments.items() if env.get("skills_dir")] == expected[1], dataset


@pytest.mark.parametrize("dataset", sorted(DATASET_PATHS))
def test_problem_statement_is_the_upstream_instruction_with_no_note(
    benchmark: TerminalBenchBenchmark, dataset: str
) -> None:
    root = Path(DATASET_PATHS[dataset])
    for task_id, task in benchmark.datasets[dataset].items():
        assert task["problem_statement"] == (root / task_id / "instruction.md").read_text(), task_id
        assert "MCP Servers:" not in task["problem_statement"], task_id


class RecordingSandbox:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.labels: dict[str, str] = {}
        self.created_at = None

    async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> ExecResult:
        self.commands.append(command)
        return ExecResult(exit_code=0, output="")

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        self.commands.append(f"upload {remote_path}")


def test_setup_task_writes_no_agent_configuration(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_start(*args: Any) -> None:
        pass

    async def drain(stream: Any) -> list[Any]:
        return [chunk async for chunk in stream]

    monkeypatch.setattr(service_module, "start_compose_runtime", fake_start)
    sandbox = RecordingSandbox()
    chunks = asyncio.run(
        drain(benchmark.setup_task("medical-claims-processing", sandbox, dataset="terminal-bench-4.0"))
    )  # pyright: ignore[reportArgumentType]

    assert not any("/bundle" in c or ".codex" in c for c in sandbox.commands)
    assert not any("Codex" in str(getattr(chunk, "data", "")) and "MCP" in str(chunk.data) for chunk in chunks)
    prompt = benchmark.datasets["terminal-bench-4.0"]["medical-claims-processing"]["problem_statement"]
    assert any(chunk.data == f"Problem statement uploaded to /tmp/problem_statement.md\n{prompt}" for chunk in chunks)
    assert "MCP" not in prompt
