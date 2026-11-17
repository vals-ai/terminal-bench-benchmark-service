"""`setup_task` writes the task folder. The problem statement file and Codex setup stay as they were."""

import asyncio
import tomllib
from collections.abc import Callable
from typing import Any

import pytest
from benchmark_service.sandbox import ExecResult
from benchmark_service.schemas import StreamMessageChunk, StreamResultChunk

from terminal_bench_benchmark_service import benchmark_service as service_module
from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark
from terminal_bench_benchmark_service.task_folder import (
    MCP_SERVER_FIELDS,
    TASK_DIR,
    build_task_folder,
    task_toml,
    write_task_folder,
)

DATASETS = ("default", "terminal-bench-2.0", "terminal-bench-2.1", "terminal-bench-4.0", "terminal-bench-4.1")
PLAYWRIGHT = {"name": "playwright", "transport": "sse", "url": "http://playwright-mcp:3080/sse"}
# Fields of Harbor's `MCPServerConfig` at v0.26.0 (src/harbor/models/task/config.py).
HARBOR_MCP_FIELDS = {"name", "transport", "url", "command", "args", "headers"}


@pytest.fixture(scope="module")
def benchmark() -> TerminalBenchBenchmark:
    return asyncio.run(TerminalBenchBenchmark.create())


class FolderSandbox:
    """Sandbox fake that records commands and uploads."""

    name = "fake"
    state = "started"

    def __init__(
        self,
        *,
        task_dir_exists: bool = False,
        mkdir_exit_code: int = 0,
        upload_error: Exception | None = None,
    ) -> None:
        self.task_dir_exists = task_dir_exists
        self.mkdir_exit_code = mkdir_exit_code
        self.upload_error = upload_error
        self.commands: list[str] = []
        self.uploads: dict[str, bytes] = {}
        self.upload_order: list[str] = []
        self.labels: dict[str, str] = {}
        self.created_at = None

    async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> ExecResult:
        self.commands.append(command)
        if command.startswith(f"test -e {TASK_DIR}"):
            return ExecResult(exit_code=0 if self.task_dir_exists else 1, output="")
        if command == f"mkdir -p {TASK_DIR}":
            return ExecResult(
                exit_code=self.mkdir_exit_code, output="Permission denied" if self.mkdir_exit_code else ""
            )
        return ExecResult(exit_code=0, output="")

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        if self.upload_error is not None and remote_path.startswith(TASK_DIR):
            raise self.upload_error
        self.uploads[remote_path] = content
        self.upload_order.append(remote_path)


def _plain_task(benchmark: TerminalBenchBenchmark, dataset: str) -> str:
    """A task that runs in the sandbox itself, with no compose services."""
    for task_id in benchmark.get_dataset(dataset):
        if benchmark._compose_source(task_id, dataset) is None:  # pyright: ignore[reportPrivateUsage]
            return task_id
    raise AssertionError(f"No plain task in {dataset}")


def _setup(benchmark: TerminalBenchBenchmark, task_id: str, dataset: str, sandbox: FolderSandbox) -> list[Any]:
    async def run() -> list[Any]:
        return [chunk async for chunk in benchmark.setup_task(task_id, sandbox, dataset=dataset)]  # pyright: ignore[reportArgumentType]

    return asyncio.run(run())


def _messages(chunks: list[Any]) -> list[str]:
    return [chunk.data for chunk in chunks if isinstance(chunk, StreamMessageChunk)]


def _all_tasks(benchmark: TerminalBenchBenchmark) -> list[tuple[str, str]]:
    return [(dataset, task_id) for dataset in DATASETS for task_id in benchmark.get_dataset(dataset)]


def _folder(benchmark: TerminalBenchBenchmark, dataset: str, task_id: str) -> dict[str, bytes]:
    task = benchmark.get_dataset(dataset)[task_id]
    return build_task_folder(
        task_id=task_id,
        instruction=(benchmark._task_dir(task_id, dataset) / "instruction.md").read_bytes(),  # pyright: ignore[reportPrivateUsage]
        timeout_sec=task["task_definition"]["agent"]["timeout_sec"],
        mcp_servers=task["task_definition"].get("environment", {}).get("mcp_servers", []),
    )


# --- the folder holds what the contract says, for every served task ---


def test_every_served_task_gets_a_valid_folder(benchmark: TerminalBenchBenchmark) -> None:
    tasks = _all_tasks(benchmark)
    assert len(tasks) > 300
    for dataset, task_id in tasks:
        task = benchmark.get_dataset(dataset)[task_id]
        upstream = task["task_definition"]
        files = _folder(benchmark, dataset, task_id)

        assert list(files) == ["instruction.md", "task.toml"]
        parsed = tomllib.loads(files["task.toml"].decode())
        assert set(parsed) <= {"schema_version", "metadata", "agent", "environment"}, (dataset, task_id)
        assert parsed["metadata"] == {"benchmark": "terminal-bench", "task_id": task_id}
        assert parsed["agent"] == {"timeout_sec": float(upstream["agent"]["timeout_sec"])}
        assert set(parsed.get("environment", {})) <= {"mcp_servers"}

        wanted = [
            {key: server[key] for key in MCP_SERVER_FIELDS if key in server}
            for server in upstream.get("environment", {}).get("mcp_servers", [])
        ]
        assert parsed.get("environment", {}).get("mcp_servers", []) == wanted


def test_instruction_is_the_upstream_file_byte_for_byte(benchmark: TerminalBenchBenchmark) -> None:
    for dataset, task_id in _all_tasks(benchmark):
        upstream = (benchmark._task_dir(task_id, dataset) / "instruction.md").read_bytes()  # pyright: ignore[reportPrivateUsage]
        assert _folder(benchmark, dataset, task_id)["instruction.md"] == upstream


def test_instruction_keeps_crlf_and_non_ascii_bytes() -> None:
    text = "Line one\r\nJosé Núñez\r\n".encode()
    assert build_task_folder(task_id="t", instruction=text, timeout_sec=60, mcp_servers=[])["instruction.md"] == text


def test_instruction_has_no_mcp_note_but_the_problem_statement_keeps_it(benchmark: TerminalBenchBenchmark) -> None:
    for dataset in ("terminal-bench-4.0", "terminal-bench-4.1"):
        task = benchmark.get_dataset(dataset)["medical-claims-processing"]
        instruction = _folder(benchmark, dataset, "medical-claims-processing")["instruction.md"].decode()
        assert "MCP Servers:" not in instruction
        assert "MCP Servers:" in task["problem_statement"]
        assert task["problem_statement"].startswith(instruction.replace("\r\n", "\n"))


def test_task_toml_leaves_out_verifier_solution_metadata_and_resources(benchmark: TerminalBenchBenchmark) -> None:
    text = _folder(benchmark, "terminal-bench-4.1", "medical-claims-processing")["task.toml"].decode()
    for leaked in ("[verifier", "[solution", "author", "difficulty", "cpus", "memory_mb", "storage_mb", "artifacts"):
        assert leaked not in text
    assert tomllib.loads(text) == {
        "schema_version": "1.3",
        "metadata": {"benchmark": "terminal-bench", "task_id": "medical-claims-processing"},
        "agent": {"timeout_sec": 28800.0},
        "environment": {"mcp_servers": [PLAYWRIGHT]},
    }


def test_mcp_fields_match_harbor() -> None:
    assert set(MCP_SERVER_FIELDS) == HARBOR_MCP_FIELDS


def test_task_toml_copies_stdio_headers_and_unusual_text() -> None:
    servers = [
        {"name": "fs 🚀", "transport": "stdio", "command": "npx", "args": ["-y", 'srv"\\ü', "a\nb"]},
        {"name": "web", "transport": "http", "url": "http://h/mcp", "headers": {"Authorization": "Bearer ${TOKEN}"}},
        {"name": "env", "transport": "stdio", "command": "x", "env": {"A": "b"}},
    ]
    parsed = tomllib.loads(task_toml('id"1', 5, servers).decode())
    assert parsed["metadata"]["task_id"] == 'id"1'
    assert parsed["environment"]["mcp_servers"] == [
        {"name": "fs 🚀", "transport": "stdio", "command": "npx", "args": ["-y", 'srv"\\ü', "a\nb"]},
        {"name": "web", "transport": "http", "url": "http://h/mcp", "headers": {"Authorization": "Bearer ${TOKEN}"}},
        {"name": "env", "transport": "stdio", "command": "x"},
    ]


def test_task_toml_without_servers_has_no_environment_table() -> None:
    assert "environment" not in tomllib.loads(task_toml("t", 60, []).decode())


def test_task_toml_rejects_text_it_cannot_write() -> None:
    with pytest.raises(ValueError, match="not valid TOML"):
        _ = task_toml("t", 60, [{"name": "bad\x7f", "transport": "sse", "url": "http://h/sse"}])


# --- setup_task ---


def test_setup_task_writes_the_folder_and_keeps_the_problem_statement(benchmark: TerminalBenchBenchmark) -> None:
    dataset = "terminal-bench-2.0"
    task_id = _plain_task(benchmark, dataset)
    sandbox = FolderSandbox()
    chunks = _setup(benchmark, task_id, dataset, sandbox)

    problem = benchmark.get_dataset(dataset)[task_id]["problem_statement"]
    assert sandbox.uploads["/tmp/problem_statement.md"] == problem.encode()
    assert (
        sandbox.uploads[f"{TASK_DIR}/instruction.md"]
        == (
            benchmark._task_dir(task_id, dataset) / "instruction.md"  # pyright: ignore[reportPrivateUsage]
        ).read_bytes()
    )
    assert tomllib.loads(sandbox.uploads[f"{TASK_DIR}/task.toml"].decode())["metadata"]["task_id"] == task_id
    # task.toml comes last, so a folder that has it is complete.
    assert sandbox.upload_order[-2:] == [f"{TASK_DIR}/instruction.md", f"{TASK_DIR}/task.toml"]
    assert f"mkdir -p {TASK_DIR}" in sandbox.commands
    assert any(f"Task folder written to {TASK_DIR}" in message for message in _messages(chunks))
    assert isinstance(chunks[-1], StreamResultChunk) and chunks[-1].data == {"status": "ok"}


def test_setup_task_skips_the_folder_when_the_path_exists(benchmark: TerminalBenchBenchmark) -> None:
    dataset = "terminal-bench-2.0"
    task_id = _plain_task(benchmark, dataset)
    sandbox = FolderSandbox(task_dir_exists=True)
    chunks = _setup(benchmark, task_id, dataset, sandbox)

    assert not any(path.startswith(TASK_DIR) for path in sandbox.uploads)
    assert f"mkdir -p {TASK_DIR}" not in sandbox.commands
    assert [m for m in _messages(chunks) if "skipped" in m] == [
        f"Task folder skipped: {TASK_DIR} already exists in the sandbox."
    ]
    problem = benchmark.get_dataset(dataset)[task_id]["problem_statement"]
    assert sandbox.uploads["/tmp/problem_statement.md"] == problem.encode()
    assert isinstance(chunks[-1], StreamResultChunk) and chunks[-1].data == {"status": "ok"}


@pytest.mark.parametrize(
    "make_sandbox",
    [
        lambda: FolderSandbox(mkdir_exit_code=1),
        lambda: FolderSandbox(upload_error=RuntimeError("disk full")),
    ],
    ids=["mkdir-fails", "upload-fails"],
)
def test_setup_task_still_succeeds_when_the_folder_cannot_be_written(
    benchmark: TerminalBenchBenchmark, make_sandbox: Callable[[], FolderSandbox]
) -> None:
    dataset = "terminal-bench-2.0"
    task_id = _plain_task(benchmark, dataset)
    sandbox = make_sandbox()
    chunks = _setup(benchmark, task_id, dataset, sandbox)

    assert any(message.startswith("Task folder not written:") for message in _messages(chunks))
    problem = benchmark.get_dataset(dataset)[task_id]["problem_statement"]
    assert sandbox.uploads["/tmp/problem_statement.md"] == problem.encode()
    assert isinstance(chunks[-1], StreamResultChunk) and chunks[-1].data == {"status": "ok"}


def test_a_failed_upload_leaves_no_task_toml() -> None:
    sandbox = FolderSandbox(upload_error=RuntimeError("disk full"))
    files = {"instruction.md": b"x", "task.toml": b"y"}
    message = asyncio.run(write_task_folder(sandbox, files))  # pyright: ignore[reportArgumentType]
    assert message == "Task folder not written: disk full"
    assert f"{TASK_DIR}/task.toml" not in sandbox.uploads


def test_retrieve_task_still_returns_the_problem_statement_path(benchmark: TerminalBenchBenchmark) -> None:
    response = asyncio.run(benchmark.retrieve_task("medical-claims-processing", dataset="terminal-bench-4.0"))
    assert response.problem_path == "/tmp/problem_statement.md"


# --- Terminal-Bench 4 tasks run in compose services, so the folder goes through ComposeSandbox ---


class OuterSandbox(FolderSandbox):
    """The DinD sandbox that Valkyrie creates. Compose commands reach the main service through it."""

    async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> ExecResult:
        self.commands.append(command)
        if "test -e /task" in command:
            return ExecResult(exit_code=0 if self.task_dir_exists else 1, output="")
        return ExecResult(exit_code=0, output="")


def _compose_setup(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch, task_id: str, outer: OuterSandbox
) -> list[Any]:
    async def fake_start(*args: Any) -> None:
        pass

    monkeypatch.setattr(service_module, "start_compose_runtime", fake_start)
    return _setup(benchmark, task_id, "terminal-bench-4.0", outer)


def test_compose_task_gets_the_folder_in_the_main_service(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch
) -> None:
    outer = OuterSandbox()
    chunks = _compose_setup(benchmark, monkeypatch, "medical-claims-processing", outer)

    mkdir = [c for c in outer.commands if c.endswith("sh -lc 'mkdir -p /task'") and "docker compose" in c]
    assert len(mkdir) == 1 and " exec " in mkdir[0] and " main " in mkdir[0]
    assert any("cat > /task/instruction.md" in c for c in outer.commands)
    assert any("cat > /task/task.toml" in c for c in outer.commands)
    task_toml_uploads = [c for c in outer.commands if "cat > /task/task.toml" in c]
    assert outer.commands.index(task_toml_uploads[0]) > outer.commands.index(
        next(c for c in outer.commands if "cat > /task/instruction.md" in c)
    )
    assert any(message.startswith(f"Task folder written to {TASK_DIR}") for message in _messages(chunks))


def test_takens_embedding_lean_skips_the_folder_because_it_uses_the_path(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The task's own Lean project is /task. The guard finds the path and writes nothing there."""
    outer = OuterSandbox(task_dir_exists=True)
    chunks = _compose_setup(benchmark, monkeypatch, "takens-embedding-lean", outer)

    assert not any("mkdir -p /task" in c or "cat > /task" in c for c in outer.commands)
    assert not any(path.startswith(TASK_DIR) for path in outer.uploads)
    assert f"Task folder skipped: {TASK_DIR} already exists in the sandbox." in _messages(chunks)
    assert isinstance(chunks[-1], StreamResultChunk) and chunks[-1].data == {"status": "ok"}


def test_setup_task_still_succeeds_when_the_folder_cannot_be_built(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(**_: Any) -> dict[str, bytes]:
        raise ValueError("bad value")

    monkeypatch.setattr(service_module, "build_task_folder", broken)
    dataset = "terminal-bench-2.0"
    task_id = _plain_task(benchmark, dataset)
    sandbox = FolderSandbox()
    chunks = _setup(benchmark, task_id, dataset, sandbox)

    assert "Task folder not written: bad value" in _messages(chunks)
    assert not any(path.startswith(TASK_DIR) for path in sandbox.uploads)
    assert isinstance(chunks[-1], StreamResultChunk) and chunks[-1].data == {"status": "ok"}
