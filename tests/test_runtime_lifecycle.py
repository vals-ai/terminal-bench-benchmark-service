"""Unit coverage for TBench4 compose setup, collection, and cleanup ordering."""

import asyncio
from collections.abc import AsyncGenerator, Mapping
from typing import Any

import pytest
from benchmark_service import ComposeSandbox, ComposeSource, ExecResult, Sandbox
from benchmark_service.schemas import StreamResultChunk

import terminal_bench_benchmark_service.benchmark_service as service_module
from terminal_bench_benchmark_service import isolated_verifier
from terminal_bench_benchmark_service.benchmark_service import MAX_DAYTONA_VCPU, TerminalBenchBenchmark


class FakeSandbox(Sandbox):
    def __init__(self, sandbox_id: str = "agent") -> None:
        self._id = sandbox_id
        self.commands: list[str] = []
        self.uploads: dict[str, bytes] = {}

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._id

    @property
    def state(self) -> str:
        return "started"

    async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> ExecResult:
        self.commands.append(command)
        return ExecResult(exit_code=0, output="PRESENT\n")

    async def command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env_vars: Mapping[str, str] | None = None,
    ) -> AsyncGenerator[str, None]:
        self.commands.append(command)
        yield ""

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        self.uploads[remote_path] = content

    async def download_file(self, remote_path: str) -> bytes:
        return b"archive"


@pytest.fixture
def benchmark() -> TerminalBenchBenchmark:
    return asyncio.run(TerminalBenchBenchmark.create())


def test_compose_setup_starts_runtime_before_using_main_service(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(_test_compose_setup_starts_runtime_before_using_main_service(benchmark, monkeypatch))


async def _test_compose_setup_starts_runtime_before_using_main_service(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch
) -> None:
    outer = FakeSandbox()
    started: dict[str, Any] = {}

    async def fake_start(*args: Any) -> None:
        started["args"] = args

    monkeypatch.setattr(service_module, "start_compose_runtime", fake_start)

    chunks = [chunk async for chunk in benchmark.setup_task("ctr-optimization", outer, dataset="terminal-bench-4.0")]

    assert isinstance(started["args"][-1], FakeSandbox)
    assert isinstance(started["args"][4], dict)
    assert any(isinstance(chunk, StreamResultChunk) for chunk in chunks)
    assert any("docker compose" in command and "main" in command for command in outer.commands)


def test_failed_compose_setup_cleans_up_runtime(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(_test_failed_compose_setup_cleans_up_runtime(benchmark, monkeypatch))


async def _test_failed_compose_setup_cleans_up_runtime(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch
) -> None:
    outer = FakeSandbox()
    stopped: list[tuple[str, Sandbox]] = []

    async def fake_start(*args: Any) -> None:
        raise RuntimeError("compose startup failed")

    async def fake_stop(task_id: str, sandbox: Sandbox) -> None:
        stopped.append((task_id, sandbox))

    monkeypatch.setattr(service_module, "start_compose_runtime", fake_start)
    monkeypatch.setattr(service_module, "stop_compose_runtime", fake_stop)

    with pytest.raises(RuntimeError, match="compose startup failed"):
        _ = [chunk async for chunk in benchmark.setup_task("ctr-optimization", outer, dataset="terminal-bench-4.0")]

    assert stopped == [("ctr-optimization", outer)]


def test_compose_setup_passes_normalized_cpu_to_nested_runtime(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(_test_compose_setup_passes_normalized_cpu_to_nested_runtime(benchmark, monkeypatch))


async def _test_compose_setup_passes_normalized_cpu_to_nested_runtime(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch
) -> None:
    outer = FakeSandbox()
    started: dict[str, Any] = {}

    async def fake_start(*args: Any) -> None:
        started["args"] = args

    monkeypatch.setattr(service_module, "start_compose_runtime", fake_start)

    _ = [chunk async for chunk in benchmark.setup_task("live-database-cutover", outer, dataset="terminal-bench-4.0")]

    assert started["args"][4]["cpus"] == MAX_DAYTONA_VCPU


def test_compose_evaluation_preserves_outer_and_tears_down(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(_test_compose_evaluation_preserves_outer_and_tears_down(benchmark, monkeypatch))


async def _test_compose_evaluation_preserves_outer_and_tears_down(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch
) -> None:
    outer = FakeSandbox()
    observed: dict[str, Any] = {}

    async def fake_evaluate(
        task_id: str,
        sandbox: Sandbox,
        dataset: str | None = None,
        *,
        outer_sandbox: Sandbox | None,
        runtime_source: ComposeSource | None,
    ) -> AsyncGenerator[StreamResultChunk, None]:
        observed.update(sandbox=sandbox, outer=outer_sandbox, source=runtime_source)
        yield StreamResultChunk(type="result", data={"ok": True})

    async def fake_stop(task_id: str, sandbox: Sandbox) -> None:
        observed.update(stopped_task=task_id, stopped_sandbox=sandbox)

    monkeypatch.setattr(benchmark, "_evaluate_in_isolated_verifier", fake_evaluate)
    monkeypatch.setattr(service_module, "stop_compose_runtime", fake_stop)

    _ = [chunk async for chunk in benchmark.evaluate_instance("ctr-optimization", outer, dataset="terminal-bench-4.0")]

    assert isinstance(observed["sandbox"], ComposeSandbox)
    assert observed["outer"] is outer
    assert isinstance(observed["source"], ComposeSource)
    assert observed["stopped_task"] == "ctr-optimization"
    assert observed["stopped_sandbox"] is outer


def test_sidecar_collect_and_artifact_use_the_declared_service(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(_test_sidecar_collect_and_artifact_use_the_declared_service(benchmark, monkeypatch))


async def _test_sidecar_collect_and_artifact_use_the_declared_service(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch
) -> None:
    outer = FakeSandbox("outer")
    main = FakeSandbox("main")
    source = benchmark._compose_source("ctr-optimization", "terminal-bench-4.0")  # pyright: ignore[reportPrivateUsage]
    assert source is not None

    collect_chunks = [
        chunk
        async for chunk in benchmark._run_collect_hooks(  # pyright: ignore[reportPrivateUsage]
            "ctr-optimization",
            "terminal-bench-4.0",
            main,
            outer_sandbox=outer,
            runtime_source=source,
            services={"api"},
        )
    ]
    assert collect_chunks
    assert any(" api " in command and "healthz" in command for command in outer.commands)

    sidecar = FakeSandbox("sidecar")
    monkeypatch.setattr(service_module, "compose_service_sandbox", lambda *_args: sidecar)
    monkeypatch.setattr(benchmark, "_check_expansion", lambda *_args: _completed())

    verifier = FakeSandbox("verifier")
    artifact = isolated_verifier.parse_artifacts([{"source": "/shared/verify_snapshot.json", "service": "api"}])[0]
    await benchmark._carry_artifact(  # pyright: ignore[reportPrivateUsage]
        artifact,
        main,
        verifier,
        outer_sandbox=outer,
        runtime_source=source,
    )

    assert sidecar.commands
    assert sidecar.commands[0].startswith("if [ -e")
    assert not main.commands


async def _completed() -> None:
    return None
