import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import pytest
from benchmark_service.sandbox import (
    DaytonaProviderConfig,
    ExecResult,
    ModalProviderConfig,
    Sandbox,
    SandboxCreateRequest,
    SandboxProvider,
    SandboxQuery,
    SnapshotSource,
)
from benchmark_service.sandbox.daytona import DaytonaSandbox
from benchmark_service.schemas import EvaluateResponseRequest, StreamChunk, StreamMessageChunk, StreamResultChunk
from pydantic import ValidationError

from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark
from terminal_bench_benchmark_service.eval_resume import (
    SNAPSHOT_TIMEOUT_SECONDS,
    EvalResumeState,
    create_daytona_snapshot,
)


class FakeSandbox(Sandbox):
    def __init__(self, sandbox_id: str = "sandbox") -> None:
        self._id = sandbox_id
        self.commands: list[str] = []

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
        return ExecResult(exit_code=0, output="")

    async def command(
        self, command: str, *, cwd: str | None = None, timeout: float | None = None
    ) -> AsyncGenerator[str, None]:
        self.commands.append(command)
        if False:
            yield ""

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        pass

    async def download_file(self, remote_path: str) -> bytes:
        return b""


class FakeProvider(SandboxProvider):
    def __init__(self) -> None:
        self.sandbox = FakeSandbox("resume-sandbox")
        self.create_request: SandboxCreateRequest | None = None
        self.deleted: list[str] = []

    async def create_sandbox(self, request: SandboxCreateRequest) -> Sandbox:
        self.create_request = request
        return self.sandbox

    async def get_sandbox(self, instance_id: str) -> Sandbox:
        raise AssertionError("resume must create a fresh sandbox")

    async def delete_sandbox(self, instance_id: str) -> None:
        self.deleted.append(instance_id)

    async def list_sandboxes(self, query: SandboxQuery) -> AsyncGenerator[Sandbox, None]:
        if False:
            yield self.sandbox


def service() -> TerminalBenchBenchmark:
    benchmark = TerminalBenchBenchmark()
    task = {
        "problem_statement": "Fix it",
        "task_definition": {
            "environment": {
                "cpus": 2,
                "memory": "4G",
                "storage": "10G",
                "docker_image": "example/task:latest",
            },
            "agent": {"timeout_sec": 300},
            "verifier": {"timeout_sec": 60},
        },
    }
    benchmark.datasets = {"default": {"task-1": task}, "other": {"task-1": task}}
    return benchmark


def daytona_config() -> DaytonaProviderConfig:
    return DaytonaProviderConfig(DAYTONA_API_KEY="key", DAYTONA_API_URL="url", DAYTONA_TARGET="target")


def use_provider(monkeypatch: pytest.MonkeyPatch, provider: FakeProvider) -> None:
    monkeypatch.setattr(DaytonaProviderConfig, "create_provider", lambda _config: provider)


async def collect(stream: AsyncGenerator[StreamChunk, None]) -> list[StreamChunk]:
    return [chunk async for chunk in stream]


def test_stream_evaluate_response_preserves_text_evaluation() -> None:
    benchmark = service()
    benchmark.datasets["default"]["task-1"]["answer"] = "done"
    request = EvaluateResponseRequest(task_id="task-1", response="done")

    chunks = asyncio.run(collect(benchmark.stream_evaluate_response(request)))

    assert chunks[-1].data["score"] == 1.0


def test_checkpoint_is_emitted_before_verifier_setup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = service()
    events: list[str] = []

    async def snapshot(_sandbox: Sandbox, _name: str) -> None:
        events.append("snapshot")

    async def fail_copy(_sandbox: Sandbox, _task_id: str, _dataset: str | None = None) -> str:
        events.append("verifier")
        raise RuntimeError("injected verifier setup failure")

    monkeypatch.setattr("terminal_bench_benchmark_service.benchmark_service.create_daytona_snapshot", snapshot)
    monkeypatch.setattr(benchmark, "_copy_test_files", fail_copy)

    chunks = asyncio.run(collect(benchmark.evaluate_instance("task-1", FakeSandbox())))

    assert events == ["snapshot", "verifier"]
    assert [chunk.type for chunk in chunks] == ["eval_resume_state", "message", "error", "result"]
    state = EvalResumeState.model_validate(chunks[0].data)
    assert state.task_id == "task-1"
    assert state.dataset == "default"
    assert chunks[-1].data["exception_info"] == "injected verifier setup failure"


def test_resume_uses_fresh_snapshot_sandbox_without_setup_and_preserves_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = service()
    provider = FakeProvider()
    use_provider(monkeypatch, provider)
    state = EvalResumeState.create("task-1", "default")
    expected_result: dict[str, Any] = {
        "task_name": "task-1",
        "trial_name": "task-1-evaluation",
        "verifier_result": {"rewards": {"score": 1.0}, "output": "pass"},
        "exception_info": None,
    }
    evaluated: list[Sandbox] = []

    async def evaluator(
        _task_id: str, sandbox: Sandbox, dataset: str | None = None
    ) -> AsyncGenerator[StreamChunk, None]:
        evaluated.append(sandbox)
        yield StreamMessageChunk(type="message", data="same verifier")
        yield StreamResultChunk(type="result", data=expected_result)

    async def forbidden_setup(*_args: object, **_kwargs: object) -> AsyncGenerator[StreamChunk, None]:
        raise AssertionError("setup_task must not run when resuming a filesystem snapshot")
        yield StreamResultChunk(type="result", data={})

    monkeypatch.setattr(benchmark, "_evaluate_snapshot", evaluator)
    monkeypatch.setattr(benchmark, "setup_task", forbidden_setup)
    request = EvaluateResponseRequest(
        task_id="task-1",
        eval_resume_state=state.model_dump(mode="json"),
        sandbox_provider=daytona_config(),
    )

    chunks = asyncio.run(collect(benchmark.stream_evaluate_response(request)))

    assert chunks[0].type == "eval_resume_state"
    assert chunks[0].data == state.model_dump(mode="json")
    assert chunks[-1].data == expected_result
    assert evaluated == [provider.sandbox]
    assert provider.deleted == [provider.sandbox.id]
    assert provider.create_request is not None
    assert provider.create_request.source == SnapshotSource(snapshot=state.snapshot_name)
    assert provider.create_request.name.startswith("tb-eval-run-v1-")


def test_resume_deletes_temporary_sandbox_when_verifier_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = service()
    provider = FakeProvider()
    use_provider(monkeypatch, provider)
    state = EvalResumeState.create("task-1", "default")

    async def fail_evaluation(
        _task_id: str, _sandbox: Sandbox, dataset: str | None = None
    ) -> AsyncGenerator[StreamChunk, None]:
        raise RuntimeError("injected evaluation failure")
        yield StreamResultChunk(type="result", data={})

    monkeypatch.setattr(benchmark, "_evaluate_snapshot", fail_evaluation)
    request = EvaluateResponseRequest(
        task_id="task-1",
        eval_resume_state=state.model_dump(mode="json"),
        sandbox_provider=daytona_config(),
    )

    with pytest.raises(RuntimeError, match="injected evaluation failure"):
        asyncio.run(collect(benchmark.stream_evaluate_response(request)))

    assert provider.deleted == [provider.sandbox.id]


@pytest.mark.parametrize(
    ("task_id", "dataset", "error"),
    [("other-task", None, "task_id mismatch"), ("task-1", "other", "dataset mismatch")],
)
def test_resume_rejects_mismatched_state(task_id: str, dataset: str | None, error: str) -> None:
    benchmark = service()
    state = EvalResumeState.create("task-1", "default")
    request = EvaluateResponseRequest(task_id=task_id, dataset=dataset, eval_resume_state=state.model_dump(mode="json"))

    with pytest.raises(ValueError, match=error):
        asyncio.run(collect(benchmark.stream_evaluate_response(request, dataset=dataset)))


@pytest.mark.parametrize("snapshot_name", ["unrelated-snapshot", f"tb-eval-resume-v1-{'0' * 12}-{'1' * 32}"])
def test_resume_rejects_malformed_or_noncanonical_snapshot_pointer(snapshot_name: str) -> None:
    benchmark = service()
    state = EvalResumeState.create("task-1", "default").model_dump(mode="json")
    state["snapshot_name"] = snapshot_name
    request = EvaluateResponseRequest(task_id="task-1", eval_resume_state=state)

    with pytest.raises(ValidationError):
        asyncio.run(collect(benchmark.stream_evaluate_response(request)))


def test_snapshot_pointers_are_unique() -> None:
    assert (
        EvalResumeState.create("task-1", "default").snapshot_name
        != EvalResumeState.create("task-1", "default").snapshot_name
    )


def test_resume_rejects_non_daytona_provider() -> None:
    benchmark = service()
    state = EvalResumeState.create("task-1", "default")
    request = EvaluateResponseRequest(
        task_id="task-1", eval_resume_state=state.model_dump(mode="json"), sandbox_provider=ModalProviderConfig()
    )

    with pytest.raises(ValueError, match="Daytona sandbox_provider"):
        asyncio.run(collect(benchmark.stream_evaluate_response(request)))


def test_snapshot_adapter_is_guarded_and_calls_daytona_filesystem_hook() -> None:
    with pytest.raises(ValueError, match="Daytona sandbox"):
        asyncio.run(create_daytona_snapshot(FakeSandbox(), "snapshot"))

    calls: list[tuple[str, int]] = []

    async def create_snapshot(name: str, timeout: int) -> None:
        calls.append((name, timeout))

    sandbox = DaytonaSandbox(SimpleNamespace(_experimental_create_snapshot=create_snapshot))
    asyncio.run(create_daytona_snapshot(sandbox, "snapshot"))

    assert calls == [("snapshot", SNAPSHOT_TIMEOUT_SECONDS)]
