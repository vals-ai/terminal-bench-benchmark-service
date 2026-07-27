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

import terminal_bench_benchmark_service.benchmark_service as service_module
import terminal_bench_benchmark_service.eval_resume as resume_module
from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark
from terminal_bench_benchmark_service.eval_resume import (
    SNAPSHOT_TIMEOUT_SECONDS,
    EvalResumeState,
    create_daytona_snapshot,
)


class FakeSandbox(Sandbox):
    def __init__(self, sandbox_id: str = "sandbox") -> None:
        self._id = sandbox_id
        self._sandbox = SimpleNamespace(labels={"Id": "run-123"})
        self.commands: list[str] = []
        self.files: dict[str, bytes] = {}

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
        self.files[remote_path] = content

    async def download_file(self, remote_path: str) -> bytes:
        return self.files.get(remote_path, b"")


class FakeProvider(SandboxProvider):
    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.sandbox = FakeSandbox("resume-sandbox")
        self.sandbox.files = dict(files or {})
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


def resume_state() -> EvalResumeState:
    benchmark = service()
    return EvalResumeState.create(
        "task-1",
        "default",
        "run-123",
        benchmark._task_contract_sha256("task-1", "default"),
    )


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


def test_initial_response_collects_expired_snapshots_without_eval_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = service()
    benchmark.datasets["default"]["task-1"]["answer"] = "done"
    expired_name = f"{resume_module.SNAPSHOT_PREFIX}-{'a' * 12}-6553f100{'1' * 24}"

    class SnapshotService:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        async def list(self, page: int, limit: int) -> SimpleNamespace:
            assert (page, limit) == (1, 100)
            return SimpleNamespace(
                items=[SimpleNamespace(name=expired_name)],
                total_pages=1,
            )

        async def delete(self, snapshot: SimpleNamespace) -> None:
            self.deleted.append(snapshot.name)

    snapshots = SnapshotService()
    provider = FakeProvider()
    provider._daytona = SimpleNamespace(snapshot=snapshots)  # type: ignore[attr-defined]
    use_provider(monkeypatch, provider)
    request = EvaluateResponseRequest(
        task_id="task-1",
        response="done",
        sandbox_provider=daytona_config(),
    )

    chunks = asyncio.run(collect(benchmark.stream_evaluate_response(request)))

    assert chunks[-1].data["score"] == 1.0
    assert snapshots.deleted == [expired_name]


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
    state = resume_state()
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
    assert provider.create_request.labels["Id"] == "run-123"


def test_resume_deletes_temporary_sandbox_when_verifier_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = service()
    provider = FakeProvider()
    use_provider(monkeypatch, provider)
    state = resume_state()

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
    state = resume_state()
    request = EvaluateResponseRequest(task_id=task_id, dataset=dataset, eval_resume_state=state.model_dump(mode="json"))

    with pytest.raises(ValueError, match=error):
        asyncio.run(collect(benchmark.stream_evaluate_response(request, dataset=dataset)))


@pytest.mark.parametrize("snapshot_name", ["unrelated-snapshot", f"tb-eval-resume-v1-{'0' * 12}-{'1' * 32}"])
def test_resume_rejects_malformed_or_noncanonical_snapshot_pointer(snapshot_name: str) -> None:
    benchmark = service()
    state = resume_state().model_dump(mode="json")
    state["snapshot_name"] = snapshot_name
    request = EvaluateResponseRequest(task_id="task-1", eval_resume_state=state)

    with pytest.raises(ValidationError):
        asyncio.run(collect(benchmark.stream_evaluate_response(request)))


@pytest.mark.parametrize("version", [True, 1.0])
def test_resume_rejects_non_exact_integer_version_before_checkpoint_or_provider(
    version: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = service()
    state = resume_state().model_dump(mode="json")
    state["version"] = version
    request = EvaluateResponseRequest(
        task_id="task-1",
        eval_resume_state=state,
        sandbox_provider=daytona_config(),
    )
    chunks: list[StreamChunk] = []

    def forbidden_provider(_config: DaytonaProviderConfig) -> SandboxProvider:
        raise AssertionError("invalid resume state must not create a provider")

    monkeypatch.setattr(DaytonaProviderConfig, "create_provider", forbidden_provider)

    async def consume() -> None:
        async for chunk in benchmark.stream_evaluate_response(request):
            chunks.append(chunk)

    with pytest.raises(ValidationError):
        asyncio.run(consume())

    assert chunks == []


def test_snapshot_pointers_are_unique() -> None:
    assert resume_state().snapshot_name != resume_state().snapshot_name


def test_resume_rejects_non_daytona_provider() -> None:
    benchmark = service()
    state = resume_state()
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


@pytest.mark.parametrize(
    ("task_id", "expected_command"),
    [
        ("nginx-request-logging", "nginx -t"),
        ("pypi-server", "/opt/pypiserver/venv/bin/pypi-server"),
        ("qemu-alpine-ssh", "qemu-system-x86_64"),
    ],
)
def test_resume_rehydrates_process_dependent_tasks(task_id: str, expected_command: str) -> None:
    benchmark = service()
    sandbox = FakeSandbox()

    asyncio.run(benchmark._rehydrate_snapshot(task_id, sandbox))

    assert expected_command in sandbox.commands[-1]


def test_cancelled_sandbox_creation_deletes_late_created_sandbox() -> None:
    async def run_check() -> None:
        provider = FakeProvider()
        started = asyncio.Event()
        release = asyncio.Event()

        async def delayed_create(_request: SandboxCreateRequest) -> Sandbox:
            started.set()
            await release.wait()
            return provider.sandbox

        provider.create_sandbox = delayed_create  # type: ignore[method-assign]
        request = SandboxCreateRequest(
            source=SnapshotSource(snapshot="snapshot"),
            resources=service_module.Resources(vcpu=1, memory=1, disk=1),
            name="resume",
            labels={},
            env_vars={},
            auto_stop_interval=15,
            create_timeout=600,
        )
        task = asyncio.create_task(service_module._create_owned_sandbox(provider, request))
        await started.wait()
        task.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert provider.deleted == [provider.sandbox.id]

    asyncio.run(run_check())


def test_checkpoint_and_resume_preserve_state_at_the_real_verifier_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = service()
    original = FakeSandbox("original")
    original.files["/app/agent-result"] = b"solved"
    snapshots: dict[str, dict[str, bytes]] = {}

    async def capture(sandbox: Sandbox, name: str) -> None:
        snapshots[name] = dict(getattr(sandbox, "files"))

    async def stop_after_checkpoint(*_args: object, **_kwargs: object) -> AsyncGenerator[StreamChunk, None]:
        raise RuntimeError("stop after checkpoint")
        yield StreamResultChunk(type="result", data={})

    monkeypatch.setattr(service_module, "create_daytona_snapshot", capture)
    monkeypatch.setattr(benchmark, "_evaluate_snapshot", stop_after_checkpoint)
    emitted: list[StreamChunk] = []
    with pytest.raises(RuntimeError, match="stop after checkpoint"):
        asyncio.run(collect_with_output(benchmark.evaluate_instance("task-1", original), emitted))

    state = EvalResumeState.model_validate(emitted[0].data)
    provider = FakeProvider(snapshots[state.snapshot_name])
    use_provider(monkeypatch, provider)

    async def copy_tests(sandbox: Sandbox, _task_id: str, _dataset: str | None = None) -> str:
        assert getattr(sandbox, "files")["/app/agent-result"] == b"solved"
        return "run-verifier"

    async def stream_test(*_args: object, **_kwargs: object) -> AsyncGenerator[str, None]:
        yield "verifier passed"

    async def reward(sandbox: Sandbox) -> dict[str, float]:
        assert getattr(sandbox, "files")["/app/agent-result"] == b"solved"
        return {"score": 1.0}

    monkeypatch.undo()
    use_provider(monkeypatch, provider)
    monkeypatch.setattr(benchmark, "_copy_test_files", copy_tests)
    monkeypatch.setattr(benchmark, "_stream_command_with_retry", stream_test)
    monkeypatch.setattr(benchmark, "_retrieve_reward", reward)
    request = EvaluateResponseRequest(
        task_id="task-1",
        eval_resume_state=state.model_dump(mode="json"),
        sandbox_provider=daytona_config(),
    )

    resumed = asyncio.run(collect(benchmark.stream_evaluate_response(request)))

    assert resumed[-1].data["verifier_result"]["rewards"] == {"score": 1.0}


async def collect_with_output(stream: AsyncGenerator[StreamChunk, None], output: list[StreamChunk]) -> None:
    async for chunk in stream:
        output.append(chunk)


def test_resume_rejects_changed_task_contract_before_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = service()
    state = resume_state()
    benchmark.datasets["default"]["task-1"]["task_definition"]["environment"]["docker_image"] = "changed"
    request = EvaluateResponseRequest(
        task_id="task-1",
        eval_resume_state=state.model_dump(mode="json"),
        sandbox_provider=daytona_config(),
    )

    def forbidden_provider(_config: DaytonaProviderConfig) -> SandboxProvider:
        raise AssertionError("contract mismatch must not create a provider")

    monkeypatch.setattr(DaytonaProviderConfig, "create_provider", forbidden_provider)

    with pytest.raises(ValueError, match="task contract"):
        asyncio.run(collect(benchmark.stream_evaluate_response(request)))


def test_snapshot_janitor_deletes_only_expired_owned_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    old_time = 1_700_000_000
    new_time = old_time + resume_module.SNAPSHOT_RETENTION_SECONDS
    monkeypatch.setattr(resume_module.time, "time", lambda: old_time)
    old = resume_state()
    monkeypatch.setattr(resume_module.time, "time", lambda: new_time)
    fresh = resume_state()

    class SnapshotService:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        async def list(self, page: int, limit: int) -> SimpleNamespace:
            assert (page, limit) == (1, 100)
            return SimpleNamespace(
                items=[SimpleNamespace(name=old.snapshot_name), SimpleNamespace(name=fresh.snapshot_name)],
                total_pages=1,
            )

        async def delete(self, snapshot: SimpleNamespace) -> None:
            self.deleted.append(snapshot.name)

    snapshots = SnapshotService()
    provider = SimpleNamespace(_daytona=SimpleNamespace(snapshot=snapshots))

    asyncio.run(
        resume_module.cleanup_expired_daytona_snapshots(
            provider,
            now_seconds=new_time + 1,
        )
    )

    assert snapshots.deleted == [old.snapshot_name]


def test_failed_snapshot_creation_attempts_to_delete_partial_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed: list[str] = []

    async def create_snapshot(_name: str, _timeout: int) -> None:
        raise RuntimeError("snapshot timed out")

    class SnapshotsApi:
        def __init__(self, _client: object) -> None:
            pass

        async def get_snapshot(self, name: str) -> SimpleNamespace:
            assert name == "snapshot"
            return SimpleNamespace(id="snapshot-id")

        async def remove_snapshot(self, snapshot_id: str) -> None:
            removed.append(snapshot_id)

    import daytona_api_client_async

    monkeypatch.setattr(daytona_api_client_async, "SnapshotsApi", SnapshotsApi)
    inner = SimpleNamespace(
        _experimental_create_snapshot=create_snapshot,
        _sandbox_api=SimpleNamespace(api_client=object()),
    )

    with pytest.raises(RuntimeError, match="snapshot timed out"):
        asyncio.run(create_daytona_snapshot(DaytonaSandbox(inner), "snapshot"))

    assert removed == ["snapshot-id"]
