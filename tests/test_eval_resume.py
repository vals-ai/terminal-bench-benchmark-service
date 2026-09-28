"""TBench4 persists what it grades, streams that as resume state, and grades from it again."""

import asyncio
import hashlib
import io
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from benchmark_service.context import sandbox_provider_scope
from botocore.exceptions import ClientError
from benchmark_service.sandbox import (
    ExecResult,
    Sandbox,
    SandboxCreateRequest,
    SandboxProvider,
    SandboxQuery,
)
from benchmark_service.schemas import (
    EvaluateResponseRequest,
    StreamChunk,
    StreamErrorChunk,
    StreamEvalResumeStateChunk,
    StreamResultChunk,
)
from pydantic import ValidationError

from terminal_bench_benchmark_service import eval_resume
from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark
from tests.test_runtime_lifecycle import FakeSandbox

TASK = "wdm-design"
DATASET = "terminal-bench-4.0"
ARCHIVE = b"archive"
ARCHIVE_SHA = hashlib.sha256(ARCHIVE).hexdigest()


class VerifierSandbox(FakeSandbox):
    """Answers the verifier's own commands: measured sizes, an unpack, a reward."""

    def __init__(self, sandbox_id: str, log: list[str]) -> None:
        super().__init__(sandbox_id)
        self.log = log

    async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> ExecResult:
        self.commands.append(command)
        self.log.append(f"verifier exec: {command[:40]}")
        if "reward" in command:
            return ExecResult(exit_code=0, output='{"reward": 1.0}\n')
        if "wc -c" in command or "gzip -dc" in command:
            return ExecResult(exit_code=0, output="10 1\n")
        return ExecResult(exit_code=0, output="")

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        self.log.append(f"verifier upload: {remote_path}")
        self.uploads[remote_path] = content


class FakeProvider(SandboxProvider):
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.created: list[SandboxCreateRequest] = []
        self.deleted: list[str] = []

    async def create_sandbox(self, request: SandboxCreateRequest) -> Sandbox:
        self.created.append(request)
        self.log.append("verifier created")
        return VerifierSandbox(f"verifier-{len(self.created)}", self.log)

    async def get_sandbox(self, instance_id: str) -> Sandbox:
        raise NotImplementedError

    async def delete_sandbox(self, instance_id: str) -> None:
        self.deleted.append(instance_id)
        self.log.append("verifier deleted")

    async def list_sandboxes(self, query: SandboxQuery) -> AsyncGenerator[Sandbox, None]:
        raise NotImplementedError
        yield


class LoggingAgentSandbox(FakeSandbox):
    def __init__(self, log: list[str]) -> None:
        super().__init__("agent")
        self.log = log

    @property
    def labels(self) -> dict[str, str]:
        return {"Id": "run-123", "Benchmark": "terminal-bench"}

    async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> ExecResult:
        self.log.append(f"agent exec: {command[:40]}")
        return await super().exec(command, cwd=cwd, timeout=timeout)


@pytest.fixture
def benchmark() -> TerminalBenchBenchmark:
    return asyncio.run(TerminalBenchBenchmark.create())


@pytest.fixture
def local_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv(eval_resume.LOCAL_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(eval_resume.BUCKET_ENV, raising=False)
    return tmp_path


def _state(benchmark: TerminalBenchBenchmark, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "version": 1,
        "run_id": "run-123",
        "task_id": TASK,
        "dataset": DATASET,
        "task_contract_sha256": benchmark._task_contract_sha256(TASK, DATASET),  # pyright: ignore[reportPrivateUsage]
        "labels": {"Id": "run-123"},
        "artifacts": [
            {
                "source": "/app/design.npy",
                "s3_key": eval_resume.artifact_key("run-123", TASK, ARCHIVE_SHA),
                "sha256": ARCHIVE_SHA,
                "size_bytes": len(ARCHIVE),
            },
            {"source": "/app/meta.json"},
        ],
    }
    fields.update(overrides)
    return fields


def _store_archive(root: Path, key: str, content: bytes = ARCHIVE) -> None:
    path = root.joinpath(*key.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _result(chunks: list[StreamChunk]) -> dict[str, Any]:
    results = [chunk for chunk in chunks if isinstance(chunk, StreamResultChunk)]
    assert len(results) == 1
    return results[0].data


# --- state model -----------------------------------------------------------


def test_state_rejects_non_canonical_keys_and_partial_artifacts(benchmark: TerminalBenchBenchmark) -> None:
    eval_resume.EvalResumeState.model_validate(_state(benchmark))

    artifacts = _state(benchmark)["artifacts"]
    with pytest.raises(ValidationError, match="canonical"):
        eval_resume.EvalResumeState.model_validate(
            _state(benchmark, artifacts=[{**artifacts[0], "s3_key": "other/prefix/x.tar.gz"}, artifacts[1]])
        )
    with pytest.raises(ValidationError, match="together"):
        eval_resume.EvalResumeState.model_validate(
            _state(benchmark, artifacts=[{"source": "/app/design.npy", "sha256": ARCHIVE_SHA}])
        )
    with pytest.raises(ValidationError, match="distinct"):
        eval_resume.EvalResumeState.model_validate(_state(benchmark, artifacts=[artifacts[1], artifacts[1]]))
    with pytest.raises(ValidationError, match="exact JSON integers"):
        eval_resume.EvalResumeState.model_validate(
            _state(benchmark, artifacts=[{**artifacts[0], "size_bytes": 7.0}, artifacts[1]])
        )
    with pytest.raises(ValidationError, match="identifiers"):
        eval_resume.EvalResumeState.model_validate(_state(benchmark, run_id="../escape"))
    with pytest.raises(ValidationError):
        eval_resume.EvalResumeState.model_validate(_state(benchmark, version=2))


def test_run_id_prefers_the_run_label_and_stays_path_safe() -> None:
    assert eval_resume.run_id_from_labels({"Id": "run-1"}, "sbx") == "run-1"
    assert eval_resume.run_id_from_labels(None, "sbx") == "sbx"
    hashed = eval_resume.run_id_from_labels({"Id": "a/b c"}, "sbx")
    assert "/" not in hashed and len(hashed) == 32


def test_load_artifact_checks_length_and_digest(local_store: Path) -> None:
    good = eval_resume.PersistedArtifact(
        source="/app/x", s3_key=eval_resume.artifact_key("r", TASK, ARCHIVE_SHA), sha256=ARCHIVE_SHA, size_bytes=7
    )
    _store_archive(local_store, good.s3_key or "", ARCHIVE + b"!")
    with pytest.raises(ValueError, match="byte-length"):
        asyncio.run(eval_resume.load_artifact(good))

    wrong_sha = "0" * 64
    tampered = eval_resume.PersistedArtifact(
        source="/app/x", s3_key=eval_resume.artifact_key("r", TASK, wrong_sha), sha256=wrong_sha, size_bytes=7
    )
    _store_archive(local_store, tampered.s3_key or "")
    with pytest.raises(ValueError, match="SHA-256"):
        asyncio.run(eval_resume.load_artifact(tampered))

    with pytest.raises(ValueError, match="not persisted"):
        asyncio.run(eval_resume.load_artifact(eval_resume.PersistedArtifact(source="/app/x")))


class FakeS3:
    """The two S3 calls the store makes, over an in-memory bucket."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.puts: list[dict[str, Any]] = []

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.puts.append(kwargs)
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = kwargs["Body"]
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": Key}}, "GetObject")
        body = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(body), "ContentLength": len(body)}


def test_s3_store_round_trips_and_reports_missing_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(eval_resume.LOCAL_DIR_ENV, raising=False)
    monkeypatch.setenv(eval_resume.BUCKET_ENV, "tb-bucket")
    s3 = FakeS3()
    monkeypatch.setattr(eval_resume, "_s3_client", lambda: s3)

    persisted = asyncio.run(eval_resume.persist_artifact("run-1", TASK, "/app/x", ARCHIVE))
    assert persisted.s3_key == eval_resume.artifact_key("run-1", TASK, ARCHIVE_SHA)
    assert s3.puts == [
        {"Bucket": "tb-bucket", "Key": persisted.s3_key, "Body": ARCHIVE, "ContentType": "application/gzip"}
    ]
    assert asyncio.run(eval_resume.load_artifact(persisted)) == ARCHIVE

    s3.objects.clear()
    with pytest.raises(RuntimeError, match="Failed to load"):
        asyncio.run(eval_resume.load_artifact(persisted))


# --- first evaluation ------------------------------------------------------


def test_isolated_evaluation_streams_state_after_persisting_and_before_the_verifier(
    benchmark: TerminalBenchBenchmark, local_store: Path
) -> None:
    log: list[str] = []
    provider = FakeProvider(log)
    agent = LoggingAgentSandbox(log)

    async def run() -> list[StreamChunk]:
        with sandbox_provider_scope(provider):
            return [
                chunk
                async for chunk in benchmark._evaluate_in_isolated_verifier(  # pyright: ignore[reportPrivateUsage]
                    TASK, agent, DATASET
                )
            ]

    chunks = asyncio.run(run())

    states = [chunk for chunk in chunks if isinstance(chunk, StreamEvalResumeStateChunk)]
    assert len(states) == 1
    state = eval_resume.EvalResumeState.model_validate(states[0].data)
    assert state.run_id == "run-123"
    assert state.labels == {"Id": "run-123", "Benchmark": "terminal-bench"}
    assert [artifact.source for artifact in state.artifacts] == ["/app/design.npy", "/app/meta.json"]
    for artifact in state.artifacts:
        assert artifact.present
        assert artifact.sha256 == ARCHIVE_SHA
        assert artifact.s3_key == eval_resume.artifact_key("run-123", TASK, ARCHIVE_SHA)
        assert local_store.joinpath(*artifact.s3_key.split("/")).read_bytes() == ARCHIVE

    # The state chunk precedes the verifier; a grading fault then has something to resume from.
    state_index = chunks.index(states[0])
    assert all(not isinstance(chunk, StreamResultChunk) for chunk in chunks[:state_index])
    before_verifier = log[: log.index("verifier created")]
    assert before_verifier and all(entry.startswith("agent exec") for entry in before_verifier)

    assert _result(chunks)["verifier_result"] == {"rewards": {"score": 1.0}, "output": ""}
    assert provider.created[0].labels == {"Id": "run-123", "Benchmark": "terminal-bench", "Role": "verifier"}
    assert provider.deleted == ["verifier-1"]
    # Restored from the store, not from the agent's sandbox.
    uploads = [entry for entry in log if entry.startswith("verifier upload")]
    assert len(uploads) == 2 and log.index("verifier created") < log.index(uploads[0])


def test_isolated_evaluation_without_a_store_is_a_grading_fault_before_any_verifier(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(eval_resume.LOCAL_DIR_ENV, raising=False)
    monkeypatch.delenv(eval_resume.BUCKET_ENV, raising=False)
    log: list[str] = []
    provider = FakeProvider(log)

    async def run() -> list[StreamChunk]:
        with sandbox_provider_scope(provider):
            return [
                chunk
                async for chunk in benchmark._evaluate_in_isolated_verifier(  # pyright: ignore[reportPrivateUsage]
                    TASK, LoggingAgentSandbox(log), DATASET
                )
            ]

    chunks = asyncio.run(run())

    assert not any(isinstance(chunk, StreamEvalResumeStateChunk) for chunk in chunks)
    assert not provider.created
    result = _result(chunks)
    assert result["verifier_result"] is None
    assert result["exception_info"] is not None
    assert eval_resume.BUCKET_ENV in result["exception_info"]
    assert any(isinstance(chunk, StreamErrorChunk) for chunk in chunks)


# --- eval-only retry -------------------------------------------------------


class FakeProviderConfig:
    def __init__(self, provider: FakeProvider) -> None:
        self.provider = provider

    def create_provider(self) -> FakeProvider:
        return self.provider


def _resume_request(
    benchmark: TerminalBenchBenchmark, provider: FakeProvider | None, **overrides: Any
) -> EvaluateResponseRequest:
    request = EvaluateResponseRequest(task_id=TASK, eval_resume_state=_state(benchmark, **overrides), dataset=DATASET)
    if provider is not None:
        object.__setattr__(request, "sandbox_provider", FakeProviderConfig(provider))
    return request


def _resume(benchmark: TerminalBenchBenchmark, request: EvaluateResponseRequest) -> list[StreamChunk]:
    async def run() -> list[StreamChunk]:
        return [chunk async for chunk in benchmark.stream_evaluate_response(request, dataset=DATASET)]

    return asyncio.run(run())


def test_resume_grades_persisted_artifacts_in_a_fresh_verifier(
    benchmark: TerminalBenchBenchmark, local_store: Path
) -> None:
    _store_archive(local_store, eval_resume.artifact_key("run-123", TASK, ARCHIVE_SHA))
    log: list[str] = []
    provider = FakeProvider(log)

    chunks = _resume(benchmark, _resume_request(benchmark, provider))

    states = [chunk for chunk in chunks if isinstance(chunk, StreamEvalResumeStateChunk)]
    assert len(states) == 1
    assert states[0].data == eval_resume.EvalResumeState.model_validate(_state(benchmark)).model_dump(mode="json")
    assert chunks.index(states[0]) < chunks.index(_first(chunks, StreamResultChunk))
    assert _result(chunks)["verifier_result"] == {"rewards": {"score": 1.0}, "output": ""}
    assert len(provider.created) == 1
    assert provider.created[0].labels == {"Id": "run-123", "Role": "verifier"}
    assert provider.created[0].name.startswith("tb-verifier-wdm-design-run-123-")
    assert provider.deleted == ["verifier-1"]
    assert len([entry for entry in log if entry.startswith("verifier upload")]) == 1
    assert not any(entry.startswith("agent exec") for entry in log)
    assert any(chunk.data == "Artifact not produced by the agent: /app/meta.json" for chunk in chunks)


def test_resume_reports_a_missing_archive_as_a_grading_fault_and_cleans_up(
    benchmark: TerminalBenchBenchmark, local_store: Path
) -> None:
    log: list[str] = []
    provider = FakeProvider(log)

    chunks = _resume(benchmark, _resume_request(benchmark, provider))

    result = _result(chunks)
    assert result["verifier_result"] is None
    assert result["exception_info"] is not None
    assert "FileNotFoundError" in result["exception_info"]
    assert provider.deleted == ["verifier-1"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"task_id": "ctr-optimization", "artifacts": [{"source": "/app/design.npy"}, {"source": "/app/meta.json"}]},
            "task_id mismatch",
        ),
        ({"dataset": "default"}, "dataset mismatch"),
        ({"task_contract_sha256": "0" * 64}, "task contract"),
        ({"artifacts": [{"source": "/app/meta.json"}]}, "declared artifacts"),
        ({"artifacts": [{"source": "/app/design.npy"}, {"source": "/app/meta.json"}, {"source": "/etc"}]}, "declared"),
        ({"version": "1"}, "not a terminal-bench resume state"),
    ],
)
def test_resume_rejects_state_that_does_not_fit_the_deployed_task(
    benchmark: TerminalBenchBenchmark, local_store: Path, overrides: dict[str, Any], message: str
) -> None:
    provider = FakeProvider([])
    with pytest.raises(ValueError, match=message):
        _resume(benchmark, _resume_request(benchmark, provider, **overrides))
    assert not provider.created


def test_resume_needs_a_provider_and_refuses_agent_sandbox_datasets(
    benchmark: TerminalBenchBenchmark, local_store: Path
) -> None:
    with pytest.raises(ValueError, match="sandbox_provider"):
        _resume(benchmark, _resume_request(benchmark, None))

    request = EvaluateResponseRequest(
        task_id=TASK, eval_resume_state=_state(benchmark, dataset="terminal-bench-2.1"), dataset="terminal-bench-2.1"
    )
    with pytest.raises(ValueError, match="grades in the agent's sandbox"):
        asyncio.run(_drain(benchmark.stream_evaluate_response(request, dataset="terminal-bench-2.1")))


def _first(chunks: list[StreamChunk], kind: type[StreamChunk]) -> StreamChunk:
    return next(chunk for chunk in chunks if isinstance(chunk, kind))


async def _drain(stream: AsyncGenerator[StreamChunk, None]) -> list[StreamChunk]:
    return [chunk async for chunk in stream]
