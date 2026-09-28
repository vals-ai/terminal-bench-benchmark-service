"""Durable copies of the artifacts an isolated verifier grades, for eval-only retry.

TBench4 grades in a sandbox the agent never had: the declared artifacts are
packed in the agent's sandbox and re-materialized in a fresh verifier. Once
the agent sandbox is gone, only those packed archives can reproduce the
grading input, so each one is written to S3 before the verifier is created and
the state that names them is streamed to the tracker. A later eval-only retry
recreates the verifier from that state alone.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Literal, NotRequired, Protocol, TypedDict, cast

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from terminal_bench_benchmark_service import isolated_verifier

BUCKET_ENV = "TERMINAL_BENCH_EVAL_STATE_BUCKET"
LOCAL_DIR_ENV = "TERMINAL_BENCH_EVAL_STATE_LOCAL_DIR"
_ARTIFACT_PREFIX = "terminal-bench/eval-resume"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_LABEL_CHARS = 256


class _StreamingBody(Protocol):
    def read(self, amount: int = -1) -> bytes: ...

    def close(self) -> None: ...


class _GetObjectResponse(TypedDict):
    Body: _StreamingBody
    ContentLength: NotRequired[int]


class _S3Client(Protocol):
    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> object: ...

    def get_object(self, *, Bucket: str, Key: str) -> _GetObjectResponse: ...


class PersistedArtifact(BaseModel):
    """One declared artifact as the agent left it: an archive in S3, or absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    s3_key: str | None = None
    sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=0, le=isolated_verifier.MAX_ARTIFACT_BYTES)

    @field_validator("size_bytes", mode="before")
    @classmethod
    def validate_exact_integer(cls, value: object) -> object:
        if value is not None and type(value) is not int:
            raise ValueError("artifact sizes must use exact JSON integers")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("artifact digests must be lowercase SHA-256 values")
        return value

    @property
    def present(self) -> bool:
        return self.s3_key is not None

    @model_validator(mode="after")
    def validate_shape(self) -> PersistedArtifact:
        stored = (self.s3_key is not None, self.sha256 is not None, self.size_bytes is not None)
        if any(stored) and not all(stored):
            raise ValueError("a stored artifact needs its key, digest, and size together")
        return self


class EvalResumeState(BaseModel):
    """Everything an eval-only retry needs to grade a TBench4 submission again."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    run_id: str
    task_id: str
    dataset: str
    task_contract_sha256: str
    labels: dict[str, str] = Field(default_factory=dict)
    artifacts: list[PersistedArtifact]

    @field_validator("version", mode="before")
    @classmethod
    def validate_exact_integer(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("resume-state integer fields must use exact JSON integers")
        return value

    @field_validator("run_id", "task_id", "dataset")
    @classmethod
    def validate_component(cls, value: str) -> str:
        if not _SAFE_COMPONENT.fullmatch(value):
            raise ValueError("resume-state identifiers may contain only letters, numbers, '.', '_', and '-'")
        return value

    @field_validator("task_contract_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("resume-state digests must be lowercase SHA-256 values")
        return value

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, value: dict[str, str]) -> dict[str, str]:
        for key, label in value.items():
            if not key or len(key) > MAX_LABEL_CHARS or len(label) > MAX_LABEL_CHARS:
                raise ValueError("resume-state labels must be short, non-empty strings")
        return value

    @model_validator(mode="after")
    def validate_artifact_keys(self) -> EvalResumeState:
        sources = [artifact.source for artifact in self.artifacts]
        if len(set(sources)) != len(sources):
            raise ValueError("resume-state artifacts must have distinct sources")
        for artifact in self.artifacts:
            if artifact.s3_key is None or artifact.sha256 is None:
                continue
            if artifact.s3_key != artifact_key(self.run_id, self.task_id, artifact.sha256):
                raise ValueError("artifact s3_key does not match the canonical terminal-bench artifact path")
        return self


def artifact_key(run_id: str, task_id: str, sha256: str) -> str:
    """Return the only S3 key these validated state fields may address."""
    return f"{_ARTIFACT_PREFIX}/{run_id}/{task_id}/{sha256}.tar.gz"


def task_contract_sha256(task_definition: Mapping[str, Any], verifier_image: str) -> str:
    """Digest of the parts of a task that decide what the verifier grades."""
    contract = {
        "artifacts": task_definition.get("artifacts"),
        "verifier": task_definition.get("verifier"),
        "verifier_image": verifier_image,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def run_id_from_labels(labels: Mapping[str, str] | None, fallback: str) -> str:
    """The Valkyrie run id when the sandbox carries it, else the sandbox's own id."""
    candidate = (labels or {}).get("Id", "").strip() or fallback
    if not _SAFE_COMPONENT.fullmatch(candidate):
        candidate = hashlib.sha256(candidate.encode()).hexdigest()[:32]
    return candidate


def is_configured() -> bool:
    return bool(os.environ.get(BUCKET_ENV)) or _local_root() is not None


async def persist_artifact(run_id: str, task_id: str, source: str, content: bytes) -> PersistedArtifact:
    """Write one packed artifact where an eval-only retry can find it."""
    if len(content) > isolated_verifier.MAX_ARTIFACT_BYTES:
        raise ValueError(f"Artifact {source} exceeds the {isolated_verifier.MAX_ARTIFACT_BYTES} byte transfer limit")
    sha256 = hashlib.sha256(content).hexdigest()
    key = artifact_key(run_id, task_id, sha256)
    await _put_object(key, content)
    return PersistedArtifact(source=source, s3_key=key, sha256=sha256, size_bytes=len(content))


async def load_artifact(artifact: PersistedArtifact) -> bytes:
    """Fetch and integrity-check one persisted artifact archive."""
    if artifact.s3_key is None or artifact.sha256 is None or artifact.size_bytes is None:
        raise ValueError(f"Artifact {artifact.source} was not persisted")
    content = await _get_object(artifact.s3_key, artifact.size_bytes)
    if len(content) != artifact.size_bytes:
        raise ValueError(f"Persisted artifact {artifact.source} failed its byte-length integrity check")
    if hashlib.sha256(content).hexdigest() != artifact.sha256:
        raise ValueError(f"Persisted artifact {artifact.source} failed its SHA-256 integrity check")
    return content


def _local_root() -> Path | None:
    value = os.environ.get(LOCAL_DIR_ENV)
    return Path(value).expanduser() if value else None


def _local_path(key: str) -> Path:
    key_path = PurePosixPath(key)
    if key_path.is_absolute() or ".." in key_path.parts:
        raise ValueError("Invalid terminal-bench eval-resume artifact key")
    root = _local_root()
    if root is None:
        raise RuntimeError(f"{LOCAL_DIR_ENV} is not configured")
    return root.joinpath(*key_path.parts)


def _bucket() -> str:
    value = os.environ.get(BUCKET_ENV)
    if not value:
        raise RuntimeError(f"{BUCKET_ENV} is not configured")
    return value


def _s3_client() -> _S3Client:
    return cast(
        _S3Client,
        boto3.client(  # pyright: ignore[reportUnknownMemberType]
            "s3",
            region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
        ),
    )


async def _put_object(key: str, content: bytes) -> None:
    if _local_root() is not None:
        path = _local_path(key)

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

        await asyncio.to_thread(write)
        return

    bucket = _bucket()

    def put() -> None:
        _s3_client().put_object(Bucket=bucket, Key=key, Body=content, ContentType="application/gzip")

    try:
        await asyncio.to_thread(put)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to persist terminal-bench artifact at {key}") from exc


async def _get_object(key: str, expected_size: int) -> bytes:
    if _local_root() is not None:
        path = _local_path(key)

        def read_bounded() -> bytes:
            with path.open("rb") as handle:
                return handle.read(expected_size + 1)

        return await asyncio.to_thread(read_bounded)

    bucket = _bucket()

    def get() -> bytes:
        response = _s3_client().get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            content_length = response.get("ContentLength")
            if content_length is not None and content_length > expected_size:
                raise ValueError("Persisted terminal-bench artifact exceeds its declared byte length")
            return body.read(expected_size + 1)
        finally:
            body.close()

    try:
        content = await asyncio.to_thread(get)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to load terminal-bench artifact at {key}") from exc
    if len(content) > expected_size:
        raise ValueError("Persisted terminal-bench artifact exceeds its declared byte length")
    return content
