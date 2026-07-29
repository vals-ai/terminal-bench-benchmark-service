"""Durable Terminal-Bench evaluation checkpoints."""

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, Literal, cast
from uuid import uuid4

from benchmark_service.sandbox import Sandbox, SandboxError
from benchmark_service.sandbox.daytona import DaytonaSandbox
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SNAPSHOT_PREFIX = "tb-eval-resume-v1"
SNAPSHOT_TIMEOUT_SECONDS = 600
SNAPSHOT_JANITOR_TIMEOUT_SECONDS = 60
SNAPSHOT_RETENTION_SECONDS = 30 * 24 * 60 * 60


def _task_binding(task_id: str, dataset: str, task_contract_sha256: str) -> str:
    identity = json.dumps([dataset, task_id, task_contract_sha256], separators=(",", ":"))
    return hashlib.sha256(identity.encode()).hexdigest()[:12]


def _snapshot_name(task_id: str, dataset: str, task_contract_sha256: str, nonce: str) -> str:
    return f"{SNAPSHOT_PREFIX}-{_task_binding(task_id, dataset, task_contract_sha256)}-{nonce}"


class EvalResumeState(BaseModel):
    """A task-bound, bearer-style pointer to an agent filesystem snapshot."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    task_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_name: str = Field(pattern=rf"^{SNAPSHOT_PREFIX}-[0-9a-f]{{12}}-[0-9a-f]{{32}}$")

    @field_validator("version", mode="before")
    @classmethod
    def validate_exact_version(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError("eval_resume_state version must be exact integer 1")
        return value

    @classmethod
    def create(
        cls,
        task_id: str,
        dataset: str,
        run_id: str,
        task_contract_sha256: str,
    ) -> "EvalResumeState":
        nonce = f"{int(time.time()):08x}{uuid4().hex[:24]}"
        return cls(
            task_id=task_id,
            dataset=dataset,
            run_id=run_id,
            task_contract_sha256=task_contract_sha256,
            snapshot_name=_snapshot_name(task_id, dataset, task_contract_sha256, nonce),
        )

    @model_validator(mode="after")
    def require_canonical_snapshot_name(self) -> "EvalResumeState":
        nonce = self.snapshot_name.rsplit("-", 1)[-1]
        if self.snapshot_name != _snapshot_name(
            self.task_id,
            self.dataset,
            self.task_contract_sha256,
            nonce,
        ):
            raise ValueError("eval_resume_state snapshot_name is not canonical for its task and dataset")
        return self


def resume_sandbox_name(state: EvalResumeState) -> str:
    binding = _task_binding(state.task_id, state.dataset, state.task_contract_sha256)
    return f"tb-eval-run-v1-{binding}-{uuid4().hex}"


def _snapshot_created_at(snapshot_name: str) -> int | None:
    if not snapshot_name.startswith(f"{SNAPSHOT_PREFIX}-"):
        return None
    try:
        return int(snapshot_name.rsplit("-", 1)[-1][:8], 16)
    except ValueError:
        return None


async def cleanup_expired_daytona_snapshots(provider: object, now_seconds: int | None = None) -> None:
    """Best-effort age-based cleanup for snapshots owned by this feature."""
    daytona = cast(Any, getattr(provider, "_daytona", None))
    snapshot_service = getattr(daytona, "snapshot", None)
    api_client = getattr(daytona, "_api_client", None)
    if snapshot_service is None and api_client is None:
        inner = getattr(provider, "_sandbox", None)
        sandbox_api = getattr(inner, "_sandbox_api", None)
        api_client = getattr(sandbox_api, "api_client", None)

    if api_client is not None:
        from daytona_api_client_async import SnapshotsApi

        snapshot_service = SnapshotsApi(api_client)

        async def list_snapshots(page: int) -> Any:
            return await snapshot_service.get_all_snapshots(
                page=page,
                limit=100,
                name=f"{SNAPSHOT_PREFIX}-",
            )

        async def delete_snapshot(snapshot: Any) -> None:
            await snapshot_service.remove_snapshot(snapshot.id)

    elif snapshot_service is not None:

        async def list_snapshots(page: int) -> Any:
            return await snapshot_service.list(page=page, limit=100)

        async def delete_snapshot(snapshot: Any) -> None:
            await snapshot_service.delete(snapshot)

    else:
        return

    cutoff = (int(time.time()) if now_seconds is None else now_seconds) - SNAPSHOT_RETENTION_SECONDS
    page = 1
    expired: list[Any] = []
    while True:
        result = await list_snapshots(page)
        expired.extend(
            snapshot
            for snapshot in result.items
            if (created_at := _snapshot_created_at(snapshot.name)) is not None and created_at < cutoff
        )
        if page >= result.total_pages:
            break
        page += 1
    for snapshot in expired:
        await delete_snapshot(snapshot)


async def create_daytona_snapshot(sandbox: Sandbox, snapshot_name: str) -> None:
    """Use Daytona's filesystem snapshot hook until CBS exposes one."""
    if not isinstance(sandbox, DaytonaSandbox):
        raise ValueError("Terminal-Bench eval resume requires a Daytona sandbox")

    inner = getattr(sandbox, "_sandbox", None)
    snapshot_hook = getattr(inner, "_experimental_create_snapshot", None)
    if not callable(snapshot_hook):
        raise SandboxError("The installed Daytona SDK does not support filesystem snapshots")

    create_snapshot = cast(Callable[[str, float | None], Awaitable[None]], snapshot_hook)
    try:
        await create_snapshot(snapshot_name, SNAPSHOT_TIMEOUT_SECONDS)
    except (Exception, asyncio.CancelledError):
        sandbox_api = getattr(inner, "_sandbox_api", None)
        api_client = getattr(sandbox_api, "api_client", None)
        if api_client is not None:
            with suppress(Exception):
                from daytona_api_client_async import SnapshotsApi

                snapshots = SnapshotsApi(api_client)
                snapshot = await snapshots.get_snapshot(snapshot_name)
                await snapshots.remove_snapshot(snapshot.id)
        raise
