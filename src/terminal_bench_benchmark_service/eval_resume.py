"""Durable Terminal-Bench evaluation checkpoints."""

import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Literal, cast
from uuid import uuid4

from benchmark_service.sandbox import Sandbox, SandboxError
from benchmark_service.sandbox.daytona import DaytonaSandbox
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SNAPSHOT_PREFIX = "tb-eval-resume-v1"
SNAPSHOT_TIMEOUT_SECONDS = 600


def _task_binding(task_id: str, dataset: str) -> str:
    identity = json.dumps([dataset, task_id], separators=(",", ":"))
    return hashlib.sha256(identity.encode()).hexdigest()[:12]


def _snapshot_name(task_id: str, dataset: str, nonce: str) -> str:
    return f"{SNAPSHOT_PREFIX}-{_task_binding(task_id, dataset)}-{nonce}"


class EvalResumeState(BaseModel):
    """A task-bound, bearer-style pointer to an agent filesystem snapshot."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    task_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    snapshot_name: str = Field(pattern=rf"^{SNAPSHOT_PREFIX}-[0-9a-f]{{12}}-[0-9a-f]{{32}}$")

    @field_validator("version", mode="before")
    @classmethod
    def validate_exact_version(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError("eval_resume_state version must be exact integer 1")
        return value

    @classmethod
    def create(cls, task_id: str, dataset: str) -> "EvalResumeState":
        return cls(task_id=task_id, dataset=dataset, snapshot_name=_snapshot_name(task_id, dataset, uuid4().hex))

    @model_validator(mode="after")
    def require_canonical_snapshot_name(self) -> "EvalResumeState":
        nonce = self.snapshot_name.rsplit("-", 1)[-1]
        if self.snapshot_name != _snapshot_name(self.task_id, self.dataset, nonce):
            raise ValueError("eval_resume_state snapshot_name is not canonical for its task and dataset")
        return self


def resume_sandbox_name(state: EvalResumeState) -> str:
    return f"tb-eval-run-v1-{_task_binding(state.task_id, state.dataset)}-{uuid4().hex}"


async def create_daytona_snapshot(sandbox: Sandbox, snapshot_name: str) -> None:
    """Use Daytona's filesystem snapshot hook until CBS exposes one."""
    if not isinstance(sandbox, DaytonaSandbox):
        raise ValueError("Terminal-Bench eval resume requires a Daytona sandbox")

    inner = getattr(sandbox, "_sandbox", None)
    snapshot_hook = getattr(inner, "_experimental_create_snapshot", None)
    if not callable(snapshot_hook):
        raise SandboxError("The installed Daytona SDK does not support filesystem snapshots")

    create_snapshot = cast(Callable[[str, float | None], Awaitable[None]], snapshot_hook)
    await create_snapshot(snapshot_name, SNAPSHOT_TIMEOUT_SECONDS)
