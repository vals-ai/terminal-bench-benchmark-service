"""Run Terminal-Bench 4 compose tasks inside a sandbox-local Docker daemon."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from benchmark_service import ComposeSandbox, ComposeSource, ImageSource, Sandbox
from benchmark_service.sandbox.types import ExecResult

_COMPOSE_ROOT = "/terminal-bench"
_ENVIRONMENT_ROOT = f"{_COMPOSE_ROOT}/environment"
_COMPOSE_FILE = f"{_COMPOSE_ROOT}/task.json"
_RUNTIME_FILE = f"{_COMPOSE_ROOT}/runtime.json"
_BUNDLE_DIR = "/bundle"
_DIND_IMAGE = "docker:28.3.3-dind@sha256:a56b3bdde89315ed2cc0e4906e582b5033d93bf20d9cb9510c2cdd4e7f7690b1"
_DOCKER_READY_ATTEMPTS = 30
_DOCKER_READY_INTERVAL_SECONDS = 1.0
_DEFAULT_READINESS_TIMEOUT_SECONDS = 60.0
_EMPTY_COMPOSE_FILE = b'{"services":{"main":{}}}\n'


def compose_runtime_source(task_id: str, task_image: str, sidecar_images: Mapping[str, str]) -> ComposeSource:
    """Build the source describing a Terminal-Bench compose runtime.

    The benchmark-service contract creates the outer image and leaves the
    service responsible for starting Docker Compose inside it. The command is
    deliberately self-contained: it only references paths staged by
    :func:`start_compose_runtime` and digest-pinned images from the release
    manifest.
    """

    project = _project_name(task_id)
    command = " ".join(
        [
            "docker",
            "compose",
            "--project-name",
            shlex.quote(project),
            "--project-directory",
            shlex.quote(_ENVIRONMENT_ROOT),
            "--file",
            shlex.quote(_COMPOSE_FILE),
            "--file",
            shlex.quote(_RUNTIME_FILE),
        ]
    )
    return ComposeSource(outer=ImageSource(image=_DIND_IMAGE), compose_command=command)


def compose_service_sandbox(outer: Sandbox, source: ComposeSource, service: str) -> ComposeSandbox:
    """Route operations through one service in the compose runtime."""

    return ComposeSandbox(outer, source.model_copy(update={"service": service}))


async def start_compose_runtime(
    task_dir: Path,
    task_id: str,
    task_image: str,
    sidecar_images: Mapping[str, str],
    resources: Mapping[str, Any],
    sandbox: Sandbox,
) -> None:
    """Stage a task and start its main and sidecar services.

    All service images are published by the pinned TBench4 release. The task's
    compose file remains authoritative for dependencies, health checks,
    networking, and environment; the runtime overlay supplies immutable images
    and mounts the agent bundle. ``--no-build`` prevents Compose from
    rebuilding the source-level ``build`` entries that remain in the task file.
    """

    await _run(sandbox, "dockerd-entrypoint.sh dockerd > /var/log/dockerd.log 2>&1 &", timeout=10)
    await _wait_for_docker(sandbox)
    await _stage_files(task_dir / "environment", task_image, sidecar_images, resources, sandbox)

    compose = compose_runtime_source(task_id, task_image, sidecar_images).compose_command
    services = await _run(sandbox, f"{compose} config --services", timeout=60)
    if "main" not in services.output.splitlines():
        raise RuntimeError("Terminal-Bench compose runtime requires a `main` service")

    timeout = _build_timeout(resources)
    await _run(sandbox, f"{compose} pull", timeout=timeout)
    await _run(sandbox, f"{compose} up -d --no-build", timeout=timeout)
    prepare_main = "mkdir -p /bundle /logs/agent /logs/verifier /logs/terminus2 && chmod -R a+rwX /bundle /logs"
    await _run(sandbox, f"{compose} exec -T -u 0 main sh -lc {shlex.quote(prepare_main)}", timeout=60)
    await _run(sandbox, f"{compose} exec -T main true", timeout=_readiness_timeout(resources))


async def stop_compose_runtime(task_id: str, sandbox: Sandbox) -> None:
    """Stop and remove task services without masking evaluation failures."""
    try:
        result = await sandbox.exec(
            f"docker compose --project-name {shlex.quote(_project_name(task_id))} "
            f"--project-directory {shlex.quote(_ENVIRONMENT_ROOT)} "
            f"--file {shlex.quote(_COMPOSE_FILE)} --file {shlex.quote(_RUNTIME_FILE)} "
            "down --volumes --remove-orphans",
            timeout=120,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Could not stop Terminal-Bench compose runtime: {result.output[-1000:]}")
    finally:
        # A partial startup can leave dockerd alive even when Compose cannot
        # parse or tear down the incomplete project. Never let that daemon
        # survive a failed setup/evaluation cleanup attempt.
        try:
            await sandbox.exec("pkill -TERM dockerd || true", timeout=10)
        except Exception:
            pass


async def stop_compose_main(task_id: str, sandbox: Sandbox) -> None:
    """Stop the agent service before collecting state from dependent sidecars."""
    result = await sandbox.exec(
        f"docker compose --project-name {shlex.quote(_project_name(task_id))} "
        f"--project-directory {shlex.quote(_ENVIRONMENT_ROOT)} "
        f"--file {shlex.quote(_COMPOSE_FILE)} --file {shlex.quote(_RUNTIME_FILE)} "
        "stop main",
        timeout=120,
    )
    if result.exit_code != 0:
        raise RuntimeError(f"Could not stop Terminal-Bench main service: {result.output[-1000:]}")


async def _stage_files(
    environment_dir: Path,
    task_image: str,
    sidecar_images: Mapping[str, str],
    resources: Mapping[str, Any],
    sandbox: Sandbox,
) -> None:
    await _run(sandbox, f"mkdir -p {shlex.quote(_ENVIRONMENT_ROOT)}", timeout=30)
    runtime = runtime_compose_definition(task_image, sidecar_images, resources)
    compose_file = environment_dir / "docker-compose.yaml"
    compose_content = compose_file.read_bytes() if compose_file.is_file() else _EMPTY_COMPOSE_FILE
    await sandbox.upload_file(_COMPOSE_FILE, compose_content)
    await sandbox.upload_file(_RUNTIME_FILE, json.dumps(runtime, sort_keys=True).encode())
    await _run(sandbox, f"mkdir -p {shlex.quote(_BUNDLE_DIR)}", timeout=30)


def runtime_compose_definition(
    task_image: str, sidecar_images: Mapping[str, str], resources: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the JSON Compose overlay used to run one pinned task."""
    main: dict[str, Any] = {
        "image": task_image,
        "pull_policy": "always",
        "command": ["sh", "-c", "sleep infinity"],
        "volumes": [f"{_BUNDLE_DIR}:{_BUNDLE_DIR}"],
        "deploy": {
            "resources": {
                "limits": {
                    "cpus": str(resources.get("cpus", 1)),
                    "memory": f"{resources.get('memory_mb', 1024)}m",
                }
            }
        },
    }
    gpu_count = int(resources.get("gpu", resources.get("gpus", 0)) or 0)
    if gpu_count:
        # The Daytona GPU allocation belongs to the outer sandbox. The pinned
        # DIND image has no NVIDIA runtime, so a nested --gpus/device
        # reservation would fail before the task starts. Privileged mode passes
        # the outer device namespace through to the task container.
        main["privileged"] = True

    return {
        "services": {
            "main": main,
            **{service: {"image": image, "pull_policy": "always"} for service, image in sorted(sidecar_images.items())},
        }
    }


def _project_name(task_id: str) -> str:
    project = re.sub(r"[^a-z0-9_-]", "-", task_id.lower())
    return project if project and project[0].isalnum() else f"0{project}"


def _build_timeout(resources: Mapping[str, Any]) -> float:
    value: Any = resources.get("build_timeout_sec", 600)
    return max(float(value), 60.0)


def _readiness_timeout(resources: Mapping[str, Any]) -> float:
    healthcheck_value: Any = resources.get("healthcheck")
    if not isinstance(healthcheck_value, dict):
        return _DEFAULT_READINESS_TIMEOUT_SECONDS
    healthcheck = cast(dict[str, Any], healthcheck_value)
    return max(
        _DEFAULT_READINESS_TIMEOUT_SECONDS,
        float(healthcheck.get("start_period_sec", 0))
        + float(healthcheck.get("retries", 0))
        * (float(healthcheck.get("interval_sec", 0)) + float(healthcheck.get("timeout_sec", 0))),
    )


async def _wait_for_docker(sandbox: Sandbox) -> None:
    for attempt in range(_DOCKER_READY_ATTEMPTS):
        result = await sandbox.exec("docker info", timeout=10)
        if result.exit_code == 0:
            return
        if attempt < _DOCKER_READY_ATTEMPTS - 1:
            await asyncio.sleep(_DOCKER_READY_INTERVAL_SECONDS)
    raise RuntimeError("Docker daemon did not become ready inside the compose sandbox")


async def _run(sandbox: Sandbox, command: str, *, timeout: float) -> ExecResult:
    result = await sandbox.exec(command, timeout=timeout)
    if result.exit_code != 0:
        raise RuntimeError(f"Command failed: {command}\n{result.output}")
    return result
