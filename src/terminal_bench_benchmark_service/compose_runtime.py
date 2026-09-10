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
_REOWNED_TASK_IMAGE = "terminal-bench/main:reowned"
_UNMAPPED_OWNER_ERROR = "failed to Lchown"
_REOWN_SCRIPT = f"{_COMPOSE_ROOT}/reown.py"
_REOWN_SCRIPT_SOURCE = """\
import sys
import tarfile


def id_ranges(path):
    ranges = []
    with open(path) as id_map:
        for line in id_map:
            start, _, count = (int(field) for field in line.split())
            ranges.append(range(start, start + count))
    return ranges


uids = id_ranges(sys.argv[1])
gids = id_ranges(sys.argv[2])
with (
    tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as source,
    tarfile.open(fileobj=sys.stdout.buffer, mode="w|", format=tarfile.PAX_FORMAT) as target,
):
    for member in source:
        if not any(member.uid in ids for ids in uids):
            member.uid = 0
            member.pax_headers.pop("uid", None)
        if not any(member.gid in ids for ids in gids):
            member.gid = 0
            member.pax_headers.pop("gid", None)
        member.uname = member.gname = ""
        target.addfile(member, source.extractfile(member) if member.isreg() else None)
"""


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
    timeout = _build_timeout(resources)
    main_image = await _pull_task_image(sandbox, task_image, timeout)
    await _stage_files(task_dir / "environment", main_image, sidecar_images, resources, sandbox)

    compose = compose_runtime_source(task_id, task_image, sidecar_images).compose_command
    services = await _run(sandbox, f"{compose} config --services", timeout=60)
    if "main" not in services.output.splitlines():
        raise RuntimeError("Terminal-Bench compose runtime requires a `main` service")

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


async def _pull_task_image(sandbox: Sandbox, task_image: str, timeout: float) -> str:
    """Pull the task image, re-importing it with root-owned files when its owners are unmappable.

    The sandbox's user namespace maps a bounded ID range and the nested daemon
    rejects layers owned by IDs outside it, so such an image is flattened with
    ``crane export``, its unmapped owners rewritten, and re-imported locally
    with the original runtime config.
    """
    pull = await sandbox.exec(f"docker pull {shlex.quote(task_image)}", timeout=timeout)
    if pull.exit_code == 0:
        return task_image
    if _UNMAPPED_OWNER_ERROR not in pull.output:
        raise RuntimeError(f"Command failed: docker pull {task_image}\n{pull.output}")

    await _run(sandbox, "apk add --no-cache crane python3", timeout=300)
    config = await _run(sandbox, f"crane config {shlex.quote(task_image)}", timeout=120)
    changes = " ".join(f"-c {shlex.quote(change)}" for change in import_changes(json.loads(config.output)["config"]))
    await sandbox.upload_file(_REOWN_SCRIPT, _REOWN_SCRIPT_SOURCE.encode())
    await _run(
        sandbox,
        f"set -o pipefail && crane export {shlex.quote(task_image)} - "
        f"| python3 {_REOWN_SCRIPT} /proc/self/uid_map /proc/self/gid_map "
        f"| docker import {changes} - {_REOWNED_TASK_IMAGE}",
        timeout=timeout,
    )
    return _REOWNED_TASK_IMAGE


def import_changes(config: Mapping[str, Any]) -> list[str]:
    """Return the ``docker import -c`` instructions that carry an image's runtime config."""
    env: list[str] = config.get("Env") or []
    changes = [f"ENV {variable}" for variable in env]
    if config.get("WorkingDir"):
        changes.append(f"WORKDIR {config['WorkingDir']}")
    if config.get("User"):
        changes.append(f"USER {config['User']}")
    if config.get("Entrypoint"):
        changes.append(f"ENTRYPOINT {json.dumps(config['Entrypoint'])}")
    if config.get("Cmd"):
        changes.append(f"CMD {json.dumps(config['Cmd'])}")
    return changes


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
        "pull_policy": "never",
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
