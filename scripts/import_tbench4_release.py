#!/usr/bin/env python3
"""Convert Harbor's TBench4 prebuilt-image release into a service manifest.

The upstream release asset describes each task as a list of environment,
verifier, and (where applicable) sidecar images. The benchmark service needs a
smaller task -> image mapping and the image working directory, which is not
included in that asset. This script derives the latter from the pinned source
Dockerfiles and records sidecar tasks as unsupported until the service can run
the compose runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any, cast

_DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}$")
_WORKDIR_RE = re.compile(r"^WORKDIR\s+(\S+)\s*$", re.IGNORECASE)


def _workdir(task_dir: Path) -> str:
    dockerfile = task_dir / "environment" / "Dockerfile"
    workdirs: list[str] = []
    for raw_line in dockerfile.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        match = _WORKDIR_RE.fullmatch(line)
        if match:
            workdirs.append(match.group(1))
    if not workdirs:
        raise ValueError(f"{dockerfile} has no WORKDIR")
    return workdirs[-1]


def _image(images: list[dict[str, Any]], kind: str, task_id: str) -> dict[str, Any]:
    matches = [image for image in images if image.get("kind") == kind]
    if len(matches) != 1:
        raise ValueError(f"{task_id} has {len(matches)} {kind} images; expected exactly one")
    image = matches[0]
    pinned_ref = image.get("pinned_ref")
    if not isinstance(pinned_ref, str) or not _DIGEST_RE.search(pinned_ref):
        raise ValueError(f"{task_id} {kind} image is not digest-pinned")
    return image


def build_manifest(source_manifest_path: Path, tasks_root: Path) -> dict[str, Any]:
    source_manifest = cast(dict[str, Any], json.loads(source_manifest_path.read_text()))
    raw_tasks_value = source_manifest.get("tasks")
    if not isinstance(raw_tasks_value, list):
        raise ValueError("TBench4 release manifest must contain a task list")
    raw_tasks = cast(list[object], raw_tasks_value)

    tasks: dict[str, dict[str, Any]] = {}
    unsupported_tasks: list[str] = []
    source_sha = "unknown"
    for raw_task_value in raw_tasks:
        if not isinstance(raw_task_value, dict):
            raise ValueError(f"Invalid task entry: {raw_task_value!r}")
        raw_task = cast(dict[str, Any], raw_task_value)
        task_id = raw_task.get("task")
        if source_sha == "unknown" and isinstance(raw_task.get("source_sha"), str):
            source_sha = raw_task["source_sha"]
        images_value = raw_task.get("images")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"Invalid task id: {task_id!r}")
        if task_id in tasks:
            raise ValueError(f"Duplicate task id: {task_id}")
        if not isinstance(images_value, list):
            raise ValueError(f"{task_id} has an invalid image list")
        images: list[dict[str, Any]] = []
        for image_value in cast(list[object], images_value):
            if not isinstance(image_value, dict):
                raise ValueError(f"{task_id} has an invalid image entry: {image_value!r}")
            images.append(cast(dict[str, Any], image_value))

        environment = _image(images, "environment", task_id)
        verifier = _image(images, "verifier", task_id)
        sidecars = [image for image in images if image.get("kind") == "sidecar"]
        task_definition = tomllib.loads((tasks_root / task_id / "task.toml").read_text())
        verifier_definition: Any = task_definition.get("verifier", {})
        has_collect_hooks = False
        if isinstance(verifier_definition, dict):
            verifier_table = cast(dict[str, Any], verifier_definition)
            has_collect_hooks = bool(verifier_table.get("collect"))
        unsupported_artifact_keys: set[str] = set()
        artifacts: Any = task_definition.get("artifacts")
        if isinstance(artifacts, list):
            for artifact_value in cast(list[object], artifacts):
                if isinstance(artifact_value, dict):
                    artifact = cast(dict[str, Any], artifact_value)
                    unsupported_artifact_keys.update(str(key) for key in artifact if key not in {"source", "service"})

        entry: dict[str, Any] = {
            "image": environment["pinned_ref"],
            "verifier_image": verifier["pinned_ref"],
            "workdir": _workdir(tasks_root / task_id),
        }
        unsupported_reasons: list[str] = []
        if sidecars:
            unsupported_reasons.append("requires sidecar services, which this service does not provision yet")
            entry["sidecars"] = [
                {
                    "role": image.get("role"),
                    "services": image.get("compose_services", []),
                }
                for image in sidecars
            ]
        if has_collect_hooks:
            unsupported_reasons.append("uses verifier.collect hooks, which this service does not execute yet")
        if unsupported_artifact_keys:
            keys = ", ".join(sorted(unsupported_artifact_keys))
            unsupported_reasons.append(f"declares artifacts with unsupported keys ({keys})")
        if unsupported_reasons:
            entry["unsupported_reason"] = f"TBench4 task {'; '.join(unsupported_reasons)}"
            unsupported_tasks.append(task_id)
        tasks[task_id] = entry

    return {
        "schema_version": 1,
        "dataset": "terminal-bench-4.0",
        "dataset_package": source_manifest.get("dataset_package"),
        "release_tag": source_manifest.get("release_tag"),
        "source_repository": "https://github.com/harbor-framework/terminal-bench",
        "source_sha": source_sha,
        "source_manifest_sha256": hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
        "image_repository": source_manifest.get("image_repository"),
        "task_count": len(tasks),
        "image_count": source_manifest.get("image_count"),
        "unsupported_tasks": sorted(unsupported_tasks),
        "tasks": dict(sorted(tasks.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--tasks-root", type=Path, default=Path("datasets/terminal-bench-4/tasks"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.source_manifest, args.tasks_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}: {manifest['task_count']} tasks, {len(manifest['unsupported_tasks'])} unsupported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
