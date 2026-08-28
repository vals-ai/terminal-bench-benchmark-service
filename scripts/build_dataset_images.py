#!/usr/bin/env python3
"""Build and publish task images for datasets that ship only Dockerfiles.

Terminal-Bench 2.x tasks name an already-published image in their `task.toml`.
Newer datasets ship an `environment/Dockerfile` for the agent environment and a
`tests/Dockerfile` for the grader instead, so the images have to be built and
pushed before a run can start. This writes the resulting task -> image mapping
to the manifest the service reads at `retrieve_task` time.

Sandboxes pull anonymously, so the target registry must allow unauthenticated
pulls -- verifier images included. Verifier images carry the graders and their
reference data, so this is only acceptable because the upstream dataset
repository is already public. Do not reuse this script for a dataset whose tests
are not.

Usage:
    python scripts/build_dataset_images.py \\
        --dataset terminal-bench-science \\
        --registry docker.io/valsai \\
        [--tasks 3x2pt-inference,diag-chipseq] [--platform linux/amd64] [--no-push]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLATFORM = "linux/amd64"


@dataclass(frozen=True)
class TaskBuild:
    task_id: str
    directory: Path

    @property
    def environment_dir(self) -> Path:
        return self.directory / "environment"

    @property
    def tests_dir(self) -> Path:
        return self.directory / "tests"

    def is_compose(self) -> bool:
        """Multi-container tasks need the compose runtime, not a single image."""
        return any(
            (self.environment_dir / name).exists()
            for name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")
        )


def discover_tasks(tasks_root: Path) -> list[TaskBuild]:
    """Every directory under tasks_root holding a task.toml, keyed by its slug."""
    builds: dict[str, TaskBuild] = {}
    for task_toml in sorted(tasks_root.rglob("task.toml")):
        directory = task_toml.parent
        if directory.name in builds:
            raise SystemExit(f"Duplicate task slug {directory.name} in {tasks_root}")
        builds[directory.name] = TaskBuild(task_id=directory.name, directory=directory)
    return list(builds.values())


def dataset_version(dataset_root: Path) -> str:
    """Record which upstream commit the images were built from."""
    result = subprocess.run(
        ["git", "-C", str(dataset_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def build_timeout(task: TaskBuild) -> float | None:
    task_toml = task.directory / "task.toml"
    with open(task_toml, "rb") as handle:
        definition = tomllib.load(handle)
    timeout = definition.get("environment", {}).get("build_timeout_sec")
    return float(timeout) if isinstance(timeout, int | float) else None


def run(command: list[str], *, timeout: float | None) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, timeout=timeout)


def build_image(context: Path, tag: str, platform: str, timeout: float | None, push: bool) -> None:
    run(["docker", "build", "--platform", platform, "-t", tag, str(context)], timeout=timeout)
    if push:
        run(["docker", "push", tag], timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset directory name under datasets/")
    parser.add_argument("--registry", required=True, help="Registry and namespace, e.g. docker.io/valsai")
    parser.add_argument("--tag", default=None, help="Image tag (default: the dataset checkout's short commit)")
    parser.add_argument("--tasks", default="", help="Comma-separated task slugs (default: every task)")
    parser.add_argument("--platform", default=DEFAULT_PLATFORM)
    parser.add_argument("--no-push", action="store_true", help="Build locally without pushing")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest output path (default: datasets/images/<dataset>.json)",
    )
    args = parser.parse_args()

    dataset_root = REPO_ROOT / "datasets" / args.dataset
    tasks_root = dataset_root / "tasks"
    if not tasks_root.is_dir():
        raise SystemExit(f"No tasks directory at {tasks_root}; run `make install-submodules` first.")

    version = dataset_version(dataset_root)
    tag = args.tag or version[:12]
    manifest_path = Path(args.manifest) if args.manifest else REPO_ROOT / "datasets" / "images" / f"{args.dataset}.json"

    selected = {slug for slug in args.tasks.split(",") if slug}
    tasks = [task for task in discover_tasks(tasks_root) if not selected or task.task_id in selected]
    if selected:
        missing = selected - {task.task_id for task in tasks}
        if missing:
            raise SystemExit(f"Unknown task slugs: {', '.join(sorted(missing))}")

    entries: dict[str, dict[str, str]] = {}
    skipped: list[str] = []
    failed: list[str] = []

    for task in tasks:
        if task.is_compose():
            # Recorded rather than silently dropped: the run must be able to tell
            # "not supported yet" apart from "forgot to build it".
            skipped.append(task.task_id)
            print(f"skip {task.task_id}: multi-container task needs the compose runtime", flush=True)
            continue

        image = f"{args.registry}/tbs-env-{task.task_id}:{tag}"
        verifier_image = f"{args.registry}/tbs-verifier-{task.task_id}:{tag}"
        timeout = build_timeout(task)
        try:
            build_image(task.environment_dir, image, args.platform, timeout, not args.no_push)
            build_image(task.tests_dir, verifier_image, args.platform, timeout, not args.no_push)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            failed.append(task.task_id)
            print(f"FAILED {task.task_id}: {error}", file=sys.stderr, flush=True)
            continue

        entries[task.task_id] = {"image": image, "verifier_image": verifier_image}

    manifest = {
        "dataset": args.dataset,
        "dataset_commit": version,
        "tag": tag,
        "platform": args.platform,
        "unsupported_tasks": sorted(skipped),
        "tasks": dict(sorted(entries.items())),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nwrote {manifest_path}: {len(entries)} built, {len(skipped)} unsupported, {len(failed)} failed")
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
