"""Smoke tests that the supported Terminal-Bench datasets load from disk."""

import asyncio
import json
from pathlib import Path

import pytest
from benchmark_service import ComposeSource
from benchmark_service.v1_schemas import V1Task

from scripts.import_tbench4_release import build_manifest
import terminal_bench_benchmark_service.benchmark_service as service_module
from terminal_bench_benchmark_service.benchmark_service import (
    MAX_DAYTONA_DISK_GB,
    MAX_DAYTONA_VCPU,
    TerminalBenchBenchmark,
)


def test_load_terminal_bench_datasets() -> None:
    benchmark = asyncio.run(TerminalBenchBenchmark.create())
    datasets = benchmark.datasets

    assert set(datasets.keys()) == {
        "default",
        "terminal-bench-2.0",
        "terminal-bench-2.1",
        "terminal-bench-4.0",
    }
    assert datasets["default"] is datasets["terminal-bench-4.0"], "`default` must alias terminal-bench-4.0"
    assert len(datasets["default"]) == 66

    for name in ("terminal-bench-2.0", "terminal-bench-2.1", "terminal-bench-4.0"):
        tasks = datasets[name]
        assert len(tasks) > 0, f"{name} loaded zero tasks"
        sample = next(iter(tasks.values()))
        assert "problem_statement" in sample
        assert "task_definition" in sample

    assert len(datasets["terminal-bench-4.0"]) == 66


def test_terminal_bench_4_uses_pinned_images_and_preserves_resources() -> None:
    benchmark = asyncio.run(TerminalBenchBenchmark.create())

    default_task = asyncio.run(benchmark.retrieve_task("atrx-vep-crispr"))
    assert isinstance(default_task.source, ComposeSource)
    assert benchmark._task_image("atrx-vep-crispr", "terminal-bench-4.0").startswith(  # type: ignore[reportPrivateUsage]
        "harborframework/terminal-bench:"
    )
    assert default_task.cwd == "/app"

    task = asyncio.run(benchmark.retrieve_task("atrx-vep-crispr", dataset="terminal-bench-4.0"))
    assert isinstance(task.source, ComposeSource)
    task_image = benchmark._task_image("atrx-vep-crispr", "terminal-bench-4.0")  # type: ignore[reportPrivateUsage]
    assert task_image.startswith("harborframework/terminal-bench:")
    assert "@sha256:" in task_image
    assert task.cwd == "/app"
    assert task.resources.gpu == 0

    gpu_task = asyncio.run(benchmark.retrieve_task("fp8-rmsnorm-gemm", dataset="terminal-bench-4.0"))
    assert gpu_task.resources.gpu == 1
    assert gpu_task.resources.gpu_type == "H100"

    jax_task = asyncio.run(benchmark.retrieve_task("jax-speedrun-gpu", dataset="terminal-bench-4.0"))
    assert jax_task.resources.disk == MAX_DAYTONA_DISK_GB

    database_task = asyncio.run(benchmark.retrieve_task("live-database-cutover", dataset="terminal-bench-4.0"))
    assert database_task.resources.vcpu == MAX_DAYTONA_VCPU
    assert benchmark._verifier_resources("live-database-cutover", "terminal-bench-4.0").vcpu == MAX_DAYTONA_VCPU  # pyright: ignore[reportPrivateUsage]

    manifest = benchmark._image_manifest("terminal-bench-4.0")  # pyright: ignore[reportPrivateUsage]
    assert manifest["release_tag"] == "v4.0.0"
    assert len(manifest["tasks"]) == 66
    assert manifest["unsupported_tasks"] == []
    assert all("@sha256:" in entry["image"] for entry in manifest["tasks"].values())
    assert all("@sha256:" in entry["verifier_image"] for entry in manifest["tasks"].values())
    assert all(
        "@sha256:" in sidecar["image"] for entry in manifest["tasks"].values() for sidecar in entry.get("sidecars", [])
    )


def test_tbench4_does_not_silently_cap_non_daytona_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = asyncio.run(TerminalBenchBenchmark.create())
    monkeypatch.setattr(service_module, "_request_sandbox_provider", lambda: object())

    with pytest.raises(ValueError, match="bound provider"):
        benchmark._sandbox_resources(  # pyright: ignore[reportPrivateUsage]
            "jax-speedrun-gpu",
            "terminal-bench-4.0",
            {"cpus": 1, "memory_mb": 1024, "storage_mb": 1000 * 1024},
        )


def test_terminal_bench_4_retrieves_every_task_and_preserves_runtime_features() -> None:
    benchmark = asyncio.run(TerminalBenchBenchmark.create())

    task_ids = sorted(benchmark.datasets["terminal-bench-4.0"])
    responses = [asyncio.run(benchmark.retrieve_task(task_id, dataset="terminal-bench-4.0")) for task_id in task_ids]
    assert len(responses) == 66
    assert all(response.source is not None for response in responses)

    compose_response = asyncio.run(benchmark.retrieve_task("ctr-optimization", dataset="terminal-bench-4.0"))
    assert isinstance(compose_response.source, ComposeSource)
    assert "runtime.json" in compose_response.source.compose_command

    image_response = asyncio.run(benchmark.retrieve_task("atrx-vep-crispr", dataset="terminal-bench-4.0"))
    assert isinstance(image_response.source, ComposeSource)

    assert benchmark._gradeable_artifacts("shadow-relay", "terminal-bench-4.0")  # pyright: ignore[reportPrivateUsage]
    artifacts = benchmark._gradeable_artifacts("vba-userform-port", "terminal-bench-4.0")  # pyright: ignore[reportPrivateUsage]
    assert artifacts[0].exclude == (
        "node_modules",
        ".venv",
        ".vite",
        ".git",
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        ".npm",
        ".cache",
    )


def test_list_tasks_exposes_all_tbench4_tasks_through_the_v1_contract() -> None:
    benchmark = asyncio.run(TerminalBenchBenchmark.create())

    tasks = asyncio.run(benchmark.list_tasks(dataset="terminal-bench-4.0"))

    assert len(tasks) == 66
    assert [task.id for task in tasks] == sorted(benchmark.datasets["terminal-bench-4.0"])
    assert all(isinstance(task, V1Task) for task in tasks)
    assert all(task.question and task.timeout is not None for task in tasks)
    assert all(set(task.model_dump()) == {"id", "question", "timeout"} for task in tasks)


def test_terminal_bench_4_manifest_is_reproducible() -> None:
    source_manifest = Path("datasets/images/terminal-bench-4-prebuilt.json")
    generated_manifest = Path("datasets/images/terminal-bench-4.json")

    assert build_manifest(source_manifest, Path("datasets/terminal-bench-4/tasks")) == json.loads(
        generated_manifest.read_text()
    )
