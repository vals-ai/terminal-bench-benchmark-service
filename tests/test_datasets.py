"""Smoke tests that the supported Terminal-Bench datasets load from disk."""

import asyncio
import json
from pathlib import Path

from benchmark_service import ComposeSource

from scripts.import_tbench4_release import build_manifest
from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark


def test_load_terminal_bench_datasets() -> None:
    benchmark = asyncio.run(TerminalBenchBenchmark.create())
    datasets = benchmark.datasets

    assert set(datasets.keys()) == {
        "default",
        "terminal-bench-2.0",
        "terminal-bench-2.1",
        "terminal-bench-4.0",
    }
    assert datasets["default"] is datasets["terminal-bench-2.1"], "`default` must alias terminal-bench-2.1"

    for name in ("terminal-bench-2.0", "terminal-bench-2.1", "terminal-bench-4.0"):
        tasks = datasets[name]
        assert len(tasks) > 0, f"{name} loaded zero tasks"
        sample = next(iter(tasks.values()))
        assert "problem_statement" in sample
        assert "task_definition" in sample

    assert len(datasets["terminal-bench-4.0"]) == 66


def test_terminal_bench_4_uses_pinned_images_and_preserves_resources() -> None:
    benchmark = asyncio.run(TerminalBenchBenchmark.create())

    task = asyncio.run(benchmark.retrieve_task("atrx-vep-crispr", dataset="terminal-bench-4.0"))
    assert task.source.image.startswith("harborframework/terminal-bench:")
    assert "@sha256:" in task.source.image
    assert task.cwd == "/app"
    assert task.resources.gpu == 0

    gpu_task = asyncio.run(benchmark.retrieve_task("fp8-rmsnorm-gemm", dataset="terminal-bench-4.0"))
    assert gpu_task.resources.gpu == 1
    assert gpu_task.resources.gpu_type == "H100"

    manifest = benchmark._image_manifest("terminal-bench-4.0")  # pyright: ignore[reportPrivateUsage]
    assert manifest["release_tag"] == "v4.0.0"
    assert len(manifest["tasks"]) == 66
    assert manifest["unsupported_tasks"] == []
    assert all("@sha256:" in entry["image"] for entry in manifest["tasks"].values())
    assert all("@sha256:" in entry["verifier_image"] for entry in manifest["tasks"].values())
    assert all(
        "@sha256:" in sidecar["image"] for entry in manifest["tasks"].values() for sidecar in entry.get("sidecars", [])
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


def test_terminal_bench_4_manifest_is_reproducible() -> None:
    source_manifest = Path("datasets/images/terminal-bench-4-prebuilt.json")
    generated_manifest = Path("datasets/images/terminal-bench-4.json")

    assert build_manifest(source_manifest, Path("datasets/terminal-bench-4/tasks")) == json.loads(
        generated_manifest.read_text()
    )
