"""Smoke test that every terminal-bench dataset actually loads from disk.

Requires submodules to be initialized (`make install-submodules`).
"""

import asyncio
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark


def test_load_terminal_bench_datasets() -> None:
    datasets = asyncio.run(TerminalBenchBenchmark.create()).datasets

    assert set(datasets.keys()) == {
        "default",
        "terminal-bench-2.0",
        "terminal-bench-2.1",
        "terminal-bench-science",
    }
    assert datasets["default"] is datasets["terminal-bench-2.1"], "`default` must alias terminal-bench-2.1"

    for name in ("terminal-bench-2.0", "terminal-bench-2.1", "terminal-bench-science"):
        tasks = datasets[name]
        assert len(tasks) > 0, f"{name} loaded zero tasks"
        sample = next(iter(tasks.values()))
        assert "problem_statement" in sample
        assert "task_definition" in sample


def test_terminal_bench_science_loads_nested_tasks_by_slug() -> None:
    """Science tasks nest as <domain>/<field>/<slug> but keep their slug as the id."""
    benchmark = asyncio.run(TerminalBenchBenchmark.create())

    tasks = benchmark.datasets["terminal-bench-science"]

    assert len(tasks) == 70, "dataset submodule is not at the pinned v0.1.0"
    assert "3x2pt-inference" in tasks, "task ids must be bare upstream slugs"
    assert not any("/" in task_id for task_id in tasks), "nested paths must not leak into task ids"

    task_dir = benchmark._task_dir("3x2pt-inference", "terminal-bench-science")
    assert task_dir.parts[-3:] == ("physical-sciences", "astronomy", "3x2pt-inference")
    assert (task_dir / "tests" / "test.sh").exists()


def test_science_task_retrieval_needs_an_image_manifest(tmp_path: Path) -> None:
    """A missing manifest must name itself rather than surface as a KeyError.

    Pointed at a path that cannot exist so the check holds both before the
    manifest is built and after it is committed.
    """
    benchmark = asyncio.run(TerminalBenchBenchmark.create())
    spec = benchmark._DATASETS["terminal-bench-science"]  # pyright: ignore[reportPrivateUsage]
    absent = replace(spec, image_manifest=Path(str(tmp_path)) / "not-built.json")

    with patch.dict(benchmark._DATASETS, {"terminal-bench-science": absent}):  # pyright: ignore[reportPrivateUsage]
        benchmark._image_manifests.pop("terminal-bench-science", None)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(FileNotFoundError, match="image manifest"):
            asyncio.run(benchmark.retrieve_task("3x2pt-inference", dataset="terminal-bench-science"))
