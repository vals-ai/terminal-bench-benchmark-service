"""Smoke test that every terminal-bench dataset actually loads from disk.

Requires submodules to be initialized (`make install-submodules`).
"""

import asyncio

from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark


def test_load_terminal_bench_datasets() -> None:
    datasets = asyncio.run(TerminalBenchBenchmark().load_datasets())

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
    benchmark = TerminalBenchBenchmark()
    datasets = asyncio.run(benchmark.load_datasets())

    tasks = datasets["terminal-bench-science"]

    assert len(tasks) == 70, "dataset submodule is not at the pinned v0.1.0"
    assert "3x2pt-inference" in tasks, "task ids must be bare upstream slugs"
    assert not any("/" in task_id for task_id in tasks), "nested paths must not leak into task ids"

    task_dir = benchmark._task_dir("3x2pt-inference", "terminal-bench-science")
    assert task_dir.parts[-3:] == ("physical-sciences", "astronomy", "3x2pt-inference")
    assert (task_dir / "tests" / "test.sh").exists()
