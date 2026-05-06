"""Smoke test that both terminal-bench datasets actually load from disk.

Requires submodules to be initialized (`make install-submodules`).
"""

import asyncio

from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark


def test_load_terminal_bench_2_0_and_2_1() -> None:
    datasets = asyncio.run(TerminalBenchBenchmark().load_datasets())

    assert set(datasets.keys()) == {"default", "terminal-bench-2.0", "terminal-bench-2.1"}
    assert datasets["default"] is datasets["terminal-bench-2.0"], "`default` must alias terminal-bench-2.0"

    for name in ("terminal-bench-2.0", "terminal-bench-2.1"):
        tasks = datasets[name]
        assert len(tasks) > 0, f"{name} loaded zero tasks"
        sample = next(iter(tasks.values()))
        assert "problem_statement" in sample
        assert "task_definition" in sample
