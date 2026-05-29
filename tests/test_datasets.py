"""Smoke test that both terminal-bench datasets actually load from disk.

Requires submodules to be initialized (`make install-submodules`).
"""

import asyncio

import pytest

from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark


def test_calculate_final_score_uses_mean_terminal_reward() -> None:
    result = asyncio.run(
        TerminalBenchBenchmark().calculate_final_score(
            {
                "task-1": {"verifier_result": {"rewards": {"score": 1.0}}},
                "task-2": {"verifier_result": {"rewards": {"score": 0.25}}},
                "task-3": None,
            }
        )
    )

    assert result.score == 41.66666666666667
    assert result.metadata["total_tasks"] == 3
    assert result.metadata["resolved_tasks"] == 1
    assert result.metadata["unresolved_tasks"] == 2
    assert (
        result.metadata["score_types"]["score"]["description"]
        == "Mean Terminal-Bench verifier reward across requested tasks."
    )
    population = result.metadata["results"]["full"]
    assert population["aggregated_metrics"]["total"]["extra"]["mean_reward"] == pytest.approx(result.score / 100)
    tasks = result.metadata["tasks"]
    assert [task["task_id"] for task in tasks] == ["task-1", "task-2", "task-3"]
    assert [task["status"] for task in tasks] == ["resolved", "unresolved", "unresolved"]
    assert [task["scores"]["score"]["value"] for task in tasks] == [100.0, 25.0, 0.0]
    assert [task["aggregated_metrics"]["metadata"]["mean_reward"] for task in tasks] == [1.0, 0.25, 0.0]


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
