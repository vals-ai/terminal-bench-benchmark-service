"""Tests for the multi-dataset support in TerminalBenchBenchmark.

These tests don't talk to Daytona — they exercise the dataset selection logic
(`_get_dataset_location`, `_load_tasks_from_directory`, `load_datasets`) that
routes the `dataset` parameter through `retrieve_task` / `_copy_test_files`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark


@pytest.fixture
def benchmark() -> TerminalBenchBenchmark:
    return TerminalBenchBenchmark()


def test_dataset_locations_registered(benchmark: TerminalBenchBenchmark) -> None:
    """`default`, `terminal-bench-2.0`, and `terminal-bench-2.1` must all be registered."""
    assert "default" in benchmark._DATASET_LOCATIONS
    assert "terminal-bench-2.0" in benchmark._DATASET_LOCATIONS
    assert "terminal-bench-2.1" in benchmark._DATASET_LOCATIONS


def test_default_is_alias_for_terminal_bench_2_0(benchmark: TerminalBenchBenchmark) -> None:
    """The `default` dataset must resolve to the same on-disk path as `terminal-bench-2.0`."""
    assert benchmark._get_dataset_location("default") == benchmark._get_dataset_location("terminal-bench-2.0")


def test_terminal_bench_2_1_path_uses_tasks_subdirectory(benchmark: TerminalBenchBenchmark) -> None:
    """terminal-bench-2.1 stores task directories under a `tasks/` subdir, unlike 2.0."""
    location = benchmark._get_dataset_location("terminal-bench-2.1")
    assert location == Path("datasets/terminal-bench-2.1/tasks")
    assert benchmark._get_dataset_location("terminal-bench-2.0") != location


def test_get_dataset_location_none_returns_default(benchmark: TerminalBenchBenchmark) -> None:
    assert benchmark._get_dataset_location(None) == benchmark._get_dataset_location("default")


def test_get_dataset_location_unknown_raises(benchmark: TerminalBenchBenchmark) -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        benchmark._get_dataset_location("not-a-real-dataset")


def _write_minimal_task(task_dir: Path) -> None:
    """Create a minimal valid task layout under `task_dir`."""
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Solve the puzzle.")
    (task_dir / "task.toml").write_text(
        "\n".join(
            [
                'docker_image_id = "example/task:1"',
                "max_agent_timeout_sec = 60.0",
                "max_test_timeout_sec = 30.0",
                "[difficulty]",
                'level = "easy"',
                "",
            ]
        )
    )


def test_load_tasks_from_directory_skips_non_directories(
    benchmark: TerminalBenchBenchmark, tmp_path: Path
) -> None:
    """Non-directory entries (like dataset.toml/README.md in tb2.1) must be skipped."""
    _write_minimal_task(tmp_path / "task-one")
    _write_minimal_task(tmp_path / "task-two")
    (tmp_path / "dataset.toml").write_text("name = 'demo'\n")
    (tmp_path / "README.md").write_text("# demo\n")

    tasks = benchmark._load_tasks_from_directory(tmp_path)

    assert set(tasks.keys()) == {"task-one", "task-two"}
    assert tasks["task-one"]["problem_statement"] == "Solve the puzzle."
    assert tasks["task-one"]["task_definition"]["max_agent_timeout_sec"] == 60.0


def test_load_tasks_from_directory_skips_hidden(
    benchmark: TerminalBenchBenchmark, tmp_path: Path
) -> None:
    """Hidden directories (e.g. `.git`) must be skipped."""
    _write_minimal_task(tmp_path / "real-task")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")

    tasks = benchmark._load_tasks_from_directory(tmp_path)

    assert set(tasks.keys()) == {"real-task"}


def test_load_datasets_shares_parsed_tasks_across_aliases(
    benchmark: TerminalBenchBenchmark, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Datasets that point at the same path should share the parsed task dict by reference."""
    shared = tmp_path / "shared"
    other = tmp_path / "other"
    _write_minimal_task(shared / "alpha-task")
    _write_minimal_task(other / "beta-task")

    monkeypatch.setattr(
        benchmark,
        "_DATASET_LOCATIONS",
        {
            "default": shared,
            "alias-for-shared": shared,
            "other": other,
        },
    )

    datasets = asyncio.run(benchmark.load_datasets())

    assert set(datasets.keys()) == {"default", "alias-for-shared", "other"}
    assert datasets["default"] is datasets["alias-for-shared"]
    assert datasets["default"] is not datasets["other"]
    assert set(datasets["default"].keys()) == {"alpha-task"}
    assert set(datasets["other"].keys()) == {"beta-task"}


def test_load_datasets_raises_when_path_missing(
    benchmark: TerminalBenchBenchmark, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        benchmark,
        "_DATASET_LOCATIONS",
        {"default": tmp_path / "does-not-exist"},
    )

    with pytest.raises(FileNotFoundError, match="not found"):
        asyncio.run(benchmark.load_datasets())
