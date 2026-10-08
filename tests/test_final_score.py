"""The final score and the vals-format metadata both post-run hooks read from it."""

import asyncio
import json
from typing import Any

import pytest

from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark

# Fields a vals_format.v1 task row may carry; any other makes the reader drop the run.
TASK_ROW_FIELDS = {
    "task_id",
    "category",
    "tags",
    "status",
    "output",
    "retries",
    "scores",
    "aggregated_metrics",
    "evaluations",
    "error",
    "turns",
    "extra",
}


def reward(value: float) -> dict[str, Any]:
    return {"task_name": "t", "verifier_result": {"rewards": {"score": value}, "output": "..."}, "exception_info": None}


def grading_fault(exception_info: str) -> dict[str, Any]:
    return {"task_name": "t", "verifier_result": None, "exception_info": exception_info}


def score(results: dict[str, Any]) -> Any:
    return asyncio.run(TerminalBenchBenchmark().calculate_final_score(results))


def rows(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["task_id"]: row for row in score(results).metadata["tasks"]}


def test_the_score_is_the_mean_reward_with_errored_tasks_as_zero() -> None:
    result = score({"a": reward(1.0), "b": reward(0.0), "c": None, "d": grading_fault("VerifierTimeoutError: slow")})
    assert result.score == 25.0


def test_the_legacy_view_still_gets_its_counts() -> None:
    """Terminal-Bench 2 still reads these."""
    metadata = score({"a": reward(1.0), "b": reward(0.25), "c": None}).metadata
    assert metadata["total_tasks"] == 3
    assert metadata["resolved_tasks"] == 1
    assert metadata["unresolved_tasks"] == 2


def test_an_empty_run_is_refused() -> None:
    with pytest.raises(ValueError, match="There must be at least one evaluation result"):
        score({})


def test_the_metadata_declares_what_the_hook_needs() -> None:
    metadata = score({"a": reward(1.0)}).metadata
    assert set(metadata) >= {"score_types", "results", "primary_population", "tasks", "usage_components"}
    assert metadata["score_types"]["score"]["unit"] == "percent"
    assert metadata["usage_components"] == [{"component": "generation.model"}]
    assert metadata["primary_population"] == "full"
    assert list(metadata["results"]) == ["full"]
    json.dumps(metadata)


def test_the_full_population_carries_the_published_score_and_consistent_counts() -> None:
    result = score(
        {
            "a": reward(1.0),
            "b": reward(0.0),
            "c": reward(0.25),
            "d": None,
            "e": grading_fault("VerifierEnvironmentError: no reward"),
        }
    )
    full = result.metadata["results"]["full"]

    assert full["scores"]["score"] == {"value": result.score, "stderr": None, "extra": {}}
    assert full["counts"] == {
        "total": 5,
        "by_status": {"resolved": 1, "unresolved": 2, "evaluation_error": 1, "error": 1},
        "extra": {},
    }
    assert full["counts"]["total"] == sum(full["counts"]["by_status"].values())
    assert full["selection"] is None


def test_an_all_pass_run_has_only_resolved_tasks() -> None:
    result = score({"a": reward(1.0), "b": reward(1.0)})
    assert result.score == 100.0
    assert result.metadata["results"]["full"]["counts"]["by_status"] == {"resolved": 2}
    assert [row["status"] for row in result.metadata["tasks"]] == ["resolved", "resolved"]


def test_every_submitted_task_gets_a_row_in_order_with_its_reward() -> None:
    result = score({"a": reward(1.0), "b": None, "c": reward(0.25), "d": reward(0.0)})
    tasks = result.metadata["tasks"]

    assert [task["task_id"] for task in tasks] == ["a", "b", "c", "d"]
    assert [task["status"] for task in tasks] == ["resolved", "error", "unresolved", "unresolved"]
    assert [task["scores"]["score"]["value"] for task in tasks] == [100.0, 0.0, 25.0, 0.0]


def test_task_rows_carry_no_field_the_schema_rejects() -> None:
    for task in score({"a": reward(1.0), "b": None, "c": grading_fault("VerifierTimeoutError: slow")}).metadata[
        "tasks"
    ]:
        assert set(task) <= TASK_ROW_FIELDS


def test_a_task_that_never_produced_a_result_is_an_error_row_left_for_the_hook_to_explain() -> None:
    for missing in (None, {}):
        row = rows({"a": missing})["a"]
        assert row["status"] == "error"
        assert row["scores"]["score"]["value"] == 0.0
        assert "error" not in row


def test_a_verifier_that_failed_is_an_evaluation_error_named_for_its_exception() -> None:
    row = rows(
        {
            "timeout": grading_fault("VerifierTimeoutError: Command execution exceeded timeout of 900.0s"),
            "env": grading_fault("VerifierEnvironmentError: Artifact /app/ packs to more than the transfer limit"),
            "other": grading_fault("RuntimeError: boom"),
            "bare": grading_fault("Command execution exceeded timeout of 900.0s"),
        }
    )

    assert {task_id: r["extra"]["exception_name"] for task_id, r in row.items()} == {
        "timeout": "VerifierTimeoutError",
        "env": "VerifierEnvironmentError",
        "other": "RuntimeError",
        "bare": "UnknownError",
    }
    for r in row.values():
        assert r["status"] == "evaluation_error"
        assert r["scores"]["score"]["value"] == 0.0
        assert r["error"]["phase"] == "evaluation"
    assert row["timeout"]["error"]["message"] == "VerifierTimeoutError: Command execution exceeded timeout of 900.0s"


def test_a_verifier_that_reported_no_reward_is_an_evaluation_error() -> None:
    row = rows(
        {"a": {"task_name": "t", "verifier_result": {"rewards": None, "output": "..."}, "exception_info": None}}
    )["a"]
    assert row["status"] == "evaluation_error"
    assert row["extra"] == {"exception_name": "RewardFileNotFoundError"}


def test_a_reward_outranks_an_exception_the_service_recovered_from() -> None:
    """The score prefers the verifier result, so the row does too."""
    recovered = {**reward(1.0), "exception_info": "RuntimeError: stream dropped"}
    row = rows({"a": recovered})["a"]
    assert row["status"] == "resolved"
    assert row["extra"] == {}
    assert "error" not in row
