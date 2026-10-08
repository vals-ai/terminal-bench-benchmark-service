"""Vals-format score metadata for Terminal-Bench final scoring."""

import re
from collections.abc import Mapping
from typing import Any

SCORE_TYPES = {
    "score": {
        "unit": "percent",
        "description": "Mean per-task reward across submitted tasks; an errored task counts as 0.",
    },
}
# The agent calls the model directly and runs its commands in the sandbox, so there is
# no model-proxy tool telemetry to count.
USAGE_COMPONENTS = [{"component": "generation.model"}]

# Status labels, in the order a population lists them.
RESOLVED = "resolved"
UNRESOLVED = "unresolved"
# The verifier ran but produced no reward: it timed out, its environment failed, or it wrote nothing.
EVALUATION_ERROR = "evaluation_error"
# The task never produced an evaluation result, so the service sees only `None`; the run-level
# `task_errors` message, which the post-run hook attaches to the row, says why.
ERROR = "error"
STATUSES = (RESOLVED, UNRESOLVED, EVALUATION_ERROR, ERROR)

# `_grading_fault` writes `exception_info` as `<ExceptionName>: <message>`.
_EXCEPTION_NAME = re.compile(r"^([A-Za-z_]\w*(?:Error|Exception|Timeout)): ")
UNKNOWN_EXCEPTION = "UnknownError"
NO_REWARD_EXCEPTION = "RewardFileNotFoundError"


def build_vals_format_metadata(
    results: Mapping[str, dict[str, Any] | None], scores: Mapping[str, float]
) -> dict[str, Any]:
    """Describe a run the way the vals-format post-run hook reads it.

    `scores` holds each task's reward as `calculate_final_score` computed it, so the rows
    cannot disagree with the published score.
    """
    rows = [_task_row(task_id, results[task_id], score) for task_id, score in scores.items()]
    by_status = {status: count for status in STATUSES if (count := sum(row["status"] == status for row in rows))}
    mean_score = sum(scores.values()) / len(scores)
    return {
        "score_types": SCORE_TYPES,
        "results": {
            "full": {
                "scores": {"score": _score(mean_score * 100)},
                "counts": {"total": len(rows), "by_status": by_status, "extra": {}},
                "selection": None,
                "aggregated_metrics": {"total": {}, "average_per_task": {}},
                "extra": {},
            },
        },
        "primary_population": "full",
        "tasks": rows,
        "usage_components": USAGE_COMPONENTS,
    }


def _score(value: float) -> dict[str, Any]:
    """Uncertainty is left to the export, which measures it across a model's runs."""
    return {"value": value, "stderr": None, "extra": {}}


def _task_row(task_id: str, result: dict[str, Any] | None, score: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task_id": task_id,
        "scores": {"score": _score(score * 100)},
        "evaluations": [],
        "retries": [],
        "extra": {},
    }
    if not result:
        row["status"] = ERROR
        return row

    verifier: dict[str, Any] = result.get("verifier_result") or {}
    if verifier.get("rewards"):
        row["status"] = RESOLVED if score == 1.0 else UNRESOLVED
        return row

    exception_info: str | None = result.get("exception_info")
    if exception_info:
        match = _EXCEPTION_NAME.match(exception_info)
        name = match.group(1) if match else UNKNOWN_EXCEPTION
        message = exception_info
    else:
        name = NO_REWARD_EXCEPTION
        message = "The verifier reported no reward."
    row["status"] = EVALUATION_ERROR
    row["error"] = {"phase": "evaluation", "message": message}
    row["extra"] = {"exception_name": name}
    return row
