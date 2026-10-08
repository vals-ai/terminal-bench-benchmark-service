#!/usr/bin/env python3
"""Write the vals-format score declaration for a finished Terminal-Bench run.

Downloads the run's stored result document, reruns `calculate_final_score` on it, checks the score matches the
stored `final_score`, and writes a copy whose `final_evaluation.properties` is the returned metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import boto3
from botocore.exceptions import ClientError

from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark

RESULT_FILE = "terminal-bench.json"
DECLARATION_FILE = "vals_format_result_data.json"
# The keys of a stored result that calculate_final_score receives.
RESULT_KEYS = ("task_name", "trial_name", "verifier_result", "exception_info")
SCORE_TOLERANCE = 1e-9
DATASET_TASKS = Path(__file__).resolve().parents[1] / "datasets" / "terminal-bench-4" / "tasks"


class DeclarationError(Exception):
    """The declaration cannot be trusted, so nothing is written."""


def dataset_task_ids(tasks_dir: Path) -> list[str]:
    return sorted(path.name for path in tasks_dir.iterdir() if path.is_dir() and not path.name.startswith("."))


def score_inputs(document: dict[str, Any], tasks_dir: Path = DATASET_TASKS) -> dict[str, dict[str, Any] | None]:
    """Rebuild calculate_final_score's input: stored results, and `None` for each errored task.

    A force-stopped task appears in neither `evaluation_results` nor `task_errors`, so a shortfall against
    `total_tasks` is filled from the dataset. The population must then match `total_tasks` exactly.
    """
    results: dict[str, Any] = document["evaluation_results"]
    inputs: dict[str, dict[str, Any] | None] = {
        task_id: {key: value for key, value in result.items() if key in RESULT_KEYS}
        for task_id, result in results.items()
    }
    for task_id in document.get("task_errors") or {}:
        inputs.setdefault(task_id, None)

    total = document["final_evaluation"]["properties"].get("total_tasks")
    if total is not None:
        if len(inputs) < total:
            for task_id in dataset_task_ids(tasks_dir):
                inputs.setdefault(task_id, None)
        if total != len(inputs):
            raise DeclarationError(f"the run scored {total} tasks, but its document accounts for {len(inputs)}")
    return inputs


def declare(document: dict[str, Any], tasks_dir: Path = DATASET_TASKS) -> dict[str, Any]:
    final = asyncio.run(TerminalBenchBenchmark().calculate_final_score(score_inputs(document, tasks_dir)))
    stored = document["final_evaluation"]["final_score"]
    if abs(final.score - stored) >= SCORE_TOLERANCE:
        raise DeclarationError(f"the service scores the run {final.score}, but it stored {stored}")
    return {**document, "final_evaluation": {**document["final_evaluation"], "properties": final.metadata}}


def download(s3: Any, bucket: str, run_id: str, result_file: str) -> dict[str, Any]:
    body = s3.get_object(Bucket=bucket, Key=f"benchmarks/{run_id}/{result_file}")["Body"].read()
    return json.loads(body)


def upload(s3: Any, bucket: str, run_id: str, declaration: dict[str, Any], force: bool) -> str:
    key = f"benchmarks/{run_id}/{DECLARATION_FILE}"
    exists = True
    existing: Any = None
    try:
        existing = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except ClientError as error:
        code = cast(dict[str, Any], error.response).get("Error", {}).get("Code")
        if code not in ("NoSuchKey", "404"):
            raise
        exists = False
    body = json.dumps(declaration).encode()
    if exists and existing == json.loads(body):
        return "unchanged"
    if exists and not force:
        raise DeclarationError(f"s3://{bucket}/{key} already holds a different declaration; --force replaces it")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    return "replaced" if exists else "created"


def main(argv: Sequence[str] | None = None, s3: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _ = parser.add_argument("run_id")
    _ = parser.add_argument("--bucket", required=True, help="bucket holding the run documents")
    _ = parser.add_argument("--result-file", default=RESULT_FILE, help="run-level result file; default %(default)s")
    _ = parser.add_argument(
        "--out-dir", type=Path, default=Path("."), help=f"where {DECLARATION_FILE} is written; default %(default)s"
    )
    _ = parser.add_argument("--upload", action="store_true", help="put the declaration in S3")
    _ = parser.add_argument("--force", action="store_true", help="let --upload replace a different declaration")
    _ = parser.add_argument("--profile", default=None, help="AWS profile; default is the standard credential chain")
    _ = parser.add_argument(
        "--tasks-dir", type=Path, default=DATASET_TASKS, help="dataset tasks directory; default %(default)s"
    )
    args = parser.parse_args(argv)

    if s3 is None:
        s3 = cast(Any, boto3.Session(profile_name=args.profile).client("s3"))  # pyright: ignore[reportUnknownMemberType]
    try:
        document = download(s3, args.bucket, args.run_id, args.result_file)
        declaration = declare(document, args.tasks_dir)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _ = (args.out_dir / DECLARATION_FILE).write_text(json.dumps(declaration))
        uploaded = upload(s3, args.bucket, args.run_id, declaration, args.force) if args.upload else "no"
    except DeclarationError as error:
        print(f"{args.run_id}: {error}", file=sys.stderr)
        return 1

    counts = declaration["final_evaluation"]["properties"]["results"]["full"]["counts"]["by_status"]
    print(f"{args.run_id} score={declaration['final_evaluation']['final_score']} statuses={counts} uploaded={uploaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
