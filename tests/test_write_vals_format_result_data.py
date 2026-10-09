"""The script that writes a finished run's score declaration."""

import io
import json
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from scripts.write_vals_format_result_data import main

RUN_ID = "11111111-2222-3333-4444-555555555555"
RESULT_KEY = f"benchmarks/{RUN_ID}/terminal-bench.json"
DECLARATION_KEY = f"benchmarks/{RUN_ID}/vals_format_result_data.json"


class StubS3:
    """In-memory S3 that records puts."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.puts: list[str] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: Any) -> None:
        self.puts.append(Key)
        self.objects[Key] = Body


def graded(reward: float) -> dict[str, Any]:
    return {
        "task_name": "t",
        "trial_name": "t-evaluation",
        "verifier_result": {"rewards": {"score": reward}, "output": "..."},
        "exception_info": None,
        "agent_caused_exit_reason": "TIMEOUT",
        "attempts": 1,
    }


def run_document(final_score: float = 100 / 3, total: int = 3) -> dict[str, Any]:
    """Two graded tasks and one errored task with no stored result."""
    return {
        "benchmark_id": RUN_ID,
        "status": "FINISHED",
        "final_evaluation": {
            "final_score": final_score,
            "properties": {"total_tasks": total, "resolved_tasks": 1, "unresolved_tasks": 2},
        },
        "evaluation_results": {"a": graded(1.0), "b": graded(0.0)},
        "task_errors": {"c": "Sandbox error: Agent command failed with exit code 139"},
    }


def s3_with(document: dict[str, Any], **objects: bytes) -> StubS3:
    return StubS3(
        {RESULT_KEY: json.dumps(document).encode(), **{f"benchmarks/{RUN_ID}/{k}": v for k, v in objects.items()}}
    )


def run(s3: StubS3, tmp_path: Path, *flags: str, dataset: tuple[str, ...] = ("a", "b", "c")) -> int:
    """Run against a temporary dataset directory that also holds a non-task file."""
    tasks = tmp_path / "dataset"
    tasks.mkdir(exist_ok=True)
    for task_id in dataset:
        (tasks / task_id).mkdir(exist_ok=True)
    (tasks / "README.md").write_text("not a task")
    return main([RUN_ID, "--bucket", "bucket", "--out-dir", str(tmp_path), "--tasks-dir", str(tasks), *flags], s3=s3)


def written(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "vals_format_result_data.json").read_text())


def test_the_declaration_replaces_the_properties_and_keeps_the_rest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = run_document()
    s3 = s3_with(document)

    assert run(s3, tmp_path) == 0

    declaration = written(tmp_path)
    assert declaration["final_evaluation"]["final_score"] == document["final_evaluation"]["final_score"]
    assert declaration["final_evaluation"]["properties"]["primary_population"] == "full"
    assert declaration["final_evaluation"]["properties"]["results"]["full"]["counts"]["total"] == 3
    assert {key: value for key, value in declaration.items() if key != "final_evaluation"} == {
        key: value for key, value in document.items() if key != "final_evaluation"
    }
    assert s3.puts == []
    assert "uploaded=no" in capsys.readouterr().out


def test_a_task_valkyrie_errored_is_filled_in_as_an_error_row(tmp_path: Path) -> None:
    assert run(s3_with(run_document()), tmp_path) == 0

    properties = written(tmp_path)["final_evaluation"]["properties"]
    rows = {row["task_id"]: row for row in properties["tasks"]}
    assert rows["c"]["status"] == "error"
    assert rows["c"]["scores"]["score"]["value"] == 0.0
    assert properties["results"]["full"]["counts"]["by_status"] == {"resolved": 1, "unresolved": 1, "error": 1}


def test_a_task_valkyrie_force_stopped_is_filled_in_from_the_dataset(tmp_path: Path) -> None:
    document = run_document(final_score=25.0, total=4)

    assert run(s3_with(document), tmp_path, dataset=("a", "b", "c", "d")) == 0

    properties = written(tmp_path)["final_evaluation"]["properties"]
    rows = {row["task_id"]: row for row in properties["tasks"]}
    assert set(rows) == {"a", "b", "c", "d"}
    assert rows["d"]["status"] == "error"
    assert properties["results"]["full"]["counts"]["by_status"] == {"resolved": 1, "unresolved": 1, "error": 2}


def test_a_dataset_larger_than_the_stored_total_aborts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(s3_with(run_document(total=4)), tmp_path, dataset=("a", "b", "c", "d", "e")) == 1

    assert "scored 4 tasks, but its document accounts for 5" in capsys.readouterr().err


def test_a_score_the_service_does_not_reproduce_aborts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    s3 = s3_with(run_document(final_score=50.0))

    assert run(s3, tmp_path, "--upload") == 1

    assert "stored 50.0" in capsys.readouterr().err
    assert not (tmp_path / "vals_format_result_data.json").exists()
    assert s3.puts == []


def test_a_population_that_does_not_match_the_stored_total_aborts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(s3_with(run_document(total=4)), tmp_path) == 1

    assert "scored 4 tasks" in capsys.readouterr().err
    assert not (tmp_path / "vals_format_result_data.json").exists()


def test_upload_creates_the_declaration(tmp_path: Path) -> None:
    s3 = s3_with(run_document())

    assert run(s3, tmp_path, "--upload") == 0

    assert s3.puts == [DECLARATION_KEY]
    assert json.loads(s3.objects[DECLARATION_KEY]) == written(tmp_path)


def test_upload_does_not_replace_a_different_declaration_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    s3 = s3_with(run_document(), **{"vals_format_result_data.json": b'{"older": true}'})

    assert run(s3, tmp_path, "--upload") == 1

    assert "--force" in capsys.readouterr().err
    assert s3.puts == []
    assert s3.objects[DECLARATION_KEY] == b'{"older": true}'


def test_force_replaces_a_different_declaration(tmp_path: Path) -> None:
    s3 = s3_with(run_document(), **{"vals_format_result_data.json": b'{"older": true}'})

    assert run(s3, tmp_path, "--upload", "--force") == 0

    assert s3.puts == [DECLARATION_KEY]
    assert json.loads(s3.objects[DECLARATION_KEY]) == written(tmp_path)


def test_an_identical_declaration_is_left_alone(tmp_path: Path) -> None:
    s3 = s3_with(run_document())
    assert run(s3, tmp_path, "--upload") == 0

    assert run(s3, tmp_path, "--upload") == 0

    assert s3.puts == [DECLARATION_KEY]


def test_an_existing_null_declaration_is_not_replaced_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    s3 = s3_with(run_document(), **{"vals_format_result_data.json": b"null"})

    assert run(s3, tmp_path, "--upload") == 1

    assert "--force" in capsys.readouterr().err
    assert s3.puts == []
    assert s3.objects[DECLARATION_KEY] == b"null"
