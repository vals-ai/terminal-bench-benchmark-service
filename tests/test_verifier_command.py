from pathlib import Path

from terminal_bench_benchmark_service.benchmark_service import prepare_test_file, with_pinned_image_tools


def test_pinned_image_tools_precede_agent_installed_tools() -> None:
    assert with_pinned_image_tools("bash /tests/test.sh") == "PATH=/bin:$PATH bash /tests/test.sh"


def test_reshard_c4_verifier_uses_cached_shard_with_unseen_nonce() -> None:
    source = b"""import hashlib\nfrom datasets import load_dataset\nDECOMPRESS_SCRIPT = "/app/decompress.py"\n"""
    source += b"""dataset = load_dataset(\n        "allenai/c4",\n        data_files={"train": ["en/c4-train.00009-of-01024.json.gz"]},\n        split="train",\n    )\n"""
    source += b"""del item["timestamp"]\n                    f.write(json.dumps(item) + "\\n")\n"""

    prepared = prepare_test_file("reshard-c4-data", Path("test_outputs.py"), source).decode()

    assert "c4-train-*.arrow" in prepared
    assert "Dataset.from_file" in prepared
    assert "concatenate_datasets" in prepared
    assert "dataset = load_dataset(" not in prepared
    assert "c4-train.00009-of-01024.json.gz" not in prepared
    assert "EVAL_NONCE = uuid.uuid4().hex" in prepared
    assert 'item["_vals_eval_nonce"] = EVAL_NONCE' in prepared


def test_other_verifiers_are_unchanged() -> None:
    source = b"test source"
    assert prepare_test_file("other-task", Path("test_outputs.py"), source) is source
