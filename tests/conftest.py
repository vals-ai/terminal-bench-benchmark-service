import pytest

from terminal_bench_benchmark_service import compose_runtime


@pytest.fixture(autouse=True)
def ecr_registry(monkeypatch: pytest.MonkeyPatch) -> str:
    """Stands in for the service role's ECR credentials, so no test reaches AWS."""
    registry = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
    monkeypatch.setattr(compose_runtime, "_ecr_credentials", lambda: (registry, "s3cret"))
    return registry
