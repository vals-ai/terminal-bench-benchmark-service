import asyncio
import json

import pytest
from benchmark_service.vals import auth

from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark


def test_hosted_auth_admits_an_allowlisted_descope_key(monkeypatch: pytest.MonkeyPatch) -> None:
    async def exchange(_project_id: str, _access_key: str) -> dict[str, object]:
        return {"tenants": {"vals.ai": {}}}

    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("DESCOPE_PROJECT_ID", "project")
    monkeypatch.delenv("BENCHMARK_CATALOG_API_URL", raising=False)
    monkeypatch.setenv(
        "DESCOPE_TENANT_ALLOWLIST_JSON", json.dumps({"tenants": {"vals.ai": {"datasets": ["terminal-bench-4.0"]}}})
    )
    monkeypatch.setattr(auth, "_exchange_descope_access_key", exchange)
    auth.clear_allowlist_cache()
    auth.clear_auth_cache()

    tenant = asyncio.run(TerminalBenchBenchmark().resolve_tenant({"x-descope-api-key": "key"}))

    assert tenant == "vals.ai"
