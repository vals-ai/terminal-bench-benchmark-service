import asyncio
import tomllib
from typing import Any

import pytest
from benchmark_service.sandbox import ExecResult

from terminal_bench_benchmark_service import benchmark_service as service_module
from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark
from terminal_bench_benchmark_service.codex_mcp import (
    codex_http_url,
    codex_mcp_servers_toml,
    configure_codex_mcp_servers,
    merge_codex_config,
)

PLAYWRIGHT = [{"name": "playwright", "transport": "sse", "url": "http://playwright-mcp:3080/sse"}]
PLAYWRIGHT_HTTP_URL = "http://playwright-mcp:3080/mcp"
EXISTING_CONFIG = "[feedback]\nenabled = true\n"


class ConfigSandbox:
    """Sandbox fake: one agent bundle under /bundle and an optional existing Codex config."""

    def __init__(self, agent: str, existing: str = "") -> None:
        self.agent = agent
        self.existing = existing
        self.commands: list[str] = []
        self.labels: dict[str, str] = {}
        self.created_at = None

    async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> ExecResult:
        self.commands.append(command)
        if "for dir in /bundle/" in command:
            return ExecResult(exit_code=0, output=f"{self.agent}\n")
        if "cat " in command and ".codex/config.toml" in command and "mkdir" not in command:
            return ExecResult(exit_code=0, output=self.existing)
        return ExecResult(exit_code=0, output="")

    @property
    def written(self) -> str | None:
        writes = [c for c in self.commands if c.startswith('mkdir -p "$HOME/.codex"')]
        if not writes:
            return None
        assert len(writes) == 1
        return writes[0].split(" printf %s ", 1)[1].rsplit(" > ", 1)[0].strip("'")


def test_sse_server_points_codex_at_streamable_http_endpoint() -> None:
    assert codex_mcp_servers_toml(PLAYWRIGHT) == f'[mcp_servers.playwright]\nurl = "{PLAYWRIGHT_HTTP_URL}"\n'


@pytest.mark.parametrize(
    ("server", "expected"),
    [
        ({"transport": "sse", "url": "http://h:1/sse"}, "http://h:1/mcp"),
        ({"transport": "sse", "url": "http://h:1/sse?token=a"}, "http://h:1/mcp?token=a"),
        ({"transport": "sse", "url": "http://h:1/events"}, "http://h:1/events"),
        ({"transport": "streamable_http", "url": "http://h:1/sse"}, "http://h:1/sse"),
        ({"transport": "http", "url": "http://h:1/mcp"}, "http://h:1/mcp"),
    ],
)
def test_codex_http_url(server: dict[str, Any], expected: str) -> None:
    assert codex_http_url(server) == expected


def test_toml_for_stdio_server_with_env() -> None:
    servers = [{"name": "fs", "transport": "stdio", "command": "npx", "args": ["-y", "srv"], "env": {"TOKEN": "x"}}]
    text = codex_mcp_servers_toml(servers)
    assert text == '[mcp_servers.fs]\ncommand = "npx"\nargs = ["-y", "srv"]\n[mcp_servers.fs.env]\nTOKEN = "x"\n'
    assert tomllib.loads(text)["mcp_servers"]["fs"] == {"command": "npx", "args": ["-y", "srv"], "env": {"TOKEN": "x"}}


def test_toml_keeps_non_ascii_and_quoted_keys_parseable() -> None:
    servers = [{"name": "docs 🚀", "transport": "streamable_http", "url": "http://mcp:1/mcp?q=\"ünï\\🚀\""}]
    assert tomllib.loads(codex_mcp_servers_toml(servers)) == {"mcp_servers": {"docs 🚀": {"url": servers[0]["url"]}}}


def test_merge_keeps_existing_config() -> None:
    merged = merge_codex_config(EXISTING_CONFIG, PLAYWRIGHT)
    assert merged.startswith(EXISTING_CONFIG)
    assert tomllib.loads(merged) == {
        "feedback": {"enabled": True},
        "mcp_servers": {"playwright": {"url": PLAYWRIGHT_HTTP_URL}},
    }


def test_merge_rejects_duplicate_server_table() -> None:
    with pytest.raises(ValueError, match="conflict"):
        merge_codex_config('[mcp_servers.playwright]\nurl = "http://other"\n', PLAYWRIGHT)


def test_codex_bundle_gets_servers_written() -> None:
    sandbox = ConfigSandbox("codex_catalog", EXISTING_CONFIG)
    assert asyncio.run(configure_codex_mcp_servers(sandbox, PLAYWRIGHT)) is True  # pyright: ignore[reportArgumentType]
    assert sandbox.written == merge_codex_config(EXISTING_CONFIG, PLAYWRIGHT)


def test_native_bundle_is_untouched() -> None:
    sandbox = ConfigSandbox("terminalbench_mini_sweagent", EXISTING_CONFIG)
    assert asyncio.run(configure_codex_mcp_servers(sandbox, PLAYWRIGHT)) is False  # pyright: ignore[reportArgumentType]
    assert sandbox.written is None
    assert not any("cat " in c for c in sandbox.commands)


def test_no_mcp_servers_means_no_sandbox_access() -> None:
    sandbox = ConfigSandbox("codex_catalog")
    assert asyncio.run(configure_codex_mcp_servers(sandbox, [])) is False  # pyright: ignore[reportArgumentType]
    assert sandbox.commands == []


@pytest.fixture
def benchmark() -> TerminalBenchBenchmark:
    return asyncio.run(TerminalBenchBenchmark.create())


class SetupSandbox(ConfigSandbox):
    async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> ExecResult:
        result = await super().exec(command, cwd=cwd, timeout=timeout)
        return result if result.output else ExecResult(exit_code=0, output="PRESENT\n")

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        pass


def _setup(benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch, task_id: str, agent: str) -> SetupSandbox:
    sandbox = SetupSandbox(agent, EXISTING_CONFIG)

    async def fake_start(*args: Any) -> None:
        pass

    monkeypatch.setattr(service_module, "start_compose_runtime", fake_start)
    _ = asyncio.run(_drain(benchmark.setup_task(task_id, sandbox, dataset="terminal-bench-4.0")))  # pyright: ignore[reportArgumentType]
    return sandbox


async def _drain(stream: Any) -> None:
    async for _ in stream:
        pass


def test_setup_task_registers_declared_servers_for_codex(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = _setup(benchmark, monkeypatch, "medical-claims-processing", "codex_catalog")
    writes = [c for c in sandbox.commands if 'mkdir -p "$HOME/.codex"' in c]
    assert len(writes) == 1
    assert "docker compose" in writes[0] and "main" in writes[0]
    assert "[mcp_servers.playwright]" in writes[0]
    assert PLAYWRIGHT_HTTP_URL in writes[0]
    assert "playwright-mcp:3080/sse" not in writes[0]
    assert "[feedback]" in writes[0]


def test_setup_task_without_mcp_servers_leaves_codex_config_alone(
    benchmark: TerminalBenchBenchmark, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = _setup(benchmark, monkeypatch, "ctr-optimization", "codex_catalog")
    assert sandbox.written is None
    assert not any("/bundle" in c or ".codex" in c for c in sandbox.commands)
