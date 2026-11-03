"""Register a task's `[[environment.mcp_servers]]` with the Codex CLI.

Codex only reaches MCP servers listed under `[mcp_servers.<name>]` in
`$HOME/.codex/config.toml`. Harbor's Codex adapter writes those entries from
the task config; the native agent has no MCP client and keeps the prompt note.
The agent bundle's installer appends its own tables to the same file and the
launcher passes its settings as `-c` overrides, so the file is merged, never
replaced.
"""

import json
import re
import shlex
import tomllib
from typing import Any

from benchmark_service.sandbox import Sandbox

from terminal_bench_benchmark_service.utils import with_retry

CODEX_AGENT_PREFIX = "codex"
CODEX_CONFIG_PATH = '"$HOME/.codex/config.toml"'
_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(name: str) -> str:
    return name if _BARE_KEY.match(name) else json.dumps(name, ensure_ascii=False)


def _toml_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"  # pyright: ignore[reportUnknownVariableType]
    raise TypeError(f"Unsupported MCP server value: {value!r}")


def codex_mcp_servers_toml(mcp_servers: list[dict[str, Any]]) -> str:
    """Render the `[mcp_servers.<name>]` tables Harbor's Codex adapter writes for a task."""
    text = ""
    for server in mcp_servers:
        name = server["name"]
        entry: dict[str, Any] = (
            {"command": server["command"], "args": list(server.get("args", []))}
            if server["transport"] == "stdio"
            else {"url": server["url"]}
        )
        text += f"[mcp_servers.{_toml_key(name)}]\n"
        text += "".join(f"{key} = {_toml_value(value)}\n" for key, value in entry.items())
        env: dict[str, str] = server.get("env") or {}
        if env:
            text += f"[mcp_servers.{_toml_key(name)}.env]\n"
            text += "".join(f"{_toml_key(key)} = {_toml_value(value)}\n" for key, value in env.items())
    return text


def merge_codex_config(existing: str, mcp_servers: list[dict[str, Any]]) -> str:
    """Append the task's MCP tables to the current config, keeping every existing line."""
    merged = existing
    if merged and not merged.endswith("\n"):
        merged += "\n"
    if merged:
        merged += "\n"
    merged += codex_mcp_servers_toml(mcp_servers)
    try:
        tomllib.loads(merged)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Task MCP servers conflict with the existing Codex config: {exc}") from exc
    return merged


async def bundled_agent_names(sandbox: Sandbox) -> list[str]:
    result = await with_retry(
        sandbox,
        lambda: sandbox.exec('for dir in /bundle/*/; do [ -d "$dir" ] && basename "$dir"; done; true', timeout=30),
    )
    return [line.strip() for line in result.output.splitlines() if line.strip()]


async def configure_codex_mcp_servers(sandbox: Sandbox, mcp_servers: list[dict[str, Any]]) -> bool:
    """Write the task's MCP servers into Codex's config when a Codex bundle runs the task.

    Returns whether the config was written. Other agents are left untouched.
    """
    if not mcp_servers:
        return False
    if not any(name.startswith(CODEX_AGENT_PREFIX) for name in await bundled_agent_names(sandbox)):
        return False
    current = await with_retry(
        sandbox, lambda: sandbox.exec(f"cat {CODEX_CONFIG_PATH} 2>/dev/null || true", timeout=30)
    )
    merged = merge_codex_config(current.output, mcp_servers)
    write = await with_retry(
        sandbox,
        lambda: sandbox.exec(
            f'mkdir -p "$HOME/.codex" && printf %s {shlex.quote(merged)} > {CODEX_CONFIG_PATH}', timeout=30
        ),
    )
    if write.exit_code != 0:
        raise RuntimeError(f"Could not write Codex MCP config: {write.output[-1000:]}")
    return True
