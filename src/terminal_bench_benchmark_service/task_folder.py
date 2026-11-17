"""The task folder `setup_task` writes into the sandbox.

The folder uses Harbor's file names. An agent package reads `/task/instruction.md`
and `/task/task.toml` and needs no benchmark code. The reader rules are in the
Agent Standardization design.

`task.toml` is built here. It is not the upstream file. It holds only what the
agent can learn from the problem statement today: the task id, the agent time
limit, and the MCP servers. `[verifier]`, `[solution]`, the upstream `[metadata]`
and the resource keys stay out.
"""

import json
import logging
import tomllib
from collections.abc import Mapping
from typing import Any

from benchmark_service.sandbox import Sandbox

from terminal_bench_benchmark_service.utils import with_retry

logger = logging.getLogger(__name__)

# The one place that names the folder path. The path is an open design decision.
TASK_DIR = "/task"

SCHEMA_VERSION = "1.3"
BENCHMARK = "terminal-bench"

# The `[[environment.mcp_servers]]` fields of Harbor's `MCPServerConfig`, in file order.
MCP_SERVER_FIELDS = ("name", "transport", "url", "command", "args", "headers")


def _string(value: str) -> str:
    # A JSON string is a TOML basic string, except for DEL and non-BMP escapes.
    # `ensure_ascii=False` avoids the escapes, and `task_toml` checks the result.
    return json.dumps(value, ensure_ascii=False)


def _value(value: Any) -> str:
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, list):
        return "[" + ", ".join(_value(item) for item in value) + "]"  # pyright: ignore[reportUnknownVariableType]
    raise TypeError(f"Unsupported MCP server value: {value!r}")


def _server_entries(server: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Return the fields to copy and their TOML text for one MCP server."""
    fields: dict[str, Any] = {key: server[key] for key in MCP_SERVER_FIELDS if key in server}
    headers: dict[str, str] = fields.pop("headers", None) or {}
    text = "[[environment.mcp_servers]]\n"
    text += "".join(f"{key} = {_value(value)}\n" for key, value in fields.items())
    if headers:
        text += "[environment.mcp_servers.headers]\n"
        text += "".join(f"{_string(key)} = {_string(value)}\n" for key, value in headers.items())
        fields["headers"] = dict(headers)
    return fields, text


def task_toml(task_id: str, timeout_sec: float, mcp_servers: list[dict[str, Any]]) -> bytes:
    """Build the `task.toml` text. Raise `ValueError` if it does not read back as written."""
    servers: list[dict[str, Any]] = []
    text = (
        f'schema_version = "{SCHEMA_VERSION}"\n'
        "\n"
        "[metadata]\n"
        f'benchmark = "{BENCHMARK}"\n'
        f"task_id = {_string(task_id)}\n"
        "\n"
        "[agent]\n"
        "# Informational. The Tracker enforces this limit from retrieve_task's\n"
        "# agent_timeout. Nothing reads this value yet.\n"
        f"timeout_sec = {float(timeout_sec)!r}\n"
    )
    for server in mcp_servers:
        fields, entry = _server_entries(server)
        servers.append(fields)
        text += "\n" + entry

    expected: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {"benchmark": BENCHMARK, "task_id": task_id},
        "agent": {"timeout_sec": float(timeout_sec)},
    }
    if servers:
        expected["environment"] = {"mcp_servers": servers}
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"task.toml for `{task_id}` is not valid TOML: {error}") from error
    if parsed != expected:
        raise ValueError(f"task.toml for `{task_id}` does not read back as written")
    return text.encode()


def build_task_folder(
    *, task_id: str, instruction: bytes, timeout_sec: float, mcp_servers: list[dict[str, Any]]
) -> dict[str, bytes]:
    """The folder's files, keyed by path relative to `TASK_DIR`.

    `instruction.md` holds the upstream bytes, with no MCP note. `task.toml` comes last,
    so a folder that has `task.toml` is a complete folder.
    """
    return {
        "instruction.md": instruction,
        "task.toml": task_toml(task_id, timeout_sec, mcp_servers),
    }


async def _upload(sandbox: Sandbox, path: str, content: bytes) -> None:
    await with_retry(sandbox, lambda: sandbox.upload_file(path, content))


async def write_task_folder(sandbox: Sandbox, files: Mapping[str, bytes]) -> str:
    """Write the folder and return one line for the setup stream.

    If `TASK_DIR` exists already, the sandbox belongs to a task that uses the path,
    so nothing is written. A failure to write is not an error for the run: the
    problem statement file still holds the task, and a folder without `task.toml`
    is not read.
    """
    present = await with_retry(sandbox, lambda: sandbox.exec(f"test -e {TASK_DIR} || test -L {TASK_DIR}", timeout=30))
    if present.exit_code == 0:
        return f"Task folder skipped: {TASK_DIR} already exists in the sandbox."
    try:
        made = await with_retry(sandbox, lambda: sandbox.exec(f"mkdir -p {TASK_DIR}", timeout=30))
        if made.exit_code != 0:
            raise RuntimeError(f"mkdir {TASK_DIR} failed: {made.output[-500:]}")
        for relative, content in files.items():
            await _upload(sandbox, f"{TASK_DIR}/{relative}", content)
    except Exception as error:
        logger.warning("Could not write the task folder %s", TASK_DIR, exc_info=True)
        return f"Task folder not written: {error}"
    return f"Task folder written to {TASK_DIR}: {', '.join(files)}"
