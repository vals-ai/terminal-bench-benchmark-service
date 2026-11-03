import asyncio

from benchmark_service.context import tool_context_scope
from benchmark_service.schemas import MCPToolContext, NativeToolContext

from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark


def test_retrieve_task_acknowledges_mcp_without_changing_task_metadata() -> None:
    service = asyncio.run(TerminalBenchBenchmark.create())
    native = asyncio.run(service.retrieve_task("atrx-vep-crispr"))
    context = MCPToolContext(mode="mcp", model="openai/test-model")
    with tool_context_scope(context):
        selected = asyncio.run(service.retrieve_task("atrx-vep-crispr"))

    assert native.tool_context == NativeToolContext(mode="native")
    assert selected.tool_context == context
    assert selected.model_dump(exclude={"tool_context"}) == native.model_dump(exclude={"tool_context"})
