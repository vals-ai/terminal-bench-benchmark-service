"""A quiet grader must not look like a dead stream to the client."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest

from terminal_bench_benchmark_service.utils import lines_or_keepalive


async def _collect(lines: AsyncGenerator[str, None], interval: float) -> list[str | None]:
    return [line async for line in lines_or_keepalive(lines, interval)]


def test_lines_pass_through_in_order_with_no_keepalive_while_they_flow() -> None:
    async def lines() -> AsyncGenerator[str, None]:
        for line in ("a", "b", "c"):
            yield line

    assert asyncio.run(_collect(lines(), 5)) == ["a", "b", "c"]


def test_a_silent_stretch_yields_keepalives_and_loses_no_line() -> None:
    async def lines() -> AsyncGenerator[str, None]:
        yield "a"
        await asyncio.sleep(0.2)
        yield "b"

    seen = asyncio.run(_collect(lines(), 0.02))

    assert [line for line in seen if line is not None] == ["a", "b"]
    assert None in seen[1:-1]


def test_a_timeout_held_by_the_stream_still_fires_between_lines() -> None:
    """The verifier timeout wraps the whole stream, so it has to keep counting while nothing arrives."""

    async def lines() -> AsyncGenerator[str, None]:
        try:
            async with asyncio.timeout(0.1):
                for _ in range(5000):
                    yield "a"
                await asyncio.sleep(5)
        except asyncio.TimeoutError:
            raise TimeoutError("grader timed out")

    async def read_slowly() -> None:
        async for _ in lines_or_keepalive(lines(), 0.02):
            await asyncio.sleep(0.0001)

    with pytest.raises(TimeoutError, match="grader timed out"):
        asyncio.run(read_slowly())


def test_closing_the_consumer_stops_the_stream() -> None:
    stopped = asyncio.Event()

    async def lines() -> AsyncGenerator[str, None]:
        try:
            yield "a"
            await asyncio.sleep(5)
        finally:
            stopped.set()

    async def run() -> None:
        consumer = lines_or_keepalive(lines(), 5)
        assert await anext(consumer) == "a"
        await consumer.aclose()
        assert stopped.is_set()

    asyncio.run(run())
