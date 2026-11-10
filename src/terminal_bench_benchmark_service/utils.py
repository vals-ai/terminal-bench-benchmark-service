import asyncio
import contextlib
from collections.abc import AsyncGenerator
from typing import Any, Callable

from benchmark_service.sandbox import Sandbox, SandboxError, SandboxNotFoundError
from tenacity import retry, retry_if_exception_type, retry_if_not_exception_type, stop_after_attempt, wait_exponential


async def with_retry(sandbox: Sandbox, fn: Callable[..., Any]) -> Any:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=(retry_if_exception_type(SandboxError) & retry_if_not_exception_type(SandboxNotFoundError)),
        reraise=True,
    )
    async def _attempt() -> Any:
        return await fn()

    try:
        return await _attempt()
    except SandboxError as e:
        raise SandboxError(f"{e} | sandbox={sandbox.name} state={sandbox.state}") from e


# The benchmark client drops a stream that carries no message for an hour.
KEEPALIVE_SECONDS = 600.0


async def lines_or_keepalive(lines: AsyncGenerator[str, None], interval: float) -> AsyncGenerator[str | None, None]:
    """Yield each line, and None whenever `interval` passes without one.

    `lines` is drained by a task of its own, so a timeout it holds stays in one task and
    keeps counting while nothing arrives. The queue is unbounded so that task only ever
    waits inside `lines`, where that timeout can reach it.
    """
    queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()

    async def drain() -> None:
        try:
            async for line in lines:
                queue.put_nowait(line)
            queue.put_nowait(None)
        except Exception as error:
            queue.put_nowait(error)

    task = asyncio.create_task(drain())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), interval)
            except TimeoutError:
                yield None
                continue
            if isinstance(item, Exception):
                raise item
            if item is None:
                return
            yield item
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
