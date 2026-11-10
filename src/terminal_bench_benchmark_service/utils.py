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

    A grader can run quietly for longer than the client waits, so the caller needs a
    moment to say it is still alive. `lines` is drained by one task of its own: a timeout
    it holds stays in a single task and keeps counting between lines.
    """
    queue: asyncio.Queue[tuple[str | None, BaseException | None]] = asyncio.Queue(maxsize=1024)

    async def drain() -> None:
        try:
            async for line in lines:
                await queue.put((line, None))
        except Exception as error:
            await queue.put((None, error))
        else:
            await queue.put((None, None))

    task = asyncio.create_task(drain())
    try:
        while True:
            try:
                line, error = await asyncio.wait_for(queue.get(), interval)
            except TimeoutError:
                yield None
                continue
            if error is not None:
                raise error
            if line is None:
                return
            yield line
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
