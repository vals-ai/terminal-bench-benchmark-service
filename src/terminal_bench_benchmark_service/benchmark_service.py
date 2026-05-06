"""Example benchmark service implementation."""

import asyncio
import json
import tomllib
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from benchmark_service import BenchmarkService
from benchmark_service.schemas import (
    EvaluateResponseRequest,
    FinalScoreResult,
    Resources,
    RetrieveTaskResponse,
    StreamChunk,
    StreamErrorChunk,
    StreamMessageChunk,
    StreamResultChunk,
)
from benchmark_service.utils import stream_command
from daytona import AsyncSandbox, DaytonaError, FileUpload
from daytona.common.process import ExecuteResponse
from pydantic import model_validator

from terminal_bench_benchmark_service.utils import with_retry


class OverrideResources(Resources):
    @staticmethod
    def _normalize_resource(value: Any, fallback_key: str | None, data: dict[str, Any]) -> int | None:
        """Normalize resource value: handle string with 'G' suffix or _mb variant."""
        if value is not None:
            if isinstance(value, str):
                return int(value.strip("G"))
            # If the actual key is memory_mb/storage_mb, convert from MB to GB
            if fallback_key and fallback_key in data:
                return value // 1024
            return value

        if fallback_key:
            mb_value = data.get(fallback_key)
            if mb_value is not None:
                return mb_value // 1024

        return None

    @model_validator(mode="before")
    @classmethod
    def align_resources(cls, data: Any) -> Any:
        # Handle both input formats and field names
        vcpu = data.get("cpus") or data.get("vcpu")
        memory = data.get("memory") or data.get("memory_mb")
        storage = data.get("storage") or data.get("storage_mb")

        # Normalize memory and storage if needed
        memory = cls._normalize_resource(memory, "memory_mb" if "memory_mb" in data else None, data)
        storage = cls._normalize_resource(storage, "storage_mb" if "storage_mb" in data else None, data)

        # Validate all resources are specified
        if vcpu is None or memory is None or storage is None:
            raise ValueError("All resource values must be specified")

        # Return dict with field names that match the model
        return {"vcpu": vcpu, "memory": memory, "disk": storage}


class TerminalBenchBenchmark(BenchmarkService):
    """TODO: Replace this example with your benchmark implementation.

    This example shows a simple text-based Q&A benchmark.
    Modify it to load your own dataset and implement your evaluation logic.
    """

    # Map dataset name -> directory containing per-task subdirectories.
    # `default` is kept as an alias for terminal-bench-2.0 for backward compatibility.
    _DATASET_LOCATIONS: dict[str, Path] = {
        "default": Path("datasets/terminal-bench-2"),
        "terminal-bench-2.0": Path("datasets/terminal-bench-2"),
        "terminal-bench-2.1": Path("datasets/terminal-bench-2.1/tasks"),
    }

    def _get_dataset_location(self, dataset: str | None) -> Path:
        """Return the on-disk path containing per-task directories for the given dataset."""
        key = dataset or "default"
        if key not in self._DATASET_LOCATIONS:
            raise ValueError(
                f"Unknown dataset '{key}'. Available datasets: {', '.join(self._DATASET_LOCATIONS.keys())}"
            )
        return self._DATASET_LOCATIONS[key]

    async def _upload_test_files(self, sandbox: AsyncSandbox, tests_path: Path) -> None:
        """Upload test files from local dataset to sandbox /tests directory."""
        files_to_upload: list[FileUpload] = []
        for test_file in tests_path.rglob("*"):
            if test_file.is_file():
                relative = test_file.relative_to(tests_path)
                with open(test_file, "rb") as f:
                    files_to_upload.append(
                        FileUpload(
                            source=f.read(),
                            destination=f"/tests/{relative}",
                        )
                    )

        if files_to_upload:
            # Ensure all subdirectories exist before uploading
            subdirs = {
                str(Path(f.destination).parent)
                for f in files_to_upload
                if Path(f.destination).parent != Path("/tests")
            }
            for subdir in sorted(subdirs):
                await with_retry(sandbox, lambda s=subdir: sandbox.process.exec(f"mkdir -p {s}"))
            await with_retry(sandbox, lambda: sandbox.fs.upload_files(files_to_upload))

    async def _copy_test_files(self, sandbox: AsyncSandbox, task_id: str, dataset: str | None = None) -> str:
        """Copy test files from local dataset into sandbox /tests directory.

        Returns the test command to run (test.sh path or default).
        """
        task_path = self._get_dataset_location(dataset) / task_id
        tests_path = task_path / "tests"

        await with_retry(sandbox, lambda: sandbox.process.exec("rm -rf /tests && mkdir -p /tests && chmod 755 /tests"))

        # Upload test files
        await self._upload_test_files(sandbox, tests_path)

        await with_retry(sandbox, lambda: sandbox.process.exec("chmod +x /tests/test.sh"))

        return "bash /tests/test.sh"

    async def _retrieve_reward(self, sandbox: AsyncSandbox) -> dict[str, Any] | None:
        try:
            result: ExecuteResponse = await with_retry(
                sandbox, lambda: sandbox.process.exec("cat /logs/verifier/reward.txt")
            )
            if result.exit_code == 0:
                return {"score": float(result.result.strip())}
        except Exception:
            pass
        return None

    def _get_verifier_timeout(self, task_id: str, dataset: str | None = None) -> float:
        """Extract verifier timeout from task definition."""
        task = self.get_dataset(dataset)[task_id]
        task_def = task.get("task_definition", {})
        verifier_config: dict[str, Any] = task_def.get("verifier", {})
        verifier_timeout_value = verifier_config.get("timeout_sec")

        if verifier_timeout_value is None:
            raise ValueError(f"Verifier timeout_sec not found in task definition for `{task_id}`")

        return float(verifier_timeout_value)

    async def _stream_command_with_timeout(
        self, sandbox: AsyncSandbox, command: str, cwd: str, timeout: float
    ) -> AsyncGenerator[str, None]:
        """Stream command output with timeout."""
        try:
            async with asyncio.timeout(timeout):
                async for line in stream_command(sandbox, command, cwd, ignore_error=True):
                    yield line
        except asyncio.TimeoutError:
            raise TimeoutError(f"Command execution exceeded timeout of {timeout}s")

    async def _stream_command_with_retry(
        self, sandbox: AsyncSandbox, command: str, cwd: str, timeout: float, retries: int = 3
    ) -> AsyncGenerator[str, None]:
        """Stream command output with retry on transient Daytona errors. Timeout is never retried."""
        for attempt in range(retries):
            try:
                async for line in self._stream_command_with_timeout(sandbox, command, cwd, timeout):
                    yield line
                return
            except TimeoutError:
                raise
            except (DaytonaError, RuntimeError):
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2**attempt)
                yield f"Stream interrupted, retrying (attempt {attempt + 2}/{retries})..."

    def _parse_rewards_from_output(self, output: str) -> dict[str, Any] | None:
        """Parse Harbor-style rewards from test output.

        Looks for:
        - reward.json: JSON dict with {key: float|int}
        - reward.txt: Single float value

        Returns:
            Rewards dict or None if not found
        """
        lines = output.split("\n")

        # Look for reward.json output
        for line in lines:
            line = line.strip()
            if line.startswith("{") and "reward" in line.lower():
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        # Look for reward.txt output (single float)
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith("["):
                try:
                    value: float = float(line)
                    if 0.0 <= value <= 1.0:
                        return {"score": value}
                except (ValueError, TypeError):
                    continue

        return None

    def _extract_score(self, result: dict[str, Any] | None) -> float:
        """Extract the score from a single evaluation result. Returns 0.0 for errored/missing tasks."""
        if not result:
            return 0.0

        # Prefer verifier_result (the authoritative score) over exception_info.
        # A task may have exception_info set (e.g. a transient streaming error) but still
        # have a valid verifier_result if the test completed and wrote reward.txt.
        verifier: dict[str, Any] = result.get("verifier_result") or {}
        rewards: dict[str, Any] = verifier.get("rewards") or {}
        score = float(rewards.get("score", 0.0))

        # Only fall back to 0.0 via exception_info when there is no verifier_result at all.
        if not verifier and result.get("exception_info"):
            return 0.0

        return score

    async def load_datasets(self) -> dict[str, dict[str, Any]]:
        """Load the benchmark datasets."""
        # Datasets that share an on-disk path are loaded once and shared by reference
        # to avoid duplicated parsing work (e.g. `default` and `terminal-bench-2.0`).
        loaded_by_path: dict[Path, dict[str, Any]] = {}
        datasets: dict[str, dict[str, Any]] = {}

        for name, location in self._DATASET_LOCATIONS.items():
            if not location.exists():
                raise FileNotFoundError(
                    f"Dataset `{name}` not found at `{location}`. Run `make install-submodules` to install datasets."
                )

            if location not in loaded_by_path:
                loaded_by_path[location] = self._load_tasks_from_directory(location)

            datasets[name] = loaded_by_path[location]

        return datasets

    def _load_tasks_from_directory(self, location: Path) -> dict[str, Any]:
        """Load all tasks from a single dataset directory."""
        dataset: dict[str, Any] = {}
        for task_path in location.iterdir():
            # Skip hidden directories and non-directories (e.g. dataset.toml manifests)
            if task_path.name.startswith(".") or not task_path.is_dir():
                continue

            task: dict[str, Any] = {}

            # Read the problem statement
            problem_path: Path = task_path / "instruction.md"
            task_toml_path: Path = task_path / "task.toml"

            # Parse the problem
            if not problem_path.exists():
                raise FileNotFoundError(f"No instructions found at `{problem_path}`")

            with open(problem_path) as f:
                task["problem_statement"] = f.read()

            # Parse the task toml file
            if not task_toml_path.exists():
                raise FileNotFoundError(f"No task definition found at `{task_toml_path}`")

            with open(task_toml_path, "rb") as f:
                task["task_definition"] = tomllib.load(f)

            dataset[task_path.name] = task

        return dataset

    async def retrieve_task(
        self, task_id: str, skip_validation: bool = False, dataset: str | None = None
    ) -> RetrieveTaskResponse:
        """Retrieve task metadata."""
        if not skip_validation:
            await self.validate_task_ids([task_id], dataset=dataset)

        task = self.get_dataset(dataset)[task_id]

        problem_statement: str = task.get("problem_statement")

        if not problem_statement:
            raise ValueError(f"Missing problem statement for `{task_id}`")

        task_def = task.get("task_definition", {})
        environment: dict[str, Any] = task_def.get("environment", {})

        # Validate resources are correctly set
        resources: OverrideResources = OverrideResources.model_validate(environment)

        # Use docker image from registry
        docker_image = environment.get("docker_image")

        if not docker_image:
            raise ValueError(f"Docker image is required for task `{task_id}`")

        # Use the full url
        formatted_docker_image: str = f"docker.io/{docker_image}"

        # Extract agent timeout from task definition
        agent_config: dict[str, Any] = task_def.get("agent", {})
        agent_timeout_value = agent_config.get("timeout_sec")

        if agent_timeout_value is None:
            raise ValueError(f"Agent timeout_sec not found in task definition for `{task_id}`")

        agent_timeout: float = agent_timeout_value

        return RetrieveTaskResponse(
            docker_image=formatted_docker_image,
            problem_path="/tmp/problem_statement.md",
            cwd="/workspace" if task_id == "prove-plus-comm" else "/app",
            agent_timeout=agent_timeout,
            resources=resources,
        )

    async def setup_task(
        self, task_id: str, sandbox: AsyncSandbox, dataset: str | None = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Setup task in sandbox by copying problem statement."""
        task = self.get_dataset(dataset)[task_id]
        problem_statement: str = task.get("problem_statement")

        # Prevent interactive prompts (e.g. tzdata timezone selection during apt-get install).
        # Write to both /etc/environment (read by PAM login sessions) and /etc/bash.bashrc
        # (read by non-login interactive shells, which is what the PTY agent uses).
        await with_retry(
            sandbox,
            lambda: sandbox.process.exec(
                'grep -q "DEBIAN_FRONTEND" /etc/environment || echo "DEBIAN_FRONTEND=noninteractive" >> /etc/environment;'
                ' grep -q "DEBIAN_FRONTEND" /etc/bash.bashrc || echo "export DEBIAN_FRONTEND=noninteractive" >> /etc/bash.bashrc'
            ),
        )

        if problem_statement:
            await with_retry(
                sandbox,
                lambda: sandbox.fs.upload_files(
                    [FileUpload(source=problem_statement.encode(), destination="/tmp/problem_statement.md")]
                ),
            )
            yield StreamMessageChunk(
                type="message", data=f"Problem statement uploaded to /tmp/problem_statement.md\n{problem_statement}"
            )
        else:
            yield StreamErrorChunk(type="error", data=f"Missing problem statement for task {task_id}")

        yield StreamResultChunk(type="result", data={"status": "ok"})

    async def evaluate_response(self, request: EvaluateResponseRequest, dataset: str | None = None) -> Any:
        """Evaluate a text response."""
        task = self.get_dataset(dataset)[request.task_id]

        # Simple string comparison
        is_correct = request.response.strip() == task["answer"]

        # Return evaluation result as a dict (you can use any structure)
        return {
            "task_id": request.task_id,
            "resolved": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "expected": task["answer"],
            "received": request.response.strip(),
        }

    async def evaluate_instance(
        self, task_id: str, sandbox: AsyncSandbox, dataset: str | None = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Pure evaluation - run test suite in sandbox and return Harbor TrialResult format."""
        exception_info: str | None = None
        verifier_result: dict[str, Any] | None = None
        test_output: str = ""
        is_success = True
        streaming_timed_out = False

        try:
            # Notification that we are starting evaluation
            yield StreamMessageChunk(type="message", data=f"Starting evaluation for task: {task_id}")

            # Ensure log directories exist before running tests
            await with_retry(
                sandbox,
                lambda: sandbox.process.exec("mkdir -p /logs/agent /logs/verifier && chmod -R 755 /logs"),
            )

            # Copy test files from dataset into sandbox and get test command
            test_script = await self._copy_test_files(sandbox, task_id, dataset)

            # Caffe processes linger when agent gets OOM killed
            if task_id == "caffe-cifar-10":
                await with_retry(sandbox, lambda: sandbox.process.exec("pkill -9 caffe || true"))

            # Get verifier timeout from task definition
            verifier_timeout = self._get_verifier_timeout(task_id, dataset)

            # Start running the tests
            yield StreamMessageChunk(type="message", data=f"Running tests for {task_id}...")

            # Run the test, collect the test output and stream the logs to the client.
            # Use retries=1 (no retry) to avoid re-running test.sh on streaming errors —
            # if the connection drops after the test completes, a retry would re-run the
            # test and could overwrite a passing reward.txt with a failing result.
            try:
                async for line in self._stream_command_with_retry(
                    sandbox, test_script, "/workspace" if task_id == "prove-plus-comm" else "/app", verifier_timeout,
                    retries=1,
                ):
                    test_output += line + "\n"
                    yield StreamMessageChunk(type="message", data=line)
            except TimeoutError as e:
                is_success = False
                streaming_timed_out = True
                exception_info = str(e)
            except (DaytonaError, RuntimeError) as e:
                # Streaming failed but the test may have already completed and written reward.txt.
                # Mark as failed tentatively; we will try to recover via _retrieve_reward below.
                is_success = False
                exception_info = str(e)

            # Always try to read the reward file unless the test definitively timed out.
            # This recovers scores when streaming drops after test.sh has already written
            # the reward — without this, a transient Daytona network error causes a false
            # negative even when the agent solved the task correctly.
            if not streaming_timed_out:
                rewards = await self._retrieve_reward(sandbox)

                if rewards:
                    # Test completed and wrote a valid reward — streaming error is irrelevant.
                    is_success = True
                    exception_info = None
                elif is_success:
                    # Streaming succeeded but no reward file — fall back to output parsing.
                    rewards = self._parse_rewards_from_output(test_output)
                else:
                    rewards = None

                reward_msg = f"with rewards: {rewards}" if rewards else "(no rewards found)"
                status_prefix = "✓" if is_success else "✗ (streaming error)"
                yield StreamMessageChunk(type="message", data=f"{status_prefix} Tests finished {reward_msg}")

                # Format final result
                verifier_result = {
                    "rewards": rewards,
                    "output": test_output,
                }

            else:
                yield StreamMessageChunk(
                    type="message",
                    data=f"✗ Tests timed out: {exception_info}",
                )

        except Exception as e:
            is_success = False
            exception_info = str(e)
            yield StreamErrorChunk(type="error", data=f"Evaluation error for {task_id}: {exception_info}")

        # Create trial result
        trial_result: dict[str, Any] = {
            "task_name": task_id,
            "trial_name": f"{task_id}-evaluation",
            "verifier_result": verifier_result if is_success else None,
            "exception_info": exception_info,
        }

        yield StreamResultChunk(type="result", data=(trial_result))

    async def calculate_final_score(
        self, evaluation_results: dict[str, Any], dataset: str | None = None
    ) -> FinalScoreResult:
        """Calculate final score across all evaluations using Harbor-style reward aggregation."""
        if not evaluation_results:
            raise ValueError("There must be at least one evaluation result")

        # Map each task to its score (errored/missing tasks get 0.0)
        task_scores: dict[str, float] = {
            task_id: self._extract_score(result) for task_id, result in evaluation_results.items()
        }

        total_count = len(task_scores)
        resolved = sum(1 for s in task_scores.values() if s == 1.0)
        mean_score = sum(task_scores.values()) / total_count

        metadata = {
            "total_tasks": total_count,
            "resolved_tasks": resolved,
            "unresolved_tasks": total_count - resolved,
        }

        return FinalScoreResult(score=mean_score * 100, metadata=metadata)
