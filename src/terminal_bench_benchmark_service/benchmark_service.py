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
from daytona import AsyncSandbox, FileUpload
from daytona.common.process import ExecuteResponse
from pydantic import model_validator


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

    _DATASET_LOCATION: Path = Path("datasets/terminal-bench-2")

    async def _upload_test_files(self, sandbox: AsyncSandbox, tests_path: Path) -> None:
        """Upload test files from local dataset to sandbox /tests directory."""
        files_to_upload: list[FileUpload] = []
        for test_file in tests_path.iterdir():
            if test_file.is_file():
                with open(test_file, "rb") as f:
                    files_to_upload.append(
                        FileUpload(
                            source=f.read(),
                            destination=f"/tests/{test_file.name}",
                        )
                    )

        if files_to_upload:
            await sandbox.fs.upload_files(files_to_upload)

    async def _copy_test_files(self, sandbox: AsyncSandbox, task_id: str) -> str:
        """Copy test files from local dataset into sandbox /tests directory.

        Returns the test command to run (test.sh path or default).
        """
        task_path = self._DATASET_LOCATION / task_id
        tests_path = task_path / "tests"

        # Create /tests directory
        await sandbox.fs.create_folder("/tests", "755")

        # Upload test files
        await self._upload_test_files(sandbox, tests_path)

        # Use test.sh if it exists, otherwise default to pytest

        await sandbox.process.exec("chmod +x /tests/test.sh")

        return "bash /tests/test.sh"

    async def _retrieve_reward(self, sandbox: AsyncSandbox) -> dict[str, Any] | None:
        try:
            result: ExecuteResponse = await sandbox.process.exec("cat /logs/verifier/reward.txt")
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
        start_time = asyncio.get_event_loop().time()
        async for line in stream_command(sandbox, command, cwd, ignore_error=True):
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Command execution exceeded timeout of {timeout}s")
            yield line

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
        if not result or result.get("exception_info"):
            return 0.0

        verifier: dict[str, Any] = result.get("verifier_result") or {}
        rewards: dict[str, Any] = verifier.get("rewards") or {}
        return float(rewards.get("score", 0.0))

    async def load_datasets(self) -> dict[str, dict[str, Any]]:
        """Load the benchmark datasets."""
        if not self._DATASET_LOCATION.exists():
            raise FileNotFoundError("Please run `make install-submodules` to install the dataset.")

        dataset: dict[str, Any] = {}
        for task_path in self._DATASET_LOCATION.iterdir():
            # Skip hidden directories and non-directories
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

        return {"default": dataset}

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

        # Prevent interactive prompts
        await sandbox.process.exec('echo "DEBIAN_FRONTEND=noninteractive" >> /etc/environment')

        if problem_statement:
            await sandbox.fs.upload_files(
                [FileUpload(source=problem_statement.encode(), destination="/tmp/problem_statement.md")]
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

        try:
            # Notification that we are starting evaluation
            yield StreamMessageChunk(type="message", data=f"Starting evaluation for task: {task_id}")

            # Copy test files from dataset into sandbox and get test command
            test_script = await self._copy_test_files(sandbox, task_id)

            # Get verifier timeout from task definition
            verifier_timeout = self._get_verifier_timeout(task_id, dataset)

            # Start running the tests
            yield StreamMessageChunk(type="message", data=f"Running tests for {task_id}...")

            # Run the test, collect the test output and stream the logs to the client
            try:
                async for line in self._stream_command_with_timeout(
                    sandbox, test_script, "/workspace" if task_id == "prove-plus-comm" else "/app", verifier_timeout
                ):
                    test_output += line + "\n"
                    yield StreamMessageChunk(type="message", data=line)
            except TimeoutError as e:
                is_success = False
                exception_info = str(e)
            except RuntimeError as e:
                is_success = False
                exception_info = str(e)

            # Create Harbor TrialResult format
            if is_success:
                # Read rewards from file written by test script
                rewards = await self._retrieve_reward(sandbox)

                # Parse from the test output if not found
                if not rewards:
                    rewards = self._parse_rewards_from_output(test_output)

                reward_msg = f"with rewards: {rewards}" if rewards else "(no rewards found)"
                yield StreamMessageChunk(type="message", data=f"✓ Tests finished {reward_msg}")

                # Format final result
                verifier_result = {
                    "rewards": rewards,
                    "output": test_output,
                }

            else:
                yield StreamMessageChunk(
                    type="message",
                    data=f"✗ Tests failed: {exception_info}",
                )

        except Exception as e:
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
