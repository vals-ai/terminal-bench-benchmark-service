"""Example benchmark service implementation."""

import json
import tomllib
from collections import defaultdict
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast

from benchmark_service import BenchmarkService
from benchmark_service.schemas import (
    EvaluateResponseRequest,
    FinalScoreResult,
    Resources,
    RetrieveTaskResponse,
    StreamChunk,
    StreamErrorChunk,
    StreamMessageChunk,
)
from benchmark_service.utils import stream_command
from daytona import AsyncSandbox
from daytona.common.process import ExecuteResponse
from pydantic import model_validator


class OverrideResources(Resources):
    @model_validator(mode="before")
    @classmethod
    def align_resources(cls, data: Any) -> Any:
        vcpu = data.get("cpus")
        memory = data.get("memory")
        storage = data.get("storage")

        # Validate all resources are specified
        resources = [vcpu, memory, storage]
        if any(resource is None for resource in [vcpu, memory, storage]):
            raise ValueError("All resource values must be specified", resources)

        # Parse out the string
        memory = int(memory.strip("G"))
        storage = int(storage.strip("G"))

        return cls(vcpu=vcpu, memory=memory, disk=storage)


class TerminalBenchBenchmark(BenchmarkService):
    """TODO: Replace this example with your benchmark implementation.

    This example shows a simple text-based Q&A benchmark.
    Modify it to load your own dataset and implement your evaluation logic.
    """

    _DATASET_LOCATION: Path = Path("datasets/terminal-bench-2")

    async def _retrieve_reward(self, sandbox: AsyncSandbox) -> dict[str, Any] | None:
        try:
            result: ExecuteResponse = await sandbox.process.exec("cat /logs/verifier/reward.txt")
            if result.exit_code == 0:
                return {"score": float(result.result.strip())}
        except (ValueError, TypeError, Exception):
            pass
        return None

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
                    return {"reward": value}
                except (ValueError, TypeError):
                    continue

        return None

    def _should_skip_result(self, result: dict[str, Any]) -> bool:
        """Check if result should be skipped from aggregation."""
        return not result or result.get("exception_info") or not result.get("verifier_result", {}).get("rewards")

    async def load_datasets(self) -> dict[str, dict[str, Any]]:
        """Load the benchmark datasets."""
        if not self._DATASET_LOCATION.exists():
            raise FileNotFoundError("Please run `make install-submodules` to install the dataset.")

        dataset: dict[str, Any] = {}
        for task_path in self._DATASET_LOCATION.iterdir():
            task: dict[str, Any] = {}

            # Read the problem statement
            problem_path: Path = Path(task_path / "instruction.md")
            task_toml_path: Path = Path(task_path / "task.toml")

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

            dataset[task_path.stem] = task

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

        environment: dict[str, Any] = task.get("environment", {})

        # Validate resources are correctly set
        resources: OverrideResources = OverrideResources.model_validate(environment)

        # Use docker image from registry
        docker_image = environment.get("docker_image")

        if not docker_image:
            raise ValueError(f"Docker image is required for task `{task_id}`")

        # Use the full url
        formatted_docker_image: str = f"docker.io/{docker_image}"

        return RetrieveTaskResponse(
            docker_image=formatted_docker_image,
            problem_statement=problem_statement,
            request_setup=False,
            cwd="/app",
            resources=resources,
        )

    async def setup_task(
        self, task_id: str, sandbox: AsyncSandbox, dataset: str | None = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Setup task in sandbox (not needed for this example)."""
        yield StreamErrorChunk(
            type="error",
            data="There is no setup required for terminal bench 2, all dependencies are coupled with the docker image.",
        )

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

            # Fetch the data for the task
            task_data = self.get_dataset(dataset)[task_id]
            task_def = task_data.get("task_definition", {})

            # Get test script from task definition
            test_script = task_def.get("tests", {}).get("command", "pytest")

            # Start running the tests
            yield StreamMessageChunk(type="message", data=f"Running tests for {task_id}...")

            # Run the test, collect the test output and stream the logs to the client
            try:
                async for line in stream_command(sandbox, test_script, "/app", ignore_error=True):
                    test_output += line + "\n"
                    yield StreamMessageChunk(type="message", data=line)
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
                yield StreamMessageChunk(type="message", data=f"✓ Tests passed {reward_msg}")

                # Format final result
                verifier_result = {
                    "rewards": rewards if rewards else {"passed": 1.0},
                    "test_output": test_output,
                    "passed": True,
                }

            else:
                yield StreamMessageChunk(
                    type="message",
                    data=f"✗ Tests failed: {exception_info}",
                )

        except Exception as e:
            exception_info = str(e)
            yield StreamErrorChunk(type="error", data=f"Evaluation error for {task_id}: {exception_info}")

        # Create verified result if successful
        harbor_verifier_result: dict[str, Any] | None = None
        if is_success and verifier_result:
            harbor_verifier_result = {
                "rewards": verifier_result.get("rewards", 1.0),
                "output": test_output,
            }

        # Create trial result
        trial_result: dict[str, Any] = {
            "task_name": task_id,
            "trial_name": f"{task_id}-evaluation",
            "verifier_result": harbor_verifier_result,
            "exception_info": exception_info,
        }

        yield StreamMessageChunk(type="message", data=json.dumps(trial_result, indent=4))

    async def calculate_final_score(
        self, evaluation_results: dict[str, Any], dataset: str | None = None
    ) -> FinalScoreResult:
        """Calculate final score across all evaluations using Harbor-style reward aggregation."""
        if not evaluation_results:
            raise ValueError("There must be at least one evaluation result")

        resolved: int = 0
        total_count: int = len(evaluation_results)
        reward_stats: dict[str, list[float | int]] = defaultdict(list)

        for result in evaluation_results.values():
            if self._should_skip_result(result):
                continue

            resolved += 1
            verifier = result["verifier_result"]
            rewards = verifier.get("rewards")

            # Parse the rewards covering multiple return formats
            if isinstance(rewards, dict):
                for key, value in cast(dict[str, Any], rewards).items():
                    if isinstance(value, (int, float)):
                        reward_stats[str(key)].append(value)
            else:
                reward_stats["reward"].append(rewards)

        # Calculate aggregated metrics
        success_rate = resolved / total_count * 100

        # Compute mean for each reward key
        aggregated_rewards: dict[str, float] = {}
        for key, values in reward_stats.items():
            if values:
                aggregated_rewards[key] = sum(values) / len(values)

        # Average of all aggregated rewards
        overall_score = sum(aggregated_rewards.values()) / len(aggregated_rewards) if aggregated_rewards else 0.0

        # Metadata about the run
        metadata = {
            "total_tasks": total_count,
            "resolved_tasks": resolved,
            "unresolved_tasks": total_count - resolved,
            "reward_stats": aggregated_rewards,
            "success_rate": success_rate,
        }

        return FinalScoreResult(score=overall_score, metadata=metadata)
