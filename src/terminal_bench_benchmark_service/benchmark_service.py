"""Example benchmark service implementation."""

import asyncio
import json
import logging
import tomllib
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

from benchmark_service import BenchmarkService
from benchmark_service.sandbox import ExecResult, ImageSource, Sandbox, SandboxError
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
from pydantic import model_validator

from terminal_bench_benchmark_service.utils import with_retry

logger = logging.getLogger(__name__)


def with_pinned_image_tools(command: str) -> str:
    """Prefer verifier tools pinned by the task image over agent-installed tools."""
    return f"PATH=/bin:$PATH {command}"


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


@dataclass(frozen=True)
class DatasetSpec:
    """Where a dataset's tasks live and how this service has to run them."""

    tasks_root: Path
    # Terminal-Bench 2.x keeps every task directly under tasks_root. Newer
    # datasets group them, e.g. tasks/<domain>/<field>/<slug>.
    nested: bool = False
    # Tasks that ship no environment.docker_image resolve their image here.
    image_manifest: Path | None = None
    # Grade in a separate, network-blocked sandbox rather than the agent's own,
    # for datasets whose tasks declare verifier.environment_mode = "separate".
    grades_in_separate_sandbox: bool = False


class TerminalBenchBenchmark(BenchmarkService):
    """TODO: Replace this example with your benchmark implementation.

    This example shows a simple text-based Q&A benchmark.
    Modify it to load your own dataset and implement your evaluation logic.
    """

    # Map dataset name -> how to load and run it.
    # `default` aliases the latest Terminal-Bench dataset.
    _DATASETS: ClassVar[dict[str, DatasetSpec]] = {
        "default": DatasetSpec(Path("datasets/terminal-bench-2.1/tasks")),
        "terminal-bench-2.0": DatasetSpec(Path("datasets/terminal-bench-2")),
        "terminal-bench-2.1": DatasetSpec(Path("datasets/terminal-bench-2.1/tasks")),
        "terminal-bench-science": DatasetSpec(
            Path("datasets/terminal-bench-science/tasks"),
            nested=True,
            image_manifest=Path("datasets/images/terminal-bench-science.json"),
            grades_in_separate_sandbox=True,
        ),
    }

    def __init__(self) -> None:
        super().__init__()
        # dataset name -> task id -> task directory, filled by load_datasets.
        # Task directories are held explicitly because a nested dataset's id no
        # longer reconstructs its path.
        self._task_paths: dict[str, dict[str, Path]] = {}
        # dataset name -> parsed image manifest, read lazily on first use.
        self._image_manifests: dict[str, dict[str, Any]] = {}

    def _dataset_spec(self, dataset: str | None) -> DatasetSpec:
        """Return how to load and run the given dataset."""
        key = dataset or "default"
        if key not in self._DATASETS:
            raise ValueError(f"Unknown dataset '{key}'. Available datasets: {', '.join(self._DATASETS.keys())}")
        return self._DATASETS[key]

    def _task_dir(self, task_id: str, dataset: str | None = None) -> Path:
        """Return the on-disk directory for one task."""
        key = dataset or "default"
        try:
            return self._task_paths[key][task_id]
        except KeyError as error:
            raise ValueError(f"Unknown task '{task_id}' in dataset '{key}'") from error

    async def _upload_test_files(self, sandbox: Sandbox, tests_path: Path) -> None:
        """Upload test files from local dataset to sandbox /tests directory."""
        files_to_upload: list[tuple[str, bytes]] = []
        for test_file in tests_path.rglob("*"):
            if test_file.is_file():
                relative = test_file.relative_to(tests_path)
                with open(test_file, "rb") as f:
                    files_to_upload.append((f"/tests/{relative}", f.read()))

        if files_to_upload:
            # Ensure all subdirectories exist before uploading
            subdirs: set[str] = {
                str(Path(destination).parent)
                for destination, _ in files_to_upload
                if Path(destination).parent != Path("/tests")
            }
            for subdir in sorted(subdirs):
                await with_retry(sandbox, lambda: sandbox.exec(f"mkdir -p {subdir}"))
            for destination, content in files_to_upload:
                await with_retry(sandbox, lambda: sandbox.upload_file(destination, content))

    async def _copy_test_files(self, sandbox: Sandbox, task_id: str, dataset: str | None = None) -> str:
        """Copy test files from local dataset into sandbox /tests directory.

        Returns the test command to run (test.sh path or default).
        """
        task_path = self._task_dir(task_id, dataset)
        tests_path = task_path / "tests"

        await with_retry(sandbox, lambda: sandbox.exec("rm -rf /tests && mkdir -p /tests && chmod 755 /tests"))

        # Upload test files
        await self._upload_test_files(sandbox, tests_path)

        await with_retry(sandbox, lambda: sandbox.exec("chmod +x /tests/test.sh"))

        return "bash /tests/test.sh"

    async def _retrieve_reward(self, sandbox: Sandbox) -> dict[str, Any] | None:
        try:
            result: ExecResult = await with_retry(
                sandbox, lambda: sandbox.exec("cat /logs/verifier/reward.txt")
            )
            if result.exit_code == 0:
                return {"score": float(result.output.strip())}
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
        self, sandbox: Sandbox, command: str, cwd: str, timeout: float
    ) -> AsyncGenerator[str, None]:
        """Stream command output with timeout."""
        try:
            async with asyncio.timeout(timeout):
                async for line in stream_command(sandbox, command, cwd, ignore_error=True):
                    yield line
        except asyncio.TimeoutError:
            raise TimeoutError(f"Command execution exceeded timeout of {timeout}s")

    async def _stream_command_with_retry(
        self, sandbox: Sandbox, command: str, cwd: str, timeout: float, retries: int = 3
    ) -> AsyncGenerator[str, None]:
        """Stream command output with retry on transient sandbox errors. Timeout is never retried."""
        for attempt in range(retries):
            try:
                async for line in self._stream_command_with_timeout(sandbox, command, cwd, timeout):
                    yield line
                return
            except TimeoutError:
                raise
            except (SandboxError, RuntimeError):
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
        # to avoid duplicated parsing work (e.g. `default` and `terminal-bench-2.1`).
        loaded_by_path: dict[Path, tuple[dict[str, Any], dict[str, Path]]] = {}
        datasets: dict[str, dict[str, Any]] = {}
        self._task_paths = {}

        for name, spec in self._DATASETS.items():
            location = spec.tasks_root
            if not location.exists():
                raise FileNotFoundError(
                    f"Dataset `{name}` not found at `{location}`. Run `make install-submodules` to install datasets."
                )

            if location not in loaded_by_path:
                loaded_by_path[location] = self._load_tasks_from_directory(location, nested=spec.nested)

            tasks, task_paths = loaded_by_path[location]
            datasets[name] = tasks
            self._task_paths[name] = task_paths

        return datasets

    def _task_directories(self, location: Path, *, nested: bool) -> dict[str, Path]:
        """Return task id -> directory for one dataset root.

        A flat dataset takes every immediate subdirectory. A nested one takes
        every directory holding a `task.toml` at any depth, keyed by its own
        name so ids stay the bare task slugs the upstream registry publishes.
        """
        if not nested:
            return {
                path.name: path
                for path in sorted(location.iterdir())
                if path.is_dir() and not path.name.startswith(".")
            }

        directories: dict[str, Path] = {}
        for task_toml in sorted(location.rglob("task.toml")):
            path = task_toml.parent
            if any(part.startswith(".") for part in path.relative_to(location).parts):
                continue
            if path.name in directories:
                raise ValueError(
                    f"Duplicate task id `{path.name}` in `{location}`: "
                    f"{directories[path.name]} and {path}"
                )
            directories[path.name] = path
        return directories

    def _load_tasks_from_directory(
        self, location: Path, *, nested: bool = False
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """Load all tasks from a single dataset directory."""
        dataset: dict[str, Any] = {}
        task_paths = self._task_directories(location, nested=nested)
        for task_id, task_path in task_paths.items():
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

            dataset[task_id] = task

        return dataset, task_paths

    def _image_manifest(self, dataset: str | None) -> dict[str, Any]:
        """Read the dataset's task -> image manifest, cached per dataset."""
        spec = self._dataset_spec(dataset)
        if spec.image_manifest is None:
            return {}
        key = dataset or "default"
        cached = self._image_manifests.get(key)
        if cached is not None:
            return cached
        if not spec.image_manifest.exists():
            raise FileNotFoundError(
                f"Dataset `{key}` needs an image manifest at `{spec.image_manifest}`, "
                "built by scripts/build_dataset_images.py."
            )
        with open(spec.image_manifest, "rb") as f:
            manifest = cast(dict[str, Any], json.loads(f.read()))
        self._image_manifests[key] = manifest
        return manifest

    def _manifest_entry(self, task_id: str, dataset: str | None) -> dict[str, Any]:
        manifest = self._image_manifest(dataset)
        tasks = cast(dict[str, Any], manifest.get("tasks", {}))
        entry = tasks.get(task_id)
        if not isinstance(entry, dict):
            raise ValueError(
                f"No image published for task `{task_id}` in dataset `{dataset or 'default'}`; "
                "rebuild the dataset image manifest."
            )
        return cast(dict[str, Any], entry)

    def _task_image(self, task_id: str, dataset: str | None) -> str:
        """Return the fully qualified agent image for a task."""
        task = self.get_dataset(dataset)[task_id]
        environment: dict[str, Any] = task.get("task_definition", {}).get("environment", {})
        docker_image = environment.get("docker_image")
        if docker_image:
            # Terminal-Bench 2.x publishes unqualified Docker Hub references.
            return f"docker.io/{docker_image}"

        image = self._manifest_entry(task_id, dataset).get("image")
        if not isinstance(image, str) or not image:
            raise ValueError(f"Image manifest entry for `{task_id}` has no `image`")
        return image

    def _task_cwd(self, task_id: str, dataset: str | None) -> str:
        """Working directory the agent and grader start in."""
        # Terminal-Bench 2.x ships one task that works out of /workspace; the id is
        # scoped to those datasets so a same-named task elsewhere cannot inherit it.
        if task_id == "prove-plus-comm" and not self._dataset_spec(dataset).nested:
            return "/workspace"
        return "/app"

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

        # Terminal-Bench 2.x tasks name a published image; datasets that ship only
        # a Dockerfile resolve their pre-built image through the dataset manifest.
        formatted_docker_image = self._task_image(task_id, dataset)

        # Extract agent timeout from task definition
        agent_config: dict[str, Any] = task_def.get("agent", {})
        agent_timeout_value = agent_config.get("timeout_sec")

        if agent_timeout_value is None:
            raise ValueError(f"Agent timeout_sec not found in task definition for `{task_id}`")

        agent_timeout: float = agent_timeout_value

        return RetrieveTaskResponse(
            source=ImageSource(image=formatted_docker_image),
            problem_path="/tmp/problem_statement.md",
            cwd=self._task_cwd(task_id, dataset),
            agent_timeout=agent_timeout,
            resources=resources,
        )

    async def setup_task(
        self, task_id: str, sandbox: Sandbox, dataset: str | None = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Setup task in sandbox by copying problem statement."""
        task = self.get_dataset(dataset)[task_id]
        problem_statement: str = task.get("problem_statement")

        # Prevent interactive prompts (e.g. tzdata timezone selection during apt-get install).
        # Write to both /etc/environment (read by PAM login sessions) and /etc/bash.bashrc
        # (read by non-login interactive shells, which is what the PTY agent uses).
        await with_retry(
            sandbox,
            lambda: sandbox.exec(
                'grep -q "DEBIAN_FRONTEND" /etc/environment || echo "DEBIAN_FRONTEND=noninteractive" >> /etc/environment;'
                ' grep -q "DEBIAN_FRONTEND" /etc/bash.bashrc || echo "export DEBIAN_FRONTEND=noninteractive" >> /etc/bash.bashrc'
            ),
        )

        if problem_statement:
            await with_retry(
                sandbox,
                lambda: sandbox.upload_file("/tmp/problem_statement.md", problem_statement.encode()),
            )
            yield StreamMessageChunk(
                type="message", data=f"Problem statement uploaded to /tmp/problem_statement.md\n{problem_statement}"
            )
        else:
            yield StreamErrorChunk(type="error", data=f"Missing problem statement for task {task_id}")

        yield StreamResultChunk(type="result", data={"status": "ok"})

    async def evaluate_response(self, request: EvaluateResponseRequest, dataset: str | None = None) -> Any:
        """Evaluate a text response."""
        if request.response is None:
            raise ValueError("response is required for terminal-bench evaluate_response")

        task = self.get_dataset(dataset)[request.task_id]

        response = request.response.strip()
        is_correct = response == task["answer"]

        return {
            "task_id": request.task_id,
            "resolved": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "expected": task["answer"],
            "received": response,
        }

    async def evaluate_instance(
        self, task_id: str, sandbox: Sandbox, dataset: str | None = None
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
                lambda: sandbox.exec("mkdir -p /logs/agent /logs/verifier && chmod -R 755 /logs"),
            )

            # Copy test files from dataset into sandbox and get test command
            test_script = await self._copy_test_files(sandbox, task_id, dataset)

            # Caffe processes linger when agent gets OOM killed
            if task_id == "caffe-cifar-10" and not self._dataset_spec(dataset).nested:
                await with_retry(sandbox, lambda: sandbox.exec("pkill -9 caffe || true"))

            # Get verifier timeout from task definition
            verifier_timeout = self._get_verifier_timeout(task_id, dataset)

            # Start running the tests
            yield StreamMessageChunk(type="message", data=f"Running tests for {task_id}...")

            # Task images install their pinned verifier tools in /bin, but an agent may
            # create an older user-local executable with the same name. Keep the image's
            # controlled toolchain ahead of user-local paths during evaluation.
            test_script = with_pinned_image_tools(test_script)

            # Run the test, collect the test output and stream the logs to the client.
            # Use retries=1 (no retry) to avoid re-running test.sh on streaming errors —
            # if the connection drops after the test completes, a retry would re-run the
            # test and could overwrite a passing reward.txt with a failing result.
            try:
                async for line in self._stream_command_with_retry(
                    sandbox, test_script, self._task_cwd(task_id, dataset), verifier_timeout,
                    retries=1,
                ):
                    test_output += line + "\n"
                    yield StreamMessageChunk(type="message", data=line)
            except TimeoutError as e:
                is_success = False
                streaming_timed_out = True
                exception_info = str(e)
            except (SandboxError, RuntimeError) as e:
                # Streaming failed but the test may have already completed and written reward.txt.
                # Mark as failed tentatively; we will try to recover via _retrieve_reward below.
                is_success = False
                exception_info = str(e)

            # Always try to read the reward file unless the test definitively timed out.
            # This recovers scores when streaming drops after test.sh has already written
            # the reward — without this, a transient sandbox network error causes a false
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
