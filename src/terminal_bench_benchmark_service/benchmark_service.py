"""Example benchmark service implementation."""

import asyncio
import json
import logging
import shlex
import tomllib
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

from benchmark_service import BenchmarkService, ComposeSandbox, ComposeSource
from benchmark_service.context import current_sandbox_provider
from benchmark_service.sandbox import (
    ExecResult,
    ImageSource,
    Sandbox,
    SandboxCreateRequest,
    SandboxError,
    SandboxProvider,
)
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

from terminal_bench_benchmark_service import isolated_verifier
from terminal_bench_benchmark_service.compose_runtime import (
    compose_runtime_source,
    compose_service_sandbox,
    start_compose_runtime,
    stop_compose_main,
    stop_compose_runtime,
)
from terminal_bench_benchmark_service.utils import with_retry

logger = logging.getLogger(__name__)

# Deletes retry internally for minutes; past this the sweeper can have it.
VERIFIER_DELETE_TIMEOUT_SECONDS = 120

# Process-wide: the bound is the container's memory, not one instance's.
_ARTIFACT_TRANSFERS = asyncio.Semaphore(isolated_verifier.MAX_CONCURRENT_TRANSFERS)
# Verifier sandboxes are created by this service, so the runner's own
# creation cap does not see them.
_VERIFIER_CREATES = asyncio.Semaphore(isolated_verifier.MAX_CONCURRENT_VERIFIER_CREATES)


def _request_sandbox_provider() -> SandboxProvider | None:
    """Return the provider serving the current request, when one is bound."""
    return current_sandbox_provider()


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
        vcpu: Any = data.get("cpus") or data.get("vcpu")
        memory: Any = data.get("memory") or data.get("memory_mb")
        storage: Any = data.get("storage") or data.get("storage_mb")
        gpu: Any = data.get("gpu")
        if gpu is None:
            gpu = data.get("gpus", 0)
        gpu_type: str | None = data.get("gpu_type") if isinstance(data.get("gpu_type"), str) else None
        if gpu_type is None:
            gpu_types: Any = data.get("gpu_types")
            if isinstance(gpu_types, list) and gpu_types:
                first_gpu_type = cast(object, gpu_types[0])
                if isinstance(first_gpu_type, str):
                    gpu_type = first_gpu_type

        # Normalize memory and storage if needed
        memory = cls._normalize_resource(memory, "memory_mb" if "memory_mb" in data else None, data)
        storage = cls._normalize_resource(storage, "storage_mb" if "storage_mb" in data else None, data)

        # Validate all resources are specified
        if vcpu is None or memory is None or storage is None:
            raise ValueError("All resource values must be specified")

        # Return dict with field names that match the model
        return {"vcpu": vcpu, "memory": memory, "disk": storage, "gpu": gpu, "gpu_type": gpu_type}


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
        "default": DatasetSpec(
            Path("datasets/terminal-bench-4/tasks"),
            image_manifest=Path("datasets/images/terminal-bench-4.json"),
            grades_in_separate_sandbox=True,
        ),
        "terminal-bench-2.0": DatasetSpec(Path("datasets/terminal-bench-2")),
        "terminal-bench-2.1": DatasetSpec(Path("datasets/terminal-bench-2.1/tasks")),
        "terminal-bench-4.0": DatasetSpec(
            Path("datasets/terminal-bench-4/tasks"),
            image_manifest=Path("datasets/images/terminal-bench-4.json"),
            grades_in_separate_sandbox=True,
        ),
    }

    # Both are filled by load_datasets, which is the hook `create()` calls; it
    # builds the instance with __new__, so __init__ never runs.
    _task_paths: dict[str, dict[str, Path]]
    _image_manifests: dict[str, dict[str, Any]]

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
            result: ExecResult = await with_retry(sandbox, lambda: sandbox.exec("cat /logs/verifier/reward.txt"))
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
        # to avoid duplicated parsing work (e.g. `default` and `terminal-bench-4.0`).
        loaded_by_path: dict[Path, tuple[dict[str, Any], dict[str, Path]]] = {}
        datasets: dict[str, dict[str, Any]] = {}
        self._task_paths = {}
        self._image_manifests = {}

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
                    f"Duplicate task id `{path.name}` in `{location}`: {directories[path.name]} and {path}"
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
            if task_id in cast(list[str], manifest.get("unsupported_tasks", [])):
                raise ValueError(f"Task `{task_id}` is not supported by this runtime")
            raise ValueError(
                f"No image published for task `{task_id}` in dataset `{dataset or 'default'}`; "
                "rebuild the dataset image manifest."
            )
        return cast(dict[str, Any], entry)

    def _supported_manifest_entry(self, task_id: str, dataset: str | None) -> dict[str, Any]:
        entry = self._manifest_entry(task_id, dataset)
        reason = entry.get("unsupported_reason")
        if isinstance(reason, str) and reason:
            raise ValueError(f"Task `{task_id}` is not supported by this runtime: {reason}")
        return entry

    def _task_image(self, task_id: str, dataset: str | None) -> str:
        """Return the fully qualified agent image for a task."""
        task = self.get_dataset(dataset)[task_id]
        environment: dict[str, Any] = task.get("task_definition", {}).get("environment", {})
        docker_image = environment.get("docker_image")
        if docker_image:
            # Terminal-Bench 2.x publishes unqualified Docker Hub references.
            return f"docker.io/{docker_image}"

        image = self._supported_manifest_entry(task_id, dataset).get("image")
        if not isinstance(image, str) or not image:
            raise ValueError(f"Image manifest entry for `{task_id}` has no `image`")
        return image

    def _verifier_image(self, task_id: str, dataset: str | None) -> str:
        """Return the image the grader runs in, for datasets that isolate it."""
        image = self._supported_manifest_entry(task_id, dataset).get("verifier_image")
        if not isinstance(image, str) or not image:
            raise ValueError(f"Image manifest entry for `{task_id}` has no `verifier_image`")
        return image

    def _compose_sidecar_images(self, task_id: str, dataset: str | None) -> dict[str, str]:
        """Return service names and pinned images for a TBench4 compose task."""
        sidecars_value = self._supported_manifest_entry(task_id, dataset).get("sidecars", [])
        if not isinstance(sidecars_value, list):
            raise ValueError(f"Image manifest entry for `{task_id}` has an invalid sidecar list")
        sidecars = cast(list[object], sidecars_value)

        images: dict[str, str] = {}
        for sidecar_value in sidecars:
            if not isinstance(sidecar_value, dict):
                raise ValueError(f"Image manifest entry for `{task_id}` has an invalid sidecar entry")
            sidecar = cast(dict[str, Any], sidecar_value)
            service = sidecar.get("service")
            image = sidecar.get("image")
            if not isinstance(service, str) or not service or not isinstance(image, str) or not image:
                raise ValueError(f"Image manifest entry for `{task_id}` has an invalid sidecar image")
            if service in images and images[service] != image:
                raise ValueError(f"Image manifest entry for `{task_id}` assigns multiple images to {service!r}")
            images[service] = image
        return images

    def _compose_source(self, task_id: str, dataset: str | None) -> ComposeSource | None:
        if self._dataset_spec(dataset).image_manifest is None:
            return None
        sidecars = self._compose_sidecar_images(task_id, dataset)
        if not sidecars:
            return None
        return compose_runtime_source(task_id, self._task_image(task_id, dataset), sidecars)

    def _task_cwd(self, task_id: str, dataset: str | None) -> str:
        """Working directory the agent starts in.

        For a dataset whose images are built here, this is the image's own
        WORKDIR, recorded in the manifest: most of these tasks put their data and
        starter files somewhere other than /app, and a sandbox started elsewhere
        gets an empty directory the tracker created.
        """
        # Terminal-Bench 2.x ships one task that works out of /workspace; the id is
        # scoped to those datasets so a same-named task elsewhere cannot inherit it.
        if task_id == "prove-plus-comm" and not self._dataset_spec(dataset).nested:
            return "/workspace"
        if self._dataset_spec(dataset).image_manifest is not None:
            workdir = self._manifest_entry(task_id, dataset).get("workdir")
            if isinstance(workdir, str) and workdir:
                return workdir
            # An image that declares no WORKDIR starts in `/`. Handing it /app
            # would put the agent in a directory the image never prepared.
            return "/"
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
        # a Dockerfile resolve their pinned image through the dataset manifest.
        formatted_docker_image = self._task_image(task_id, dataset)
        runtime_source = self._compose_source(task_id, dataset)

        agent_config: dict[str, Any] = task_def.get("agent", {})
        agent_timeout_value = agent_config.get("timeout_sec")

        if agent_timeout_value is None:
            raise ValueError(f"Agent timeout_sec not found in task definition for `{task_id}`")

        agent_timeout: float = agent_timeout_value

        return RetrieveTaskResponse(
            source=runtime_source or ImageSource(image=formatted_docker_image),
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
        outer_sandbox = sandbox
        compose_started = False
        setup_succeeded = False

        try:
            # Valkyrie creates the outer DinD sandbox before calling setup_task.
            # Start the compose services here, then use the main-service wrapper for
            # the same setup operations the agent will receive later.
            runtime_source = self._compose_source(task_id, dataset)
            if runtime_source is not None:
                # Mark this before startup: a partial `pull`/`up` still needs a
                # best-effort `down` if startup or the client stream fails.
                compose_started = True
                environment = task.get("task_definition", {}).get("environment", {})
                await start_compose_runtime(
                    self._task_dir(task_id, dataset),
                    task_id,
                    self._task_image(task_id, dataset),
                    self._compose_sidecar_images(task_id, dataset),
                    environment,
                    sandbox,
                )
                sandbox = ComposeSandbox(sandbox, runtime_source)

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

            # Set this before yielding the terminal setup chunk so a consumer
            # that stops after receiving it does not tear down a healthy runtime.
            setup_succeeded = True
            yield StreamResultChunk(type="result", data={"status": "ok"})
        finally:
            if compose_started and not setup_succeeded:
                try:
                    await stop_compose_runtime(task_id, outer_sandbox)
                except Exception:
                    logger.warning(
                        "Could not clean up compose runtime after setup failure for %s", task_id, exc_info=True
                    )

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
        if self._dataset_spec(dataset).grades_in_separate_sandbox:
            runtime_source = self._compose_source(task_id, dataset)
            outer_sandbox = sandbox if runtime_source is not None else None
            runtime_sandbox = ComposeSandbox(sandbox, runtime_source) if runtime_source is not None else sandbox
            try:
                async for chunk in self._evaluate_in_isolated_verifier(
                    task_id,
                    runtime_sandbox,
                    dataset,
                    outer_sandbox=outer_sandbox,
                    runtime_source=runtime_source,
                ):
                    yield chunk
            finally:
                if runtime_source is not None and outer_sandbox is not None:
                    try:
                        await stop_compose_runtime(task_id, outer_sandbox)
                    except Exception:
                        logger.warning("Could not clean up compose runtime for %s", task_id, exc_info=True)
            return

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
                    sandbox,
                    test_script,
                    self._task_cwd(task_id, dataset),
                    verifier_timeout,
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

    def _verifier_resources(self, task_id: str, dataset: str | None) -> OverrideResources:
        """Resources for the grading environment.

        A task may size its verifier separately from its agent -- a grader that
        replays a hundred instances can need several times the agent's memory --
        so `[verifier.environment]` overlays `[environment]` rather than
        replacing it, and a task that says nothing keeps the agent's sizing.
        """
        task_definition = self.get_dataset(dataset)[task_id].get("task_definition", {})
        environment: dict[str, Any] = dict(task_definition.get("environment", {}))
        environment.update(task_definition.get("verifier", {}).get("environment", {}))
        return OverrideResources.model_validate(environment)

    async def _create_verifier_sandbox(
        self,
        provider: SandboxProvider,
        task_id: str,
        agent_sandbox: Sandbox,
        dataset: str | None,
        verifier_timeout: float,
        attempt: str,
    ) -> Sandbox:
        """Start a sandbox from the task's verifier image with egress blocked."""
        request = SandboxCreateRequest(
            source=ImageSource(image=self._verifier_image(task_id, dataset)),
            name=isolated_verifier.verifier_sandbox_name(task_id, agent_sandbox.id, attempt),
            resources=self._verifier_resources(task_id, dataset),
            network_block_all=True,
            auto_stop_interval=isolated_verifier.auto_stop_minutes(verifier_timeout),
            create_timeout=isolated_verifier.VERIFIER_CREATE_TIMEOUT_SECONDS,
            # The run's labels, so the verifier is attributable to the same
            # benchmark and task; the age-based sweeper collects any stranded.
            labels={**(agent_sandbox.labels or {}), "Role": "verifier"},
            env_vars={},
        )
        try:
            async with _VERIFIER_CREATES:
                return await provider.create_sandbox(request)
        except SandboxError as error:
            raise isolated_verifier.VerifierEnvironmentError(
                f"Could not start the verifier sandbox for `{task_id}`: {error}"
            ) from error

    async def _carry_artifact(
        self,
        artifact: isolated_verifier.ArtifactSpec,
        agent_sandbox: Sandbox,
        verifier: Sandbox,
        *,
        outer_sandbox: Sandbox | None = None,
        runtime_source: ComposeSource | None = None,
    ) -> str | None:
        """Re-materialize one declared artifact in the verifier at its original path.

        Files and directories both travel as a tar archive, staged and checked in
        the verifier before anything is put in place. Every bound is enforced on
        bytes this process holds or on the verifier's own tooling, because the
        agent is root in the sandbox the archive comes from.

        Returns a note when the agent never produced the artifact; the grader
        still runs and decides what a missing submission is worth. Anything else
        raises, because it says nothing about the model.
        """
        source_sandbox = agent_sandbox
        if artifact.service is not None:
            if outer_sandbox is None or runtime_source is None:
                raise isolated_verifier.VerifierEnvironmentError(
                    f"Artifact {artifact.source} names compose service {artifact.service!r}, "
                    "but no compose runtime is active"
                )
            source_sandbox = compose_service_sandbox(outer_sandbox, runtime_source, artifact.service)

        source = artifact.source.rstrip("/") or "/"
        archive = f"/tmp/{isolated_verifier.artifact_archive_name(source)}"

        present = await with_retry(
            source_sandbox, lambda: source_sandbox.exec(isolated_verifier.exists_command(source))
        )
        if present.exit_code != 0:
            raise isolated_verifier.VerifierEnvironmentError(
                f"Could not look for artifact {artifact.source}: {present.output.strip()[-500:]}"
            )
        if present.output.strip().splitlines()[-1].strip() != isolated_verifier.PRESENT:
            return f"Artifact not produced by the agent: {artifact.source}"

        followed = await with_retry(
            source_sandbox, lambda: source_sandbox.exec(isolated_verifier.dir_symlink_command(source))
        )
        if followed.exit_code != 0:
            raise isolated_verifier.VerifierEnvironmentError(
                f"Artifact {artifact.source} contains a symlinked directory, whose contents "
                "packing cannot carry: grading it would mark the model down for output it made"
            )

        packed = await with_retry(
            source_sandbox,
            lambda: source_sandbox.exec(isolated_verifier.pack_command(source, archive, artifact.exclude)),
        )
        if packed.exit_code != 0:
            raise isolated_verifier.VerifierEnvironmentError(
                f"Could not pack artifact {artifact.source}: {packed.output.strip()[-500:]}"
            )
        if isolated_verifier.fabricated_content(packed.output):
            # Zero-padded to its listed length: the grader would parse a
            # right-sized file with a fabricated tail and fail it.
            raise isolated_verifier.VerifierEnvironmentError(
                f"Artifact {artifact.source} changed while being packed: {packed.output.strip()[-500:]}"
            )

        async with _ARTIFACT_TRANSFERS:
            try:
                content = await self._download_bounded(source_sandbox, archive, artifact.source)
            finally:
                await source_sandbox.exec(f"rm -f {shlex.quote(archive)}")
            await with_retry(verifier, lambda: verifier.upload_file(archive, content))

        await self._check_expansion(verifier, archive, artifact.source)

        unpacked = await verifier.exec(isolated_verifier.unpack_command(source, archive))
        if unpacked.exit_code != 0:
            raise isolated_verifier.VerifierEnvironmentError(
                f"Could not unpack artifact {artifact.source} in the verifier: {unpacked.output.strip()[-500:]}"
            )
        return None

    async def _download_bounded(self, sandbox: Sandbox, archive: str, source: str) -> bytes:
        """Read an archive, refusing it as soon as it passes the transfer bound."""
        content = bytearray()
        async for chunk in sandbox.stream_download(archive):
            content.extend(chunk)
            if len(content) > isolated_verifier.MAX_ARTIFACT_BYTES:
                raise isolated_verifier.VerifierEnvironmentError(
                    f"Artifact {source} packs to more than the "
                    f"{isolated_verifier.MAX_ARTIFACT_BYTES} byte transfer limit"
                )
        return bytes(content)

    async def _check_expansion(self, verifier: Sandbox, archive: str, source: str) -> None:
        """Measure the archive with the verifier's own tar before unpacking it."""
        measured = await verifier.exec(isolated_verifier.expanded_size_command(archive))
        if measured.exit_code != 0:
            raise isolated_verifier.VerifierEnvironmentError(
                f"Could not measure artifact {source}: {measured.output.strip()[-500:]}"
            )
        try:
            expanded, members = (int(value) for value in measured.output.strip().splitlines()[-1].split())
        except ValueError as error:
            raise isolated_verifier.VerifierEnvironmentError(
                f"Could not measure artifact {source}: {measured.output.strip()[-500:]}"
            ) from error
        if expanded > isolated_verifier.MAX_EXPANDED_ARTIFACT_BYTES:
            raise isolated_verifier.VerifierEnvironmentError(
                f"Artifact {source} expands to {expanded} bytes, over the "
                f"{isolated_verifier.MAX_EXPANDED_ARTIFACT_BYTES} byte limit"
            )
        if members > isolated_verifier.MAX_ARTIFACT_MEMBERS:
            raise isolated_verifier.VerifierEnvironmentError(
                f"Artifact {source} holds {members} members, over the "
                f"{isolated_verifier.MAX_ARTIFACT_MEMBERS} member limit"
            )

    def _gradeable_artifacts(self, task_id: str, dataset: str | None) -> list[isolated_verifier.ArtifactSpec]:
        """Parse the task's declared artifacts for the isolated verifier."""
        task_def = self.get_dataset(dataset)[task_id].get("task_definition", {})
        try:
            return isolated_verifier.parse_artifacts(task_def.get("artifacts"))
        except isolated_verifier.UnsupportedArtifactError as error:
            raise ValueError(f"Task `{task_id}` declares an artifact this runtime cannot honour: {error}") from error

    async def _run_collect_hooks(
        self,
        task_id: str,
        dataset: str | None,
        agent_sandbox: Sandbox,
        *,
        outer_sandbox: Sandbox | None,
        runtime_source: ComposeSource | None,
        services: set[str] | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Run Harbor verifier collect hooks before carrying artifacts."""
        task_def = self.get_dataset(dataset)[task_id].get("task_definition", {})
        verifier_def = task_def.get("verifier", {})
        hooks = isolated_verifier.parse_collect_hooks(verifier_def.get("collect"))
        for hook in hooks:
            if services is not None and hook.service not in services:
                continue
            target = agent_sandbox
            if hook.service != "main":
                if outer_sandbox is None or runtime_source is None:
                    raise isolated_verifier.VerifierEnvironmentError(
                        f"Collect hook for {task_id} targets compose service {hook.service!r}, "
                        "but no compose runtime is active"
                    )
                target = compose_service_sandbox(outer_sandbox, runtime_source, hook.service)

            yield StreamMessageChunk(
                type="message",
                data=f"Running collect hook for {task_id} on {hook.service}: {hook.command}",
            )
            command = hook.command
            if hook.user is not None:
                command = f"su -s /bin/sh {shlex.quote(hook.user)} -c {shlex.quote(command)}"
            result = await with_retry(target, lambda: target.exec(command, timeout=hook.timeout_sec))
            if result.exit_code != 0:
                raise isolated_verifier.VerifierEnvironmentError(
                    f"Collect hook for {task_id} on {hook.service} failed with exit code "
                    f"{result.exit_code}: {result.output.strip()[-1000:]}"
                )
            if result.output.strip():
                yield StreamMessageChunk(type="message", data=result.output.strip())

    async def _evaluate_in_isolated_verifier(
        self,
        task_id: str,
        sandbox: Sandbox,
        dataset: str | None = None,
        *,
        outer_sandbox: Sandbox | None = None,
        runtime_source: ComposeSource | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Grade in a second sandbox the agent never had access to.

        The agent's environment decides nothing here: the grader runs from the
        task's own verifier image, sees only the declared artifacts, and has no
        network. Failing to build that environment is reported as a grading
        fault rather than a zero, so a broken run is retried instead of being
        published as a model's score.
        """
        exception_info: str | None = None
        verifier_result: dict[str, Any] | None = None
        test_output = ""
        verifier: Sandbox | None = None
        provider = _request_sandbox_provider()
        attempt = uuid.uuid4().hex[:8]

        # One try/finally around everything: a consumer that disconnects while
        # the grader is still streaming closes this generator, and the verifier
        # sandbox has to go with it.
        try:
            try:
                yield StreamMessageChunk(type="message", data=f"Starting isolated evaluation for task: {task_id}")

                if provider is None:
                    raise isolated_verifier.VerifierEnvironmentError(
                        f"Dataset `{dataset}` grades in a separate sandbox, which needs the request's sandbox "
                        "provider. Either create-benchmark-service predates "
                        "benchmark_service.context, or the caller bound the provider around this "
                        "generator's creation rather than around draining it."
                    )

                try:
                    artifacts = self._gradeable_artifacts(task_id, dataset)
                except ValueError as error:
                    raise isolated_verifier.VerifierEnvironmentError(str(error)) from error

                async for chunk in self._run_collect_hooks(
                    task_id,
                    dataset,
                    sandbox,
                    outer_sandbox=outer_sandbox,
                    runtime_source=runtime_source,
                    services={"main"},
                ):
                    yield chunk

                verifier_timeout = self._get_verifier_timeout(task_id, dataset)
                # Everything up to the grader has its own bound: only the grader
                # itself is allowed to take the task's verifier timeout.
                try:
                    async with asyncio.timeout(isolated_verifier.PREPARE_TIMEOUT_SECONDS):
                        verifier = await self._create_verifier_sandbox(
                            provider, task_id, sandbox, dataset, verifier_timeout, attempt
                        )
                except TimeoutError as error:
                    raise isolated_verifier.VerifierEnvironmentError(
                        f"Verifier sandbox for `{task_id}` was not ready within "
                        f"{isolated_verifier.PREPARE_TIMEOUT_SECONDS:g}s"
                    ) from error

                # Artifacts land first; the reward directory is emptied after them,
                # so nothing an archive planted there survives to be read as a score.
                deadline = asyncio.get_running_loop().time() + isolated_verifier.PREPARE_BUDGET_SECONDS
                main_artifacts = [artifact for artifact in artifacts if artifact.service in (None, "main")]
                sidecar_artifacts = [artifact for artifact in artifacts if artifact.service not in (None, "main")]
                task_def = self.get_dataset(dataset)[task_id].get("task_definition", {})
                verifier_def = task_def.get("verifier", {})
                collect_hooks = isolated_verifier.parse_collect_hooks(verifier_def.get("collect"))
                sidecar_hooks = [hook for hook in collect_hooks if hook.service != "main"]
                for artifact in main_artifacts:
                    # Announced before each transfer so the client's idle watchdog
                    # sees progress across a long series of them.
                    yield StreamMessageChunk(type="message", data=f"Carrying artifact {artifact.source}")
                    # Bounded per artifact rather than around the loop: a timeout
                    # spanning a yield would keep running while nobody is driving
                    # this generator. The shared deadline caps their sum.
                    budget = min(
                        isolated_verifier.PREPARE_TIMEOUT_SECONDS,
                        deadline - asyncio.get_running_loop().time(),
                    )
                    try:
                        async with asyncio.timeout(budget):
                            note = await self._carry_artifact(
                                artifact,
                                sandbox,
                                verifier,
                                outer_sandbox=outer_sandbox,
                                runtime_source=runtime_source,
                            )
                    except TimeoutError as error:
                        raise isolated_verifier.VerifierEnvironmentError(
                            f"Artifact {artifact.source} did not transfer within {max(budget, 0):.0f}s"
                        ) from error
                    if note:
                        yield StreamMessageChunk(type="message", data=note)

                if sidecar_artifacts or sidecar_hooks:
                    if outer_sandbox is None or runtime_source is None:
                        raise isolated_verifier.VerifierEnvironmentError(
                            f"Task {task_id} declares sidecar artifacts but no compose runtime is active"
                        )
                    try:
                        await stop_compose_main(task_id, outer_sandbox)
                    except Exception as error:
                        raise isolated_verifier.VerifierEnvironmentError(
                            f"Could not stop main before collecting sidecar state for {task_id}: {error}"
                        ) from error

                    async for chunk in self._run_collect_hooks(
                        task_id,
                        dataset,
                        sandbox,
                        outer_sandbox=outer_sandbox,
                        runtime_source=runtime_source,
                        services={
                            *[artifact.service for artifact in sidecar_artifacts if artifact.service is not None],
                            *[hook.service for hook in sidecar_hooks],
                        },
                    ):
                        yield chunk

                    for artifact in sidecar_artifacts:
                        yield StreamMessageChunk(type="message", data=f"Carrying artifact {artifact.source}")
                        budget = min(
                            isolated_verifier.PREPARE_TIMEOUT_SECONDS,
                            deadline - asyncio.get_running_loop().time(),
                        )
                        try:
                            async with asyncio.timeout(budget):
                                note = await self._carry_artifact(
                                    artifact,
                                    sandbox,
                                    verifier,
                                    outer_sandbox=outer_sandbox,
                                    runtime_source=runtime_source,
                                )
                        except TimeoutError as error:
                            raise isolated_verifier.VerifierEnvironmentError(
                                f"Artifact {artifact.source} did not transfer within {max(budget, 0):.0f}s"
                            ) from error
                        if note:
                            yield StreamMessageChunk(type="message", data=note)

                # /tests is the image's: uploading ours over it would delete data
                # generated at build time and undo its permission hardening.
                await with_retry(verifier, lambda: verifier.exec(isolated_verifier.prepare_logs_command()))
                test_script = isolated_verifier.GRADE_COMMAND

                yield StreamMessageChunk(type="message", data=f"Running isolated tests for {task_id}...")

                # From `/`: most verifier images define no WORKDIR, and every grader
                # addresses its inputs absolutely. No pinned-tools prefix either --
                # the grader runs in its own image.
                async for line in self._stream_command_with_retry(
                    verifier, test_script, "/", verifier_timeout, retries=1
                ):
                    test_output += line + "\n"
                    yield StreamMessageChunk(type="message", data=line)

                reward = await with_retry(verifier, lambda: verifier.exec(isolated_verifier.read_reward_command()))
                if reward.exit_code != 0:
                    # These verifiers write a reward for every graded outcome, a failed
                    # submission included. No reward means grading itself did not finish.
                    raise isolated_verifier.VerifierEnvironmentError(
                        f"Verifier wrote no reward for `{task_id}`: {reward.output.strip()[-2000:]}"
                    )

                try:
                    rewards = {"score": isolated_verifier.parse_reward(reward.output)}
                except (ValueError, IndexError) as error:
                    raise isolated_verifier.VerifierEnvironmentError(
                        f"Verifier wrote an unusable reward for `{task_id}`: {error}"
                    ) from error
                yield StreamMessageChunk(type="message", data=f"✓ Isolated tests finished with rewards: {rewards}")
                verifier_result = {"rewards": rewards, "output": test_output}

            except TimeoutError as e:
                exception_info = f"VerifierTimeoutError: {e}"
                yield StreamMessageChunk(type="message", data=f"✗ Isolated tests timed out: {e}")
            except isolated_verifier.VerifierEnvironmentError as e:
                exception_info = f"VerifierEnvironmentError: {e}"
                yield StreamErrorChunk(type="error", data=exception_info)
            except Exception as e:
                exception_info = f"{type(e).__name__}: {e}"
                yield StreamErrorChunk(type="error", data=f"Evaluation error for {task_id}: {exception_info}")

            # The verdict reaches the consumer before the sandbox is deleted:
            # the generator only resumes into the finally on the next chunk
            # request, so a slow delete cannot hold the result back.
            yield StreamResultChunk(
                type="result",
                data={
                    "task_name": task_id,
                    "trial_name": f"{task_id}-evaluation",
                    "verifier_result": verifier_result,
                    "exception_info": exception_info,
                },
            )
        finally:
            if verifier is not None and provider is not None:
                await self._delete_verifier_sandbox(provider, verifier.id)

    async def _delete_verifier_sandbox(self, provider: SandboxProvider, sandbox_id: str) -> None:
        """Delete the verifier sandbox, surviving cancellation.

        Shielded and re-awaited so a cancelled trial does not abandon the delete
        half-way, and bounded so a provider that retries for minutes cannot hold
        the trial open. Anything still stranded is collected by the sweeper that
        reaps sandboxes by age.
        """
        cleanup = asyncio.create_task(provider.delete_sandbox(sandbox_id))
        cancellation: asyncio.CancelledError | None = None
        while not cleanup.done():
            try:
                await asyncio.wait_for(asyncio.shield(cleanup), VERIFIER_DELETE_TIMEOUT_SECONDS)
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
            except TimeoutError:
                break
            except Exception:
                break

        if cleanup.done():
            try:
                cleanup.result()
            except asyncio.CancelledError:
                if cancellation is None:
                    raise
            except Exception:
                logger.exception("Failed to delete verifier sandbox %s", sandbox_id)
        else:
            logger.warning("Verifier sandbox %s left to the sweeper", sandbox_id)

        if cancellation is not None:
            raise cancellation

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
