# terminal-bench Service

## Getting started

Edit `src/terminal_bench_benchmark_service/benchmark_service.py` and implement the six abstract methods on `ExampleBenchmark`:

| Method | What to do |
|--------|-------------|
| `load_datasets()` | Return `dict[dataset_name, dict[task_id, ...]]` of all datasets |
| `retrieve_task()` | Return docker image, problem statement, and resource requirements |
| `setup_task()` | Prepare the sandbox before evaluation (upload files, run install commands) |
| `evaluate_response()` | Score a plain-text response without a sandbox |
| `evaluate_instance()` | Run tests / evaluation scripts in the sandbox |
| `calculate_final_score()` | Aggregate per-task results into a single score |

`setup_task` and `evaluate_instance` are async generators — yield `StreamMessageChunk`, `StreamErrorChunk`, and finally `StreamResultChunk`.

## Datasets

The service loads Terminal-Bench datasets from git submodules under `datasets/`.

| Dataset name | Source path | Layout |
|--------------|-------------|--------|
| `default` | `datasets/terminal-bench-4/tasks` | flat |
| `terminal-bench-2.1` | `datasets/terminal-bench-2.1/tasks` | flat |
| `terminal-bench-2.0` | `datasets/terminal-bench-2` | flat |
| `terminal-bench-4.0` | `datasets/terminal-bench-4/tasks` | flat |

Requests that omit `dataset` use `default`, which aliases `terminal-bench-4.0`.
The explicit `terminal-bench-2.0` and `terminal-bench-2.1` names remain available
for callers that need the published 2.x datasets.

## Terminal-Bench 4

`terminal-bench-4.0` is pinned to the upstream `v4.0.0` source tag and Harbor's
published prebuilt-image release. The release asset is kept as
`datasets/images/terminal-bench-4-prebuilt.json`; its images are mirrored and the
checked-in service manifest is generated from it with:

```bash
python -m scripts.mirror_tb4_images
python -m scripts.import_tbench4_release \
  --source-manifest datasets/images/terminal-bench-4-prebuilt.json \
  --output datasets/images/terminal-bench-4.json
```

The manifest points task and sidecar images at digest-identical copies in the
`benchmark-services/terminal-bench` ECR repository, in `us-east-1` of the account the service runs in, made by
`scripts/mirror_tb4_images.py`, so the nested daemon never pulls from Docker Hub. The sandbox provider pulls the verifier images and the
`docker:28.3.3-dind` outer image itself, so those stay on Docker Hub.

The service supports all 66 v4 CPU/GPU tasks. Plain tasks use their digest-pinned
environment and verifier images. Compose tasks use the digest-pinned
`docker:28.3.3-dind` outer sandbox and digest-pinned service images from the
release manifest. `verifier.collect` hooks run at their declared service and
phase, and artifact `exclude` patterns are applied while packaging the original
source path for the isolated verifier. The task Dockerfile's working directory
is preserved.

Before pulling, the nested daemon logs in to that registry with a short-lived token from the
service's AWS credentials. They need `ecr:GetAuthorizationToken`, plus `ecr:BatchGetImage` and
`ecr:GetDownloadUrlForLayer` on the repository, so local TB4 runs need such credentials too.
It logs out once the services are up, before the agent runs.

Each packed artifact is written to S3 before the verifier is created, and the
state naming those objects is streamed as `eval_resume_state`. A grading
fault is then retried eval-only: `/ws/evaluate-response` recreates the
verifier from the request's `sandbox_provider` and restores the artifacts from
S3 instead of rerunning the agent. `TERMINAL_BENCH_EVAL_STATE_BUCKET` must be
set for TBench4 grading; `TERMINAL_BENCH_EVAL_STATE_LOCAL_DIR` substitutes a
local directory for tests and `make dev`.

## Development

```bash
make install   # install dependencies
make dev       # run local server
make test      # run tests
make lint      # run lint checks
make typecheck # run type checks
make install-submodules # Install the datasets
```
