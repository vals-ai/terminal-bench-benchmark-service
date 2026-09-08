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
| `default` | `datasets/terminal-bench-2.1/tasks` | flat |
| `terminal-bench-2.1` | `datasets/terminal-bench-2.1/tasks` | flat |
| `terminal-bench-2.0` | `datasets/terminal-bench-2` | flat |
| `terminal-bench-4.0` | `datasets/terminal-bench-4/tasks` | flat |

Requests that omit `dataset` use `default`, which currently aliases
`terminal-bench-2.1`. TBench4 is opt-in to preserve existing Terminal-Bench 2.x
callers while its remaining runtime features are integrated.

## Terminal-Bench 4

`terminal-bench-4.0` is pinned to the upstream `v4.0.0` source tag and Harbor's
published prebuilt-image release. The release asset is kept as
`datasets/images/terminal-bench-4-prebuilt.json`; the checked-in service
manifest is generated from it with:

```bash
python scripts/import_tbench4_release.py \
  --source-manifest datasets/images/terminal-bench-4-prebuilt.json \
  --output datasets/images/terminal-bench-4.json
```

The service supports all 66 v4 CPU/GPU tasks. Plain tasks use their digest-pinned
environment and verifier images. Compose tasks use the digest-pinned
`docker:28.3.3-dind` outer sandbox and digest-pinned service images from the
release manifest. `verifier.collect` hooks run at their declared service and
phase, and artifact `exclude` patterns are applied while packaging the original
source path for the isolated verifier. The task Dockerfile's working directory
is preserved.

## Development

```bash
make install   # install dependencies
make dev       # run local server
make test      # run tests
make lint      # run lint checks
make typecheck # run type checks
make install-submodules # Install the datasets
```
