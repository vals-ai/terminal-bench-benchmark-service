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

The service supports v4 CPU/GPU tasks that use one environment image and a
separate verifier image. Both image references are digest-pinned, and the task
Dockerfile's working directory is preserved. The current v4 release has 66
tasks; 52 are supported by this runtime. Fourteen are explicitly refused at
retrieval time because they require sidecar services, `verifier.collect` hooks,
or artifact fields this service does not execute yet. They remain in the
manifest so support can be added without silently changing the imported set.

## Development

```bash
make install   # install dependencies
make dev       # run local server
make test      # run tests
make lint      # run lint checks
make typecheck # run type checks
make install-submodules # Install the datasets
```
