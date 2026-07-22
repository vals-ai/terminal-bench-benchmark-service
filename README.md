# terminal-bench Service

## Getting started

Edit `src/terminal_bench_benchmark_service/benchmark_service.py` and implement the six abstract methods on `ExampleBenchmark`:

| Method | What to do |
|--------|------------|
| `load_datasets()` | Return `dict[dataset_name, dict[task_id, ...]]` of all datasets |
| `retrieve_task()` | Return docker image, problem statement, and resource requirements |
| `setup_task()` | Prepare the sandbox before evaluation (upload files, run install commands) |
| `evaluate_response()` | Score a plain-text response without a sandbox |
| `evaluate_instance()` | Run tests / evaluation scripts in the sandbox |
| `calculate_final_score()` | Aggregate per-task results into a single score |

`setup_task` and `evaluate_instance` are async generators — yield `StreamMessageChunk`, `StreamErrorChunk`, and finally `StreamResultChunk`.

## Datasets

The service loads Terminal-Bench datasets from git submodules under `datasets/`.

| Dataset name | Source path |
|--------------|-------------|
| `default` | `datasets/terminal-bench-2.1/tasks` |
| `terminal-bench-2.1` | `datasets/terminal-bench-2.1/tasks` |
| `terminal-bench-2.0` | `datasets/terminal-bench-2` |

Requests that omit `dataset` use `default`, which currently aliases `terminal-bench-2.1`.

## Evaluation retry

Terminal-Bench checkpoints the post-agent Daytona filesystem before verification.
Retries restore that snapshot without rerunning the agent. The service restarts the
runtime state required by `nginx-request-logging`, `pypi-server`, and
`qemu-alpine-ssh`; checkpoints are bound to the task image, tests, verifier code,
dataset, and original run ID. Owned snapshots expire after 30 days through the
best-effort Daytona janitor that runs on retry.

## Development

```bash
make install   # install dependencies
make dev       # run local server
make test      # run tests
make help      # list all commands
make install-submodules # Install the datasets
```
