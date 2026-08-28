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

| Dataset name | Source path | Layout |
|--------------|-------------|--------|
| `default` | `datasets/terminal-bench-2.1/tasks` | flat |
| `terminal-bench-2.1` | `datasets/terminal-bench-2.1/tasks` | flat |
| `terminal-bench-2.0` | `datasets/terminal-bench-2` | flat |
| `terminal-bench-science` | `datasets/terminal-bench-science/tasks` | `<domain>/<field>/<slug>` |

Requests that omit `dataset` use `default`, which currently aliases `terminal-bench-2.1`.

Task ids are always the bare task slug, including for nested datasets, so they
match the ids the upstream registry publishes.

## Terminal-Bench Science

This dataset differs from Terminal-Bench 2.x in two ways that the service has to
handle rather than the tasks.

**Images are not published upstream.** Its tasks ship an `environment/Dockerfile`
and a `tests/Dockerfile` instead of naming an image, so both have to be built and
pushed before a run, and the resulting references recorded in a manifest:

```bash
python scripts/build_dataset_images.py --dataset terminal-bench-science --registry <registry>/<namespace>
```

That writes `datasets/images/terminal-bench-science.json`, which `retrieve_task`
reads. Commit it: the service image is built from this repository, so an
uncommitted manifest means every `retrieve_task` for the dataset fails. Sandboxes
pull anonymously, so the registry must allow unauthenticated pulls.

Each task declares a build budget that upstream's runner allows, which is not
what a cold cache on other hardware needs -- two of six measured builds ran past
theirs, one at 4343s against 1000s. Pass `--build-timeout` well above the
largest, and read the `failed:` line before committing the manifest: the script
exits non-zero on any failure but still writes what it built.

**Grading happens in a separate sandbox.** Its tasks declare
`verifier.environment_mode = "separate"` with `network_mode = "no-network"`: the
grader's dependencies live in the task's verifier image and are deliberately not
installed at grade time. The service therefore starts a second sandbox from that
image with egress blocked, re-materializes the task's declared artifacts at their
original paths, runs the grader there, and reads back
`/logs/verifier/reward.txt`. Nesting a container inside the agent's own sandbox
would be cheaper, but the agent is root there for its whole phase and could
tamper with the grader or its result.

This needs the request's sandbox provider, which a create-benchmark-service
release has to expose (`benchmark_service.context.current_sandbox_provider`).
Against an older framework release the isolated path reports a grading fault
instead of producing a score.

Five tasks are multi-container and are not supported by this single-container
runtime; the build script records them under `unsupported_tasks` in the manifest
rather than dropping them silently.

## Development

```bash
make install   # install dependencies
make dev       # run local server
make test      # run tests
make help      # list all commands
make install-submodules # Install the datasets
```
