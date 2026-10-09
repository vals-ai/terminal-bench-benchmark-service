"""Phase policies for the pinned Terminal-Bench task releases."""

from benchmark_service.schemas import BenchmarkEgressPlan

# Temporary migration coverage for historical agent transports.
MODEL_HOSTS = [
    "*.cursor.sh",
    "*.cursorapi.com",
    "downloads.cursor.com",
    "api.anthropic.com",
    "api.cohere.ai",
    "api.deepseek.com",
    "api.devin.ai",
    "api.meta.ai",
    "api.minimax.io",
    "api.mistral.ai",
    "api.moonshot.ai",
    "api.openai.com",
    "api.x.ai",
    "api.xiaomimimo.com",
    "api.z.ai",
    "app.devin.ai",
    "dashscope-intl.aliyuncs.com",
    "dashscope.aliyuncs.com",
    "dev.model-gateway.vals.ai",
    "generativelanguage.googleapis.com",
    "inference.poolside.ai",
    "integrate.api.nvidia.com",
    "model-gateway.vals.ai",
    "open.bigmodel.cn",
    "server.codeium.com",
    "us-west-1.api.x.ai",
]
APT_HOSTS = ["archive.ubuntu.com", "security.ubuntu.com", "deb.debian.org", "security.debian.org"]
PYTHON_HOSTS = ["pypi.org", "files.pythonhosted.org"]
GITHUB_HOSTS = [
    "github.com",
    "raw.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
    "codeload.github.com",
]
COMPOSE_HOSTS = [
    "auth.docker.io",
    "registry-1.docker.io",
    "production.cloudfront.docker.com",
    "dl-cdn.alpinelinux.org",
]


def task_egress(task_id: str, *, separate_verifier: bool) -> BenchmarkEgressPlan:
    package_hosts = APT_HOSTS + PYTHON_HOSTS + GITHUB_HOSTS + ["astral.sh"]
    run_hosts = MODEL_HOSTS + package_hosts
    if separate_verifier:
        cuda_hosts = ["developer.download.nvidia.com"] if task_id in {"fp8-rmsnorm-gemm", "jax-speedrun-gpu"} else []
        return BenchmarkEgressPlan(
            setup_task=COMPOSE_HOSTS + APT_HOSTS + cuda_hosts,
            run=sorted(set(run_hosts + ["registry.npmjs.org"] + cuda_hosts)),
            # This covers artifact collection in the original sandbox, not the child verifier.
            evaluation=[],
        )
    if task_id in {"pytorch-model-cli", "sam-cell-seg"}:
        package_hosts += ["download.pytorch.org"]
        run_hosts += ["download.pytorch.org"]
    if task_id in {"hf-model-inference", "reshard-c4-data"}:
        hub_hosts = ["huggingface.co", "us.aws.cdn.hf.co", "cas-server.xethub.hf.co"]
        run_hosts += hub_hosts
        package_hosts += hub_hosts
    if task_id == "build-pov-ray":
        run_hosts += ["www.povray.org"]
    if task_id == "build-pmars":
        run_hosts += ["www.koth.org", "koth.org"]
    if task_id == "extract-moves-from-video":
        run_hosts += ["www.youtube.com", "youtube.com", "googlevideo.com", "*.googlevideo.com"]
    return BenchmarkEgressPlan(
        setup_task=[],
        run=sorted(set(run_hosts)),
        evaluation=sorted(set(package_hosts)),
    )
