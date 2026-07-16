from terminal_bench_benchmark_service.benchmark_service import with_pinned_image_tools


def test_pinned_image_tools_precede_agent_installed_tools() -> None:
    assert with_pinned_image_tools("bash /tests/test.sh") == "PATH=/bin:$PATH bash /tests/test.sh"
