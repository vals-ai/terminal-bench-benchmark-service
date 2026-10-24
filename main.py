"""Benchmark service entry point."""

from benchmark_service.vals import ValsBenchmarkServiceApp

from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark

app = ValsBenchmarkServiceApp(TerminalBenchBenchmark)
