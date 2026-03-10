"""Benchmark service entry point."""

from benchmark_service import BenchmarkServiceApp

from terminal_bench_benchmark_service.benchmark_service import TerminalBenchBenchmark

app = BenchmarkServiceApp(TerminalBenchBenchmark)
