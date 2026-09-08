import os

# Define absolute path to output directory
BENCHMARK_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_results")

# Core benchmark constants
WARMUP_ITERATIONS = 10
MIN_MICROBENCHMARK_ITERATIONS = 30
SEEDS = [42, 123, 456, 789, 2026]

# Standard lambdas to test for trade-offs
LAMBDAS = [0, 0.0001, 0.001, 0.01, 0.1, 1.0]

# List of schedulers to run
SCHEDULERS = [
    "Always-Skip",
    "Always-Zstd",
    "Always-Brotli",
    "Always-Gzip",
    "Always-LZ4",
    "Fixed-Batch",
    "Random-Action",
    "Length-Threshold",
    "Heuristic",
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "LinUCB"
]

# Baseline codecs for microbenchmarks
CODECS = [
    "SKIP",
    "ZSTD",
    "Brotli",
    "Gzip",
    "LZ4",
    # Reference-only codecs (not in adaptive action space)
    "zlib",
    "bz2",
    "lzma"
]

def ensure_benchmark_dirs():
    os.makedirs(BENCHMARK_RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(BENCHMARK_RESULTS_DIR, "figures"), exist_ok=True)
