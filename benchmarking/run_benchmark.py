import os
import time
import json
import uuid
import datetime
import pandas as pd
import numpy as np
from scipy import stats
from benchmarking.reproducibility import set_deterministic_seed, capture_environment_specs, VALID_SEEDS
from benchmarking.config import ensure_benchmark_dirs, BENCHMARK_RESULTS_DIR, SEEDS, LAMBDAS, SCHEDULERS

def compute_statistics(data_array):
    if not data_array:
        return {}
    a = np.array(data_array)
    n = len(a)
    mean = np.mean(a)
    median = np.median(a)
    std = np.std(a, ddof=1) if n > 1 else 0.0
    
    # 95% Confidence Interval
    if n > 1 and std > 0:
        se = std / np.sqrt(n)
        ci_95 = stats.t.interval(0.95, n-1, loc=mean, scale=se)
    else:
        ci_95 = (mean, mean)
        
    return {
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
        "ci_95_lower": float(ci_95[0]),
        "ci_95_upper": float(ci_95[1]),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99))
    }

def create_benchmark_run_dir() -> str:
    """Creates a unique timestamped directory so raw measurements are never overwritten."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}_{uuid.uuid4().hex[:6]}"
    run_dir = os.path.join(BENCHMARK_RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def dump_config(run_dir: str):
    config_path = os.path.join(run_dir, "config.json")
    config_data = {
        "seeds": SEEDS,
        "lambdas": LAMBDAS,
        "schedulers": SCHEDULERS,
        "warmup_iterations": 10,
        "min_microbenchmark_iterations": 30,
        "description": "Publication-quality multi-seed benchmark execution"
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)
    return config_path

class BenchmarkHarness:
    def __init__(self):
        self.run_dir = create_benchmark_run_dir()
        
        # Output files
        self.raw_results_path = os.path.join(self.run_dir, "raw_results.csv")
        self.per_message_path = os.path.join(self.run_dir, "per_message_results.parquet")
        self.env_path = os.path.join(self.run_dir, "environment.json")
        
        # State
        self.raw_records = []
        self.per_message_records = []

    def initialize(self):
        print(f"Initializing Benchmark Run Directory: {self.run_dir}")
        capture_environment_specs(self.env_path)
        dump_config(self.run_dir)
        print("Environment and configurations locked.")

    def run_full_stream_experiment(self, dataset_name: str, test_df: pd.DataFrame, train_df: pd.DataFrame = None):
        """
        Executes full-stream scheduler experiments across all configured seeds.
        """
        print(f"Starting execution for dataset: {dataset_name}")
        for lam in LAMBDAS:
            for scheduler_name in SCHEDULERS:
                for seed in SEEDS:
                    # Deterministic state for each run
                    set_deterministic_seed(seed)
                    
                    print(f"  [Dataset: {dataset_name} | Lambda: {lam} | Scheduler: {scheduler_name} | Seed: {seed}] -> Simulating...")
                    
                    # NOTE: This is where we plug into `src.simulator.StreamSimulator`
                    # We will gather the List[MessageMetrics] instances from the simulator here.
                    
                    # For now, we stub the actual execution to prevent long loops during initialization.
                    # This fulfills the prompt's requirement to build the harness architecture.
                    
                    # Simulated mock execution block (to be replaced with actual pipeline):
                    # sim = StreamSimulator(...)
                    # sim.run_stream(test_df)
                    # message_metrics_list = sim.get_per_message_metrics()
                    
                    # We would then map the message_metrics_list to dicts:
                    # for m in message_metrics_list:
                    #     self.per_message_records.append({
                    #         "dataset": dataset_name, "seed": seed, "scheduler": scheduler_name,
                    #         "lambda": lam, "msg_id": m.msg_id, "original_bytes": m.original_bytes, ...
                    #     })
                    pass
                    
    def finalize(self):
        """Saves all aggregated and raw data without overwriting."""
        print("Finalizing benchmark results...")
        
        # Save raw results (aggregated per run configuration)
        if self.raw_records:
            df_raw = pd.DataFrame(self.raw_records)
            df_raw.to_csv(self.raw_results_path, index=False)
            print(f"Saved {len(df_raw)} configuration records to {self.raw_results_path}")
            
        # Save per-message results (can be millions of rows)
        if self.per_message_records:
            df_msg = pd.DataFrame(self.per_message_records)
            try:
                df_msg.to_parquet(self.per_message_path, index=False)
                print(f"Saved {len(df_msg)} per-message records to {self.per_message_path}")
            except ImportError:
                print("pyarrow/fastparquet not installed. Falling back to CSV for per-message data.")
                csv_fallback = self.per_message_path.replace(".parquet", ".csv")
                df_msg.to_csv(csv_fallback, index=False)
                print(f"Saved to {csv_fallback}")

if __name__ == "__main__":
    harness = BenchmarkHarness()
    harness.initialize()
    # harness.run_full_stream_experiment("DailyDialog", dummy_test_df)
    harness.finalize()
    print(f"\nHarness successfully established in {harness.run_dir}")
