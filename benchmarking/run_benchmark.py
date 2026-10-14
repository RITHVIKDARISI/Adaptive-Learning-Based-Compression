import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import json
import uuid
import datetime
import pandas as pd
import numpy as np
from scipy import stats

from benchmarking.reproducibility import set_deterministic_seed, capture_environment_specs, VALID_SEEDS
from benchmarking.config import ensure_benchmark_dirs, BENCHMARK_RESULTS_DIR, SEEDS, LAMBDAS, SCHEDULERS

from src.features import FeatureExtractor
from src.bandit import LinUCBBandit
from src.simulator import StreamSimulator
from src.manager import BatchCacheManager
from src.scheduler import OfflineLabelGenerator, SupervisedScheduler, HeuristicScheduler

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

def create_benchmark_run_dir(base_dir: str = None) -> str:
    """Creates a unique timestamped directory so raw measurements are never overwritten."""
    parent = base_dir or BENCHMARK_RESULTS_DIR
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}_{uuid.uuid4().hex[:6]}"
    run_dir = os.path.join(parent, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def dump_config(run_dir: str, seeds: list = None, lambdas: list = None, schedulers: list = None):
    config_path = os.path.join(run_dir, "config.json")
    config_data = {
        "seeds": seeds if seeds is not None else SEEDS,
        "lambdas": lambdas if lambdas is not None else LAMBDAS,
        "schedulers": schedulers if schedulers is not None else SCHEDULERS,
        "warmup_iterations": 10,
        "min_microbenchmark_iterations": 30,
        "description": "Publication-quality multi-seed benchmark execution with 8-stage latency accounting and wall-clock throughput"
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)
    return config_path

def build_scheduler(scheduler_name: str, seed: int, lam: float, X_train=None, y_train=None):
    """Instantiates a deterministic scheduler model for benchmarking."""
    name = scheduler_name.strip()
    if name in ["Always-Skip", "Always_Skip"]:
        return lambda f: (0, 0.0)
    elif name in ["Always-Zstd", "Always_Zstd"]:
        return lambda f: (1, 0.0)
    elif name in ["Always-Brotli", "Always_Brotli"]:
        return lambda f: (2, 0.0)
    elif name in ["Always-Gzip", "Always_Gzip"]:
        return lambda f: (3, 0.0)
    elif name in ["Always-LZ4", "Always_LZ4"]:
        return lambda f: (4, 0.0)
    elif name in ["Fixed-Batch", "Fixed_Batch"]:
        return lambda f: (5, 0.0)
    elif name in ["Random-Action", "Random_Action"]:
        rng = np.random.RandomState(seed)
        return lambda f: (int(rng.randint(0, 6)), 0.0)
    elif name in ["Length-Threshold", "Length_Threshold"]:
        return lambda f: (0 if f.get("char_len", 0) < 25 else 1, 0.0)
    elif name == "Heuristic":
        return HeuristicScheduler(skip_threshold=25)
    elif name in ["Logistic Regression", "Logistic-Regression"]:
        model = SupervisedScheduler('logistic_regression', random_state=seed)
        if X_train is not None and y_train is not None and len(set(y_train)) > 1:
            model.fit(X_train, y_train)
        return model
    elif name in ["Decision Tree", "Decision-Tree"]:
        model = SupervisedScheduler('decision_tree', random_state=seed)
        if X_train is not None and y_train is not None and len(set(y_train)) > 1:
            model.fit(X_train, y_train)
        return model
    elif name in ["Random Forest", "Random-Forest"]:
        model = SupervisedScheduler('random_forest', random_state=seed)
        if X_train is not None and y_train is not None and len(set(y_train)) > 1:
            model.fit(X_train, y_train)
        return model
    elif name in ["Gradient Boosting", "Gradient-Boosting"]:
        model = SupervisedScheduler('gradient_boosting', random_state=seed)
        if X_train is not None and y_train is not None and len(set(y_train)) > 1:
            model.fit(X_train, y_train)
        return model
    elif name in ["LinUCB", "LinUCB-Bandit"]:
        return LinUCBBandit(d=9, K=6, alpha=0.5, lambda_param=lam)
    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")

class BenchmarkHarness:
    def __init__(self, output_dir: str = None, seeds: list = None, lambdas: list = None, schedulers: list = None):
        if output_dir:
            self.run_dir = output_dir
            os.makedirs(self.run_dir, exist_ok=True)
        else:
            self.run_dir = create_benchmark_run_dir()
            
        self.seeds = seeds if seeds is not None else SEEDS
        self.lambdas = lambdas if lambdas is not None else [0.01]
        self.schedulers = schedulers if schedulers is not None else SCHEDULERS
        
        # Output files
        self.raw_results_path = os.path.join(self.run_dir, "raw_results.csv")
        self.per_message_path = os.path.join(self.run_dir, "per_message_results.parquet")
        self.env_path = os.path.join(self.run_dir, "environment.json")
        self.config_path = os.path.join(self.run_dir, "config.json")
        self.summary_comparison_path = os.path.join(self.run_dir, "timing_comparison_summary.csv")
        
        # State
        self.raw_records = []
        self.per_message_records = []

    def initialize(self):
        print(f"Initializing Benchmark Run Directory: {self.run_dir}")
        capture_environment_specs(self.env_path)
        dump_config(self.run_dir, seeds=self.seeds, lambdas=self.lambdas, schedulers=self.schedulers)
        print("Environment and configurations locked.")

    def run_full_stream_experiment(self, dataset_name: str, test_df: pd.DataFrame, train_df: pd.DataFrame = None):
        """
        Executes full-stream scheduler experiments across all configured seeds, schedulers, and lambdas.
        """
        print(f"\n==================== Starting execution for dataset: {dataset_name} ====================")
        
        # Pre-extract training features if train_df is provided
        X_train = None
        if train_df is not None:
            print(f"Extracting features on train split ({len(train_df)} rows)...")
            fe_train = FeatureExtractor()
            X_train = [fe_train.extract_features(r.text, r.timestamp) for r in train_df.itertuples()]
            
        for lam in self.lambdas:
            y_train = None
            if train_df is not None:
                # Generate offline optimal labels for the supervised models under lambda
                print(f"Generating offline training labels for lambda={lam}...")
                label_gen = OfflineLabelGenerator(lambda_param=lam)
                y_train = label_gen.generate_labels(train_df["text"].tolist())
                
            for scheduler_name in self.schedulers:
                for seed in self.seeds:
                    set_deterministic_seed(seed)
                    model = build_scheduler(scheduler_name, seed, lam, X_train, y_train)
                    
                    manager = BatchCacheManager()
                    sim = StreamSimulator(manager, lambda_param=lam)
                    
                    # Single time.perf_counter() around the entire stream loop
                    summary = sim.run_stream(test_df, model)
                    
                    print(
                        f"  [{dataset_name} | lam={lam} | {scheduler_name:18s} | seed={seed:4d}] "
                        f"-> wall_sec={summary['wall_clock_time_sec']:.2f}s | "
                        f"thru={summary['throughput_msg_per_sec']:.1f} msg/s | "
                        f"mean_e2e={summary['mean_e2e_latency_us']:.2f} us | "
                        f"old_e2e={summary['mean_old_e2e_latency_us']:.2f} us"
                    )
                    
                    rec = {
                        "dataset": dataset_name,
                        "lambda": lam,
                        "scheduler": scheduler_name,
                        "seed": seed,
                        **summary
                    }
                    self.raw_records.append(rec)
                    
                    # Record per-message metrics
                    for s in sim.stats:
                        self.per_message_records.append({
                            "dataset": dataset_name,
                            "lambda": lam,
                            "scheduler": scheduler_name,
                            "seed": seed,
                            "msg_id": s["msg_id"],
                            "action": s["action"],
                            "cache_hit": s["cache_hit"],
                            "orig_size": s["orig_size"],
                            "comp_size": s["comp_size"],
                            "e2e_lat_us": s["e2e_lat_us"],
                            "old_e2e_lat_us": s.get("old_e2e_lat_us", s["e2e_lat_us"]),
                            "feat_lat_us": s["feat_lat_us"],
                            "cache_lookup_lat_us": s["cache_lookup_lat_us"],
                            "scheduler_lat_us": s["scheduler_lat_us"],
                            "comp_lat_us": s["comp_lat_us"],
                            "decomp_lat_us": s["decomp_lat_us"],
                            "queue_wait_lat_us": s["queue_wait_lat_us"],
                            "cache_store_lat_us": s["cache_store_lat_us"],
                            "cache_fill_comp_lat_us": s["cache_fill_comp_lat_us"],
                        })

    def finalize(self):
        """Saves all aggregated and raw data without overwriting."""
        print("\nFinalizing benchmark results...")
        
        # Save raw results (aggregated per run configuration)
        if self.raw_records:
            df_raw = pd.DataFrame(self.raw_records)
            df_raw.to_csv(self.raw_results_path, index=False)
            print(f"Saved {len(df_raw)} configuration records to {self.raw_results_path}")
            
            # Compute comparison summary across seeds
            grouped = df_raw.groupby(["dataset", "scheduler", "lambda"]).agg(
                mean_e2e_latency_us=("mean_e2e_latency_us", "mean"),
                mean_old_e2e_latency_us=("mean_old_e2e_latency_us", "mean"),
                mean_throughput_msg_per_sec=("throughput_msg_per_sec", "mean"),
                mean_throughput_mb_s=("throughput_mb_s", "mean"),
                mean_compression_ratio=("compression_ratio", "mean"),
                mean_cache_hit_rate=("cache_hit_rate", "mean")
            ).reset_index()
            
            grouped["delta_latency_us"] = grouped["mean_e2e_latency_us"] - grouped["mean_old_e2e_latency_us"]
            grouped["pct_latency_increase"] = 100.0 * (grouped["delta_latency_us"] / grouped["mean_old_e2e_latency_us"].replace(0, np.nan))
            grouped.to_csv(self.summary_comparison_path, index=False)
            print(f"Saved latency comparison summary to {self.summary_comparison_path}")
            
        # Save per-message results
        if self.per_message_records:
            df_msg = pd.DataFrame(self.per_message_records)
            try:
                df_msg.to_parquet(self.per_message_path, index=False)
                print(f"Saved {len(df_msg)} per-message records to {self.per_message_path}")
            except Exception as e:
                print(f"Parquet save failed ({e}). Falling back to CSV.")
                csv_fallback = self.per_message_path.replace(".parquet", ".csv")
                df_msg.to_csv(csv_fallback, index=False)
                print(f"Saved to {csv_fallback}")

def run_all_benchmarks(output_dir: str = None, seeds: list = None, lambdas: list = None):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = output_dir or os.path.join(project_root, "benchmark_results", "v2", "timing")
    
    harness = BenchmarkHarness(
        output_dir=target_dir,
        seeds=seeds or SEEDS,
        lambdas=lambdas or [0.01],
        schedulers=SCHEDULERS
    )
    harness.initialize()
    
    datasets = {
        "DailyDialog": os.path.join(project_root, "data", "dailydialog_clean.csv"),
        "Sentiment140": os.path.join(project_root, "data", "sentiment140_clean.csv"),
        "SMS-Spam-Ham": os.path.join(project_root, "data", "sms_spam_ham_clean.csv")
    }
    
    for name, path in datasets.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prepared dataset not found: {path}")
        df = pd.read_csv(path)
        split_idx = len(df) // 2
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        harness.run_full_stream_experiment(name, test_df=test_df, train_df=train_df)
        
    harness.finalize()
    print(f"\nAll benchmark runs completed successfully! Output in: {harness.run_dir}")

if __name__ == "__main__":
    run_all_benchmarks()
