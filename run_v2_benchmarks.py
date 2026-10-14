"""
Master v2 benchmark runner.
Runs ALL experiments from scratch and writes results to benchmark_results/v2/.
Never overwrites existing result files.

Produces:
  - benchmark_results/v2/experiment_results.json  (full scheduler comparison across 3 datasets x 4 lambdas)
  - benchmark_results/v2/environment.json           (machine + package specs)
  - benchmark_results/v2/dataset_statistics.csv      (per-dataset stats)
  - benchmark_results/v2/pareto_frontier.png          (Figure 1: Pareto)
  - benchmark_results/v2/feature_importance.png       (Figure 2: DT importance)
  - benchmark_results/v2/final_comparison_table.csv   (for paper table)
  - data/MANIFEST.json                                (dataset manifest with SHA256)
  - SUMMARY_FOR_PAPER.md                             (summary for paper)
"""
import os
import sys
import json
import time
import hashlib
import platform
import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.engine import CompressionEngine
from src.features import FeatureExtractor
from src.bandit import LinUCBBandit
from src.simulator import StreamSimulator
from src.manager import BatchCacheManager
from src.scheduler import (
    OfflineLabelGenerator, SupervisedScheduler, HeuristicScheduler,
    ACTION_MAP
)

# ── Config ──────────────────────────────────────────────────────────────
V2_DIR = os.path.join(PROJECT_ROOT, "benchmark_results", "v2")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LAMBDAS = [0.001, 0.01, 0.1, 1.0]
SEED = 42

os.makedirs(V2_DIR, exist_ok=True)

# ── Helpers ─────────────────────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def capture_environment():
    import psutil
    env = {
        "timestamp": datetime.datetime.now().isoformat(),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": sys.version,
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "packages": {}
    }
    for pkg_name, import_name in [
        ("zstandard", "zstandard"), ("brotli", "brotli"), ("lz4", "lz4"),
        ("scikit-learn", "sklearn"), ("pandas", "pandas"), ("numpy", "numpy"),
        ("matplotlib", "matplotlib"), ("datasketch", "datasketch"),
        ("scipy", "scipy"), ("psutil", "psutil")
    ]:
        try:
            mod = __import__(import_name)
            env["packages"][pkg_name] = getattr(mod, "__version__", getattr(mod, "VERSION", "unknown"))
        except ImportError:
            env["packages"][pkg_name] = "not installed"
    return env

def load_dataset(name, filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset {path} not found. Run data_loader first.")
    df = pd.read_csv(path)
    print(f"  Loaded {name}: {len(df)} rows from {filename}")
    return df

# ── Core benchmark ──────────────────────────────────────────────────────

def run_stream_benchmark(dataset_name, df, lambdas):
    """Run full benchmark for one dataset. Returns nested dict of results."""
    np.random.seed(SEED)
    
    split_idx = len(df) // 2
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Extract training features
    fe_train = FeatureExtractor()
    X_train = []
    for _, row in train_df.iterrows():
        feats = fe_train.extract_features(row["text"], row["timestamp"])
        X_train.append(feats)
    
    results = {}
    
    # ── Static baselines (lambda-independent) ──
    print(f"  Running static baselines...")
    static_baselines = {
        "Always-Skip":  lambda f: (0, 0.0),
        "Always-Zstd":  lambda f: (1, 0.0),
        "Always-Brotli": lambda f: (2, 0.0),
        "Always-Gzip":  lambda f: (3, 0.0),
        "Always-LZ4":   lambda f: (4, 0.0),
        "Fixed-Batch":  lambda f: (5, 0.0),
        "Heuristic":    HeuristicScheduler(skip_threshold=25),
    }
    
    for name, model in static_baselines.items():
        manager = BatchCacheManager()
        sim = StreamSimulator(manager)
        sim.reset()
        for _, row in test_df.iterrows():
            sim.run_message(row["msg_id"], row["text"], row["timestamp"], model)
        sim.finalize_stream(test_df.iloc[-1]["timestamp"], model)
        results[name] = {"static": sim.get_summary_metrics()}
    
    # ── Lambda-dependent schedulers ──
    for lam in lambdas:
        print(f"  Lambda = {lam}...")
        lam_key = f"lambda_{lam}"
        results[lam_key] = {}
        
        # Generate offline labels
        label_gen = OfflineLabelGenerator(lambda_param=lam)
        y_train = label_gen.generate_labels(train_df["text"].tolist())
        
        # Supervised schedulers
        supervised_models = {
            "Logistic-Regression": SupervisedScheduler('logistic_regression'),
            "Decision-Tree":      SupervisedScheduler('decision_tree'),
            "Random-Forest":      SupervisedScheduler('random_forest'),
            "Gradient-Boosting":  SupervisedScheduler('gradient_boosting'),
        }
        
        for m_name, m_wrapper in supervised_models.items():
            if len(set(y_train)) < 2:
                default_class = y_train[0] if y_train else 0
                class DummyModel:
                    def predict(self, f): return default_class, 1.0
                model_to_use = DummyModel()
            else:
                m_wrapper.fit(X_train, y_train)
                model_to_use = m_wrapper
            
            manager = BatchCacheManager()
            sim = StreamSimulator(manager, lambda_param=lam)
            sim.reset()
            for _, row in test_df.iterrows():
                sim.run_message(row["msg_id"], row["text"], row["timestamp"], model_to_use)
            sim.finalize_stream(test_df.iloc[-1]["timestamp"], model_to_use)
            
            metrics = sim.get_summary_metrics()
            
            # Capture feature importances for DT
            if m_name == "Decision-Tree" and hasattr(model_to_use, 'get_feature_importances'):
                metrics["feature_importances"] = m_wrapper.get_feature_importances()
            
            results[lam_key][m_name] = metrics
        
        # LinUCB bandit (online)
        bandit = LinUCBBandit(d=9, K=6, alpha=0.5, lambda_param=0.1)
        manager = BatchCacheManager()
        sim = StreamSimulator(manager, lambda_param=lam)
        sim.reset()
        for _, row in test_df.iterrows():
            sim.run_message(row["msg_id"], row["text"], row["timestamp"], bandit)
        sim.finalize_stream(test_df.iloc[-1]["timestamp"], bandit)
        results[lam_key]["LinUCB-Bandit"] = sim.get_summary_metrics()
    
    return results


# ── Dataset statistics ──────────────────────────────────────────────────

def compute_dataset_stats(datasets):
    rows = []
    for name, (filename, df) in datasets.items():
        texts = df["text"].dropna().tolist()
        lengths = [len(t.encode('utf-8')) for t in texts]
        rows.append({
            "Dataset": name,
            "File": filename,
            "Rows": len(df),
            "Mean_Len_Bytes": np.mean(lengths),
            "Median_Len_Bytes": np.median(lengths),
            "Min_Len_Bytes": np.min(lengths),
            "Max_Len_Bytes": np.max(lengths),
            "Std_Len_Bytes": np.std(lengths),
            "Unique_Texts": len(set(texts)),
            "Duplicate_Rate": 1.0 - len(set(texts)) / len(texts),
        })
    return pd.DataFrame(rows)


# ── Figures ─────────────────────────────────────────────────────────────

def plot_pareto_figure(all_results, output_path):
    """Figure 1: Pareto frontier across all datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    
    colors = {
        "Logistic-Regression": "#3B82F6",
        "Decision-Tree": "#10B981",
        "Random-Forest": "#F59E0B",
        "Gradient-Boosting": "#EC4899",
        "LinUCB-Bandit": "#8B5CF6",
        "Heuristic": "#6B7280",
        "Always-Skip": "#9CA3AF",
        "Always-Zstd": "#2563EB",
        "Always-Brotli": "#7C3AED",
        "Always-Gzip": "#D97706",
        "Always-LZ4": "#059669",
        "Fixed-Batch": "#DB2777",
    }
    
    for ax, (d_name, d_data) in zip(axes, all_results.items()):
        # Static baselines
        for k, v in d_data.items():
            if isinstance(v, dict) and "static" in v:
                m = v["static"]
                lat_ms = m.get("mean_e2e_latency_us", 0) / 1000.0
                ratio = m.get("compression_ratio", 1.0)
                marker = "*" if "Heuristic" in k else "D"
                size = 120 if "Heuristic" in k else 70
                ax.scatter(lat_ms, ratio, color=colors.get(k, "#64748B"),
                          s=size, marker=marker, label=k, zorder=5)
        
        # Adaptive curves across lambda
        adaptive_models = ["Logistic-Regression", "Decision-Tree", "Random-Forest",
                          "Gradient-Boosting", "LinUCB-Bandit"]
        lambda_keys = sorted([k for k in d_data.keys() if k.startswith("lambda_")])
        
        for model in adaptive_models:
            lats, ratios = [], []
            for lk in lambda_keys:
                if model in d_data[lk]:
                    m = d_data[lk][model]
                    lats.append(m.get("mean_e2e_latency_us", 0) / 1000.0)
                    ratios.append(m.get("compression_ratio", 1.0))
            if lats and ratios:
                ax.plot(lats, ratios, marker="o", linewidth=2, label=model,
                       color=colors.get(model, "#000"))
        
        ax.set_title(d_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean E2E Latency (ms)")
        ax.set_ylabel("Compression Ratio (higher = better)")
        ax.grid(True, alpha=0.3)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6,
              bbox_to_anchor=(0.5, -0.08), frameon=True, fontsize=9)
    fig.suptitle("Pareto Frontier: Compression Ratio vs Latency", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def plot_feature_importance_figure(all_results, output_path):
    """Figure 2: Decision Tree feature importances at lambda=0.01."""
    fi = None
    source_dataset = None
    for d_name, d_data in all_results.items():
        l_data = d_data.get("lambda_0.01", {})
        dt = l_data.get("Decision-Tree", {})
        if "feature_importances" in dt:
            fi = dt["feature_importances"]
            source_dataset = d_name
            break
    
    if fi is None:
        print("  ERROR: No feature importance data found!")
        return
    
    # Sort descending
    pairs = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    features = [p[0] for p in pairs][::-1]
    importances = [p[1] * 100 for p in pairs][::-1]
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.barh(features, importances, color="#10B981")
    ax.set_xlabel("Importance (%)", fontsize=11)
    ax.set_title(f"Decision Tree Feature Importance (lambda=0.01, {source_dataset})",
                fontsize=13, fontweight="bold")
    
    for bar in bars:
        w = bar.get_width()
        if w > 0.5:
            ax.text(w + 1, bar.get_y() + bar.get_height()/2,
                   f"{w:.1f}%", va="center", fontsize=9)
    
    ax.set_xlim(0, max(importances) * 1.15)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


# ── Comparison table for paper ──────────────────────────────────────────

def build_comparison_table(all_results):
    """Builds the paper's main comparison table at lambda=0.01."""
    rows = []
    lam_key = "lambda_0.01"
    
    for d_name, d_data in all_results.items():
        # Static baselines
        for sched_name in ["Always-Skip", "Always-Zstd", "Always-Brotli", "Always-Gzip",
                           "Always-LZ4", "Fixed-Batch", "Heuristic"]:
            if sched_name in d_data and "static" in d_data[sched_name]:
                m = d_data[sched_name]["static"]
                rows.append({
                    "Dataset": d_name,
                    "Scheduler": sched_name,
                    "Lambda": 0.01,
                    "Compression_Ratio": m.get("compression_ratio", 0),
                    "Space_Saving_Pct": m.get("space_saving_pct", 0),
                    "Mean_E2E_Latency_us": m.get("mean_e2e_latency_us", 0),
                    "P95_E2E_Latency_us": m.get("p95_e2e_latency_us", 0),
                    "P99_E2E_Latency_us": m.get("p99_e2e_latency_us", 0),
                    "Cache_Hit_Rate": m.get("cache_hit_rate", 0),
                    "Throughput_msg_s": m.get("throughput_msg_per_sec", 0),
                    "Total_Cost": m.get("total_compute_cost", 0),
                })
        
        # Adaptive schedulers at lambda=0.01
        l_data = d_data.get(lam_key, {})
        for sched_name in ["Logistic-Regression", "Decision-Tree", "Random-Forest",
                           "Gradient-Boosting", "LinUCB-Bandit"]:
            if sched_name in l_data:
                m = l_data[sched_name]
                rows.append({
                    "Dataset": d_name,
                    "Scheduler": sched_name,
                    "Lambda": 0.01,
                    "Compression_Ratio": m.get("compression_ratio", 0),
                    "Space_Saving_Pct": m.get("space_saving_pct", 0),
                    "Mean_E2E_Latency_us": m.get("mean_e2e_latency_us", 0),
                    "P95_E2E_Latency_us": m.get("p95_e2e_latency_us", 0),
                    "P99_E2E_Latency_us": m.get("p99_e2e_latency_us", 0),
                    "Cache_Hit_Rate": m.get("cache_hit_rate", 0),
                    "Throughput_msg_s": m.get("throughput_msg_per_sec", 0),
                    "Total_Cost": m.get("total_compute_cost", 0),
                })
    
    return pd.DataFrame(rows)


# ── MANIFEST.json ───────────────────────────────────────────────────────

def build_manifest(dataset_files):
    manifest = {}
    for name, filename in dataset_files.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            manifest[filename] = {
                "dataset_name": name,
                "sha256": sha256_file(path),
                "row_count": len(df),
                "file_size_bytes": os.path.getsize(path),
                "columns": list(df.columns),
            }
    return manifest


def df_to_markdown(df, floatfmt=".4f"):
    try:
        import tabulate
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for _, row in df.iterrows():
            formatted = []
            for val in row:
                if isinstance(val, (float, np.floating)):
                    formatted.append(f"{val:.4f}")
                else:
                    formatted.append(str(val))
            rows.append("| " + " | ".join(formatted) + " |")
        return "\n".join([header, sep] + rows)


# ── SUMMARY_FOR_PAPER.md ───────────────────────────────────────────────

def build_summary(all_results, comparison_df, dataset_stats_df, env):
    lines = []
    lines.append("# SUMMARY FOR PAPER\n")
    lines.append(f"**Generated**: {datetime.datetime.now().isoformat()}\n")
    lines.append(f"**System**: {env['os']}, {env['cpu']}, {env['ram_gb']} GB RAM\n")
    lines.append(f"**Python**: {env['python_version'].split()[0]}\n")
    lines.append("")
    
    # Dataset statistics
    lines.append("## Dataset Statistics\n")
    lines.append(df_to_markdown(dataset_stats_df))
    lines.append("")
    
    # Main comparison at lambda=0.01
    lines.append("## Main Results (lambda = 0.01)\n")
    for d_name in comparison_df["Dataset"].unique():
        lines.append(f"\n### {d_name}\n")
        sub = comparison_df[comparison_df["Dataset"] == d_name].copy()
        sub = sub.sort_values("Total_Cost")
        display_cols = ["Scheduler", "Compression_Ratio", "Space_Saving_Pct",
                       "Mean_E2E_Latency_us", "P99_E2E_Latency_us",
                       "Cache_Hit_Rate", "Total_Cost"]
        lines.append(df_to_markdown(sub[display_cols], floatfmt=".4f"))
        lines.append("")
    
    # Best scheduler per dataset
    lines.append("## Best Scheduler by Dataset (lowest total cost at lambda=0.01)\n")
    for d_name in comparison_df["Dataset"].unique():
        sub = comparison_df[comparison_df["Dataset"] == d_name]
        best = sub.loc[sub["Total_Cost"].idxmin()]
        lines.append(f"- **{d_name}**: {best['Scheduler']} "
                     f"(cost={best['Total_Cost']:.2f}, ratio={best['Compression_Ratio']:.4f}, "
                     f"latency={best['Mean_E2E_Latency_us']:.1f} us)")
    lines.append("")
    
    # Feature importances
    lines.append("## Decision Tree Feature Importances (lambda=0.01)\n")
    for d_name, d_data in all_results.items():
        l_data = d_data.get("lambda_0.01", {})
        dt = l_data.get("Decision-Tree", {})
        if "feature_importances" in dt:
            fi = dt["feature_importances"]
            lines.append(f"### {d_name}\n")
            sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            for feat, imp in sorted_fi:
                lines.append(f"- {feat}: {imp*100:.2f}%")
            lines.append("")
            break
    
    # Lambda sweep summary
    lines.append("## Lambda Sweep Summary\n")
    lines.append("| Dataset | Lambda | Best Scheduler | Cost | Ratio |")
    lines.append("|---------|--------|---------------|------|-------|")
    for d_name, d_data in all_results.items():
        for lam in LAMBDAS:
            lam_key = f"lambda_{lam}"
            if lam_key in d_data:
                best_name = None
                best_cost = float('inf')
                best_ratio = 0
                for sched_name, metrics in d_data[lam_key].items():
                    cost = metrics.get("total_compute_cost", float('inf'))
                    if cost < best_cost:
                        best_cost = cost
                        best_name = sched_name
                        best_ratio = metrics.get("compression_ratio", 0)
                if best_name:
                    lines.append(f"| {d_name} | {lam} | {best_name} | {best_cost:.2f} | {best_ratio:.4f} |")
    lines.append("")
    
    lines.append("## Figures\n")
    lines.append("1. `benchmark_results/v2/pareto_frontier.png` - Pareto frontier across 3 datasets")
    lines.append("2. `benchmark_results/v2/feature_importance.png` - Decision Tree feature importance breakdown")
    lines.append("")
    
    lines.append("## Artifacts\n")
    lines.append("- `benchmark_results/v2/experiment_results.json` - Full raw results")
    lines.append("- `benchmark_results/v2/final_comparison_table.csv` - Main comparison table")
    lines.append("- `benchmark_results/v2/dataset_statistics.csv` - Dataset statistics")
    lines.append("- `benchmark_results/v2/environment.json` - Environment specs")
    lines.append("- `data/MANIFEST.json` - Dataset provenance and checksums")
    
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    start = time.time()
    print("=" * 60)
    print("V2 BENCHMARK RUNNER - Paper-Quality Results")
    print("=" * 60)
    
    # 1. Environment
    print("\n[1/7] Capturing environment...")
    env = capture_environment()
    env_path = os.path.join(V2_DIR, "environment.json")
    with open(env_path, "w") as f:
        json.dump(env, f, indent=4)
    print(f"  Saved {env_path}")
    
    # 2. Load datasets
    print("\n[2/7] Loading datasets...")
    dataset_files = {
        "DailyDialog": "dailydialog_clean.csv",
        "Sentiment140": "sentiment140_clean.csv",
        "NUS-SMS": "nussms_clean.csv",
    }
    datasets = {}
    for name, filename in dataset_files.items():
        df = load_dataset(name, filename)
        datasets[name] = (filename, df)
    
    # 3. Dataset statistics
    print("\n[3/7] Computing dataset statistics...")
    ds_stats = compute_dataset_stats(datasets)
    ds_stats_path = os.path.join(V2_DIR, "dataset_statistics.csv")
    ds_stats.to_csv(ds_stats_path, index=False)
    print(f"  Saved {ds_stats_path}")
    
    results_path = os.path.join(V2_DIR, "experiment_results.json")
    rerun = "--force" in sys.argv
    if os.path.exists(results_path) and not rerun:
        print(f"\n[4/7] Loading existing experiment results from {results_path} (pass --force to re-run)...")
        with open(results_path, "r") as f:
            all_results = json.load(f)
    else:
        # 4. Run benchmarks
        print("\n[4/7] Running benchmarks (this may take several minutes)...")
        all_results = {}
        for name, (filename, df) in datasets.items():
            print(f"\n  ---- {name} ----")
            all_results[name] = run_stream_benchmark(name, df, LAMBDAS)
        
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\n  Saved {results_path}")
    
    # 5. Build comparison table
    print("\n[5/7] Building comparison table...")
    comparison_df = build_comparison_table(all_results)
    table_path = os.path.join(V2_DIR, "final_comparison_table.csv")
    comparison_df.to_csv(table_path, index=False)
    print(f"  Saved {table_path}")
    
    # 6. Generate figures
    print("\n[6/7] Generating figures...")
    plot_pareto_figure(all_results, os.path.join(V2_DIR, "pareto_frontier.png"))
    plot_feature_importance_figure(all_results, os.path.join(V2_DIR, "feature_importance.png"))
    
    # 7. Build manifest and summary
    print("\n[7/7] Building MANIFEST and SUMMARY...")
    manifest = build_manifest(dataset_files)
    manifest_path = os.path.join(DATA_DIR, "MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
    print(f"  Saved {manifest_path}")
    
    summary_text = build_summary(all_results, comparison_df, ds_stats, env)
    summary_path = os.path.join(PROJECT_ROOT, "SUMMARY_FOR_PAPER.md")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"  Saved {summary_path}")
    
    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Results in: {V2_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
