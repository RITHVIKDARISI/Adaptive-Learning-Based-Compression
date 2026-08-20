"""
CLI Dataset Demo Utility for Faculty Presentation.

Usage:
    python test_dataset_demo.py
    python test_dataset_demo.py --dataset data/dailydialog_clean.csv --samples 50
    python test_dataset_demo.py --dataset data/nussms_clean.csv --scheduler "LinUCB"
"""

import os
import sys
import argparse
import time
import pandas as pd

from src.manager import BatchCacheManager
from src.features import FeatureExtractor
from src.engine import CompressionEngine
from src.scheduler import HeuristicScheduler, SupervisedScheduler, OfflineLabelGenerator, ACTION_MAP
from src.bandit import LinUCBBandit
from src.utils import get_data_dir

# Fix console encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_dataset(dataset_path: str) -> pd.DataFrame:
    if not os.path.exists(dataset_path):
        data_dir_path = os.path.join(get_data_dir(), os.path.basename(dataset_path))
        if os.path.exists(data_dir_path):
            dataset_path = data_dir_path
        else:
            print(f"❌ Error: Dataset file '{dataset_path}' not found.")
            sys.exit(1)

    df = pd.read_csv(dataset_path)
    if "text" not in df.columns:
        str_cols = [c for c in df.columns if df[c].dtype == "object"]
        if str_cols:
            df["text"] = df[str_cols[0]]
        else:
            print("❌ Error: Dataset CSV must contain a text column.")
            sys.exit(1)
    return df


def get_trained_scheduler(name: str, lambda_param: float = 0.01, train_path: str = None):
    name_lower = name.lower()
    if "heuristic" in name_lower:
        return HeuristicScheduler(skip_threshold=25)
    elif "bandit" in name_lower or "linucb" in name_lower:
        return LinUCBBandit(d=9, K=6, alpha=0.5, lambda_param=lambda_param)

    model_type = "decision_tree"
    if "logistic" in name_lower or "regression" in name_lower:
        model_type = "logistic_regression"
    elif "forest" in name_lower:
        model_type = "random_forest"

    # Train model on bootstrap sample
    if train_path and os.path.exists(train_path):
        df_train = pd.read_csv(train_path)
    else:
        default_train = os.path.join(get_data_dir(), "dailydialog_clean.csv")
        if os.path.exists(default_train):
            df_train = pd.read_csv(default_train)
        else:
            df_train = None

    if df_train is not None and "text" in df_train.columns:
        messages = df_train["text"].dropna().head(300).tolist()
    else:
        messages = [
            "Sample message text for training initialization " * 4,
            "Short message",
            "Hey how are you?",
            "Critical server error occurred in thread main at database connection timeout.",
            "Ok", "Thanks", "See you soon!",
            "Compression algorithms evaluate payload density and redundancy across token streams." * 3
        ] * 40

    timestamps = [i * 0.5 for i in range(len(messages))]
    fe = FeatureExtractor()
    features = [fe.extract_features(m, ts) for m, ts in zip(messages, timestamps)]
    label_gen = OfflineLabelGenerator(lambda_param=lambda_param)
    labels = label_gen.generate_labels(messages)

    model = SupervisedScheduler(model_type=model_type)
    model.fit(features, labels)
    return model


def main():
    parser = argparse.ArgumentParser(description="Adaptive Compression Scheduler - Dataset CLI Evaluation")
    parser.add_argument("--dataset", type=str, default="data/dailydialog_clean.csv", help="Path to CSV dataset file")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--scheduler", type=str, default="Decision Tree", help="Scheduler model (Heuristic, Decision Tree, Random Forest, Logistic Regression, LinUCB)")
    parser.add_argument("--lambda_param", type=float, default=0.01, help="Latency weight penalty parameter")
    args = parser.parse_args()

    print("=" * 70)
    print(" 📡 ADAPTIVE COMPRESSION SCHEDULER — DATASET EVALUATION DEMO")
    print("=" * 70)
    print(f" * Dataset File   : {args.dataset}")
    print(f" * Samples Count  : {args.samples}")
    print(f" * Scheduler Model: {args.scheduler}")
    print(f" * Lambda Param   : {args.lambda_param}")
    print("=" * 70)

    df = load_dataset(args.dataset)
    messages = df["text"].dropna().head(args.samples).tolist()

    scheduler = get_trained_scheduler(args.scheduler, lambda_param=args.lambda_param)
    manager = BatchCacheManager()
    fe = FeatureExtractor()
    engine = CompressionEngine()

    records = []
    total_raw_bytes = 0
    total_comp_bytes = 0
    cache_hits = 0
    action_counts = {}

    print(f"\nProcessing {len(messages)} messages...\n")
    print(f"{'Msg #':<6} | {'Original':<10} | {'Compressed':<10} | {'Ratio':<8} | {'Saved %':<9} | {'Action':<12} | {'Latency (µs)':<12}")
    print("-" * 80)

    for i, msg in enumerate(messages):
        msg_str = str(msg)
        ts = i * 0.5
        raw_bytes = msg_str.encode("utf-8")
        orig_size = len(raw_bytes)
        total_raw_bytes += orig_size

        start_time = time.perf_counter()

        # 1. Exact-match cache lookup
        cached = manager.cache_lookup(msg_str)
        if cached is not None:
            cache_hits += 1
            action_name = "CACHE_REUSE"
            comp_bytes, _ = cached
            comp_size = len(comp_bytes)
            elapsed_us = (time.perf_counter() - start_time) * 1_000_000
        else:
            # 2. Feature extraction
            feat = fe.extract_features(msg_str, ts)

            # 3. Scheduler action decision
            if hasattr(scheduler, "predict"):
                action_idx, sched_lat = scheduler.predict(feat)
            else:
                action_idx, sched_lat = scheduler(feat)
            action_name = ACTION_MAP.get(action_idx, "SKIP")

            # 4. Action execution
            if action_name == "SKIP":
                comp_size = orig_size
            elif action_name in ["ZSTD", "BROTLI", "GZIP", "LZ4"]:
                comp_bytes, comp_lat = engine.compress(msg_str, action_name)
                comp_size = len(comp_bytes)
                manager.cache_store(msg_str, comp_bytes, action_name)
            elif action_name == "BATCH":
                comp_size = max(int(orig_size * 0.5), 1)

            # 5. Online bandit update
            if hasattr(scheduler, "update"):
                reward = -(comp_size + args.lambda_param * sched_lat)
                scheduler.update(action_idx, feat, reward)

            elapsed_us = (time.perf_counter() - start_time) * 1_000_000

        total_comp_bytes += comp_size
        ratio = comp_size / orig_size if orig_size > 0 else 1.0
        saved_pct = ((orig_size - comp_size) / orig_size * 100.0) if orig_size > 0 else 0.0
        action_counts[action_name] = action_counts.get(action_name, 0) + 1

        records.append({
            "msg_id": i + 1,
            "orig_size": orig_size,
            "comp_size": comp_size,
            "ratio": ratio,
            "saved_pct": saved_pct,
            "action": action_name,
            "latency_us": elapsed_us
        })

        if i < 15 or i >= len(messages) - 5:
            print(f"{i + 1:<6} | {orig_size:<10} | {comp_size:<10} | {ratio:<8.3f} | {saved_pct:<+8.1f}% | {action_name:<12} | {elapsed_us:<12.1f}")
        elif i == 15:
            print(f"{'...':<6} | {'...':<10} | {'...':<10} | {'...':<8} | {'...':<9} | {'...':<12} | {'...':<12}")

    overall_ratio = total_comp_bytes / total_raw_bytes if total_raw_bytes > 0 else 1.0
    overall_saved = ((total_raw_bytes - total_comp_bytes) / total_raw_bytes * 100.0) if total_raw_bytes > 0 else 0.0
    mean_lat = sum(r["latency_us"] for r in records) / len(records) if records else 0.0
    cache_rate = (cache_hits / len(records) * 100.0) if records else 0.0

    print("\n📊 EVALUATION SUMMARY METRICS")
    print("=" * 40)
    print(f" Total Messages Processed : {len(records)}")
    print(f" Total Original Data Size : {total_raw_bytes:,} bytes")
    print(f" Total Compressed Size    : {total_comp_bytes:,} bytes")
    print(f" Overall Compression Ratio: {overall_ratio:.3f}")
    print(f" Overall Space Saved      : {overall_saved:+.2f}%")
    print(f" Mean End-to-End Latency  : {mean_lat:.2f} µs")
    print(f" Cache Hit Rate           : {cache_rate:.1f}%")
    print("=" * 40)

    print("\n⚡ ACTION BREAKDOWN")
    print("-" * 30)
    for act, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(records) * 100.0
        print(f" {act:<12}: {cnt:>4} msgs ({pct:>5.1f}%)")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
