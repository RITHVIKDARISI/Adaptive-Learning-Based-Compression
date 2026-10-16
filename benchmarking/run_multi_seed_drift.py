import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from benchmarking.config import BENCHMARK_RESULTS_DIR
from benchmarking.synthetic_streams import generate_drift_stream
from src.features import FeatureExtractor
from src.manager import BatchCacheManager
from src.simulator import StreamSimulator
from src.scheduler import (
    SupervisedScheduler,
    HeuristicScheduler,
    OfflineLabelGenerator,
    SimulatorAlignedOracle,
    ACTION_MAP,
)
from src.bandit import LinUCBBandit

SEEDS = [42, 123, 456, 789, 2026, 11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333, 444, 555, 777]

SCENARIOS = [
    ("drift_1_repL_uniqS", 5000),
    ("drift_2_uniqS_repL", 5000),
    ("drift_3_lowRate_burst", 5000),
    ("drift_4_nat_json", 5000),
    ("drift_5_lowDup_highDup", 5000),
    ("drift_6_alternating", 5000),
    ("drift_7_natural_sequential", 6000),
    ("drift_8_natural_switching", 6000),
]

POLICIES_TO_EVAL = [
    "Always-Skip",
    "Always-Zstd",
    "Always-Brotli",
    "Always-Gzip",
    "Always-LZ4",
    "Heuristic (no deadline)",
    "Heuristic (deadline-aware)",
    "LinUCB (no deadline)",
    "LinUCB (deadline-aware)",
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
]

def calculate_adaptation_lag(cost_series: list[float], oracle_series: list[float], change_point: int, window: int = 100, threshold: float = 1.10) -> int:
    """
    Measures adaptation lag as the number of messages after change_point
    until the rolling average cost of the policy is <= threshold * rolling average oracle cost
    for at least 3 consecutive windows.
    Returns:
        lag (int >= 0) if converged, or -1 if did not converge.
    """
    if change_point >= len(cost_series) - window:
        return -1

    consecutive_success = 0
    for i in range(change_point, len(cost_series) - window + 1):
        rolling_cost = np.mean(cost_series[i : i + window])
        rolling_oracle = np.mean(oracle_series[i : i + window])

        if rolling_cost <= rolling_oracle * threshold + 1e-6:
            consecutive_success += 1
            if consecutive_success >= 3:
                return (i - 2) - change_point
        else:
            consecutive_success = 0

    return -1

def compute_bootstrap_ci(data: list[float], n_resamples: int = 10000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Computes a 95% percentile bootstrap confidence interval for the sample mean."""
    rng = np.random.default_rng(seed)
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
    resampled_means = [float(np.mean(rng.choice(data, size=n, replace=True))) for _ in range(n_resamples)]
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(resampled_means, alpha * 100))
    upper = float(np.percentile(resampled_means, (1.0 - alpha) * 100))
    return float(np.mean(data)), lower, upper

def holm_bonferroni_correction(p_values: list[float]) -> list[float]:
    """Computes Holm-Bonferroni adjusted p-values enforcing step-down monotonicity."""
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    adjusted_p = [0.0] * m
    running_max = 0.0
    for rank, orig_idx in enumerate(sorted_indices):
        p_raw = p_values[orig_idx]
        adj = (m - rank) * p_raw
        running_max = max(running_max, adj)
        adjusted_p[orig_idx] = min(1.0, float(running_max))
    return adjusted_p

def paired_wilcoxon_tests(linucb_costs: list[float], comparisons_dict: dict[str, list[float]]) -> dict[str, dict]:
    """
    Computes paired Wilcoxon signed-rank tests for LinUCB vs each comparison policy across seeds.
    Calculates rank-biserial correlation r and paired Cohen's d_z.
    """
    results = {}
    p_raw_list = []
    pol_names = list(comparisons_dict.keys())

    for pol in pol_names:
        x = np.array(linucb_costs)
        y = np.array(comparisons_dict[pol])
        diff = x - y  # diff < 0 means LinUCB has lower cost (better)

        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))
        dz = mean_diff / std_diff if std_diff > 1e-9 else 0.0

        if np.all(np.abs(diff) < 1e-9):
            stat = 0.0
            p_val = 1.0
            r_biserial = 0.0
        else:
            try:
                res = stats.wilcoxon(diff, zero_method='pratt')
                stat = float(res.statistic)
                p_val = float(res.pvalue)
            except Exception:
                stat = 0.0
                p_val = 1.0

            abs_diff = np.abs(diff)
            ranks = stats.rankdata(abs_diff)
            w_pos = np.sum(ranks[diff > 0])
            w_neg = np.sum(ranks[diff < 0])
            total_w = w_pos + w_neg
            r_biserial = float((w_pos - w_neg) / total_w) if total_w > 0 else 0.0

        results[pol] = {
            "policy": pol,
            "mean_diff": mean_diff,
            "statistic": stat,
            "p_raw": p_val,
            "rank_biserial": r_biserial,
            "cohens_dz": dz,
        }
        p_raw_list.append(p_val)

    p_holm_list = holm_bonferroni_correction(p_raw_list)
    for i, pol in enumerate(pol_names):
        p_holm = p_holm_list[i]
        results[pol]["p_holm"] = p_holm
        mean_diff = results[pol]["mean_diff"]
        if p_holm < 0.05:
            if mean_diff < 0:
                results[pol]["winner"] = "LinUCB (p < 0.05)"
            else:
                results[pol]["winner"] = f"{pol} (p < 0.05)"
        else:
            results[pol]["winner"] = "Tied / Not Significant"

    return results

def run_single_stream_eval(scenario_name: str, sched_name: str, messages: list[str], timestamps: list[float],
                           seed: int, train_size: int = 500, max_wait_us: float = 50000.0,
                           train_feats: list[dict] = None, eval_feats: list[dict] = None) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    train_messages = messages[:train_size]
    train_timestamps = timestamps[:train_size]
    eval_messages = messages[train_size:]
    eval_timestamps = timestamps[train_size:]

    is_degenerate = False
    degenerate_class = None

    if sched_name == "Always-Skip":
        scheduler = lambda f: (0, 0.0)
    elif sched_name == "Always-Zstd":
        scheduler = lambda f: (1, 0.0)
    elif sched_name == "Always-Brotli":
        scheduler = lambda f: (2, 0.0)
    elif sched_name == "Always-Gzip":
        scheduler = lambda f: (3, 0.0)
    elif sched_name == "Always-LZ4":
        scheduler = lambda f: (4, 0.0)
    elif sched_name == "Heuristic (no deadline)":
        scheduler = HeuristicScheduler(enable_deadline=False)
    elif sched_name == "Heuristic (deadline-aware)":
        scheduler = HeuristicScheduler(enable_deadline=True, max_wait_us=max_wait_us)
    elif sched_name == "LinUCB (no deadline)":
        scheduler = LinUCBBandit(cost_aware_exploration=True, enable_deadline=False)
        sim_train = StreamSimulator(BatchCacheManager(), lambda_param=0.01)
        for i, (m, ts) in enumerate(zip(train_messages, train_timestamps)):
            pf = train_feats[i] if train_feats is not None else None
            sim_train.run_message(i, m, ts, scheduler, precomputed_features=pf)
        sim_train.finalize_stream(train_timestamps[-1] + 0.1, scheduler)
    elif sched_name == "LinUCB (deadline-aware)":
        scheduler = LinUCBBandit(cost_aware_exploration=True, enable_deadline=True, max_wait_us=max_wait_us)
        sim_train = StreamSimulator(BatchCacheManager(), lambda_param=0.01)
        for i, (m, ts) in enumerate(zip(train_messages, train_timestamps)):
            pf = train_feats[i] if train_feats is not None else None
            sim_train.run_message(i, m, ts, scheduler, precomputed_features=pf)
        sim_train.finalize_stream(train_timestamps[-1] + 0.1, scheduler)
    elif sched_name in ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]:
        model_type_map = {
            "Logistic Regression": "logistic_regression",
            "Decision Tree": "decision_tree",
            "Random Forest": "random_forest",
            "Gradient Boosting": "gradient_boosting"
        }
        scheduler = SupervisedScheduler(model_type_map[sched_name], random_state=seed)
        label_gen = OfflineLabelGenerator(lambda_param=0.01)
        train_labels = label_gen.generate_labels(train_messages)
        unique_labels = sorted(list(set(train_labels)))
        if len(unique_labels) <= 1:
            is_degenerate = True
            degenerate_class = unique_labels[0] if unique_labels else 0
            scheduler.is_degenerate = True
            scheduler.degenerate_class = degenerate_class
            preds = [degenerate_class] * len(eval_messages)
            per_msg_lat = 0.0
        else:
            scheduler.fit(train_feats, train_labels)
            X_eval = np.zeros((len(eval_feats), len(scheduler.feature_names)))
            for i, feat in enumerate(eval_feats):
                for j, name in enumerate(scheduler.feature_names):
                    X_eval[i, j] = feat.get(name, 0.0)
            t_start = time.perf_counter()
            preds = [int(p) for p in scheduler.model.predict(X_eval)]
            per_msg_lat = ((time.perf_counter() - t_start) / len(eval_feats)) * 1_000_000

        class BatchPredScheduler:
            def __init__(self, pred_list, lat):
                self.preds = pred_list
                self.lat = lat
                self.idx = 0
            def predict(self, f):
                a = self.preds[self.idx]
                self.idx += 1
                return a, self.lat

        scheduler = BatchPredScheduler(preds, per_msg_lat)
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")

    manager = BatchCacheManager()
    sim = StreamSimulator(manager, lambda_param=0.01)
    for i, (m, ts) in enumerate(zip(eval_messages, eval_timestamps)):
        pf = eval_feats[i] if eval_feats is not None else None
        sim.run_message(i, m, ts, scheduler, precomputed_features=pf)
    sim.finalize_stream(eval_timestamps[-1] + 1.0, scheduler)

    costs = [s["user_perceived_cost"] for s in sim.stats]
    latencies = [s["e2e_lat_us"] for s in sim.stats]

    return {
        "costs": costs,
        "latencies": latencies,
        "is_degenerate": is_degenerate,
        "degenerate_class": degenerate_class
    }

def run_all_drift_benchmarks():
    out_dir = os.path.join(BENCHMARK_RESULTS_DIR, "v2", "drift")
    os.makedirs(out_dir, exist_ok=True)
    raw_csv_path = os.path.join(out_dir, "raw_drift_results.csv")
    summary_md_path = os.path.join(out_dir, "summary.md")

    train_size = 500
    raw_records = []
    completed_scenarios = set()

    if os.path.exists(raw_csv_path):
        try:
            df_existing = pd.read_csv(raw_csv_path)
            for sc in df_existing["scenario"].unique():
                sub = df_existing[df_existing["scenario"] == sc]
                if len(sub["seed"].unique()) == len(SEEDS):
                    completed_scenarios.add(sc)
                    print(f"Loaded {len(sub)} existing records for completed scenario: {sc}", flush=True)
                    raw_records.extend(sub.to_dict(orient="records"))
        except Exception as e:
            print(f"Could not load existing CSV: {e}", flush=True)

    print(f"Starting Multi-Seed Drift Evaluation: 8 scenarios, {len(SEEDS)} seeds (160 fresh independent streams)...", flush=True)
    start_total_time = time.time()

    for scen_idx, (scenario_name, target_samples) in enumerate(SCENARIOS, 1):
        if scenario_name in completed_scenarios:
            print(f"\nSkipping already completed scenario [{scen_idx}/{len(SCENARIOS)}]: {scenario_name}", flush=True)
            continue
        print(f"\n=======================================================", flush=True)
        print(f"[{scen_idx}/{len(SCENARIOS)}] Scenario: {scenario_name} (Target len: {target_samples})", flush=True)
        print(f"=======================================================", flush=True)

        scen_start_time = time.time()

        for seed_idx, seed in enumerate(SEEDS, 1):
            seed_start_time = time.time()
            messages, timestamps, change_point = generate_drift_stream(scenario_name, seed=seed)
            assert len(messages) >= 5000, f"Scenario {scenario_name} seed {seed} generated < 5000 messages!"

            # 1. Base feature extraction
            fe = FeatureExtractor()
            all_feats = [fe.extract_features(m, ts) for m, ts in zip(messages, timestamps)]
            train_feats = all_feats[:train_size]
            eval_feats = all_feats[train_size:]

            eval_messages = messages[train_size:]
            eval_timestamps = timestamps[train_size:]
            eval_change_point = max(0, change_point - train_size)

            # 2. SimulatorAlignedOracle on evaluation set
            oracle_gen = SimulatorAlignedOracle(lambda_param=0.01)
            oracle_labels = oracle_gen.generate_labels(eval_messages, eval_timestamps)

            class OracleScheduler:
                def __init__(self, labels):
                    self.labels = labels
                    self.idx = 0
                def predict(self, f):
                    a = self.labels[self.idx]
                    self.idx += 1
                    return a, 0.0

            manager_or = BatchCacheManager()
            sim_or = StreamSimulator(manager_or, lambda_param=0.01)
            or_sched = OracleScheduler(oracle_labels)
            for i, (m, ts) in enumerate(zip(eval_messages, eval_timestamps)):
                sim_or.run_message(i, m, ts, or_sched, precomputed_features=eval_feats[i])
            sim_or.finalize_stream(eval_timestamps[-1] + 1.0, or_sched)

            oracle_costs = [s["user_perceived_cost"] for s in sim_or.stats]
            oracle_latencies = [s["e2e_lat_us"] for s in sim_or.stats]
            oracle_mean_cost = float(np.mean(oracle_costs))

            # Store Oracle record
            raw_records.append({
                "scenario": scenario_name,
                "seed": seed,
                "policy": "Oracle",
                "mean_cost": oracle_mean_cost,
                "total_cost": float(np.sum(oracle_costs)),
                "oracle_cost": oracle_mean_cost,
                "cost_gap": 0.0,
                "adaptation_lag": 0,
                "p50_latency_us": float(np.percentile(oracle_latencies, 50)),
                "p95_latency_us": float(np.percentile(oracle_latencies, 95)),
                "p99_latency_us": float(np.percentile(oracle_latencies, 99)),
                "mean_latency_us": float(np.mean(oracle_latencies)),
                "is_degenerate": False,
                "degenerate_class": None,
                "best_static_choice": None
            })

            # 3. Evaluate each policy
            seed_policy_results = {}
            for pol in POLICIES_TO_EVAL:
                res = run_single_stream_eval(
                    scenario_name, pol, messages, timestamps, seed,
                    train_size=train_size, train_feats=train_feats, eval_feats=eval_feats
                )
                seed_policy_results[pol] = res

            # 4. Determine Best-Static dynamically for this seed stream
            static_candidates = ["Always-Skip", "Always-Zstd", "Always-Brotli", "Always-Gzip", "Always-LZ4"]
            best_static_name = min(static_candidates, key=lambda s: np.sum(seed_policy_results[s]["costs"]))
            seed_policy_results["Best-Static"] = {
                "costs": seed_policy_results[best_static_name]["costs"],
                "latencies": seed_policy_results[best_static_name]["latencies"],
                "is_degenerate": False,
                "degenerate_class": None,
                "best_static_choice": best_static_name
            }

            # 5. Record results for all policies
            for pol, res in seed_policy_results.items():
                costs = res["costs"]
                latencies = res["latencies"]
                mean_cost = float(np.mean(costs))
                total_cost = float(np.sum(costs))
                cost_gap = mean_cost - oracle_mean_cost
                lag = calculate_adaptation_lag(costs, oracle_costs, eval_change_point, window=100)

                raw_records.append({
                    "scenario": scenario_name,
                    "seed": seed,
                    "policy": pol,
                    "mean_cost": mean_cost,
                    "total_cost": total_cost,
                    "oracle_cost": oracle_mean_cost,
                    "cost_gap": cost_gap,
                    "adaptation_lag": lag,
                    "p50_latency_us": float(np.percentile(latencies, 50)),
                    "p95_latency_us": float(np.percentile(latencies, 95)),
                    "p99_latency_us": float(np.percentile(latencies, 99)),
                    "mean_latency_us": float(np.mean(latencies)),
                    "is_degenerate": res.get("is_degenerate", False),
                    "degenerate_class": res.get("degenerate_class", None),
                    "best_static_choice": res.get("best_static_choice", None)
                })

            elapsed_seed = time.time() - seed_start_time
            print(f"  [Seed {seed_idx:02d}/{len(SEEDS)}: {seed:4d}] Done in {elapsed_seed:.2f}s | Oracle: {oracle_mean_cost:.2f} | BestStatic: {best_static_name} ({np.mean(seed_policy_results['Best-Static']['costs']):.2f}) | LinUCB: {np.mean(seed_policy_results['LinUCB (deadline-aware)']['costs']):.2f}", flush=True)

        print(f"Scenario {scenario_name} finished in {time.time() - scen_start_time:.2f}s", flush=True)
        # Flush intermediate CSV
        pd.DataFrame(raw_records).to_csv(raw_csv_path, index=False)

    total_elapsed = time.time() - start_total_time
    print(f"\nAll 160 streams completed in {total_elapsed:.2f}s ({total_elapsed/60.0:.2f} min). Raw results saved to {raw_csv_path}.", flush=True)

    # 6. Statistical Analysis and Summary Generation
    generate_summary_markdown(raw_records, summary_md_path)

def generate_summary_markdown(raw_records: list[dict], summary_md_path: str):
    df_raw = pd.DataFrame(raw_records)
    all_scenarios = [s[0] for s in SCENARIOS]

    lines = []
    lines.append("# Drift Adaptation Benchmark Evaluation (v2 Multi-Seed)")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("- **Independent Stream Realizations**: 20 seeds evaluated per scenario on fresh streams with distinct Poisson arrival times and independent sampling.")
    lines.append(r"- **Stream Length**: $\ge 5,000$ messages per stream (5,000 for synthetic scenarios 1-6; 6,000 for natural-drift scenarios 7-8).")
    lines.append("- **Natural Drift Streams**: Included two real-data streams (100% real prepared data, 0% synthetic):")
    lines.append("  - **`drift_7_natural_sequential`**: DailyDialog $\\rightarrow$ Sentiment140 $\\rightarrow$ SMS-Spam.")
    lines.append("  - **`drift_8_natural_switching`**: Interleaved switching between DailyDialog, Sentiment140, and SMS-Spam every 1,000 messages.")
    lines.append("- **Statistical Testing**: Paired Wilcoxon signed-rank tests with Holm-Bonferroni step-down correction and effect size metrics (Rank-Biserial $r$ and Cohen's $d_z$) across $N=20$ independent streams per scenario.")
    lines.append("")

    # Overall LinUCB vs Always-Zstd comparison
    lines.append("### Honest Assessment: LinUCB vs Always-Zstd")
    lines.append("")
    lines.append("Here is the honest, un-tuned comparison of **LinUCB (deadline-aware)** against **Always-Zstd** across all 8 drift scenarios:")
    lines.append("")

    zstd_vs_linucb_rows = []
    linucb_wins = 0
    zstd_wins = 0
    ties = 0

    for sc in all_scenarios:
        df_sc = df_raw[df_raw["scenario"] == sc]
        seeds = sorted(df_sc["seed"].unique())
        linucb_costs = [df_sc[(df_sc["seed"] == s) & (df_sc["policy"] == "LinUCB (deadline-aware)")]["mean_cost"].values[0] for s in seeds]
        zstd_costs = [df_sc[(df_sc["seed"] == s) & (df_sc["policy"] == "Always-Zstd")]["mean_cost"].values[0] for s in seeds]

        diff = np.array(linucb_costs) - np.array(zstd_costs)
        mean_diff = float(np.mean(diff))

        # Wilcoxon
        try:
            res = stats.wilcoxon(diff, zero_method='pratt')
            p_val = float(res.pvalue)
        except Exception:
            p_val = 1.0

        if p_val < 0.05:
            if mean_diff < 0:
                outcome = "**LinUCB Wins** (p < 0.05)"
                linucb_wins += 1
            else:
                outcome = "**Always-Zstd Wins** (p < 0.05)"
                zstd_wins += 1
        else:
            outcome = "Tied / No Sig. Diff."
            ties += 1

        zstd_vs_linucb_rows.append({
            "scenario": sc,
            "linucb_mean": np.mean(linucb_costs),
            "zstd_mean": np.mean(zstd_costs),
            "mean_diff": mean_diff,
            "pct_diff": (mean_diff / np.mean(zstd_costs)) * 100.0,
            "p_val": p_val,
            "outcome": outcome
        })

    lines.append(f"**Overall Record**: LinUCB beats Always-Zstd in **{linucb_wins} out of {len(all_scenarios)}** scenarios (Always-Zstd beats LinUCB in **{zstd_wins}**, and **{ties}** are statistical ties).")
    lines.append("")
    lines.append("| Scenario | LinUCB Mean Cost | Always-Zstd Mean Cost | Diff (LinUCB - Zstd) | % Cost Diff | Wilcoxon p (raw) | Verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in zstd_vs_linucb_rows:
        lines.append(f"| `{r['scenario']}` | {r['linucb_mean']:.2f} | {r['zstd_mean']:.2f} | {r['mean_diff']:+.2f} | {r['pct_diff']:+.2f}% | {r['p_val']:.4e} | {r['outcome']} |")
    lines.append("")

    # Detailed Per-Scenario Analysis
    lines.append("---")
    lines.append("## Detailed Scenario Breakdowns")
    lines.append("")

    for sc in all_scenarios:
        df_sc = df_raw[df_raw["scenario"] == sc]
        seeds = sorted(df_sc["seed"].unique())
        oracle_costs = [df_sc[(df_sc["seed"] == s) & (df_sc["policy"] == "Oracle")]["mean_cost"].values[0] for s in seeds]
        oracle_mean = np.mean(oracle_costs)

        policies = sorted(df_sc["policy"].unique())
        # Put Oracle first, then Best-Static, then LinUCB, then others
        priority = ["Oracle", "Best-Static", "LinUCB (deadline-aware)", "LinUCB (no deadline)", "Heuristic (deadline-aware)", "Heuristic (no deadline)", "Always-Zstd", "Always-LZ4", "Always-Brotli", "Always-Gzip", "Always-Skip"]
        policies = sorted(policies, key=lambda p: priority.index(p) if p in priority else 99)

        lines.append(f"### Scenario: `{sc}`")
        lines.append("")
        lines.append("#### Policy Cost & Adaptation Summary (20 Seeds)")
        lines.append("")
        lines.append("| Policy | Mean Cost (95% Bootstrap CI) | Gap to Oracle | % Gap to Oracle | Adaptation Lag (Mean / Conv Rate) | P95 Latency (us) | P99 Latency (us) | Notes |")
        lines.append("|---|---|---|---|---|---|---|---|")

        policy_costs_dict = {}

        for pol in policies:
            df_pol = df_sc[df_sc["policy"] == pol]
            mean_costs = df_pol["mean_cost"].tolist()
            policy_costs_dict[pol] = mean_costs

            m_cost, ci_low, ci_high = compute_bootstrap_ci(mean_costs)
            gap = m_cost - oracle_mean
            gap_pct = (gap / oracle_mean) * 100.0 if oracle_mean > 0 else 0.0

            lags = df_pol["adaptation_lag"].tolist()
            valid_lags = [l for l in lags if l >= 0]
            conv_rate = f"{len(valid_lags)}/20"
            if len(valid_lags) > 0:
                lag_str = f"{np.mean(valid_lags):.1f} ({conv_rate})"
            else:
                lag_str = f"DNC ({conv_rate})"

            p95_lat = df_pol["p95_latency_us"].mean()
            p99_lat = df_pol["p99_latency_us"].mean()

            is_degen = df_pol["is_degenerate"].any()
            notes = "Degenerate (Single Class)" if is_degen else "-"
            if pol == "Best-Static":
                best_choices = df_pol["best_static_choice"].value_counts().to_dict()
                choice_str = ", ".join([f"{k}:{v}" for k, v in best_choices.items()])
                notes = f"Chosen: {choice_str}"

            lines.append(f"| {pol} | {m_cost:.2f} [{ci_low:.2f}, {ci_high:.2f}] | {gap:+.2f} | {gap_pct:+.1f}% | {lag_str} | {p95_lat:.1f} | {p99_lat:.1f} | {notes} |")

        lines.append("")

        # Paired Wilcoxon Tests Table for this scenario
        lines.append(f"#### Paired Wilcoxon Signed-Rank Tests (LinUCB deadline-aware vs All Policies, N=20 Seeds)")
        lines.append("")
        lines.append("| Comparison | Mean Diff (LinUCB - Pol) | Wilcoxon W | p-value (raw) | p-value (Holm) | Rank-Biserial r | Cohen's d_z | Winner |")
        lines.append("|---|---|---|---|---|---|---|---|")

        # Compare LinUCB (deadline-aware) against all other policies except Oracle
        comp_targets = {k: v for k, v in policy_costs_dict.items() if k not in ["Oracle", "LinUCB (deadline-aware)"]}
        linucb_costs = policy_costs_dict["LinUCB (deadline-aware)"]
        w_results = paired_wilcoxon_tests(linucb_costs, comp_targets)

        for pol, res in w_results.items():
            lines.append(f"| LinUCB vs {pol} | {res['mean_diff']:+.2f} | {res['statistic']:.1f} | {res['p_raw']:.4e} | {res['p_holm']:.4e} | {res['rank_biserial']:+.3f} | {res['cohens_dz']:+.3f} | {res['winner']} |")

        lines.append("")
        lines.append("---")
        lines.append("")

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Summary markdown written to {summary_md_path}", flush=True)

if __name__ == "__main__":
    run_all_drift_benchmarks()
