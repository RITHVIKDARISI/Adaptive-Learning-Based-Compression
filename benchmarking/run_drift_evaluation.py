import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from benchmarking.config import BENCHMARK_RESULTS_DIR, ensure_benchmark_dirs, SEEDS
from src.utils import get_data_dir
from src.simulator import StreamSimulator
from src.manager import BatchCacheManager
from src.scheduler import (
    SupervisedScheduler,
    HeuristicScheduler,
    OfflineLabelGenerator,
    SimulatorAlignedOracle,
    ACTION_MAP,
)
from src.bandit import LinUCBBandit
from src.features import FeatureExtractor
from benchmarking.reproducibility import set_deterministic_seed
from benchmarking.synthetic_streams import generate_all_streams

def load_drift_stream(name):
    path = os.path.join(get_data_dir(), "synthetic_robustness", f"{name}.csv")
    if not os.path.exists(path):
        print(f"Synthetic stream {name} not found, generating...", flush=True)
        generate_all_streams()
    df = pd.read_csv(path)
    return df["text"].tolist(), df["timestamp"].tolist()

def eval_drift(name, sched_name, messages, timestamps, seed, train_size=500, max_wait_us=50000.0, train_feats=None, eval_feats=None):
    set_deterministic_seed(seed)

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
        # Warm up on the 500-message training portion
        sim_train = StreamSimulator(BatchCacheManager(), lambda_param=0.01)
        for i, (m, ts) in enumerate(zip(train_messages, train_timestamps)):
            pf = train_feats[i] if train_feats is not None else None
            sim_train.run_message(i, m, ts, scheduler, precomputed_features=pf)
        sim_train.finalize_stream(train_timestamps[-1] + 0.1, scheduler)
    elif sched_name == "LinUCB (deadline-aware)":
        scheduler = LinUCBBandit(cost_aware_exploration=True, enable_deadline=True, max_wait_us=max_wait_us)
        # Warm up on the 500-message training portion
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
            if seed == 42:
                print(f"[Degenerate Policy] Scenario '{name}', Policy '{sched_name}': Single class ({degenerate_class}: {ACTION_MAP.get(degenerate_class, 'UNKNOWN')}) in training labels. Reported as degenerate.", flush=True)
            scheduler.is_degenerate = True
            scheduler.degenerate_class = degenerate_class
        else:
            scheduler.fit(train_feats, train_labels)
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")

    manager = BatchCacheManager()
    sim = StreamSimulator(manager, lambda_param=0.01)

    for i, (m, ts) in enumerate(zip(eval_messages, eval_timestamps)):
        pf = eval_feats[i] if eval_feats is not None else None
        sim.run_message(i, m, ts, scheduler, precomputed_features=pf)
    sim.finalize_stream(eval_timestamps[-1] + 1.0, scheduler)

    costs = [s["user_perceived_cost"] for s in sim.stats]
    actions = [s["action"] for s in sim.stats]
    savings = [s["orig_size"] - s["comp_size"] for s in sim.stats]
    latencies = [s["e2e_lat_us"] for s in sim.stats]

    return {
        "costs": costs,
        "actions": actions,
        "savings": savings,
        "latencies": latencies,
        "is_degenerate": is_degenerate,
        "degenerate_class": degenerate_class
    }

def calculate_adaptation_lag(cost_series, oracle_series, change_point, window=100):
    if change_point >= len(cost_series):
        return -1

    consecutive_success = 0
    for i in range(change_point, len(cost_series) - window + 1):
        rolling_cost = np.mean(cost_series[i:i+window])
        rolling_oracle = np.mean(oracle_series[i:i+window])

        if rolling_cost <= rolling_oracle * 1.10:
            consecutive_success += 1
            if consecutive_success >= 3:
                return i - change_point
        else:
            consecutive_success = 0

    return -1

def run_drift_benchmarks():
    out_dir = os.path.join(BENCHMARK_RESULTS_DIR, "v2", "baselines")
    os.makedirs(out_dir, exist_ok=True)

    scenarios = [
        ("drift_1_repL_uniqS", 5000),
        ("drift_2_uniqS_repL", 5000),
        ("drift_3_lowRate_burst", 5000),
        ("drift_4_nat_json", 5000),
        ("drift_5_lowDup_highDup", 5000),
        ("drift_6_alternating", 500)
    ]

    schedulers_to_run = [
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
        "Gradient Boosting"
    ]
    active_seeds = [42, 123, 456, 789, 2026]

    all_results = []
    scenario_series = {}
    lag_results = {s[0]: {} for s in scenarios}

    train_size = 500

    for scenario, change_interval in scenarios:
        print(f"\n=== Running scenario: {scenario} ===", flush=True)
        messages, timestamps = load_drift_stream(scenario)

        # Precompute base features once per scenario for high performance
        print(f"Precomputing base features for {len(messages)} messages...", flush=True)
        fe = FeatureExtractor()
        all_features = [fe.extract_features(m, ts) for m, ts in zip(messages, timestamps)]
        train_feats = all_features[:train_size]
        eval_feats = all_features[train_size:]

        eval_messages = messages[train_size:]
        eval_timestamps = timestamps[train_size:]

        # Oracle baseline via SimulatorAlignedOracle
        print(f"Computing SimulatorAlignedOracle labels and simulation...", flush=True)
        oracle_gen = SimulatorAlignedOracle(lambda_param=0.01)
        oracle_labels = oracle_gen.generate_labels(eval_messages, eval_timestamps)

        # Simulate Oracle on the evaluation stream
        manager_or = BatchCacheManager()
        sim_or = StreamSimulator(manager_or, lambda_param=0.01)

        class OracleScheduler:
            def __init__(self, labels):
                self.labels = labels
                self.idx = 0
            def predict(self, f):
                a = self.labels[self.idx]
                self.idx += 1
                return a, 0.0

        or_sched = OracleScheduler(oracle_labels)
        for i, (m, ts) in enumerate(zip(eval_messages, eval_timestamps)):
            sim_or.run_message(i, m, ts, or_sched, precomputed_features=eval_feats[i])
        sim_or.finalize_stream(eval_timestamps[-1] + 1.0, or_sched)

        oracle_costs = [s["user_perceived_cost"] for s in sim_or.stats]
        oracle_savings = [s["orig_size"] - s["comp_size"] for s in sim_or.stats]
        oracle_latencies = [s["e2e_lat_us"] for s in sim_or.stats]
        oracle_actions = [s["action"] for s in sim_or.stats]

        print(f"Evaluating {len(schedulers_to_run)} policies across {len(active_seeds)} seeds...", flush=True)
        for seed in active_seeds:
            seed_results = {}
            for sched in schedulers_to_run:
                res = eval_drift(
                    scenario, sched, messages, timestamps, seed,
                    train_size=train_size, train_feats=train_feats, eval_feats=eval_feats
                )
                seed_results[sched] = res

            # Determine Best-Static dynamically among SKIP, ZSTD, BROTLI, GZIP, LZ4 by total cost
            static_candidates = ["Always-Skip", "Always-Zstd", "Always-Brotli", "Always-Gzip", "Always-LZ4"]
            best_static_name = min(static_candidates, key=lambda s: sum(seed_results[s]["costs"]))
            seed_results["Best-Static"] = {
                "costs": seed_results[best_static_name]["costs"],
                "actions": seed_results[best_static_name]["actions"],
                "savings": seed_results[best_static_name]["savings"],
                "latencies": seed_results[best_static_name]["latencies"],
                "is_degenerate": False,
                "degenerate_class": None,
                "best_action": best_static_name
            }

            # Add Oracle to seed_results
            seed_results["Oracle"] = {
                "costs": oracle_costs,
                "actions": oracle_actions,
                "savings": oracle_savings,
                "latencies": oracle_latencies,
                "is_degenerate": False,
                "degenerate_class": None
            }

            for sched, res in seed_results.items():
                costs = res["costs"]
                actions = res["actions"]
                savings = res["savings"]
                latencies = res["latencies"]
                is_degen = res.get("is_degenerate", False)
                degen_cls = res.get("degenerate_class", None)

                df_temp = pd.DataFrame({
                    "cost": costs,
                    "oracle": oracle_costs,
                    "action": actions,
                    "saving": savings,
                    "latency": latencies
                })
                df_temp["regret"] = df_temp["cost"] - df_temp["oracle"]

                rolling_cost = df_temp["cost"].rolling(100).mean().mean()
                rolling_saving = df_temp["saving"].rolling(100).mean().mean()
                rolling_latency = df_temp["latency"].rolling(100).mean().mean()
                rolling_regret = df_temp["regret"].rolling(100).mean().mean()

                cum_regret = df_temp["regret"].sum()

                # Adaptation lag calculation
                eval_change_point = max(0, change_interval - train_size)
                lag = calculate_adaptation_lag(costs, oracle_costs, eval_change_point)

                if sched not in lag_results[scenario]:
                    lag_results[scenario][sched] = []
                if lag != -1:
                    lag_results[scenario][sched].append(lag)

                all_results.append({
                    "Scenario": scenario,
                    "Scheduler": sched,
                    "Seed": seed,
                    "Total_Cost": sum(costs),
                    "Mean_Cost": float(np.mean(costs)),
                    "P50_Latency_us": float(np.percentile(latencies, 50)),
                    "P95_Latency_us": float(np.percentile(latencies, 95)),
                    "P99_Latency_us": float(np.percentile(latencies, 99)),
                    "Mean_Latency_us": float(np.mean(latencies)),
                    "Rolling_Cost_Avg": rolling_cost,
                    "Rolling_Saving_Avg": rolling_saving,
                    "Rolling_Latency_Avg": rolling_latency,
                    "Rolling_Regret_Avg": rolling_regret,
                    "Cumulative_Regret": cum_regret,
                    "Adaptation_Lag": lag,
                    "Is_Degenerate": is_degen,
                    "Degenerate_Class": degen_cls,
                    "Best_Static_Choice": res.get("best_action", None)
                })

                # Save series for plotting from seed 42
                if seed == 42:
                    if scenario == "drift_6_alternating":
                        if "drift_6" not in scenario_series:
                            scenario_series["drift_6"] = {}
                        scenario_series["drift_6"][sched] = df_temp
                    if scenario == "drift_1_repL_uniqS":
                        if "drift_1" not in scenario_series:
                            scenario_series["drift_1"] = {}
                        scenario_series["drift_1"][sched] = df_temp

    df_res = pd.DataFrame(all_results)
    out_path = os.path.join(out_dir, "drift_results.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\n[Saved] Drift evaluation results saved to {out_path}", flush=True)

    generate_plots(scenario_series, lag_results, df_res, scenarios, out_dir=out_dir)
    return df_res

def generate_plots(scenario_series, lag_results, df_res, scenarios, out_dir):
    print("Generating baseline evaluation plots...", flush=True)
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    # 1. linucb_cumulative_regret.png
    plt.figure(figsize=(10, 6))
    series_d1 = scenario_series.get("drift_1", {})
    plotted = ["Always-Skip", "Always-Zstd", "Best-Static", "Heuristic (no deadline)", "Heuristic (deadline-aware)", "LinUCB (deadline-aware)", "Decision Tree", "Random Forest"]
    for sched in plotted:
        df_temp = series_d1.get(sched)
        if df_temp is not None:
            plt.plot(df_temp["regret"].cumsum(), label=sched)
    plt.axvline(x=4500, color='r', linestyle='--', label='Change Point (after 500-msg train)')
    plt.title('Cumulative Regret over Time (Scenario 1: repL -> uniqS)')
    plt.xlabel('Evaluation Message Index')
    plt.ylabel('Cumulative Regret')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "linucb_cumulative_regret.png"))
    plt.close()

    # 2. linucb_actions_over_time.png
    plt.figure(figsize=(12, 6))
    df_linucb = scenario_series.get("drift_6", {}).get("LinUCB (deadline-aware)")
    if df_linucb is not None:
        actions_dummies = pd.get_dummies(df_linucb["action"])
        actions_rolling = actions_dummies.rolling(100).mean()
        for col in actions_rolling.columns:
            plt.plot(actions_rolling[col], label=f"Action: {col}")
        for i in range(500, 9500, 500):
            plt.axvline(x=i, color='gray', linestyle=':', alpha=0.5)
        plt.title('LinUCB (Deadline-Aware) Action Frequencies over Time (Scenario 6: Alternating)')
        plt.xlabel('Evaluation Message Index')
        plt.ylabel('Action Probability (Rolling 100)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "linucb_actions_over_time.png"))
    plt.close()

    # 3. distribution_shift_cost.png
    plt.figure(figsize=(10, 6))
    for sched in ["Best-Static", "Heuristic (no deadline)", "Heuristic (deadline-aware)", "LinUCB (no deadline)", "LinUCB (deadline-aware)"]:
        df_temp = series_d1.get(sched)
        if df_temp is not None:
            plt.plot(df_temp["cost"].rolling(100).mean(), label=sched, alpha=0.8)
    plt.axvline(x=4500, color='r', linestyle='--', label='Change Point')
    plt.title('Rolling Cost under Distribution Shift (Scenario 1)')
    plt.xlabel('Evaluation Message Index')
    plt.ylabel('Rolling Cost (100 msgs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "distribution_shift_cost.png"))
    plt.close()

    # 4. adaptation_lag.png
    lag_plot_data = []
    for sc_name, sched_lags in lag_results.items():
        for sched, lags in sched_lags.items():
            if sched in ["Oracle", "Best-Static", "Always-Skip", "Always-Zstd", "Always-Brotli", "Always-Gzip", "Always-LZ4"]:
                continue
            valid_lags = [l for l in lags if l != -1]
            avg_lag = np.mean(valid_lags) if valid_lags else 4500
            lag_plot_data.append({"Scenario": sc_name.replace("drift_", ""), "Scheduler": sched, "Lag": avg_lag})

    if lag_plot_data:
        df_lag = pd.DataFrame(lag_plot_data)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_lag, x="Scenario", y="Lag", hue="Scheduler")
        plt.title('Adaptation Lag by Scenario and Scheduler')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "adaptation_lag.png"))
        plt.close()
    print("Baseline evaluation plots generated and saved.", flush=True)

if __name__ == "__main__":
    run_drift_benchmarks()
