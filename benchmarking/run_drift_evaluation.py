import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from benchmarking.config import BENCHMARK_RESULTS_DIR, ensure_benchmark_dirs, SEEDS
from src.utils import get_data_dir
from src.simulator import StreamSimulator
from src.manager import BatchCacheManager
from src.scheduler import SupervisedScheduler, HeuristicScheduler, CorrectedOfflineLabelGenerator
from src.bandit import LinUCBBandit
from benchmarking.reproducibility import set_deterministic_seed
from benchmarking.synthetic_streams import generate_all_streams

def load_drift_stream(name):
    path = os.path.join(get_data_dir(), "synthetic_robustness", f"{name}.csv")
    if not os.path.exists(path):
        print(f"Synthetic stream {name} not found, generating...")
        generate_all_streams()
    df = pd.read_csv(path)
    return df["text"].tolist(), df["timestamp"].tolist()

def eval_drift(name, sched_name, messages, timestamps, seed):
    set_deterministic_seed(seed)
    manager = BatchCacheManager()
    sim = StreamSimulator(manager, lambda_param=0.01)
    
    if sched_name == "Always-Skip":
        scheduler = lambda f: (0, 0.0)
    elif sched_name == "Always-Zstd":
        scheduler = lambda f: (1, 0.0)
    elif sched_name == "Heuristic":
        scheduler = HeuristicScheduler()
    elif sched_name == "Decision Tree":
        # Untrained DT for drift (acts as static baseline post-init)
        scheduler = SupervisedScheduler('decision_tree', random_state=seed)
        # Mock fit so it doesn't crash
        from src.features import FeatureExtractor
        scheduler.fit([FeatureExtractor().extract_features(m, ts) for m,ts in zip(messages[:10], timestamps[:10])], [0]*10)
    elif sched_name == "LinUCB":
        scheduler = LinUCBBandit(cost_aware_exploration=True)
    else:
        raise ValueError(sched_name)
        
    for i, (m, ts) in enumerate(zip(messages, timestamps)):
        sim.run_message(i, m, ts, scheduler)
    sim.finalize_stream(timestamps[-1] + 1.0, scheduler)
    
    costs = [s["user_perceived_cost"] for s in sim.stats]
    actions = [s["action"] for s in sim.stats]
    savings = [s["orig_size"] - s["comp_size"] for s in sim.stats]
    latencies = [s["e2e_lat_us"] for s in sim.stats]
    
    return costs, actions, savings, latencies

def calculate_adaptation_lag(cost_series, oracle_series, change_point, window=100):
    # Number of messages until rolling cost remains within 10% of new regime oracle
    # for 3 consecutive windows
    if change_point >= len(cost_series): return -1
    
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
    ensure_benchmark_dirs()
    scenarios = [
        ("drift_1_repL_uniqS", 5000),
        ("drift_2_uniqS_repL", 5000),
        ("drift_3_lowRate_burst", 5000),
        ("drift_4_nat_json", 5000),
        ("drift_5_lowDup_highDup", 5000),
        ("drift_6_alternating", 500) # Every 500
    ]
    
    schedulers = ["Always-Skip", "Always-Zstd", "Heuristic", "Decision Tree", "LinUCB"]
    active_seeds = [42, 123, 456, 789, 2026]
    
    all_results = []
    
    scenario_series = {}  # Store time series for plotting
    lag_results = {s[0]: {} for s in scenarios}
    
    for scenario, change_interval in scenarios:
        print(f"Running scenario: {scenario}")
        messages, timestamps = load_drift_stream(scenario)
        
        # Oracle baseline
        gen = CorrectedOfflineLabelGenerator(lambda_param=0.01)
        oracle_labels = gen.generate_labels(messages, timestamps)
        
        # We simulate the Oracle to get true costs
        manager = BatchCacheManager()
        sim_or = StreamSimulator(manager, lambda_param=0.01)
        
        class OracleScheduler:
            def __init__(self, labels): self.labels, self.idx = labels, 0
            def predict(self, f):
                a = self.labels[self.idx]; self.idx += 1; return a, 0.0
                
        or_sched = OracleScheduler(oracle_labels)
        for i, (m, ts) in enumerate(zip(messages, timestamps)):
            sim_or.run_message(i, m, ts, or_sched)
        sim_or.finalize_stream(timestamps[-1] + 1.0, or_sched)
        
        oracle_costs = [s["user_perceived_cost"] for s in sim_or.stats]
        oracle_savings = [s["orig_size"] - s["comp_size"] for s in sim_or.stats]
        oracle_latencies = [s["e2e_lat_us"] for s in sim_or.stats]
        oracle_actions = [s["action"] for s in sim_or.stats]
        
        for seed in active_seeds:
            seed_results = {}
            for sched in schedulers:
                c, a, s_v, l = eval_drift(scenario, sched, messages, timestamps, seed)
                seed_results[sched] = {"costs": c, "actions": a, "savings": s_v, "latencies": l}
                
            # Determine Best-Static dynamically for this stream and seed
            skip_cost = sum(seed_results["Always-Skip"]["costs"])
            zstd_cost = sum(seed_results["Always-Zstd"]["costs"])
            best_static_name = "Always-Skip" if skip_cost < zstd_cost else "Always-Zstd"
            seed_results["Best-Static"] = seed_results[best_static_name]
            
            # Add Oracle to seed_results
            seed_results["Oracle"] = {"costs": oracle_costs, "actions": oracle_actions, "savings": oracle_savings, "latencies": oracle_latencies}
            
            for sched, res in seed_results.items():
                costs = res["costs"]
                actions = res["actions"]
                savings = res["savings"]
                latencies = res["latencies"]
                
                df_temp = pd.DataFrame({"cost": costs, "oracle": oracle_costs, "action": actions, "saving": savings, "latency": latencies})
                df_temp["regret"] = df_temp["cost"] - df_temp["oracle"]
                
                rolling_cost = df_temp["cost"].rolling(100).mean().mean()
                rolling_saving = df_temp["saving"].rolling(100).mean().mean()
                rolling_latency = df_temp["latency"].rolling(100).mean().mean()
                rolling_regret = df_temp["regret"].rolling(100).mean().mean()
                
                cum_regret = df_temp["regret"].sum()
                
                # Adaptation lag calculation
                change_point = change_interval
                lag = calculate_adaptation_lag(costs, oracle_costs, change_point)
                
                if sched not in lag_results[scenario]:
                    lag_results[scenario][sched] = []
                if lag != -1:
                    lag_results[scenario][sched].append(lag)
                
                all_results.append({
                    "Scenario": scenario,
                    "Scheduler": sched,
                    "Seed": seed,
                    "Total_Cost": sum(costs),
                    "Rolling_Cost_Avg": rolling_cost,
                    "Rolling_Saving_Avg": rolling_saving,
                    "Rolling_Latency_Avg": rolling_latency,
                    "Rolling_Regret_Avg": rolling_regret,
                    "Cumulative_Regret": cum_regret,
                    "Adaptation_Lag": lag
                })
                
                # Save series for plotting (just from seed 42 to avoid clutter)
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
    out_path = os.path.join(BENCHMARK_RESULTS_DIR, "drift_results.csv")
    df_res.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    
    generate_plots(scenario_series, lag_results, df_res, scenarios)

def generate_plots(scenario_series, lag_results, df_res, scenarios):
    print("Generating plots...")
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    
    # 1. linucb_cumulative_regret.png
    plt.figure(figsize=(10, 6))
    series_d1 = scenario_series.get("drift_1", {})
    for sched, df_temp in series_d1.items():
        if sched == "Oracle": continue
        plt.plot(df_temp["regret"].cumsum(), label=sched)
    plt.axvline(x=5000, color='r', linestyle='--', label='Change Point')
    plt.title('Cumulative Regret over Time (Scenario 1: repL -> uniqS)')
    plt.xlabel('Message Index')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_RESULTS_DIR, "linucb_cumulative_regret.png"))
    plt.close()
    
    # 2. linucb_actions_over_time.png
    plt.figure(figsize=(12, 6))
    df_linucb = scenario_series.get("drift_6", {}).get("LinUCB")
    if df_linucb is not None:
        actions_dummies = pd.get_dummies(df_linucb["action"])
        actions_rolling = actions_dummies.rolling(100).mean()
        for col in actions_rolling.columns:
            plt.plot(actions_rolling[col], label=col)
        for i in range(500, 10000, 500):
            plt.axvline(x=i, color='gray', linestyle=':', alpha=0.5)
        plt.title('LinUCB Action Frequencies over Time (Scenario 6: Alternating)')
        plt.xlabel('Message Index')
        plt.ylabel('Action Probability (Rolling 100)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(BENCHMARK_RESULTS_DIR, "linucb_actions_over_time.png"))
    plt.close()
    
    # 3. distribution_shift_cost.png
    plt.figure(figsize=(10, 6))
    for sched, df_temp in series_d1.items():
        plt.plot(df_temp["cost"].rolling(100).mean(), label=sched, alpha=0.8)
    plt.axvline(x=5000, color='r', linestyle='--', label='Change Point')
    plt.title('Rolling Cost (Scenario 1)')
    plt.xlabel('Message Index')
    plt.ylabel('Rolling Cost (100 msgs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_RESULTS_DIR, "distribution_shift_cost.png"))
    plt.close()
    
    # 4. adaptation_lag.png
    lag_plot_data = []
    for sc_name, sched_lags in lag_results.items():
        for sched, lags in sched_lags.items():
            if sched == "Oracle" or sched == "Best-Static": continue
            valid_lags = [l for l in lags if l != -1]
            avg_lag = np.mean(valid_lags) if valid_lags else 5000  # Cap at 5000 if no adaptation
            lag_plot_data.append({"Scenario": sc_name.replace("drift_", ""), "Scheduler": sched, "Lag": avg_lag})
            
    df_lag = pd.DataFrame(lag_plot_data)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_lag, x="Scenario", y="Lag", hue="Scheduler")
    plt.title('Adaptation Lag by Scenario and Scheduler')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_RESULTS_DIR, "adaptation_lag.png"))
    plt.close()
    print("Plots saved.")

if __name__ == "__main__":
    run_drift_benchmarks()
