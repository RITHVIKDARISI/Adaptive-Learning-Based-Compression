import os
import pandas as pd
import numpy as np

from benchmarking.config import BENCHMARK_RESULTS_DIR, ensure_benchmark_dirs
from src.utils import get_data_dir
from src.simulator import StreamSimulator
from src.manager import BatchCacheManager
from src.scheduler import SupervisedScheduler, HeuristicScheduler, ACTION_MAP
from src.bandit import LinUCBBandit

def get_subset():
    path = os.path.join(get_data_dir(), "dailydialog_clean.csv")
    df = pd.read_csv(path).head(1000)
    return df["text"].tolist(), df["timestamp"].tolist()

def eval_system(name, messages, timestamps, scheduler, cache_cap, batch_to, batch_sz, mod_feat=None, enable_deadline=False, max_wait_us=50000.0):
    manager = BatchCacheManager(
        cache_capacity=cache_cap if cache_cap > 0 else 1,
        timeout_seconds=batch_to,
        max_batch_size=batch_sz
    )
    if cache_cap == 0:
        class DummyCache:
            def get(self, k): return None
            def put(self, k, v): pass
            def clear(self): pass
            def __len__(self): return 0
        manager.cache = DummyCache()

    sim = StreamSimulator(manager, lambda_param=0.01)

    # Configure deadline on scheduler if applicable
    if hasattr(scheduler, 'set_deadline'):
        scheduler.set_deadline(max_wait_us=max_wait_us, enable_deadline=enable_deadline)

    class WrappedScheduler:
        def __init__(self, sched, mod):
            self.sched = sched
            self.mod = mod

        def predict(self, f):
            if self.mod == "no_rep":
                f["repetition_score"] = 0.0
            if self.mod == "no_arr":
                f["arrival_rate"] = 0.0
            if self.mod == "char_only":
                c = f["char_len"]
                for k in list(f.keys()):
                    if k not in ["queue_len", "batch_max_size", "batch_timeout_seconds"]:
                        f[k] = 0.0
                f["char_len"] = c
            if hasattr(self.sched, 'predict'):
                return self.sched.predict(f)
            return self.sched(f)

        def update(self, a, f, r):
            if hasattr(self.sched, 'update'):
                self.sched.update(a, f, r)

    w_sched = WrappedScheduler(scheduler, mod_feat)

    for i, (m, ts) in enumerate(zip(messages, timestamps)):
        sim.run_message(i, m, ts, w_sched)
    sim.finalize_stream(timestamps[-1] + 1.0, w_sched)

    metrics = sim.get_summary_metrics()
    metrics["Variant"] = name
    metrics["Deadline_Mask"] = "ON" if enable_deadline else "OFF"
    metrics["Max_Wait_us"] = max_wait_us if enable_deadline else None
    metrics["Mean_Cost"] = metrics["total_user_perceived_cost"] / max(metrics["total_messages"], 1)
    metrics["P95_Latency_us"] = metrics["p95_e2e_latency_us"]
    metrics["P99_Latency_us"] = metrics["p99_e2e_latency_us"]

    # Cache usage
    metrics["cache_evictions"] = manager.cache_evictions
    metrics["cache_items"] = len(manager.cache)

    cache_mem_bytes = 0
    if hasattr(manager.cache, 'cache'):
        for k, v in manager.cache.cache.items():
            cache_mem_bytes += len(k.encode('utf-8')) + len(v[0]) + len(v[1].encode('utf-8'))
    metrics["cache_memory_bytes"] = cache_mem_bytes

    return metrics

def run_ablation_study():
    out_dir = os.path.join(BENCHMARK_RESULTS_DIR, "v2", "baselines")
    os.makedirs(out_dir, exist_ok=True)

    messages, timestamps = get_subset()
    results = []

    for enable_deadline in [False, True]:
        dl_tag = " [Deadline ON]" if enable_deadline else " [Deadline OFF]"
        print(f"\n--- Running DailyDialog 1,000-message Ablation with Deadline {'ON' if enable_deadline else 'OFF'} ---")

        def add(name, sched, cache=1000, batch_to=0.5, batch_sz=50, mod_feat=None):
            print(f"Running {name}{dl_tag}...")
            results.append(eval_system(
                name=f"{name}{dl_tag}",
                messages=messages,
                timestamps=timestamps,
                scheduler=sched,
                cache_cap=cache,
                batch_to=batch_to,
                batch_sz=batch_sz,
                mod_feat=mod_feat,
                enable_deadline=enable_deadline
            ))

        base_sched = HeuristicScheduler(enable_deadline=enable_deadline)

        # Architectural ablations
        add("A. Scheduler only", base_sched, cache=0, batch_sz=1)
        add("B. Scheduler + cache", base_sched, cache=1000, batch_sz=1)
        add("C. Scheduler + batch", base_sched, cache=0, batch_sz=50)
        add("D. Full system", base_sched, cache=1000, batch_sz=50)

        def cache_only_sched(f): return 1, 5.0
        add("E. Cache-only", cache_only_sched, cache=1000, batch_sz=1)

        def batch_only_sched(f):
            if enable_deadline:
                from src.scheduler import estimate_batch_wait_us
                if estimate_batch_wait_us(f) > 50000.0:
                    return 1, 5.0
            return 5, 5.0
        add("F. Batch-only", batch_only_sched, cache=0, batch_sz=50)

        # Feature ablations
        add("G. No repetition", base_sched, mod_feat="no_rep")
        add("H. No arrival_rate", base_sched, mod_feat="no_arr")
        add("I. char_len only", base_sched, mod_feat="char_only")
        add("J. all features", base_sched)

        # Bandit variants
        lin_no_cost = LinUCBBandit(cost_aware_exploration=False, enable_deadline=enable_deadline)
        add("K. LinUCB (no cost-aware)", lin_no_cost)

        lin_cost = LinUCBBandit(cost_aware_exploration=True, enable_deadline=enable_deadline)
        add("L. LinUCB (cost-aware)", lin_cost)

    df = pd.DataFrame(results)
    out_path = os.path.join(out_dir, "ablation_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[Saved] DailyDialog Ablation saved to {out_path}")

    # Cache ablation
    print("\nRunning Cache Capacity Ablation...")
    c_res = []
    base_sched_cache = HeuristicScheduler(enable_deadline=True)
    for cap in [0, 10, 100, 1000, 10000]:
        m = eval_system(f"Cap_{cap}", messages, timestamps, base_sched_cache, cache_cap=cap, batch_to=0.5, batch_sz=1, enable_deadline=True)
        m["Capacity"] = cap
        c_res.append(m)
    c_df = pd.DataFrame(c_res)
    c_path = os.path.join(out_dir, "cache_ablation.csv")
    c_df.to_csv(c_path, index=False)
    print(f"[Saved] Cache Ablation saved to {c_path}")

    # Batch ablation
    print("\nRunning Batch Parameter Ablation...")
    b_res = []
    base_sched_batch = HeuristicScheduler(enable_deadline=True)
    for sz in [2, 4, 8, 16, 32]:
        for to in [0.005, 0.010, 0.050, 0.100, 0.500]:
            m = eval_system(f"Sz{sz}_To{to}", messages, timestamps, base_sched_batch, cache_cap=0, batch_to=to, batch_sz=sz, enable_deadline=True)
            m["BatchSize"] = sz
            m["Timeout"] = to
            b_res.append(m)
    b_df = pd.DataFrame(b_res)
    b_path = os.path.join(out_dir, "batch_ablation.csv")
    b_df.to_csv(b_path, index=False)
    print(f"[Saved] Batch Ablation saved to {b_path}")
    print("Done DailyDialog Ablation Study.")

if __name__ == "__main__":
    run_ablation_study()
