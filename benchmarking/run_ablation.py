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

def eval_system(name, messages, timestamps, scheduler, cache_cap, batch_to, batch_sz, mod_feat=None):
    manager = BatchCacheManager(timeout_seconds=batch_to, max_batch_size=batch_sz)
    manager.cache_capacity = cache_cap
    if cache_cap == 0:
        # Mock LRUCache to completely bypass
        class DummyCache:
            def get(self, k): return None
            def put(self, k, v): pass
            def clear(self): pass
            def __len__(self): return 0
        manager.cache = DummyCache()
        
    sim = StreamSimulator(manager, lambda_param=0.01)
    
    class WrappedScheduler:
        def __init__(self, sched, mod):
            self.sched = sched
            self.mod = mod
        def predict(self, f):
            if self.mod == "no_rep": f["repetition_score"] = 0.0
            if self.mod == "no_arr": f["arrival_rate"] = 0.0
            if self.mod == "char_only":
                c = f["char_len"]
                for k in f.keys(): f[k] = 0.0
                f["char_len"] = c
            if hasattr(self.sched, 'predict'): return self.sched.predict(f)
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
    
    # Also add cache usage
    metrics["cache_evictions"] = manager.cache_evictions
    metrics["cache_items"] = len(manager.cache)
    
    # Cache memory usage (approximate bytes of cached texts and compressed bytes)
    cache_mem_bytes = 0
    if hasattr(manager.cache, 'cache'):
        for k, v in manager.cache.cache.items():
            cache_mem_bytes += len(k.encode('utf-8')) + len(v[0]) + len(v[1].encode('utf-8'))
    metrics["cache_memory_bytes"] = cache_mem_bytes
    
    return metrics

def run_ablation_study():
    ensure_benchmark_dirs()
    messages, timestamps = get_subset()
    
    results = []
    
    def add(name, sched, cache=1000, batch_to=0.5, batch_sz=50, mod_feat=None):
        print(f"Running {name}...")
        results.append(eval_system(name, messages, timestamps, sched, cache, batch_to, batch_sz, mod_feat))
        
    dt = SupervisedScheduler('decision_tree')
    # Use random labels just for ablation throughput structure; in real benchmark we train
    # For now we use Heuristic as a strong base for architectural ablation
    base_sched = HeuristicScheduler()
    
    add("A. Scheduler only", base_sched, cache=0, batch_sz=1)
    add("B. Scheduler + cache", base_sched, cache=1000, batch_sz=1)
    add("C. Scheduler + batch", base_sched, cache=0, batch_sz=50)
    add("D. Full system", base_sched, cache=1000, batch_sz=50)
    
    def cache_only_sched(f): return 1, 5.0 # Always compress, let cache intercept
    add("E. Cache-only", cache_only_sched, cache=1000, batch_sz=1)
    
    def batch_only_sched(f): return 5, 5.0 # Always batch
    add("F. Batch-only", batch_only_sched, cache=0, batch_sz=50)
    
    add("G. No repetition", base_sched, mod_feat="no_rep")
    add("H. No arrival_rate", base_sched, mod_feat="no_arr")
    add("I. char_len only", base_sched, mod_feat="char_only")
    add("J. all features", base_sched)
    
    lin_no_cost = LinUCBBandit(cost_aware_exploration=False)
    add("K. LinUCB (no cost-aware)", lin_no_cost)
    
    lin_cost = LinUCBBandit(cost_aware_exploration=True)
    add("L. LinUCB (cost-aware)", lin_cost)
    
    df = pd.DataFrame(results)
    path = os.path.join(BENCHMARK_RESULTS_DIR, "ablation_results.csv")
    df.to_csv(path, index=False)
    print(f"Saved {path}")
    
    # Cache Ablation
    print("\\nRunning Cache Ablation...")
    c_res = []
    for cap in [0, 10, 100, 1000, 10000]:
        m = eval_system(f"Cap_{cap}", messages, timestamps, base_sched, cache_cap=cap, batch_to=0.5, batch_sz=1)
        m["Capacity"] = cap
        c_res.append(m)
    pd.DataFrame(c_res).to_csv(os.path.join(BENCHMARK_RESULTS_DIR, "cache_ablation.csv"), index=False)
    
    # Batch Ablation
    print("\\nRunning Batch Ablation...")
    b_res = []
    for sz in [2, 4, 8, 16, 32]:
        for to in [0.005, 0.010, 0.050, 0.100, 0.500]:
            m = eval_system(f"Sz{sz}_To{to}", messages, timestamps, base_sched, cache_cap=0, batch_to=to, batch_sz=sz)
            m["BatchSize"] = sz
            m["Timeout"] = to
            b_res.append(m)
    pd.DataFrame(b_res).to_csv(os.path.join(BENCHMARK_RESULTS_DIR, "batch_ablation.csv"), index=False)
    print("Done Ablation.")

if __name__ == "__main__":
    run_ablation_study()
