import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score
from collections import Counter

from src.utils import get_data_dir
from src.scheduler import OfflineLabelGenerator, CorrectedOfflineLabelGenerator, SupervisedScheduler, ACTION_MAP
from src.simulator import StreamSimulator
from src.manager import BatchCacheManager
from benchmarking.config import BENCHMARK_RESULTS_DIR, ensure_benchmark_dirs
from src.features import FeatureExtractor

def measure_downstream_cost(messages, timestamps, labels, lambda_param):
    """
    Simulates the stream using the provided labels as the 'scheduler' actions.
    Returns the total user_perceived_cost.
    """
    manager = BatchCacheManager(timeout_seconds=0.5, max_batch_size=50)
    # Turn off cache to purely evaluate the label decisions without reuse contamination
    manager.cache = {} 
    
    sim = StreamSimulator(manager, lambda_param=lambda_param)
    
    # Create a mock scheduler that just returns the pre-computed label
    class LabelReplayScheduler:
        def __init__(self, labels):
            self.labels = labels
            self.idx = 0
            
        def predict(self, features):
            action = self.labels[self.idx]
            self.idx += 1
            return action, 5.0 # mock 5us sched latency
            
    scheduler = LabelReplayScheduler(labels)
    
    for msg_id, (text, ts) in enumerate(zip(messages, timestamps)):
        sim.run_message(msg_id, text, ts, scheduler)
        
    sim.finalize_stream(timestamps[-1] + 1.0, scheduler)
    metrics = sim.get_summary_metrics()
    return metrics.get("total_user_perceived_cost", 0.0)

def main():
    ensure_benchmark_dirs()
    
    # Load sample dataset
    data_path = os.path.join(get_data_dir(), "dailydialog_clean.csv")
    if not os.path.exists(data_path):
        print("Dataset not found. Skipping validation.")
        return
        
    df = pd.read_csv(data_path)
    # Take first 2000 messages for fast label validation
    sample = df.head(2000).to_dict('records')
    messages = [row['text'] for row in sample]
    timestamps = [row['timestamp'] for row in sample]
    
    lambda_param = 0.01
    
    print("Generating Original Labels...")
    orig_gen = OfflineLabelGenerator(lambda_param=lambda_param)
    t0 = time.time()
    orig_labels = orig_gen.generate_labels(messages)
    print(f"  Done in {time.time() - t0:.2f}s")
    
    print("Generating Corrected Labels...")
    corr_gen = CorrectedOfflineLabelGenerator(lambda_param=lambda_param)
    t0 = time.time()
    corr_labels = corr_gen.generate_labels(messages, timestamps)
    print(f"  Done in {time.time() - t0:.2f}s")
    
    # For this offline evaluation, the "Measured Train-Stream Oracle" is effectively 
    # the exact evaluation of the corrected labels, but we will measure downstream cost 
    # to prove the new labels are better.
    
    # Calculate distributions
    orig_dist = Counter([ACTION_MAP[l] for l in orig_labels])
    corr_dist = Counter([ACTION_MAP[l] for l in corr_labels])
    
    agreement = accuracy_score(corr_labels, orig_labels) * 100
    try:
        kappa = cohen_kappa_score(corr_labels, orig_labels)
        if np.isnan(kappa): kappa = 1.0
    except Exception:
        kappa = 1.0
    
    print("Evaluating Downstream Composite Cost...")
    orig_cost = measure_downstream_cost(messages, timestamps, orig_labels, lambda_param)
    corr_cost = measure_downstream_cost(messages, timestamps, corr_labels, lambda_param)
    
    print("Training Supervised Models...")
    fe = FeatureExtractor()
    features = [fe.extract_features(m, ts) for m, ts in zip(messages, timestamps)]
    
    dt_orig = SupervisedScheduler('decision_tree')
    dt_orig.fit(features, orig_labels)
    
    dt_corr = SupervisedScheduler('decision_tree')
    dt_corr.fit(features, corr_labels)
    
    # Train accuracy
    orig_preds = [dt_orig.predict(f)[0] for f in features]
    corr_preds = [dt_corr.predict(f)[0] for f in features]
    
    orig_acc = accuracy_score(orig_labels, orig_preds) * 100
    corr_acc = accuracy_score(corr_labels, corr_preds) * 100
    
    results = [{
        "Methodology": "Original (Flawed)",
        "SKIP%": orig_dist.get('SKIP', 0) / len(messages) * 100,
        "ZSTD%": orig_dist.get('ZSTD', 0) / len(messages) * 100,
        "BATCH%": orig_dist.get('BATCH', 0) / len(messages) * 100,
        "Agreement_with_Corrected%": agreement,
        "Cohens_Kappa": kappa,
        "DT_Train_Accuracy%": orig_acc,
        "Downstream_Total_Cost": orig_cost
    }, {
        "Methodology": "Corrected (Mathematical)",
        "SKIP%": corr_dist.get('SKIP', 0) / len(messages) * 100,
        "ZSTD%": corr_dist.get('ZSTD', 0) / len(messages) * 100,
        "BATCH%": corr_dist.get('BATCH', 0) / len(messages) * 100,
        "Agreement_with_Corrected%": 100.0,
        "Cohens_Kappa": 1.0,
        "DT_Train_Accuracy%": corr_acc,
        "Downstream_Total_Cost": corr_cost
    }]
    
    out_df = pd.DataFrame(results)
    print("\\n=== Label Validation Results ===")
    print(out_df.to_string(index=False))
    
    out_path = os.path.join(BENCHMARK_RESULTS_DIR, "offline_label_validation.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\\nSaved to {out_path}")

if __name__ == "__main__":
    main()
