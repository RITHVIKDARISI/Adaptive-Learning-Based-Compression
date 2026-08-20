import os
import json
import time
import pandas as pd
import numpy as np
from src.utils import get_data_dir, log_environment_specs
from src.manager import BatchCacheManager
from src.features import FeatureExtractor
from src.bandit import LinUCBBandit
from src.simulator import StreamSimulator
from src.scheduler import OfflineLabelGenerator, SupervisedScheduler, HeuristicScheduler

def run_dataset_benchmark(dataset_name: str, clean_csv_path: str, lambdas: list[float]) -> dict:
    print(f"\n==================== Running Benchmark on {dataset_name} ====================")
    df = pd.read_csv(clean_csv_path)
    
    # Split into 50% train (for supervised training) and 50% test (for stream simulation)
    split_idx = len(df) // 2
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    results = {}
    
    # 1. Feature extraction for Train Set
    print("Extracting training features...")
    fe_train = FeatureExtractor()
    X_train = []
    # Feed messages sequentially to simulate features (repetition, arrival rate) properly
    for idx, row in train_df.iterrows():
        feats = fe_train.extract_features(row["text"], row["timestamp"])
        X_train.append(feats)
        
    # We will run simulation on the test set for all baselines and schedulers
    # Baselines that don't depend on lambda can be run once:
    print("Running static baselines...")
    static_baselines = {
        "Always-Skip": lambda f: (0, 0.0),
        "Always-Zstd": lambda f: (1, 0.0),
        "Always-Brotli": lambda f: (2, 0.0),
        "Always-Gzip": lambda f: (3, 0.0),
        "Always-LZ4": lambda f: (4, 0.0),
        "Fixed-Batch": lambda f: (5, 0.0),
        "Length-Threshold": lambda f: (0 if f["char_len"] < 25 else 1, 0.0),
        "Heuristic": HeuristicScheduler(skip_threshold=25)
    }
    
    for name, model in static_baselines.items():
        manager = BatchCacheManager()
        sim = StreamSimulator(manager)
        sim.reset()
        for _, row in test_df.iterrows():
            sim.run_message(row["msg_id"], row["text"], row["timestamp"], model)
        sim.finalize_stream(test_df.iloc[-1]["timestamp"], model)
        results[name] = {"static": sim.get_summary_metrics()}

    # Schedulers that depend on lambda:
    for lam in lambdas:
        print(f"\nEvaluating Lambda = {lam}...")
        lam_key = f"lambda_{lam}"
        results[lam_key] = {}
        
        # A. Generate Offline Optimal Labels for Train set
        label_gen = OfflineLabelGenerator(lambda_param=lam)
        y_train = label_gen.generate_labels(train_df["text"].tolist())
        
        # B. Train Supervised Schedulers
        supervised_models = {
            "Logistic-Regression": SupervisedScheduler('logistic_regression'),
            "Decision-Tree": SupervisedScheduler('decision_tree'),
            "Random-Forest": SupervisedScheduler('random_forest'),
            "Gradient-Boosting": SupervisedScheduler('gradient_boosting')
        }
        
        for m_name, m_wrapper in supervised_models.items():
            # Check if there are at least two classes to train
            if len(set(y_train)) < 2:
                # Fallback to default class if training is not possible
                default_class = y_train[0] if y_train else 0
                class DummyModel:
                    def predict(self, f): return default_class, 1.0
                model_to_use = DummyModel()
            else:
                m_wrapper.fit(X_train, y_train)
                model_to_use = m_wrapper
                
            # Run Test Simulation
            manager = BatchCacheManager()
            sim = StreamSimulator(manager, lambda_param=lam)
            sim.reset()
            for _, row in test_df.iterrows():
                sim.run_message(row["msg_id"], row["text"], row["timestamp"], model_to_use)
            sim.finalize_stream(test_df.iloc[-1]["timestamp"], model_to_use)
            
            results[lam_key][m_name] = sim.get_summary_metrics()
            
            # Export feature importances for tree models (Decision Tree is primary)
            if m_name == "Decision-Tree" and hasattr(model_to_use, 'get_feature_importances'):
                results[lam_key][m_name]["feature_importances"] = m_wrapper.get_feature_importances()

        # C. Run Online Contextual Bandit (LinUCB)
        # The bandit learns online on the test set stream
        bandit = LinUCBBandit(d=9, K=6, alpha=0.5, lambda_param=0.1)
        manager = BatchCacheManager()
        sim = StreamSimulator(manager, lambda_param=lam)
        sim.reset()
        
        for _, row in test_df.iterrows():
            sim.run_message(row["msg_id"], row["text"], row["timestamp"], bandit)
        sim.finalize_stream(test_df.iloc[-1]["timestamp"], bandit)
        
        results[lam_key]["LinUCB-Bandit"] = sim.get_summary_metrics()
        
    return results

def main():
    log_environment_specs()  # Log specs first
    
    # Path setup
    data_dir = get_data_dir()
    datasets = {
        "DailyDialog": os.path.join(data_dir, "dailydialog_clean.csv"),
        "Sentiment140": os.path.join(data_dir, "sentiment140_clean.csv"),
        "NUS-SMS": os.path.join(data_dir, "nussms_clean.csv")
    }
    
    # Check that datasets exist
    missing = [name for name, path in datasets.items() if not os.path.exists(path)]
    if missing:
        print(f"Missing prepared datasets: {missing}. Running data loader first...")
        from src.data_loader import prepare_all_datasets
        prepare_all_datasets()
        
    # We select 4 lambda values to represent compression-latency trade-offs:
    # 0.001: low latency cost (prefers Broti/Zstd/LZ4)
    # 0.01: moderate latency cost
    # 0.1: high latency cost
    # 1.0: extremely high latency cost (prefers SKIP/exact cache reuse)
    lambdas = [0.001, 0.01, 0.1, 1.0]
    
    all_results = {}
    for name, path in datasets.items():
        all_results[name] = run_dataset_benchmark(name, path, lambdas)
        
    # Save all results to data/experiment_results.json
    output_path = os.path.join(data_dir, "experiment_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\nSuccess! Benchmarking complete. Saved results to {output_path}")

if __name__ == "__main__":
    main()
