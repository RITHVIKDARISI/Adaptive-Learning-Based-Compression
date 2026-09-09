import os
import hashlib
import json
import pandas as pd
import numpy as np
from collections import Counter
import math

from src.utils import get_data_dir
from benchmarking.config import BENCHMARK_RESULTS_DIR, ensure_benchmark_dirs

def compute_entropy(text: str) -> float:
    if not text:
        return 0.0
    freqs = Counter(text)
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in freqs.values())

def file_sha256(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def audit_datasets():
    ensure_benchmark_dirs()
    data_dir = get_data_dir()
    
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    
    stats_list = []
    manifest = {}
    
    for f in files:
        filepath = os.path.join(data_dir, f)
        df = pd.read_csv(filepath)
        
        file_hash = file_sha256(filepath)
        
        row_count = len(df)
        unique_texts = df["text"].nunique()
        duplicate_rate = 1.0 - (unique_texts / max(row_count, 1))
        
        byte_lengths = df["text"].apply(lambda x: len(str(x).encode('utf-8')))
        
        avg_bytes = byte_lengths.mean()
        median_bytes = byte_lengths.median()
        p95_bytes = np.percentile(byte_lengths, 95)
        
        entropies = df["text"].apply(lambda x: compute_entropy(str(x)))
        mean_entropy = entropies.mean()
        
        # Near-duplicate heuristic (same length and entropy within 5%)
        # For simplicity in this audit, we estimate near-duplicates by length collisions
        # that are not exact duplicates.
        length_counts = byte_lengths.value_counts()
        potential_near_dupes = sum(count for count in length_counts if count > 1) - (row_count - unique_texts)
        near_dupe_freq = max(0, potential_near_dupes) / max(row_count, 1)
        
        # Timestamp distribution
        if "timestamp" in df.columns:
            dt = df["timestamp"].diff().dropna()
            mean_arrival_interval = dt.mean()
            arrival_rate_hz = 1.0 / mean_arrival_interval if mean_arrival_interval > 0 else 0
        else:
            arrival_rate_hz = 0
            
        # Determine provenance
        # If "source_type" isn't present, we must label UNKNOWN since it could be synthetic fallback
        if "source_type" in df.columns:
            sources = df["source_type"].unique()
            if len(sources) == 1 and sources[0] == "REAL":
                provenance = "REAL"
            elif len(sources) == 1 and sources[0] == "SYNTHETIC":
                provenance = "SYNTHETIC"
            else:
                provenance = "MIXED"
        else:
            provenance = "UNKNOWN"
            
        stats = {
            "dataset": f,
            "row_count": row_count,
            "unique_rows": unique_texts,
            "duplicate_rate": duplicate_rate,
            "average_bytes": avg_bytes,
            "median_bytes": median_bytes,
            "p95_bytes": p95_bytes,
            "mean_entropy": mean_entropy,
            "exact_duplicate_freq": duplicate_rate,
            "near_duplicate_freq": near_dupe_freq,
            "arrival_rate_hz": arrival_rate_hz,
            "provenance": provenance,
            "sha256": file_hash
        }
        stats_list.append(stats)
        
        manifest[f] = {
            "provenance": provenance,
            "sha256": file_hash,
            "row_count": row_count
        }
        
    df_stats = pd.DataFrame(stats_list)
    stats_path = os.path.join(BENCHMARK_RESULTS_DIR, "dataset_statistics.csv")
    df_stats.to_csv(stats_path, index=False)
    
    manifest_path = os.path.join(BENCHMARK_RESULTS_DIR, "dataset_manifest.json")
    with open(manifest_path, "w") as out_f:
        json.dump(manifest, out_f, indent=4)
        
    print(f"Audited {len(files)} datasets.")
    print(f"Saved {stats_path}")
    print(f"Saved {manifest_path}")

if __name__ == "__main__":
    audit_datasets()
