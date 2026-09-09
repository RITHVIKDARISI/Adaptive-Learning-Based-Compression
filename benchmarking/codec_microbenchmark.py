import os
import time
import json
import numpy as np
import pandas as pd
import string
import random
import matplotlib.pyplot as plt

from benchmarking.config import BENCHMARK_RESULTS_DIR, ensure_benchmark_dirs
from benchmarking.metrics import MessageMetrics
from benchmarking.instrumentation import measure_time_us, verify_lossless_reconstruction
from src.engine import CompressionEngine

# Force consistent visual styles
plt.style.use('ggplot')

CODECS = ["SKIP", "ZSTD", "BROTLI", "GZIP", "LZ4", "ZLIB", "BZ2", "LZMA"]

def generate_synthetic_diagnostics() -> dict[str, list[str]]:
    random.seed(42)
    def rand_str(l, chars): return ''.join(random.choices(chars, k=l))
    
    # 1. Random high-entropy ASCII
    high_entropy = [rand_str(random.randint(10, 500), string.ascii_letters + string.digits + string.punctuation) for _ in range(100)]
    
    # 2. Highly repetitive ASCII
    repetitive = [(rand_str(1, string.ascii_letters) * random.randint(10, 500)) for _ in range(100)]
    
    # 3. Natural English (mocked via simple words)
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "hello", "world", "this", "is", "a", "test"]
    natural = [" ".join(random.choices(words, k=random.randint(5, 50))) for _ in range(100)]
    
    # 4. Emoji-heavy UTF-8
    emojis = ["😀", "🚀", "🔥", "❤️", "👍", "🙌", "😂", "✨", "🎉", "💡"]
    emoji_heavy = ["".join(random.choices(emojis, k=random.randint(5, 50))) for _ in range(100)]
    
    # 5. JSON-like text
    json_like = [f'{{"id": {random.randint(1,1000)}, "status": "ok", "value": "{rand_str(5, string.ascii_letters)}"}}' for _ in range(100)]
    
    # 6. Log messages
    logs = [f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] INFO: User {random.randint(1,100)} logged in successfully.' for _ in range(100)]
    
    # 7. Repeated templates
    templates = [f"Order {random.randint(1000, 9999)} has been shipped." for _ in range(100)]
    
    # 8. Unique short strings
    short = [rand_str(random.randint(2, 10), string.ascii_lowercase) for _ in range(100)]
    
    return {
        "HighEntropy": high_entropy,
        "Repetitive": repetitive,
        "Natural": natural,
        "EmojiHeavy": emoji_heavy,
        "JSON": json_like,
        "Logs": logs,
        "Templates": templates,
        "Short": short
    }

def get_length_bucket(byte_len: int) -> str:
    if byte_len <= 15: return "0-15"
    if byte_len <= 31: return "16-31"
    if byte_len <= 63: return "32-63"
    if byte_len <= 127: return "64-127"
    if byte_len <= 255: return "128-255"
    if byte_len <= 511: return "256-511"
    if byte_len <= 1023: return "512-1023"
    return "1024+"

def run_microbenchmarks():
    ensure_benchmark_dirs()
    engine = CompressionEngine()
    
    datasets = generate_synthetic_diagnostics()
    # If standard datasets exist, load a sample from them too
    from src.utils import get_data_dir
    data_dir = get_data_dir()
    for ds_name, file_name in [("DailyDialog", "dailydialog_clean.csv"), ("Sentiment140", "sentiment140_clean.csv"), ("NUS-SMS", "nussms_clean.csv")]:
        path = os.path.join(data_dir, file_name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            sample = df["text"].dropna().sample(n=min(500, len(df)), random_state=42).tolist()
            datasets[ds_name] = sample
            
    results = []
    
    print("Starting codec microbenchmarks...")
    for ds_name, messages in datasets.items():
        print(f"  Benchmarking {ds_name} ({len(messages)} samples)...")
        for msg in messages:
            raw_bytes = msg.encode('utf-8')
            byte_len = len(raw_bytes)
            bucket = get_length_bucket(byte_len)
            
            for codec in CODECS:
                # Warmup
                for _ in range(5):
                    c_tmp, _ = engine.compress(msg, codec)
                    engine.decompress(c_tmp, codec)
                    
                # Measurements
                comp_lats = []
                decomp_lats = []
                c_bytes = b""
                for _ in range(30):
                    with measure_time_us() as t_comp:
                        c_bytes, _ = engine.compress(msg, codec)
                    comp_lats.append(t_comp.elapsed_us)
                    
                    with measure_time_us() as t_decomp:
                        d_text, _ = engine.decompress(c_bytes, codec)
                    decomp_lats.append(t_decomp.elapsed_us)
                    
                # Lossless check
                d_text, _ = engine.decompress(c_bytes, codec, raw_bytes=True)
                lossless = verify_lossless_reconstruction(msg, d_text)
                
                # Metrics wrapper
                m = MessageMetrics(
                    msg_id=0, original_bytes=byte_len, encoded_bytes=len(c_bytes),
                    codec=codec
                )
                
                results.append({
                    "dataset": ds_name,
                    "codec": codec,
                    "length_bucket": bucket,
                    "original_bytes": byte_len,
                    "compressed_bytes": len(c_bytes),
                    "compression_factor": m.compression_factor,
                    "compressed_fraction": m.compressed_fraction,
                    "space_saving_pct": m.space_saving_pct,
                    "comp_lat_p50": np.percentile(comp_lats, 50),
                    "comp_lat_p95": np.percentile(comp_lats, 95),
                    "comp_lat_p99": np.percentile(comp_lats, 99),
                    "decomp_lat_p50": np.percentile(decomp_lats, 50),
                    "decomp_lat_p95": np.percentile(decomp_lats, 95),
                    "decomp_lat_p99": np.percentile(decomp_lats, 99),
                    "lossless_success": lossless
                })
                
    df = pd.DataFrame(results)
    raw_path = os.path.join(BENCHMARK_RESULTS_DIR, "codec_microbenchmarks.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved raw microbenchmark data to {raw_path}")
    
    # Aggregation by length
    agg_funcs = {
        "original_bytes": "count", # N
        "compression_factor": "mean",
        "compressed_fraction": "mean",
        "space_saving_pct": "mean",
        "comp_lat_p50": "mean",
        "decomp_lat_p50": "mean"
    }
    df_by_length = df.groupby(["codec", "length_bucket"]).agg(agg_funcs).reset_index()
    df_by_length.rename(columns={"original_bytes": "N"}, inplace=True)
    
    # Compute Throughput MB/s: (Mean Bytes / 1024^2) / (Mean Latency us / 1,000,000)
    # Average original bytes in bucket
    avg_bytes = df.groupby(["codec", "length_bucket"])["original_bytes"].mean().reset_index()
    df_by_length["avg_bytes"] = avg_bytes["original_bytes"]
    
    df_by_length["comp_throughput_MB_s"] = np.where(
        df_by_length["comp_lat_p50"] > 0,
        (df_by_length["avg_bytes"] / 1048576) / (df_by_length["comp_lat_p50"] / 1000000),
        np.nan
    )
    df_by_length["decomp_throughput_MB_s"] = np.where(
        df_by_length["decomp_lat_p50"] > 0,
        (df_by_length["avg_bytes"] / 1048576) / (df_by_length["decomp_lat_p50"] / 1000000),
        np.nan
    )
    
    len_path = os.path.join(BENCHMARK_RESULTS_DIR, "codec_by_length.csv")
    df_by_length.to_csv(len_path, index=False)
    print(f"Saved aggregated length data to {len_path}")
    
    # Generate Plots
    generate_plots(df_by_length)
    
def generate_plots(df_by_length: pd.DataFrame):
    fig_dir = os.path.join(BENCHMARK_RESULTS_DIR, "figures")
    order = ["0-15", "16-31", "32-63", "64-127", "128-255", "256-511", "512-1023", "1024+"]
    
    df = df_by_length.copy()
    df['length_bucket'] = pd.Categorical(df['length_bucket'], categories=order, ordered=True)
    df = df.sort_values(['codec', 'length_bucket'])
    
    # 1. Ratio vs Length
    plt.figure(figsize=(10, 6))
    for codec in CODECS:
        c_df = df[df['codec'] == codec]
        plt.plot(c_df['length_bucket'], c_df['compression_factor'], marker='o', label=codec)
    plt.axhline(1.0, color='red', linestyle='--', label='No Compression Boundary')
    plt.title('Compression Factor by Message Length\n(Higher is Better)')
    plt.ylabel('Compression Factor (Original/Compressed)')
    plt.xlabel('Message Length (UTF-8 Bytes)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "codec_ratio_vs_length.png"))
    plt.close()
    
    # 2. Savings vs Length
    plt.figure(figsize=(10, 6))
    for codec in CODECS:
        c_df = df[df['codec'] == codec]
        plt.plot(c_df['length_bucket'], c_df['space_saving_pct'], marker='o', label=codec)
    plt.axhline(0.0, color='red', linestyle='--', label='No Savings Boundary')
    plt.title('Space Savings % by Message Length\n(Higher is Better)')
    plt.ylabel('Space Savings %')
    plt.xlabel('Message Length (UTF-8 Bytes)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "codec_savings_vs_length.png"))
    plt.close()
    
    # 3. Latency vs Length
    plt.figure(figsize=(10, 6))
    for codec in CODECS:
        c_df = df[df['codec'] == codec]
        plt.plot(c_df['length_bucket'], c_df['comp_lat_p50'], marker='o', label=codec)
    plt.title('Compression Latency (p50) by Message Length\n(Lower is Better)')
    plt.ylabel('Latency (microseconds)')
    plt.xlabel('Message Length (UTF-8 Bytes)')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "codec_latency_vs_length.png"))
    plt.close()
    
    # 4. Throughput
    plt.figure(figsize=(10, 6))
    for codec in CODECS:
        c_df = df[df['codec'] == codec]
        plt.plot(c_df['length_bucket'], c_df['comp_throughput_MB_s'], marker='o', label=codec)
    plt.title('Compression Throughput by Message Length\n(Higher is Better)')
    plt.ylabel('Throughput (MB/s)')
    plt.xlabel('Message Length (UTF-8 Bytes)')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "codec_throughput.png"))
    plt.close()

if __name__ == "__main__":
    run_microbenchmarks()
