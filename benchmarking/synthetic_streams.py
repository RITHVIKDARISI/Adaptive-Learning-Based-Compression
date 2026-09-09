import os
import random
import string
import pandas as pd
import numpy as np

from src.utils import get_data_dir

def generate_poisson_timestamps(n_messages: int, rate_hz: float = 1.0) -> list[float]:
    timestamps = []
    current_time = 0.0
    for _ in range(n_messages):
        current_time += np.random.exponential(scale=1.0 / rate_hz)
        timestamps.append(current_time)
    return timestamps

def generate_message(length: int, entropy: str) -> str:
    if entropy == "high":
        chars = string.ascii_letters + string.digits + string.punctuation
        return ''.join(random.choices(chars, k=length))
    elif entropy == "medium":
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "data", "stream", "compress"]
        return ' '.join(random.choices(words, k=max(1, length // 5)))[:length].ljust(length, 'a')
    else: # low
        return ('a' * (length // 2)) + ('b' * (length - length // 2))

def create_robustness_stream(n_samples: int, dup_prob: float, length: int, rate: float, entropy: str) -> pd.DataFrame:
    messages = []
    pool = []
    
    # Pre-generate a small pool for duplicates
    for _ in range(10):
        pool.append(generate_message(length, entropy))
        
    for _ in range(n_samples):
        if random.random() < dup_prob:
            messages.append(random.choice(pool))
        else:
            messages.append(generate_message(length, entropy))
            
    timestamps = generate_poisson_timestamps(n_samples, rate_hz=rate)
    return pd.DataFrame({
        "msg_id": range(n_samples),
        "text": messages,
        "timestamp": timestamps,
        "source_type": "SYNTHETIC_ROBUSTNESS"
    })

def create_drift_scenario_1() -> pd.DataFrame:
    # repetitive long text -> unique short text
    df1 = create_robustness_stream(5000, dup_prob=0.8, length=512, rate=10, entropy="low")
    df2 = create_robustness_stream(5000, dup_prob=0.0, length=32, rate=10, entropy="high")
    df2["timestamp"] += df1["timestamp"].max() + 0.1
    df2["msg_id"] += len(df1)
    return pd.concat([df1, df2], ignore_index=True)

def create_drift_scenario_2() -> pd.DataFrame:
    # unique short text -> repetitive long text
    df1 = create_robustness_stream(5000, dup_prob=0.0, length=32, rate=10, entropy="high")
    df2 = create_robustness_stream(5000, dup_prob=0.8, length=512, rate=10, entropy="low")
    df2["timestamp"] += df1["timestamp"].max() + 0.1
    df2["msg_id"] += len(df1)
    return pd.concat([df1, df2], ignore_index=True)

def create_drift_scenario_3() -> pd.DataFrame:
    # low arrival rate -> burst traffic
    df1 = create_robustness_stream(5000, dup_prob=0.2, length=128, rate=1, entropy="medium")
    df2 = create_robustness_stream(5000, dup_prob=0.2, length=128, rate=1000, entropy="medium")
    df2["timestamp"] += df1["timestamp"].max() + 0.1
    df2["msg_id"] += len(df1)
    return pd.concat([df1, df2], ignore_index=True)

def create_drift_scenario_4() -> pd.DataFrame:
    # natural English -> JSON/log-style messages
    df1 = create_robustness_stream(5000, dup_prob=0.1, length=128, rate=10, entropy="medium")
    
    # Custom JSON logs
    json_msgs = []
    for _ in range(5000):
        json_msgs.append(f'{{"user": "u{random.randint(1,100)}", "action": "click", "ts": {time.time()}}}')
    df2 = pd.DataFrame({
        "msg_id": range(len(df1), len(df1)+5000),
        "text": json_msgs,
        "timestamp": generate_poisson_timestamps(5000, 10),
        "source_type": "SYNTHETIC_ROBUSTNESS"
    })
    df2["timestamp"] += df1["timestamp"].max() + 0.1
    return pd.concat([df1, df2], ignore_index=True)

def create_drift_scenario_5() -> pd.DataFrame:
    # low duplicate rate -> high duplicate rate
    df1 = create_robustness_stream(5000, dup_prob=0.05, length=128, rate=10, entropy="medium")
    df2 = create_robustness_stream(5000, dup_prob=0.90, length=128, rate=10, entropy="medium")
    df2["timestamp"] += df1["timestamp"].max() + 0.1
    df2["msg_id"] += len(df1)
    return pd.concat([df1, df2], ignore_index=True)

def create_drift_scenario_6() -> pd.DataFrame:
    # alternating regimes every 500 messages
    dfs = []
    current_ts = 0.0
    for i in range(20): # 20 * 500 = 10,000 messages
        if i % 2 == 0:
            df = create_robustness_stream(500, dup_prob=0.8, length=512, rate=100, entropy="low")
        else:
            df = create_robustness_stream(500, dup_prob=0.0, length=32, rate=5, entropy="high")
            
        df["timestamp"] += current_ts
        df["msg_id"] += i * 500
        current_ts = df["timestamp"].max() + 0.1
        dfs.append(df)
        
    return pd.concat(dfs, ignore_index=True)

def generate_all_streams():
    random.seed(42)
    np.random.seed(42)
    
    out_dir = os.path.join(get_data_dir(), "synthetic_robustness")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Drift Scenarios (Prompt 9)
    print("Generating Drift Scenarios...")
    create_drift_scenario_1().to_csv(os.path.join(out_dir, "drift_1_repL_uniqS.csv"), index=False)
    create_drift_scenario_2().to_csv(os.path.join(out_dir, "drift_2_uniqS_repL.csv"), index=False)
    create_drift_scenario_3().to_csv(os.path.join(out_dir, "drift_3_lowRate_burst.csv"), index=False)
    create_drift_scenario_4().to_csv(os.path.join(out_dir, "drift_4_nat_json.csv"), index=False)
    create_drift_scenario_5().to_csv(os.path.join(out_dir, "drift_5_lowDup_highDup.csv"), index=False)
    create_drift_scenario_6().to_csv(os.path.join(out_dir, "drift_6_alternating.csv"), index=False)
    
    # 2. Base Robustness Grids (Prompt 8)
    # We won't generate ALL combinations to avoid 50GB of CSVs. We'll pick key axes.
    print("Generating Base Robustness Grids...")
    for dup in [0.0, 0.25, 0.75]:
        for length in [32, 128, 512]:
            df = create_robustness_stream(2000, dup_prob=dup, length=length, rate=10, entropy="medium")
            df.to_csv(os.path.join(out_dir, f"grid_dup{int(dup*100)}_len{length}.csv"), index=False)
            
    print(f"Saved all streams to {out_dir}")

if __name__ == "__main__":
    import time
    generate_all_streams()
