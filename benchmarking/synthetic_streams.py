import os
import random
import string
import time
import pandas as pd
import numpy as np

from src.utils import get_data_dir

def generate_poisson_timestamps(n_messages: int, rate_hz: float = 1.0, rng: np.random.Generator = None) -> list[float]:
    if rng is None:
        rng = np.random.default_rng()
    intervals = rng.exponential(scale=1.0 / rate_hz, size=n_messages)
    return list(np.cumsum(intervals))

def generate_message(length: int, entropy: str, prng: random.Random = None) -> str:
    if prng is None:
        prng = random.Random()
    if entropy == "high":
        chars = string.ascii_letters + string.digits + string.punctuation
        return ''.join(prng.choices(chars, k=length))
    elif entropy == "medium":
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "data", "stream", "compress"]
        return ' '.join(prng.choices(words, k=max(1, length // 5)))[:length].ljust(length, 'a')
    else:  # low
        return ('a' * (length // 2)) + ('b' * (length - length // 2))

def create_robustness_stream(n_samples: int, dup_prob: float, length: int, rate: float, entropy: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    pool = [generate_message(length, entropy, prng=prng) for _ in range(10)]
    messages = []
    for _ in range(n_samples):
        if prng.random() < dup_prob:
            messages.append(prng.choice(pool))
        else:
            messages.append(generate_message(length, entropy, prng=prng))
    timestamps = generate_poisson_timestamps(n_samples, rate_hz=rate, rng=rng)
    return pd.DataFrame({
        "msg_id": range(n_samples),
        "text": messages,
        "timestamp": timestamps,
        "source_type": "SYNTHETIC_ROBUSTNESS"
    })

def create_drift_scenario_1(seed: int = 42, n_samples: int = 5000) -> pd.DataFrame:
    # repetitive long text -> unique short text
    n1 = n_samples // 2
    n2 = n_samples - n1
    df1 = create_robustness_stream(n1, dup_prob=0.8, length=512, rate=10, entropy="low", seed=seed)
    df2 = create_robustness_stream(n2, dup_prob=0.0, length=32, rate=10, entropy="high", seed=seed + 10000)
    df2["timestamp"] += df1["timestamp"].max() + 0.1
    df2["msg_id"] += len(df1)
    return pd.concat([df1, df2], ignore_index=True)

def create_drift_scenario_2(seed: int = 42, n_samples: int = 5000) -> pd.DataFrame:
    # unique short text -> repetitive long text
    n1 = n_samples // 2
    n2 = n_samples - n1
    df1 = create_robustness_stream(n1, dup_prob=0.0, length=32, rate=10, entropy="high", seed=seed)
    df2 = create_robustness_stream(n2, dup_prob=0.8, length=512, rate=10, entropy="low", seed=seed + 10000)
    df2["timestamp"] += df1["timestamp"].max() + 0.1
    df2["msg_id"] += len(df1)
    return pd.concat([df1, df2], ignore_index=True)

def create_drift_scenario_3(seed: int = 42, n_samples: int = 5000) -> pd.DataFrame:
    # low arrival rate -> burst traffic
    n1 = n_samples // 2
    n2 = n_samples - n1
    df1 = create_robustness_stream(n1, dup_prob=0.2, length=128, rate=1, entropy="medium", seed=seed)
    df2 = create_robustness_stream(n2, dup_prob=0.2, length=128, rate=1000, entropy="medium", seed=seed + 10000)
    df2["timestamp"] += df1["timestamp"].max() + 0.1
    df2["msg_id"] += len(df1)
    return pd.concat([df1, df2], ignore_index=True)

def create_drift_scenario_4(seed: int = 42, n_samples: int = 5000) -> pd.DataFrame:
    # natural English -> JSON/log-style messages
    n1 = n_samples // 2
    n2 = n_samples - n1
    df1 = create_robustness_stream(n1, dup_prob=0.1, length=128, rate=10, entropy="medium", seed=seed)

    prng = random.Random(seed + 10000)
    rng = np.random.default_rng(seed + 10000)
    json_msgs = []
    base_ts = 1700000000.0 + (seed % 1000) * 1000
    for i in range(n2):
        action = prng.choice(["click", "view", "purchase", "login", "search"])
        json_msgs.append(f'{{"user": "u{prng.randint(1, 100)}", "action": "{action}", "ts": {base_ts + i * 0.1:.1f}}}')
    df2 = pd.DataFrame({
        "msg_id": range(len(df1), len(df1) + n2),
        "text": json_msgs,
        "timestamp": generate_poisson_timestamps(n2, rate_hz=10, rng=rng),
        "source_type": "SYNTHETIC_ROBUSTNESS"
    })
    df2["timestamp"] += df1["timestamp"].max() + 0.1
    return pd.concat([df1, df2], ignore_index=True)

def create_drift_scenario_5(seed: int = 42, n_samples: int = 5000) -> pd.DataFrame:
    # low duplicate rate -> high duplicate rate
    n1 = n_samples // 2
    n2 = n_samples - n1
    df1 = create_robustness_stream(n1, dup_prob=0.05, length=128, rate=10, entropy="medium", seed=seed)
    df2 = create_robustness_stream(n2, dup_prob=0.90, length=128, rate=10, entropy="medium", seed=seed + 10000)
    df2["timestamp"] += df1["timestamp"].max() + 0.1
    df2["msg_id"] += len(df1)
    return pd.concat([df1, df2], ignore_index=True)

def create_drift_scenario_6(seed: int = 42, n_samples: int = 5000, block_size: int = 500) -> pd.DataFrame:
    # alternating regimes every 500 messages
    n_blocks = max(2, n_samples // block_size)
    dfs = []
    current_ts = 0.0
    for i in range(n_blocks):
        block_seed = seed + i * 1000
        if i % 2 == 0:
            df = create_robustness_stream(block_size, dup_prob=0.8, length=512, rate=100, entropy="low", seed=block_seed)
        else:
            df = create_robustness_stream(block_size, dup_prob=0.0, length=32, rate=5, entropy="high", seed=block_seed)
        df["timestamp"] += current_ts
        df["msg_id"] += i * block_size
        current_ts = df["timestamp"].max() + 0.1
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def create_drift_scenario_7_natural_sequential(seed: int = 42, n_per_dataset: int = 2000) -> pd.DataFrame:
    """
    Natural drift stream (a): DailyDialog then Sentiment140 then SMS-Spam.
    100% real prepared data, no synthetic data.
    Random subsets sampled per seed without replacement.
    Total messages: 3 * n_per_dataset = 6,000 messages.
    """
    rng = np.random.default_rng(seed)
    data_dir = get_data_dir()

    df_dd = pd.read_csv(os.path.join(data_dir, "dailydialog_clean.csv"))
    df_s140 = pd.read_csv(os.path.join(data_dir, "sentiment140_clean.csv"))
    df_sms = pd.read_csv(os.path.join(data_dir, "sms_spam_ham_clean.csv"))

    idx_dd = rng.choice(len(df_dd), size=min(n_per_dataset, len(df_dd)), replace=False)
    idx_s140 = rng.choice(len(df_s140), size=min(n_per_dataset, len(df_s140)), replace=False)
    idx_sms = rng.choice(len(df_sms), size=min(n_per_dataset, len(df_sms)), replace=False)

    sub_dd = df_dd.iloc[idx_dd]["text"].tolist()
    sub_s140 = df_s140.iloc[idx_s140]["text"].tolist()
    sub_sms = df_sms.iloc[idx_sms]["text"].tolist()

    messages = sub_dd + sub_s140 + sub_sms
    timestamps = generate_poisson_timestamps(len(messages), rate_hz=10.0, rng=rng)

    return pd.DataFrame({
        "msg_id": range(len(messages)),
        "text": messages,
        "timestamp": timestamps,
        "source_type": "NATURAL_SEQUENTIAL"
    })

def create_drift_scenario_8_natural_switching(seed: int = 42, block_size: int = 1000) -> pd.DataFrame:
    """
    Natural drift stream (b): Switches between DailyDialog, Sentiment140, SMS-Spam every 1,000 messages.
    100% real prepared data, no synthetic data.
    Random disjoint subsets sampled per seed.
    Total: 6 blocks of 1,000 = 6,000 messages (DD -> S140 -> SMS -> DD -> S140 -> SMS).
    """
    rng = np.random.default_rng(seed)
    data_dir = get_data_dir()

    df_dd = pd.read_csv(os.path.join(data_dir, "dailydialog_clean.csv"))
    df_s140 = pd.read_csv(os.path.join(data_dir, "sentiment140_clean.csv"))
    df_sms = pd.read_csv(os.path.join(data_dir, "sms_spam_ham_clean.csv"))

    # Sample 2,000 non-overlapping rows from each dataset
    idx_dd = rng.choice(len(df_dd), size=2 * block_size, replace=False)
    idx_s140 = rng.choice(len(df_s140), size=2 * block_size, replace=False)
    idx_sms = rng.choice(len(df_sms), size=2 * block_size, replace=False)

    texts_dd1 = df_dd.iloc[idx_dd[:block_size]]["text"].tolist()
    texts_dd2 = df_dd.iloc[idx_dd[block_size:]]["text"].tolist()

    texts_s140_1 = df_s140.iloc[idx_s140[:block_size]]["text"].tolist()
    texts_s140_2 = df_s140.iloc[idx_s140[block_size:]]["text"].tolist()

    texts_sms1 = df_sms.iloc[idx_sms[:block_size]]["text"].tolist()
    texts_sms2 = df_sms.iloc[idx_sms[block_size:]]["text"].tolist()

    blocks = [texts_dd1, texts_s140_1, texts_sms1, texts_dd2, texts_s140_2, texts_sms2]
    messages = [msg for block in blocks for msg in block]
    timestamps = generate_poisson_timestamps(len(messages), rate_hz=10.0, rng=rng)

    return pd.DataFrame({
        "msg_id": range(len(messages)),
        "text": messages,
        "timestamp": timestamps,
        "source_type": "NATURAL_SWITCHING"
    })

def generate_drift_stream(scenario_name: str, seed: int = 42) -> tuple[list[str], list[float], int]:
    """
    Generates a fresh drift stream for a given scenario and seed.
    Returns:
        (messages, timestamps, change_point)
    """
    if scenario_name == "drift_1_repL_uniqS":
        df = create_drift_scenario_1(seed=seed, n_samples=5000)
        change_point = 2500
    elif scenario_name == "drift_2_uniqS_repL":
        df = create_drift_scenario_2(seed=seed, n_samples=5000)
        change_point = 2500
    elif scenario_name == "drift_3_lowRate_burst":
        df = create_drift_scenario_3(seed=seed, n_samples=5000)
        change_point = 2500
    elif scenario_name == "drift_4_nat_json":
        df = create_drift_scenario_4(seed=seed, n_samples=5000)
        change_point = 2500
    elif scenario_name == "drift_5_lowDup_highDup":
        df = create_drift_scenario_5(seed=seed, n_samples=5000)
        change_point = 2500
    elif scenario_name == "drift_6_alternating":
        df = create_drift_scenario_6(seed=seed, n_samples=5000, block_size=500)
        change_point = 1000
    elif scenario_name == "drift_7_natural_sequential":
        df = create_drift_scenario_7_natural_sequential(seed=seed, n_per_dataset=2000)
        change_point = 2000
    elif scenario_name == "drift_8_natural_switching":
        df = create_drift_scenario_8_natural_switching(seed=seed, block_size=1000)
        change_point = 1000
    else:
        raise ValueError(f"Unknown scenario name: {scenario_name}")

    return df["text"].tolist(), df["timestamp"].tolist(), change_point

def generate_all_streams():
    """Generates legacy offline files for backward compatibility."""
    out_dir = os.path.join(get_data_dir(), "synthetic_robustness")
    os.makedirs(out_dir, exist_ok=True)

    create_drift_scenario_1(seed=42, n_samples=10000).to_csv(os.path.join(out_dir, "drift_1_repL_uniqS.csv"), index=False)
    create_drift_scenario_2(seed=42, n_samples=10000).to_csv(os.path.join(out_dir, "drift_2_uniqS_repL.csv"), index=False)
    create_drift_scenario_3(seed=42, n_samples=10000).to_csv(os.path.join(out_dir, "drift_3_lowRate_burst.csv"), index=False)
    create_drift_scenario_4(seed=42, n_samples=10000).to_csv(os.path.join(out_dir, "drift_4_nat_json.csv"), index=False)
    create_drift_scenario_5(seed=42, n_samples=10000).to_csv(os.path.join(out_dir, "drift_5_lowDup_highDup.csv"), index=False)
    create_drift_scenario_6(seed=42, n_samples=10000, block_size=500).to_csv(os.path.join(out_dir, "drift_6_alternating.csv"), index=False)

if __name__ == "__main__":
    generate_all_streams()
