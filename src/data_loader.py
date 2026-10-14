import os
import re
import sys
import random
import argparse
import pandas as pd
import numpy as np
from datasets import load_dataset
from src.utils import get_data_dir

def generate_poisson_timestamps(n_messages: int, mean_interval_sec: float = 1.0) -> list[float]:
    """Generates synthetic timestamps using an exponential distribution for inter-arrival times."""
    timestamps = []
    current_time = 0.0
    for _ in range(n_messages):
        interval = np.random.exponential(scale=mean_interval_sec)
        current_time += interval
        timestamps.append(current_time)
    return timestamps

# High-fidelity synthetic text generators for offline/fallback mode
def generate_synthetic_conversational(n_samples: int) -> list[str]:
    greetings = ["hi", "hello", "hey there", "good morning", "good evening", "how are you?", "how is it going?"]
    questions = ["what are you doing?", "are you busy?", "did you finish the homework?", "want to grab lunch?", "what time is it?"]
    answers = ["nothing much", "just working", "not yet", "sure, let's go!", "it's 5 o'clock", "fine, thanks", "i'm good"]
    closings = ["bye", "see you later", "talk to you tomorrow", "good night", "take care", "have a nice day!"]
    
    pool = greetings + questions + answers + closings
    messages = []
    
    # Introduce explicit repetitions to simulate cache hits & repetition features
    for _ in range(n_samples):
        if random.random() < 0.25:  # 25% chance of exact repeat of a common message
            messages.append(random.choice(greetings + closings))
        else:
            messages.append(random.choice(pool) + f" {random.randint(1, 100) if random.random() < 0.1 else ''}")
            
    return messages

def generate_synthetic_tweets(n_samples: int) -> list[str]:
    templates = [
        "loving the weather today! #sunny",
        "just had the best coffee ever",
        "working on my new machine learning project",
        "so tired, need some sleep...",
        "congrats to the team for the launch! 🚀",
        "who is watching the game tonight?",
        "learning python is so much fun",
        "traffic is terrible today 😠",
        "cannot wait for the weekend!"
    ]
    messages = []
    for _ in range(n_samples):
        if random.random() < 0.15:
            messages.append(random.choice(templates))
        else:
            msg = random.choice(templates)
            if random.random() < 0.5:
                msg += f" {random.choice(['#coding', '#life', '#fun', '!!!', ':)'])}"
            messages.append(msg)
    return messages

def generate_synthetic_sms(n_samples: int) -> list[str]:
    templates = [
        "Hey, where are you?",
        "Ok, see you there",
        "Can you call me back?",
        "I will be late by 10 mins",
        "Please bring some milk on your way home",
        "Got it, thanks!",
        "Let me know when you reach",
        "Happy birthday!",
        "Are we still on for tonight?"
    ]
    messages = []
    for _ in range(n_samples):
        if random.random() < 0.20:
            messages.append(random.choice(templates))
        else:
            messages.append(random.choice(templates))
    return messages

def prepare_dailydialog(n_samples: int = 5000, no_synthetic: bool = False, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    print("Loading DailyDialog...")
    messages = []
    sources = []
    
    # 1. Try loading refs/convert/parquet revision of official dataset
    try:
        print("Attempting to load daily_dialog via refs/convert/parquet...")
        dataset = load_dataset("daily_dialog", revision="refs/convert/parquet", split="train")
        for item in dataset:
            if "dialog" in item:
                for utterance in item["dialog"]:
                    clean_text = utterance.strip()
                    if clean_text and len(clean_text) <= 500:
                        messages.append(clean_text)
                        sources.append("real")
    except Exception as e:
        print(f"Hugging Face load_dataset failed: {e}. Trying hf_hub_download for daily_dialog refs/convert/parquet...")
        try:
            from huggingface_hub import hf_hub_download
            pq_path = hf_hub_download(
                repo_id="daily_dialog",
                filename="default/train/0000.parquet",
                revision="refs/convert/parquet",
                repo_type="dataset"
            )
            df_pq = pd.read_parquet(pq_path)
            for dialog in df_pq["dialog"]:
                for utterance in dialog:
                    clean_text = utterance.strip()
                    if clean_text and len(clean_text) <= 500:
                        messages.append(clean_text)
                        sources.append("real")
        except Exception as e_hub:
            print(f"hf_hub_download failed: {e_hub}. Trying data-only community mirror...")
            try:
                dataset = load_dataset("roskoN/dailydialog", split="train")
                for item in dataset:
                    dialog_key = "dialog" if "dialog" in item else ("dialogue" if "dialogue" in item else None)
                    if dialog_key:
                        for utterance in item[dialog_key]:
                            clean_text = utterance.strip()
                            if clean_text and len(clean_text) <= 500:
                                messages.append(clean_text)
                                sources.append("real")
            except Exception as e2:
                if no_synthetic:
                    raise RuntimeError(
                        f"Failed to download DailyDialog from Hugging Face: {e2}. "
                        f"Fallback to synthetic data is disabled (--no-synthetic is set)."
                    ) from e2
                print(f"Community mirror failed: {e2}. Bypassing network: generating high-fidelity synthetic conversational data...")
                syn = generate_synthetic_conversational(n_samples)
                messages = syn
                sources = ["synthetic"] * len(syn)

    # Truncate / sample or pad
    if n_samples is not None:
        if len(messages) > n_samples:
            random.seed(seed)
            indices = random.sample(range(len(messages)), n_samples)
            messages = [messages[i] for i in indices]
            sources = [sources[i] for i in indices]
        elif len(messages) < n_samples:
            if no_synthetic:
                raise RuntimeError(
                    f"DailyDialog has only {len(messages)} real messages, fewer than requested {n_samples}. "
                    f"Padding is disabled with --no-synthetic."
                )
            pad_count = n_samples - len(messages)
            messages += generate_synthetic_conversational(pad_count)
            sources += ["synthetic"] * pad_count

    if not messages:
        if no_synthetic:
            raise RuntimeError("No messages available for DailyDialog and --no-synthetic is set.")
        syn = generate_synthetic_conversational(n_samples or 5000)
        messages = syn
        sources = ["synthetic"] * len(syn)

    timestamps = generate_poisson_timestamps(len(messages), mean_interval_sec=0.5)
    output_path = os.path.join(get_data_dir(), "dailydialog_clean.csv")
    
    df = pd.DataFrame({
        "msg_id": range(len(messages)),
        "text": messages,
        "timestamp": timestamps,
        "source": sources
    })
    df.to_csv(output_path, index=False)
    source_counts = dict(df["source"].value_counts())
    print(f"DailyDialog prepared: {len(df)} rows. Source breakdown: {source_counts}")
    print(f"Saved {len(df)} DailyDialog messages to {output_path}")
    return df

def prepare_sentiment140(n_samples: int = 5000, no_synthetic: bool = False, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    print("Loading Sentiment140...")
    messages = []
    sources = []
    pool_size = (n_samples * 2) if n_samples else 10000
    
    # 1. Try loading refs/convert/parquet revision of official dataset
    try:
        print("Attempting to load sentiment140 via refs/convert/parquet (streaming)...")
        dataset = load_dataset("sentiment140", revision="refs/convert/parquet", split="train", streaming=True)
        for item in dataset:
            text_key = "text" if "text" in item else ("tweet" if "tweet" in item else None)
            if text_key:
                clean_text = re_clean_tweet(item[text_key])
                if clean_text and len(clean_text) <= 280:
                    messages.append(clean_text)
                    sources.append("real")
                    if len(messages) >= pool_size:
                        break
    except Exception as e:
        print(f"Hugging Face refs/convert/parquet failed: {e}. Trying bdanko/sentiment140...")
        try:
            dataset = load_dataset("bdanko/sentiment140", split="train", streaming=True)
            for item in dataset:
                text_key = "text" if "text" in item else ("tweet" if "tweet" in item else None)
                if text_key:
                    clean_text = re_clean_tweet(item[text_key])
                    if clean_text and len(clean_text) <= 280:
                        messages.append(clean_text)
                        sources.append("real")
                        if len(messages) >= pool_size:
                            break
        except Exception as e2:
            if no_synthetic:
                raise RuntimeError(
                    f"Failed to download Sentiment140 from Hugging Face: {e2}. "
                    f"Fallback to synthetic data is disabled (--no-synthetic is set)."
                ) from e2
            print(f"bdanko/sentiment140 failed: {e2}. Generating high-fidelity synthetic tweet stream...")
            syn = generate_synthetic_tweets(n_samples or 5000)
            messages = syn
            sources = ["synthetic"] * len(syn)

    if n_samples is not None:
        if len(messages) > n_samples:
            random.seed(seed)
            indices = random.sample(range(len(messages)), n_samples)
            messages = [messages[i] for i in indices]
            sources = [sources[i] for i in indices]
        elif len(messages) < n_samples:
            if no_synthetic:
                raise RuntimeError(
                    f"Sentiment140 has only {len(messages)} real messages, fewer than requested {n_samples}. "
                    f"Padding is disabled with --no-synthetic."
                )
            pad_count = n_samples - len(messages)
            messages += generate_synthetic_tweets(pad_count)
            sources += ["synthetic"] * pad_count

    timestamps = generate_poisson_timestamps(len(messages), mean_interval_sec=0.2)
    output_path = os.path.join(get_data_dir(), "sentiment140_clean.csv")
    
    df = pd.DataFrame({
        "msg_id": range(len(messages)),
        "text": messages,
        "timestamp": timestamps,
        "source": sources
    })
    df.to_csv(output_path, index=False)
    source_counts = dict(df["source"].value_counts())
    print(f"Sentiment140 prepared: {len(df)} rows. Source breakdown: {source_counts}")
    print(f"Saved {len(df)} Sentiment140 messages to {output_path}")
    return df

def re_clean_tweet(text: str) -> str:
    # Remove @user tags
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Remove http links
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def prepare_sms_spam_ham(n_samples: int = None, no_synthetic: bool = False, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    print("Loading SMS Spam Collection (ham subset)...")
    messages = []
    sources = []
    
    # 1. Try loading refs/convert/parquet revision of official dataset
    try:
        print("Attempting to load sms_spam via refs/convert/parquet...")
        dataset = load_dataset("sms_spam", revision="refs/convert/parquet", split="train")
        sms_key = "sms" if "sms" in dataset.column_names else ("text" if "text" in dataset.column_names else None)
        label_key = "label" if "label" in dataset.column_names else None
        
        for item in dataset:
            is_ham = True
            if label_key:
                is_ham = item[label_key] in [0, "ham", "clean"]
            if is_ham and sms_key:
                clean_text = item[sms_key].strip()
                if clean_text:
                    messages.append(clean_text)
                    sources.append("real")
    except Exception as e:
        print(f"Hugging Face refs/convert/parquet failed: {e}. Trying ucirvine/sms_spam...")
        try:
            dataset = load_dataset("ucirvine/sms_spam", split="train")
            sms_key = "sms" if "sms" in dataset.column_names else ("text" if "text" in dataset.column_names else None)
            label_key = "label" if "label" in dataset.column_names else None
            
            for item in dataset:
                is_ham = True
                if label_key:
                    is_ham = item[label_key] in [0, "ham", "clean"]
                if is_ham and sms_key:
                    clean_text = item[sms_key].strip()
                    if clean_text:
                        messages.append(clean_text)
                        sources.append("real")
        except Exception as e2:
            if no_synthetic:
                raise RuntimeError(
                    f"Failed to download sms_spam from Hugging Face: {e2}. "
                    f"Fallback to synthetic data is disabled (--no-synthetic is set)."
                ) from e2
            print(f"ucirvine/sms_spam failed: {e2}. Generating high-fidelity synthetic SMS data...")
            syn = generate_synthetic_sms(n_samples or 5000)
            messages = syn
            sources = ["synthetic"] * len(syn)

    if n_samples is not None:
        if len(messages) > n_samples:
            random.seed(seed)
            indices = random.sample(range(len(messages)), n_samples)
            messages = [messages[i] for i in indices]
            sources = [sources[i] for i in indices]
        elif len(messages) < n_samples:
            if no_synthetic:
                raise RuntimeError(
                    f"sms_spam (ham only) has only {len(messages)} real messages, fewer than requested {n_samples}. "
                    f"Padding is disabled with --no-synthetic."
                )
            pad_count = n_samples - len(messages)
            messages += generate_synthetic_sms(pad_count)
            sources += ["synthetic"] * pad_count

    if not messages:
        if no_synthetic:
            raise RuntimeError("No messages available for sms_spam (ham only) and --no-synthetic is set.")
        syn = generate_synthetic_sms(n_samples or 5000)
        messages = syn
        sources = ["synthetic"] * len(syn)

    timestamps = generate_poisson_timestamps(len(messages), mean_interval_sec=1.5)
    
    df = pd.DataFrame({
        "msg_id": range(len(messages)),
        "text": messages,
        "timestamp": timestamps,
        "source": sources
    })
    
    # Save as canonical sms_spam_ham_clean.csv
    output_path = os.path.join(get_data_dir(), "sms_spam_ham_clean.csv")
    df.to_csv(output_path, index=False)
    
    # Also save as nussms_clean.csv for backwards compatibility with any unmigrated runners
    legacy_path = os.path.join(get_data_dir(), "nussms_clean.csv")
    df.to_csv(legacy_path, index=False)
    
    source_counts = dict(df["source"].value_counts())
    print(f"sms_spam_ham prepared: {len(df)} rows. Source breakdown: {source_counts}")
    print(f"Saved {len(df)} SMS messages to {output_path} (and legacy alias {legacy_path})")
    return df

# Backward-compatible alias
prepare_nus_sms = prepare_sms_spam_ham

def prepare_all_datasets(n_samples: int = None, no_synthetic: bool = False, seed: int = 42):
    """
    Prepares all three datasets.
    If no_synthetic is True, requests exact real samples (5000 for DailyDialog & Sentiment140,
    and all 4827 real ham messages for sms_spam_ham).
    """
    os.makedirs(get_data_dir(), exist_ok=True)
    daily_n = n_samples if n_samples is not None else 5000
    sent_n = n_samples if n_samples is not None else 5000
    sms_n = n_samples if (n_samples is not None and not no_synthetic) else (None if no_synthetic else 5000)
    
    print(f"\n==================================================")
    print(f"PREPARING DATASETS (no_synthetic={no_synthetic}, seed={seed})")
    print(f"==================================================")
    
    df_daily = prepare_dailydialog(daily_n, no_synthetic=no_synthetic, seed=seed)
    df_sent = prepare_sentiment140(sent_n, no_synthetic=no_synthetic, seed=seed)
    df_sms = prepare_sms_spam_ham(sms_n, no_synthetic=no_synthetic, seed=seed)
    
    print("\nSummary of Prepared Datasets:")
    for name, df in [("DailyDialog", df_daily), ("Sentiment140", df_sent), ("SMS Spam Ham", df_sms)]:
        print(f"  {name}: {len(df)} total rows, source breakdown: {dict(df['source'].value_counts())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset streams for compression benchmarking")
    parser.add_argument("--no-synthetic", action="store_true", help="Raise an error instead of falling back or padding")
    parser.add_argument("--n-samples", type=int, default=None, help="Target samples per dataset (default: 5000, 4827 for sms_spam_ham when no-synthetic)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    
    prepare_all_datasets(n_samples=args.n_samples, no_synthetic=args.no_synthetic, seed=args.seed)
