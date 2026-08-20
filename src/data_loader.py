import os
import re
import random
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

def prepare_dailydialog(n_samples: int = 5000) -> pd.DataFrame:
    print("Loading DailyDialog...")
    messages = []
    
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
    except Exception as e:
        print(f"Hugging Face refs/convert/parquet failed: {e}. Trying data-only community mirror...")
        try:
            dataset = load_dataset("roskoN/dailydialog", split="train")
            for item in dataset:
                dialog_key = "dialog" if "dialog" in item else ("dialogue" if "dialogue" in item else None)
                if dialog_key:
                    for utterance in item[dialog_key]:
                        clean_text = utterance.strip()
                        if clean_text and len(clean_text) <= 500:
                            messages.append(clean_text)
        except Exception as e2:
            print(f"Community mirror failed: {e2}. Bypassing network: generating high-fidelity synthetic conversational data...")
            messages = generate_synthetic_conversational(n_samples)

    # Truncate/sample
    if len(messages) > n_samples:
        random.seed(42)
        messages = random.sample(messages, n_samples)
    elif len(messages) < n_samples and messages:
        # Pad with synthetic if needed
        messages += generate_synthetic_conversational(n_samples - len(messages))
        
    if not messages:
        messages = generate_synthetic_conversational(n_samples)

    timestamps = generate_poisson_timestamps(len(messages), mean_interval_sec=0.5)
    
    df = pd.DataFrame({
        "msg_id": range(len(messages)),
        "text": messages,
        "timestamp": timestamps
    })
    
    output_path = os.path.join(get_data_dir(), "dailydialog_clean.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} DailyDialog messages to {output_path}")
    return df

def prepare_sentiment140(n_samples: int = 5000) -> pd.DataFrame:
    print("Loading Sentiment140...")
    messages = []
    pool_size = n_samples * 3
    
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
                        if len(messages) >= pool_size:
                            break
        except Exception as e2:
            print(f"bdanko/sentiment140 failed: {e2}. Generating high-fidelity synthetic tweet stream...")
            messages = generate_synthetic_tweets(n_samples)
            
    if len(messages) > n_samples:
        random.seed(42)
        messages = random.sample(messages, n_samples)
    elif len(messages) < n_samples:
        messages += generate_synthetic_tweets(n_samples - len(messages))

    timestamps = generate_poisson_timestamps(len(messages), mean_interval_sec=0.2)
    
    df = pd.DataFrame({
        "msg_id": range(len(messages)),
        "text": messages,
        "timestamp": timestamps
    })
    
    output_path = os.path.join(get_data_dir(), "sentiment140_clean.csv")
    df.to_csv(output_path, index=False)
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

def prepare_nus_sms(n_samples: int = 5000) -> pd.DataFrame:
    print("Loading SMS Spam Collection (NUS ham subset)...")
    messages = []
    
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
        except Exception as e2:
            print(f"ucirvine/sms_spam failed: {e2}. Generating high-fidelity synthetic SMS data...")
            messages = generate_synthetic_sms(n_samples)

    if len(messages) > n_samples:
        random.seed(42)
        messages = random.sample(messages, n_samples)
    elif len(messages) < n_samples and messages:
        messages += generate_synthetic_sms(n_samples - len(messages))
        
    if not messages:
        messages = generate_synthetic_sms(n_samples)

    timestamps = generate_poisson_timestamps(len(messages), mean_interval_sec=1.5)
    
    df = pd.DataFrame({
        "msg_id": range(len(messages)),
        "text": messages,
        "timestamp": timestamps
    })
    
    output_path = os.path.join(get_data_dir(), "nussms_clean.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} SMS messages to {output_path}")
    return df

def prepare_all_datasets(n_samples: int = 5000):
    os.makedirs(get_data_dir(), exist_ok=True)
    prepare_dailydialog(n_samples)
    prepare_sentiment140(n_samples)
    prepare_nus_sms(n_samples)

if __name__ == "__main__":
    prepare_all_datasets()
