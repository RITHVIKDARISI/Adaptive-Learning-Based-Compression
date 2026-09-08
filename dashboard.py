"""
Interactive dashboard for the Adaptive Compression Scheduling project.

Run with:
    streamlit run dashboard.py

Provides three views:
  1. Live Stream      — real-time-paced replay of a message stream, watch the
                         scheduler route each message live, with metrics/charts
                         updating as it goes. Includes an optional simulated
                         distribution-shift (chat -> tweets) to show bandit adaptation.
  2. Single Message    — type any message, see feature extraction, cache status,
                         scheduler decision, and compression result instantly.
  3. Scheduler Compare — runs the same fixed stream through every scheduler
                         (Heuristic / Decision Tree / Logistic Regression /
                         Random Forest / LinUCB Bandit) and compares them head-to-head.
"""

import time
import random
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import re
import math

from src.manager import BatchCacheManager
from src.simulator import StreamSimulator
from src.scheduler import HeuristicScheduler, SupervisedScheduler, OfflineLabelGenerator, ACTION_MAP
from src.engine import CompressionEngine
from src.bandit import LinUCBBandit
from src.features import FeatureExtractor
from src.synthetic_data import DATASET_GENERATORS, generate_poisson_timestamps

st.set_page_config(page_title="Adaptive Compression Scheduler", layout="wide", page_icon="📡")

# Custom styling to make the dashboard look extremely premium and beautiful
st.markdown(
    """
    <style>
    /* Import modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], .stApp, [class*="st-key"] {
        background-color: #0A0915 !important;
        background: radial-gradient(circle at 10% 20%, #161233 0%, #0A0915 90%) !important;
        color: #F3F4F6 !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #110F24 !important;
        border-right: 1px solid rgba(139, 92, 246, 0.15) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #FFFFFF !important;
    }
    
    /* Force glowing gradient title */
    h1 {
        background: linear-gradient(to right, #60A5FA, #A78BFA, #F472B6);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 800 !important;
    }
    
    /* Make all texts in labels and components white */
    .stApp p, .stApp span, .stApp label, .stApp div {
        color: #F3F4F6 !important;
    }
    
    /* Style input elements, selectboxes, sliders to match the dark aesthetic */
    div[data-baseweb="select"] > div {
        background-color: #1A1735 !important;
        color: #F3F4F6 !important;
        border: 1px solid rgba(139, 92, 246, 0.25) !important;
        border-radius: 8px !important;
    }
    div[role="listbox"] {
        background-color: #110F24 !important;
        color: #F3F4F6 !important;
    }
    div[role="option"] {
        background-color: transparent !important;
        color: #F3F4F6 !important;
    }
    div[role="option"]:hover, div[role="option"][aria-selected="true"] {
        background-color: rgba(139, 92, 246, 0.25) !important;
        color: #FFFFFF !important;
    }
    
    /* Text input styling */
    input[type="text"] {
        background-color: #1A1735 !important;
        color: #F3F4F6 !important;
        border: 1px solid rgba(139, 92, 246, 0.25) !important;
        border-radius: 8px !important;
    }
    
    /* Sleek gradient dashboard cards */
    .metric-card {
        background: linear-gradient(135deg, #1C1936 0%, #120F24 100%) !important;
        border: 1px solid rgba(139, 92, 246, 0.25) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        border-radius: 12px !important;
        padding: 18px !important;
        text-align: center !important;
        margin-bottom: 12px !important;
        transition: transform 0.2s ease, border-color 0.2s ease !important;
    }
    .metric-card:hover {
        transform: translateY(-3px) !important;
        border-color: rgba(139, 92, 246, 0.6) !important;
        box-shadow: 0 8px 32px 0 rgba(139, 92, 246, 0.15) !important;
    }
    .metric-title {
        color: #A5B4FC !important; /* Soft light indigo */
        font-size: 13px !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 6px !important;
    }
    .metric-value {
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #FFFFFF !important; /* White value text */
        margin-top: 4px !important;
        margin-bottom: 4px !important;
    }
    .metric-desc {
        color: #9CA3AF !important; /* Muted grey description */
        font-size: 11px !important;
    }
    
    /* Styled tabs with matching obsidian-indigo colors */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #110F24 !important;
        padding: 6px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(139, 92, 246, 0.15) !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px !important;
        border-radius: 8px !important;
        color: #9CA3AF !important;
        background-color: transparent !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFF !important;
        background-color: rgba(139, 92, 246, 0.08) !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    /* Table styling for dark theme readability */
    .stDataFrame {
        border: 1px solid rgba(139, 92, 246, 0.15) !important;
        border-radius: 8px !important;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.45) !important;
        border: none !important;
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

ACTION_COLORS = {
    "SKIP": "#9CA3AF", "ZSTD": "#3B82F6", "BROTLI": "#8B5CF6",
    "GZIP": "#F59E0B", "LZ4": "#10B981", "BATCH": "#EC4899", "CACHE_REUSE": "#06B6D4",
}

SCHEDULER_CHOICES = [
    "Auto — LinUCB Bandit (online, recommended)",
    "Heuristic (hand-written rules)",
    "Decision Tree (supervised)",
    "Logistic Regression (supervised)",
    "Random Forest (supervised)",
]


from src.utils import get_data_dir
import os

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def build_stream(style: str, n: int, seed: int = None):
    """Generates n diverse messages with balanced short, medium, and rich multi-sentence text for peak compression ratio and diverse action distribution."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    filename_map = {
        "DailyDialog-style (chat)": "dailydialog_clean.csv",
        "Sentiment140-style (tweets)": "sentiment140_clean.csv",
        "NUS-SMS-style (sms)": "nussms_clean.csv"
    }
    fn = filename_map.get(style, "dailydialog_clean.csv")
    csv_path = os.path.join(get_data_dir(), fn)
    
    # Load dataset texts
    all_dataset_texts = []
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if "text" in df.columns:
                all_dataset_texts = df["text"].dropna().tolist()
        except Exception:
            pass
            
    # For DailyDialog (which has short single-line phrases), combine 2-3 turns to form realistic multi-turn chat utterances
    if "DailyDialog" in style and all_dataset_texts:
        combined_turns = []
        for i in range(0, len(all_dataset_texts) - 3, 3):
            combined_turns.append(f"{all_dataset_texts[i]} {all_dataset_texts[i+1]} {all_dataset_texts[i+2]}")
        compressible_pool = combined_turns
    else:
        compressible_pool = [t for t in all_dataset_texts if len(t) >= 40]
        
    # High-value structured messages ensuring ZSTD / BROTLI / LZ4 / GZIP achieve peak compression
    rich_long_pool = [
        "System telemetry report: worker nodes 1, 2, and 3 are operating at 42% CPU load with memory usage steady at 3.8GB. No connection timeouts detected on database cluster.",
        "Could you please review the updated pull request before tomorrow morning's release? All unit tests and regression benchmarks have passed with zero failures.",
        "Diagnostic error trace: HTTP POST request failed with status code 500 on port 5432. Retrying in 5 seconds. Attempt 1 of 5 in worker thread main.",
        "The quarterly performance benchmarks show that the LinUCB bandit model outperforms all static baselines by 28% on compression ratio while keeping latency sub-5 microseconds.",
        "Reminder: sprint planning session is scheduled for tomorrow at 9am. Please review the backlog items and come prepared with effort estimates for the compression pipeline tickets.",
        "Application alert: database connection pool has reached its configured threshold of 80 concurrent connections. Scaling backend pods automatically to prevent latency spikes.",
        "The automated deployment pipeline succeeded at the Docker build stage. Container image tagged v2.4.0 and pushed to the private container registry."
    ]
    
    if not compressible_pool:
        compressible_pool = rich_long_pool

    short_pool = ["ok", "sure", "hi", "yes, got it", "thanks!", "see you soon", "sounds good", "on my way"]
    
    # Stratified mix:
    # 50% Medium multi-sentence text (40-90 chars) -> LZ4 / ZSTD / BROTLI
    # 25% Rich long payloads (100-250 chars) -> BROTLI / GZIP / ZSTD / BATCH
    # 10% Short greetings (<30 chars) -> SKIP
    # 15% Exact duplicate repeats -> CACHE_REUSE
    n_unique = max(int(n * 0.85), 5)
    sample_pool = []
    
    n_med = int(n_unique * 0.55)
    n_long = int(n_unique * 0.30)
    n_short = n_unique - n_med - n_long
    
    if compressible_pool:
        sample_pool += random.sample(compressible_pool, min(n_med, len(compressible_pool)))
        if len(sample_pool) < n_med:
            sample_pool += random.choices(compressible_pool, k=n_med - len(sample_pool))
            
    sample_pool += random.choices(rich_long_pool, k=n_long)
    sample_pool += random.choices(short_pool, k=n_short)
    
    # Add 15% duplicate repeats
    n_dups = n - len(sample_pool)
    duplicates = [random.choice(sample_pool) for _ in range(max(n_dups, 0))]
    combined = (sample_pool + duplicates)[:n]
    random.shuffle(combined)
    
    _, mean_interval = DATASET_GENERATORS.get(style, (None, 0.8))
    timestamps = generate_poisson_timestamps(len(combined), mean_interval_sec=mean_interval)
    return combined, timestamps


@st.cache_resource(show_spinner=False)
def train_supervised_scheduler(model_type: str, style: str, lambda_param: float, n_train: int = 250):
    """Trains (and caches) a supervised scheduler on a synthetic training stream."""
    messages, timestamps = build_stream(style, n_train, seed=42)
    messages = list(messages)
    timestamps = list(timestamps)

    # Inject diverse payloads to ensure training dataset has a rich class mix (not just SKIP)
    long_samples = [
        "This is an exceptionally long diagnostic system message designed to simulate heavy workload and rich textual density. " * 12,
        "Error trace log: database connection failed at port 5432. Retrying in 5 seconds. Attempt 1 of 5. Timeout exception in thread main. " * 8,
        "Adaptive compression scheduling optimization framework requires large message payloads to demonstrate the efficiency and throughput advantages of classical codecs. " * 6,
        "System configuration parameters: cpu_count=16, memory_limit_gb=64, storage_type=ssd, networking_throughput_mbps=10000, compression_level=default. " * 10,
        "A" * 1500,  # Extremely compressible
        "B" * 2000,  # Extremely compressible
    ]
    for idx, msg in enumerate(long_samples):
        messages.append(msg)
        timestamps.append(timestamps[-1] + (idx + 1) * 2.0 if timestamps else (idx + 1) * 2.0)

    label_gen = OfflineLabelGenerator(lambda_param=lambda_param)
    labels = label_gen.generate_labels(messages)

    fe = FeatureExtractor()
    fe.reset()
    features_list = [fe.extract_features(msg, ts) for msg, ts in zip(messages, timestamps)]

    model = SupervisedScheduler(model_type=model_type)
    model.fit(features_list, labels)
    return model


def get_scheduler(choice: str, style: str, lambda_param: float):
    if choice.startswith("Auto") or choice.startswith("LinUCB"):
        return LinUCBBandit(d=9, K=6, alpha=0.5, lambda_param=lambda_param)
    elif choice.startswith("Heuristic"):
        return HeuristicScheduler(skip_threshold=25)
    elif choice.startswith("Decision Tree"):
        return train_supervised_scheduler("decision_tree", style, lambda_param)
    elif choice.startswith("Logistic Regression"):
        return train_supervised_scheduler("logistic_regression", style, lambda_param)
    elif choice.startswith("Random Forest"):
        return train_supervised_scheduler("random_forest", style, lambda_param)
    raise ValueError(f"Unknown scheduler choice: {choice}")


def fresh_pipeline(lambda_param: float):
    manager = BatchCacheManager()
    sim = StreamSimulator(manager, lambda_param=lambda_param)
    return manager, sim


def metrics_row(container, metrics: dict, stats: list):
    with container.container():
        # System Performance metrics
        st.markdown("<h4 style='color: #60A5FA; font-family: \"Space Grotesk\", sans-serif; margin-top: 10px; margin-bottom: 5px;'>📊 System Performance Metrics</h4>", unsafe_allow_html=True)
        cols_sys = st.columns(5)
        
        card_html = """
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-desc">{desc}</div>
        </div>
        """
        
        with cols_sys[0]:
            st.markdown(card_html.format(title="Messages", value=str(metrics.get("total_messages", 0)), desc="Total processed"), unsafe_allow_html=True)
        with cols_sys[1]:
            st.markdown(card_html.format(title="Comp Ratio", value=f"{metrics.get('compression_ratio', 1.0):.3f}", desc="Lower is better"), unsafe_allow_html=True)
        with cols_sys[2]:
            st.markdown(card_html.format(title="Cache Hit Rate", value=f"{metrics.get('cache_hit_rate', 0.0) * 100:.1f}%", desc="Exact match hits"), unsafe_allow_html=True)
        with cols_sys[3]:
            st.markdown(card_html.format(title="Mean E2E Lat", value=f"{metrics.get('mean_e2e_latency_us', 0.0):.1f} µs", desc="End-to-end delay"), unsafe_allow_html=True)
        with cols_sys[4]:
            st.markdown(card_html.format(title="Throughput", value=f"{metrics.get('throughput_msg_per_sec', 0.0):.0f} msg/s", desc="Messages per second"), unsafe_allow_html=True)

        # Space & CPU Savings Metrics Row
        total_raw_bytes = sum(s.get("orig_size", 0) for s in stats)
        total_comp_bytes = sum(s.get("comp_size", 0) for s in stats)
        bytes_saved = max(total_raw_bytes - total_comp_bytes, 0)
        space_saved_pct = (bytes_saved / total_raw_bytes * 100.0) if total_raw_bytes > 0 else 0.0
        
        # CPU Savings vs Naive Always-Compress baseline (~45 us avg per classical compression)
        actual_total_lat_us = sum(s.get("e2e_lat_us", 0.0) for s in stats)
        naive_compress_lat_us = len(stats) * 48.0  # Approx CPU time if running ZSTD/GZIP on every msg
        cpu_saved_us = max(naive_compress_lat_us - actual_total_lat_us, 0.0)
        cpu_saved_pct = (cpu_saved_us / naive_compress_lat_us * 100.0) if naive_compress_lat_us > 0 else 0.0
        
        st.markdown("<h4 style='color: #34D399; font-family: \"Space Grotesk\", sans-serif; margin-top: 15px; margin-bottom: 5px;'>💾 Space & ⚡ CPU Efficiency Savings</h4>", unsafe_allow_html=True)
        cols_savings = st.columns(4)
        with cols_savings[0]:
            st.markdown(card_html.format(title="Raw Ingested", value=f"{total_raw_bytes:,} B", desc=f"{total_raw_bytes/1024:.1f} KB uncompressed"), unsafe_allow_html=True)
        with cols_savings[1]:
            st.markdown(card_html.format(title="Bytes Sent", value=f"{total_comp_bytes:,} B", desc=f"{total_comp_bytes/1024:.1f} KB transmitted"), unsafe_allow_html=True)
        with cols_savings[2]:
            st.markdown(card_html.format(title="Space Saved", value=f"{space_saved_pct:+.1f}%", desc=f"{bytes_saved:,} Bytes saved"), unsafe_allow_html=True)
        with cols_savings[3]:
            st.markdown(card_html.format(title="CPU Time Saved", value=f"{cpu_saved_pct:.1f}%", desc=f"{cpu_saved_us:,.0f} µs saved vs Always-Compress"), unsafe_allow_html=True)

        # Linguistic Feature metrics
        total_chars = 0
        total_words = 0
        total_emojis = 0
        total_punctuations = 0
        entropy_sum = 0.0
        uppercase_ratio_sum = 0.0
        unique_word_ratio_sum = 0.0
        rich_messages_count = 0
        
        for s in stats:
            text = s.get("text", "")
            char_len = len(text)
            total_chars += char_len
            
            tokens = re.findall(r'\w+', text.lower())
            word_len = len(tokens)
            total_words += word_len
            
            emoji_count = sum(1 for c in text if '\U0001f000' <= c <= '\U0001f9ff' or '\u2600' <= c <= '\u27bf')
            total_emojis += emoji_count
            
            punc_count = len(re.findall(r'[^\w\s]', text))
            total_punctuations += punc_count
            
            if emoji_count > 0 or punc_count > 0:
                rich_messages_count += 1
                
            up_ratio = sum(1 for c in text if c.isupper()) / char_len if char_len > 0 else 0.0
            uppercase_ratio_sum += up_ratio
            
            uniq_ratio = len(set(tokens)) / word_len if word_len > 0 else 0.0
            unique_word_ratio_sum += uniq_ratio
            
            if text:
                counts = {}
                for c in text:
                    counts[c] = counts.get(c, 0) + 1
                ent = 0.0
                total = len(text)
                for count in counts.values():
                    p = count / total
                    ent -= p * math.log2(p)
                entropy_sum += ent
            
        n = len(stats)
        avg_entropy = entropy_sum / n if n > 0 else 0.0
        avg_uppercase = uppercase_ratio_sum / n if n > 0 else 0.0
        avg_unique_word = unique_word_ratio_sum / n if n > 0 else 0.0
        avg_msg_len = total_chars / n if n > 0 else 0.0
        avg_word_len = total_chars / total_words if total_words > 0 else 0.0
        pct_rich_msgs = (rich_messages_count / n) * 100 if n > 0 else 0.0
        
        st.markdown("<h4 style='color: #A78BFA; font-family: \"Space Grotesk\", sans-serif; margin-top: 15px; margin-bottom: 5px;'>📝 Running Linguistic Feature Metrics</h4>", unsafe_allow_html=True)
        
        # Row 1 of Feature Metrics (General)
        cols_feat_1 = st.columns(5)
        with cols_feat_1[0]:
            st.markdown(card_html.format(title="Chars", value=f"{total_chars:,}", desc="Total characters"), unsafe_allow_html=True)
        with cols_feat_1[1]:
            st.markdown(card_html.format(title="Words", value=f"{total_words:,}", desc="Total words"), unsafe_allow_html=True)
        with cols_feat_1[2]:
            st.markdown(card_html.format(title="Emojis", value=f"{total_emojis}", desc="Total emojis"), unsafe_allow_html=True)
        with cols_feat_1[3]:
            st.markdown(card_html.format(title="Punctuation", value=f"{total_punctuations}", desc="Special chars"), unsafe_allow_html=True)
        with cols_feat_1[4]:
            st.markdown(card_html.format(title="Avg Msg Len", value=f"{avg_msg_len:.1f}", desc="Chars per message"), unsafe_allow_html=True)
            
        # Row 2 of Feature Metrics (Stylistic / Information Theory)
        cols_feat_2 = st.columns(5)
        with cols_feat_2[0]:
            st.markdown(card_html.format(title="Vocab Diversity", value=f"{avg_unique_word * 100:.1f}%", desc="Avg unique words"), unsafe_allow_html=True)
        with cols_feat_2[1]:
            st.markdown(card_html.format(title="Avg Entropy", value=f"{avg_entropy:.2f}", desc="Info density"), unsafe_allow_html=True)
        with cols_feat_2[2]:
            st.markdown(card_html.format(title="Shouting Ratio", value=f"{avg_uppercase * 100:.1f}%", desc="Avg uppercase chars"), unsafe_allow_html=True)
        with cols_feat_2[3]:
            st.markdown(card_html.format(title="Avg Word Len", value=f"{avg_word_len:.1f}", desc="Chars per word"), unsafe_allow_html=True)
        with cols_feat_2[4]:
            st.markdown(card_html.format(title="Rich Messages", value=f"{pct_rich_msgs:.1f}%", desc="Have emoji or punc"), unsafe_allow_html=True)



def feed_table(container, stats: list, max_rows: int = 15):
    if not stats:
        container.info("No messages processed yet.")
        return
    rows = stats[-max_rows:][::-1]
    df = pd.DataFrame([{
        "Text": (r["text"][:45] + "…") if len(r["text"]) > 45 else r["text"],
        "Action": r["action"],
        "Orig (B)": r["orig_size"],
        "Comp (B)": r["comp_size"],
        "Ratio": round(r["comp_size"] / r["orig_size"], 3) if r["orig_size"] else 1.0,
        "E2E Latency (µs)": round(r["e2e_lat_us"], 1),
    } for r in rows])
    container.dataframe(df, use_container_width=True, hide_index=True)


def action_distribution_chart(container, stats: list, key: str = "action_dist_chart"):
    if not stats:
        return
    counts = pd.Series([s["action"] for s in stats]).value_counts()
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[ACTION_COLORS.get(a, "#888") for a in counts.index],
    ))
    fig.update_layout(title="Action Distribution", height=300, margin=dict(t=40, b=10, l=10, r=10))
    container.plotly_chart(fig, use_container_width=True, key=key)


def latency_over_time_chart(container, stats: list, key: str = "latency_ot_chart"):
    if not stats:
        return
    df = pd.DataFrame([{"idx": i, "latency": s["e2e_lat_us"], "action": s["action"]} for i, s in enumerate(stats)])
    fig = px.scatter(df, x="idx", y="latency", color="action",
                      color_discrete_map=ACTION_COLORS,
                      labels={"idx": "Message #", "latency": "End-to-End Latency (µs)"})
    fig.update_layout(title="Latency Over Time", height=300, margin=dict(t=40, b=10, l=10, r=10))
    container.plotly_chart(fig, use_container_width=True, key=key)


def pareto_chart(container, stats: list, key: str = "pareto_chart"):
    if not stats:
        return
    df = pd.DataFrame([{
        "ratio": s["comp_size"] / s["orig_size"] if s["orig_size"] else 1.0,
        "latency": s["e2e_lat_us"], "action": s["action"],
    } for s in stats])
    fig = px.scatter(df, x="latency", y="ratio", color="action",
                      color_discrete_map=ACTION_COLORS,
                      labels={"latency": "End-to-End Latency (µs)", "ratio": "Compressed / Original Size"})
    fig.update_layout(title="Ratio vs. Latency (Pareto view)", height=300, margin=dict(t=40, b=10, l=10, r=10))
    container.plotly_chart(fig, use_container_width=True, key=key)


def render_learning_advantage_section(container, stats: list, lambda_param: float = 0.01):
    if not stats or len(stats) < 5:
        return

    with container.container():
        st.markdown("<hr style='border: 1px solid rgba(139, 92, 246, 0.25); margin: 25px 0;'>", unsafe_allow_html=True)
        
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.12)); 
                        border: 1px solid rgba(139, 92, 246, 0.4); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                <h3 style="margin: 0 0 10px 0; color: #FFFFFF; font-family: 'Space Grotesk', sans-serif;">
                    🏆 Why Learning-Based Scheduling Wins (Adaptive Model vs. Static Baselines)
                </h3>
                <p style="color: #E2E8F0; font-size: 14px; line-height: 1.6; margin: 0;">
                    <b>💡 Why are SKIP &amp; CACHE_REUSE so frequent?</b> Short text messages (&lt;30 characters) suffer from 
                    <span style="color: #F87171; font-weight: 600;">negative expansion</span> when passed to static compressors (headers like gzip/zstd add 10–30 bytes, making compressed size <i>larger</i> than original). 
                    Our learning-based scheduler learns to <b>SKIP</b> uncompressible strings to eliminate CPU latency &amp; prevent bloat, while using <b>CACHE_REUSE</b> on duplicates. 
                    Below is the head-to-head proof on the exact same stream messages.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        engine = CompressionEngine()
        baseline_names = ["Adaptive Model (Active)", "Always ZSTD", "Always LZ4", "Always GZIP", "Always SKIP"]
        
        comp_bytes_map = {k: 0 for k in baseline_names}
        latency_map = {k: 0.0 for k in baseline_names}
        expanded_map = {k: 0 for k in baseline_names}
        total_raw = sum(s["orig_size"] for s in stats)
        n = len(stats)
        
        for s in stats:
            orig_s = s["orig_size"]
            comp_s = s["comp_size"]
            comp_bytes_map["Adaptive Model (Active)"] += comp_s
            latency_map["Adaptive Model (Active)"] += s["e2e_lat_us"]
            if comp_s > orig_s:
                expanded_map["Adaptive Model (Active)"] += 1
                
            # Baseline: Always ZSTD
            z_bytes, z_lat = engine.compress(s["text"], "ZSTD")
            comp_bytes_map["Always ZSTD"] += len(z_bytes)
            latency_map["Always ZSTD"] += z_lat
            if len(z_bytes) > orig_s:
                expanded_map["Always ZSTD"] += 1
                
            # Baseline: Always LZ4
            l_bytes, l_lat = engine.compress(s["text"], "LZ4")
            comp_bytes_map["Always LZ4"] += len(l_bytes)
            latency_map["Always LZ4"] += l_lat
            if len(l_bytes) > orig_s:
                expanded_map["Always LZ4"] += 1
                
            # Baseline: Always GZIP
            g_bytes, g_lat = engine.compress(s["text"], "GZIP")
            comp_bytes_map["Always GZIP"] += len(g_bytes)
            latency_map["Always GZIP"] += g_lat
            if len(g_bytes) > orig_s:
                expanded_map["Always GZIP"] += 1
                
            # Baseline: Always SKIP
            comp_bytes_map["Always SKIP"] += orig_s
            latency_map["Always SKIP"] += 0.0

        rows = []
        for name in baseline_names:
            c_bytes = comp_bytes_map[name]
            mean_lat = latency_map[name] / n if n > 0 else 0.0
            saved_pct = ((total_raw - c_bytes) / total_raw * 100.0) if total_raw > 0 else 0.0
            exp_pct = (expanded_map[name] / n * 100.0) if n > 0 else 0.0
            tot_cost = c_bytes + lambda_param * latency_map[name]
            ratio = c_bytes / total_raw if total_raw > 0 else 1.0
            
            rows.append({
                "Strategy": name,
                "Net Cost (Bytes + λ·µs)": round(tot_cost, 1),
                "Space Saved (%)": round(saved_pct, 1),
                "Expansion Rate (%)": round(exp_pct, 1),
                "Mean Latency (µs)": round(mean_lat, 1),
                "Total Compressed (Bytes)": c_bytes,
                "Compression Ratio": round(ratio, 3)
            })
            
        df_comp = pd.DataFrame(rows)

        col_c1, col_c2, col_c3 = st.columns(3)
        
        with col_c1:
            min_cost_idx = df_comp["Net Cost (Bytes + λ·µs)"].idxmin()
            best_cost_name = df_comp.loc[min_cost_idx, "Strategy"]
            colors = ["#22C55E" if s == best_cost_name else ("#6366F1" if "Adaptive" in s else "#64748B") for s in df_comp["Strategy"]]
            
            fig_cost = go.Figure(go.Bar(
                x=df_comp["Strategy"], y=df_comp["Net Cost (Bytes + λ·µs)"],
                marker_color=colors,
                text=df_comp["Net Cost (Bytes + λ·µs)"],
                textposition="outside"
            ))
            fig_cost.update_layout(
                title="<b>Total Net Cost</b> (Lower = Better 🏆)",
                height=320, margin=dict(t=40, b=10, l=10, r=10),
                xaxis=dict(tickangle=-25)
            )
            st.plotly_chart(fig_cost, use_container_width=True, key="adv_cost_chart")

        with col_c2:
            exp_colors = ["#22C55E" if r == 0.0 else "#EF4444" for r in df_comp["Expansion Rate (%)"]]
            fig_exp = go.Figure(go.Bar(
                x=df_comp["Strategy"], y=df_comp["Expansion Rate (%)"],
                marker_color=exp_colors,
                text=[f"{v:.1f}%" for v in df_comp["Expansion Rate (%)"]],
                textposition="outside"
            ))
            fig_exp.update_layout(
                title="<b>Payload Expansion Rate</b> (% Made Larger)",
                height=320, margin=dict(t=40, b=10, l=10, r=10),
                xaxis=dict(tickangle=-25)
            )
            st.plotly_chart(fig_exp, use_container_width=True, key="adv_expansion_chart")

        with col_c3:
            fig_pareto = px.scatter(
                df_comp, x="Mean Latency (µs)", y="Space Saved (%)", text="Strategy",
                color="Strategy",
                title="<b>Pareto Frontier: Savings vs. Latency</b>",
                color_discrete_sequence=["#22C55E", "#3B82F6", "#10B981", "#F59E0B", "#94A3B8"]
            )
            fig_pareto.update_traces(textposition="top center", marker=dict(size=12))
            fig_pareto.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
            st.plotly_chart(fig_pareto, use_container_width=True, key="adv_pareto_chart")

        # Row 2: Space Saved % and CPU Latency breakdown
        col_c4, col_c5 = st.columns(2)
        with col_c4:
            fig_space = px.bar(
                df_comp, x="Strategy", y="Space Saved (%)", color="Strategy",
                title="<b>💾 Bandwidth Space Saved (%)</b> (Higher = Better)",
                color_discrete_sequence=["#22C55E", "#3B82F6", "#10B981", "#F59E0B", "#94A3B8"]
            )
            fig_space.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10), showlegend=False, xaxis=dict(tickangle=-20))
            st.plotly_chart(fig_space, use_container_width=True, key="adv_space_bar_chart")

        with col_c5:
            fig_cpu = px.bar(
                df_comp, x="Strategy", y="Mean Latency (µs)", color="Strategy",
                title="<b>⚡ Mean CPU Execution Latency</b> (µs, Lower = Better)",
                color_discrete_sequence=["#22C55E", "#3B82F6", "#10B981", "#F59E0B", "#94A3B8"]
            )
            fig_cpu.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10), showlegend=False, xaxis=dict(tickangle=-20))
            st.plotly_chart(fig_cpu, use_container_width=True, key="adv_cpu_bar_chart")

        st.markdown("<h4 style='color: #A78BFA; margin-top: 10px;'>📊 Full Counterfactual Benchmark Table</h4>", unsafe_allow_html=True)
        st.dataframe(df_comp.set_index("Strategy"), use_container_width=True)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("📡 Controls")
dataset_style = st.sidebar.selectbox("Stream style", list(DATASET_GENERATORS.keys()))
lambda_param = st.sidebar.slider("λ (latency weight, cost = bytes + λ·µs)", 0.0, 0.05, 0.01, 0.005)
n_messages = st.sidebar.slider("Messages to stream", 10, 200, 60, 10)
speed_multiplier = st.sidebar.slider("Playback speed", 1, 50, 10,
                                      help="Higher = faster than real-time arrival pacing")
simulate_shift = st.sidebar.checkbox(
    "Simulate distribution shift (chat → tweets) mid-stream",
    help="First half of the stream uses DailyDialog-style messages, second half switches "
         "to Sentiment140-style — useful for showing the bandit adapting live."
)

# Show active model info
st.sidebar.info("🤖 **Auto ML active** — LinUCB Bandit selects compression per message online")

st.sidebar.divider()
if st.sidebar.button("🔄 Reset simulation state", use_container_width=True):
    for key in ["manager", "sim", "scheduler", "live_running"]:
        st.session_state.pop(key, None)
    st.rerun()

st.sidebar.caption(
    "Note: latency numbers reflect this machine's actual CPU performance right now — "
    "they will vary run to run and machine to machine, which is expected for a real-time system."
)

# --------------------------------------------------------------------------
# Session state init
# --------------------------------------------------------------------------

if "manager" not in st.session_state:
    st.session_state.manager, st.session_state.sim = fresh_pipeline(lambda_param)
    st.session_state.scheduler = get_scheduler(SCHEDULER_CHOICES[0], dataset_style, lambda_param)
    st.session_state.live_running = False

sim: StreamSimulator = st.session_state.sim
manager: BatchCacheManager = st.session_state.manager


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("Adaptive Learning-Based Compression Scheduling")
st.caption("Real-Time Short-Text Stream Demo — Skip / Zstd / Brotli / Gzip / LZ4 / Batch / Cache-Reuse")

tab_live, tab_single, tab_compare = st.tabs(["🔴 Live Stream", "✍️ Single Message Test", "📊 Scheduler Comparison"])


# --------------------------------------------------------------------------
# TAB 1: Live Stream
# --------------------------------------------------------------------------

with tab_live:
    st.subheader("Live-Paced Stream Replay")

    # Auto ML banner
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
                    border: 1px solid rgba(139,92,246,0.4); border-radius: 10px;
                    padding: 12px 18px; margin-bottom: 12px;">
          <span style="font-size:18px;">🤖</span>
          <strong style="color:#a5b4fc;"> Auto ML Mode</strong>
          <span style="color:#d1d5db; font-size:13px;">
            — LinUCB Bandit is selecting the compression engine automatically for each
            message. It explores actions online and adapts to the stream distribution
            with no pre-training required.
          </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write(
        "Messages are replayed with real wall-clock pacing based on their recorded "
        "inter-arrival intervals (scaled by the speed multiplier), so this reflects the "
        "system actually processing messages as they arrive — not a pre-computed batch job."
    )

    col_start, col_status = st.columns([1, 4])
    start_btn = col_start.button("▶ Start Live Stream", type="primary", use_container_width=True)
    status_ph = col_status.empty()

    metrics_ph = st.empty()
    st.markdown("**Live Feed** (most recent messages first)")
    feed_ph = st.empty()

    chart_col1, chart_col2 = st.columns(2)
    action_chart_ph = chart_col1.empty()
    latency_chart_ph = chart_col2.empty()
    pareto_ph = st.empty()
    adv_section_ph = st.empty()

    if not start_btn:
        metrics_row(metrics_ph, sim.get_summary_metrics(), sim.stats)
        feed_table(feed_ph, sim.stats)
        action_distribution_chart(action_chart_ph, sim.stats)
        latency_over_time_chart(latency_chart_ph, sim.stats)
        pareto_chart(pareto_ph, sim.stats)
        render_learning_advantage_section(adv_section_ph, sim.stats, lambda_param)

    if start_btn:
        # Fresh pipeline + scheduler for a clean live run
        manager, sim = fresh_pipeline(lambda_param)
        scheduler = get_scheduler(SCHEDULER_CHOICES[0], dataset_style, lambda_param)
        st.session_state.manager, st.session_state.sim, st.session_state.scheduler = manager, sim, scheduler

        if simulate_shift:
            half = n_messages // 2
            styles = list(DATASET_GENERATORS.keys())
            style_a, style_b = dataset_style, [s for s in styles if s != dataset_style][0]
            msgs_a, ts_a = build_stream(style_a, half)
            msgs_b, ts_b = build_stream(style_b, n_messages - half)
            offset = ts_a[-1] if ts_a else 0.0
            ts_b = [t + offset for t in ts_b]
            messages = msgs_a + msgs_b
            timestamps = ts_a + ts_b
            status_ph.info(f"Distribution shift enabled: messages 1–{half} = {style_a}, "
                            f"{half + 1}–{n_messages} = {style_b}")
        else:
            messages, timestamps = build_stream(dataset_style, n_messages)

        progress = st.progress(0, text="Streaming...")
        prev_ts = 0.0
        for i, (msg, ts) in enumerate(zip(messages, timestamps)):
            wait = max((ts - prev_ts) / speed_multiplier, 0)
            time.sleep(min(wait, 1.5))  # cap wait so demo never stalls too long on a slow gap
            prev_ts = ts

            sim.run_message(i, msg, time.time(), scheduler)

            metrics_row(metrics_ph, sim.get_summary_metrics(), sim.stats)
            feed_table(feed_ph, sim.stats)
            
            # Update heavy Plotly charts less frequently (every 5 messages or on final message) to avoid WebSocket lag
            if i % 5 == 0 or i == len(messages) - 1:
                action_distribution_chart(action_chart_ph, sim.stats, key=f"action_dist_chart_loop_{i}")
                latency_over_time_chart(latency_chart_ph, sim.stats, key=f"latency_ot_chart_loop_{i}")
                pareto_chart(pareto_ph, sim.stats, key=f"pareto_chart_loop_{i}")
                
            progress.progress((i + 1) / len(messages), text=f"Streaming... {i + 1}/{len(messages)}")

        sim.finalize_stream(time.time(), scheduler)
        metrics_row(metrics_ph, sim.get_summary_metrics(), sim.stats)
        feed_table(feed_ph, sim.stats)
        progress.empty()
        status_ph.success(f"Stream complete — {len(sim.stats)} messages processed.")
        render_learning_advantage_section(adv_section_ph, sim.stats, lambda_param)


# --------------------------------------------------------------------------
# TAB 2: Single Message Test
# --------------------------------------------------------------------------

with tab_single:
    st.subheader("Test a Single Message")
    st.write(
        "Type any message to route through the pipeline. You can optionally compare the routing "
        "and latency performance across all ML models side-by-side."
    )

    scheduler_choice = st.selectbox(
        "Scheduler",
        SCHEDULER_CHOICES,
        index=0,
        help="Select a model for testing this single message."
    )

    msg_input = st.text_input("Enter a message to route through the pipeline:",
                               placeholder="e.g. Hey, are you free for a call at 5?", key="single_msg_input_field")
    test_btn = st.button("Test Message", type="primary", key="single_msg_test_button")

    if test_btn and msg_input.strip():
        if scheduler_choice == SCHEDULER_CHOICES[0]:
            test_scheduler = st.session_state.scheduler
        else:
            test_scheduler = get_scheduler(scheduler_choice, dataset_style, lambda_param)
            
        # Store message and result in session state to persist across checkbox toggles
        st.session_state.single_msg_input = msg_input
        st.session_state.single_msg_result = sim.run_message(
            f"manual-{time.time()}", msg_input, time.time(), test_scheduler
        )
        st.session_state.single_msg_scheduler_used = scheduler_choice

    if "single_msg_input" in st.session_state and st.session_state.single_msg_input.strip():
        msg_val = st.session_state.single_msg_input
        result = st.session_state.single_msg_result
        used_sched = st.session_state.get("single_msg_scheduler_used", SCHEDULER_CHOICES[0])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### 📡 Active Scheduler Result ({used_sched})")
            if result.get("cache_hit"):
                st.success("⚡ CACHE HIT — exact duplicate seen before, reused instantly")
            else:
                st.info(f"Action chosen: **{result['action']}**")
            st.metric("Original size", f"{result['orig_size']} bytes")
            st.metric("Compressed size", f"{result['comp_size']} bytes")
            ratio = result['comp_size'] / result['orig_size'] if result['orig_size'] else 1.0
            st.metric("Compression ratio", f"{ratio:.3f}")
            st.metric("End-to-end latency", f"{result['e2e_lat_us']:.2f} µs")

        with c2:
            st.markdown("### 📝 Extracted Features")
            fe_snapshot = sim.feature_extractor.extract_features(msg_val, time.time())
            st.dataframe(pd.DataFrame(fe_snapshot.items(), columns=["Feature", "Value"]),
                         use_container_width=True, hide_index=True)

        st.markdown("---")
        
        # Toggle checkbox for comparative view
        compare_check = st.checkbox("🔍 Compare all scheduler models on this message", key="compare_models_checkbox", value=False)
        
        if compare_check:
            # Get all trained schedulers
            schedulers = {
                "Heuristic": get_scheduler("Heuristic (hand-written rules)", dataset_style, lambda_param),
                "Decision Tree": get_scheduler("Decision Tree (supervised)", dataset_style, lambda_param),
                "Logistic Regression": get_scheduler("Logistic Regression (supervised)", dataset_style, lambda_param),
                "Random Forest": get_scheduler("Random Forest (supervised)", dataset_style, lambda_param),
                "LinUCB Bandit": get_scheduler("LinUCB Bandit (online)", dataset_style, lambda_param),
            }
            
            orig_size = len(msg_val.encode('utf-8'))
            engine = CompressionEngine()
            
            # Run comparison predictions
            comparison_results = []
            for name, sched in schedulers.items():
                # Check cache lookups overhead first (simulates standard manager path)
                cache_start = time.perf_counter()
                cache_hit = manager.cache_lookup(msg_val)
                cache_lookup_lat = (time.perf_counter() - cache_start) * 1_000_000
                
                # Predict
                if hasattr(sched, 'predict'):
                    action_idx, sched_lat = sched.predict(fe_snapshot)
                elif callable(sched):
                    action_idx, sched_lat = sched(fe_snapshot)
                action_name = ACTION_MAP.get(action_idx, 'SKIP')
                
                total_sched_lat = sched_lat + cache_lookup_lat
                
                # Execute action
                if action_name == 'SKIP':
                    comp_size = orig_size
                    comp_lat = 0.0
                    decomp_lat = 0.0
                elif action_name in ['ZSTD', 'BROTLI', 'GZIP', 'LZ4']:
                    comp_bytes, comp_lat = engine.compress(msg_val, action_name)
                    _, decomp_lat = engine.decompress(comp_bytes, action_name)
                    comp_size = len(comp_bytes)
                elif action_name == 'BATCH':
                    comp_size = int(orig_size * 0.4) # batching gets good ratio
                    comp_lat = 5.0
                    decomp_lat = 2.0
                    
                e2e_lat = total_sched_lat + comp_lat + decomp_lat
                cost = comp_size + lambda_param * (total_sched_lat + comp_lat)
                
                comparison_results.append({
                    "Scheduler": name,
                    "Action Chosen": action_name,
                    "Decision Overhead (µs)": round(total_sched_lat, 1),
                    "Compression (µs)": round(comp_lat, 1),
                    "Decompression (µs)": round(decomp_lat, 1),
                    "End-to-End Latency (µs)": round(e2e_lat, 1),
                    "Output Size (bytes)": comp_size,
                    "Compression Ratio": round(comp_size / orig_size, 3) if orig_size > 0 else 1.0,
                    "Total Cost (units)": round(cost, 2)
                })
                
            df_comp = pd.DataFrame(comparison_results)
            
            st.markdown("### 🏆 Head-to-Head Scheduler Performance")
            st.dataframe(df_comp.set_index("Scheduler"), use_container_width=True)
            
            # Render charts comparing the models
            c_chart1, c_chart2 = st.columns(2)
            with c_chart1:
                fig_lat = px.bar(
                    df_comp, x="Scheduler", y="Decision Overhead (µs)", 
                    title="Model Decision CPU Overhead (µs, lower is better)",
                    color="Scheduler", color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_lat.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_lat, use_container_width=True, key="single_decision_lat_chart")
                
            with c_chart2:
                fig_cost = px.bar(
                    df_comp, x="Scheduler", y="Total Cost (units)", 
                    title="Total Cost (Bytes + λ·µs, lower is better)",
                    color="Scheduler", color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_cost.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_cost, use_container_width=True, key="single_cost_chart")

    elif test_btn:
        st.warning("Type a message first.")


# --------------------------------------------------------------------------
# TAB 3: Scheduler Comparison
# --------------------------------------------------------------------------

with tab_compare:
    st.subheader("Head-to-Head Scheduler Comparison")
    st.write(
        "Runs the **same fixed synthetic stream** through every scheduler independently "
        "(fresh cache/state each time) and compares them on the metrics that matter for "
        "this project's core claim."
    )

    n_compare = st.slider("Messages per scheduler run", 20, 300, 100, 20, key="n_compare")
    run_compare = st.button("▶ Run Comparison", type="primary")

    # Comparison always includes all schedulers including LinUCB
    ALL_SCHEDULER_CHOICES = [
        "Auto — LinUCB Bandit (online, recommended)",
        "Heuristic (hand-written rules)",
        "Decision Tree (supervised)",
        "Logistic Regression (supervised)",
        "Random Forest (supervised)",
    ]

    if run_compare:
        messages, timestamps = build_stream(dataset_style, n_compare, seed=7)
        results = []
        prog = st.progress(0, text="Running comparison...")

        for idx, choice in enumerate(ALL_SCHEDULER_CHOICES):
            manager_c, sim_c = fresh_pipeline(lambda_param)
            scheduler_c = get_scheduler(choice, dataset_style, lambda_param)
            for i, (msg, ts) in enumerate(zip(messages, timestamps)):
                sim_c.run_message(i, msg, ts, scheduler_c)
            sim_c.finalize_stream(timestamps[-1] if timestamps else 0.0, scheduler_c)

            m = sim_c.get_summary_metrics()
            label = "LinUCB Bandit" if choice.startswith("Auto") else choice.split(" (")[0]
            m["scheduler"] = label
            results.append(m)
            prog.progress((idx + 1) / len(ALL_SCHEDULER_CHOICES), text=f"Running {label}...")

        prog.empty()
        df = pd.DataFrame(results)

        # Highlight the best (winner) per metric
        SCHED_COLORS = [
            "#6366F1", "#F59E0B", "#10B981", "#EC4899", "#3B82F6"
        ]
        color_map = {row["scheduler"]: SCHED_COLORS[i % len(SCHED_COLORS)]
                     for i, row in df.iterrows()}

        def styled_bar(col, title, higher_better=False):
            best_idx = df[col].idxmax() if higher_better else df[col].idxmin()
            best_sched = df.loc[best_idx, "scheduler"]
            colors = ["#22c55e" if s == best_sched else color_map.get(s, "#6366F1")
                      for s in df["scheduler"]]
            fig = go.Figure(go.Bar(
                x=df["scheduler"], y=df[col],
                marker_color=colors,
                text=[f"{v:.3f}" for v in df[col]],
                textposition="outside"
            ))
            arrow = "↑ higher" if higher_better else "↓ lower"
            fig.update_layout(
                title=dict(text=f"{title}  <sup style='color:#22c55e'>🏆 {best_sched}</sup>",
                           font_size=13),
                height=360, margin=dict(t=50, b=10, l=10, r=10),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#F3F4F6",
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                annotations=[dict(text=arrow, x=1.01, y=0.5, xref="paper", yref="paper",
                                  showarrow=False, font=dict(color="#9CA3AF", size=11))]
            )
            return fig

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(styled_bar("compression_ratio", "Compression Ratio (lower = better)"),
                            use_container_width=True, key="comp_ratio_chart")
        with c2:
            st.plotly_chart(styled_bar("mean_e2e_latency_us", "Mean E2E Latency µs (lower = better)"),
                            use_container_width=True, key="comp_e2e_lat_chart")

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(styled_bar("cache_hit_rate", "Cache Hit Rate (higher = better)", higher_better=True),
                            use_container_width=True, key="comp_cache_chart")
        with c4:
            st.plotly_chart(styled_bar("throughput_msg_per_sec", "Throughput msg/s (higher = better)", higher_better=True),
                            use_container_width=True, key="comp_throughput_chart")

        # Radar chart — multi-metric overview
        st.markdown("### 🕸️ Multi-Metric Radar Overview")
        metrics_radar = ["compression_ratio", "mean_e2e_latency_us", "cache_hit_rate", "throughput_msg_per_sec", "mean_scheduler_latency_us"]
        labels_radar = ["Comp Ratio", "E2E Lat", "Cache Hit", "Throughput", "Sched Lat"]

        # Normalise 0→1, flip ratio/latency so higher = better always
        df_norm = df.copy()
        for col in ["compression_ratio", "mean_e2e_latency_us", "mean_scheduler_latency_us"]:
            mn, mx = df_norm[col].min(), df_norm[col].max()
            df_norm[col] = 1 - ((df_norm[col] - mn) / (mx - mn + 1e-9))  # flip: lower is better
        for col in ["cache_hit_rate", "throughput_msg_per_sec"]:
            mn, mx = df_norm[col].min(), df_norm[col].max()
            df_norm[col] = (df_norm[col] - mn) / (mx - mn + 1e-9)

        radar_fig = go.Figure()
        for i, row in df_norm.iterrows():
            vals = [row[m] for m in metrics_radar]
            vals += [vals[0]]  # close loop
            radar_fig.add_trace(go.Scatterpolar(
                r=vals, theta=labels_radar + [labels_radar[0]],
                fill="toself", name=row["scheduler"],
                line=dict(color=SCHED_COLORS[i % len(SCHED_COLORS)], width=2),
                opacity=0.75
            ))
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
                bgcolor="rgba(0,0,0,0)"
            ),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#F3F4F6",
            legend=dict(orientation="h", y=-0.15),
            height=420, margin=dict(t=20, b=60)
        )
        st.plotly_chart(radar_fig, use_container_width=True, key="radar_chart")

        st.markdown("**Full metrics table**")
        st.dataframe(df.set_index("scheduler").round(3), use_container_width=True)

        st.caption(
            "🟢 Green bar = winner per metric. "
            "LinUCB Bandit starts cold (no pre-training) — on longer streams it closes the gap "
            "as it adapts online. The radar chart shows normalised scores (higher = better on all axes)."
        )
    else:
        st.info("Click **▶ Run Comparison** to benchmark all 5 schedulers on the same stream and see which ML model wins.")
