# Adaptive Learning-Based Compression Scheduling for Real-Time Short-Text Streams

This repository implements the research prototype for **Adaptive Learning-Based Compression Scheduling**. It targets real-time short-text streams (such as chat, tweets, and SMS) and learns an optimal per-message decision policy to maximize compression savings while minimizing latency overhead.

> **Scope Note**: Heavy LLM-based neural compression codecs (e.g. LLMZip / FineZip) were intentionally scoped out to preserve real-time streaming throughput. The action space is strictly composed of classical codecs, batching, and skip options.

---

## Core Pipeline Architecture

```
Incoming Stream Message
         │
         ▼
[ Exact-Match Cache Lookup (SHA-256) ]
   ├── (Hit)  ──► Decompress / Reuse Cached Payload ──► Output (Near-Zero Latency)
   └── (Miss) ──► Continue
         │
         ▼
[ 9-D Feature Extraction ] ──► (Length, Entropy, Repetition, Arrival Rate, Punctuation/Emoji Ratios)
         │
         ▼
[ Scheduler Model Policy ] ──► [ Heuristic | Logistic | Decision Tree | Random Forest | Gradient Boosting | LinUCB Bandit ]
         │
         ▼
Selected Action Execution:
 ├── SKIP   ──► Pass-through raw text
 ├── ZSTD   ──► Single Zstandard compression
 ├── BROTLI ──► Single Brotli compression
 ├── GZIP   ──► Single Gzip compression
 ├── LZ4    ──► Single LZ4 compression
 └── BATCH  ──► Queue message; flush on size limit (10) or timeout (500ms) with 4-byte length-prefix framing
```

> **Data Preparation vs Runtime**: Message cleaning (e.g. Twitter regex cleanup) is performed offline during dataset preparation in `data_loader.py`. Online tokenization happens inside the Feature Extractor at runtime. Exact duplicate detection is handled dynamically by the SHA-256 Cache Lookup.

---

## Core Architecture & Cost Function

The framework consists of five main modules:
1. **Classical Compression Engines** ([`src/engine.py`](file:///c:/Users/rithv/Downloads/Adaptive%20Learning-Based%20Compression/src/engine.py)): Interfaces for Zstandard (Zstd), Brotli, Gzip, and LZ4, measuring compression and decompression times strictly in microseconds ($\mu s$).
2. **Feature Extractor** ([`src/features.py`](file:///c:/Users/rithv/Downloads/Adaptive%20Learning-Based%20Compression/src/features.py)): Computes per-message character/word length, Shannon entropy, vocabulary diversity, text complexity (uppercase, punctuation, emoji ratios), and real-time arrival rate. It also computes a MinHash Jaccard similarity score against a rolling window of recent messages. (Note: Language detection was excluded from the online feature vector to minimize inference latency).
3. **Batch & Cache Manager** ([`src/manager.py`](file:///c:/Users/rithv/Downloads/Adaptive%20Learning-Based%20Compression/src/manager.py)):
   - **Exact-Match Cache**: Keyed by SHA-256 hashes of the raw message text. Bypasses feature extraction and the scheduler on cache hits to guarantee lossless duplication removal with near-zero latency.
   - **Batcher Queue**: Bounded queue aggregating messages scheduled for batching. Uses **4-byte big-endian length-prefix binary framing** to guarantee round-trip losslessness even when messages contain newlines. Flushes when queue size reaches limit (10) or stream arrival timestamp exceeds timeout (500ms).
4. **Contextual Bandit Scheduler** ([`src/bandit.py`](file:///c:/Users/rithv/Downloads/Adaptive%20Learning-Based%20Compression/src/bandit.py)): An online reinforcement learning scheduler utilizing a **LinUCB (Linear Upper Confidence Bound)** policy with optional cost-aware exploration bounds. The bandit predicts actions from `[SKIP, ZSTD, BROTLI, GZIP, LZ4, BATCH]` and updates parameters online based on reward $R = -\text{Cost}$, where:
   $$\text{Cost} = \text{Size (bytes)} + \lambda \times \text{Latency (\mu s)}$$
   Varying $\lambda$ constructs different operating points on the Pareto frontier.
5. **Supervised Schedulers & Label Generator** ([`src/scheduler.py`](file:///c:/Users/rithv/Downloads/Adaptive%20Learning-Based%20Compression/src/scheduler.py)):
   - **Offline Label Generator**: Grid-searches optimal actions on historical data using microsecond-level costs. BATCH cost is evaluated using a single global approximation (average size share & compression latency from the first 10 messages + 50ms queue penalty).
   - **Supervised Classifiers**: Logistic Regression, Decision Tree (max_depth=5), Random Forest, and Gradient Boosting.
   - **Heuristic Scheduler**: Rule-based baseline.

---

## Installation & Setup

1. **Python version**: Python 3.10+ is recommended.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Run Guide

### 1. Run Automated Unit Tests
Verify mathematical correctness, microsecond unit alignment, timeout flushing, length-prefixed batch losslessness, and online bandit updates:
```bash
python -m unittest discover -s src -p "test_*.py"
```

### 2. Prepare Datasets
Download and clean the Sentiment140, DailyDialog, and NUS SMS datasets (uses `random.seed(42)` sampling to avoid temporal bias, with synthetic fallback data generation for offline environments):
```bash
python -m src.data_loader
```

### 3. Run Benchmarking Experiments
Execute all schedulers across multiple values of $\lambda$:
```bash
python -m experiments.run_experiments
```

### 4. Plot Results
Generate Pareto Frontier curves, scheduler decision latencies, and feature importances:
```bash
python -m experiments.plot_results
```

---

## Evaluation & Methodology Notes

- **Simulated Latency**: End-to-end latencies are computed via discrete-event simulation over recorded arrival timestamps ($queue\_wait\_us = (current\_time - timestamp) \times 10^6$), incorporating exact measured microsecond CPU execution times for compression/decompression and scheduler inference.
- **Losslessness Verification**: Every compressor action (including batch decompression) explicitly verifies byte-for-byte losslessness against original input strings before output.
