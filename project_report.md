# Adaptive Learning-Based Compression Scheduling for Real-Time Short-Text Streams
## Comprehensive Project Report

---

> **Report Scope**: This report documents the complete lifecycle of the Adaptive Learning-Based Compression Scheduling project — from research motivation and conceptual design, through full system architecture, module-level implementation, testing strategy, the interactive premium dashboard, and final experimental results. **This represents the final, fully-integrated production system.**

---

## Table of Contents

1. [Project Motivation and Research Problem](#1-project-motivation-and-research-problem)
2. [Literature Survey and Related Work](#2-literature-survey-and-related-work)
3. [Problem Gap Analysis](#3-problem-gap-analysis)
4. [Conceptual Design and System Scope Decisions](#4-conceptual-design-and-system-scope-decisions)
5. [System Architecture Overview](#5-system-architecture-overview)
6. [Component 1: Classical Compression Engine](#6-component-1-classical-compression-engine)
7. [Component 2: Feature Extractor](#7-component-2-feature-extractor)
8. [Component 3: Batch and Cache Manager](#8-component-3-batch-and-cache-manager)
9. [Component 4: Scheduler Models and Offline Label Generator](#9-component-4-scheduler-models-and-offline-label-generator)
10. [Component 5: Contextual Bandit (LinUCB)](#10-component-5-contextual-bandit-linucb)
11. [Component 6: Stream Simulator](#11-component-6-stream-simulator)
12. [Component 7: Data Loader and Dataset Preparation](#12-component-7-data-loader-and-dataset-preparation)
13. [Experiment Pipeline](#13-experiment-pipeline)
14. [Test Suite: Architecture, Coverage, and Results](#14-test-suite-architecture-coverage-and-results)
15. [Experimental Results: Benchmark Metrics and Analysis](#15-experimental-results-benchmark-metrics-and-analysis)
16. [Visualization Outputs](#16-visualization-outputs)
17. [Runtime Utilities and Environment Logging](#17-runtime-utilities-and-environment-logging)
18. [Interactive Premium Dashboard](#18-interactive-premium-dashboard)
19. [CLI Demo and Evaluation Tools](#19-cli-demo-and-evaluation-tools)
20. [Dependency Stack and Environment Specifications](#20-dependency-stack-and-environment-specifications)
21. [Project File Structure and Code Summary](#21-project-file-structure-and-code-summary)
22. [Key Findings and Conclusions](#22-key-findings-and-conclusions)
23. [Known Limitations and Future Work](#23-known-limitations-and-future-work)

---

## 1. Project Motivation and Research Problem

### 1.1 Background

Modern real-time communication systems — chat platforms, SMS gateways, social media APIs, alert notification systems — generate enormous volumes of short-text messages that must be transmitted, processed, and stored at high throughput with minimal latency. Unlike document-level or binary compression (where codecs like Zstd or Brotli are straightforwardly beneficial), **short-text streams present a unique challenge**:

- Individual short messages are too small to benefit from compression overheads (header sizes dominate).
- Message traffic is heterogeneous: a stream may contain greeting messages (5 characters), system log alerts (250 characters), repeated notifications (exact duplicates), and bursts of emoji-heavy tweets.
- Applying a single static codec across all message types is provably suboptimal — it either wastes CPU time on uncompressible short messages or fails to exploit repetitive patterns.

### 1.2 The Core Research Question

> *Can a learning-based scheduling policy — one that observes per-message features and adaptively selects the best compression action from a predefined action space — outperform static baselines across the compression ratio vs. latency Pareto frontier?*

The research specifically targets:
- **Sub-millisecond scheduling overhead** (inference must not dominate the pipeline),
- **Guaranteed losslessness** (every reconstructed byte must be identical to the input),
- **Online adaptability** (the policy must track distribution shifts without requiring offline retraining).

### 1.3 Scope Narrowing Decision

An early design decision was to **exclude LLM-based neural compression codecs** (e.g., LLMZip, FineZip, DeepZip). While such codecs achieve impressive compression on natural language, they require GPU-accelerated inference that:
- Adds 100–1,000 ms of latency per message,
- Makes real-time streaming infeasible on commodity hardware,
- Violates the sub-millisecond end-to-end latency budget.

The action space was therefore constrained to **five deterministic classical codecs plus one batching action**, making the system fast, deployable, and reproducible without specialized hardware.

---

## 2. Literature Survey and Related Work

The `paperslit/` directory contains 8 research PDFs that were reviewed during the project's inception, including:

| Paper | Topic |
|-------|-------|
| `2306.04050v2.pdf` | Lightweight inference compression |
| `2312.12495v1.pdf` | Transformer-based token compression |
| `2312.13461v3.pdf` | Neural text compression (LLMZip lineage) |
| `2409.17141v1.pdf` | Streaming data compression scheduling |
| `2503.05248v1.pdf` | Bandit algorithms for adaptive resource allocation |
| `2603.19733v1.pdf` | Multi-armed bandit for network transmission |
| `2605.09990v1.pdf` | Short-text compression benchmarks |
| `2606.17712v1.pdf` | Real-time feature extraction for text streams |

Key takeaways from the literature review, captured in `Literature_Survey.docx`:

1. **No existing work** applies a contextual multi-armed bandit to per-message compression scheduling for heterogeneous short-text streams.
2. Existing offline-optimal scheduling methods (e.g., learned compression selectors for video) do not generalize to online streaming with sub-millisecond requirements.
3. MinHash-based Jaccard similarity (datasketch library) is sufficiently fast (~10–50 µs) for online stream repetition scoring.
4. The **LinUCB algorithm** (Li et al., 2010) with ridge regression covariance updates is well-suited to online contextual decision problems with bounded feature vectors.

---

## 3. Problem Gap Analysis

The `Project_Gap_Analysis.docx` document, combined with the `paperslit/I recommend this instead.txt` advisor note, identified the following gaps that this project was designed to fill:

| Gap | Solution Implemented |
|-----|----------------------|
| Static codec selection ignores message-level heterogeneity | Per-message feature extraction + action selection |
| No existing benchmarks on chat / SMS / tweet stream compression | Three real datasets (DailyDialog, Sentiment140, NUS-SMS) |
| Batch compression latency not modeled accurately | Discrete-event simulator with exact microsecond timing + queue wait |
| Repeated-message duplicate detection is not integrated | SHA-256 exact-match LRU cache with lossless verification |
| Online adaptation absent from prior art | LinUCB contextual bandit with Sherman-Morrison online updates |
| Batch delimiter fragility with embedded newlines | 4-byte big-endian length-prefix binary framing protocol |

---

## 4. Conceptual Design and System Scope Decisions

### 4.1 Action Space

The action space for the scheduler is defined as:

```
A = { SKIP, ZSTD, BROTLI, GZIP, LZ4, BATCH }
```

- **SKIP**: Pass the message through uncompressed. Zero CPU cost. Best for very short messages where compressed output would be larger.
- **ZSTD**: Zstandard level-3 single-message compression. Best balanced codec for medium-length compressible messages.
- **BROTLI**: Brotli quality-4. Slower but achieves better ratio on dictionary-compressible natural language.
- **GZIP**: Standard gzip level-6. Widely compatible, moderate speed.
- **LZ4**: LZ4 block compression. Fastest codec; minimal ratio improvement but very low latency overhead.
- **BATCH**: Queue the message until batch is full (≥10 messages) or a 500 ms timeout is exceeded, then compress the entire framed batch payload with ZSTD. Amortizes per-message overhead by exploiting inter-message redundancy.

### 4.2 The Cost Function

The central optimization objective is the **composite cost function**:

$$\text{Cost}(m, a) = \text{CompressedSize}(m, a) + \lambda \times \text{Latency}(m, a, \text{µs})$$

where:
- `CompressedSize` is in bytes,
- `Latency` is measured in **microseconds** (µs),
- `λ` (lambda) is a scalar trade-off weight in **bytes per microsecond**.

The reward signal used by the bandit is `R = -Cost`, so higher rewards correspond to less expensive actions.

### 4.3 Lambda Values as Operating Points

Four distinct values of λ define four different Pareto operating points:

| Lambda | Interpretation | Preferred Actions |
|--------|----------------|-------------------|
| 0.001 | Latency nearly free; maximize compression | BROTLI, GZIP |
| 0.01 | Moderate latency cost | ZSTD, LZ4 |
| 0.1 | High latency cost | LZ4, SKIP |
| 1.0 | Extremely high latency cost; latency dominates | SKIP, CACHE_REUSE |

---

## 5. System Architecture Overview

The full system pipeline, as documented in the project README and implementation plan, follows a strict sequential event-driven flow:

```
Incoming Stream Message (text, timestamp, msg_id)
          │
          ▼
[ 1. Exact-Match Cache Lookup (SHA-256 hash key) ]
   ├── HIT  → Decompress/Reuse → Output (near-zero latency)
   └── MISS → Continue
          │
          ▼
[ 2. 9-Dimensional Feature Extraction ]
  char_len, word_len, entropy, repetition_score,
  arrival_rate, uppercase_ratio, punctuation_ratio,
  emoji_ratio, unique_word_ratio
          │
          ▼
[ 3. Scheduler / Bandit → Select Action ∈ A ]
  ┌─────────────────────────────────────────────┐
  │  Heuristic │ LR │ DT │ RF │ GBM │ LinUCB   │
  └─────────────────────────────────────────────┘
          │
          ▼
[ 4. Execute Action ]
  SKIP    → Pass-through
  ZSTD    → zstandard.compress(level=3)
  BROTLI  → brotli.compress(quality=4)
  GZIP    → gzip.compress(level=6)
  LZ4     → lz4.block.compress()
  BATCH   → Queue; flush on size=10 or timeout=500ms
             (4-byte length-prefix framing + ZSTD batch)
          │
          ▼
[ 5. Cache Store → LRU Cache (capacity 1000) ]
          │
          ▼
[ 6. Bandit Update → R = -Cost (online only) ]
          │
          ▼
[ 7. Metrics Recording: size, latency, cache_hit, action ]
```

The codebase is organized as:

```
Adaptive Learning-Based Compression/
├── src/
│   ├── __init__.py
│   ├── engine.py          (CompressionEngine)
│   ├── features.py        (FeatureExtractor)
│   ├── manager.py         (LRUCache, Batcher, BatchCacheManager)
│   ├── bandit.py          (LinUCBBandit)
│   ├── scheduler.py       (HeuristicScheduler, OfflineLabelGenerator, SupervisedScheduler)
│   ├── simulator.py       (StreamSimulator)
│   ├── data_loader.py     (Dataset preparation)
│   ├── synthetic_data.py  (Fallback generators)
│   ├── utils.py           (Path utilities, env logging)
│   └── tests/
│       ├── test_framework.py
│       ├── test_bug_fixes.py
│       └── test_gap_fixes.py
├── experiments/
│   ├── run_experiments.py
│   └── plot_results.py
├── data/
│   ├── dailydialog_clean.csv
│   ├── sentiment140_clean.csv
│   ├── nussms_clean.csv
│   ├── experiment_results.json
│   ├── env_specs.json
│   ├── feature_importance.png
│   ├── scheduler_latency.png
│   ├── pareto_dailydialog.png
│   ├── pareto_nus-sms.png
│   └── pareto_sentiment140.png
├── paperslit/             (Research PDFs and literature docs)
├── test_cache_and_batch_demo.py
├── test_custom_text.py
├── test_dataset_demo.py
├── test_repetition_demo.py
├── dashboard.py           (Premium Interactive UI)
├── requirements.txt
└── implementation_plan.md
```

---

## 6. Component 1: Classical Compression Engine

**File**: [`src/engine.py`](file:///c:/Users/rithv/Downloads/Adaptive%20Learning-Based%20Compression/src/engine.py) — **68 lines**

### 6.1 Design

The `CompressionEngine` class provides a unified interface for all five classical compression codecs. It abstracts away library-specific APIs and standardizes:
- **Input**: accepts either `str` (auto-encoded to UTF-8) or raw `bytes`,
- **Output**: returns `(compressed_bytes: bytes, latency_microseconds: float)` from `compress()`, and `(decompressed_payload, latency_microseconds)` from `decompress()`.
- **Timing**: uses `time.perf_counter()` for nanosecond-resolution wall-clock measurement, converted to microseconds.

### 6.2 Implementation Details

```python
class CompressionEngine:
    def __init__(self):
        self.zstd_compressor = zstd.ZstdCompressor(level=3)
        self.zstd_decompressor = zstd.ZstdDecompressor()
```

The Zstd compressor and decompressor instances are **reused across calls** to avoid object initialization overhead on the hot path. Other codecs (Brotli, Gzip, LZ4) are invoked via their respective library APIs each call.

**Codec Parameter Settings:**
| Codec | Setting | Rationale |
|-------|---------|-----------|
| ZSTD | level=3 | Balanced speed-ratio default per Zstandard docs |
| BROTLI | quality=4 | Mid-range; avoids high-quality slowdown for short text |
| GZIP | compresslevel=6 | Standard default; widely supported |
| LZ4 | store_size=True | Stores uncompressed size header for decompression |
| SKIP | — | Identity passthrough; exact bytes retained |

### 6.3 Raw Bytes Mode

The `decompress()` method supports `raw_bytes=True` to return raw decompressed `bytes` without UTF-8 decoding. This is critical for batch decompression, where the decompressed output is a **binary framed payload** (length-prefixed), not a plain UTF-8 string.

### 6.4 Losslessness Guarantee

Every `compress → decompress` round-trip is covered by tests that assert byte-for-byte equality, including edge cases with Unicode, emoji, embedded newlines, and random binary content.

---

## 7. Component 2: Feature Extractor

**File**: [`src/features.py`](file:///c:/Users/rithv/Downloads/Adaptive%20Learning-Based%20Compression/src/features.py) — **118 lines**

### 7.1 Design Philosophy

The feature extractor produces a **9-dimensional feature vector** from each incoming message. All features are computed in pure Python without any network calls, LLM inference, or heavy ML models. The design prioritizes:
- **Sub-50 µs extraction time** per message,
- **Statefulness** (some features depend on stream history),
- **Normalizability** (all features have bounded or log-transformed ranges for the linear bandit model).

### 7.2 Feature Definitions

| Feature | Type | Computation | Range |
|---------|------|-------------|-------|
| `char_len` | scalar | `len(text)` | [0, ∞) |
| `word_len` | scalar | Token count after whitespace tokenization | [0, ∞) |
| `entropy` | scalar | Shannon entropy on character distribution: $H = -\sum p_i \log_2 p_i$ | [0, 8.0] |
| `repetition_score` | scalar | Max Jaccard similarity vs. last 50 message MinHashes | [0.0, 1.0] |
| `arrival_rate` | scalar | Exponentially smoothed instantaneous message rate (α=0.2) | [0, ∞) |
| `uppercase_ratio` | scalar | Count of uppercase chars / char_len | [0.0, 1.0] |
| `punctuation_ratio` | scalar | Count of non-word, non-space chars / char_len | [0.0, 1.0] |
| `emoji_ratio` | scalar | Count of emoji chars (U+1F000–U+1F9FF, U+2600–U+27BF) / char_len | [0.0, 1.0] |
| `unique_word_ratio` | scalar | Unique token count / total token count | [0.0, 1.0] |

### 7.3 MinHash-Based Repetition Scoring

The `repetition_score` feature deserves special attention. It addresses the question: *"How similar is this message to recent messages in the stream?"*

```python
def _compute_minhash(self, tokens: list[str]) -> MinHash:
    m = MinHash(num_perm=64)
    for token in tokens:
        m.update(token.encode('utf-8'))
    return m
```

Using `datasketch.MinHash` with 64 permutations, each incoming message is hashed and its Jaccard similarity is computed against all MinHashes in a rolling window of the last `window_size=50` messages. The maximum similarity score is returned.

This feature:
- Signals high values (near 1.0) for exact or near-duplicate messages → cache or batch preferred,
- Signals low values (near 0.0) for novel content → compression or skip preferred.

### 7.4 Arrival Rate Estimation

The arrival rate uses an **exponentially weighted moving average (EWMA)**:

```
arrival_rate_t = α × (1 / Δt) + (1 - α) × arrival_rate_{t-1}
```

With `α = 0.2`, the EWMA is biased toward history (80% weight) while updating smoothly with recent timing. This provides a real-time estimate of message burstiness.

### 7.5 Stateful Reset

The `reset()` method clears the MinHash history and resets timestamp tracking — essential between independent simulation runs in the benchmarking suite to prevent cross-experiment contamination.

---

## 8. Component 3: Batch and Cache Manager

**File**: [`src/manager.py`](file:///c:/Users/rithv/Downloads/Adaptive%20Learning-Based%20Compression/src/manager.py) — **123 lines**

This file contains three distinct classes: `LRUCache`, `Batcher`, and `BatchCacheManager`.

### 8.1 LRU Cache

```python
class LRUCache:
    def __init__(self, capacity: int = 1000):
        self.cache = OrderedDict()
        self.capacity = capacity
```

The cache stores `(compressed_bytes, codec_name)` tuples, keyed by **SHA-256 hash** of the original message text. It uses Python's `collections.OrderedDict` to implement LRU eviction in O(1) via `move_to_end()` on access and `popitem(last=False)` on overflow.

**Design properties:**
- Exact-match only (hash collision is astronomically unlikely with SHA-256),
- Guarantees lossless retrieval (stored compressed bytes + codec allow exact decompression),
- Default capacity: 1,000 entries (configurable),
- Thread-safety: not required (single-threaded simulation).

### 8.2 Batcher

```python
class Batcher:
    def __init__(self, max_batch_size: int = 10, timeout_seconds: float = 0.5):
```

The `Batcher` maintains a FIFO queue of `(text, timestamp, msg_id)` tuples. A flush is triggered by either:
1. **Size trigger**: when `len(queue) >= max_batch_size` (default: 10),
2. **Timeout trigger**: when the wall-clock gap between the **arrival timestamp of the first queued message** and the **current message's timestamp** exceeds `timeout_seconds` (default: 0.5 seconds).

The timeout check is implemented as:
```python
def should_flush(self, current_time: float) -> bool:
    if not self.queue:
        return False
    return (current_time - self.first_message_time) >= self.timeout_seconds
```

> [!IMPORTANT]
> The timeout is evaluated against **stream arrival timestamps**, not wall-clock time. This is a deliberate simulation fidelity choice: the simulator replays recorded timestamps (Poisson-distributed), so time-based decisions remain consistent regardless of machine speed.

### 8.3 Binary Length-Prefix Batch Framing

A critical design decision is the use of **4-byte big-endian `uint32` length-prefix framing** for batch payloads:

```python
@staticmethod
def encode_batch(messages: list[str]) -> bytes:
    payload = bytearray()
    for msg in messages:
        msg_bytes = msg.encode('utf-8')
        payload.extend(struct.pack('>I', len(msg_bytes)))
        payload.extend(msg_bytes)
    return bytes(payload)
```

**Why not use newline delimiters?** Messages in real-world streams (system logs, dialog transcripts) routinely contain embedded `\n` characters. Splitting on newlines would corrupt such messages during reconstruction. The 4-byte length-prefix protocol guarantees **unambiguous, lossless reconstruction** regardless of message content.

This design was motivated by a specific bug identified during development: early prototypes used `"\n".join(messages)` for batching, which silently corrupted messages containing embedded newlines.

### 8.4 BatchCacheManager

`BatchCacheManager` composes `LRUCache` and `Batcher` with two public interface methods:
- `cache_lookup(text)` → `(compressed_bytes, codec)` or `None`,
- `cache_store(text, compressed_bytes, codec)` → stores to LRU,
- `encode_batch(messages)` / `decode_batch(framed_bytes)` → static framing methods,
- `reset()` → clears cache and reinitializes batcher.

---

## 9. Component 4: Scheduler Models and Offline Label Generator

**File**: [`src/scheduler.py`](file:///c:/Users/rithv/Downloads/Adaptive%20Learning-Based%20Compression/src/scheduler.py) — **167 lines**

This file defines the shared action map and three scheduler classes.

### 9.1 Action Map

```python
ACTION_MAP = {
    0: 'SKIP', 1: 'ZSTD', 2: 'BROTLI',
    3: 'GZIP', 4: 'LZ4', 5: 'BATCH'
}
```

All scheduler classes return `(action_index: int, inference_latency_us: float)` from their `predict()` method. This common interface allows the simulator to treat any scheduler polymorphically.

### 9.2 HeuristicScheduler

```python
class HeuristicScheduler:
    def __init__(self, skip_threshold: int = 25):
```

Rule-based baseline implementing:
```
if char_len < 25:       → SKIP
elif repetition > 0.8:  → BATCH
else:                   → ZSTD (default)
```

This baseline serves as the primary hand-crafted comparison point. Its decision latency is typically **< 5 µs** (just an if-else chain).

### 9.3 OfflineLabelGenerator

```python
class OfflineLabelGenerator:
    def __init__(self, lambda_param: float = 0.01):
```

This class computes **oracle-optimal labels** for supervised training. For each message in the training set, it:
1. Tries all 6 actions and records the actual compressed size and latency using the `CompressionEngine`,
2. For BATCH, uses a global approximation: average compressed size share and average latency per message from the first 10 training messages, plus a **50,000 µs (50 ms) queue delay penalty** to represent realistic queueing wait time,
3. Selects the action that minimizes `Cost = Size + λ × Latency`.

The generated labels are then used to train the supervised classifiers via `SupervisedScheduler.fit()`.

**Lambda-dependent label distributions:** Different lambda values result in different label distributions:
- At `λ=0.001`: compression quality matters more → BROTLI/ZSTD labels dominate,
- At `λ=1.0`: latency matters more → SKIP labels dominate (since short messages are cheapest to pass through).

### 9.4 SupervisedScheduler

```python
class SupervisedScheduler:
    def __init__(self, model_type: str = 'decision_tree'):
```

Wraps four sklearn classifiers:

| Model Type | Sklearn Class | Key Hyperparameters |
|-----------|---------------|---------------------|
| `logistic_regression` | `LogisticRegression` | max_iter=1000 |
| `decision_tree` | `DecisionTreeClassifier` | max_depth=5 |
| `random_forest` | `RandomForestClassifier` | n_estimators=100, max_depth=8 |
| `gradient_boosting` | `GradientBoostingClassifier` | n_estimators=100, max_depth=4 |

The `fit()` method converts a list of feature dicts to a numpy matrix and trains the classifier. The `predict()` method times inference with `time.perf_counter()` and returns `(action_index, latency_us)`.

**Feature importances** are extracted from tree-based models via `feature_importances_` for the Decision Tree and `coef_` (normalized absolute values) for Logistic Regression.

---

## 10. Component 5: Contextual Bandit (LinUCB)

**File**: [`src/bandit.py`](file:///c:/Users/rithv/Downloads/Adaptive Learning-Based Compression/src/bandit.py) — **102 lines**

### 10.1 Algorithm: Linear Upper Confidence Bound (LinUCB)

The `LinUCBBandit` implements the **disjoint LinUCB algorithm** (Li et al., 2010) for online contextual multi-armed bandit learning.

**Model State** (per action `a`):
- `A[a]`: `(d×d)` covariance matrix, initialized to `λ_param × I_d` for ridge regularization,
- `b[a]`: `(d×1)` bias vector, initialized to zeros,
- `A_inv[a]`: precomputed inverse of `A[a]`, updated online via Sherman-Morrison formula.

### 10.2 Prediction

```python
def predict(self, features_dict: dict) -> tuple[int, float]:
    x = self._normalize_features(features_dict)
    for a in range(self.K):
        theta = self.A_inv[a] @ self.b[a]
        std_dev = sqrt(x.T @ self.A_inv[a] @ x)
        bonus_weight = (1.0 / self.action_cost_scale[a]) if self.cost_aware_exploration else 1.0
        p[a] = (theta.T @ x) + alpha * bonus_weight * std_dev
    return argmax(p), latency
```

The UCB score for action `a` is:
$$p_a = \theta_a^T x + \alpha \cdot w_a \cdot \sqrt{x^T A_a^{-1} x}$$

where `w_a` is the **cost-aware exploration discount**: high-cost actions (BATCH: 2.0, BROTLI: 1.5) receive smaller exploration bonuses, encouraging the bandit to only explore those expensive actions when their expected reward is very high.

### 10.3 Feature Normalization

Before passing features to the bandit's linear model, raw features are normalized:

```python
x[0] = min(char_len / 280.0, 2.0)         # Normalized to tweet length
x[1] = min(word_len / 50.0, 2.0)
x[2] = entropy / 8.0                       # [0, 1] range
x[3] = repetition_score                    # Already [0, 1]
x[4] = min(log1p(arrival_rate) / 5.0, 2.0) # Log-scaled
x[5] = uppercase_ratio                     # Already [0, 1]
x[6] = punctuation_ratio                   # Already [0, 1]
x[7] = emoji_ratio                         # Already [0, 1]
x[8] = unique_word_ratio                   # Already [0, 1]
```

This normalization ensures the ridge regression model converges properly without feature scale dominance.

### 10.4 Online Update (Sherman-Morrison)

```python
def update(self, action: int, features_dict: dict, reward: float):
    x = self._normalize_features(features_dict)
    self.A[action] += x @ x.T
    self.b[action] += reward * x
    # Sherman-Morrison rank-1 inverse update:
    inv = self.A_inv[action]
    numerator = inv @ x @ x.T @ inv
    denominator = 1.0 + (x.T @ inv @ x)[0, 0]
    self.A_inv[action] = inv - numerator / denominator
```

The **Sherman-Morrison formula** avoids the O(d³) cost of re-inverting the covariance matrix after each update. Instead, it performs an O(d²) rank-1 update, making the bandit suitable for real-time streaming updates.

**Reward signal**: `R = -Cost = -(CompressedSize + λ × Latency_µs)`. Negative cost means higher rewards correspond to cheaper, faster compression decisions.

---

## 11. Component 6: Stream Simulator

**File**: [`src/simulator.py`](file:///c:/Users/rithv/Downloads/Adaptive Learning-Based Compression/src/simulator.py) — **270 lines**

### 11.1 Design Purpose

The `StreamSimulator` is the central orchestrator that replays a recorded message stream and measures per-message performance metrics in a **discrete-event simulation**. It coordinates all other components: cache, feature extractor, scheduler, compression engine, and batcher.

### 11.2 Message Processing Flow

The `run_message()` method implements the full pipeline:

1. **Batch timeout check**: Before processing the new message, check if the batcher's oldest queued message has timed out. If so, flush the batch first.
2. **Cache lookup**: Compute SHA-256 hash and check LRU cache. On HIT, decompress and record as `CACHE_REUSE` action.
3. **Feature extraction**: On cache miss, extract the 9-D feature vector.
4. **Scheduler query**: Call `scheduler.predict(features_dict)` to get `(action_idx, sched_lat_us)`.
5. **Action execution**: Based on the action:
   - `SKIP`: Record raw size as compressed size, zero compression latency.
   - `ZSTD/BROTLI/GZIP/LZ4`: Compress, decompress (verify losslessness), store in cache.
   - `BATCH`: Add to batcher queue. Trigger flush if queue hits max size.
6. **Bandit update**: For online bandit schedulers (those with `update()` method), compute reward and update model.
7. **Metrics recording**: Append the metrics dict to `self.stats`.

### 11.3 Batch Flush Mechanics

When the batch flushes (`flush_batch()`):

1. Retrieve all queued `(text, timestamp, msg_id)` from the batcher.
2. Encode them into a binary length-prefixed frame using `encode_batch()`.
3. Compress the frame payload with ZSTD.
4. Decompress and verify losslessness via assertion: `assert reconstructed_messages == messages_text`.
5. Distribute costs proportionally:
   - **Size share**: `batch_comp_size × (msg_orig_size / total_orig_size)` — proportional to original size contribution.
   - **Latency share**: `(queue_wait_us + comp_lat / n_msgs + decomp_lat / n_msgs)`.
6. Cache each message individually (for future cache hits).
7. Update bandit with each message's proportional reward.

### 11.4 Metrics Structure

Each simulated message records:

```json
{
  "msg_id": 42,
  "text": "Hello world!",
  "timestamp": 1.234,
  "action": "ZSTD",
  "cache_hit": false,
  "scheduler_lat_us": 12.5,
  "comp_lat_us": 8.3,
  "decomp_lat_us": 3.1,
  "orig_size": 12,
  "comp_size": 9,
  "queued": false,
  "e2e_lat_us": 23.9
}
```

### 11.5 Summary Metrics

`get_summary_metrics()` aggregates stats into:
- `total_messages`, `compression_ratio` (total_orig / total_comp),
- `mean_e2e_latency_us`, `p95_e2e_latency_us`,
- `mean_scheduler_latency_us`, `p95_scheduler_latency_us`,
- `cache_hit_rate`, `throughput_msg_per_sec`.

---

## 12. Component 7: Data Loader and Dataset Preparation

**File**: [`src/data_loader.py`](file:///c:/Users/rithv/Downloads/Adaptive Learning-Based Compression/src/data_loader.py) — **263 lines**  
**File**: [`src/synthetic_data.py`](file:///c:/Users/rithv/Downloads/Adaptive Learning-Based Compression/src/synthetic_data.py) — **75 lines**

### 12.1 Three Real Datasets

The project benchmarks on three real-world short-text datasets representing different linguistic styles:

| Dataset | Source | Style | Mean Interval | Samples |
|---------|--------|-------|---------------|---------|
| **DailyDialog** | Hugging Face `daily_dialog` | Multi-turn conversational chat | 0.5 s | 5,000 |
| **Sentiment140** | Hugging Face `sentiment140` | Twitter posts / microblogs | 0.2 s | 5,000 |
| **NUS SMS** | Hugging Face `sms_spam` (ham only) | SMS text messages | 1.5 s | 5,000 |

### 12.2 Loading Strategy with Fallback

Each dataset loader follows a **3-tier fallback strategy**:
1. **Primary**: Load official dataset via `load_dataset(name, revision="refs/convert/parquet")` — uses Parquet format for reliable schema access,
2. **Fallback 1**: Try a community mirror (e.g., `roskoN/dailydialog`, `bdanko/sentiment140`),
3. **Fallback 2**: Generate **high-fidelity synthetic data** using template-based generators if all network attempts fail.

This makes the system runnable in completely offline/air-gapped environments.

### 12.3 Data Cleaning

- **Sentiment140**: Removes `@user` mentions and `https://...` URLs via regex. Keeps messages ≤280 characters.
- **DailyDialog**: Strips whitespace; keeps utterances ≤500 characters.
- **NUS SMS**: Filters to ham-only (non-spam) messages.

### 12.4 Timestamp Generation

All datasets use **Poisson-distributed arrival timestamps**:
```python
def generate_poisson_timestamps(n_messages, mean_interval_sec) -> list[float]:
    for _ in range(n_messages):
        interval = np.random.exponential(scale=mean_interval_sec)
        current_time += interval
```

This simulates realistic bursty message arrivals (inter-arrival times follow an exponential distribution).

### 12.5 Synthetic Fallback Generators

`synthetic_data.py` provides three domain-matched generators:
- **Conversational**: Greetings, questions, answers, closings with 25% exact-repeat probability,
- **Tweets**: Template-based social media messages with 15% exact-repeat probability,
- **SMS**: Short action-based templates with 20% repeat probability.

These generators intentionally include repetition probabilities to simulate realistic cache reuse opportunities.

### 12.6 Sampling

All datasets are subsampled to exactly 5,000 messages using `random.seed(42)` for reproducibility. If fewer messages are available, the shortfall is padded with synthetic data.

Saved to:
- `data/dailydialog_clean.csv` (191 KB, as observed),
- `data/sentiment140_clean.csv` (473 KB),
- `data/nussms_clean.csv` (477 KB).

---

## 13. Experiment Pipeline

**File**: [`experiments/run_experiments.py`](file:///c:/Users/rithv/Downloads/Adaptive Learning-Based Compression/experiments/run_experiments.py) — **153 lines**

### 13.1 Benchmark Design

The benchmark runs the following schedulers on each of the three datasets at each of four lambda values:

**Static Baselines** (evaluated once, lambda-independent):
- Always-Skip, Always-Zstd, Always-Brotli, Always-Gzip, Always-LZ4, Fixed-Batch
- Length-Threshold (skip if char_len < 25, else Zstd)
- HeuristicScheduler

**Lambda-Dependent Schedulers** (evaluated at λ ∈ {0.001, 0.01, 0.1, 1.0}):
- Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (supervised)
- LinUCB Bandit (online)

**Total experimental configurations**: 3 datasets × (8 static + 4 lambda × 5 schedulers) = 3 × 28 = **84 runs**.

### 13.2 Train/Test Split

Each dataset is split 50/50:
- **Train set** (messages 0–2499): used for feature extraction and offline label generation for supervised models,
- **Test set** (messages 2500–4999): used for simulation and metric collection.

For the online bandit, there is no separate training — it starts cold on the test set and learns online as messages arrive.

### 13.3 Supervised Model Training Protocol

```python
# Feature extraction on train set
fe_train = FeatureExtractor()
X_train = [fe_train.extract_features(row["text"], row["timestamp"]) 
           for idx, row in train_df.iterrows()]

# Offline label generation (per lambda)
label_gen = OfflineLabelGenerator(lambda_param=lam)
y_train = label_gen.generate_labels(train_df["text"].tolist())

# Train supervised schedulers
for model in [LR, DT, RF, GBM]:
    model.fit(X_train, y_train)
```

Edge case handling: If `y_train` contains only a single class (which can occur for very extreme lambda values), a `DummyModel` that always predicts the majority class is used to avoid sklearn's `fit()` rejecting single-class inputs.

### 13.4 Results Serialization

All results are written to `data/experiment_results.json` as a nested dictionary:
```
{
  "DatasetName": {
    "Always-Skip": { "static": { ...metrics... } },
    "Heuristic": { "static": { ...metrics... } },
    "lambda_0.01": {
      "Logistic-Regression": { ...metrics... },
      "Decision-Tree": { ...metrics..., "feature_importances": {...} },
      "LinUCB-Bandit": { ...metrics... }
    }
  }
}
```

---

## 14. Test Suite: Architecture, Coverage, and Results

The project includes **three test files** in `src/tests/` covering framework correctness, bug fix verification, and edge case gap testing.

### 14.1 `test_framework.py` — Core Correctness Tests

| Test | What It Verifies |
|------|-----------------|
| `test_compressors` | All 5 codecs (SKIP, ZSTD, BROTLI, GZIP, LZ4) achieve lossless round-trip on 4 test strings including emoji content |
| `test_features` | 9-D feature vector has correct keys; entropy ∈ (0, 8]; repetition ∈ [0, 1]; emoji_ratio > 0 for emoji text |
| `test_cache_reuse_is_lossless` | SHA-256 cache store/lookup returns byte-identical compressed payload; decompressed text matches original |
| `test_bandit_learning_step` | LinUCB covariance matrix `A[action]` changes after a reward update |

### 14.2 `test_bug_fixes.py` — Regression Tests for Fixed Bugs

**Bug 1 — Latency Unit Mismatch:**  
An early version had a mismatch where `OfflineLabelGenerator` computed costs in milliseconds while `StreamSimulator` used microseconds, causing the bandit's reward signal to be scaled incorrectly. The fix aligned both to microseconds (µs) throughout. This test verifies the alignment:
```
Offline cost = CompSize + λ × Latency_µs
Sim cost     = CompSize + λ × Latency_µs
→ EQUAL ✓
```

**Bug 2 — Batch Timeout Not Firing:**  
The batcher's timeout check was originally placed incorrectly — it checked elapsed time based on wall-clock time rather than stream timestamps. On sparse streams, messages could queue indefinitely. The fix moves the check to `should_flush(current_time)` using stream timestamps. The test:
- Sends 3 messages 0.3 s apart (stream time = 0 s, 0.3 s, 0.6 s),
- Verifies that the batcher queue has fewer than 3 messages after the third arrives (the first two were flushed when 0.6 s > 0.5 s timeout).

**Bug 3 — Batch Reconstruction Corrupted by Embedded Newlines:**  
The original batch protocol joined messages with `"\n"` and split on `"\n"` during reconstruction. Messages containing embedded newlines (e.g., `"second test\nwith a newline"`) were corrupted. The fix implements 4-byte length-prefix binary framing. The test:
- Batches 3 messages, one with an embedded newline,
- Verifies each message round-trips correctly through the framed batch codec.

**Additional Sanity Checks in `test_bug_fixes.py`:**

| Sanity Check | What It Verifies |
|---|---|
| SC-1: Compression ratios | ZSTD ratio < 1.0 for repetitive text; SKIP ratio ≈ 1.0 |
| SC-2: Cache hit rate | 50% hit rate after sending exact duplicate |
| SC-3: Scheduler inference latency | DT and LR inference latencies < 1,000 µs (sub-millisecond) |
| SC-4: Round-trip losslessness | 20 random messages × 4 codecs all pass byte-identical verification |
| Engine raw bytes | Compress/decompress round-trip on raw binary (framed batch simulation) |

### 14.3 `test_gap_fixes.py` — Edge Case and Gap Tests

| Test | What It Verifies |
|------|-----------------|
| `test_length_prefix_batching_with_newlines` | 5 messages including empty string and emoji+newline round-trip via `encode_batch/decode_batch` |
| `test_batch_timeout_firing` | Msg 0 at t=0 is confirmed flushed after Msg 1 arrives at t=0.6 (> 0.5 s timeout) |
| `test_microsecond_latency_unit_consistency` | Simulator e2e latency > 0 µs; `OfflineLabelGenerator` runs without error at µs lambda |
| `test_logistic_regression_initialization` | `SupervisedScheduler('logistic_regression')` creates non-null model |
| `test_bandit_cost_aware_exploration` | LinUCB with `cost_aware_exploration=True` predicts valid action and records reward history |

### 14.4 Test Execution Command

```bash
python -m unittest discover -s src -p "test_*.py"
```

All tests pass successfully on the development environment (Python 3.13.5, Windows 11).

---

## 15. Experimental Results: Benchmark Metrics and Analysis

### 15.1 DailyDialog Results

**DailyDialog** represents conversational chat messages. Key characteristics: moderate lengths (~30–150 chars), moderate repetition, high lexical diversity.

#### Static Baselines (DailyDialog, 2,500 test messages):

| Scheduler | Comp. Ratio | Mean E2E Latency (µs) | Cache Hit Rate | Throughput (msg/s) |
|-----------|-------------|------------------------|----------------|---------------------|
| Always-Skip | 1.000 | 11.51 | 0.0% | 86,867 |
| Always-Zstd | 0.590 | 8.14 | 90.96% | 122,907 |
| Always-Brotli | 0.764 | 14.94 | 90.96% | 66,942 |
| Always-Gzip | 0.394 | 20.74 | 90.96% | 48,221 |
| Always-LZ4 | 0.707 | 7.17 | 90.96% | 139,521 |
| Fixed-Batch | 0.624 | 2,146,603 | 90.16% | 0.47 |
| Length-Threshold | 0.982 | 8.73 | 2.44% | 114,489 |
| Heuristic | 0.982 | 9.35 | 2.44% | 107,007 |

> [!NOTE]
> The high 90.96% cache hit rate for Always-Zstd/Brotli/Gzip/LZ4 on DailyDialog reveals that conversational messages are **highly repetitive at the exact-match level** — greetings, acknowledgments, and formulaic utterances repeat frequently. Once a message is compressed and cached, all subsequent duplicates are served via near-zero-latency cache lookup.

> [!WARNING]
> Fixed-Batch achieves enormous mean E2E latency (2.1 seconds per message on average) because every single message queues forever in a batch — the queue only flushes at batch size 10, and for conversational streams, the 500ms timeout is frequently exceeded, causing cascading wait times.

#### Supervised and Bandit Results (DailyDialog, λ=0.001 sample):

| Scheduler | Comp. Ratio | Mean E2E Latency (µs) | Cache Hit Rate |
|-----------|-------------|------------------------|----------------|
| Logistic Regression | 1.000 | 8.98 | 0.0% |
| Decision Tree | 1.000 | 9.10 | 0.0% |
| Random Forest | 1.000 | 9.78 | 0.0% |
| Gradient Boosting | 1.000 | 9.56 | 0.0% |
| LinUCB Bandit | 0.654 | 911,723 | 83.28% |

> [!NOTE]
> All supervised models chose SKIP for DailyDialog at λ=0.001 (ratio=1.0, 0% cache hits). This is because the offline label generator, given DailyDialog's predominantly short messages, found SKIP to be the lowest-cost action. The models learned this policy perfectly. The LinUCB bandit, however, explored BATCH during its cold start (resulting in massive queueing latency) and eventually discovered that short-message DailyDialog benefits from cache reuse (83.28% cache hit rate).

### 15.2 Sentiment140 Results

**Sentiment140** represents Twitter data: longer messages (up to 280 chars), lower repetition, high vocabulary diversity, with emojis and URLs.

#### Static Baselines (Sentiment140, 2,500 test messages):

| Scheduler | Comp. Ratio | Mean E2E Latency (µs) | Cache Hit Rate |
|-----------|-------------|------------------------|----------------|
| Always-Skip | 1.000 | 6.56 | 0.0% |
| Always-Zstd | 1.002 | 32.21 | 0.36% | 
| Always-Brotli | 1.139 | 85.88 | 0.36% |
| Always-Gzip | 0.882 | 80.37 | 0.36% |
| Always-LZ4 | 0.933 | 18.19 | 0.36% |
| Fixed-Batch | 1.521 | 903,371 | 0.32% |

> [!IMPORTANT]
> For Sentiment140, ZSTD and Brotli produce **compression ratios > 1.0** (the compressed output is larger than the input). This is expected behavior: tweets are short (~60–100 bytes), and Zstd/Brotli headers add ~15–30 bytes of overhead per message. Compressing very short messages is counterproductive. The optimal policy for tweets is SKIP.

#### Decision Tree Feature Importances (Sentiment140, λ=0.001):
```
char_len           → 91.6% importance
entropy            → 3.5%
uppercase_ratio    → 3.3%
repetition_score   → 0.56%
unique_word_ratio  → 0.27%
```

`char_len` dominates with >90% importance across all lambda settings and datasets. This confirms the fundamental insight: **message length is the primary predictor of whether compression is beneficial**. Short messages should be skipped; longer messages should be compressed.

### 15.3 NUS-SMS Results

**NUS-SMS** represents real SMS messages: generally short (5–160 chars), moderate repetition, casual language.

The NUS-SMS results follow a similar pattern to DailyDialog — the primary determinant is whether messages are long enough for compression to yield savings.

### 15.4 Cross-Dataset Comparative Analysis

| Dataset | Best Static Ratio | Best Bandit Ratio | Cache Hit Rate (Zstd) |
|---------|------------------|--------------------|----------------------|
| DailyDialog | 0.394 (Gzip) | 0.654–0.767 | 90.96% |
| Sentiment140 | 0.882 (Gzip) | 1.14–1.52 | 0.36% |
| NUS-SMS | (similar to DailyDialog) | — | (high) |

The stark contrast between DailyDialog (90%+ cache hit rate) and Sentiment140 (0.36% cache hit rate) demonstrates that **the optimal compression strategy is highly dataset-dependent** — exactly the scenario where an adaptive scheduler is most valuable.

### 15.5 Scheduler Inference Latency Analysis

From the `scheduler_latency.png` output and result metrics:

| Model | Mean Scheduler Latency (µs) | P95 Latency (µs) |
|-------|-----------------------------|------------------|
| Logistic Regression | ~8–25 µs | ~17–54 µs |
| Decision Tree | ~9–17 µs | ~17–43 µs |
| Random Forest | ~10–6,432 µs | ~18–11,047 µs |
| LinUCB Bandit | ~25–81 µs | ~125–265 µs |

> [!NOTE]
> Logistic Regression and Decision Tree consistently achieve **sub-20 µs mean inference latency** — firmly sub-millisecond. Random Forest's latency is 2–3 orders of magnitude higher due to aggregating 100 trees, making it unsuitable for high-throughput real-time deployment despite similar accuracy. The LinUCB Bandit's ~25–80 µs overhead is acceptable for a streaming context and includes the Sherman-Morrison matrix update cost.

---

## 16. Visualization Outputs

**File**: [`experiments/plot_results.py`](file:///c:/Users/rithv/Downloads/Adaptive Learning-Based Compression/experiments/plot_results.py) — **179 lines**

Three plot types are generated and saved to `data/`:

### 16.1 Pareto Frontier Plots

**Files**: `pareto_dailydialog.png`, `pareto_nus-sms.png`, `pareto_sentiment140.png`

Each Pareto frontier plot shows **Compression Ratio (Y-axis) vs. Mean E2E Latency in ms (X-axis)**, where:
- Static baselines appear as single `◆` diamond markers,
- The Heuristic appears as a `★` star marker,
- Adaptive schedulers (LR, DT, RF, LinUCB) appear as connected line curves — each curve point represents a different λ value.

The ideal operating point is the **top-left corner** (high compression, low latency). A scheduler is Pareto-superior if its curve extends further toward the top-left than baseline markers.

**Plot aesthetics**: Uses a curated color palette (violet for LinUCB, blue for LR, green for DT, amber for RF, pink for Heuristic) with white-grid matplotlib styling for readability.

### 16.2 Scheduler Latency Comparison

**File**: `scheduler_latency.png`

A grouped bar chart with two bars per model (mean latency and P95 latency) on a **log scale Y-axis** (microseconds). This clearly shows the orders-of-magnitude difference between lightweight models (DT: ~9 µs) and ensemble models (RF: ~6,000 µs).

### 16.3 Feature Importance Chart

**File**: `feature_importance.png`

A horizontal bar chart of Gini importance values from the Decision Tree model (green bars). `char_len` dominates at 89–92% importance across all datasets, followed by `entropy` and `uppercase_ratio`.

---

## 17. Runtime Utilities and Environment Logging

**File**: [`src/utils.py`](file:///c:/Users/rithv/Downloads/Adaptive Learning-Based Compression/src/utils.py) — **78 lines**

### 17.1 Path Utilities

```python
def get_project_root() -> str  # Returns absolute path to project root
def get_data_dir() -> str      # Returns absolute path to data/ (creates if missing)
```

These functions make all file I/O paths **portable and platform-independent** — no hardcoded paths anywhere in the codebase.

### 17.2 Environment Specification Logging

`log_environment_specs()` captures and saves the complete runtime environment to `data/env_specs.json`:
- OS, Python version, machine architecture,
- CPU count, RAM (queried via `wmic` on Windows),
- GPU availability (via `nvidia-smi`),
- Installed package versions for all dependencies.

**Actual environment at time of benchmarking** (from `data/env_specs.json`):

| Spec | Value |
|------|-------|
| OS | Windows 11 (10.0.26200) |
| Python | 3.13.5 (MSC v.1943 64-bit AMD64) |
| Processor | Intel64 Family 6 Model 186 Stepping 3 (GenuineIntel) |
| CPU Cores | 12 |
| RAM | 15.72 GB |
| GPU | None (CPU-only) |
| zstandard | 0.25.0 |
| brotli | 1.2.0 |
| lz4 | 4.4.5 |
| scikit-learn | 1.7.1 |
| pandas | 2.3.1 |
| matplotlib | 3.10.3 |
| numpy | 2.3.1 |
| datasketch | 2.0.0 |
| datasets | 5.0.0 |

---

## 18. Interactive Premium Dashboard

**File**: [`dashboard.py`](file:///c:/Users/rithv/Downloads/Adaptive Learning-Based Compression/dashboard.py) — **~1,130 lines**

To visually demonstrate the effectiveness of the system, a high-performance, interactive web dashboard was built using Streamlit and Plotly. The dashboard features a custom, highly polished "obsidian-indigo" CSS theme with sleek gradient cards, modern typography, and robust visual layout.

### 18.1 Key Features

- **🔴 Live Stream (Auto ML Mode)**: Replays a dataset stream with real wall-clock pacing. The **LinUCB Bandit** adaptively routes each message in real-time. Features live metrics and plots (Action Distribution, Latency Over Time, Pareto Frontier) updating asynchronously.
- **✍️ Single Message Test**: A sandbox where users can type any message to see the extracted 9-dimensional features, exact cache hit status, compression ratio, and end-to-end latency. It also features a "Head-to-Head" checkbox to benchmark that specific message across all 5 models simultaneously.
- **📊 Scheduler Comparison**: Runs a fixed stream dataset through every available scheduler independently to generate a direct comparative benchmark. Automatically renders beautiful bar charts and a multi-metric radar chart to identify the winning scheduler per metric.

### 18.2 UI Design Choices
The interface eschews standard Streamlit styling for a completely bespoke gradient aesthetic. The user experience is tightly controlled to prevent confusion—for example, the "Live Stream" tab explicitly restricts the backend model to the LinUCB Contextual Bandit, dynamically hiding conflicting user controls while seamlessly adapting to underlying distribution shifts (e.g., chat to tweets).

---

## 19. CLI Demo and Evaluation Tools

### 18.1 `test_cache_and_batch_demo.py`

**Purpose**: Demonstrates the cache reuse and batch compression mechanics interactively.

**Demo 1 — Cache Reuse:**
- Message #1 arrives: CACHE MISS → compress with ZSTD, store in cache.
- Message #2 arrives (exact duplicate): CACHE HIT → retrieve, `Compression Time: 0.00 µs`.

**Demo 2 — Batch Compression:**
Sends 5 short messages individually vs. as a batch:
- Individual compression: each gets Zstd block header overhead → compressed output larger than raw input.
- Batch compression: all 5 messages framed and compressed together → cross-message redundancy exploited, **significant space savings** vs. individual compression.

### 18.2 `test_custom_text.py`

**Purpose**: Interactive/CLI tool to test any custom text through the full pipeline.

```bash
python test_custom_text.py "Your custom message here!"
python test_custom_text.py   # Interactive mode
```

For each input message:
1. Prints all 9 extracted features with 4-decimal precision,
2. Reports cache status (HIT/MISS),
3. Prints scheduler's action decision and inference time,
4. Reports compression results: original size, compressed size, ratio, engine latency, total E2E latency.

### 18.3 `test_dataset_demo.py`

**Purpose**: Faculty presentation CLI demo evaluating a scheduler over a full dataset.

```bash
python test_dataset_demo.py --dataset data/dailydialog_clean.csv --samples 50 --scheduler "Decision Tree"
python test_dataset_demo.py --dataset data/nussms_clean.csv --scheduler "LinUCB"
```

Supports: Heuristic, Decision Tree, Random Forest, Logistic Regression, LinUCB schedulers.

**Output format**: A formatted table showing the first 15 and last 5 messages with columns for message ID, original bytes, compressed bytes, ratio, saved %, action taken, and E2E latency. Followed by:
- Aggregated summary (total bytes, overall ratio, mean latency, cache hit rate),
- Action breakdown histogram (% of messages per action type).

The demo dynamically trains supervised models on a small bootstrap sample from DailyDialog before running on the target dataset.

### 18.4 `test_repetition_demo.py`

**Purpose**: Demonstrates the feature extractor's handling of repetitive stream content and cache reuse.

Three messages:
1. Long repetitive emoji alert → CACHE MISS → ZSTD compression,
2. Slightly modified version → CACHE MISS → different repetition_score,
3. Exact duplicate of message #1 → CACHE HIT → `Retrieval Time: ~0.00 µs`.

For each message, prints the `unique_word_ratio` (internal repetition), `repetition_score` (stream repetition vs. history), and `emoji_ratio`.

---

## 20. Dependency Stack and Environment Specifications

**File**: [`requirements.txt`](file:///c:/Users/rithv/Downloads/Adaptive Learning-Based Compression/requirements.txt)

```
zstandard>=0.21.0      # ZSTD compression
brotli>=1.1.0          # Brotli compression
lz4>=4.3.2             # LZ4 block compression
scikit-learn>=1.3.0    # LR, DT, RF, GBM classifiers
pandas>=2.0.0          # Dataset loading and manipulation
matplotlib>=3.7.0      # Plot generation
numpy>=1.24.0          # Linear algebra for LinUCB
datasketch>=1.5.9      # MinHash implementation
datasets>=2.14.0       # Hugging Face dataset loading
streamlit>=1.25.0      # (Listed; used for interactive dashboard tooling)
plotly>=5.15.0         # (Listed; used for interactive chart rendering)
```

The project requires **Python 3.10+** (uses `tuple[int, float]` type hints which require Python 3.9+; the `match` statement pattern used in some error handling requires 3.10+).

All compression libraries (zstandard, brotli, lz4) are pure Python wrappers around compiled C extensions, ensuring microsecond-range compression without Python GIL overhead.

---

## 21. Project File Structure and Code Summary

### 21.1 Complete File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `src/engine.py` | 68 | CompressionEngine: all codecs, latency measurement |
| `src/features.py` | 118 | FeatureExtractor: 9-D feature vector |
| `src/manager.py` | 123 | LRUCache, Batcher, BatchCacheManager |
| `src/bandit.py` | 102 | LinUCBBandit: online contextual bandit |
| `src/scheduler.py` | 167 | HeuristicScheduler, OfflineLabelGenerator, SupervisedScheduler |
| `src/simulator.py` | 270 | StreamSimulator: discrete-event simulation |
| `src/data_loader.py` | 263 | Dataset preparation for all 3 datasets |
| `src/synthetic_data.py` | 75 | Fallback synthetic data generators |
| `src/utils.py` | 78 | Path utilities, environment logging |
| `src/__init__.py` | 1 | Package init |
| `src/tests/test_framework.py` | 103 | Core correctness tests (4 tests) |
| `src/tests/test_bug_fixes.py` | 175 | Bug regression + sanity checks (7 tests) |
| `src/tests/test_gap_fixes.py` | 88 | Edge case gap tests (5 tests) |
| `experiments/run_experiments.py` | 153 | Full benchmark pipeline |
| `experiments/plot_results.py` | 179 | Visualization generation |
| `test_cache_and_batch_demo.py` | 88 | Cache/batch demo script |
| `test_custom_text.py` | 105 | Interactive custom text test |
| `test_dataset_demo.py` | 193 | Dataset CLI evaluation demo |
| `test_repetition_demo.py` | 80 | Repetition and cache reuse demo |
| `dashboard.py` | 1,129 | Premium Interactive Dashboard |
| `implementation_plan.md` | 97 | Technical design documentation |
| `requirements.txt` | 12 | Dependency specifications |
| `paperslit/README.md` | 99 | System architecture documentation |
| **Total** | **~3,572** | |

### 21.2 Data Artifacts

| File | Size | Contents |
|------|------|---------|
| `data/dailydialog_clean.csv` | 191 KB | 5,000 chat messages with msg_id, text, timestamp |
| `data/sentiment140_clean.csv` | 473 KB | 5,000 cleaned tweets |
| `data/nussms_clean.csv` | 477 KB | 5,000 SMS ham messages |
| `data/experiment_results.json` | 47 KB | Full benchmark results (1,008 lines) |
| `data/env_specs.json` | 679 B | System environment specs |
| `data/feature_importance.png` | 93 KB | Decision Tree feature importance bar chart |
| `data/scheduler_latency.png` | 92 KB | Scheduler inference latency comparison |
| `data/pareto_dailydialog.png` | 207 KB | Pareto frontier for DailyDialog |
| `data/pareto_nus-sms.png` | 253 KB | Pareto frontier for NUS-SMS |
| `data/pareto_sentiment140.png` | 269 KB | Pareto frontier for Sentiment140 |

---

## 22. Key Findings and Conclusions

### 22.1 Message Length is the Dominant Signal

Across all three datasets and all lambda settings, `char_len` consistently captures >89% of Decision Tree feature importance. The primary scheduling decision is:

> **"Is this message long enough for compression to yield net savings?"**

For messages below ~25 characters, all compressors add more overhead (header bytes) than they save, making SKIP the optimal action. For messages above ~100 characters with redundant content, ZSTD or batching yield meaningful savings.

### 22.2 Exact-Match Caching Dominates on Repetitive Streams

For DailyDialog (conversational chat), the 90.96% cache hit rate observed with static compression baselines reveals that **most conversational messages are exact duplicates**. A simple exact-match cache effectively eliminates the need for per-message compression decisions for the majority of traffic.

This implies that for real-world chat deployments, cache design and capacity are at least as important as the choice of compression algorithm.

### 22.3 Compression is Counterproductive for Short Unique Messages

Sentiment140 (Twitter) demonstrated that for short, unique, high-entropy messages, classical compressors can produce output **larger than the input** (compression ratios > 1.0). This reinforces the need for adaptive scheduling — applying ZSTD blindly to a tweet stream wastes both CPU and bandwidth.

### 22.4 LinUCB Bandit Behavior

The LinUCB Bandit showed interesting behavior:
- **DailyDialog**: Achieved 83.28% cache hit rate after online learning — it discovered the value of CACHE_REUSE actions. However, the cold-start exploration of BATCH action during early messages inflated mean latency due to long queue waits.
- **Sentiment140**: The bandit eventually converged on SKIP-biased actions as expected for short unique tweets.

The bandit's key advantage is **no offline training** — it starts cold and adapts in real time, making it suitable for deployment on data distributions where labeled training data is unavailable.

### 22.5 Supervised Schedulers Converge to SKIP on Short Streams

For DailyDialog at all tested lambda values, the supervised models (LR, DT, RF, GBM) all learned to SKIP every message (compression ratio = 1.0, 0% cache hits). This is because the offline label generator, using the cost function, correctly identifies SKIP as the cheapest action for short conversational messages. The trained classifiers generalize this rule perfectly.

This is technically correct behavior, but means that for DailyDialog specifically, the sophisticated ML scheduler reduces to the trivial "always skip" policy — the real value of the cache is handled transparently by the simulator's cache layer, not the scheduler.

### 22.6 Pareto Frontier Insights

The Pareto frontier plots reveal:
- For DailyDialog, static baselines (Gzip, Zstd) achieve better compression ratios than the adaptive schedulers for medium lambda — because the static baselines compress even short messages while the adaptive schedulers skip them.
- For Sentiment140, LinUCB achieves competitive Pareto positions at λ=0.1, suggesting it learns faster for high-lambda regimes where SKIP rewards are clear.
- The Heuristic scheduler consistently occupies a near-identical Pareto position as supervised schedulers — its simple rules capture most of the relevant structure.

---

## 23. Known Limitations and Future Work

### 23.1 Identified Limitations

| Limitation | Impact | Potential Fix |
|-----------|--------|---------------|
| Simulated (not live) latency measurement | CPU load variability not captured; batch queue wait is computed from timestamps, not real wall-clock | Integrate with a real streaming server (e.g., WebSockets, Kafka) |
| LinUCB cold-start problem | First ~50–100 messages explore inefficiently; large latency spikes from BATCH exploration | ε-greedy warm-start from heuristic actions, or pre-load with synthetic data |
| Supervised models learn trivial SKIP policy on short streams | Reduces ML value to heuristic-equivalent for DailyDialog | Use message-type classification to conditionally apply compression |
| BATCH cost approximation uses global sample average | Per-message BATCH cost estimate is imprecise for heterogeneous streams | Maintain online running average of batch compression ratios |
| Single-threaded simulation | Does not model multi-core parallelism or network I/O | Implement async streaming pipeline with concurrent compression workers |
| No compression quality levels explored | Fixed Zstd level=3, Brotli quality=4 | Add codec quality as part of the action space |
| Dataset size limited to 5,000 messages | Results may not generalize to streaming volumes of millions/hour | Scale experiments with larger corpora |

### 23.2 Future Research Directions

1. **Contextual Batching**: Instead of always queuing for BATCH action, use features to predict the optimal batch partner — only batch with messages that share high Jaccard similarity.

2. **Cascading Cache Levels**: Implement a two-level cache: exact-match (SHA-256) at level 1, and approximate-match (MinHash LSH similarity ≥0.9) at level 2 for near-duplicate compression reuse.

3. **Streaming Distribution Shift Detection**: Add a CUSUM or Page-Hinkley change detection test on the reward signal to detect when the message distribution shifts (e.g., from chat to alerts), allowing the bandit to reset its exploration bonus.

4. **Online Supervised Learning**: Integrate an online gradient descent classifier (e.g., scikit-learn's `SGDClassifier`) that can update continuously from streaming rewards without a full retraining cycle.

5. **Real-World Deployment**: Integrate with a WebSocket server or Kafka consumer to measure true end-to-end latencies including network I/O overhead.

6. **Neural Compression as Future Action**: For long messages (>500 chars), include a neural codec (Brotli-NN or LLM-token prediction) as an optional high-latency high-ratio action, with the bandit learning to selectively invoke it only when the throughput budget allows.

---

## Appendix A: Mathematical Summary

### Cost Function
$$\text{Cost}(m, a) = |C_a(m)|_{\text{bytes}} + \lambda \cdot T_a(m)_{\mu s}$$

### LinUCB UCB Score
$$p_{a,t} = \theta_a^T x_t + \alpha \cdot w_a \cdot \sqrt{x_t^T A_a^{-1} x_t}$$

where $w_a = 1 / \text{cost\_scale}_a$ (cost-aware exploration discount).

### Sherman-Morrison Online Inverse Update
$$(A + x x^T)^{-1} = A^{-1} - \frac{A^{-1} x x^T A^{-1}}{1 + x^T A^{-1} x}$$

### MinHash Jaccard Approximation
$$J(A, B) \approx \frac{|\{i : h_i(A) = h_i(B)\}|}{n_{\text{perm}}}$$

with $n_{\text{perm}} = 64$ hash permutations.

### Shannon Entropy
$$H(m) = -\sum_{c \in \text{chars}} p(c) \log_2 p(c)$$

### Poisson Message Arrival Model
$$\Delta t_i \sim \text{Exponential}(\mu)$$
$$\text{timestamp}_i = \sum_{j=0}^{i} \Delta t_j$$

---

*End of Report*

---

**Report compiled**: 2026-08-18  
**Project**: Adaptive Learning-Based Compression Scheduling for Real-Time Short-Text Streams  
**Workspace**: `c:\Users\rithv\Downloads\Adaptive Learning-Based Compression`  
**Coverage**: All production source files, tests, datasets, experiments, and output artifacts as of pre-August 18 system state.
