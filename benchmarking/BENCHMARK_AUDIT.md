# Benchmark Validity Audit

This document contains a research-grade audit of the experimental methodology currently present in the Adaptive Learning-Based Compression repository, evaluated against 25 criteria essential for peer-reviewed publication.

## A. Compression-ratio formulas and their interpretation
- **Severity**: CRITICAL
- **File**: `src/simulator.py`, `experiments/plot_results.py`, `project_report.md`
- **Function**: `StreamSimulator.get_summary_metrics()`
- **Problem**: `simulator.py` calculates `comp_ratio = total_orig / total_comp`. This means values > 1 indicate successful compression. However, `plot_results.py` labels the Y-axis as "Compression Ratio (Compressed / Original)" and `project_report.md` explicitly states "ZSTD and Brotli produce compression ratios > 1.0 (the compressed output is larger than the input)".
- **Consequence**: The conclusions derived from the code vs the report are completely inverted. A ratio of 1.5 in code means space was saved; a ratio of 1.5 in the report means space was wasted.
- **Proposed Correction**: Deprecate the ambiguous term "compression ratio". Replace entirely with unambiguous metrics: `compression_factor = original/compressed` (>1 = good), `compressed_fraction = compressed/original` (<1 = good), and `space_saving_pct = 100 * (1 - compressed/original)`.

## B. Feature-extraction latency
- **Severity**: CRITICAL
- **File**: `src/simulator.py`
- **Function**: `StreamSimulator.process_message()`
- **Problem**: The time taken by `FeatureExtractor.extract()` (which includes expensive MinHash string operations) is completely absent from the `e2e_lat_us` and `scheduler_lat_us` calculations.
- **Consequence**: Adaptive schedulers (DT, RF, LinUCB) appear artificially fast because the heaviest part of their decision pipeline is not billed to their end-to-end latency, making the comparison against static algorithms highly unfair.
- **Proposed Correction**: Wrap feature extraction in a timing block and explicitly add `feature_latency_us` to the total `e2e_latency_us` and compute costs.

## C. Throughput calculation
- **Severity**: HIGH
- **File**: `src/simulator.py`
- **Function**: `StreamSimulator.get_summary_metrics()`
- **Problem**: Throughput is derived mathematically as `len(self.stats) / sum(e2e_lat_us)`.
- **Consequence**: This is not a real throughput measurement. It ignores Python loop overhead, I/O, garbage collection, and simulator framework costs.
- **Proposed Correction**: Use real wall-clock elapsed time (`time.perf_counter()`) across the entire stream simulation loop to compute true `msg/s` and `MB/s`.

## D. Cache fairness between baselines
- **Severity**: HIGH
- **File**: `src/simulator.py`
- **Function**: `StreamSimulator.process_message()`
- **Problem**: Compression actions (ZSTD, etc.) populate the cache. The `Always-Skip` baseline does not compress, so it does not populate the cache. Thus, if a stream is highly repetitive, ZSTD will get >90% cache hits (yielding near-zero latency for duplicates) while SKIP must process every duplicate separately.
- **Consequence**: The static baseline comparison is confounded by cache dynamics. ZSTD's high throughput on DailyDialog is actually due to the cache, not the codec.
- **Proposed Correction**: Perform full ablation studies separating the cache from the scheduler (e.g., Scheduler with cache OFF, Scheduler with cache ON).

## E. Hidden batch cache cost
- **Severity**: HIGH
- **File**: `src/manager.py`
- **Function**: `BatchCacheManager.process_batch()`
- **Problem**: When a batch flushes, the messages are batch-compressed, but then *each individual message* in the batch is separately Zstd-compressed just to populate the exact-match cache for future use.
- **Consequence**: This extra individual Zstd compression takes CPU time but is not charged to the `queue_wait_latency_us` or the message's compute cost.
- **Proposed Correction**: Explicitly time the individual cache-population compression and add it to `cache_store_latency_us` or disable it via a flag during pure batching experiments.

## F. Offline BATCH model mismatch
- **Severity**: CRITICAL
- **File**: `src/scheduler.py`
- **Function**: `OfflineLabelGenerator._simulate_action_costs()`
- **Problem**: The offline label generator assumes a fixed `BATCH_QUEUE_PENALTY_US = 50_000` (50ms). However, the actual simulator (`BatchCacheManager`) defaults to a timeout of 0.5 seconds (500,000 µs).
- **Consequence**: Supervised models (DT, RF) are trained on fake labels that severely underestimate the actual queue penalty, causing them to incorrectly prefer BATCH in situations where the real simulator will timeout and incur massive latency.
- **Proposed Correction**: Use actual batch queue wait times measured during training-stream simulation to generate realistic oracle labels, avoiding fixed heuristics.

## G. BATCH label size allocation underestimation
- **Severity**: HIGH
- **File**: `src/scheduler.py`
- **Function**: `OfflineLabelGenerator._simulate_action_costs()`
- **Problem**: `avg_batch_size` is calculated by dividing total batch output by batch capacity. Then, when assigning cost to message $i$, it is scaled again by the fraction of $i$'s length to the total capacity length.
- **Consequence**: The batch cost is effectively divided twice, making BATCH appear significantly cheaper to the offline supervised models than it actually is mathematically.
- **Proposed Correction**: Derive exact proportional shares: `compressed_batch_bytes * (original_bytes_i / sum_batch_original_bytes)`.

## H. Dataset provenance and synthetic fallback
- **Severity**: HIGH
- **File**: `src/data_loader.py`
- **Function**: `load_datasets()`
- **Problem**: If datasets are missing, the code silently falls back to synthetic generation via `src.synthetic_data`.
- **Consequence**: A user running the benchmark could unknowingly publish results based entirely on synthetic mock data instead of the claimed DailyDialog/Sentiment140 datasets.
- **Proposed Correction**: Datasets must strictly enforce provenance. Use SHA-256 hashes for raw datasets and explicit failure/warnings if real datasets are missing. Add explicit metadata sidecars.

## Other Methodological Issues
- **Random Seeds (12)**: Not explicitly set universally; NumPy and Python random seeds are unconstrained, making experiments non-reproducible.
- **Oracle Comparison (24)**: No formal optimal oracle baseline is tracked during the benchmark to compute regret properly.
- **Machine/CPU variability (20)**: CPU load isn't isolated; wall-clock latency tests are susceptible to background OS noise. Need robust statistical aggregation across multiple seeds.
- **Lossless reconstruction checks (21)**: No validation that the decompression actually yields the exact original byte stream.

---

### Conclusion

**BENCHMARK_VALIDITY_SCORE = 45/100**

The current codebase establishes an excellent architectural foundation, but the measurement and logging layer contains critical mathematical and logical flaws that would invalidate the results in a peer-reviewed context. These must be repaired before executing formal benchmarks.
