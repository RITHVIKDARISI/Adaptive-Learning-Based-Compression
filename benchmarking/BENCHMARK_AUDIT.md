# Rigorous Benchmark Validity Audit

This document is a research-grade audit of the experimental methodology present in the `RITHVIKDARISI/Adaptive-Learning-Based-Compression` repository. 

**BENCHMARK_VALIDITY_SCORE = 32/100**

While the system architecture is conceptually sound, the current measurement harness cannot support claims in a peer-reviewed research paper. The codebase contains severe timing omissions, mathematical conflicts, and a lack of statistical rigor.

Below is an explicit, exhaustive audit of the 25 requested methodological components.

---

### 1. compression-ratio formulas and their interpretation
- **Severity**: CRITICAL
- **File**: `src/simulator.py` (Line 243) vs `experiments/plot_results.py` (Line 89)
- **Problem**: `simulator.py` calculates `comp_ratio = total_orig / total_comp`. This means values > 1 indicate successful compression. However, `plot_results.py` labels the Y-axis as "Compression Ratio (Compressed / Original)", and the report assumes <1 means compression.
- **Consequence**: The quantitative conclusions in the report are inverted relative to the raw JSON output.
- **Proposed Correction**: Deprecate "compression ratio" and replace with `compression_factor = original/compressed` and `compressed_fraction = compressed/original`.

### 2. bytes saved / storage reduction calculations
- **Severity**: HIGH
- **File**: `src/simulator.py`
- **Problem**: Total bytes saved is not calculated or reported as a percentage, leading to reliance on the ambiguous "ratio" metric.
- **Proposed Correction**: Add `space_saving_pct = 100 * (1 - compressed/original)`.

### 3. compression and decompression timing
- **Severity**: CRITICAL
- **File**: `src/engine.py`
- **Problem**: Only compression time is measured via `time.perf_counter()`. Decompression is completely unimplemented and un-timed.
- **Consequence**: The total end-to-end latency metric is missing the decompression penalty, which severely biases results in favor of slow-to-decompress codecs.
- **Proposed Correction**: Implement decompression for all codecs and measure `decompression_latency_us`.

### 4. scheduler latency
- **Severity**: MEDIUM
- **File**: `src/scheduler.py`
- **Problem**: Inference is timed using `time.perf_counter()`, but standard model caching/JIT compilation warm-up is ignored.
- **Proposed Correction**: Implement warm-up iterations before measuring latency.

### 5. feature-extraction latency
- **Severity**: CRITICAL
- **File**: `src/simulator.py` (Line 142+)
- **Problem**: The time taken by `FeatureExtractor.extract_features()` (which includes expensive MinHash operations) is not measured and is excluded from `e2e_lat_us`.
- **Consequence**: Adaptive schedulers appear artificially fast since the heaviest part of their pipeline is unbilled.
- **Proposed Correction**: Wrap feature extraction in a timer and add `feature_latency_us` to `e2e_latency_us`.

### 6. cache lookup and cache insertion latency
- **Severity**: HIGH
- **File**: `src/manager.py`
- **Problem**: Hashing operations (`hashlib.sha256`) and OrderedDict manipulation in `LRUCache` are un-timed.
- **Consequence**: The overhead of cache management is invisible in the metrics.
- **Proposed Correction**: Add explicit `cache_lookup_latency_us` and `cache_store_latency_us` metrics.

### 7. batching queue latency
- **Severity**: MEDIUM
- **File**: `src/simulator.py`
- **Problem**: Queueing time is computed via synthetic timestamp arithmetic (`current_ts - first_msg_ts`), not actual wall-clock queue duration.
- **Proposed Correction**: Continue using logical simulation time for queueing, but clearly decouple it from active CPU compute time in reporting.

### 8. hidden computational costs
- **Severity**: HIGH
- **File**: `src/simulator.py` (BATCH flush logic)
- **Problem**: When a batch flushes, the batch payload is generated, but then every individual message is *separately* compressed with ZSTD simply to populate the exact-match cache.
- **Consequence**: This takes significant CPU time but is not charged to the message's `total_compute_latency_us`.
- **Proposed Correction**: Either disable cache-insertion during batching, or charge the individual ZSTD compression time to the message.

### 9. throughput calculation
- **Severity**: CRITICAL
- **File**: `src/simulator.py` (Line 257)
- **Problem**: Throughput is derived mathematically as `len(self.stats) / sum(e2e_lat_us)`.
- **Consequence**: This completely ignores Python loop overhead, I/O, and framework costs. It is not a real throughput measurement.
- **Proposed Correction**: Measure true wall-clock time across the entire stream loop to compute true `msg/s` and `MB/s`.

### 10. train/test leakage
- **Severity**: LOW
- **File**: `experiments/run_experiments.py`
- **Problem**: Data is split 50/50 for train/test. State leakage is avoided because `sim.reset()` correctly clears the `BatchCacheManager`.

### 11. benchmark reproducibility
- **Severity**: HIGH
- **File**: `experiments/run_experiments.py`
- **Problem**: The models (`DecisionTreeClassifier`, `LogisticRegression`) lack a hardcoded `random_state` upon initialization.
- **Consequence**: Results will drift between runs.
- **Proposed Correction**: Enforce a global `random_state` mapping to the overarching seed.

### 12. random seeds
- **Severity**: HIGH
- **File**: `src/data_loader.py`
- **Problem**: `random.seed(42)` is called mid-function, globally mutating the random state unpredictably. `np.random.exponential` is entirely unseeded.
- **Proposed Correction**: Set a strict global seed array `[42, 123, 456, 789, 2026]` at the absolute beginning of the benchmark harness.

### 13. dataset provenance
- **Severity**: HIGH
- **File**: `src/data_loader.py`
- **Problem**: Network failures cause silent fallbacks to community mirrors or pure synthetic generation. The output CSV does not track its provenance.
- **Consequence**: Published results could unknowingly be based on synthetic mocks instead of the claimed official datasets.
- **Proposed Correction**: Calculate SHA-256 hashes of the dataset files and include a `"source": "REAL" | "SYNTHETIC"` metadata flag.

### 14. synthetic-data fallback
- **Severity**: HIGH
- **File**: `src/data_loader.py`
- **Problem**: See (13). The fallback is completely silent to the downstream experiment runner.
- **Proposed Correction**: Issue a massive `WARNING` or require an explicit `--allow-synthetic` flag.

### 15. batch cost approximation
- **Severity**: HIGH
- **File**: `src/scheduler.py` (`OfflineLabelGenerator`)
- **Problem**: The offline labels use a global average batch size and latency derived *only from the first 10 messages*.
- **Consequence**: Highly inaccurate labels for non-stationary streams.
- **Proposed Correction**: Perform a realistic training-only batch simulation to assign actual fractional costs to each message.

### 16. supervised offline-label validity
- **Severity**: CRITICAL
- **File**: `src/scheduler.py`
- **Problem**: `OfflineLabelGenerator` uses `batch_queue_penalty = 50000.0` (50ms). The actual `BatchCacheManager` uses `0.5` seconds (500ms).
- **Consequence**: Supervised models are trained to think BATCH is 10x faster than it actually is.
- **Proposed Correction**: Sync the timeout variables exactly.

### 17. LinUCB reward correctness
- **Severity**: MEDIUM
- **File**: `src/bandit.py`
- **Problem**: The reward computation correctly subtracts cost, but the cost lacks `feature_latency_us` and `cache_latency_us`.
- **Proposed Correction**: Update the global cost function to use `total_compute_latency_us` rather than just `codec_latency`.

### 18. cache fairness between different baselines
- **Severity**: CRITICAL
- **File**: `experiments/run_experiments.py`
- **Problem**: `Always-Skip` does not populate the cache. `Always-Zstd` does. Duplicate messages get ~0ms latency under Zstd, making Zstd incorrectly appear to have higher compute throughput than Skip.
- **Consequence**: The comparison is irreparably confounded. 
- **Proposed Correction**: Run an ablation where the cache is turned off for all schedulers to measure pure codec/scheduler performance.

### 19. warm-up effects
- **Severity**: MEDIUM
- **File**: `experiments/run_experiments.py`
- **Problem**: First invocations of classifiers pay Python instantiation overhead.
- **Proposed Correction**: Add 10 warm-up iterations before capturing latencies.

### 20. machine/CPU variability
- **Severity**: HIGH
- **File**: `experiments/run_experiments.py`
- **Problem**: Only one seed/run is executed. CPU load spikes can easily bias latency timings.
- **Proposed Correction**: Benchmark each configuration across multiple independent seeds and aggregate with median/P95 and standard deviations.

### 21. lossless reconstruction checks
- **Severity**: CRITICAL
- **File**: `src/engine.py`
- **Problem**: There is zero verification that the bytes coming out of decompression actually match the input bytes.
- **Consequence**: A bug in frame headers could result in garbage decompression and silently pass the benchmark.
- **Proposed Correction**: Implement a strict `SHA-256(original) == SHA-256(reconstructed)` verification check.

### 22. statistical testing
- **Severity**: HIGH
- **File**: `experiments/plot_results.py`
- **Problem**: Means are reported directly. No confidence intervals, standard deviations, or effect sizes are provided.
- **Proposed Correction**: Output raw seed data, compute 95% CIs, and perform Wilcoxon signed-rank tests for scheduler comparisons.

### 23. baseline completeness
- **Severity**: MEDIUM
- **File**: `experiments/run_experiments.py`
- **Problem**: Missing pure non-adaptive context codecs (`zlib`, `lzma`, `bz2`) to establish standard bounds.
- **Proposed Correction**: Add standard library codecs to the microbenchmark suite as reference points.

### 24. oracle comparison
- **Severity**: HIGH
- **File**: `experiments/run_experiments.py`
- **Problem**: `LinUCB` regret is not tracked. There is no `PER-MESSAGE-ORACLE` baseline to identify the theoretical maximum performance bound.
- **Proposed Correction**: Compute the true optimal action cost for each test message in hindsight to plot cumulative regret properly.

### 25. experimental artifact completeness
- **Severity**: HIGH
- **File**: `experiments/run_experiments.py`
- **Problem**: Aggregated JSON is saved, but not raw, per-message, per-seed CSV/Parquet data.
- **Proposed Correction**: Export `raw_results.csv` and `per_message_results.parquet` before any aggregation occurs.
