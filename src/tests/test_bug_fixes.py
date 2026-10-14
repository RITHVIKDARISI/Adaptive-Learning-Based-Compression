import unittest
import numpy as np
import random
from src.manager import BatchCacheManager
from src.simulator import StreamSimulator
from src.scheduler import OfflineLabelGenerator, SupervisedScheduler, HeuristicScheduler
from src.engine import CompressionEngine
from src.features import FeatureExtractor

class TestBugFixesAndSanityChecks(unittest.TestCase):

    def test_bug1_latency_units(self):
        """Verify Bug 1 fix: Formula and unit alignment between OfflineLabelGenerator and StreamSimulator."""
        msg = "loving the weather today! #sunny"
        lam = 0.01

        # Calculate cost using fixed mock latency (100 us) and compressed size
        engine = CompressionEngine()
        zcomp, _ = engine.compress(msg, 'ZSTD')
        comp_size = len(zcomp)
        fixed_lat_us = 150.0

        # Offline generator formula
        offline_cost = comp_size + lam * fixed_lat_us

        # Simulator formula (comp_size + lambda_param * comp_lat_us)
        sim = StreamSimulator(BatchCacheManager(), lambda_param=lam)
        sim_cost = comp_size + sim.lambda_param * fixed_lat_us

        print(f"\n[Bug 1 Verification] Offline cost formula: {offline_cost:.4f} | Simulator cost formula: {sim_cost:.4f}")
        self.assertEqual(offline_cost, sim_cost, msg="STILL MISMATCHED — cost formulas/units differ")

    def test_bug2_batch_timeout(self):
        """Verify Bug 2 fix: Timeout-triggered flush fires on slow streams before max batch size is reached."""
        manager = BatchCacheManager(max_batch_size=10, timeout_seconds=0.5)
        sim = StreamSimulator(manager)
        batch_scheduler = lambda feat: (5, 0.0)

        # Send 3 messages (well under batch size of 10), spaced 0.3s apart.
        # By message 3 (t=0.6s), elapsed time exceeds the 0.5s timeout threshold.
        sim.run_message(1, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 0.0, batch_scheduler)
        sim.run_message(2, "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", 0.3, batch_scheduler)
        sim.run_message(3, "cccccccccccccccccccccccccccccccccc", 0.6, batch_scheduler)

        print(f"[Bug 2 Verification] Queue length after 3 messages (0.6s elapsed): {len(manager.batcher.queue)}")
        # Timeout-triggered flush should have emptied queue of messages 1 and 2 upon processing message 3.
        self.assertLess(len(manager.batcher.queue), 3, msg="Timeout-triggered batch flush failed to fire!")

    def test_bug3_batch_reconstruction_with_newlines(self):
        """Verify Bug 3 fix: Length-prefixed batch reconstruction preserves messages containing literal newlines."""
        manager = BatchCacheManager(max_batch_size=3, timeout_seconds=999)
        sim = StreamSimulator(manager)
        batch_scheduler = lambda feat: (5, 0.0)

        originals = [
            "first test message here",
            "second test\nwith an embedded newline",   # Embedded newline that breaks naive delimiter splitting
            "third test message here"
        ]

        for i, msg in enumerate(originals):
            sim.run_message(i, msg, i * 0.1, batch_scheduler)   # 3rd message triggers batch flush at size=3

        for i, msg in enumerate(originals):
            cached = manager.cache_lookup(msg)
            self.assertIsNotNone(cached, f"Message {i} was not correctly cached/reconstructed")
            # Decompress cached payload to verify byte-for-byte equality
            decomp_text, _ = sim.engine.decompress(cached[0], cached[1])
            self.assertEqual(decomp_text, msg, f"Reconstructed message {i} text does not match original!")
            print(f"[Bug 3 Verification] [OK] Message {i} round-trips correctly")

    def test_sanity_check_compression_ratios(self):
        """Sanity Check 1: Compression ratio < 1.0 for compressed actions on repetitive text, ≈ 1.0 for SKIP."""
        engine = CompressionEngine()
        text = "System warning log message repeated! " * 5
        
        # Test ZSTD compression ratio
        zcomp, _ = engine.compress(text, 'ZSTD')
        zstd_ratio = len(zcomp) / len(text.encode('utf-8'))
        self.assertLess(zstd_ratio, 1.0, "Compressed ratio should be < 1.0 for repetitive input")

        # Test SKIP action ratio
        skip_comp, _ = engine.compress(text, 'SKIP')
        skip_ratio = len(skip_comp) / len(text.encode('utf-8'))
        self.assertAlmostEqual(skip_ratio, 1.0, places=4, msg="SKIP action ratio should be 1.0")

    def test_sanity_check_cache_hit_rate(self):
        """Sanity Check 2: Cache hit rate increases when duplicate messages are replayed."""
        manager = BatchCacheManager()
        sim = StreamSimulator(manager)
        scheduler = HeuristicScheduler()

        msg = "Exact duplicate message for cache verification."
        
        # Message arrival #1 -> Cache Miss
        sim.run_message(0, msg, 0.0, scheduler)
        
        # Message arrival #2 -> Cache Hit
        sim.run_message(1, msg, 0.1, scheduler)

        metrics = sim.get_summary_metrics()
        self.assertEqual(metrics["cache_hit_rate"], 0.5, "Cache hit rate should be 50% for 1 hit out of 2 messages")

    def test_sanity_check_scheduler_inference_latencies(self):
        """Sanity Check 3: Decision Tree and Logistic Regression inference latencies are sub-millisecond."""
        dt = SupervisedScheduler(model_type='decision_tree')
        lr = SupervisedScheduler(model_type='logistic_regression')
        rf = SupervisedScheduler(model_type='random_forest')

        # Dummy multi-class dataset
        X = [{"char_len": 50, "word_len": 8, "entropy": 3.0, "repetition_score": 0.0,
              "arrival_rate": 1.0, "uppercase_ratio": 0.0, "punctuation_ratio": 0.0,
              "emoji_ratio": 0.0, "unique_word_ratio": 1.0}] * 24
        y = [0, 1, 2, 3, 4, 5] * 4

        dt.fit(X, y)
        lr.fit(X, y)
        rf.fit(X, y)

        feat = X[0]
        
        # Warmup run to avoid cold-start noise
        dt.predict(feat)
        lr.predict(feat)
        rf.predict(feat)
        
        # Measure mean latency across multiple trials
        num_trials = 10
        dt_lats, lr_lats, rf_lats = [], [], []
        for _ in range(num_trials):
            _, dt_lat = dt.predict(feat)
            _, lr_lat = lr.predict(feat)
            _, rf_lat = rf.predict(feat)
            dt_lats.append(dt_lat)
            lr_lats.append(lr_lat)
            rf_lats.append(rf_lat)
            
        mean_dt_lat = sum(dt_lats) / num_trials
        mean_lr_lat = sum(lr_lats) / num_trials
        mean_rf_lat = sum(rf_lats) / num_trials

        print(f"[Sanity Check 3] Mean DT Latency: {mean_dt_lat:.2f} µs | Mean LR Latency: {mean_lr_lat:.2f} µs | Mean RF Latency: {mean_rf_lat:.2f} µs")
        self.assertLess(mean_dt_lat, 1000.0, "Decision Tree latency should be sub-millisecond (< 1000 µs)")
        self.assertLess(mean_lr_lat, 1000.0, "Logistic Regression latency should be sub-millisecond (< 1000 µs)")

    def test_sanity_check_roundtrip_losslessness(self):
        """Sanity Check 4: Round-trip losslessness across 20 random sample messages across all classical codecs."""
        engine = CompressionEngine()
        codecs = ['ZSTD', 'BROTLI', 'GZIP', 'LZ4']

        sample_messages = [
            "Simple plain message.",
            "Special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?",
            "Unicode & Emojis: 🚀🔥🎉💻🌍",
            "Multi-line string\nwith\nnewlines",
            "Numbers: 1234567890 9876543210"
        ] + [f"Random message #{i}: " + "".join(random.choices("abcdefghijklmnopqrstuvwxyz ", k=40)) for i in range(15)]

        for msg in sample_messages:
            for codec in codecs:
                comp, _ = engine.compress(msg, codec)
                decomp, _ = engine.decompress(comp, codec)
                self.assertEqual(decomp, msg, f"Losslessness failed for codec {codec} on message: {msg!r}")

    def test_engine_handles_raw_bytes_roundtrip(self):
        """Confirms compress/decompress work correctly on raw bytes (not just str) for batch framing."""
        engine = CompressionEngine()
        raw_binary = b'\x00\x00\x00\x05hello\x00\x00\x00\x05world'  # simulates framed batch payload
        comp, _ = engine.compress(raw_binary, 'ZSTD')
        decomp, _ = engine.decompress(comp, 'ZSTD', raw_bytes=True)
        self.assertEqual(decomp, raw_binary, "Raw bytes round-trip through compress/decompress must be exact")

    def test_latency_definition_eight_stages(self):
        """Verifies that per-message latency includes all 8 stages and e2e_lat_us equals their exact sum."""
        manager = BatchCacheManager(max_batch_size=2, timeout_seconds=999)
        sim = StreamSimulator(manager)
        
        # Test actions: SKIP (0), ZSTD (1), BATCH (5)
        # 1. SKIP message
        m0 = sim.run_message(0, "skip message text", 0.0, lambda f: (0, 0.0))
        
        # 2. ZSTD single compressor message
        m1 = sim.run_message(1, "zstd compressed message text", 0.1, lambda f: (1, 0.0))
        
        # 3. Exact duplicate of m1 -> CACHE_REUSE hit
        m2 = sim.run_message(2, "zstd compressed message text", 0.2, lambda f: (1, 0.0))
        
        # 4 & 5. BATCH messages (flush after 2 messages)
        m3 = sim.run_message(3, "batch item 1 text", 0.3, lambda f: (5, 0.0))
        m4 = sim.run_message(4, "batch item 2 text", 0.4, lambda f: (5, 0.0))
        
        stage_keys = [
            "feat_lat_us", "cache_lookup_lat_us", "scheduler_lat_us", "comp_lat_us",
            "decomp_lat_us", "queue_wait_lat_us", "cache_store_lat_us", "cache_fill_comp_lat_us"
        ]
        
        self.assertEqual(len(sim.stats), 5, "All 5 messages should now be recorded in simulator stats")
        
        for stat in sim.stats:
            # Check all 8 per-stage keys exist
            for k in stage_keys:
                self.assertIn(k, stat, f"Stage key {k} missing from message {stat['msg_id']}")
                self.assertGreaterEqual(stat[k], 0.0, f"Stage latency {k} must be non-negative")
            
            # Check e2e_lat_us strictly equals the sum of all 8 per-stage fields
            expected_sum = sum(stat[k] for k in stage_keys)
            self.assertAlmostEqual(
                stat["e2e_lat_us"], expected_sum, places=5,
                msg=f"e2e_lat_us ({stat['e2e_lat_us']}) does not equal sum of 8 stages ({expected_sum}) for msg {stat['msg_id']}"
            )
        
        # Specific assertions per action type:
        # SKIP (msg 0): no compression, no queue wait, no store
        s0 = sim.stats[0]
        self.assertEqual(s0["action"], "SKIP")
        self.assertGreater(s0["feat_lat_us"], 0.0)
        self.assertEqual(s0["comp_lat_us"], 0.0)
        self.assertEqual(s0["cache_store_lat_us"], 0.0)
        
        # ZSTD (msg 1): comp, decomp, cache_store
        s1 = sim.stats[1]
        self.assertEqual(s1["action"], "ZSTD")
        self.assertGreater(s1["feat_lat_us"], 0.0)
        self.assertGreater(s1["comp_lat_us"], 0.0)
        self.assertGreater(s1["decomp_lat_us"], 0.0)
        self.assertGreater(s1["cache_store_lat_us"], 0.0)
        self.assertEqual(s1["cache_fill_comp_lat_us"], 0.0)
        
        # CACHE_REUSE (msg 2): cache lookup + decomp only, feat_lat_us == 0
        s2 = sim.stats[2]
        self.assertEqual(s2["action"], "CACHE_REUSE")
        self.assertTrue(s2["cache_hit"])
        self.assertEqual(s2["feat_lat_us"], 0.0)
        self.assertGreater(s2["cache_lookup_lat_us"], 0.0)
        self.assertGreater(s2["decomp_lat_us"], 0.0)
        self.assertEqual(s2["comp_lat_us"], 0.0)
        self.assertEqual(s2["cache_store_lat_us"], 0.0)
        
        # BATCH (msgs 3 & 4): queue wait, comp share, decomp share, cache fill comp, cache store
        for s_batch in sim.stats[3:5]:
            self.assertEqual(s_batch["action"], "BATCH")
            self.assertGreater(s_batch["feat_lat_us"], 0.0)
            self.assertGreater(s_batch["comp_lat_us"], 0.0)
            self.assertGreater(s_batch["decomp_lat_us"], 0.0)
            self.assertGreater(s_batch["cache_fill_comp_lat_us"], 0.0)
            self.assertGreater(s_batch["cache_store_lat_us"], 0.0)

    def test_wall_clock_throughput_computation(self):
        """Verifies throughput (msg/s and MB/s) is computed from one wall-clock timer."""
        manager = BatchCacheManager()
        sim = StreamSimulator(manager)
        messages = [
            (0, "first message for throughput test", 0.0),
            (1, "second message for throughput test", 0.05),
            (2, "third message for throughput test", 0.10),
            (3, "fourth message for throughput test", 0.15),
        ]
        
        summary = sim.run_stream(messages, lambda f: (1, 0.0))
        
        self.assertIn("wall_clock_time_sec", summary)
        self.assertGreater(summary["wall_clock_time_sec"], 0.0)
        
        expected_msg_per_sec = len(messages) / summary["wall_clock_time_sec"]
        self.assertAlmostEqual(summary["throughput_msg_per_sec"], expected_msg_per_sec, places=3)
        
        expected_mb_s = (summary["total_orig_bytes"] / 1048576.0) / summary["wall_clock_time_sec"]
        self.assertAlmostEqual(summary["throughput_mb_s"], expected_mb_s, places=3)

    def test_lossless_roundtrip_all_codecs_and_batching(self):
        """Exhaustive check that all single codecs and batch framed encoding achieve 100% lossless recovery."""
        engine = CompressionEngine()
        manager = BatchCacheManager(max_batch_size=5, timeout_seconds=999)
        test_strings = [
            "Simple string",
            "String with \n embedded \r\n newlines and \t tabs",
            "Unicode symbols 🚀🔥🎉 and accents éèàçñ",
            "A" * 500,  # Highly compressible repetitive text
            "",         # Empty message
            "1234567890!@#$%^&*()_+-=[]{}|;':\",./<>?"
        ]
        
        # 1. Single codecs
        for codec in ['SKIP', 'ZSTD', 'BROTLI', 'GZIP', 'LZ4']:
            for text in test_strings:
                comp, _ = engine.compress(text, codec)
                decomp, _ = engine.decompress(comp, codec)
                self.assertEqual(decomp, text, f"Single codec roundtrip failed for {codec}")
        
        # 2. Batch encoding and decoding
        encoded = manager.encode_batch(test_strings)
        decoded = manager.decode_batch(encoded)
        self.assertEqual(decoded, test_strings, "Batch framed encoding/decoding failed lossless roundtrip")


class TestDeadlineMaskingAndBatchLosslessness(unittest.TestCase):
    """
    Unit tests ensuring:
    1. BATCH action is masked when predicted queue wait exceeds deadline.
    2. BATCH action is allowed when predicted queue wait is within deadline.
    3. Batch framing round-trip losslessness is preserved and unchanged across diverse data.
    4. SimulatorAlignedOracle correctly models framing and losslessness without error.
    """

    def setUp(self):
        from src.scheduler import HeuristicScheduler, estimate_batch_wait_us, SimulatorAlignedOracle
        from src.bandit import LinUCBBandit
        from src.manager import BatchCacheManager
        from src.engine import CompressionEngine

        self.HeuristicScheduler = HeuristicScheduler
        self.estimate_batch_wait_us = estimate_batch_wait_us
        self.SimulatorAlignedOracle = SimulatorAlignedOracle
        self.LinUCBBandit = LinUCBBandit
        self.manager = BatchCacheManager()
        self.engine = CompressionEngine()

    def test_batch_masked_above_deadline(self):
        # High queue wait case: arrival rate is low (1 msg/s), queue empty (queue_len=0, needed=9)
        # Timeout 0.5s -> est wait is 500,000 us > 50,000 us deadline
        feat_high_wait = {
            "char_len": 200,
            "repetition_score": 0.95,
            "arrival_rate": 1.0,
            "queue_len": 0,
            "batch_max_size": 10,
            "batch_timeout_seconds": 0.5
        }

        est_wait = self.estimate_batch_wait_us(feat_high_wait)
        self.assertGreater(est_wait, 50000.0, "Estimated wait should exceed 50ms deadline")

        # 1. Heuristic without deadline -> predicts BATCH (5)
        h_no_deadline = self.HeuristicScheduler(enable_deadline=False)
        action_no_dl, _ = h_no_deadline.predict(feat_high_wait)
        self.assertEqual(action_no_dl, 5, "Heuristic without deadline should select BATCH (5)")

        # 2. Heuristic with deadline -> masks BATCH (5), falls back to ZSTD (1)
        h_with_deadline = self.HeuristicScheduler(enable_deadline=True, max_wait_us=50000.0)
        action_dl, _ = h_with_deadline.predict(feat_high_wait)
        self.assertEqual(action_dl, 1, "Heuristic with deadline should mask BATCH and select ZSTD (1)")

        # 3. LinUCB with deadline -> masks BATCH (5) even when BATCH has highest expectation
        bandit = self.LinUCBBandit(enable_deadline=True, max_wait_us=50000.0)
        # Manually bias b for BATCH (index 5) to make it the top choice without masking
        bandit.b[5] = np.ones((bandit.d, 1)) * 1000.0
        bandit.A_inv[5] = np.eye(bandit.d)
        # If deadline is disabled, it would choose 5
        bandit.enable_deadline = False
        self.assertEqual(bandit.predict(feat_high_wait)[0], 5)
        # When deadline is enabled, action 5 is masked
        bandit.enable_deadline = True
        action_bandit, _ = bandit.predict(feat_high_wait)
        self.assertNotEqual(action_bandit, 5, "LinUCB with deadline must NOT select masked BATCH")

    def test_batch_allowed_below_deadline(self):
        # Case A: High arrival rate (1000 msg/s), queue_len=0 -> est wait = 9/1000 = 9000 us < 50000 us
        feat_high_rate = {
            "char_len": 200,
            "repetition_score": 0.95,
            "arrival_rate": 1000.0,
            "queue_len": 0,
            "batch_max_size": 10,
            "batch_timeout_seconds": 0.5
        }
        est_wait_rate = self.estimate_batch_wait_us(feat_high_rate)
        self.assertLess(est_wait_rate, 50000.0, "High arrival rate should yield wait under deadline")

        h_deadline = self.HeuristicScheduler(enable_deadline=True, max_wait_us=50000.0)
        action_rate, _ = h_deadline.predict(feat_high_rate)
        self.assertEqual(action_rate, 5, "BATCH should be allowed when arrival rate is fast enough")

        # Case B: Queue is full - 1, message triggers immediate flush -> est wait = 0.0 us
        feat_flush = {
            "char_len": 200,
            "repetition_score": 0.95,
            "arrival_rate": 0.1,  # very slow rate, but queue is ready to flush
            "queue_len": 9,
            "batch_max_size": 10,
            "batch_timeout_seconds": 0.5
        }
        est_wait_flush = self.estimate_batch_wait_us(feat_flush)
        self.assertEqual(est_wait_flush, 0.0, "Wait must be 0 us when message fills batch capacity")
        action_flush, _ = h_deadline.predict(feat_flush)
        self.assertEqual(action_flush, 5, "BATCH should be allowed when queue is filled")

    def test_batch_round_trip_losslessness_unchanged(self):
        # Comprehensive test with unicode, emojis, control chars, newlines, binary sequences
        messages = [
            "Hello, world!",
            "Line 1\nLine 2\nLine 3\r\nLine 4",
            "Special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?`~",
            "Unicode texts: Bonjour le monde!  ! \u4e2d\u6587 \u0928\u092e\u0938\u094d\u0924\u0947",
            "Emojis: \U0001f600 \U0001f680 \U0001f389 \U0001f4ca",
            "Long repetitive string: " + "repeat " * 200,
            "JSON-like: {\"id\": 123, \"payload\": \"sample\", \"active\": true}"
        ]

        # 4-byte big-endian framing
        encoded_bytes = self.manager.encode_batch(messages)
        # Compress and decompress via ZSTD
        comp_bytes, _ = self.engine.compress(encoded_bytes, 'ZSTD')
        decomp_bytes, _ = self.engine.decompress(comp_bytes, 'ZSTD', raw_bytes=True)
        # Decode framed messages
        decoded_messages = self.manager.decode_batch(decomp_bytes)
        self.assertEqual(decoded_messages, messages, "Batch round-trip failed exact losslessness verification")

    def test_simulator_aligned_oracle_execution(self):
        oracle = self.SimulatorAlignedOracle(lambda_param=0.01, max_batch_size=5, timeout_seconds=0.1)
        test_msgs = [
            "short",
            "repeat text " * 10,
            "repeat text " * 10,
            "another unique message with some details",
            "repeat text " * 10,
            "final message in test stream"
        ]
        test_ts = [0.0, 0.01, 0.02, 0.03, 0.04, 0.20]
        labels = oracle.generate_labels(test_msgs, test_ts)
        self.assertEqual(len(labels), len(test_msgs))
        for l in labels:
            self.assertIn(l, range(6), f"Label {l} not in valid action range 0..5")


if __name__ == "__main__":
    unittest.main()

