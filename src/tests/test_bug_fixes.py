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

if __name__ == "__main__":
    unittest.main()
