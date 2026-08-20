import unittest
import numpy as np
from src.engine import CompressionEngine
from src.features import FeatureExtractor
from src.manager import BatchCacheManager
from src.bandit import LinUCBBandit
from src.scheduler import HeuristicScheduler

class TestCompressionFramework(unittest.TestCase):
    def setUp(self):
        self.engine = CompressionEngine()
        self.manager = BatchCacheManager(cache_capacity=10)
        self.feature_extractor = FeatureExtractor()
        self.bandit = LinUCBBandit(d=9, K=6, alpha=0.5)

    def test_compressors(self):
        """Tests that all classical compressors achieve correct lossless recovery."""
        test_strings = [
            "hello world!",
            "adaptive compression scheduling is novel",
            "emoji test: 🌟🔥🚀",
            "A" * 100  # Highly repetitive to check compression ratio
        ]
        codecs = ['SKIP', 'ZSTD', 'BROTLI', 'GZIP', 'LZ4']
        
        for codec in codecs:
            for text in test_strings:
                comp, comp_lat = self.engine.compress(text, codec)
                decomp, decomp_lat = self.engine.decompress(comp, codec)
                self.assertEqual(text, decomp, f"Lossless check failed for codec: {codec}")
                self.assertGreaterEqual(comp_lat, 0.0)
                self.assertGreaterEqual(decomp_lat, 0.0)

    def test_features(self):
        """Tests correctness of feature extractor dimensions and value ranges."""
        text = "Hello, world! This is a test message. 🚀"
        timestamp = 1718000000.0
        
        feats = self.feature_extractor.extract_features(text, timestamp)
        
        # Verify keys
        expected_keys = {
            "char_len", "word_len", "entropy", "repetition_score", "arrival_rate",
            "uppercase_ratio", "punctuation_ratio", "emoji_ratio", "unique_word_ratio"
        }
        self.assertEqual(set(feats.keys()), expected_keys)
        
        # Verify ranges
        self.assertEqual(feats["char_len"], len(text))
        self.assertGreater(feats["entropy"], 0.0)
        self.assertLessEqual(feats["entropy"], 8.0)
        self.assertGreaterEqual(feats["repetition_score"], 0.0)
        self.assertLessEqual(feats["repetition_score"], 1.0)
        self.assertGreaterEqual(feats["arrival_rate"], 0.0)
        self.assertGreaterEqual(feats["uppercase_ratio"], 0.0)
        self.assertLessEqual(feats["uppercase_ratio"], 1.0)
        self.assertGreater(feats["punctuation_ratio"], 0.0)
        self.assertGreater(feats["emoji_ratio"], 0.0)

    def test_cache_reuse_is_lossless(self):
        """Asserts exact-match cache hits retrieve byte-identical text (lossless correctness)."""
        msg = "critical authentication otp: 9384"
        codec = 'ZSTD'
        comp_bytes, _ = self.engine.compress(msg, codec)
        
        # Store in cache
        self.manager.cache_store(msg, comp_bytes, codec)
        
        # Lookup
        lookup_res = self.manager.cache_lookup(msg)
        self.assertIsNotNone(lookup_res)
        
        retrieved_bytes, retrieved_codec = lookup_res
        self.assertEqual(retrieved_codec, codec)
        
        decomp, _ = self.engine.decompress(retrieved_bytes, retrieved_codec)
        self.assertEqual(decomp, msg, "Decompressed cache payload does not match original message!")

    def test_bandit_learning_step(self):
        """Verifies that the contextual bandit model updates its state after a feedback step."""
        feats = {
            "char_len": 50, "word_len": 8, "entropy": 3.4, "repetition_score": 0.0, "arrival_rate": 2.0,
            "uppercase_ratio": 0.05, "punctuation_ratio": 0.1, "emoji_ratio": 0.0, "unique_word_ratio": 1.0
        }
        
        # Action prediction
        action_idx, pred_lat = self.bandit.predict(feats)
        self.assertIn(action_idx, list(range(6)))
        self.assertGreater(pred_lat, 0.0)
        
        # Record pre-update covariance matrix elements
        pre_A = self.bandit.A[action_idx].copy()
        
        # Update reward
        self.bandit.update(action_idx, feats, reward=10.0)
        
        # Check that covariance matrix has changed
        post_A = self.bandit.A[action_idx]
        self.assertFalse(np.array_equal(pre_A, post_A), "Bandit covariance matrix did not update after training feedback!")

if __name__ == "__main__":
    unittest.main()
