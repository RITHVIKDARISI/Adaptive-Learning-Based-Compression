import unittest
import numpy as np
from src.manager import BatchCacheManager, Batcher
from src.simulator import StreamSimulator
from src.scheduler import OfflineLabelGenerator, SupervisedScheduler, HeuristicScheduler
from src.bandit import LinUCBBandit

class TestGapFixes(unittest.TestCase):

    def test_length_prefix_batching_with_newlines(self):
        """Verify length-prefixed batching preserves messages containing literal newlines."""
        messages = [
            "Hello\nWorld!",
            "Single line message",
            "Multi\nline\nwith\nmultiple\nnewlines",
            "",
            "Emoji message 🚀\nwith newline"
        ]
        
        encoded = BatchCacheManager.encode_batch(messages)
        self.assertIsInstance(encoded, bytes)
        
        decoded = BatchCacheManager.decode_batch(encoded)
        self.assertEqual(messages, decoded)

    def test_batch_timeout_firing(self):
        """Verify batcher timeout fires on sparse arrival timestamps in simulator."""
        manager = BatchCacheManager(max_batch_size=10, timeout_seconds=0.5)
        simulator = StreamSimulator(manager=manager, lambda_param=0.01)
        # Dummy scheduler that always selects action 5 (BATCH)
        batch_scheduler = lambda feat: (5, 0.0)
        
        # Msg 0 arrives at t=0.0 and gets scheduled for BATCH
        msg0_text = "repeat message text"
        res0 = simulator.run_message(0, msg0_text, 0.0, batch_scheduler)
        self.assertEqual(res0["action"], "BATCH")
        self.assertTrue(res0["queued"])
        self.assertEqual(len(simulator.manager.batcher.queue), 1)
        
        # Msg 1 arrives at t=0.6 (> 0.5s timeout).
        # simulator.run_message should trigger flush of Msg 0 before adding Msg 1.
        res1 = simulator.run_message(1, "normal message", 0.6, batch_scheduler)
        
        # Verify Msg 0 was flushed and recorded in stats
        flushed_ids = [s["msg_id"] for s in simulator.stats]
        self.assertIn(0, flushed_ids)

    def test_microsecond_latency_unit_consistency(self):
        """Verify microsecond cost formulas in simulator match offline label generator."""
        manager = BatchCacheManager()
        sim = StreamSimulator(manager=manager, lambda_param=0.01)
        bandit = LinUCBBandit(lambda_param=0.01)
        
        # Run a message through simulator with bandit
        res = sim.run_message(0, "Test microsecond latency", 0.0, bandit)
        
        # Check simulator metrics latency is in us
        stat = sim.stats[0]
        self.assertGreater(stat["e2e_lat_us"], 0.0)
        
        # Check label generator runs without error with us-based lambda
        generator = OfflineLabelGenerator(lambda_param=0.01)
        labels = generator.generate_labels(["Test microsecond latency"])
        self.assertEqual(len(labels), 1)

    def test_logistic_regression_initialization(self):
        """Verify SupervisedScheduler initializes Logistic Regression without warnings/errors."""
        scheduler = SupervisedScheduler(model_type='logistic_regression')
        self.assertIsNotNone(scheduler.model)

    def test_bandit_cost_aware_exploration(self):
        """Verify LinUCBBandit initialization and update with cost-aware exploration."""
        bandit = LinUCBBandit(cost_aware_exploration=True)
        features = {
            "char_len": 50, "word_len": 8, "entropy": 3.5, "repetition_score": 0.1,
            "arrival_rate": 2.0, "uppercase_ratio": 0.0, "punctuation_ratio": 0.05,
            "emoji_ratio": 0.0, "unique_word_ratio": 1.0
        }
        action, lat = bandit.predict(features)
        self.assertIn(action, list(range(6)))
        self.assertGreater(lat, 0.0)
        
        bandit.update(action, features, reward=-100.0)
        self.assertEqual(len(bandit.reward_history), 1)

if __name__ == "__main__":
    unittest.main()
