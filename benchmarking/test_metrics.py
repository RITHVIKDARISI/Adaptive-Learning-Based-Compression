import unittest
from benchmarking.metrics import MessageMetrics
from benchmarking.instrumentation import verify_lossless_reconstruction

class TestMetrics(unittest.TestCase):
    def test_compression_factor(self):
        # 100 bytes down to 50 bytes -> ratio 2.0
        m = MessageMetrics(msg_id=1, original_bytes=100, encoded_bytes=50)
        self.assertEqual(m.compression_factor, 2.0)
        self.assertEqual(m.compressed_fraction, 0.5)
        self.assertEqual(m.space_saving_pct, 50.0)

    def test_expansion(self):
        # 10 bytes expanded to 30 bytes (Zstd header overhead)
        m = MessageMetrics(msg_id=2, original_bytes=10, encoded_bytes=30)
        self.assertEqual(m.compression_factor, 10/30)
        self.assertEqual(m.compressed_fraction, 3.0)
        self.assertEqual(m.space_saving_pct, -200.0)

    def test_zero_bytes(self):
        # Edge cases
        m = MessageMetrics(msg_id=3, original_bytes=0, encoded_bytes=0)
        self.assertEqual(m.compression_factor, 1.0)
        self.assertEqual(m.compressed_fraction, 1.0)
        self.assertEqual(m.space_saving_pct, 0.0)

    def test_latency_summation(self):
        m = MessageMetrics(
            msg_id=4,
            original_bytes=100,
            encoded_bytes=50,
            feature_latency_us=10.0,
            cache_lookup_latency_us=2.0,
            scheduler_inference_latency_us=15.0,
            compression_latency_us=30.0,
            cache_store_latency_us=5.0,
            queue_wait_latency_us=500.0
        )
        # Compute latency: 10 + 2 + 15 + 30 + 5 = 62.0
        self.assertEqual(m.total_compute_latency_us, 62.0)
        # E2E latency: 62.0 + 500 = 562.0
        self.assertEqual(m.e2e_latency_us, 562.0)

    def test_lossless_reconstruction(self):
        original = "Hello World! 👋"
        # Correct bytes
        self.assertTrue(verify_lossless_reconstruction(original, original.encode('utf-8')))
        # Incorrect bytes
        self.assertFalse(verify_lossless_reconstruction(original, b"Hello World!"))

if __name__ == '__main__':
    unittest.main()
