import time
import numpy as np
from src.engine import CompressionEngine
from src.features import FeatureExtractor
from src.manager import BatchCacheManager

class StreamSimulator:
    def __init__(self, manager: BatchCacheManager, lambda_param: float = 0.01):
        """
        Stream Simulator.
        Args:
            manager: BatchCacheManager instance.
            lambda_param: Cost weight for latency (bytes/us).
        """
        self.manager = manager
        self.lambda_param = lambda_param
        self.engine = CompressionEngine()
        self.feature_extractor = FeatureExtractor()
        
        # Action map matching scheduler.py
        self.action_names = {
            0: 'SKIP',
            1: 'ZSTD',
            2: 'BROTLI',
            3: 'GZIP',
            4: 'LZ4',
            5: 'BATCH'
        }
        self.reset()

    def reset(self):
        self.manager.reset()
        self.feature_extractor.reset()
        self.stats = []  # List of metrics per message
        self.pending_batch_metrics = {}  # msg_id -> dict with partially filled metrics
        self.wall_clock_time_sec = None

    def run_message(self, msg_id: int, text: str, timestamp: float, scheduler) -> dict:
        """
        Simulates an incoming message.
        Args:
            msg_id: Unique message identifier.
            text: Raw message content.
            timestamp: Epoch arrival timestamp in seconds.
            scheduler: A scheduler model with a `.predict(features_dict)` method, or a bandit.
        Returns:
            Dict containing immediate metrics for the message (might be marked as queued for BATCH).
        """
        # 0. Check Batch Queue Timeout before processing current message
        if self.manager.batcher.should_flush(timestamp):
            self.flush_batch(timestamp, scheduler)

        # 1. Check Exact-Match Cache Lookup
        cache_start = time.perf_counter()
        cache_hit = self.manager.cache_lookup(text)
        cache_lookup_lat_us = (time.perf_counter() - cache_start) * 1_000_000

        if cache_hit is not None:
            compressed_bytes, codec = cache_hit
            # Verify losslessness by decompressing
            decomp_text, decomp_lat_us = self.engine.decompress(compressed_bytes, codec)

            feat_lat_us = 0.0
            scheduler_lat_us = 0.0
            comp_lat_us = 0.0
            queue_wait_lat_us = 0.0
            cache_store_lat_us = 0.0
            cache_fill_comp_lat_us = 0.0

            e2e_lat_us = (
                feat_lat_us
                + cache_lookup_lat_us
                + scheduler_lat_us
                + comp_lat_us
                + decomp_lat_us
                + queue_wait_lat_us
                + cache_store_lat_us
                + cache_fill_comp_lat_us
            )
            old_e2e_lat_us = cache_lookup_lat_us + decomp_lat_us

            metrics = {
                "msg_id": msg_id,
                "text": text,
                "timestamp": timestamp,
                "action": "CACHE_REUSE",
                "cache_hit": True,
                "feat_lat_us": feat_lat_us,
                "cache_lookup_lat_us": cache_lookup_lat_us,
                "scheduler_lat_us": scheduler_lat_us,
                "comp_lat_us": comp_lat_us,
                "decomp_lat_us": decomp_lat_us,
                "queue_wait_lat_us": queue_wait_lat_us,
                "cache_store_lat_us": cache_store_lat_us,
                "cache_fill_comp_lat_us": cache_fill_comp_lat_us,
                "orig_size": len(text.encode('utf-8')),
                "comp_size": len(compressed_bytes),
                "queued": False,
                "e2e_lat_us": e2e_lat_us,
                "old_e2e_lat_us": old_e2e_lat_us,
                "compute_cost": len(compressed_bytes) + self.lambda_param * cache_lookup_lat_us,
                "user_perceived_cost": len(compressed_bytes) + self.lambda_param * e2e_lat_us,
            }
            self.stats.append(metrics)
            return metrics

        # 2. Extract Features (Cache Miss)
        feat_start = time.perf_counter()
        features = self.feature_extractor.extract_features(text, timestamp)
        feat_lat_us = (time.perf_counter() - feat_start) * 1_000_000

        # 3. Query Scheduler
        if hasattr(scheduler, 'predict'):
            action_idx, scheduler_lat_us = scheduler.predict(features)
        elif callable(scheduler):
            action_idx, scheduler_lat_us = scheduler(features)
        else:
            raise ValueError("Scheduler must either have a predict() method or be callable.")
        action_name = self.action_names.get(action_idx, 'SKIP')

        if action_name == 'SKIP':
            comp_lat_us = 0.0
            decomp_lat_us = 0.0
            queue_wait_lat_us = 0.0
            cache_store_lat_us = 0.0
            cache_fill_comp_lat_us = 0.0

            e2e_lat_us = (
                feat_lat_us
                + cache_lookup_lat_us
                + scheduler_lat_us
                + comp_lat_us
                + decomp_lat_us
                + queue_wait_lat_us
                + cache_store_lat_us
                + cache_fill_comp_lat_us
            )
            old_e2e_lat_us = cache_lookup_lat_us + scheduler_lat_us

            orig_size = len(text.encode('utf-8'))
            metrics = {
                "msg_id": msg_id,
                "text": text,
                "timestamp": timestamp,
                "action": "SKIP",
                "cache_hit": False,
                "feat_lat_us": feat_lat_us,
                "cache_lookup_lat_us": cache_lookup_lat_us,
                "scheduler_lat_us": scheduler_lat_us,
                "comp_lat_us": comp_lat_us,
                "decomp_lat_us": decomp_lat_us,
                "queue_wait_lat_us": queue_wait_lat_us,
                "cache_store_lat_us": cache_store_lat_us,
                "cache_fill_comp_lat_us": cache_fill_comp_lat_us,
                "orig_size": orig_size,
                "comp_size": orig_size,
                "queued": False,
                "e2e_lat_us": e2e_lat_us,
                "old_e2e_lat_us": old_e2e_lat_us,
                "compute_cost": orig_size + self.lambda_param * (feat_lat_us + cache_lookup_lat_us + scheduler_lat_us),
                "user_perceived_cost": orig_size + self.lambda_param * e2e_lat_us,
            }

            if hasattr(scheduler, 'update'):
                cost = metrics["comp_size"] + self.lambda_param * metrics["e2e_lat_us"]
                scheduler.update(action_idx, features, -cost)

            self.stats.append(metrics)
            return metrics

        elif action_name in ['ZSTD', 'BROTLI', 'GZIP', 'LZ4']:
            # Single classical compressor immediate action
            comp_bytes, comp_lat_us = self.engine.compress(text, action_name)
            decomp_text, decomp_lat_us = self.engine.decompress(comp_bytes, action_name)

            # Store in cache
            c_start = time.perf_counter()
            self.manager.cache_store(text, comp_bytes, action_name)
            cache_store_lat_us = (time.perf_counter() - c_start) * 1_000_000

            queue_wait_lat_us = 0.0
            cache_fill_comp_lat_us = 0.0

            e2e_lat_us = (
                feat_lat_us
                + cache_lookup_lat_us
                + scheduler_lat_us
                + comp_lat_us
                + decomp_lat_us
                + queue_wait_lat_us
                + cache_store_lat_us
                + cache_fill_comp_lat_us
            )
            old_e2e_lat_us = cache_lookup_lat_us + scheduler_lat_us + comp_lat_us + decomp_lat_us

            orig_size = len(text.encode('utf-8'))
            comp_size = len(comp_bytes)
            metrics = {
                "msg_id": msg_id,
                "text": text,
                "timestamp": timestamp,
                "action": action_name,
                "cache_hit": False,
                "feat_lat_us": feat_lat_us,
                "cache_lookup_lat_us": cache_lookup_lat_us,
                "scheduler_lat_us": scheduler_lat_us,
                "comp_lat_us": comp_lat_us,
                "decomp_lat_us": decomp_lat_us,
                "queue_wait_lat_us": queue_wait_lat_us,
                "cache_store_lat_us": cache_store_lat_us,
                "cache_fill_comp_lat_us": cache_fill_comp_lat_us,
                "orig_size": orig_size,
                "comp_size": comp_size,
                "queued": False,
                "e2e_lat_us": e2e_lat_us,
                "old_e2e_lat_us": old_e2e_lat_us,
                "compute_cost": comp_size + self.lambda_param * (
                    feat_lat_us + cache_lookup_lat_us + scheduler_lat_us + comp_lat_us + cache_store_lat_us
                ),
                "user_perceived_cost": comp_size + self.lambda_param * e2e_lat_us,
            }

            if hasattr(scheduler, 'update'):
                cost = metrics["comp_size"] + self.lambda_param * metrics["e2e_lat_us"]
                scheduler.update(action_idx, features, -cost)

            self.stats.append(metrics)
            return metrics

        elif action_name == 'BATCH':
            # Add to batch queue
            trigger_flush = self.manager.batcher.add(text, timestamp, msg_id)

            # Store features, timestamps, and scheduling latency for when the batch flushes
            self.pending_batch_metrics[msg_id] = {
                "msg_id": msg_id,
                "text": text,
                "timestamp": timestamp,
                "features": features,
                "action_idx": action_idx,
                "feat_lat_us": feat_lat_us,
                "cache_lookup_lat_us": cache_lookup_lat_us,
                "scheduler_lat_us": scheduler_lat_us,
            }

            if trigger_flush:
                self.flush_batch(timestamp, scheduler)

            return {"msg_id": msg_id, "action": "BATCH", "queued": True}

    def flush_batch(self, current_time: float, scheduler):
        """Flushes the batcher queue, compresses the framed batch, and computes shared costs."""
        batch = self.manager.batcher.flush()
        if not batch:
            return

        # batch element format: (text, timestamp, msg_id)
        messages_text = [item[0] for item in batch]
        framed_payload = self.manager.encode_batch(messages_text)

        # Compress batch using Zstd as default batch codec
        comp_bytes, comp_lat = self.engine.compress(framed_payload, 'ZSTD')
        decomp_bytes, decomp_lat = self.engine.decompress(comp_bytes, 'ZSTD', raw_bytes=True)

        # Explicit losslessness verification
        reconstructed_messages = self.manager.decode_batch(decomp_bytes)
        assert reconstructed_messages == messages_text, "Batch decompression losslessness verification failed!"

        batch_size = len(comp_bytes)
        num_messages = len(batch)
        total_orig_len = sum(len(item[0].encode('utf-8')) for item in batch)

        # Process metrics for each message in the batch
        for text, timestamp, msg_id in batch:
            pending = self.pending_batch_metrics.pop(msg_id)
            orig_size = len(text.encode('utf-8'))

            # Size share proportional to original length
            size_share = batch_size * (orig_size / max(total_orig_len, 1))

            queue_wait_lat_us = (current_time - timestamp) * 1_000_000
            comp_lat_share = comp_lat / num_messages
            decomp_lat_share = decomp_lat / num_messages

            # Store individually in exact-match cache for future hits (measures cache-fill comp and store time)
            indiv_comp, cache_fill_comp_lat_us = self.engine.compress(text, 'ZSTD')
            c_start = time.perf_counter()
            self.manager.cache_store(text, indiv_comp, 'ZSTD')
            cache_store_lat_us = (time.perf_counter() - c_start) * 1_000_000

            feat_lat_us = pending["feat_lat_us"]
            cache_lookup_lat_us = pending["cache_lookup_lat_us"]
            scheduler_lat_us = pending["scheduler_lat_us"]
            comp_lat_us = comp_lat_share
            decomp_lat_us = decomp_lat_share

            e2e_lat_us = (
                feat_lat_us
                + cache_lookup_lat_us
                + scheduler_lat_us
                + comp_lat_us
                + decomp_lat_us
                + queue_wait_lat_us
                + cache_store_lat_us
                + cache_fill_comp_lat_us
            )
            old_e2e_lat_us = (
                cache_lookup_lat_us
                + scheduler_lat_us
                + queue_wait_lat_us
                + comp_lat_share
                + decomp_lat_share
            )

            comp_size = int(size_share)
            compute_cost = comp_size + self.lambda_param * (
                feat_lat_us
                + cache_lookup_lat_us
                + scheduler_lat_us
                + comp_lat_us
                + cache_store_lat_us
                + cache_fill_comp_lat_us
            )
            user_perceived_cost = comp_size + self.lambda_param * e2e_lat_us

            metrics = {
                "msg_id": msg_id,
                "text": text,
                "timestamp": timestamp,
                "action": "BATCH",
                "cache_hit": False,
                "feat_lat_us": feat_lat_us,
                "cache_lookup_lat_us": cache_lookup_lat_us,
                "scheduler_lat_us": scheduler_lat_us,
                "comp_lat_us": comp_lat_us,
                "decomp_lat_us": decomp_lat_us,
                "queue_wait_lat_us": queue_wait_lat_us,
                "cache_store_lat_us": cache_store_lat_us,
                "cache_fill_comp_lat_us": cache_fill_comp_lat_us,
                "orig_size": orig_size,
                "comp_size": comp_size,
                "queued": False,
                "e2e_lat_us": e2e_lat_us,
                "old_e2e_lat_us": old_e2e_lat_us,
                "compute_cost": compute_cost,
                "user_perceived_cost": user_perceived_cost,
            }

            # Update bandit online reward if applicable (latency in us)
            if hasattr(scheduler, 'update'):
                cost = metrics["comp_size"] + self.lambda_param * metrics["e2e_lat_us"]
                scheduler.update(pending["action_idx"], pending["features"], -cost)

            self.stats.append(metrics)

    def finalize_stream(self, current_time: float, scheduler):
        """Forces flushing any remaining messages in the batch queue at the end of the stream."""
        if self.manager.batcher.queue:
            self.flush_batch(current_time, scheduler)

    def run_stream(self, stream_data, scheduler) -> dict:
        """
        Runs the entire stream of messages through the simulator.
        Measures total wall-clock time with time.perf_counter() around the whole stream loop.

        Args:
            stream_data: Iterable of items. Can be a pandas DataFrame (with msg_id, text, timestamp)
                         or an iterable of (msg_id, text, timestamp) tuples or dicts.
            scheduler: Scheduler model or callable.
        Returns:
            Summary metrics dict.
        """
        self.reset()

        if hasattr(stream_data, 'itertuples'):
            items = [(getattr(row, 'msg_id', idx), row.text, row.timestamp)
                     for idx, row in enumerate(stream_data.itertuples())]
        elif isinstance(stream_data, list):
            items = stream_data
        else:
            items = list(stream_data)

        t_start = time.perf_counter()
        last_timestamp = 0.0
        for item in items:
            if isinstance(item, tuple):
                m_id, txt, ts = item[0], item[1], item[2]
            elif isinstance(item, dict):
                m_id, txt, ts = item["msg_id"], item["text"], item["timestamp"]
            else:
                m_id, txt, ts = getattr(item, "msg_id"), getattr(item, "text"), getattr(item, "timestamp")
            last_timestamp = ts
            self.run_message(m_id, txt, ts, scheduler)

        self.finalize_stream(last_timestamp, scheduler)
        self.wall_clock_time_sec = time.perf_counter() - t_start
        return self.get_summary_metrics(wall_clock_time_sec=self.wall_clock_time_sec)

    def get_summary_metrics(self, wall_clock_time_sec: float = None) -> dict:
        """Computes aggregate performance metrics from simulated stats."""
        if not self.stats:
            return {}

        total_orig = sum(s["orig_size"] for s in self.stats)
        total_comp = sum(s["comp_size"] for s in self.stats)

        comp_ratio = total_orig / total_comp if total_comp > 0 else 1.0

        avg_e2e_lat = float(np.mean([s["e2e_lat_us"] for s in self.stats]))
        p50_e2e_lat = float(np.percentile([s["e2e_lat_us"] for s in self.stats], 50))
        p95_e2e_lat = float(np.percentile([s["e2e_lat_us"] for s in self.stats], 95))
        p99_e2e_lat = float(np.percentile([s["e2e_lat_us"] for s in self.stats], 99))

        avg_old_e2e_lat = float(np.mean([s.get("old_e2e_lat_us", s["e2e_lat_us"]) for s in self.stats]))

        avg_sched_lat = float(np.mean([s["scheduler_lat_us"] for s in self.stats]))
        p95_sched_lat = float(np.percentile([s["scheduler_lat_us"] for s in self.stats], 95))

        cache_hits = sum(1 for s in self.stats if s["action"] == "CACHE_REUSE")
        cache_hit_rate = cache_hits / len(self.stats)

        # Calculate throughput (messages per second & MB/s) from one time.perf_counter() timer
        if wall_clock_time_sec is not None:
            elapsed_sec = float(wall_clock_time_sec)
        elif self.wall_clock_time_sec is not None:
            elapsed_sec = float(self.wall_clock_time_sec)
        else:
            elapsed_sec = sum(s["e2e_lat_us"] for s in self.stats) / 1_000_000

        throughput = len(self.stats) / elapsed_sec if elapsed_sec > 0 else 0.0
        throughput_mb_s = (total_orig / 1048576.0) / elapsed_sec if elapsed_sec > 0 else 0.0

        import psutil
        peak_rss_mb = psutil.Process().memory_info().rss / 1048576.0

        total_compute_cost = sum(s["compute_cost"] for s in self.stats)
        total_user_cost = sum(s["user_perceived_cost"] for s in self.stats)

        cache_comp_savings = sum(s["orig_size"] - s["comp_size"] for s in self.stats if s["action"] == "CACHE_REUSE")

        return {
            "total_messages": len(self.stats),
            "total_orig_bytes": total_orig,
            "total_comp_bytes": total_comp,
            "compression_ratio": comp_ratio,
            "space_saving_pct": 100 * (1 - (total_comp / total_orig)) if total_orig > 0 else 0.0,
            "cache_compression_savings_bytes": cache_comp_savings,
            "mean_e2e_latency_us": avg_e2e_lat,
            "mean_old_e2e_latency_us": avg_old_e2e_lat,
            "p50_e2e_latency_us": p50_e2e_lat,
            "p95_e2e_latency_us": p95_e2e_lat,
            "p99_e2e_latency_us": p99_e2e_lat,
            "mean_feat_latency_us": float(np.mean([s["feat_lat_us"] for s in self.stats])),
            "mean_cache_lookup_latency_us": float(np.mean([s["cache_lookup_lat_us"] for s in self.stats])),
            "mean_scheduler_latency_us": avg_sched_lat,
            "p95_scheduler_latency_us": p95_sched_lat,
            "mean_comp_latency_us": float(np.mean([s["comp_lat_us"] for s in self.stats])),
            "mean_decomp_latency_us": float(np.mean([s["decomp_lat_us"] for s in self.stats])),
            "mean_queue_wait_latency_us": float(np.mean([s["queue_wait_lat_us"] for s in self.stats])),
            "mean_cache_store_latency_us": float(np.mean([s["cache_store_lat_us"] for s in self.stats])),
            "mean_cache_fill_comp_latency_us": float(np.mean([s["cache_fill_comp_lat_us"] for s in self.stats])),
            "cache_hit_rate": cache_hit_rate,
            "wall_clock_time_sec": elapsed_sec,
            "throughput_msg_per_sec": throughput,
            "throughput_mb_s": throughput_mb_s,
            "peak_rss_mb": peak_rss_mb,
            "total_compute_cost": total_compute_cost,
            "total_user_perceived_cost": total_user_cost
        }

