import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.engine import CompressionEngine

ACTION_MAP = {
    0: 'SKIP',
    1: 'ZSTD',
    2: 'BROTLI',
    3: 'GZIP',
    4: 'LZ4',
    5: 'BATCH'
}

REV_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}

def estimate_batch_wait_us(features_dict: dict) -> float:
    """
    Estimates the remaining queue wait latency in microseconds for BATCH action
    based on the current queue length, batch capacity, batch timeout, and arrival rate.
    """
    queue_len = features_dict.get("queue_len", 0)
    batch_max_size = features_dict.get("batch_max_size", 10)
    batch_timeout_sec = features_dict.get("batch_timeout_seconds", 0.5)
    arrival_rate = features_dict.get("arrival_rate", 1.0)

    needed = max(0, batch_max_size - (queue_len + 1))
    if needed == 0:
        return 0.0

    if arrival_rate > 0:
        est_wait_s = min(needed / arrival_rate, batch_timeout_sec)
    else:
        est_wait_s = batch_timeout_sec

    return est_wait_s * 1_000_000.0

class HeuristicScheduler:
    def __init__(self, skip_threshold: int = 25, max_wait_us: float = 50000.0, enable_deadline: bool = False):
        self.skip_threshold = skip_threshold
        self.max_wait_us = max_wait_us
        self.enable_deadline = enable_deadline

    def set_deadline(self, max_wait_us: float = None, enable_deadline: bool = True):
        """Sets deadline parameters and toggles deadline masking."""
        if max_wait_us is not None:
            self.max_wait_us = max_wait_us
        self.enable_deadline = enable_deadline

    def predict(self, features_dict: dict) -> tuple[int, float]:
        """
        Predicts action using hand-written heuristic rules.
        If enable_deadline is True and predicted queue wait exceeds max_wait_us,
        BATCH is masked and falls back to ZSTD.
        Returns:
            (action_index, latency_microseconds)
        """
        start_time = time.perf_counter()
        
        char_len = features_dict.get("char_len", 0)
        repetition = features_dict.get("repetition_score", 0.0)
        
        # Heuristic rules:
        # If very short, skip.
        # If highly repetitive, batch or compress.
        if char_len < self.skip_threshold:
            action = 0  # SKIP
        elif repetition > 0.8:
            if self.enable_deadline and self.max_wait_us is not None:
                predicted_wait = estimate_batch_wait_us(features_dict)
                if predicted_wait > self.max_wait_us:
                    action = 1  # Mask BATCH -> fallback to ZSTD
                else:
                    action = 5  # BATCH
            else:
                action = 5  # BATCH
        else:
            action = 1  # ZSTD
            
        latency = (time.perf_counter() - start_time) * 1_000_000
        return action, latency

class OfflineLabelGenerator:
    def __init__(self, lambda_param: float = 0.01):
        """
        Generates offline-optimal labels using size + lambda * latency.
        Note that lambda is in bytes per microsecond (us).
        e.g., lambda = 0.01 means 1 byte is worth 100 microseconds.
        """
        self.lambda_param = lambda_param
        self.engine = CompressionEngine()

    def generate_labels(self, messages: list[str]) -> list[int]:
        """
        Computes the optimal action for each message by trying all of them.
        Returns a list of action indices.
        """
        labels = []
        
        # Estimate batch size & latency on a sample of 10 messages (global approximation)
        sample_messages = messages[:10] if len(messages) >= 10 else messages
        batch_text = "\n".join(sample_messages)
        batch_comp, batch_comp_lat = self.engine.compress(batch_text, 'ZSTD')
        avg_batch_size = len(batch_comp) / max(len(sample_messages), 1)
        avg_batch_lat = batch_comp_lat / max(len(sample_messages), 1)
        
        # Batching queue delay penalty: 50 ms (50,000 microseconds)
        batch_queue_penalty = 50000.0

        for msg in messages:
            costs = {}
            raw_bytes = msg.encode('utf-8')
            
            # 0. SKIP
            costs[0] = len(raw_bytes) + self.lambda_param * 0.0
            
            # 1. ZSTD
            zcomp, zlat = self.engine.compress(msg, 'ZSTD')
            costs[1] = len(zcomp) + self.lambda_param * zlat
            
            # 2. BROTLI
            bcomp, blat = self.engine.compress(msg, 'BROTLI')
            costs[2] = len(bcomp) + self.lambda_param * blat
            
            # 3. GZIP
            gcomp, glat = self.engine.compress(msg, 'GZIP')
            costs[3] = len(gcomp) + self.lambda_param * glat
            
            # 4. LZ4
            lcomp, llat = self.engine.compress(msg, 'LZ4')
            costs[4] = len(lcomp) + self.lambda_param * llat
            
            # 5. BATCH
            msg_size_share = avg_batch_size * (len(raw_bytes) / max(sum(len(m.encode()) for m in sample_messages), 1))
            costs[5] = msg_size_share + self.lambda_param * (avg_batch_lat + batch_queue_penalty)
            
            optimal_action = min(costs, key=costs.get)
            labels.append(optimal_action)
            
        return labels

class SimulatorAlignedOracle:
    def __init__(self, lambda_param: float = 0.01, max_batch_size: int = 10, timeout_seconds: float = 0.5, cache_capacity: int = 1000):
        """
        Exact Simulator-Aligned Oracle Comparator.
        Matches src/simulator.py exactly:
        - 4-byte big-endian length-prefix binary framing (BatchCacheManager.encode_batch).
        - Runtime batcher queue capacity (default 10) and timeout (default 0.5s).
        - Runtime LRU cache lookup, exact-match reuse, and store mechanics.
        - Exact 8-stage latency accounting (feat, cache lookup, comp, decomp, queue wait, cache store, cache fill).
        """
        self.lambda_param = lambda_param
        self.max_batch_size = max_batch_size
        self.timeout_seconds = timeout_seconds
        self.cache_capacity = cache_capacity
        self.engine = CompressionEngine()

    def generate_labels(self, messages: list[str], timestamps: list[float]) -> list[int]:
        """
        Simulates optimal stream compression decisions matching runtime simulator mechanics.
        Returns the sequence of optimal action indices (0-5).
        """
        from src.manager import BatchCacheManager, LRUCache
        labels = [0] * len(messages)
        cache = LRUCache(capacity=self.cache_capacity)
        
        # 1. Evaluate non-batch costs for every message taking exact cache and latency into account
        costs_matrix = []
        for i, (msg, ts) in enumerate(zip(messages, timestamps)):
            raw_bytes = msg.encode('utf-8')
            msg_costs = {}
            
            # Cache lookup simulation
            cached = cache.get(msg)
            if cached is not None:
                cached_bytes, cached_codec = cached
                _, decomp_lat = self.engine.decompress(cached_bytes, cached_codec)
                # Cache reuse cost
                cache_cost = len(cached_bytes) + self.lambda_param * (1.0 + decomp_lat)
                # If cached, assign cache cost to all actions as lower bound
                for a in range(6):
                    msg_costs[a] = cache_cost
                costs_matrix.append(msg_costs)
                continue

            # 0. SKIP
            msg_costs[0] = len(raw_bytes) + self.lambda_param * 10.0  # sched overhead
            
            # 1. ZSTD
            zcomp, zlat = self.engine.compress(msg, 'ZSTD')
            _, zdecomp = self.engine.decompress(zcomp, 'ZSTD')
            msg_costs[1] = len(zcomp) + self.lambda_param * (zlat + zdecomp + 15.0)
            
            # 2. BROTLI
            bcomp, blat = self.engine.compress(msg, 'BROTLI')
            _, bdecomp = self.engine.decompress(bcomp, 'BROTLI')
            msg_costs[2] = len(bcomp) + self.lambda_param * (blat + bdecomp + 15.0)
            
            # 3. GZIP
            gcomp, glat = self.engine.compress(msg, 'GZIP')
            _, gdecomp = self.engine.decompress(gcomp, 'GZIP')
            msg_costs[3] = len(gcomp) + self.lambda_param * (glat + gdecomp + 15.0)
            
            # 4. LZ4
            lcomp, llat = self.engine.compress(msg, 'LZ4')
            _, ldecomp = self.engine.decompress(lcomp, 'LZ4')
            msg_costs[4] = len(lcomp) + self.lambda_param * (llat + ldecomp + 15.0)
            
            # Default placeholder for BATCH (to be updated during batch simulation)
            msg_costs[5] = float('inf')
            
            # Update cache with best single compressor for future hits
            best_single_action = min([1, 2, 3, 4], key=lambda a: msg_costs[a])
            best_comp_payloads = {1: (zcomp, 'ZSTD'), 2: (bcomp, 'BROTLI'), 3: (gcomp, 'GZIP'), 4: (lcomp, 'LZ4')}
            cache.put(msg, best_comp_payloads[best_single_action])
            
            costs_matrix.append(msg_costs)

        # 2. Simulate exact runtime batching with 4-byte length prefix framing and capacity/timeout
        batch_queue = []
        batch_start_time = None

        def flush_oracle_batch(flush_time: float, current_queue: list):
            if not current_queue:
                return
            batch_texts = [item[0] for item in current_queue]
            framed_payload = BatchCacheManager.encode_batch(batch_texts)
            
            comp_bytes, comp_lat = self.engine.compress(framed_payload, 'ZSTD')
            decomp_bytes, decomp_lat = self.engine.decompress(comp_bytes, 'ZSTD', raw_bytes=True)
            assert BatchCacheManager.decode_batch(decomp_bytes) == batch_texts, "Oracle batch decoding losslessness verification failed!"
            
            batch_size = len(comp_bytes)
            num_messages = len(current_queue)
            total_orig_len = sum(len(t.encode('utf-8')) for t in batch_texts)

            for text, ts, orig_idx in current_queue:
                orig_size = len(text.encode('utf-8'))
                size_share = batch_size * (orig_size / max(total_orig_len, 1))
                queue_wait_us = (flush_time - ts) * 1_000_000
                comp_share = comp_lat / num_messages
                decomp_share = decomp_lat / num_messages
                
                # Cache-fill and store
                _, fill_lat = self.engine.compress(text, 'ZSTD')
                store_lat = 5.0
                
                batch_cost = size_share + self.lambda_param * (queue_wait_us + comp_share + decomp_share + fill_lat + store_lat + 10.0)
                costs_matrix[orig_idx][5] = batch_cost

        for i, (msg, ts) in enumerate(zip(messages, timestamps)):
            if not batch_queue:
                batch_start_time = ts
            if batch_queue and (ts - batch_start_time >= self.timeout_seconds or len(batch_queue) >= self.max_batch_size):
                flush_oracle_batch(ts, batch_queue)
                batch_queue = []
                batch_start_time = ts
            batch_queue.append((msg, ts, i))

        if batch_queue:
            flush_oracle_batch(batch_queue[-1][1], batch_queue)

        # 3. Assign optimal labels
        for i, msg_costs in enumerate(costs_matrix):
            labels[i] = min(msg_costs, key=msg_costs.get)

        return labels

# Explicit backward-compatibility alias
CorrectedOfflineLabelGenerator = SimulatorAlignedOracle

class SupervisedScheduler:
    def __init__(self, model_type: str = 'decision_tree', random_state: int = 42):
        self.model_type = model_type
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=random_state)
        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(max_depth=5, random_state=random_state)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=random_state)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.feature_names = [
            "char_len", "word_len", "entropy", "repetition_score", "arrival_rate",
            "uppercase_ratio", "punctuation_ratio", "emoji_ratio", "unique_word_ratio"
        ]

    def _dict_to_array(self, features_dict: dict) -> np.ndarray:
        return np.array([[features_dict[name] for name in self.feature_names]])

    def fit(self, X: list[dict], y: list[int]):
        """Trains the classifier on a list of feature dicts and labels."""
        unique_classes = np.unique(y)
        if len(unique_classes) <= 1:
            self.is_degenerate = True
            self.degenerate_class = int(unique_classes[0]) if len(unique_classes) == 1 else 0
            return
        self.is_degenerate = False
        X_matrix = np.zeros((len(X), len(self.feature_names)))
        for i, feat in enumerate(X):
            for j, name in enumerate(self.feature_names):
                X_matrix[i, j] = feat.get(name, 0.0)
        self.model.fit(X_matrix, y)

    def predict(self, features_dict: dict) -> tuple[int, float]:
        """
        Predicts action using the trained classifier.
        Returns:
            (action_index, latency_microseconds)
        """
        start_time = time.perf_counter()
        if getattr(self, 'is_degenerate', False):
            latency = (time.perf_counter() - start_time) * 1_000_000
            return self.degenerate_class, latency
        x = self._dict_to_array(features_dict)
        pred = int(self.model.predict(x)[0])
        latency = (time.perf_counter() - start_time) * 1_000_000
        return pred, latency

    def get_feature_importances(self) -> dict:
        """Returns feature importances if the model supports it."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients summed across classes
            importances = np.sum(np.abs(self.model.coef_), axis=0)
            # Normalize to sum to 1
            importances /= max(np.sum(importances), 1e-6)
            return dict(zip(self.feature_names, importances))
        return {}
