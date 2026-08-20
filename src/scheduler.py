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

class HeuristicScheduler:
    def __init__(self, skip_threshold: int = 25):
        self.skip_threshold = skip_threshold

    def predict(self, features_dict: dict) -> tuple[int, float]:
        """
        Predicts action using hand-written heuristic rules.
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
            action = 5  # BATCH
        else:
            action = 1  # ZSTD (default classical compressor)
            
        latency = (time.perf_counter() - start_time) * 1_000_000
        return action, latency

class OfflineLabelGenerator:
    def __init__(self, lambda_param: float = 0.01):
        """
        Generates offline-optimal labels using size + lambda * latency.
        Note that lambda is in bytes per microsecond (us).
        e.g., lambda = 0.01 means 1 byte is worth 100 microseconds.
        
        Modeling Approximation:
        The BATCH action cost is computed using a single global approximation (average size share 
        and average batch compression latency estimated from the first 10 messages) plus a 50 ms 
        (50,000 us) queue delay penalty.
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
            
            # 0. SKIP
            raw_bytes = msg.encode('utf-8')
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
            # Shared cost approximation: message's size share + shared latency + queue penalty
            msg_size_share = avg_batch_size * (len(raw_bytes) / max(sum(len(m.encode()) for m in sample_messages), 1))
            costs[5] = msg_size_share + self.lambda_param * (avg_batch_lat + batch_queue_penalty)
            
            # Select action that minimizes Cost
            optimal_action = min(costs, key=costs.get)
            labels.append(optimal_action)
            
        return labels

class SupervisedScheduler:
    def __init__(self, model_type: str = 'decision_tree'):
        self.model_type = model_type
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(max_depth=5)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
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
