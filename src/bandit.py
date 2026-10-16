import numpy as np
import time
from typing import Union

def calibrate_action_cost_scales(train_messages: list[str], sample_size: int = 100) -> np.ndarray:
    """
    Calibrates empirical action cost scales on the training portion of a stream.

    Latency-Proportional Cost Scaling Rule:
        Before evaluation, for each action a in {0: SKIP, 1: ZSTD, 2: BROTLI, 3: GZIP, 4: LZ4, 5: BATCH}:
        1. Measure mean operation latency L_a on a calibration sample of training messages:
           - SKIP (a=0): pass-through UTF-8 encode and memory check (clamped to >= 1.0 us).
           - Codecs (a=1..4): mean compression + decompression latency on the sample.
           - BATCH (a=5): mean framed batch compression latency per message + estimated queue wait.
        2. Compute relative scale:
           scale(a) = max(1.0, L_a / L_SKIP)

        Under cost-aware exploration, the exploration bonus for action a is scaled by 1 / scale(a),
        discounting exploration of high-latency actions proportionally to their slowdown relative to SKIP.

    Returns:
        np.ndarray of shape (6,) with scale >= 1.0 for each action (scale[0] == 1.0).
    """
    from src.engine import CompressionEngine
    engine = CompressionEngine()

    sample = train_messages[:sample_size] if len(train_messages) > sample_size else train_messages
    if not sample:
        return np.ones(6)

    # 1. SKIP latency
    t0 = time.perf_counter()
    for m in sample:
        _ = len(m.encode('utf-8'))
    l_skip = max(1.0, ((time.perf_counter() - t0) * 1_000_000) / len(sample))

    # 2. Codec latencies: ZSTD, BROTLI, GZIP, LZ4
    codec_names = {1: 'ZSTD', 2: 'BROTLI', 3: 'GZIP', 4: 'LZ4'}
    latencies = {0: l_skip}

    for a_idx, c_name in codec_names.items():
        t0 = time.perf_counter()
        for m in sample:
            b, _ = engine.compress(m, c_name)
            _, _ = engine.decompress(b, c_name)
        lat = ((time.perf_counter() - t0) * 1_000_000) / len(sample)
        latencies[a_idx] = max(l_skip, lat)

    # 3. BATCH latency: batch framing + batch compress per msg + nominal queue wait (1000 us)
    try:
        from src.manager import BatchCacheManager
        mgr = BatchCacheManager()
        framed = mgr.encode_batch(sample)
        t0 = time.perf_counter()
        comp, _ = engine.compress(framed, 'ZSTD')
        _, _ = engine.decompress(comp, 'ZSTD', raw_bytes=True)
        batch_comp_per_msg = ((time.perf_counter() - t0) * 1_000_000) / len(sample)
        latencies[5] = batch_comp_per_msg + 1000.0  # nominal batch queue wait
    except Exception:
        latencies[5] = latencies[1] + 1000.0

    scales = np.zeros(6)
    for a in range(6):
        scales[a] = max(1.0, latencies[a] / l_skip)

    return scales

class LinUCBBandit:
    def __init__(self, d: int = 9, K: int = 6, alpha: float = 0.3, lambda_param: float = 0.01,
                 cost_aware_exploration: bool = False, max_wait_us: float = 50000.0, enable_deadline: bool = False,
                 action_cost_scale: np.ndarray = None, warm_start_steps: int = 0):
        """
        LinUCB Contextual Bandit with optional Cost-Aware Exploration, Deadline Masking, and Heuristic Warm-Start.

        Args:
            d: Dimension of context feature vector.
            K: Number of actions.
            alpha: Exploration parameter (controls width of confidence interval).
            lambda_param: L2 regularization parameter for ridge regression.
            cost_aware_exploration: If True, discounts exploration bonus for high-latency actions by 1 / action_cost_scale[a].
            max_wait_us: Maximum acceptable queue wait deadline in microseconds.
            enable_deadline: If True, masks BATCH action when predicted wait exceeds max_wait_us.
            action_cost_scale: Custom array of length K with empirical cost scales relative to SKIP.
            warm_start_steps: Number of initial messages to override action using the length heuristic (SKIP < 25 chars, else ZSTD).

        Latency-Proportional Cost Scaling Rule:
            When cost_aware_exploration=True, exploration bonus is multiplied by:
                bonus_weight = 1.0 / action_cost_scale[a]
            where action_cost_scale[a] = max(1.0, mean_latency(a) / mean_latency(SKIP)),
            measured empirically on the training portion of the stream.

        Length Heuristic Warm-Start:
            For the first warm_start_steps messages, predict() returns action = 0 (SKIP) if char_len < 25,
            otherwise action = 1 (ZSTD). The observed rewards update LinUCB's ridge regression parameters
            A and b, providing a warm initial prior for common actions before UCB exploration begins.
        """
        self.d = d
        self.K = K
        self.alpha = alpha
        self.lambda_param = lambda_param
        self.cost_aware_exploration = cost_aware_exploration
        self.max_wait_us = max_wait_us
        self.enable_deadline = enable_deadline
        self.action_cost_scale = np.array(action_cost_scale, dtype=float) if action_cost_scale is not None else np.ones(self.K)
        self.warm_start_steps = warm_start_steps
        self.step_count = 0
        self.reset()

    def set_action_cost_scale(self, scales: Union[list, np.ndarray]):
        """Sets empirical action cost scales for cost-aware exploration."""
        self.action_cost_scale = np.array(scales, dtype=float)

    def set_deadline(self, max_wait_us: float = None, enable_deadline: bool = True):
        """Sets deadline parameters and toggles deadline masking."""
        if max_wait_us is not None:
            self.max_wait_us = max_wait_us
        self.enable_deadline = enable_deadline

    def _normalize_features(self, features_dict: dict) -> np.ndarray:
        """Normalizes raw features to a stable [0, 1] range for linear model convergence."""
        x = np.zeros((self.d, 1))
        x[0, 0] = min(features_dict.get("char_len", 0) / 280.0, 2.0)
        x[1, 0] = min(features_dict.get("word_len", 0) / 50.0, 2.0)
        x[2, 0] = features_dict.get("entropy", 0) / 8.0
        x[3, 0] = features_dict.get("repetition_score", 0.0)
        # Log scaling for arrival rate
        x[4, 0] = min(np.log1p(features_dict.get("arrival_rate", 0.0)) / 5.0, 2.0)
        x[5, 0] = features_dict.get("uppercase_ratio", 0.0)
        x[6, 0] = features_dict.get("punctuation_ratio", 0.0)
        x[7, 0] = features_dict.get("emoji_ratio", 0.0)
        x[8, 0] = features_dict.get("unique_word_ratio", 0.0)
        return x

    def predict(self, features_dict: dict) -> tuple[int, float]:
        """
        Selects the action that maximizes the upper confidence bound.
        If warm_start_steps > 0 and step_count < warm_start_steps, uses length heuristic:
            action = 0 (SKIP) if char_len < 25, else 1 (ZSTD).
        If enable_deadline is True and predicted queue wait exceeds max_wait_us,
        BATCH (action 5) is masked.
        Returns:
            (action_index, inference_latency_microseconds)
        """
        start_time = time.perf_counter()

        # Heuristic warm-start override
        if self.warm_start_steps > 0 and self.step_count < self.warm_start_steps:
            char_len = features_dict.get("char_len", 0)
            action = 0 if char_len < 25 else 1
            self.step_count += 1
            latency = (time.perf_counter() - start_time) * 1_000_000
            return action, latency

        self.step_count += 1
        x = self._normalize_features(features_dict)
        p = np.zeros(self.K)
        
        for a in range(self.K):
            # Compute ridge regression coefficients
            theta = self.A_inv[a] @ self.b[a]
            # Standard deviation of expectation
            std_dev = np.sqrt(x.T @ self.A_inv[a] @ x)[0, 0]
            
            # Optional cost-aware exploration discount:
            # High-cost actions receive a smaller exploration bonus to avoid risky exploration.
            bonus_weight = (1.0 / self.action_cost_scale[a]) if self.cost_aware_exploration else 1.0
            
            # Upper Confidence Bound
            p[a] = (theta.T @ x)[0, 0] + self.alpha * bonus_weight * std_dev
            
        # Deadline mask on BATCH (action 5)
        if self.enable_deadline and self.max_wait_us is not None:
            from src.scheduler import estimate_batch_wait_us
            predicted_wait = estimate_batch_wait_us(features_dict)
            if predicted_wait > self.max_wait_us:
                p[5] = -np.inf
                
        action = int(np.argmax(p))
        latency = (time.perf_counter() - start_time) * 1_000_000
        return action, latency

    def update(self, action: int, features_dict: dict, reward: float):
        """
        Updates the bandit parameters with observed reward.
        Args:
            action: Chosen action index.
            features_dict: Features context.
            reward: Real-valued scalar reward (-Cost).
        """
        x = self._normalize_features(features_dict)
        # Update covariance matrix and bias vector
        self.A[action] += x @ x.T
        self.b[action] += reward * x
        
        # Sherman-Morrison formula for fast rank-1 update of matrix inverse:
        # (A + x x^T)^-1 = A^-1 - (A^-1 x x^T A^-1) / (1 + x^T A^-1 x)
        inv = self.A_inv[action]
        numerator = inv @ x @ x.T @ inv
        denominator = 1.0 + (x.T @ inv @ x)[0, 0]
        self.A_inv[action] = inv - numerator / denominator
        
        self.reward_history.append(reward)
        self.action_history.append(action)

    def reset(self):
        """Resets the bandit model state."""
        self.A = [np.eye(self.d) * self.lambda_param for _ in range(self.K)]
        self.b = [np.zeros((self.d, 1)) for _ in range(self.K)]
        self.A_inv = [np.eye(self.d) / self.lambda_param for _ in range(self.K)]
        self.reward_history = []
        self.action_history = []
        self.step_count = 0

