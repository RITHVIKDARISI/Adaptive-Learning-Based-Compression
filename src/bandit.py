import numpy as np
import time
from typing import Union

class LinUCBBandit:
    def __init__(self, d: int = 9, K: int = 6, alpha: float = 0.3, lambda_param: float = 0.01, cost_aware_exploration: bool = False):
        """
        LinUCB Contextual Bandit.
        Args:
            d: Dimension of context feature vector.
            K: Number of actions.
            alpha: Exploration parameter (controls width of confidence interval).
            lambda_param: L2 regularization parameter for ridge regression.
            cost_aware_exploration: If True, discounts exploration bonus for high-latency actions.
        """
        self.d = d
        self.K = K
        self.alpha = alpha
        self.lambda_param = lambda_param
        self.cost_aware_exploration = cost_aware_exploration
        self.action_cost_scale = np.ones(self.K)
        self.reset()

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
        Returns:
            (action_index, inference_latency_microseconds)
        """
        start_time = time.perf_counter()
        
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

