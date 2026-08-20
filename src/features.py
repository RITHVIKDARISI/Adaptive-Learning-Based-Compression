import math
import re
from datasketch import MinHash

class FeatureExtractor:
    def __init__(self, window_size: int = 50, num_perm: int = 64):
        self.window_size = window_size
        self.num_perm = num_perm
        self.minhash_history = []  # List of MinHash objects
        self.last_timestamp = None
        self.arrival_rate = 0.0
        self.alpha = 0.2  # Smoothing factor for arrival rate

    def _get_tokens(self, text: str) -> list[str]:
        # Simple whitespace tokenization, lowercased
        return re.findall(r'\w+', text.lower())

    def _compute_minhash(self, tokens: list[str]) -> MinHash:
        m = MinHash(num_perm=self.num_perm)
        for token in tokens:
            m.update(token.encode('utf-8'))
        return m

    def _compute_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        counts = {}
        for c in text:
            counts[c] = counts.get(c, 0) + 1
        entropy = 0.0
        total = len(text)
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    def _compute_repetition_score(self, current_minhash: MinHash) -> float:
        if not self.minhash_history:
            return 0.0
        max_sim = 0.0
        for past_minhash in self.minhash_history:
            sim = current_minhash.jaccard(past_minhash)
            if sim > max_sim:
                max_sim = sim
        return max_sim

    def _update_history(self, current_minhash: MinHash):
        self.minhash_history.append(current_minhash)
        if len(self.minhash_history) > self.window_size:
            self.minhash_history.pop(0)

    def _compute_arrival_rate(self, timestamp: float) -> float:
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            return 1.0  # Initial default rate
            
        time_diff = timestamp - self.last_timestamp
        self.last_timestamp = timestamp
        
        if time_diff <= 0:
            time_diff = 0.001  # Prevent division by zero
            
        current_rate = 1.0 / time_diff
        # Exponential moving average
        self.arrival_rate = self.alpha * current_rate + (1 - self.alpha) * self.arrival_rate
        return self.arrival_rate

    def extract_features(self, text: str, timestamp: float) -> dict:
        """
        Extracts features from the text message.
        Args:
            text: Message string.
            timestamp: Arrival epoch timestamp in seconds.
        Returns:
            Dict containing feature name and value.
        """
        char_len = len(text)
        tokens = self._get_tokens(text)
        word_len = len(tokens)
        
        entropy = self._compute_entropy(text)
        
        current_minhash = self._compute_minhash(tokens)
        repetition = self._compute_repetition_score(current_minhash)
        self._update_history(current_minhash)
        
        arrival_rate = self._compute_arrival_rate(timestamp)
        
        # Extended complexity features
        uppercase_ratio = sum(1 for c in text if c.isupper()) / char_len if char_len > 0 else 0.0
        
        punctuation_ratio = len(re.findall(r'[^\w\s]', text)) / char_len if char_len > 0 else 0.0
        
        # Simple emoji check (covers standard emoji block ranges)
        emoji_count = sum(1 for c in text if '\U0001f000' <= c <= '\U0001f9ff' or '\u2600' <= c <= '\u27bf')
        emoji_ratio = emoji_count / char_len if char_len > 0 else 0.0
        
        # Unique word ratio
        unique_word_ratio = len(set(tokens)) / word_len if word_len > 0 else 0.0
        
        return {
            "char_len": char_len,
            "word_len": word_len,
            "entropy": entropy,
            "repetition_score": repetition,
            "arrival_rate": arrival_rate,
            "uppercase_ratio": uppercase_ratio,
            "punctuation_ratio": punctuation_ratio,
            "emoji_ratio": emoji_ratio,
            "unique_word_ratio": unique_word_ratio
        }
        
    def reset(self):
        """Resets the state of the history trackers (useful between experiments)."""
        self.minhash_history = []
        self.last_timestamp = None
        self.arrival_rate = 0.0
