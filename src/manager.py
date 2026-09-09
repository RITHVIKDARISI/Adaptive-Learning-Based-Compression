import hashlib
import struct
import time
from collections import OrderedDict
from typing import Optional, Union

class LRUCache:
    def __init__(self, capacity: int = 1000):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.evictions = 0

    def get(self, key: str) -> Optional[tuple[bytes, str]]:
        if key not in self.cache:
            return None
        # Move to end to represent recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: tuple[bytes, str]):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove oldest (FIFO/LRU)
            self.evictions += 1

    def clear(self):
        self.cache.clear()

    def __len__(self):
        return len(self.cache)

class Batcher:
    def __init__(self, max_batch_size: int = 10, timeout_seconds: float = 0.5):
        self.max_batch_size = max_batch_size
        self.timeout_seconds = timeout_seconds
        self.queue = []  # List of (message_text, timestamp, message_id)
        self.first_message_time = None

    def add(self, text: str, timestamp: float, msg_id: int) -> bool:
        """
        Adds a message to the batch.
        Returns:
            True if adding this message triggers a batch flush, False otherwise.
        """
        if not self.queue:
            self.first_message_time = timestamp
        
        self.queue.append((text, timestamp, msg_id))
        
        if len(self.queue) >= self.max_batch_size:
            return True
        return False

    def should_flush(self, current_time: float) -> bool:
        """Checks if the batch has timed out relative to current stream time."""
        if not self.queue:
            return False
        return (current_time - self.first_message_time) >= self.timeout_seconds

    def flush(self) -> list[tuple[str, float, int]]:
        """Clears and returns the batched messages."""
        batch = self.queue
        self.queue = []
        self.first_message_time = None
        return batch

class BatchCacheManager:
    def __init__(self, cache_capacity: int = 1000, max_batch_size: int = 10, timeout_seconds: float = 0.5):
        self.cache = LRUCache(capacity=cache_capacity)
        self.batcher = Batcher(max_batch_size=max_batch_size, timeout_seconds=timeout_seconds)

    @property
    def cache_evictions(self) -> int:
        if hasattr(self.cache, 'evictions'):
            return self.cache.evictions
        return 0

    @staticmethod
    def encode_batch(messages: list[str]) -> bytes:
        """
        Encodes a list of message strings into a binary framed payload
        using 4-byte big-endian uint32 length prefixes.
        Guarantees lossless reconstruction regardless of message contents (e.g. newlines).
        """
        payload = bytearray()
        for msg in messages:
            msg_bytes = msg.encode('utf-8')
            payload.extend(struct.pack('>I', len(msg_bytes)))
            payload.extend(msg_bytes)
        return bytes(payload)

    @staticmethod
    def decode_batch(framed_bytes: bytes) -> list[str]:
        """
        Decodes a length-prefixed binary framed payload back into a list of original message strings.
        """
        messages = []
        offset = 0
        total_len = len(framed_bytes)
        while offset < total_len:
            if offset + 4 > total_len:
                raise ValueError("Corrupted length-prefixed batch payload: incomplete header")
            msg_len = struct.unpack('>I', framed_bytes[offset:offset+4])[0]
            offset += 4
            if offset + msg_len > total_len:
                raise ValueError("Corrupted length-prefixed batch payload: incomplete body")
            msg_bytes = framed_bytes[offset:offset+msg_len]
            offset += msg_len
            messages.append(msg_bytes.decode('utf-8'))
        return messages

    def get_hash(self, text: str) -> str:
        """Computes SHA-256 hash of the message text for exact-match cache keys."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def cache_lookup(self, text: str) -> Optional[tuple[bytes, str]]:
        """Looks up a message in the exact-match cache."""
        key = self.get_hash(text)
        return self.cache.get(key)

    def cache_store(self, text: str, compressed_bytes: bytes, codec: str):
        """Stores a compressed message in the exact-match cache."""
        key = self.get_hash(text)
        self.cache.put(key, (compressed_bytes, codec))

    def reset(self):
        """Resets the state of cache and batcher."""
        self.cache.clear()
        # Re-initialize batcher to clear queue
        self.batcher = Batcher(max_batch_size=self.batcher.max_batch_size, timeout_seconds=self.batcher.timeout_seconds)

