from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MessageMetrics:
    """
    Standardized per-message metrics for rigorous benchmarking.
    """
    msg_id: int
    original_bytes: int
    encoded_bytes: int
    
    # Latency tracking (all in microseconds)
    feature_latency_us: float = 0.0
    cache_lookup_latency_us: float = 0.0
    scheduler_inference_latency_us: float = 0.0
    compression_latency_us: float = 0.0
    decompression_latency_us: float = 0.0
    cache_store_latency_us: float = 0.0
    queue_wait_latency_us: float = 0.0
    
    # Actions & status
    action: str = ""
    codec: str = ""
    cache_hit: bool = False
    batch_size: int = 1
    reconstruction_success: Optional[bool] = None

    @property
    def compression_factor(self) -> float:
        """original / compressed. >1 means compression."""
        if self.encoded_bytes == 0:
            return 1.0
        return self.original_bytes / self.encoded_bytes

    @property
    def compressed_fraction(self) -> float:
        """compressed / original. <1 means compression."""
        if self.original_bytes == 0:
            return 1.0
        return self.encoded_bytes / self.original_bytes

    @property
    def space_saving_pct(self) -> float:
        """100 * (1 - compressed/original). >0 means compression."""
        if self.original_bytes == 0:
            return 0.0
        return 100.0 * (1.0 - (self.encoded_bytes / self.original_bytes))

    @property
    def total_compute_latency_us(self) -> float:
        """Sum of all active CPU compute latencies for this message."""
        return (self.feature_latency_us + 
                self.cache_lookup_latency_us + 
                self.scheduler_inference_latency_us + 
                self.compression_latency_us + 
                self.cache_store_latency_us)

    @property
    def e2e_latency_us(self) -> float:
        """Total end-to-end latency including queueing and computation."""
        return self.total_compute_latency_us + self.queue_wait_latency_us
