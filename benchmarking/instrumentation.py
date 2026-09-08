import time
import hashlib
import psutil
import os
from contextlib import contextmanager

def get_peak_rss_mb() -> float:
    """Returns the peak Resident Set Size (RSS) memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    # Note: process.memory_info().peak_wset is Windows-specific, but cross-platform
    # fallback to current RSS can be used if peak is not available.
    try:
        peak_bytes = process.memory_info().peak_wset # type: ignore
    except AttributeError:
        peak_bytes = process.memory_info().rss
    return peak_bytes / (1024 * 1024)

@contextmanager
def measure_time_us():
    """
    Context manager to accurately measure wall-clock time in microseconds.
    Usage:
        with measure_time_us() as timer:
            do_work()
        latency = timer.elapsed_us
    """
    class Timer:
        def __init__(self):
            self.elapsed_us = 0.0
            
    timer = Timer()
    start = time.perf_counter()
    try:
        yield timer
    finally:
        end = time.perf_counter()
        timer.elapsed_us = (end - start) * 1_000_000

def verify_lossless_reconstruction(original_text: str, reconstructed_bytes: bytes) -> bool:
    """
    Verifies that the decompressed bytes perfectly match the original text bytes via SHA-256.
    """
    original_bytes = original_text.encode('utf-8')
    orig_hash = hashlib.sha256(original_bytes).hexdigest()
    recon_hash = hashlib.sha256(reconstructed_bytes).hexdigest()
    return orig_hash == recon_hash

class WallClockBenchmark:
    """
    Tracks overall wall-clock metrics for an entire stream simulation.
    """
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
        self.total_messages = 0
        self.total_input_bytes = 0
        self.total_output_bytes = 0

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()

    @property
    def stream_wall_time_sec(self) -> float:
        return self.end_time - self.start_time

    @property
    def messages_per_second(self) -> float:
        if self.stream_wall_time_sec == 0:
            return 0.0
        return self.total_messages / self.stream_wall_time_sec

    @property
    def input_mb_per_second(self) -> float:
        if self.stream_wall_time_sec == 0:
            return 0.0
        return (self.total_input_bytes / (1024 * 1024)) / self.stream_wall_time_sec

    @property
    def output_mb_per_second(self) -> float:
        if self.stream_wall_time_sec == 0:
            return 0.0
        return (self.total_output_bytes / (1024 * 1024)) / self.stream_wall_time_sec
