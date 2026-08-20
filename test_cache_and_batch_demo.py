"""
Demo script showing CACHE REUSE and BATCH COMPRESSION examples.
"""

import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.manager import BatchCacheManager
from src.engine import CompressionEngine

def run_cache_and_batch_demo():
    print("="*70)
    print(" 1. CACHE REUSE DEMO (Exact-Match Lossless Retrieval)")
    print("="*70)
    
    manager = BatchCacheManager()
    engine = CompressionEngine()
    
    original_msg = "Hello team, the production server deployment is completed successfully! 🚀"
    
    # 1. First time message arrives: CACHE MISS
    raw_bytes = original_msg.encode('utf-8')
    comp_bytes, comp_lat = engine.compress(original_msg, "ZSTD")
    manager.cache_store(original_msg, comp_bytes, "ZSTD")
    
    print(f"\n📩 Message Arrival #1: \"{original_msg}\"")
    print(f"  * Status           : CACHE MISS (First time seen)")
    print(f"  * Original Size    : {len(raw_bytes)} bytes")
    print(f"  * Compressed Size  : {len(comp_bytes)} bytes")
    print(f"  * Compression Time : {comp_lat:.2f} µs")
    
    # 2. Second time message arrives: CACHE HIT
    cached_hit = manager.cache_lookup(original_msg)
    if cached_hit:
        c_bytes, c_codec = cached_hit
        print(f"\n📩 Message Arrival #2: \"{original_msg}\" (Duplicate)")
        print(f"  * Status           : ⚡ CACHE HIT (Instant Lossless Memory Lookup)")
        print(f"  * Codec Reused     : {c_codec}")
        print(f"  * Original Size    : {len(raw_bytes)} bytes")
        print(f"  * Cached Bytes     : {len(c_bytes)} bytes")
        print(f"  * Compression Time : 0.00 µs (Bypassed compressor CPU overhead!)")

    print("\n" + "="*70)
    print(" 2. BATCH COMPRESSION DEMO (Aggregating Multiple Stream Messages)")
    print("="*70)
    
    # Batch of 5 short messages arriving on the stream
    stream_messages = [
        "User 101 logged in.",
        "User 102 logged in.",
        "User 103 updated profile picture.",
        "User 101 sent a chat message.",
        "User 104 logged out."
    ]
    
    print("\n📥 Individual Stream Messages:")
    indiv_total_raw = sum(len(m.encode('utf-8')) for m in stream_messages)
    indiv_total_comp = 0
    for idx, m in enumerate(stream_messages, 1):
        c, _ = engine.compress(m, "ZSTD")
        indiv_total_comp += len(c)
        print(f"  Message #{idx}: \"{m}\" ({len(m.encode('utf-8'))} B -> compressed individually: {len(c)} B)")
        
    indiv_ratio = indiv_total_comp / indiv_total_raw
    print(f"\n❌ If Compressed Individually:")
    print(f"  * Total Raw Bytes       : {indiv_total_raw} bytes")
    print(f"  * Total Compressed Bytes: {indiv_total_comp} bytes")
    print(f"  * Individual Ratio      : {indiv_ratio:.4f} (Increased size due to 5x block header overheads!)")
    
    # Now compress as a BATCH using length-prefix framing
    batched_payload = manager.encode_batch(stream_messages)
    batched_comp, batch_lat = engine.compress(batched_payload, "ZSTD")
    batch_ratio = len(batched_comp) / indiv_total_raw
    saved_pct = (1 - batch_ratio) * 100
    
    print(f"\n✅ If Compressed as a BATCH (Combined Queue):")
    print(f"  * Batched Raw Bytes     : {indiv_total_raw} bytes")
    print(f"  * Batched Compressed    : {len(batched_comp)} bytes")
    print(f"  * Batch Compression Ratio: {batch_ratio:.4f} ({saved_pct:+.1f}% space saved!)")
    print(f"  * Batch Engine Latency  : {batch_lat:.2f} µs")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_cache_and_batch_demo()
