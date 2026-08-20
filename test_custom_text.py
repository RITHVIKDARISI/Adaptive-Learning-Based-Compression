"""
Interactive / CLI utility to test single custom text messages through the Adaptive Compression Scheduler pipeline.
Usage:
    python test_custom_text.py "Your custom text message goes here!"
    python test_custom_text.py   (Interactive mode)
"""

import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.features import FeatureExtractor
from src.manager import BatchCacheManager
from src.scheduler import HeuristicScheduler, ACTION_MAP
from src.engine import CompressionEngine

def test_text(text: str, manager: BatchCacheManager, fe: FeatureExtractor, scheduler: HeuristicScheduler):
    print("\n" + "="*60)
    print(f"[Input Text]: \"{text}\"")
    print("="*60)
    
    timestamp = time.time()
    
    # 1. Feature Extraction
    features = fe.extract_features(text, timestamp)
    print("\n[Extracted Features]:")
    for k, v in features.items():
        if isinstance(v, float):
            print(f"  * {k:20s}: {v:.4f}")
        else:
            print(f"  * {k:20s}: {v}")
            
    # 2. Check Cache
    cached_res = manager.cache_lookup(text)
    if cached_res is not None:
        comp_bytes, codec = cached_res
        print(f"\n[Cache Status]      : CACHE HIT (Lossless Reuse)")
        print(f"  * Codec Used       : {codec}")
        print(f"  * Raw Bytes        : {len(text.encode('utf-8'))} B")
        print(f"  * Compressed Bytes : {len(comp_bytes)} B")
        ratio = len(comp_bytes) / len(text.encode('utf-8')) if len(text.encode('utf-8')) > 0 else 1.0
        print(f"  * Compression Ratio: {ratio:.4f}")
        return

    print(f"\n[Cache Status]      : CACHE MISS")
    
    # 3. Scheduler Decision
    action_idx, sched_lat = scheduler.predict(features)
    action_name = ACTION_MAP.get(action_idx, "UNKNOWN")
    print(f"\n[Scheduler Action]  : {action_name} (Decision time: {sched_lat:.2f} microseconds)")
    
    # 4. Perform Action & Measure Size/Latency
    engine = CompressionEngine()
    raw_bytes = text.encode('utf-8')
    raw_len = len(raw_bytes)
    
    if action_name == "SKIP":
        comp_bytes = raw_bytes
        comp_lat = 0.0
    elif action_name == "BATCH":
        # Simulate batching
        comp_bytes, comp_lat = engine.compress(text, "ZSTD")
    else:
        comp_bytes, comp_lat = engine.compress(text, action_name)
        
    manager.cache_store(text, comp_bytes, action_name)
    
    ratio = len(comp_bytes) / raw_len if raw_len > 0 else 1.0
    saved_percent = (1 - ratio) * 100
    
    print("\n[Compression Results]:")
    print(f"  * Original Size    : {raw_len} bytes")
    print(f"  * Compressed Size  : {len(comp_bytes)} bytes")
    print(f"  * Compression Ratio: {ratio:.4f} ({saved_percent:+.1f}% space saved)")
    print(f"  * Engine Latency   : {comp_lat:.2f} microseconds")
    print(f"  * Total E2E Latency: {sched_lat + comp_lat:.2f} microseconds")
    print("="*60 + "\n")

def main():
    manager = BatchCacheManager()
    fe = FeatureExtractor()
    scheduler = HeuristicScheduler(skip_threshold=25)
    
    if len(sys.argv) > 1:
        custom_text = " ".join(sys.argv[1:])
        test_text(custom_text, manager, fe, scheduler)
    else:
        print("💡 Entering Interactive Mode. Type a text message and press Enter (or 'q' to exit).")
        while True:
            try:
                user_input = input("\nEnter message: ").strip()
                if user_input.lower() in ['q', 'quit', 'exit']:
                    print("Exiting.")
                    break
                if not user_input:
                    continue
                test_text(user_input, manager, fe, scheduler)
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    main()
