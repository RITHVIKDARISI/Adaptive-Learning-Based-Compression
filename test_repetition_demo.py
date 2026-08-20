"""
Demo script showing feature extraction and compression for large text with emojis, internal repetitions, and stream repetitions / cache reuse.
"""

import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.features import FeatureExtractor
from src.manager import BatchCacheManager
from src.scheduler import HeuristicScheduler, ACTION_MAP
from src.engine import CompressionEngine

def run_repetition_demo():
    manager = BatchCacheManager()
    fe = FeatureExtractor()
    scheduler = HeuristicScheduler(skip_threshold=25)
    engine = CompressionEngine()
    
    messages = [
        "🚀 High priority system alert! 🚀 High priority system alert! 🚀 High priority system alert! 🚀 High priority system alert!",
        "🚀 High priority system alert! 🚀 High priority system alert! 🚀 System warning issued!",
        "🚀 High priority system alert! 🚀 High priority system alert! 🚀 High priority system alert! 🚀 High priority system alert!" # Exact duplicate of Message 1
    ]
    
    t = time.time()
    for idx, text in enumerate(messages, 1):
        print(f"\n" + "="*65)
        print(f"📩 [Message #{idx}]: \"{text}\"")
        print("="*65)
        
        t += 0.05  # Simulate 50ms arrival gap
        features = fe.extract_features(text, t)
        
        print("\n[Extracted Features]:")
        print(f"  * char_len            : {features['char_len']}")
        print(f"  * word_len            : {features['word_len']}")
        print(f"  * emoji_ratio         : {features['emoji_ratio']:.4f} (Emoji detected!)")
        print(f"  * unique_word_ratio   : {features['unique_word_ratio']:.4f} (Internal repetition)")
        print(f"  * repetition_score    : {features['repetition_score']:.4f} (Stream similarity)")
        
        cached_res = manager.cache_lookup(text)
        if cached_res is not None:
            comp_bytes, codec = cached_res
            raw_len = len(text.encode('utf-8'))
            ratio = len(comp_bytes) / raw_len
            saved = (1 - ratio) * 100
            print(f"\n⚡ [Cache Status]      : CACHE HIT (Lossless Reuse)")
            print(f"  * Codec Used       : {codec}")
            print(f"  * Original Size    : {raw_len} bytes")
            print(f"  * Cached Size      : {len(comp_bytes)} bytes")
            print(f"  * Compression Ratio: {ratio:.4f} ({saved:+.1f}% space saved)")
            print(f"  * Retrieval Time   : ~0.00 microseconds")
            continue
            
        print(f"\n⚡ [Cache Status]      : CACHE MISS")
        action_idx, sched_lat = scheduler.predict(features)
        action_name = ACTION_MAP.get(action_idx, "UNKNOWN")
        
        raw_bytes = text.encode('utf-8')
        raw_len = len(raw_bytes)
        comp_bytes, comp_lat = engine.compress(text, action_name)
        manager.cache_store(text, comp_bytes, action_name)
        
        ratio = len(comp_bytes) / raw_len
        saved = (1 - ratio) * 100
        
        print(f"\n🤖 [Scheduler Action]  : {action_name} (Decision time: {sched_lat:.2f} µs)")
        print(f"\n📈 [Compression Results]:")
        print(f"  * Original Size    : {raw_len} bytes")
        print(f"  * Compressed Size  : {len(comp_bytes)} bytes")
        print(f"  * Compression Ratio: {ratio:.4f} ({saved:+.1f}% space saved)")
        print(f"  * Engine Latency   : {comp_lat:.2f} µs")
        print(f"  * Total E2E Latency: {sched_lat + comp_lat:.2f} µs")

if __name__ == "__main__":
    run_repetition_demo()
