import time
from typing import Union
import zstandard as zstd
import brotli
import gzip
import lz4.block
import zlib
import bz2
import lzma

class CompressionEngine:
    def __init__(self):
        # Initialize reusable compressors if needed
        self.zstd_compressor = zstd.ZstdCompressor(level=3)
        self.zstd_decompressor = zstd.ZstdDecompressor()

    def compress(self, text: Union[str, bytes], codec: str) -> tuple[bytes, float]:
        """
        Compresses input text or raw bytes using the specified codec.
        Returns:
            (compressed_bytes, latency_microseconds)
        """
        data = text.encode('utf-8') if isinstance(text, str) else text
        start_time = time.perf_counter()
        
        if codec == 'SKIP':
            compressed = data
        elif codec == 'ZSTD':
            compressed = self.zstd_compressor.compress(data)
        elif codec == 'BROTLI':
            compressed = brotli.compress(data, quality=4)
        elif codec == 'GZIP':
            compressed = gzip.compress(data, compresslevel=6)
        elif codec == 'LZ4':
            compressed = lz4.block.compress(data, store_size=True)
        elif codec == 'ZLIB':
            compressed = zlib.compress(data)
        elif codec == 'BZ2':
            compressed = bz2.compress(data)
        elif codec == 'LZMA':
            compressed = lzma.compress(data)
        else:
            raise ValueError(f"Unknown codec: {codec}")
            
        latency = (time.perf_counter() - start_time) * 1_000_000  # Convert to microseconds
        return compressed, latency

    def decompress(self, compressed_bytes: bytes, codec: str, raw_bytes: bool = False) -> tuple[Union[str, bytes], float]:
        """
        Decompresses input bytes using the specified codec.
        Args:
            compressed_bytes: Input compressed payload.
            codec: Codec name.
            raw_bytes: If True, returns raw decompressed bytes without UTF-8 decoding.
        Returns:
            (decompressed_payload, latency_microseconds)
        """
        start_time = time.perf_counter()
        
        if codec == 'SKIP':
            decompressed_data = compressed_bytes
        elif codec == 'ZSTD':
            decompressed_data = self.zstd_decompressor.decompress(compressed_bytes)
        elif codec == 'BROTLI':
            decompressed_data = brotli.decompress(compressed_bytes)
        elif codec == 'GZIP':
            decompressed_data = gzip.decompress(compressed_bytes)
        elif codec == 'LZ4':
            decompressed_data = lz4.block.decompress(compressed_bytes)
        elif codec == 'ZLIB':
            decompressed_data = zlib.decompress(compressed_bytes)
        elif codec == 'BZ2':
            decompressed_data = bz2.decompress(compressed_bytes)
        elif codec == 'LZMA':
            decompressed_data = lzma.decompress(compressed_bytes)
        else:
            raise ValueError(f"Unknown codec: {codec}")
            
        latency = (time.perf_counter() - start_time) * 1_000_000  # Convert to microseconds
        if raw_bytes:
            return decompressed_data, latency
        return decompressed_data.decode('utf-8'), latency
