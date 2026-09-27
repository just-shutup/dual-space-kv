"""
Dual-Space Centroid KV (Academic v4.0): Position-Constrained RoPE-Decoupled Cache Compression
"""

from .streaming_cache import DualSpaceKVCache, WindowedDualSpaceCache, DualSpaceCacheLayer

__version__ = "4.0.0"
__all__ = ["DualSpaceKVCache", "WindowedDualSpaceCache", "DualSpaceCacheLayer"]
