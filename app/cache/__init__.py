"""
OCR缓存模块
基于diskcache实现持久化图片缓存
"""

from .manager import CacheManager, get_cache_manager

__all__ = ["CacheManager", "get_cache_manager"]