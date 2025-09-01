"""
OCR缓存管理器
基于diskcache + SHA-256实现图片OCR结果持久化缓存
"""

import hashlib
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass
import diskcache

from ..logging import get_logger
from ..settings import settings

logger = get_logger("app.cache.manager")


@dataclass
class OCRCacheResult:
    """缓存的OCR结果"""
    results: List[dict]       # 完整OCR结果(置信度=0时获取)
    processing_time: float    # 原始OCR处理时间
    cached_at: float         # 缓存时间戳


class CacheManager:
    """OCR缓存管理器"""
    
    def __init__(self):
        self._cache = None
        self._executor = None
        self._initialized = False
        
    def _initialize(self):
        """延迟初始化缓存"""
        if self._initialized or not settings.CACHE_ENABLED:
            return
        
        try:
            # 初始化diskcache
            self._cache = diskcache.Cache(
                directory=settings.CACHE_DIR,
                size_limit=settings.CACHE_SIZE_LIMIT,
                eviction_policy='least-recently-used',
                timeout=1.0,  # 避免长时间锁等待
                disk_min_file_size=1024  # 小数据内嵌存储
            )
            
            # 初始化后台写入线程池
            self._executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="cache-writer"
            )
            
            self._initialized = True
            logger.info(
                "Cache manager initialized",
                extra={
                    "cache_dir": settings.CACHE_DIR,
                    "size_limit_gb": settings.CACHE_SIZE_LIMIT / (1024**3),
                    "current_size_mb": self.get_cache_size_mb()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            self._initialized = False
    
    @property
    def enabled(self) -> bool:
        """检查缓存是否启用"""
        return settings.CACHE_ENABLED and self._initialized
    
    def _compute_cache_key(self, image_bytes: bytes, model_name: str) -> str:
        """计算缓存键"""
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        return f"{model_name}:{image_hash}"
    
    def get_cached_result(self, image_bytes: bytes, model_name: str, conf_threshold: float) -> Optional[Tuple[float, List]]:
        """获取缓存的OCR结果"""
        if not self.enabled:
            return None
        
        self._initialize()
        if not self._cache:
            return None
        
        try:
            cache_key = self._compute_cache_key(image_bytes, model_name)
            cached = self._cache.get(cache_key)
            
            if cached:
                # 缓存命中 - 按置信度过滤结果
                filtered_results = self._filter_by_confidence(cached.results, conf_threshold)
                logger.debug(
                    "Cache hit",
                    extra={
                        "cache_key": cache_key[:32] + "...",
                        "original_count": len(cached.results),
                        "filtered_count": len(filtered_results),
                        "conf_threshold": conf_threshold
                    }
                )
                return 0.001, filtered_results  # 缓存命中几乎无耗时
            
            return None
            
        except Exception as e:
            logger.error(f"Cache lookup error: {e}")
            return None
    
    def cache_result_async(self, image_bytes: bytes, model_name: str, ocr_results: List, processing_time: float):
        """异步缓存OCR结果"""
        if not self.enabled:
            return
        
        self._initialize()
        if not self._cache or not self._executor:
            return
        
        try:
            cache_key = self._compute_cache_key(image_bytes, model_name)
            cache_result = OCRCacheResult(
                results=ocr_results,
                processing_time=processing_time,
                cached_at=time.time()
            )
            
            # 提交到后台线程执行写入
            self._executor.submit(self._write_to_cache, cache_key, cache_result)
            
        except Exception as e:
            logger.error(f"Cache write submission error: {e}")
    
    def _write_to_cache(self, cache_key: str, cache_result: OCRCacheResult):
        """后台线程中执行缓存写入"""
        try:
            self._cache.set(cache_key, cache_result)
            logger.debug(f"Cached result for key: {cache_key[:32]}...")
            
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    def _filter_by_confidence(self, cached_results: List, threshold: float) -> List:
        """从缓存结果中按置信度过滤"""
        filtered = []
        
        for line in cached_results:
            if isinstance(line, list) and len(line) >= 2:
                # 标准OCR结果格式: [[bbox], [text, confidence]]
                if len(line[1]) >= 2:
                    confidence = float(line[1][1])
                    if confidence >= threshold:
                        filtered.append(line)
        
        return [filtered] if filtered else []
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        if not self.enabled:
            return {"enabled": False}
        
        self._initialize()
        if not self._cache:
            return {"enabled": False, "error": "Cache not initialized"}
        
        try:
            stats = self._cache.stats(enable=True)
            return {
                "enabled": True,
                "total_keys": len(self._cache),
                "cache_size_mb": self.get_cache_size_mb(),
                "size_limit_gb": settings.CACHE_SIZE_LIMIT / (1024**3),
                "hits": getattr(stats, 'hits', 0),
                "misses": getattr(stats, 'misses', 0),
                "hit_rate": getattr(stats, 'hit_rate', 0.0),
                "evictions": getattr(stats, 'evictions', 0)
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"enabled": True, "error": str(e)}
    
    def get_cache_size_mb(self) -> float:
        """获取缓存大小(MB)"""
        if not self.enabled or not self._cache:
            return 0.0
        
        try:
            return self._cache.volume() / (1024 * 1024)
        except Exception:
            return 0.0
    
    def clear_cache(self, older_than_hours: Optional[int] = None) -> dict:
        """清理缓存"""
        if not self.enabled:
            return {"error": "Cache not enabled"}
        
        self._initialize()
        if not self._cache:
            return {"error": "Cache not initialized"}
        
        try:
            if older_than_hours:
                # 清理指定时间之前的缓存
                cutoff_time = time.time() - (older_than_hours * 3600)
                cleared_count = 0
                
                for key in list(self._cache.keys()):
                    try:
                        cached = self._cache.get(key)
                        if cached and cached.cached_at < cutoff_time:
                            del self._cache[key]
                            cleared_count += 1
                    except Exception:
                        continue
                
                return {"cleared_count": cleared_count, "criteria": f"older than {older_than_hours}h"}
            else:
                # 清理全部缓存
                original_count = len(self._cache)
                self._cache.clear()
                return {"cleared_count": original_count, "criteria": "all"}
                
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return {"error": str(e)}
    
    def close(self):
        """关闭缓存管理器"""
        if self._executor:
            self._executor.shutdown(wait=True)
        if self._cache:
            self._cache.close()


# 全局缓存管理器实例
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """获取缓存管理器实例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager