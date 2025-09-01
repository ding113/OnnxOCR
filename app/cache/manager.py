"""
OCR缓存管理器
基于diskcache + SHA-256实现图片OCR结果持久化缓存
"""

import hashlib
import time
import asyncio
import os
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Any, Dict
from dataclasses import dataclass
from pathlib import Path

from ..logging import get_logger
from ..settings import settings

logger = get_logger("app.cache.manager")


@dataclass
class OCRCacheResult:
    """缓存的OCR结果"""
    results: List[dict]       # 完整OCR结果(置信度=0时获取)
    processing_time: float    # 原始OCR处理时间
    cached_at: float         # 缓存时间戳


@dataclass
class CacheDiagnostics:
    """缓存诊断信息"""
    enabled: bool
    diskcache_available: bool
    directory_writable: bool
    initialization_error: Optional[str]
    cache_dir: str
    cache_size_mb: float
    total_keys: int
    dependency_status: Dict[str, str]


class CacheManager:
    """OCR缓存管理器"""
    
    def __init__(self):
        self._cache = None
        self._executor = None
        self._initialized = False
        self._initialization_error = None
        
        # 记录初始化状态
        logger.info(f"CacheManager starting initialization (settings.CACHE_ENABLED={settings.CACHE_ENABLED})")
        
        # 如果缓存被禁用，直接返回
        if not settings.CACHE_ENABLED:
            logger.info("Cache is disabled by configuration (CACHE_ENABLED=False)")
            return
            
        # 立即尝试初始化以便记录任何错误
        try:
            self._initialize()
        except Exception as e:
            logger.error(f"Cache initialization failed during __init__: {e}")
            self._initialization_error = str(e)
    
    def _check_dependencies(self) -> Dict[str, str]:
        """检查依赖包状态"""
        deps_status = {}
        
        # 检查diskcache
        try:
            import diskcache
            deps_status["diskcache"] = f"OK (version: {getattr(diskcache, '__version__', 'unknown')})"
        except ImportError as e:
            deps_status["diskcache"] = f"MISSING: {e}"
        except Exception as e:
            deps_status["diskcache"] = f"ERROR: {e}"
        
        # 检查其他依赖
        try:
            import sqlite3
            deps_status["sqlite3"] = f"OK (version: {sqlite3.sqlite_version})"
        except Exception as e:
            deps_status["sqlite3"] = f"ERROR: {e}"
            
        return deps_status
    
    def _check_directory_permissions(self) -> Tuple[bool, str]:
        """检查缓存目录权限"""
        try:
            cache_path = Path(settings.CACHE_DIR)
            
            # 检查目录是否存在，不存在则创建
            if not cache_path.exists():
                logger.info(f"Creating cache directory: {cache_path}")
                cache_path.mkdir(parents=True, exist_ok=True)
            
            # 检查是否可写
            test_file = cache_path / ".permission_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
                
                # 获取权限信息
                stat_info = cache_path.stat()
                permissions = oct(stat_info.st_mode)[-3:]
                
                return True, f"OK (permissions: {permissions})"
            except Exception as e:
                return False, f"NOT_WRITABLE: {e}"
                
        except Exception as e:
            return False, f"ERROR: {e}"
        
    def _initialize(self):
        """延迟初始化缓存"""
        if self._initialized:
            return
        
        if not settings.CACHE_ENABLED:
            logger.info("Cache initialization skipped (disabled)")
            return
        
        logger.info("Starting cache initialization...")
        
        try:
            # 1. 检查依赖包
            logger.debug("Step 1: Checking dependencies...")
            deps_status = self._check_dependencies()
            logger.info(f"Dependency check: {deps_status}")
            
            if ("MISSING" in deps_status.get("diskcache", "") or 
                "ERROR" in deps_status.get("diskcache", "")):
                error_msg = f"diskcache package not available: {deps_status['diskcache']}"
                logger.error(error_msg)
                self._initialization_error = error_msg
                self._initialized = False
                return
            
            # 2. 检查目录权限
            logger.debug("Step 2: Checking directory permissions...")
            dir_writable, dir_status = self._check_directory_permissions()
            logger.info(f"Directory check: {dir_status}")
            
            if not dir_writable:
                error_msg = f"Cache directory not writable: {dir_status}"
                logger.error(error_msg)
                self._initialization_error = error_msg
                self._initialized = False
                return
            
            # 3. 初始化diskcache
            logger.debug("Step 3: Initializing diskcache...")
            try:
                import diskcache
                
                logger.info(f"Initializing diskcache at {settings.CACHE_DIR}")
                self._cache = diskcache.Cache(
                    directory=settings.CACHE_DIR,
                    size_limit=settings.CACHE_SIZE_LIMIT,
                    eviction_policy='least-recently-used',
                    timeout=1.0,  # 避免长时间锁等待
                    disk_min_file_size=1024  # 小数据内嵌存储
                )
                logger.info("diskcache.Cache object created successfully")
                
                # 测试缓存操作
                logger.debug("Step 3a: Testing cache operations...")
                test_key = "__cache_test__"
                test_value = "test_value"
                
                # 测试写入
                self._cache.set(test_key, test_value, expire=5)
                logger.debug("Cache test write: OK")
                
                # 测试读取
                test_result = self._cache.get(test_key)
                logger.debug(f"Cache test read: {test_result}")
                
                # 测试删除
                self._cache.delete(test_key)
                logger.debug("Cache test delete: OK")
                
                if test_result != test_value:
                    raise RuntimeError(f"Cache test operation failed: expected '{test_value}', got '{test_result}'")
                
                logger.info("diskcache operations test passed")
                
            except ImportError as e:
                error_msg = f"Failed to import diskcache: {e}"
                logger.error(error_msg)
                self._initialization_error = error_msg
                self._initialized = False
                return
            except Exception as e:
                error_msg = f"diskcache initialization failed: {e}"
                logger.error(error_msg, exc_info=True)
                self._initialization_error = error_msg
                self._initialized = False
                return
            
            # 4. 初始化后台写入线程池
            logger.debug("Step 4: Initializing thread pool...")
            try:
                self._executor = ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="cache-writer"
                )
                logger.info("Cache thread pool initialized")
            except Exception as e:
                error_msg = f"Thread pool initialization failed: {e}"
                logger.error(error_msg, exc_info=True)
                self._initialization_error = error_msg
                self._initialized = False
                return
            
            # 5. 成功初始化
            self._initialized = True
            self._initialization_error = None  # 清除任何之前的错误
            
            cache_size_mb = self.get_cache_size_mb()
            total_keys = len(self._cache) if self._cache else 0
            
            logger.info(
                "Cache manager initialized successfully",
                extra={
                    "cache_dir": settings.CACHE_DIR,
                    "size_limit_gb": settings.CACHE_SIZE_LIMIT / (1024**3),
                    "current_size_mb": cache_size_mb,
                    "total_keys": total_keys
                }
            )
            
        except Exception as e:
            error_msg = f"Cache initialization failed with unexpected error: {e}"
            logger.error(error_msg, exc_info=True)
            self._initialization_error = error_msg
            self._initialized = False
            
            # 清理任何部分初始化的资源
            if hasattr(self, '_cache') and self._cache:
                try:
                    self._cache.close()
                except Exception:
                    pass
                self._cache = None
                
            if hasattr(self, '_executor') and self._executor:
                try:
                    self._executor.shutdown(wait=False)
                except Exception:
                    pass
                self._executor = None
    
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
    
    def get_diagnostics(self) -> CacheDiagnostics:
        """获取缓存系统详细诊断信息"""
        try:
            # 尝试初始化以获取最新状态
            self._initialize()
            
            deps_status = self._check_dependencies()
            dir_writable, dir_status = self._check_directory_permissions()
            
            # 检查diskcache是否可用
            diskcache_available = ("MISSING" not in deps_status.get("diskcache", "") and 
                                 "ERROR" not in deps_status.get("diskcache", ""))
            
            # 判断缓存是否真正可用：需要配置启用、初始化成功、依赖可用
            cache_actually_enabled = (settings.CACHE_ENABLED and 
                                    self._initialized and 
                                    diskcache_available and 
                                    dir_writable)
            
            logger.debug(
                "Cache diagnostics",
                extra={
                    "config_enabled": settings.CACHE_ENABLED,
                    "initialized": self._initialized, 
                    "diskcache_available": diskcache_available,
                    "directory_writable": dir_writable,
                    "actually_enabled": cache_actually_enabled,
                    "initialization_error": self._initialization_error
                }
            )
            
            return CacheDiagnostics(
                enabled=cache_actually_enabled,
                diskcache_available=diskcache_available,
                directory_writable=dir_writable,
                initialization_error=self._initialization_error,
                cache_dir=settings.CACHE_DIR,
                cache_size_mb=self.get_cache_size_mb(),
                total_keys=len(self._cache) if self._cache else 0,
                dependency_status=deps_status
            )
        except Exception as e:
            logger.error(f"Failed to get cache diagnostics: {e}")
            return CacheDiagnostics(
                enabled=False,
                diskcache_available=False,
                directory_writable=False,
                initialization_error=str(e),
                cache_dir=settings.CACHE_DIR,
                cache_size_mb=0.0,
                total_keys=0,
                dependency_status={"error": str(e)}
            )
    
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