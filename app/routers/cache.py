"""
缓存管理API路由
提供缓存统计、清理和配置管理功能
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..cache import get_cache_manager
from ..cache.manager import CacheDiagnostics
from ..logging import get_logger
from ..settings import settings

logger = get_logger("app.routes.cache")

router = APIRouter(prefix="/api/v2/cache", tags=["cache"])


class CacheStatsResponse(BaseModel):
    """缓存统计响应"""
    enabled: bool
    total_keys: Optional[int] = None
    cache_size_mb: Optional[float] = None
    size_limit_gb: Optional[float] = None
    hits: Optional[int] = None
    misses: Optional[int] = None
    hit_rate: Optional[float] = None
    evictions: Optional[int] = None
    error: Optional[str] = None


class CacheClearResponse(BaseModel):
    """缓存清理响应"""
    success: bool
    cleared_count: Optional[int] = None
    criteria: Optional[str] = None
    error: Optional[str] = None


class CacheConfigResponse(BaseModel):
    """缓存配置响应"""
    enabled: bool
    cache_dir: str
    size_limit_gb: float
    ttl_days: int


class CacheDiagnosticsResponse(BaseModel):
    """缓存诊断响应"""
    enabled: bool
    diskcache_available: bool
    directory_writable: bool
    initialization_error: Optional[str] = None
    cache_dir: str
    cache_size_mb: float
    total_keys: int
    dependency_status: Dict[str, str]


class CacheHealthResponse(BaseModel):
    """缓存健康检查响应"""
    status: str  # healthy, unhealthy, disabled, initializing, error
    message: Optional[str] = None
    cache_size_mb: Optional[float] = None
    total_keys: Optional[int] = None
    initialization_error: Optional[str] = None
    directory_writable: Optional[bool] = None
    diskcache_available: Optional[bool] = None


@router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """获取缓存统计信息"""
    try:
        cache_manager = get_cache_manager()
        stats = cache_manager.get_cache_stats()
        
        logger.info("Cache stats requested", extra={"stats": stats})
        
        return CacheStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to get cache stats: {str(e)}", "code": "CACHE_STATS_ERROR"}
        )


@router.post("/clear", response_model=CacheClearResponse)
async def clear_cache(older_than_hours: Optional[int] = None):
    """
    清理缓存
    
    Args:
        older_than_hours: 清理多少小时之前的缓存，不指定则清理全部
    """
    try:
        cache_manager = get_cache_manager()
        
        if not cache_manager.enabled:
            return CacheClearResponse(
                success=False,
                error="Cache is not enabled"
            )
        
        result = cache_manager.clear_cache(older_than_hours=older_than_hours)
        
        if "error" in result:
            return CacheClearResponse(
                success=False,
                error=result["error"]
            )
        
        logger.info(
            "Cache cleared",
            extra={
                "cleared_count": result.get("cleared_count", 0),
                "criteria": result.get("criteria", "unknown")
            }
        )
        
        return CacheClearResponse(
            success=True,
            cleared_count=result.get("cleared_count", 0),
            criteria=result.get("criteria", "unknown")
        )
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to clear cache: {str(e)}", "code": "CACHE_CLEAR_ERROR"}
        )


@router.get("/diagnostics", response_model=CacheDiagnosticsResponse)
async def get_cache_diagnostics():
    """获取缓存系统详细诊断信息"""
    try:
        cache_manager = get_cache_manager()
        diagnostics = cache_manager.get_diagnostics()
        
        logger.info("Cache diagnostics requested", extra={
            "enabled": diagnostics.enabled,
            "diskcache_available": diagnostics.diskcache_available,
            "directory_writable": diagnostics.directory_writable,
            "initialization_error": diagnostics.initialization_error,
            "cache_size_mb": diagnostics.cache_size_mb,
            "total_keys": diagnostics.total_keys
        })
        
        return CacheDiagnosticsResponse(
            enabled=diagnostics.enabled,
            diskcache_available=diagnostics.diskcache_available,
            directory_writable=diagnostics.directory_writable,
            initialization_error=diagnostics.initialization_error,
            cache_dir=diagnostics.cache_dir,
            cache_size_mb=diagnostics.cache_size_mb,
            total_keys=diagnostics.total_keys,
            dependency_status=diagnostics.dependency_status
        )
        
    except Exception as e:
        logger.error(f"Failed to get cache diagnostics: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to get cache diagnostics: {str(e)}", "code": "CACHE_DIAGNOSTICS_ERROR"}
        )


@router.get("/health", response_model=CacheHealthResponse)
async def cache_health():
    """检查缓存系统健康状态"""
    try:
        cache_manager = get_cache_manager()
        diagnostics = cache_manager.get_diagnostics()
        
        logger.info("Cache health check requested", extra={
            "diagnostics_enabled": diagnostics.enabled,
            "config_enabled": settings.CACHE_ENABLED,
            "diskcache_available": diagnostics.diskcache_available,
            "directory_writable": diagnostics.directory_writable,
            "initialization_error": diagnostics.initialization_error
        })
        
        # 基于配置判断基本状态
        if not settings.CACHE_ENABLED:
            return CacheHealthResponse(
                status="disabled",
                message="Cache is disabled by configuration (CACHE_ENABLED=false)",
                diskcache_available=diagnostics.diskcache_available,
                directory_writable=diagnostics.directory_writable
            )
        
        # 检查各种条件并收集问题
        issues = []
        
        if not diagnostics.diskcache_available:
            issues.append("diskcache library not available")
        
        if not diagnostics.directory_writable:
            issues.append("cache directory not writable")
            
        if diagnostics.initialization_error:
            issues.append(f"initialization failed: {diagnostics.initialization_error}")
        
        # 基于 diagnostics.enabled 判断最终状态
        if diagnostics.enabled:
            # 缓存完全可用
            status = "healthy"
            message = "Cache system is operating normally"
        elif issues:
            # 有具体问题
            status = "unhealthy"  
            message = f"Cache issues detected: {'; '.join(issues)}"
        else:
            # 配置启用但初始化未完成（可能仍在进行中）
            status = "initializing"
            message = "Cache system is enabled but not yet initialized"
            
        logger.info("Cache health check completed", extra={
            "status": status,
            "issues": issues,
            "cache_size_mb": diagnostics.cache_size_mb,
            "total_keys": diagnostics.total_keys,
            "actually_enabled": diagnostics.enabled
        })
        
        return CacheHealthResponse(
            status=status,
            message=message,
            cache_size_mb=diagnostics.cache_size_mb if diagnostics.enabled else None,
            total_keys=diagnostics.total_keys if diagnostics.enabled else None,
            initialization_error=diagnostics.initialization_error,
            directory_writable=diagnostics.directory_writable,
            diskcache_available=diagnostics.diskcache_available
        )
        
    except Exception as e:
        logger.error(f"Cache health check failed with unexpected error: {e}", exc_info=True)
        return CacheHealthResponse(
            status="error",
            message=f"Health check failed: {str(e)}"
        )


@router.get("/config", response_model=CacheConfigResponse)
async def get_cache_config():
    """获取缓存配置信息"""
    try:
        return CacheConfigResponse(
            enabled=settings.CACHE_ENABLED,
            cache_dir=settings.CACHE_DIR,
            size_limit_gb=settings.CACHE_SIZE_LIMIT / (1024**3),
            ttl_days=settings.CACHE_TTL_DAYS
        )
        
    except Exception as e:
        logger.error(f"Failed to get cache config: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to get cache config: {str(e)}", "code": "CACHE_CONFIG_ERROR"}
        )


@router.get("/health")
async def cache_health():
    """检查缓存系统健康状态"""
    try:
        cache_manager = get_cache_manager()
        
        if not cache_manager.enabled:
            return {
                "status": "disabled",
                "message": "Cache is disabled"
            }
        
        # 尝试获取统计信息来测试缓存系统
        stats = cache_manager.get_cache_stats()
        
        if "error" in stats:
            return {
                "status": "unhealthy", 
                "message": f"Cache error: {stats['error']}"
            }
        
        return {
            "status": "healthy",
            "cache_size_mb": stats.get("cache_size_mb", 0),
            "total_keys": stats.get("total_keys", 0)
        }
        
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}"
        }