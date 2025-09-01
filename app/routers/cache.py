"""
缓存管理API路由
提供缓存统计、清理和配置管理功能
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..cache import get_cache_manager
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