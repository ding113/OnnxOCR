"""
FastAPI中间件
包含请求ID注入、访问日志、异常处理、缓存监控
"""
import time
import uuid
import traceback
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .logging import get_logger
from .cache import get_cache_manager

logger = get_logger("app.middleware")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """请求ID中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 获取或生成请求ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        
        # 注入到request state中
        request.state.request_id = request_id
        
        # 调用下一个中间件/路由
        response = await call_next(request)
        
        # 在响应头中返回请求ID
        response.headers["X-Request-ID"] = request_id
        
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """访问日志中间件（含缓存监控）"""
    
    def __init__(self, app, cache_stats_interval: int = 100):
        super().__init__(app)
        self.cache_stats_interval = cache_stats_interval
        self.request_count = 0
        self.last_cache_log_time = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")
        
        # 检查是否是OCR请求
        is_ocr_request = any(path in str(request.url.path) for path in ["/ocr", "/api/v1/ocr", "/api/v2/ocr"])
        
        # 记录请求开始
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client": request.client.host if request.client else "unknown",
                "is_ocr_request": is_ocr_request,
            }
        )
        
        try:
            # 获取请求前的缓存统计（仅OCR请求）
            cache_stats_before = None
            if is_ocr_request:
                try:
                    cache_manager = get_cache_manager()
                    if cache_manager.enabled:
                        cache_stats_before = cache_manager.get_cache_stats()
                except Exception:
                    pass  # 忽略缓存统计错误，不影响正常处理
            
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 获取请求后的缓存统计并计算缓存命中情况
            cache_hit_info = {}
            if is_ocr_request and cache_stats_before:
                try:
                    cache_manager = get_cache_manager()
                    if cache_manager.enabled:
                        cache_stats_after = cache_manager.get_cache_stats()
                        
                        # 计算此次请求是否为缓存命中
                        hits_diff = cache_stats_after.get("hits", 0) - cache_stats_before.get("hits", 0)
                        misses_diff = cache_stats_after.get("misses", 0) - cache_stats_before.get("misses", 0)
                        
                        cache_hit_info = {
                            "cache_hit": hits_diff > 0,
                            "cache_miss": misses_diff > 0,
                            "cache_total_keys": cache_stats_after.get("total_keys", 0),
                            "cache_size_mb": cache_stats_after.get("cache_size_mb", 0),
                        }
                except Exception:
                    pass
            
            # 记录请求完成
            log_extra = {
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "process_time": "{:.3f}s".format(process_time),
                "is_ocr_request": is_ocr_request,
            }
            log_extra.update(cache_hit_info)
            
            logger.info("Request completed", extra=log_extra)
            
            # 周期性记录缓存统计
            self.request_count += 1
            current_time = time.time()
            if (self.request_count % self.cache_stats_interval == 0 or 
                current_time - self.last_cache_log_time > 300):  # 每300秒至少记录一次
                
                try:
                    cache_manager = get_cache_manager()
                    if cache_manager.enabled:
                        stats = cache_manager.get_cache_stats()
                        logger.info(
                            "Cache statistics",
                            extra={
                                "cache_enabled": True,
                                "total_keys": stats.get("total_keys", 0),
                                "cache_size_mb": stats.get("cache_size_mb", 0),
                                "hit_rate": stats.get("hit_rate", 0.0),
                                "hits": stats.get("hits", 0),
                                "misses": stats.get("misses", 0),
                                "evictions": stats.get("evictions", 0),
                            }
                        )
                    else:
                        logger.info("Cache statistics", extra={"cache_enabled": False})
                    
                    self.last_cache_log_time = current_time
                except Exception as e:
                    logger.warning(f"Failed to get cache statistics: {e}")
            
            # 添加处理时间头
            response.headers["X-Process-Time"] = "{:.3f}".format(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            # 记录错误
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "error": str(e),
                    "process_time": "{:.3f}s".format(process_time),
                    "traceback": traceback.format_exc(),
                    "is_ocr_request": is_ocr_request,
                }
            )
            
            raise


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """全局异常处理中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            request_id = getattr(request.state, "request_id", "unknown")
            
            logger.error(
                "Unhandled exception",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            
            # 返回统一的错误响应
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "code": "INTERNAL_ERROR",
                    "request_id": request_id,
                },
            )