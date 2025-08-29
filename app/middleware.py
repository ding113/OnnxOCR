"""
FastAPI中间件
包含请求ID注入、访问日志、异常处理
"""
import time
import uuid
import traceback
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .logging import get_logger

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
    """访问日志中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")
        
        # 记录请求开始
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client": request.client.host if request.client else "unknown",
            }
        )
        
        try:
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 记录请求完成
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "status_code": response.status_code,
                    "process_time": "{:.3f}s".format(process_time),
                }
            )
            
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