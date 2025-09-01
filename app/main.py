"""
FastAPI主应用
ASGI应用入口，挂载静态文件、路由和中间件
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .settings import settings
from .logging import setup_logging, get_logger
from .middleware import RequestIDMiddleware, AccessLogMiddleware, ExceptionHandlerMiddleware
from .engine import get_engine_manager
from .cache import get_cache_manager
from .model_downloader import get_model_downloader
from .routers import v1
from .routers import v2
from .routers import cache
from .ui import router as ui_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    setup_logging()
    logger = get_logger("app.main")
    
    logger.info("Starting OCR Service")
    logger.info("Configuration: {}".format({
        'WORKERS': settings.WORKERS,
        'MODEL_CONCURRENCY': settings.MODEL_CONCURRENCY,
        'DEFAULT_MODEL': settings.DEFAULT_MODEL,
        'CACHE_ENABLED': settings.CACHE_ENABLED,
        'USE_GPU': settings.USE_GPU
    }))
    
    # Start background PP-OCRv5-Server model download
    downloader = get_model_downloader()
    if settings.DEFAULT_MODEL == "PP-OCRv5-Server":
        logger.info("Starting background download of PP-OCRv5-Server model")
        asyncio.create_task(downloader.ensure_model_available("PP-OCRv5-Server"))
    
    # ===== CACHE SERVICE INITIALIZATION =====
    logger.info("=== Cache Service Initialization Status ===")
    
    # 检查缓存系统状态
    cache_manager = get_cache_manager()
    if settings.CACHE_ENABLED:
        diagnostics = cache_manager.get_diagnostics()
        if diagnostics.initialization_error:
            logger.error(
                "=== Cache system initialization FAILED ===",
                extra={
                    "error": diagnostics.initialization_error,
                    "diskcache_available": diagnostics.diskcache_available,
                    "directory_writable": diagnostics.directory_writable,
                    "dependency_status": diagnostics.dependency_status
                }
            )
        elif not cache_manager.enabled:
            logger.warning(
                "=== Cache system is DISABLED ===",
                extra={
                    "diskcache_available": diagnostics.diskcache_available,
                    "directory_writable": diagnostics.directory_writable
                }
            )
        else:
            logger.info(
                "=== Cache system initialized SUCCESSFULLY ===",
                extra={
                    "cache_dir": diagnostics.cache_dir,
                    "cache_size_mb": diagnostics.cache_size_mb,
                    "total_keys": diagnostics.total_keys
                }
            )
    else:
        logger.info("=== Cache system is DISABLED by configuration ===")
    
    logger.info("=== End Cache Service Initialization ===")
    # ===== END CACHE SERVICE INITIALIZATION =====
    
    # 预热模型
    engine = get_engine_manager()
    engine.warmup()
    
    logger.info("OCR Service started successfully")
    
    yield
    
    # 关闭时执行
    logger.info("Shutting down OCR Service")
    
    # 关闭缓存管理器
    cache_manager = get_cache_manager()
    cache_manager.close()


# 创建FastAPI应用
app = FastAPI(
    title="OnnxOCR Service",
    description="高性能多语言OCR服务",
    version="2.0.0",
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(ExceptionHandlerMiddleware)
app.add_middleware(AccessLogMiddleware)
app.add_middleware(RequestIDMiddleware)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

# 注册路由
app.include_router(v1.router, tags=["v1-compatibility"])
app.include_router(v2.router, tags=["v2-api"])
app.include_router(cache.router, tags=["cache-management"])
app.include_router(ui_router, tags=["web-ui"])


# 根路径重定向（如果需要）
@app.get("/health")
async def health():
    """简单健康检查"""
    return {"status": "ok"}


@app.get("/healthz")
async def healthz():
    """Kubernetes风格健康检查"""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    """就绪检查"""
    from fastapi import HTTPException
    engine = get_engine_manager()
    if not engine.ready:
        raise HTTPException(status_code=503, detail={"status": "not ready", "ready": False})
    return {"status": "ready", "ready": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    )
