"""
FastAPI应用配置管理
从环境变量读取配置，支持自适应参数调优
"""
import os
import multiprocessing
from typing import Optional


class Settings:
    """应用配置类"""
    
    # 服务器配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "5005"))
    WORKERS: int = int(os.getenv("WORKERS", min(4, multiprocessing.cpu_count() * 2)))
    THREADS: int = int(os.getenv("THREADS", "2"))
    
    # 模型配置
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "PP-OCRv5")
    MODEL_POOL_SIZE: int = int(os.getenv("MODEL_POOL_SIZE", "1"))
    MODEL_CONCURRENCY: int = int(os.getenv("MODEL_CONCURRENCY", "1"))
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() == "true"
    WARMUP: bool = os.getenv("WARMUP", "true").lower() == "true"
    
    # 上传配置
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
    MAX_CONTENT_LENGTH: int = MAX_UPLOAD_MB * 1024 * 1024
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s %(name)s %(message)s")
    
    # 目录配置
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR: str = os.path.join(BASE_DIR, "results")
    TEMPLATES_DIR: str = os.path.join(BASE_DIR, "templates")
    STATIC_DIR: str = os.path.join(BASE_DIR, "static")
    
    # 确保目录存在
    def __init__(self):
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        
        # 自适应调优
        if not os.getenv("WORKERS"):
            self.WORKERS = self._auto_workers()
        if not os.getenv("MODEL_CONCURRENCY"):
            self.MODEL_CONCURRENCY = self._auto_concurrency()
    
    def _auto_workers(self) -> int:
        """自动推导worker数量"""
        cpu_count = multiprocessing.cpu_count()
        return min(4, max(1, cpu_count * 2))
    
    def _auto_concurrency(self) -> int:
        """自动推导模型并发数"""
        # 基于CPU核数和内存考虑
        return max(1, min(2, multiprocessing.cpu_count() // 2))


# 全局配置实例
settings = Settings()