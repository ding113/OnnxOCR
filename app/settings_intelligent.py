"""
FastAPI应用配置管理
支持基于系统资源的智能自适应配置
"""
import os
import multiprocessing
import time
import logging
from typing import Optional, Tuple, Dict

# 尝试导入psutil进行精确资源监控
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class Settings:
    """智能配置类 - 基于系统资源自适应优化"""
    
    # 服务器基础配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "5005"))
    
    # 模型配置
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "PP-OCRv5-Server")
    MODEL_POOL_SIZE: int = int(os.getenv("MODEL_POOL_SIZE", "1"))
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
    
    # 缓存配置
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_DIR: str = os.getenv("CACHE_DIR", "/cache_data")
    CACHE_SIZE_LIMIT: int = int(os.getenv("CACHE_SIZE_LIMIT_GB", "10")) * 1024**3
    CACHE_TTL_DAYS: int = int(os.getenv("CACHE_TTL_DAYS", "90"))
    
    def __init__(self):
        # 智能配置 - 核心改进
        self._setup_intelligent_config()
        
        # 确保目录存在
        self._ensure_directories()
    
    def _setup_intelligent_config(self):
        """设置智能配置"""
        # 优先使用环境变量
        if os.getenv("WORKERS") and os.getenv("MODEL_CONCURRENCY"):
            self.WORKERS = int(os.getenv("WORKERS"))
            self.MODEL_CONCURRENCY = int(os.getenv("MODEL_CONCURRENCY"))
            self.THREADS = int(os.getenv("THREADS", "2"))
            print(f"Using manual configuration: WORKERS={self.WORKERS}, CONCURRENCY={self.MODEL_CONCURRENCY}")
        else:
            # 智能自适应配置
            self.WORKERS, self.MODEL_CONCURRENCY = self._calculate_optimal_config()
            self.THREADS = max(2, min(8, self.WORKERS // 4))  # 动态线程数
            
            print(f"Intelligent configuration calculated:")
            print(f"   WORKERS: {self.WORKERS}")
            print(f"   MODEL_CONCURRENCY: {self.MODEL_CONCURRENCY}")
            print(f"   THREADS: {self.THREADS}")
            print(f"   Theoretical max concurrent tasks: {self.WORKERS * self.MODEL_CONCURRENCY}")
    
    def _calculate_optimal_config(self) -> Tuple[int, int]:
        """核心智能配置算法"""
        cpu_count = multiprocessing.cpu_count()
        
        # 获取系统资源信息
        memory_info = self._get_memory_info()
        cpu_info = self._get_cpu_performance()
        
        # 配置约束和参数
        MAX_WORKERS = min(64, cpu_count)  # 最大worker数，避免管理开销
        MAX_CONCURRENCY = 8               # 最大并发数，避免锁竞争
        MIN_WORKERS = max(4, cpu_count // 24)  # 最小worker数，保证基本性能
        MIN_CONCURRENCY = 1
        
        MEMORY_PER_MODEL_GB = 1.0         # 每模型实例内存占用
        SYSTEM_RESERVE_GB = max(8, memory_info['total'] * 0.15)  # 系统预留内存
        
        # Step 1: 计算最优WORKERS数量
        if cpu_count >= 64:  # 高端服务器
            # 目标70%CPU利用率，考虑I/O等待
            target_utilization = 0.75
            io_wait_factor = 0.25
            base_workers = int(cpu_count * target_utilization / (1 - io_wait_factor))
            optimal_workers = min(base_workers, MAX_WORKERS)
        elif cpu_count >= 32:  # 中端服务器
            optimal_workers = min(cpu_count // 2, MAX_WORKERS)
        else:  # 小型服务器
            optimal_workers = min(cpu_count, MAX_WORKERS)
        
        optimal_workers = max(MIN_WORKERS, optimal_workers)
        
        # Step 2: 计算最优MODEL_CONCURRENCY
        # 内存约束
        usable_memory = memory_info['available'] - SYSTEM_RESERVE_GB
        max_models_by_memory = max(1, int(usable_memory / MEMORY_PER_MODEL_GB))
        
        # CPU约束 - 动态调整核心/并发比
        if cpu_count >= 64:
            cores_per_concurrency = 8  # 高端服务器：每8核1并发
        elif cpu_count >= 32:
            cores_per_concurrency = 6  # 中端服务器：每6核1并发
        else:
            cores_per_concurrency = 4  # 小型服务器：每4核1并发
        
        cpu_based_concurrency = max(1, cpu_count // cores_per_concurrency)
        
        # 分配约束
        memory_based_concurrency = max(1, max_models_by_memory // optimal_workers)
        
        # 选择最限制性的约束
        optimal_concurrency = min(
            cpu_based_concurrency,
            memory_based_concurrency,
            MAX_CONCURRENCY
        )
        optimal_concurrency = max(MIN_CONCURRENCY, optimal_concurrency)
        
        # Step 3: 最终验证和调整
        total_models = optimal_workers * optimal_concurrency
        estimated_memory_usage = total_models * MEMORY_PER_MODEL_GB
        
        # 内存安全检查
        memory_usage_ratio = estimated_memory_usage / memory_info['available']
        if memory_usage_ratio > 0.8:  # 不超过80%可用内存
            scale_factor = 0.8 / memory_usage_ratio
            optimal_concurrency = max(1, int(optimal_concurrency * scale_factor))
        
        # 性能优化调整
        if cpu_count >= 96:  # 96核系统特殊优化
            # 确保充分利用超高核心数
            if optimal_workers < 32:
                optimal_workers = min(48, cpu_count // 2)
                # 重新计算并发数
                optimal_concurrency = min(
                    max(2, cpu_count // 32),  # 每32核至少2个并发
                    max_models_by_memory // optimal_workers,
                    MAX_CONCURRENCY
                )
        
        # 记录配置决策
        self._log_config_decision({
            'cpu_count': cpu_count,
            'memory_total_gb': memory_info['total'],
            'memory_available_gb': memory_info['available'],
            'optimal_workers': optimal_workers,
            'optimal_concurrency': optimal_concurrency,
            'total_model_instances': optimal_workers * optimal_concurrency,
            'estimated_memory_usage_gb': optimal_workers * optimal_concurrency * MEMORY_PER_MODEL_GB,
            'memory_utilization_percent': round(
                (optimal_workers * optimal_concurrency * MEMORY_PER_MODEL_GB / memory_info['available']) * 100, 1
            )
        })
        
        return optimal_workers, optimal_concurrency
    
    def _get_memory_info(self) -> Dict[str, float]:
        """获取内存信息"""
        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                return {
                    'total': mem.total / (1024**3),
                    'available': mem.available / (1024**3),
                    'used': mem.used / (1024**3)
                }
            except Exception:
                pass
        
        # 保守估算策略
        cpu_count = multiprocessing.cpu_count()
        if cpu_count >= 64:
            # 高端双路服务器通常有256GB+内存
            estimated_total = max(256, cpu_count * 4)
        elif cpu_count >= 32:
            estimated_total = max(128, cpu_count * 3)
        else:
            estimated_total = max(64, cpu_count * 2)
        
        return {
            'total': estimated_total,
            'available': estimated_total * 0.7,  # 假设70%可用
            'used': estimated_total * 0.3
        }
    
    def _get_cpu_performance(self) -> Dict[str, float]:
        """获取CPU性能指标"""
        if PSUTIL_AVAILABLE:
            try:
                return {
                    'usage_percent': psutil.cpu_percent(interval=0.1),
                    'load_avg': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
                }
            except Exception:
                pass
        return {'usage_percent': 30, 'load_avg': 2.0}
    
    def _log_config_decision(self, config_data: Dict):
        """记录配置决策过程"""
        print(f"System Resource Analysis:")
        print(f"   CPU Cores: {config_data['cpu_count']}")
        print(f"   Total Memory: {config_data['memory_total_gb']:.1f}GB")
        print(f"   Available Memory: {config_data['memory_available_gb']:.1f}GB")
        print(f"Optimal Configuration:")
        print(f"   Workers: {config_data['optimal_workers']}")
        print(f"   Model Concurrency: {config_data['optimal_concurrency']}")
        print(f"   Total Model Instances: {config_data['total_model_instances']}")
        print(f"   Estimated Memory Usage: {config_data['estimated_memory_usage_gb']:.1f}GB")
        print(f"   Memory Utilization: {config_data['memory_utilization_percent']}%")
        print(f"Performance Estimate:")
        theoretical_qps = config_data['total_model_instances'] * (1000/300)  # 假设300ms/图
        print(f"   Theoretical Max QPS: ~{theoretical_qps:.1f} images/sec")
        print(f"   CPU Utilization Target: ~70%")
    
    def _ensure_directories(self):
        """确保目录存在"""
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        if self.CACHE_ENABLED:
            os.makedirs(self.CACHE_DIR, exist_ok=True)


# 全局配置实例
settings = Settings()