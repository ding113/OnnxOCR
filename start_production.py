#!/usr/bin/env python3
"""
Modern ONNX OCR Service - Production Startup Script
生产环境启动脚本

Features:
- Multi-worker production server
- Automatic CPU core detection
- Production-optimized settings
- Health monitoring integration
- Prometheus metrics enabled
"""

import os
import sys
import multiprocessing
from pathlib import Path

import uvicorn

def get_worker_count() -> int:
    """
    Calculate optimal worker count for production
    基于CPU核心数自动计算最优worker数量
    """
    cpu_count = multiprocessing.cpu_count()
    
    # Get worker count from environment or calculate optimal
    workers_env = os.getenv("WORKERS", "auto")
    
    if workers_env == "auto":
        # Production formula: 2 * CPU cores + 1 (but cap at 8 for memory efficiency)
        workers = min(2 * cpu_count + 1, 8)
    else:
        try:
            workers = int(workers_env)
            workers = max(1, min(workers, 16))  # Clamp between 1-16
        except ValueError:
            workers = min(2 * cpu_count + 1, 8)
    
    return workers

def main():
    """Production server startup with optimized settings"""
    
    # Environment configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5005))
    log_level = os.getenv("LOG_LEVEL", "info")
    workers = get_worker_count()
    
    print(f"[INFO] Starting Modern ONNX OCR Service - Production Mode")
    print(f"[INFO] Host: {host}:{port}")
    print(f"[INFO] Workers: {workers}")
    print(f"[INFO] CPU Cores: {multiprocessing.cpu_count()}")
    print(f"[INFO] Log Level: {log_level.upper()}")
    print(f"[INFO] Python: {sys.version}")
    print(f"[INFO] Process ID: {os.getpid()}")
    
    # Production server configuration
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        
        # Production optimizations
        access_log=True,
        use_colors=False,
        reload=False,
        
        # Server limits
        limit_concurrency=1000,
        limit_max_requests=10000,
        timeout_keep_alive=5,
        
        # Headers
        server_header=True,
        date_header=True,
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Failed to start server: {e}")
        sys.exit(1)