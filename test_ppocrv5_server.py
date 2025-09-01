#!/usr/bin/env python3
"""
PPOCRv5-Server 适配功能测试脚本
验证模型下载、切换和回退机制
"""

import asyncio
import os
import sys
import time
import tempfile
import requests
import cv2
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.model_downloader import ModelDownloadManager
from app.engine import EngineManager
from app.settings import settings


async def test_model_downloader():
    """测试模型下载管理器"""
    print("=== 测试模型下载管理器 ===")
    
    downloader = ModelDownloadManager()
    
    # 测试目录结构检查
    print(f"Server模型目录: {downloader.server_model_dir}")
    print(f"Mobile模型目录: {downloader.mobile_model_dir}")
    
    # 检查当前server模型状态
    is_complete = downloader.is_server_model_complete()
    print(f"Server模型完整性: {is_complete}")
    
    if not is_complete:
        print("开始准备PPOCRv5-Server模型...")
        success = await downloader.ensure_model_available("PP-OCRv5-Server")
        print(f"模型准备结果: {success}")
        
        if success:
            print("验证模型文件...")
            is_complete_after = downloader.is_server_model_complete()
            print(f"准备后模型完整性: {is_complete_after}")
        else:
            print("模型准备失败，将在运行时回退到mobile版本")
    else:
        print("Server模型已可用")
    
    return True


def test_engine_manager():
    """测试引擎管理器"""
    print("\n=== 测试引擎管理器 ===")
    
    # 创建引擎管理器实例
    engine = EngineManager(default_model="PP-OCRv5-Server")
    
    print(f"默认模型: {engine.default_model}")
    print(f"并发限制: {engine.concurrency}")
    
    # 预热测试
    print("开始预热...")
    engine.warmup()
    print(f"预热结果: {engine.ready}")
    
    return engine.ready


async def test_ocr_functionality():
    """测试OCR功能"""
    print("\n=== 测试OCR功能 ===")
    
    # 创建测试图像
    test_img = np.zeros((200, 400, 3), dtype=np.uint8)
    test_img.fill(255)  # 白色背景
    
    # 添加一些文本（使用OpenCV绘制）
    cv2.putText(test_img, "Test OCR", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    engine = EngineManager()
    
    # 测试Server版本
    print("测试PP-OCRv5-Server...")
    try:
        processing_time, result = await engine.run_ocr(test_img, model_name="PP-OCRv5-Server")
        print(f"Server版本处理时间: {processing_time:.3f}s")
        print(f"Server版本识别结果数量: {len(result[0]) if result and result[0] else 0}")
    except Exception as e:
        print(f"Server版本测试失败: {e}")
    
    # 测试Mobile版本对比
    print("测试PP-OCRv5 (mobile)...")
    try:
        processing_time, result = await engine.run_ocr(test_img, model_name="PP-OCRv5")
        print(f"Mobile版本处理时间: {processing_time:.3f}s")
        print(f"Mobile版本识别结果数量: {len(result[0]) if result and result[0] else 0}")
    except Exception as e:
        print(f"Mobile版本测试失败: {e}")


def test_api_endpoints():
    """测试API接口"""
    print("\n=== 测试API接口 ===")
    
    base_url = f"http://localhost:{settings.PORT}"
    
    # 测试健康检查
    try:
        response = requests.get(f"{base_url}/api/v2/healthz", timeout=5)
        print(f"健康检查状态: {response.status_code}")
        if response.status_code == 200:
            print(f"健康检查响应: {response.json()}")
    except Exception as e:
        print(f"健康检查失败: {e}")
    
    # 测试就绪检查
    try:
        response = requests.get(f"{base_url}/api/v2/readyz", timeout=5)
        print(f"就绪检查状态: {response.status_code}")
        if response.status_code == 200:
            print(f"就绪检查响应: {response.json()}")
        elif response.status_code == 503:
            print("服务未就绪，模型可能还在加载中")
    except Exception as e:
        print(f"就绪检查失败: {e}")


def test_configuration():
    """测试配置"""
    print("\n=== 测试配置 ===")
    
    print(f"默认模型: {settings.DEFAULT_MODEL}")
    print(f"模型池大小: {settings.MODEL_POOL_SIZE}")
    print(f"模型并发数: {settings.MODEL_CONCURRENCY}")
    print(f"是否使用GPU: {settings.USE_GPU}")
    print(f"是否预热: {settings.WARMUP}")
    print(f"最大上传大小: {settings.MAX_UPLOAD_MB}MB")
    
    # 验证配置的合理性
    if settings.DEFAULT_MODEL != "PP-OCRv5-Server":
        print("⚠️  警告: 默认模型不是PP-OCRv5-Server，可能影响精确度优先原则")
    else:
        print("✅ 默认模型配置正确")


async def main():
    """主测试函数"""
    print("PPOCRv5-Server 适配功能测试")
    print("=" * 50)
    
    # 测试配置
    test_configuration()
    
    # 测试模型下载管理器
    try:
        await test_model_downloader()
    except Exception as e:
        print(f"模型下载测试失败: {e}")
    
    # 测试引擎管理器
    try:
        engine_ready = test_engine_manager()
        if not engine_ready:
            print("⚠️  引擎未就绪，可能影响后续测试")
    except Exception as e:
        print(f"引擎管理器测试失败: {e}")
    
    # 测试OCR功能
    try:
        await test_ocr_functionality()
    except Exception as e:
        print(f"OCR功能测试失败: {e}")
    
    # 测试API接口（需要服务运行）
    try:
        test_api_endpoints()
    except Exception as e:
        print(f"API测试失败（可能服务未运行）: {e}")
    
    print("\n" + "=" * 50)
    print("测试完成")


if __name__ == "__main__":
    asyncio.run(main())