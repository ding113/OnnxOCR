#!/usr/bin/env python3
"""
测试PP-OCRv5-Server模型下载与加载流程
"""
import sys
import os
import asyncio
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

async def test_model_download_and_loading():
    """测试模型下载和加载流程"""
    print("Testing PP-OCRv5-Server model download and loading...")
    print("=" * 60)
    
    try:
        # 导入必要的模块
        from app.engine import get_engine_manager
        from app.model_downloader import get_model_downloader
        
        # 获取下载管理器和引擎管理器
        downloader = get_model_downloader()
        engine = get_engine_manager()
        
        print("Step 1: Checking current model file status...")
        is_complete_before = downloader.is_server_model_complete()
        print(f"PP-OCRv5-Server files complete before: {'✅' if is_complete_before else '❌'}")
        
        print("\nStep 2: Testing model acquisition with download...")
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 尝试获取PP-OCRv5-Server模型（应该触发下载或fallback）
            model = await engine.get_model("PP-OCRv5-Server")
            end_time = asyncio.get_event_loop().time()
            
            print(f"✅ Model acquisition completed in {end_time - start_time:.2f}s")
            print(f"Final model type: {type(model).__name__}")
            
            # 检查最终的文件状态
            is_complete_after = downloader.is_server_model_complete()
            print(f"PP-OCRv5-Server files complete after: {'✅' if is_complete_after else '❌'}")
            
            print("\nStep 3: Testing OCR functionality...")
            # 创建测试图像
            import numpy as np
            test_img = np.ones((64, 64, 3), dtype=np.uint8) * 255  # 白色图像
            
            # 测试OCR功能
            processing_time, results = await engine.run_ocr(test_img, "PP-OCRv5-Server")
            print(f"✅ OCR test completed in {processing_time:.3f}s")
            print(f"OCR results: {len(results[0]) if results and results[0] else 0} text regions detected")
            
        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            print(f"❌ Model acquisition failed after {end_time - start_time:.2f}s: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you are running this from the project root directory")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_model_warmup():
    """测试预热流程"""
    print("\nTesting model warmup process...")
    print("=" * 40)
    
    try:
        from app.engine import get_engine_manager
        
        engine = get_engine_manager()
        print("Starting warmup...")
        
        start_time = asyncio.get_event_loop().time()
        engine.warmup()
        end_time = asyncio.get_event_loop().time()
        
        print(f"Warmup completed in {end_time - start_time:.2f}s")
        print(f"Engine ready: {'✅' if engine.ready else '❌'}")
        
        return engine.ready
        
    except Exception as e:
        print(f"❌ Warmup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    async def main():
        print("PP-OCRv5-Server Model Download & Loading Test")
        print("=" * 50)
        
        # 测试模型下载和加载
        download_success = await test_model_download_and_loading()
        
        # 测试预热流程
        warmup_success = await test_model_warmup()
        
        print("\n" + "=" * 50)
        print("FINAL RESULTS:")
        print(f"Model Download & Loading: {'✅' if download_success else '❌'}")
        print(f"Model Warmup:            {'✅' if warmup_success else '❌'}")
        
        if download_success and warmup_success:
            print("\n🎉 All tests passed! The model download and loading mechanism is working correctly.")
            sys.exit(0)
        else:
            print("\n💥 Some tests failed. Please check the logs above for details.")
            sys.exit(1)
    
    asyncio.run(main())