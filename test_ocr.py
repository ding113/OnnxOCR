"""
使用方法:
    python test_ocr.py              # 运行所有测试
    python test_ocr.py --api        # 仅API测试
    python test_ocr.py --performance # 仅性能测试
    python test_ocr.py --batch      # 仅批量测试
"""

import asyncio
import base64
import json
import time
import argparse
import statistics
from pathlib import Path
from typing import List, Dict, Any

import cv2
import httpx
import numpy as np
from onnxocr.api import ModernONNXOCR

# 测试配置
TEST_CONFIG = {
    "base_url": "http://localhost:5005",
    "timeout": 30.0,
    "test_images_dir": Path("onnxocr/test_images"),
    "performance_iterations": 5,
    "batch_size": 3,
}

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title: str):
    """打印测试标题"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
    print(f"[TEST] {title}")
    print(f"{'='*60}{Colors.END}")

def print_success(message: str):
    """打印成功消息"""
    print(f"{Colors.GREEN}[OK] {message}{Colors.END}")

def print_error(message: str):
    """打印错误消息"""
    print(f"{Colors.RED}[ERROR] {message}{Colors.END}")

def print_info(message: str):
    """打印信息消息"""
    print(f"{Colors.BLUE}[INFO] {message}{Colors.END}")

def print_warning(message: str):
    """打印警告消息"""
    print(f"{Colors.YELLOW}[WARN] {message}{Colors.END}")

def encode_image_to_base64(image_path: Path) -> str:
    """将图像编码为Base64"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        _, buffer = cv2.imencode('.jpg', img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    except Exception as e:
        print_error(f"图像编码失败 {image_path}: {e}")
        return ""

def get_test_images() -> List[Path]:
    """获取测试图像列表"""
    if not TEST_CONFIG["test_images_dir"].exists():
        print_warning(f"测试图像目录不存在: {TEST_CONFIG['test_images_dir']}")
        return []
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    images = []
    for ext in image_extensions:
        images.extend(TEST_CONFIG["test_images_dir"].glob(f"*{ext}"))
        images.extend(TEST_CONFIG["test_images_dir"].glob(f"*{ext.upper()}"))
    
    return sorted(images)[:10]  # 限制测试图像数量

class OCRTester:
    """OCR测试类"""
    
    def __init__(self):
        self.client = None
        self.test_images = get_test_images()
        self.results = {
            "api_tests": {},
            "performance_tests": {},
            "batch_tests": {},
            "error_tests": {}
        }
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.client = httpx.AsyncClient(
            base_url=TEST_CONFIG["base_url"],
            timeout=TEST_CONFIG["timeout"]
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.client:
            await self.client.aclose()
    
    async def check_service_health(self) -> bool:
        """检查服务健康状态"""
        try:
            response = await self.client.get("/health")
            if response.status_code == 200:
                data = response.json()
                print_success(f"服务状态: {data.get('status', 'unknown')}")
                print_info(f"模型状态: {'已加载' if data.get('model_loaded') else '未加载'}")
                return True
            else:
                print_error(f"健康检查失败: HTTP {response.status_code}")
                return False
        except Exception as e:
            print_error(f"无法连接到服务: {e}")
            return False
    
    async def test_service_info(self) -> bool:
        """测试服务信息端点"""
        try:
            response = await self.client.get("/info")
            if response.status_code == 200:
                data = response.json()
                print_success("服务信息获取成功")
                print_info(f"服务: {data.get('service', 'Unknown')}")
                print_info(f"版本: {data.get('version', 'Unknown')}")
                print_info(f"框架: {data.get('framework', 'Unknown')}")
                return True
            else:
                print_error(f"服务信息获取失败: HTTP {response.status_code}")
                return False
        except Exception as e:
            print_error(f"服务信息测试失败: {e}")
            return False
    
    async def test_single_ocr(self, image_path: Path) -> Dict[str, Any]:
        """测试单个图像OCR"""
        print_info(f"测试图像: {image_path.name}")
        
        # 编码图像
        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            return {"success": False, "error": "图像编码失败"}
        
        # 准备请求数据
        request_data = {
            "image": image_base64,
            "use_angle_cls": True,
            "drop_score": 0.5
        }
        
        try:
            start_time = time.time()
            response = await self.client.post("/ocr", json=request_data)
            end_time = time.time()
            
            request_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                results_count = len(data.get("results", []))
                processing_time = data.get("processing_time", 0)
                
                print_success(f"OCR成功: 识别到 {results_count} 个文本区域")
                print_info(f"处理时间: {processing_time:.3f}s")
                print_info(f"请求时间: {request_time:.3f}s")
                
                # 显示识别结果
                for i, result in enumerate(data.get("results", [])[:3]):  # 只显示前3个
                    text = result.get("text", "").strip()
                    confidence = result.get("confidence", 0)
                    print_info(f"  {i+1}. {text} (置信度: {confidence:.3f})")
                
                return {
                    "success": True,
                    "results_count": results_count,
                    "processing_time": processing_time,
                    "request_time": request_time,
                    "data": data
                }
            else:
                print_error(f"OCR请求失败: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print_error(f"错误详情: {error_data}")
                except:
                    print_error(f"响应内容: {response.text}")
                
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": response.text
                }
        
        except Exception as e:
            print_error(f"OCR测试异常: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_batch_ocr(self, image_paths: List[Path]) -> Dict[str, Any]:
        """测试批量OCR处理"""
        print_info(f"批量测试: {len(image_paths)} 个图像")
        
        try:
            # 准备文件数据
            files = []
            for image_path in image_paths:
                with open(image_path, 'rb') as f:
                    files.append(('files', (image_path.name, f.read(), 'image/jpeg')))
            
            # 表单数据
            form_data = {
                'use_angle_cls': True,
                'drop_score': 0.5
            }
            
            start_time = time.time()
            response = await self.client.post(
                "/ocr/batch",
                files=files,
                data=form_data
            )
            end_time = time.time()
            
            request_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                total_files = data.get("total_files", 0)
                processed_files = data.get("processed_files", 0)
                total_processing_time = data.get("total_processing_time", 0)
                
                print_success(f"批量处理完成: {processed_files}/{total_files} 个文件")
                print_info(f"总处理时间: {total_processing_time:.3f}s")
                print_info(f"平均处理时间: {total_processing_time/max(processed_files, 1):.3f}s/文件")
                print_info(f"请求时间: {request_time:.3f}s")
                
                return {
                    "success": True,
                    "total_files": total_files,
                    "processed_files": processed_files,
                    "total_processing_time": total_processing_time,
                    "request_time": request_time,
                    "average_time": total_processing_time / max(processed_files, 1)
                }
            else:
                print_error(f"批量处理失败: HTTP {response.status_code}")
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": response.text
                }
        
        except Exception as e:
            print_error(f"批量测试异常: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_error_handling(self):
        """测试错误处理"""
        print_header("错误处理测试")
        
        tests = [
            {
                "name": "空请求",
                "data": {},
                "expected_status": 422
            },
            {
                "name": "无效Base64",
                "data": {"image": "invalid_base64", "use_angle_cls": True, "drop_score": 0.5},
                "expected_status": 400
            },
            {
                "name": "无效置信度",
                "data": {"image": "validbase64==", "use_angle_cls": True, "drop_score": 1.5},
                "expected_status": 422
            }
        ]
        
        for test in tests:
            try:
                response = await self.client.post("/ocr", json=test["data"])
                if response.status_code == test["expected_status"]:
                    print_success(f"{test['name']}: 错误处理正确 (HTTP {response.status_code})")
                else:
                    print_warning(f"{test['name']}: 预期 {test['expected_status']}, 得到 {response.status_code}")
            except Exception as e:
                print_error(f"{test['name']}: 测试异常 {e}")
    
    async def run_api_tests(self):
        """运行API测试"""
        print_header("API功能测试")
        
        # 健康检查
        if not await self.check_service_health():
            print_error("服务不健康，跳过API测试")
            return False
        
        # 服务信息测试
        await self.test_service_info()
        
        # OCR功能测试
        if self.test_images:
            success_count = 0
            total_tests = min(3, len(self.test_images))  # 测试前3个图像
            
            for i, image_path in enumerate(self.test_images[:total_tests]):
                print(f"\n[TEST] 测试 {i+1}/{total_tests}")
                result = await self.test_single_ocr(image_path)
                if result["success"]:
                    success_count += 1
                
                self.results["api_tests"][image_path.name] = result
            
            print(f"\n[STATS] API测试总结: {success_count}/{total_tests} 成功")
            return success_count == total_tests
        else:
            print_warning("没有找到测试图像")
            return False
    
    async def run_performance_tests(self):
        """运行性能测试"""
        print_header("性能基准测试")
        
        if not self.test_images:
            print_warning("没有测试图像，跳过性能测试")
            return
        
        test_image = self.test_images[0]
        iterations = TEST_CONFIG["performance_iterations"]
        
        print_info(f"性能测试: {test_image.name} x {iterations} 次")
        
        times = []
        for i in range(iterations):
            print_info(f"迭代 {i+1}/{iterations}")
            result = await self.test_single_ocr(test_image)
            if result["success"]:
                times.append(result["processing_time"])
            else:
                print_warning(f"迭代 {i+1} 失败")
        
        if times:
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            
            print_success("性能测试完成")
            print_info(f"平均处理时间: {avg_time:.3f}s")
            print_info(f"最快处理时间: {min_time:.3f}s")
            print_info(f"最慢处理时间: {max_time:.3f}s")
            print_info(f"标准差: {std_time:.3f}s")
            print_info(f"预估吞吐量: {1/avg_time:.1f} 图像/秒")
            
            self.results["performance_tests"] = {
                "average_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "std_time": std_time,
                "throughput": 1/avg_time
            }
    
    async def run_batch_tests(self):
        """运行批量测试"""
        print_header("批量处理测试")
        
        if len(self.test_images) < 2:
            print_warning("测试图像不足，跳过批量测试")
            return
        
        batch_size = min(TEST_CONFIG["batch_size"], len(self.test_images))
        batch_images = self.test_images[:batch_size]
        
        result = await self.test_batch_ocr(batch_images)
        self.results["batch_tests"] = result
    
    def generate_report(self):
        """生成测试报告"""
        print_header("测试报告")
        
        # API测试报告
        api_tests = self.results["api_tests"]
        if api_tests:
            successful_apis = sum(1 for r in api_tests.values() if r.get("success"))
            total_apis = len(api_tests)
            print_info(f"API测试: {successful_apis}/{total_apis} 成功")
        
        # 性能测试报告
        perf_tests = self.results["performance_tests"]
        if perf_tests:
            print_info(f"性能测试: 平均 {perf_tests['average_time']:.3f}s, 吞吐量 {perf_tests['throughput']:.1f} 图像/秒")
        
        # 批量测试报告
        batch_tests = self.results["batch_tests"]
        if batch_tests.get("success"):
            print_info(f"批量测试: {batch_tests['processed_files']}/{batch_tests['total_files']} 文件处理成功")
        
        # 生成JSON报告
        report_path = Path("test_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print_success(f"详细报告已保存: {report_path}")

async def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description="ONNX OCR 测试套件")
    parser.add_argument("--api", action="store_true", help="仅运行API测试")
    parser.add_argument("--performance", action="store_true", help="仅运行性能测试")
    parser.add_argument("--batch", action="store_true", help="仅运行批量测试")
    parser.add_argument("--error", action="store_true", help="仅运行错误处理测试")
    
    args = parser.parse_args()
    
    print_header("ONNX OCR Service 测试套件")
    print_info(f"测试服务地址: {TEST_CONFIG['base_url']}")
    
    async with OCRTester() as tester:
        # 根据参数决定运行哪些测试
        if args.api:
            await tester.run_api_tests()
        elif args.performance:
            await tester.run_performance_tests()
        elif args.batch:
            await tester.run_batch_tests()
        elif args.error:
            await tester.test_error_handling()
        else:
            # 运行所有测试
            await tester.run_api_tests()
            await tester.run_performance_tests()
            await tester.run_batch_tests()
            await tester.test_error_handling()
        
        # 生成报告
        tester.generate_report()

if __name__ == "__main__":
    try:
        asyncio.run(main())
        print_success("测试完成！")
    except KeyboardInterrupt:
        print_warning("测试被用户中断")
    except Exception as e:
        print_error(f"测试执行失败: {e}")
        import traceback
        traceback.print_exc()