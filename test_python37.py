#!/usr/bin/env python3.7
"""
Python 3.7兼容性测试脚本
测试FastAPI OCR服务是否能在Python 3.7下正常运行
"""
import sys
import os

# 检查Python版本
if sys.version_info < (3, 7) or sys.version_info >= (3, 8):
    print(f"警告：当前Python版本 {sys.version}，建议使用Python 3.7")

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有模块导入"""
    try:
        print("测试模块导入...")
        
        # 测试FastAPI应用导入
        from app.main import app
        print("✅ FastAPI应用导入成功")
        
        # 测试配置导入
        from app.settings import settings
        print("✅ 配置模块导入成功")
        
        # 测试引擎导入
        from app.engine import get_engine_manager
        print("✅ 引擎模块导入成功")
        
        # 测试路由导入
        from app.routers import v1, v2
        print("✅ 路由模块导入成功")
        
        # 测试UI模块导入
        from app import ui
        print("✅ UI模块导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    try:
        print("测试基本功能...")
        
        # 测试引擎管理器
        from app.engine import get_engine_manager
        engine = get_engine_manager()
        print("✅ 引擎管理器创建成功")
        
        # 测试配置读取
        from app.settings import settings
        print(f"✅ 配置读取成功: 默认模型={settings.DEFAULT_MODEL}")
        
        return True
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

def test_server_startup():
    """测试服务器启动（不实际启动）"""
    try:
        print("测试服务器启动配置...")
        
        # 导入FastAPI应用
        from app.main import app
        
        # 检查路由
        routes = [route.path for route in app.routes]
        print("✅ 注册的路由:")
        for route in routes:
            print(f"   - {route}")
        
        return True
    except Exception as e:
        print(f"❌ 服务器启动测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=== Python 3.7 兼容性测试 ===")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print()
    
    success_count = 0
    total_tests = 3
    
    # 运行测试
    if test_imports():
        success_count += 1
    
    if test_basic_functionality():
        success_count += 1
        
    if test_server_startup():
        success_count += 1
    
    print()
    print(f"=== 测试结果: {success_count}/{total_tests} 通过 ===")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！代码与Python 3.7兼容")
        sys.exit(0)
    else:
        print("⚠️  部分测试失败，需要修复兼容性问题")
        sys.exit(1)