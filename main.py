import asyncio
import base64
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

import cv2
import numpy as np
import structlog
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.exposition import make_wsgi_app

from onnxocr.api import ModernONNXOCR
from onnxocr.core import OCRRequest, OCRResponse, ModelSwitchRequest, config

# [GEAR] 配置结构化日志
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# [CHART] 性能监控指标 (增强版 - 支持模型版本分离)
REQUEST_COUNT = Counter(
    'ocr_requests_total', 
    'Total OCR requests', 
    ['method', 'endpoint', 'status', 'model_version']
)
REQUEST_DURATION = Histogram(
    'ocr_request_duration_seconds', 
    'OCR request duration',
    ['endpoint', 'model_version']
)
MODEL_LOAD_TIME = Histogram(
    'ocr_model_load_seconds', 
    'Model loading time',
    ['model_version']
)
OCR_INFERENCE_TIME = Histogram(
    'ocr_inference_duration_seconds',
    'OCR inference time',
    ['model_version']
)
MODEL_SWITCH_COUNT = Counter(
    'ocr_model_switches_total',
    'Total model switches',
    ['from_version', 'to_version']
)

# 🎯 兼容性模型定义 (用于批处理等特殊用途)
class OCRResult(BaseModel):
    """单个OCR识别结果"""
    text: str = Field(..., description="识别的文本内容")
    confidence: float = Field(..., description="置信度 (0-1)")
    bounding_box: List[List[float]] = Field(
        ..., 
        description="文本边界框坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"
    )

class BatchOCRResponse(BaseModel):
    """批量OCR响应模型"""
    success: bool = Field(..., description="整体处理是否成功")
    total_files: int = Field(..., description="总文件数")
    processed_files: int = Field(..., description="成功处理文件数")
    results: List[Dict[str, Any]] = Field(..., description="每个文件的处理结果")
    total_processing_time: float = Field(..., description="总处理时间")

# 🧠 全局模型实例
ocr_model: Optional[ModernONNXOCR] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 - 启动和关闭时的操作"""
    global ocr_model
    
    # 🚀 启动时预加载现代化OCR系统
    logger.info("正在初始化现代化ONNX OCR系统...")
    start_time = time.time()
    
    try:
        # 预加载现代化模型
        ocr_model = ModernONNXOCR(
            use_gpu=False,  # CPU专用版本
            use_angle_cls=True,
            drop_score=0.5
        )
        
        await ocr_model.initialize()
        
        load_time = time.time() - start_time
        MODEL_LOAD_TIME.labels(model_version=config.default_model_version).observe(load_time)
        
        logger.info(
            "现代化OCR系统初始化完成",
            load_time_seconds=load_time,
            default_model=config.default_model_version,
            available_models=await ocr_model.get_available_models()
        )
        
        # 🔥 模型预热 - 用虚拟图像进行一次推理
        logger.info("开始模型预热...")
        dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        await ocr_model.ocr_async(dummy_img)
        
        warmup_time = time.time() - start_time - load_time
        logger.info(
            "模型预热完成", 
            warmup_time_seconds=warmup_time
        )
        
    except Exception as e:
        logger.error("初始化失败", error=str(e))
        raise RuntimeError(f"OCR系统初始化失败: {e}") from e
    
    yield
    
    # 🧹 关闭时清理资源
    logger.info("正在清理OCR系统资源...")
    try:
        if ocr_model:
            await ocr_model.cleanup()
        logger.info("资源清理完成")
    except Exception as e:
        logger.error("清理资源时出错", error=str(e))

# [GLOBE] 创建FastAPI应用
app = FastAPI(
    title="[ROCKET] Modern ONNX OCR Service",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# [GLOBE] 跨域支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# [FILE] 静态文件服务
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================================
# [SHIELD] 文件验证和处理函数
# ============================================================================

# 支持的图片MIME类型
ALLOWED_MIME_TYPES = {
    'image/jpeg',
    'image/jpg', 
    'image/png',
    'image/webp',
    'image/bmp',
    'image/tiff'
}

# 文件头魔数检测 (防止伪造扩展名)
FILE_SIGNATURES = {
    b'\xff\xd8\xff': 'image/jpeg',
    b'\x89PNG\r\n\x1a\n': 'image/png',
    b'RIFF': 'image/webp',  # WebP files start with RIFF
    b'BM': 'image/bmp',
    b'II*\x00': 'image/tiff',
    b'MM\x00*': 'image/tiff'
}

async def validate_image_file(file: UploadFile) -> Dict[str, Any]:
    """
    现代化文件验证：MIME类型、文件头、大小检查
    
    Args:
        file: 上传的文件对象
        
    Returns:
        Dict containing file info and validation results
        
    Raises:
        HTTPException: 如果文件验证失败
    """
    # 检查MIME类型
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "不支持的文件格式",
                "message": f"文件类型 {file.content_type} 不被支持",
                "supported_types": list(ALLOWED_MIME_TYPES),
                "hint": "请上传 JPG, PNG, WebP, BMP 或 TIFF 格式的图片"
            }
        )
    
    # 读取文件头进行魔数检测
    file_content = await file.read()
    file_size = len(file_content)
    
    # 检查文件大小 (默认限制10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "文件过大",
                "message": f"文件大小 {file_size/1024/1024:.1f}MB 超过限制",
                "max_size_mb": max_size // 1024 // 1024,
                "hint": "请上传小于10MB的图片"
            }
        )
    
    # 检查最小文件大小
    if file_size < 100:  # 100 bytes
        raise HTTPException(
            status_code=400,
            detail={
                "error": "文件过小",
                "message": "文件可能已损坏或为空",
                "hint": "请上传有效的图片文件"
            }
        )
    
    # 文件头魔数验证
    detected_type = None
    for signature, mime_type in FILE_SIGNATURES.items():
        if file_content.startswith(signature):
            detected_type = mime_type
            break
        # WebP需要特殊检查
        elif signature == b'RIFF' and len(file_content) >= 12:
            if file_content[8:12] == b'WEBP':
                detected_type = 'image/webp'
                break
    
    if not detected_type:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "无效的图片文件",
                "message": "文件头检测失败，可能是损坏或伪造的文件",
                "hint": "请上传有效的图片文件"
            }
        )
    
    # MIME类型和文件头一致性检查
    if detected_type != file.content_type and not (
        # JPEG的特殊情况：image/jpg 和 image/jpeg 都是有效的
        (detected_type == 'image/jpeg' and file.content_type in ['image/jpeg', 'image/jpg'])
    ):
        logger.warning(
            "文件类型不匹配", 
            declared_type=file.content_type,
            detected_type=detected_type,
            filename=file.filename
        )
    
    # 重置文件指针以便后续读取
    await file.seek(0)
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "detected_type": detected_type,
        "size_bytes": file_size,
        "size_kb": round(file_size / 1024, 1),
        "size_mb": round(file_size / 1024 / 1024, 2)
    }

async def process_uploaded_image(file: UploadFile) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    处理上传的图片文件，转换为OpenCV格式
    
    Args:
        file: 验证过的上传文件
        
    Returns:
        Tuple of (image_array, file_info)
        
    Raises:
        HTTPException: 如果图片处理失败
    """
    try:
        # 读取文件内容
        file_content = await file.read()
        
        # 转换为numpy数组
        image_np = np.frombuffer(file_content, dtype=np.uint8)
        
        # 解码为OpenCV图像
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "图片解码失败", 
                    "message": "无法解析图片文件，可能文件已损坏",
                    "hint": "请确认文件是有效的图片格式"
                }
            )
        
        # 检查图片尺寸
        height, width = img.shape[:2]
        channels = img.shape[2] if len(img.shape) > 2 else 1
        
        # 尺寸限制检查 (防止过大图片导致内存问题)
        max_pixels = 50 * 1024 * 1024  # 50M pixels
        if height * width > max_pixels:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "图片尺寸过大",
                    "message": f"图片尺寸 {width}×{height} 超过限制",
                    "max_pixels": f"{max_pixels:,}",
                    "hint": "请上传小于50M像素的图片"
                }
            )
        
        # 最小尺寸检查
        if height < 10 or width < 10:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "图片尺寸过小",
                    "message": f"图片尺寸 {width}×{height} 太小",
                    "min_size": "10×10",
                    "hint": "请上传尺寸大于10×10像素的图片"
                }
            )
        
        image_info = {
            "width": width,
            "height": height,
            "channels": channels,
            "format": "multipart",
            "total_pixels": width * height,
            "aspect_ratio": round(width / height, 2)
        }
        
        return img, image_info
        
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        logger.error("图片处理失败", error=str(e), filename=file.filename)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "图片处理失败",
                "message": f"处理图片时发生内部错误: {str(e)}",
                "hint": "请稍后重试或联系技术支持"
            }
        )

# ============================================================================
# [STAR] API端点定义
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """[HOME] 服务主页 - 现代化UI界面"""
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🚀 Modern ONNX OCR Service</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 40px;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            }
            h1 { font-size: 2.5em; text-align: center; margin-bottom: 10px; }
            .subtitle { text-align: center; opacity: 0.8; margin-bottom: 40px; }
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .feature {
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            .links {
                display: flex;
                justify-content: center;
                gap: 20px;
                flex-wrap: wrap;
                margin-top: 40px;
            }
            .link {
                background: rgba(255, 255, 255, 0.2);
                color: white;
                padding: 12px 24px;
                text-decoration: none;
                border-radius: 25px;
                transition: all 0.3s ease;
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            .link:hover {
                background: rgba(255, 255, 255, 0.3);
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>[ROCKET] Modern ONNX OCR Service</h1>
            <p class="subtitle">Python 3.13 + FastAPI + ONNX Runtime 1.22.1</p>
            
            <div class="features">
                <div class="feature">
                    <h3>[LIGHTNING] 高性能异步</h3>
                    <p>FastAPI + Uvicorn多进程架构</p>
                </div>
                <div class="feature">
                    <h3>[BRAIN] 智能OCR</h3>
                    <p>PP-OCR v5模型，高精度识别</p>
                </div>
                <div class="feature">
                    <h3>[CHART] 实时监控</h3>
                    <p>Prometheus指标 + 结构化日志</p>
                </div>
                <div class="feature">
                    <h3>[GEAR] 生产就绪</h3>
                    <p>自动验证 + 错误处理 + 健康检查</p>
                </div>
            </div>
            
            <div class="links">
                <a href="/webui" class="link">[WEB] 现代化Web界面</a>
                <a href="/docs" class="link">[BOOK] API文档</a>
                <a href="/redoc" class="link">[DOCS] ReDoc</a>
                <a href="/models/available" class="link">[GEAR] 可用模型</a>
                <a href="/models/info" class="link">[INFO] 模型信息</a>
                <a href="/health" class="link">[HEALTH] 健康检查</a>
                <a href="/metrics" class="link">[CHART] 性能指标</a>
                <a href="/info" class="link">[INFO] 服务信息</a>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/webui", response_class=HTMLResponse)
async def web_ui():
    """[WEB] 现代化Web界面"""
    try:
        with open("templates/webui.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
        <body style="font-family: Arial; text-align: center; padding: 50px;">
            <h1>Web UI Not Found</h1>
            <p>Template file not found. Please check if templates/webui.html exists.</p>
            <a href="/">Back to Home</a>
        </body>
        </html>
        """

@app.get("/health")
async def health_check():
    """[HEALTH] 健康检查端点"""
    return {
        "status": "healthy",
        "model_loaded": ocr_model is not None,
        "service": "Modern ONNX OCR Service",
        "version": "2.0.0",
        "python_version": "3.13",
        "timestamp": time.time()
    }

@app.get("/info")
async def service_info():
    """[INFO] 服务详细信息"""
    return {
        "service": "Modern ONNX OCR Service",
        "version": "2.0.0",
        "python_version": "3.13",
        "framework": "FastAPI",
        "inference_engine": "ONNX Runtime 1.22.1",
        "model_type": "PP-OCR v5",
        "model_ready": ocr_model is not None,
        "features": [
            "异步高性能处理",
            "CPU多核优化",
            "实时性能监控",
            "自动API文档",
            "数据验证",
            "结构化日志",
        ],
        "endpoints": {
            "ocr": "/ocr - 单图OCR识别 (支持model_version参数)",
            "batch": "/ocr/batch - 批量文件处理 (支持model_version参数)",
            "models_available": "/models/available - 获取可用模型版本",
            "models_info": "/models/info - 获取模型详细信息",
            "models_switch": "/models/switch - 切换默认模型版本",
            "docs": "/docs - 交互式API文档",
            "metrics": "/metrics - Prometheus指标",
            "health": "/health - 健康检查"
        }
    }

@app.post("/ocr", response_model=OCRResponse)
async def ocr_service(request: OCRRequest):
    """
    🎯 OCR文本识别服务
    
    ### 功能特性
    - **高精度识别**: PP-OCR v5模型，支持中英文等多语言
    - **角度校正**: 可选的文本角度分类和校正  
    - **置信度过滤**: 可配置的识别结果置信度阈值
    - **边界框输出**: 精确的文本位置坐标信息
    
    ### 参数说明
    - **image**: Base64编码的图像数据
    - **use_angle_cls**: 是否使用角度分类器进行文本方向校正
    - **drop_score**: 置信度阈值，低于此值的识别结果将被过滤
    
    ### 返回结果
    - **success**: 处理成功状态
    - **processing_time**: 处理耗时（秒）
    - **results**: OCR识别结果列表
    - **model_info**: 使用的模型信息
    - **performance_metrics**: 详细性能指标
    """
    start_time = time.time()
    endpoint = "ocr"
    
    # 验证模型是否就绪
    if not ocr_model:
        REQUEST_COUNT.labels(
            method="POST", 
            endpoint=endpoint, 
            status="error", 
            model_version="unknown"
        ).inc()
        logger.error("模型未就绪")
        raise HTTPException(
            status_code=503, 
            detail={
                "error": "模型未就绪",
                "message": "OCR模型正在加载中，请稍后重试",
                "retry_after": 30
            }
        )
    
    try:
        # [IMAGE] 解码Base64图像
        decode_start = time.time()
        try:
            image_bytes = base64.b64decode(request.image)
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("图像解码失败，请检查Base64格式")
                
            decode_time = time.time() - decode_start
            logger.debug("图像解码完成", decode_time_seconds=decode_time)
            
        except Exception as e:
            REQUEST_COUNT.labels(
                method="POST", 
                endpoint=endpoint, 
                status="error", 
                model_version=getattr(request, 'model_version', 'unknown')
            ).inc()
            logger.error("图像解码失败", error=str(e))
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "图像解码失败",
                    "message": f"无法解码Base64图像: {str(e)}",
                    "hint": "请确保提供有效的Base64编码图像数据"
                }
            )
        
        # 🧠 异步执行现代化OCR推理
        inference_start = time.time()
        
        # 使用现代化OCR系统进行推理
        ocr_result = await ocr_model.process_base64_image(
            base64_image=request.image,
            model_version=request.model_version,
            use_angle_cls=request.use_angle_cls,
            drop_score=request.drop_score
        )
        
        inference_time = time.time() - inference_start
        OCR_INFERENCE_TIME.labels(model_version=request.model_version).observe(inference_time)
        
        # [CHART] 处理和格式化结果
        format_start = time.time()
        ocr_results = []
        
        if ocr_result["success"] and ocr_result["results"]:
            for item in ocr_result["results"]:
                ocr_results.append(OCRResult(
                    text=item["text"],
                    confidence=item["confidence"],
                    bounding_box=item["box"]
                ))
        
        format_time = time.time() - format_start
        processing_time = time.time() - start_time
        
        # 📈 记录成功指标
        REQUEST_COUNT.labels(
            method="POST", 
            endpoint=endpoint, 
            status="success", 
            model_version=request.model_version
        ).inc()
        REQUEST_DURATION.labels(
            endpoint=endpoint, 
            model_version=request.model_version
        ).observe(processing_time)
        
        logger.info(
            "OCR处理完成",
            processing_time_seconds=processing_time,
            results_count=len(ocr_results),
            decode_time_seconds=decode_time,
            inference_time_seconds=inference_time,
            format_time_seconds=format_time,
        )
        
        return OCRResponse(
            success=True,
            processing_time=processing_time,
            results=ocr_results,
            model_version=request.model_version,  # 添加缺失的字段
            image_info={  # 添加缺失的字段 - 使用正确的变量名
                "width": int(img.shape[1]) if img is not None and len(img.shape) >= 2 else 0,
                "height": int(img.shape[0]) if img is not None and len(img.shape) >= 1 else 0,
                "channels": int(img.shape[2]) if img is not None and len(img.shape) > 2 else 1,
                "format": "base64",
                "size_kb": len(request.image) // 1024
            },
            model_info={
                "model_version": request.model_version,
                "model_type": "PP-OCR",
                "use_angle_cls": request.use_angle_cls,
                "drop_score": request.drop_score,
                "inference_engine": "ONNX Runtime 1.22.1",
                "performance_metrics": ocr_result.get("performance_metrics", {})
            },
            performance_metrics={
                "decode_time": decode_time,
                "inference_time": inference_time,
                "format_time": format_time,
                "total_time": processing_time
            }
        )
        
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        # 处理未预期的异常
        REQUEST_COUNT.labels(
            method="POST", 
            endpoint=endpoint, 
            status="error", 
            model_version=getattr(request, 'model_version', 'unknown')
        ).inc()
        logger.error("OCR处理异常", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "OCR处理失败",
                "message": f"服务器内部错误: {str(e)}",
                "processing_time": time.time() - start_time
            }
        )

@app.post("/ocr/batch", response_model=BatchOCRResponse)
async def batch_ocr_service(
    files: List[UploadFile] = File(..., description="要处理的图像文件列表"),
    model_version: str = Form("v5", description="模型版本"),
    use_angle_cls: bool = Form(True, description="是否使用角度分类器"),
    drop_score: float = Form(0.5, description="置信度阈值"),
):
    """
    📦 批量OCR处理服务
    
    ### 功能特性
    - **并发处理**: 多文件异步并发处理，提高整体效率
    - **错误隔离**: 单个文件失败不影响其他文件处理
    - **进度跟踪**: 返回详细的处理进度和结果统计
    - **格式支持**: 支持JPG、PNG、BMP等常见图像格式
    
    ### 使用场景
    - 批量文档扫描识别
    - 多页PDF文档处理
    - 大量图片文本提取
    
    ### 性能优化
    - 异步并发处理，充分利用多核CPU
    - 智能错误处理，保证服务稳定性
    - 内存优化，支持大批量文件处理
    """
    start_time = time.time()
    endpoint = "batch_ocr"
    
    if not ocr_model:
        REQUEST_COUNT.labels(
            method="POST", 
            endpoint=endpoint, 
            status="error", 
            model_version=model_version
        ).inc()
        raise HTTPException(
            status_code=503, 
            detail="模型未就绪，请稍后重试"
        )
    
    if not files:
        REQUEST_COUNT.labels(
            method="POST", 
            endpoint=endpoint, 
            status="error", 
            model_version=model_version
        ).inc()
        raise HTTPException(
            status_code=400,
            detail="请提供至少一个图像文件"
        )
    
    logger.info(f"开始批量OCR处理", file_count=len(files))
    
    results = []
    processed_count = 0
    
    # [LOOP] 异步并发处理每个文件
    async def process_single_file(file: UploadFile) -> Dict[str, Any]:
        """处理单个文件的异步函数"""
        try:
            # 读取文件内容
            file_start_time = time.time()
            image_bytes = await file.read()
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            
            if img is None:
                return {
                    "filename": file.filename,
                    "success": False,
                    "error": "图像解码失败",
                    "processing_time": 0
                }
            
            # OCR处理 - 使用现代化API
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            ocr_result = await ocr_model.process_base64_image(
                base64_image=image_b64,
                model_version=model_version,
                use_angle_cls=use_angle_cls,
                drop_score=drop_score
            )
            
            file_processing_time = time.time() - file_start_time
            
            # 格式化结果
            ocr_results = []
            if ocr_result["success"] and ocr_result["results"]:
                for item in ocr_result["results"]:
                    ocr_results.append({
                        "text": item["text"],
                        "confidence": item["confidence"],
                        "bounding_box": item["box"]
                    })
            
            return {
                "filename": file.filename,
                "success": True,
                "processing_time": file_processing_time,
                "results_count": len(ocr_results),
                "results": ocr_results
            }
            
        except Exception as e:
            logger.error(f"文件处理失败", filename=file.filename, error=str(e))
            return {
                "filename": file.filename,
                "success": False,
                "error": str(e),
                "processing_time": time.time() - file_start_time if 'file_start_time' in locals() else 0
            }
    
    # 🚀 并发处理所有文件
    try:
        # 使用asyncio.gather实现并发处理
        results = await asyncio.gather(
            *[process_single_file(file) for file in files],
            return_exceptions=False
        )
        
        processed_count = sum(1 for r in results if r.get("success", False))
        
        total_processing_time = time.time() - start_time
        
        REQUEST_COUNT.labels(
            method="POST", 
            endpoint=endpoint, 
            status="success", 
            model_version=model_version
        ).inc()
        REQUEST_DURATION.labels(
            endpoint=endpoint, 
            model_version=model_version
        ).observe(total_processing_time)
        
        logger.info(
            "批量OCR处理完成",
            total_files=len(files),
            processed_files=processed_count,
            failed_files=len(files) - processed_count,
            total_processing_time_seconds=total_processing_time
        )
        
        return BatchOCRResponse(
            success=True,
            total_files=len(files),
            processed_files=processed_count,
            results=results,
            total_processing_time=total_processing_time
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(
            method="POST", 
            endpoint=endpoint, 
            status="error", 
            model_version=model_version
        ).inc()
        logger.error("批量OCR处理异常", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"批量处理失败: {str(e)}"
        )

# ============================================================================
# [ROCKET] V2 现代化文件上传接口 (推荐)
# ============================================================================

@app.post("/v2/ocr")
async def ocr_v2_service(
    file: UploadFile = File(..., description="图片文件"),
    model_version: str = Form("v5-server", description="模型版本"),
    det: bool = Form(True, description="是否启用文字检测"), 
    rec: bool = Form(True, description="是否启用文字识别"),
    cls: bool = Form(True, description="是否启用角度分类"),
    drop_score: float = Form(0.5, description="置信度阈值")
):
    """
    🚀 V2现代化OCR接口 - 推荐使用
    
    ### 优势特性
    - **内存友好**: multipart/form-data，无需base64编码
    - **高效传输**: 比base64减少33%数据量
    - **流式处理**: 大文件友好，避免内存溢出
    - **严格验证**: 文件头魔数检测，防止恶意文件
    - **NGINX兼容**: 完美支持标准nginx配置
    
    ### 文件要求
    - **格式**: JPG, PNG, WebP, BMP, TIFF
    - **大小**: 最大10MB，最小100字节
    - **尺寸**: 10×10像素到50M像素
    - **检测**: 自动文件头验证和MIME类型检查
    
    ### 返回格式
    与v1接口完全兼容，包含额外的文件信息
    """
    endpoint = "/v2/ocr"
    start_time = time.time()
    
    if not ocr_model:
        raise HTTPException(
            status_code=503,
            detail="OCR系统未就绪，请稍后重试"
        )
    
    try:
        # [SHIELD] 文件验证和处理
        validation_start = time.time()
        
        # 验证文件
        file_info = await validate_image_file(file)
        logger.debug("文件验证完成", **file_info)
        
        # 处理图片
        img, image_info = await process_uploaded_image(file)
        
        validation_time = time.time() - validation_start
        logger.debug("文件处理完成", validation_time_seconds=validation_time)
        
        # [BRAIN] 现代化OCR推理
        inference_start = time.time()
        
        ocr_result = await ocr_model.ocr_async(
            image=img,
            model_version=model_version,
            det=det,
            rec=rec,
            cls=cls,
            drop_score=drop_score
        )
        
        inference_time = time.time() - inference_start
        OCR_INFERENCE_TIME.labels(model_version=model_version).observe(inference_time)
        
        # [CHART] 格式化结果
        format_start = time.time()
        ocr_results = []
        
        if ocr_result and len(ocr_result) > 0:
            for item in ocr_result:
                if len(item) >= 2 and item[1]:
                    # 处理识别结果
                    text_info = item[1]
                    text = text_info[0] if isinstance(text_info, list) else str(text_info)
                    confidence = text_info[1] if isinstance(text_info, list) and len(text_info) > 1 else 0.0
                    
                    # 处理边界框
                    bbox = item[0] if item[0] is not None else []
                    
                    ocr_results.append(OCRResult(
                        text=text,
                        confidence=confidence,
                        bounding_box=bbox
                    ))
        
        format_time = time.time() - format_start
        processing_time = time.time() - start_time
        
        # [CHART] 记录成功指标
        REQUEST_COUNT.labels(
            method="POST",
            endpoint=endpoint,
            status="success", 
            model_version=model_version
        ).inc()
        REQUEST_DURATION.labels(
            endpoint=endpoint,
            model_version=model_version
        ).observe(processing_time)
        
        # 合并文件信息
        enhanced_image_info = {**file_info, **image_info}
        
        logger.info(
            "V2 OCR处理完成",
            filename=file.filename,
            model_version=model_version,
            processing_time_seconds=processing_time,
            results_count=len(ocr_results)
        )
        
        return OCRResponse(
            success=True,
            processing_time=processing_time,
            results=ocr_results,
            model_version=model_version,
            image_info=enhanced_image_info,
            model_info={
                "model_version": model_version,
                "model_type": "PP-OCR",
                "use_angle_cls": cls,
                "drop_score": drop_score,
                "inference_engine": "ONNX Runtime 1.22.1",
                "api_version": "v2"
            },
            performance_metrics={
                "validation_time": validation_time,
                "inference_time": inference_time,
                "format_time": format_time,
                "total_time": processing_time
            }
        )
        
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        # 处理未预期的异常
        REQUEST_COUNT.labels(
            method="POST",
            endpoint=endpoint,
            status="error",
            model_version=model_version
        ).inc()
        
        logger.error("V2 OCR处理异常", error=str(e), filename=file.filename)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "OCR处理失败",
                "message": f"处理过程中发生错误: {str(e)}",
                "api_version": "v2",
                "hint": "请检查图片格式和内容，或稍后重试"
            }
        )

@app.post("/v2/ocr/batch")
async def batch_ocr_v2_service(
    files: List[UploadFile] = File(..., description="图片文件列表"),
    model_version: str = Form("v5-server", description="模型版本"),
    det: bool = Form(True, description="是否启用文字检测"),
    rec: bool = Form(True, description="是否启用文字识别"), 
    cls: bool = Form(True, description="是否启用角度分类"),
    drop_score: float = Form(0.5, description="置信度阈值")
):
    """
    🚀 V2批量OCR接口 - 现代化多文件处理
    
    ### 批量处理特性
    - **并发处理**: 多文件异步处理，提升效率
    - **独立验证**: 每个文件独立验证和处理
    - **错误隔离**: 单个文件失败不影响其他文件
    - **详细反馈**: 每个文件的处理状态和错误信息
    
    ### 限制说明
    - **文件数量**: 最多20个文件
    - **单文件大小**: 最大10MB
    - **总大小**: 建议不超过50MB
    """
    endpoint = "/v2/ocr/batch"
    start_time = time.time()
    
    if not ocr_model:
        raise HTTPException(
            status_code=503,
            detail="OCR系统未就绪，请稍后重试"
        )
    
    # 检查文件数量限制
    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "文件数量超过限制",
                "message": f"上传了{len(files)}个文件，最多支持20个",
                "max_files": 20,
                "hint": "请分批上传文件"
            }
        )
    
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "没有文件",
                "message": "请至少上传一个图片文件",
                "hint": "检查文件选择器是否正确"
            }
        )
    
    try:
        results = []
        processed_count = 0
        total_files = len(files)
        
        logger.info(f"开始批量V2 OCR处理", total_files=total_files, model_version=model_version)
        
        # 并发处理所有文件
        for index, file in enumerate(files):
            file_start_time = time.time()
            
            try:
                # 验证和处理文件
                file_info = await validate_image_file(file)
                img, image_info = await process_uploaded_image(file)
                
                # OCR推理
                ocr_result = await ocr_model.ocr_async(
                    image=img,
                    model_version=model_version,
                    det=det,
                    rec=rec,
                    cls=cls,
                    drop_score=drop_score
                )
                
                # 格式化结果
                ocr_results = []
                if ocr_result and len(ocr_result) > 0:
                    for item in ocr_result:
                        if len(item) >= 2 and item[1]:
                            text_info = item[1]
                            text = text_info[0] if isinstance(text_info, list) else str(text_info)
                            confidence = text_info[1] if isinstance(text_info, list) and len(text_info) > 1 else 0.0
                            bbox = item[0] if item[0] is not None else []
                            
                            ocr_results.append(OCRResult(
                                text=text,
                                confidence=confidence,
                                bounding_box=bbox
                            ))
                
                file_processing_time = time.time() - file_start_time
                enhanced_image_info = {**file_info, **image_info}
                
                results.append({
                    "file_index": index,
                    "filename": file.filename,
                    "success": True,
                    "results": ocr_results,
                    "processing_time": file_processing_time,
                    "image_info": enhanced_image_info,
                    "results_count": len(ocr_results)
                })
                
                processed_count += 1
                logger.debug(f"文件处理完成", filename=file.filename, index=index, results_count=len(ocr_results))
                
            except Exception as file_error:
                file_processing_time = time.time() - file_start_time
                
                results.append({
                    "file_index": index,
                    "filename": file.filename,
                    "success": False,
                    "error": str(file_error),
                    "processing_time": file_processing_time,
                    "results_count": 0
                })
                
                logger.warning(f"文件处理失败", filename=file.filename, index=index, error=str(file_error))
        
        total_processing_time = time.time() - start_time
        
        # 记录成功指标
        REQUEST_COUNT.labels(
            method="POST",
            endpoint=endpoint,
            status="success",
            model_version=model_version
        ).inc()
        REQUEST_DURATION.labels(
            endpoint=endpoint,
            model_version=model_version
        ).observe(total_processing_time)
        
        logger.info(
            "批量V2 OCR处理完成",
            total_files=total_files,
            processed_files=processed_count,
            failed_files=total_files - processed_count,
            total_processing_time_seconds=total_processing_time
        )
        
        return BatchOCRResponse(
            success=True,
            total_files=total_files,
            processed_files=processed_count,
            results=results,
            total_processing_time=total_processing_time
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(
            method="POST",
            endpoint=endpoint,
            status="error",
            model_version=model_version
        ).inc()
        
        logger.error("批量V2 OCR处理异常", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "批量处理失败",
                "message": f"批量处理过程中发生错误: {str(e)}",
                "api_version": "v2",
                "hint": "请检查所有文件格式，或稍后重试"
            }
        )

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """[CHART] Prometheus监控指标端点"""
    return generate_latest()

@app.get("/models/available")
async def get_available_models():
    """[BOOKS] 获取可用的模型版本列表"""
    if not ocr_model:
        raise HTTPException(
            status_code=503,
            detail="OCR系统未就绪"
        )
    
    try:
        available_models = await ocr_model.get_available_models()
        return {
            "success": True,
            "available_models": available_models,
            "default_model": config.default_model_version
        }
    except Exception as e:
        logger.error("获取可用模型失败", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"获取模型信息失败: {str(e)}"
        )

@app.get("/models/info")
async def get_model_info(model_version: Optional[str] = None):
    """[INFO] 获取模型详细信息"""
    if not ocr_model:
        raise HTTPException(
            status_code=503,
            detail="OCR系统未就绪"
        )
    
    try:
        model_info = await ocr_model.get_model_info(model_version)
        return {
            "success": True,
            **model_info
        }
    except Exception as e:
        logger.error("获取模型信息失败", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"获取模型信息失败: {str(e)}"
        )

@app.post("/models/switch")
async def switch_model(request: ModelSwitchRequest):
    """[LOOP] 切换默认模型版本"""
    if not ocr_model:
        raise HTTPException(
            status_code=503,
            detail="OCR系统未就绪"
        )
    
    try:
        result = await ocr_model.switch_model(request.model_version)
        
        # 记录模型切换指标
        if result["success"]:
            MODEL_SWITCH_COUNT.labels(
                from_version=result["previous_version"],
                to_version=result["current_version"]
            ).inc()
        
        return result
    except Exception as e:
        logger.error("模型切换失败", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"模型切换失败: {str(e)}"
        )

# ============================================================================
# 🚀 应用启动配置
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import multiprocessing
    
    # Development environment configuration
    print("[INFO] Starting development server...")
    print("[DOCS] API Documentation: http://localhost:5005/docs")
    print("[METRICS] Performance Metrics: http://localhost:5005/metrics")
    print("[HEALTH] Health Check: http://localhost:5005/health")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5005,
        reload=False,  # Disable reload to avoid Prometheus metrics conflicts
        workers=1,  # Single worker for testing
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        },
    )