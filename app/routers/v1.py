"""
v1兼容路由
保持与原Flask app-service.py 100%兼容
"""
import base64
import cv2
import numpy as np
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..engine import get_engine_manager
from ..logging import get_logger

logger = get_logger("app.routes.v1")

router = APIRouter()


class OCRRequest(BaseModel):
    """OCR请求模型"""
    image: str  # base64编码的图像


class BoundingBox(BaseModel):
    """边界框坐标"""
    pass  # 使用List[List[float]]作为类型


class OCRResult(BaseModel):
    """OCR结果项"""
    text: str
    confidence: float
    bounding_box: List[List[float]]


class OCRResponse(BaseModel):
    """OCR响应模型"""
    processing_time: float
    results: List[OCRResult]


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str


@router.post("/ocr")
async def ocr_service(request: OCRRequest):
    """
    OCR服务接口 - v1兼容版本
    完全兼容原Flask接口的输入输出格式
    """
    try:
        # 验证请求数据
        if not request.image:
            raise HTTPException(
                status_code=400,
                detail={"error": "Invalid request, 'image' field is required."}
            )
        
        # 解码base64图像
        try:
            image_bytes = base64.b64decode(request.image)
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "Failed to decode image from base64."}
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={"error": "Image decoding failed: {}".format(str(e))}
            )
        
        # 获取引擎管理器
        engine = get_engine_manager()
        
        # 执行OCR
        processing_time, result = await engine.run_ocr(img)
        
        # 格式化结果 - 保持与原版完全一致的格式
        ocr_results = []
        if result and result[0]:
            for line in result[0]:
                # 确保line[0]是NumPy数组或列表
                if isinstance(line[0], (list, np.ndarray)):
                    # 将bounding_box转换为[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]格式
                    bounding_box = np.array(line[0]).reshape(4, 2).tolist()
                else:
                    bounding_box = []
                
                ocr_results.append({
                    "text": line[1][0],  # 识别文本
                    "confidence": float(line[1][1]),  # 置信度
                    "bounding_box": bounding_box  # 文本框坐标
                })
        
        # 返回结果 - 保持与原版完全一致的格式
        return {
            "processing_time": processing_time,
            "results": ocr_results
        }
        
    except Exception as e:
        # 捕获所有异常并返回错误信息 - 保持与原版一致的格式
        logger.error("OCR service error: {}".format(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "An error occurred: {}".format(str(e))}
        )