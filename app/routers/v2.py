"""
v2新接口路由
支持multipart/form-data、多文件、健康检查等新功能
"""
import os
import time
import zipfile
import tempfile
import cv2
import numpy as np
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from enum import Enum
import inspect
import os

from ..engine import get_engine_manager
from ..settings import settings
from ..logging import get_logger

logger = get_logger("app.routes.v2")

router = APIRouter(prefix="/api/v2")


class ModelName(str, Enum):
    """支持的模型名称"""
    PPOCRV5 = "PP-OCRv5"
    PPOCRV4 = "PP-OCRv4"
    CH_PPOCR_SERVER_V2 = "ch_ppocr_server_v2.0"


class OutputFormat(str, Enum):
    """输出格式"""
    JSON = "json"
    TEXT = "text"
    TSV = "tsv"
    HOCR = "hocr"


class OCRResultItem(BaseModel):
    """OCR结果项"""
    text: str
    confidence: float
    bounding_box: Optional[List[List[float]]] = None


class OCRResponse(BaseModel):
    """OCR响应 - 单文件"""
    processing_time: float
    results: List[OCRResultItem]
    preview_image: Optional[str] = None


class MultiFileOCRResponse(BaseModel):
    """OCR响应 - 多文件"""
    processing_time: float
    items: List[Dict[str, Any]]
    zip_url: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: float = time.time()


class TaskStatus(str, Enum):
    """任务状态"""
    QUEUED = "queued"
    RUNNING = "running" 
    DONE = "done"
    ERROR = "error"


class TaskResponse(BaseModel):
    """任务响应"""
    status: TaskStatus
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# 内存任务存储 (简单实现)
task_store = {}


def generate_timestamp() -> str:
    """生成时间戳"""
    return time.strftime("%Y%m%d_%H%M%S")


def process_image_to_results(img: np.ndarray, bbox: bool = True) -> List[OCRResultItem]:
    """处理单张图像，返回结果列表"""
    # 这里会在实际调用中由OCR引擎处理
    return []


def results_to_text(results: List[OCRResultItem]) -> str:
    """将OCR结果转换为文本格式"""
    return "\n".join([item.text for item in results])


def results_to_tsv(results: List[OCRResultItem]) -> str:
    """将OCR结果转换为TSV格式"""
    lines = ["text\tconfidence\tbbox"]
    for item in results:
        bbox_str = str(item.bounding_box) if item.bounding_box else ""
        lines.append("{}\t{}\t{}".format(item.text, item.confidence, bbox_str))
    return "\n".join(lines)


def results_to_hocr(results: List[OCRResultItem]) -> str:
    """将OCR结果转换为hOCR格式 (简化版)"""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"',
             '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
             '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">',
             '<head><title></title></head><body>']
    
    for i, item in enumerate(results):
        if item.bounding_box and len(item.bounding_box) >= 4:
            x1, y1 = item.bounding_box[0]
            x2, y2 = item.bounding_box[2]
            bbox = "bbox {} {} {} {}".format(int(x1), int(y1), int(x2), int(y2))
            lines.append('<span class="ocrx_word" title="{}; x_wconf {}">{}</span>'.format(
                bbox, int(item.confidence * 100), item.text))
        else:
            lines.append('<span class="ocrx_word">{}</span>'.format(item.text))
    
    lines.extend(['</body></html>'])
    return "\n".join(lines)


@router.post("/ocr")
async def ocr_v2(
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
    model_name: ModelName = ModelName.PPOCRV5,
    conf_threshold: float = Form(0.5),
    output_format: OutputFormat = OutputFormat.JSON,
    bbox: bool = Form(True),
    return_image: bool = Form(False)
):
    """
    OCR v2接口 - 支持multipart/form-data
    """
    try:
        # 处理文件输入
        file_list = []
        if files:
            # 兼容可能为异步生成器或非列表类型
            try:
                if inspect.isasyncgen(files):
                    async for f in files:  # type: ignore[arg-type]
                        file_list.append(f)
                elif isinstance(files, (list, tuple)):
                    file_list.extend(files)
                else:
                    file_list.append(files)  # type: ignore[arg-type]
            except TypeError:
                file_list.append(files)  # type: ignore[arg-type]
        if file:
            file_list.append(file)
        
        if not file_list:
            raise HTTPException(
                status_code=400,
                detail={"error": "No files provided", "code": "VALIDATION_ERROR"}
            )
        
        # 检查文件大小
        total_size = 0
        for f in file_list:
            try:
                cur = f.file.tell()
                f.file.seek(0, os.SEEK_END)
                total_size += f.file.tell()
                f.file.seek(cur, os.SEEK_SET)
            except Exception:
                probe = await f.read()
                total_size += len(probe)
                f.file.seek(0)
        
        if total_size > settings.MAX_CONTENT_LENGTH:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "Total file size exceeds {}MB limit".format(settings.MAX_UPLOAD_MB),
                    "code": "FILE_TOO_LARGE"
                }
            )
        
        engine = get_engine_manager()
        start_time = time.time()
        
        # 处理单文件情况
        if len(file_list) == 1:
            upload_file = file_list[0]
            
            # 读取文件内容
            content = await upload_file.read()
            
            # 检查文件类型
            if not upload_file.content_type or not upload_file.content_type.startswith('image/'):
                if not upload_file.filename or not upload_file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.pdf')):
                    raise HTTPException(
                        status_code=415,
                        detail={"error": "Unsupported file type", "code": "UNSUPPORTED_MEDIA_TYPE"}
                    )
            
            # 解码图像
            try:
                if upload_file.filename and upload_file.filename.lower().endswith('.pdf'):
                    # TODO: 实现PDF处理
                    raise HTTPException(
                        status_code=415,
                        detail={"error": "PDF processing not implemented yet", "code": "UNSUPPORTED_MEDIA_TYPE"}
                    )
                else:
                    # 处理图像文件
                    image_np = np.frombuffer(content, dtype=np.uint8)
                    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                    if img is None:
                        raise HTTPException(
                            status_code=400,
                            detail={"error": "Failed to decode image", "code": "VALIDATION_ERROR"}
                        )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "Image processing failed: {}".format(str(e)), "code": "VALIDATION_ERROR"}
                )
            
            # 执行OCR
            processing_time, result = await engine.run_ocr(
                img, 
                model_name=model_name.value, 
                conf_threshold=conf_threshold
            )
            
            # 格式化结果
            ocr_results = []
            if result and result[0]:
                for line in result[0]:
                    if isinstance(line[0], (list, np.ndarray)):
                        bounding_box = np.array(line[0]).reshape(4, 2).tolist() if bbox else None
                    else:
                        bounding_box = None
                    
                    ocr_results.append(OCRResultItem(
                        text=line[1][0],
                        confidence=float(line[1][1]),
                        bounding_box=bounding_box
                    ))
            
            # 根据输出格式返回
            if output_format == OutputFormat.JSON:
                return OCRResponse(
                    processing_time=processing_time,
                    results=ocr_results,
                    preview_image=None  # TODO: 实现return_image功能
                )
            elif output_format == OutputFormat.TEXT:
                return {"text": results_to_text(ocr_results), "processing_time": processing_time}
            elif output_format == OutputFormat.TSV:
                return {"tsv": results_to_tsv(ocr_results), "processing_time": processing_time}
            elif output_format == OutputFormat.HOCR:
                return {"hocr": results_to_hocr(ocr_results), "processing_time": processing_time}
        
        else:
            # 多文件处理
            timestamp = generate_timestamp()
            session_dir = os.path.join(settings.RESULTS_DIR, timestamp)
            os.makedirs(session_dir, exist_ok=True)
            
            results = []
            
            for upload_file in file_list:
                try:
                    content = await upload_file.read()
                    image_np = np.frombuffer(content, dtype=np.uint8)
                    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        logger.warning(f"Failed to decode image: {upload_file.filename}")
                        continue
                    
                    # 执行OCR
                    _, result = await engine.run_ocr(
                        img,
                        model_name=model_name.value,
                        conf_threshold=conf_threshold
                    )
                    
                    # 格式化结果
                    ocr_results = []
                    if result and result[0]:
                        for line in result[0]:
                            if isinstance(line[0], (list, np.ndarray)):
                                bounding_box = np.array(line[0]).reshape(4, 2).tolist() if bbox else None
                            else:
                                bounding_box = None
                            
                            ocr_results.append(OCRResultItem(
                                text=line[1][0],
                                confidence=float(line[1][1]),
                                bounding_box=bounding_box
                            ))
                    
                    if output_format == OutputFormat.TEXT:
                        text_content = results_to_text(ocr_results)
                        results.append({"filename": upload_file.filename, "text": text_content})
                        
                        # 保存文本文件
                        txt_filename = "{}.txt".format(os.path.splitext(upload_file.filename)[0])
                        txt_path = os.path.join(session_dir, txt_filename)
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(text_content)
                    else:
                        results.append({"filename": upload_file.filename, "results": [r.dict() for r in ocr_results]})
                
                except Exception as e:
                    logger.error("Failed to process file {}: {}".format(upload_file.filename, e))
                    results.append({"filename": upload_file.filename, "error": str(e)})
            
            total_processing_time = time.time() - start_time
            
            # 创建ZIP文件
            zip_url = None
            if output_format == OutputFormat.TEXT and results:
                zip_path = os.path.join(session_dir, "ocr_txt_{}.zip".format(timestamp))
                with zipfile.ZipFile(zip_path, "w") as zipf:
                    for txt_file in os.listdir(session_dir):
                        if txt_file.endswith(".txt"):
                            zipf.write(
                                os.path.join(session_dir, txt_file),
                                txt_file
                            )
                zip_url = "/download/{}".format(timestamp)
            
            return MultiFileOCRResponse(
                processing_time=total_processing_time,
                items=results,
                zip_url=zip_url
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("OCR v2 service error: {}".format(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "An error occurred: {}".format(str(e)), "code": "INTERNAL_ERROR"}
        )


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """获取任务状态"""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail={"error": "Task not found", "code": "NOT_FOUND"})
    
    return task_store[task_id]


@router.get("/healthz")
async def health_check():
    """健康检查"""
    return HealthResponse(status="ok")


@router.get("/readyz")
async def readiness_check():
    """就绪检查"""
    engine = get_engine_manager()
    if not engine.ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "not ready", "message": "Models not loaded"}
        )
    
    return HealthResponse(status="ready")
