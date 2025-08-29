"""
Web UI路由
复用现有templates和static，提供Web界面
"""
import os
from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List

from .settings import settings
from .logging import get_logger

logger = get_logger("app.routes.ui")

# 模板配置
templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)

# 路由器
router = APIRouter()

# 模型选项
MODEL_OPTIONS = ["PP-OCRv5", "PP-OCRv4", "ch_ppocr_server_v2.0"]


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """首页 - WebUI"""
    return templates.TemplateResponse(
        "webui.html", 
        {
            "request": request, 
            "model_options": MODEL_OPTIONS
        }
    )


@router.get("/download/{timestamp}")
async def download_zip(timestamp: str):
    """下载ZIP文件"""
    session_dir = os.path.join(settings.RESULTS_DIR, timestamp)
    zip_path = os.path.join(session_dir, "ocr_txt_{}.zip".format(timestamp))
    
    if not os.path.exists(zip_path):
        raise HTTPException(
            status_code=404, 
            detail={"error": "File not found", "code": "NOT_FOUND"}
        )
    
    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename="ocr_txt_{}.zip".format(timestamp)
    )