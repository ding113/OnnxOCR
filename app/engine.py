"""
推理引擎封装
封装现有ONNXPaddleOcr，提供并发控制和模型管理
"""
import asyncio
import time
import threading
from typing import Dict, Optional, List, Tuple, Any
import numpy as np

from onnxocr.onnx_paddleocr import ONNXPaddleOcr
from onnxocr.ocr_images_pdfs import OCRLogic
from .settings import settings
from .logging import get_logger

logger = get_logger("app.engine")


class EngineManager:
    """推理引擎管理器"""
    
    def __init__(
        self,
        pool_size: int = None,
        concurrency: int = None,
        default_model: str = None
    ):
        self.pool_size = pool_size or settings.MODEL_POOL_SIZE
        self.concurrency = concurrency or settings.MODEL_CONCURRENCY
        self.default_model = default_model or settings.DEFAULT_MODEL
        
        # 模型实例缓存
        self._models: Dict[str, ONNXPaddleOcr] = {}
        self._ocr_logic: Optional[OCRLogic] = None
        
        # 并发控制
        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._lock = threading.Lock()
        
        # 就绪状态
        self._ready = False
        
        logger.info(
            "EngineManager initialized",
            extra={
                "pool_size": self.pool_size,
                "concurrency": self.concurrency,
                "default_model": self.default_model,
                "use_gpu": settings.USE_GPU,
            }
        )
    
    def _get_model_kwargs(self, model_name: str) -> dict:
        """根据模型名称获取初始化参数"""
        kwargs = {
            "use_angle_cls": True,
            "use_gpu": settings.USE_GPU,
        }
        
        # 根据不同模型设置特定参数 - 使用相对路径
        if model_name == "PP-OCRv5":
            kwargs.update({
                "det_model_dir": "onnxocr/models/ppocrv5/det",
                "rec_model_dir": "onnxocr/models/ppocrv5/rec", 
                "cls_model_dir": "onnxocr/models/ppocrv5/cls",
                "rec_char_dict_path": "onnxocr/models/ppocrv5/ppocrv5_dict.txt",
            })
        elif model_name == "PP-OCRv4":
            kwargs.update({
                "det_model_dir": "onnxocr/models/ppocrv4/det",
                "rec_model_dir": "onnxocr/models/ppocrv4/rec",
                "cls_model_dir": "onnxocr/models/ppocrv4/cls",
            })
        elif model_name == "ch_ppocr_server_v2.0":
            kwargs.update({
                "det_model_dir": "onnxocr/models/ch_ppocr_server_v2.0/det",
                "rec_model_dir": "onnxocr/models/ch_ppocr_server_v2.0/rec",
                "cls_model_dir": "onnxocr/models/ch_ppocr_server_v2.0/cls",
                "rec_char_dict_path": "onnxocr/models/ch_ppocr_server_v2.0/ppocr_keys_v1.txt",
            })
        
        return kwargs
    
    def get_model(self, model_name: Optional[str] = None) -> ONNXPaddleOcr:
        """获取模型实例"""
        model_name = model_name or self.default_model
        
        with self._lock:
            if model_name not in self._models:
                logger.info("Loading model: {}".format(model_name))
                kwargs = self._get_model_kwargs(model_name)
                self._models[model_name] = ONNXPaddleOcr(**kwargs)
                logger.info("Model loaded: {}".format(model_name))
            
            return self._models[model_name]
    
    def get_ocr_logic(self) -> OCRLogic:
        """获取OCRLogic实例"""
        if self._ocr_logic is None:
            self._ocr_logic = OCRLogic(lambda msg: logger.debug("OCRLogic: {}".format(msg)))
        return self._ocr_logic
    
    async def run_ocr(
        self,
        img: np.ndarray,
        model_name: Optional[str] = None,
        conf_threshold: Optional[float] = None
    ) -> Tuple[float, List[List]]:
        """执行OCR识别"""
        async with self._semaphore:
            # 在线程池中执行同步OCR操作
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                self._sync_ocr,
                img,
                model_name,
                conf_threshold
            )
    
    def _sync_ocr(
        self,
        img: np.ndarray,
        model_name: Optional[str] = None,
        conf_threshold: Optional[float] = None
    ) -> Tuple[float, List[List]]:
        """同步OCR执行"""
        model = self.get_model(model_name)
        
        start_time = time.time()
        result = model.ocr(img)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # 应用置信度阈值过滤
        if conf_threshold is not None and result and result[0]:
            filtered_result = []
            for line in result[0]:
                if len(line) >= 2 and len(line[1]) >= 2:
                    confidence = float(line[1][1])
                    if confidence >= conf_threshold:
                        filtered_result.append(line)
            result = [filtered_result]
        
        return processing_time, result
    
    def warmup(self):
        """预热模型"""
        if not settings.WARMUP:
            return
        
        try:
            logger.info("Starting model warmup")
            # 创建一个小的测试图像
            test_img = np.zeros((64, 64, 3), dtype=np.uint8)
            model = self.get_model(self.default_model)
            model.ocr(test_img)
            self._ready = True
            logger.info("Model warmup completed")
        except Exception as e:
            logger.error("Model warmup failed: {}".format(e))
            self._ready = False
    
    @property
    def ready(self) -> bool:
        """检查是否已就绪"""
        return self._ready


# 全局引擎管理器实例
engine_manager = EngineManager()


def get_engine_manager() -> EngineManager:
    """获取引擎管理器实例"""
    return engine_manager