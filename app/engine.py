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
from .model_downloader import get_model_downloader
from .cache import get_cache_manager

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
        
        # 模型下载管理器
        self._downloader = get_model_downloader()
        
        # 缓存管理器
        self._cache_manager = get_cache_manager()
        
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
                "cache_enabled": settings.CACHE_ENABLED,
            }
        )
    
    def _get_model_kwargs(self, model_name: str) -> dict:
        """根据模型名称获取初始化参数"""
        kwargs = {
            "use_angle_cls": True,
            "use_gpu": settings.USE_GPU,
        }
        
        # 根据不同模型设置特定参数（确保传入的是具体的ONNX文件路径）
        # 默认（不覆盖路径）时，ONNXPaddleOcr 会使用 utils.infer_args() 中的 v5 路径
        if model_name == "PP-OCRv5":
            kwargs.update({
                "det_model_dir": "onnxocr/models/ppocrv5/det/det.onnx",
                "rec_model_dir": "onnxocr/models/ppocrv5/rec/rec.onnx",
                "cls_model_dir": "onnxocr/models/ppocrv5/cls/cls.onnx",
                "rec_char_dict_path": "onnxocr/models/ppocrv5/ppocrv5_dict.txt",
            })
        elif model_name == "PP-OCRv5-Server":
            kwargs.update({
                "det_model_dir": "onnxocr/models/ppocrv5-server/det/det.onnx",
                "rec_model_dir": "onnxocr/models/ppocrv5-server/rec/rec.onnx",
                "cls_model_dir": "onnxocr/models/ppocrv5-server/cls/cls.onnx",
                "rec_char_dict_path": "onnxocr/models/ppocrv5-server/ppocrv5_dict.txt",
            })
        elif model_name == "PP-OCRv4":
            kwargs.update({
                "det_model_dir": "onnxocr/models/ppocrv4/det/det.onnx",
                "rec_model_dir": "onnxocr/models/ppocrv4/rec/rec.onnx",
                "cls_model_dir": "onnxocr/models/ppocrv4/cls/cls.onnx",
            })
        elif model_name == "ch_ppocr_server_v2.0":
            # 注意：仓库中仅包含 det/cls 与字典，若缺少 rec 模型请按需补全
            kwargs.update({
                "det_model_dir": "onnxocr/models/ch_ppocr_server_v2.0/det/det.onnx",
                "cls_model_dir": "onnxocr/models/ch_ppocr_server_v2.0/cls/cls.onnx",
                "rec_char_dict_path": "onnxocr/models/ch_ppocr_server_v2.0/ppocr_keys_v1.txt",
            })
        
        return kwargs
    
    async def get_model(self, model_name: Optional[str] = None) -> ONNXPaddleOcr:
        """获取模型实例，自动下载缺失的模型文件并支持重试机制"""
        original_model_name = model_name or self.default_model
        model_name = original_model_name
        
        # 如果模型已缓存，直接返回
        with self._lock:
            if model_name in self._models:
                return self._models[model_name]
        
        # PP-OCRv5-Server模型需要特殊处理，支持下载和重试
        if original_model_name == "PP-OCRv5-Server":
            max_retries = 2  # 最多重试2次
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    logger.info(f"Attempting to prepare PP-OCRv5-Server model (attempt {retry_count + 1}/{max_retries + 1})")
                    
                    # 尝试确保模型文件可用（包括下载）
                    model_available = await self._downloader.ensure_model_available(model_name)
                    
                    if model_available:
                        logger.info("PP-OCRv5-Server model files are now available")
                        break
                    else:
                        retry_count += 1
                        if retry_count <= max_retries:
                            wait_time = retry_count * 10  # 递增等待时间：10s, 20s
                            logger.warning(f"PP-OCRv5-Server preparation failed, retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.warning("PP-OCRv5-Server preparation failed after all retries, falling back to PP-OCRv5")
                            model_name = "PP-OCRv5"
                            
                except Exception as e:
                    logger.error(f"Error preparing PP-OCRv5-Server model (attempt {retry_count + 1}): {e}")
                    retry_count += 1
                    if retry_count <= max_retries:
                        wait_time = retry_count * 10
                        logger.warning(f"Retrying PP-OCRv5-Server preparation in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning("PP-OCRv5-Server preparation failed after all retries, falling back to PP-OCRv5")
                        model_name = "PP-OCRv5"
        else:
            # 其他模型直接检查可用性（不需要下载）
            try:
                await self._downloader.ensure_model_available(model_name)
            except Exception as e:
                logger.warning(f"Model availability check failed for {model_name}: {e}")
        
        # 获取或创建模型实例
        with self._lock:
            if model_name not in self._models:
                logger.info("Loading model: {}".format(model_name))
                try:
                    kwargs = self._get_model_kwargs(model_name)
                    self._models[model_name] = ONNXPaddleOcr(**kwargs)
                    logger.info("Model loaded successfully: {}".format(model_name))
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    # 如果是server版本加载失败，最后一次尝试加载mobile版本
                    if model_name == "PP-OCRv5-Server":
                        logger.warning("Final fallback: loading PP-OCRv5 mobile version")
                        model_name = "PP-OCRv5"
                        if model_name not in self._models:
                            kwargs = self._get_model_kwargs(model_name)
                            self._models[model_name] = ONNXPaddleOcr(**kwargs)
                            logger.info("Fallback model loaded: {}".format(model_name))
                    else:
                        raise
            
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
            # 确保模型在主线程中已经准备好
            model_name = model_name or self.default_model
            
            # 如果请求的模型尚未加载，先在主线程中加载
            if model_name not in self._models:
                logger.info(f"Model {model_name} not in cache, loading asynchronously...")
                await self.get_model(model_name)
            
            # 现在在线程池中执行同步OCR操作
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
        """同步OCR执行（在线程池中运行，模型已预先加载）"""
        model_name = model_name or self.default_model
        
        # 模型应该已经在缓存中，直接获取
        with self._lock:
            if model_name not in self._models:
                # 这种情况不应该发生，因为run_ocr已经预先加载了模型
                # 但为了安全起见，使用同步fallback
                logger.warning(f"Model {model_name} not found in cache, using sync fallback")
                if model_name == "PP-OCRv5-Server":
                    logger.warning("Using PP-OCRv5 as sync fallback")
                    model_name = "PP-OCRv5"
                
                # 同步加载fallback模型
                if model_name not in self._models:
                    kwargs = self._get_model_kwargs(model_name)
                    self._models[model_name] = ONNXPaddleOcr(**kwargs)
                    logger.info(f"Sync fallback model loaded: {model_name}")
            
            model = self._models[model_name]
        
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
    
    async def run_ocr_with_cache(
        self,
        img: np.ndarray,
        image_bytes: bytes,
        model_name: Optional[str] = None,
        conf_threshold: Optional[float] = None
    ) -> Tuple[float, List[List]]:
        """执行OCR识别（带缓存支持）"""
        model_name = model_name or self.default_model
        conf_threshold = conf_threshold or 0.5
        
        # 尝试从缓存获取结果
        if settings.CACHE_ENABLED:
            cached_result = self._cache_manager.get_cached_result(
                image_bytes, model_name, conf_threshold
            )
            if cached_result:
                logger.debug(f"Cache hit for model {model_name}")
                return cached_result
        
        # 缓存未命中，执行OCR
        async with self._semaphore:
            # 确保模型在主线程中已经准备好
            model_name = model_name or self.default_model
            
            # 如果请求的模型尚未加载，先在主线程中加载
            with self._lock:
                if model_name not in self._models:
                    logger.info(f"Model {model_name} not in cache, loading asynchronously...")
            
            # 确保模型已加载
            if model_name not in self._models:
                await self.get_model(model_name)
            
            # 现在在线程池中执行OCR（置信度=0获取全部结果用于缓存）
            loop = asyncio.get_event_loop()
            processing_time, full_results = await loop.run_in_executor(
                None, 
                self._sync_ocr,
                img,
                model_name,
                0.0  # 缓存所有结果
            )
            
            # 异步写入缓存
            if settings.CACHE_ENABLED and full_results:
                self._cache_manager.cache_result_async(
                    image_bytes, model_name, full_results, processing_time
                )
            
            # 按用户要求的置信度过滤结果
            if conf_threshold > 0.0 and full_results and full_results[0]:
                filtered_result = []
                for line in full_results[0]:
                    if len(line) >= 2 and len(line[1]) >= 2:
                        confidence = float(line[1][1])
                        if confidence >= conf_threshold:
                            filtered_result.append(line)
                return processing_time, [filtered_result]
            
            return processing_time, full_results
    
    async def _async_get_model(self, model_name: Optional[str] = None) -> ONNXPaddleOcr:
        """异步获取模型的辅助方法"""
        return await self.get_model(model_name)
    
    def warmup(self):
        """预热模型"""
        if not settings.WARMUP:
            return
        
        try:
            logger.info("Starting model warmup")
            # 创建一个小的测试图像
            test_img = np.zeros((64, 64, 3), dtype=np.uint8)
            
            # 对于PP-OCRv5-Server，先检查是否已完整，避免多进程同时下载
            target_model = self.default_model
            if target_model == "PP-OCRv5-Server":
                if not self._downloader.is_server_model_complete():
                    logger.info("PP-OCRv5-Server model not complete, warmup will use PP-OCRv5 instead")
                    target_model = "PP-OCRv5"
                else:
                    logger.info("PP-OCRv5-Server model already complete")
            
            # 使用同步方式获取模型进行预热
            try:
                model = self._get_model_sync(target_model)
                model.ocr(test_img)
            except Exception as e:
                logger.warning(f"Warmup failed with {target_model}, trying PP-OCRv5: {e}")
                # 如果失败，回退到PP-OCRv5
                model = self._get_model_sync("PP-OCRv5")
                model.ocr(test_img)
            
            self._ready = True
            logger.info("Model warmup completed")
        except Exception as e:
            logger.error("Model warmup failed: {}".format(e))
            self._ready = False
    
    async def _async_warmup_model(self, test_img: np.ndarray):
        """异步预热模型，支持模型下载等待"""
        model = await self.get_model()
        model.ocr(test_img)
        return model
    
    def _get_model_sync(self, model_name: str) -> ONNXPaddleOcr:
        """同步获取模型实例（用于预热fallback）"""
        model_name = model_name or self.default_model
        
        # 对于预热fallback，如果server版本准备失败，直接回退到mobile版本
        if model_name == "PP-OCRv5-Server":
            try:
                # 同步检查server模型是否可用（不触发下载）
                from .model_downloader import get_model_downloader
                downloader = get_model_downloader()
                if not downloader.is_server_model_complete():
                    logger.warning("PP-OCRv5-Server model files not complete, falling back to PP-OCRv5 for warmup")
                    model_name = "PP-OCRv5"
            except Exception as e:
                logger.error(f"Failed to check server model availability: {e}")
                logger.warning("Falling back to PP-OCRv5 mobile version for warmup")
                model_name = "PP-OCRv5"
        
        # 获取或创建模型实例
        with self._lock:
            if model_name not in self._models:
                logger.info("Loading model for warmup: {}".format(model_name))
                try:
                    kwargs = self._get_model_kwargs(model_name)
                    self._models[model_name] = ONNXPaddleOcr(**kwargs)
                    logger.info("Model loaded successfully for warmup: {}".format(model_name))
                except Exception as e:
                    logger.error(f"Failed to load model {model_name} for warmup: {e}")
                    # 如果是server版本加载失败，尝试加载mobile版本
                    if model_name == "PP-OCRv5-Server":
                        logger.warning("Failed to load PP-OCRv5-Server for warmup, falling back to PP-OCRv5")
                        model_name = "PP-OCRv5"
                        if model_name not in self._models:
                            kwargs = self._get_model_kwargs(model_name)
                            self._models[model_name] = ONNXPaddleOcr(**kwargs)
                            logger.info("Fallback model loaded for warmup: {}".format(model_name))
                    else:
                        raise
            
            return self._models[model_name]
    
    @property
    def ready(self) -> bool:
        """检查是否已就绪"""
        return self._ready


# 全局引擎管理器实例
engine_manager = EngineManager()


def get_engine_manager() -> EngineManager:
    """获取引擎管理器实例"""
    return engine_manager
