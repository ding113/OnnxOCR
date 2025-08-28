"""
Modern ONNX OCR API - main entry point with model switching support
"""

import asyncio
from typing import Dict, List, Any, Optional
import numpy as np
import structlog

from .core import config, SmartModelManager
from .core.exceptions import OCRError, ModelNotFoundError, InferenceError
from .models.async_system import AsyncOCREngine
from .utils.image_ops import base64_to_cv2, validate_image

logger = structlog.get_logger()


class ModernONNXOCR:
    """
    Modern ONNX OCR class with features:
    - Dynamic model switching via API parameters
    - Fully async processing pipeline
    - Intelligent model management and caching
    - Comprehensive performance monitoring
    - Memory optimization
    """
    
    def __init__(self, **kwargs):
        """
        Initialize modern OCR system
        
        Args:
            **kwargs: Configuration overrides
        """
        # Update config with any provided overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Initialize components
        self.model_manager = SmartModelManager()
        self.ocr_engine = AsyncOCREngine(self.model_manager)
        
        # State tracking
        self._initialized = False
        self._default_model_version = config.default_model_version
        
        self.logger = logger.bind(component="ModernONNXOCR")
    
    async def initialize(self) -> None:
        """
        Initialize the OCR system and preload default model
        """
        if self._initialized:
            return
        
        self.logger.info("Initializing Modern ONNX OCR system")
        
        try:
            # Ensure directories exist
            config.ensure_directories()
            
            # Preload default model version
            await self.model_manager.preload_model(self._default_model_version)
            
            self._initialized = True
            
            self.logger.info(
                "Modern ONNX OCR system initialized successfully",
                default_model=self._default_model_version
            )
            
        except Exception as e:
            self.logger.error("Failed to initialize OCR system", error=str(e))
            raise OCRError(f"Initialization failed: {e}") from e
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    async def ocr_async(
        self,
        image: np.ndarray,
        model_version: str = "v5",
        det: bool = True,
        rec: bool = True,
        cls: bool = True,
        drop_score: float = 0.5,
        det_limit_side_len: float = 960,
        **kwargs
    ) -> List[List[Any]]:
        """
        Perform OCR with model version selection
        
        Args:
            image: Input image as numpy array
            model_version: Model version ('v4', 'v5', 'v5-server')
            det: Whether to perform text detection
            rec: Whether to perform text recognition
            cls: Whether to perform angle classification
            drop_score: Confidence threshold
            det_limit_side_len: Detection side length limit
            **kwargs: Additional arguments
            
        Returns:
            OCR results in PaddleOCR compatible format
        """
        if not self._initialized:
            await self.initialize()
        
        if not validate_image(image):
            raise InferenceError("Invalid input image")
        
        self.logger.debug(
            "Starting OCR processing",
            model_version=model_version,
            det=det,
            rec=rec,
            cls=cls,
            image_shape=image.shape
        )
        
        try:
            # Process based on requested operations
            if det and rec:
                # Full OCR pipeline
                result = await self.ocr_engine.process_ocr(
                    image=image,
                    model_version=model_version,
                    use_angle_cls=cls,
                    drop_score=drop_score,
                    det_limit_side_len=det_limit_side_len
                )
                
                if result["success"]:
                    # Convert to PaddleOCR format: [[[box], (text, confidence)]]
                    ocr_results = []
                    for item in result["results"]:
                        box = item["box"]
                        text = item["text"]
                        confidence = item["confidence"]
                        ocr_results.append([box, (text, confidence)])
                    
                    return [ocr_results]  # Wrap in list for compatibility
                else:
                    return [[]]  # Empty result
            
            elif det and not rec:
                # Detection only
                sessions = await self.model_manager.get_model_sessions(model_version)
                detector = sessions["det"]
                
                dt_boxes, metadata = await detector.detect_text(image)
                if dt_boxes is not None:
                    return [[box.tolist() for box in dt_boxes]]
                else:
                    return [[]]
            
            elif not det and rec:
                # Recognition only (image must be pre-cropped text)
                sessions = await self.model_manager.get_model_sessions(model_version)
                recognizer = sessions["rec"]
                classifier = sessions["cls"]
                
                img_list = [image]
                
                # Apply angle classification if enabled
                if cls:
                    img_list, cls_results = await classifier.classify_angles(img_list)
                
                # Recognize text
                rec_results = await recognizer.recognize_text(img_list)
                
                return [rec_results]  # Return in expected format
            
            else:
                return [[]]  # No operations requested
                
        except Exception as e:
            self.logger.error(
                "OCR processing failed",
                model_version=model_version,
                error=str(e)
            )
            raise InferenceError(f"OCR processing failed: {e}") from e
    
    def ocr(self, image: np.ndarray, **kwargs) -> List[List[Any]]:
        """
        Synchronous OCR wrapper (for backward compatibility)
        """
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, create new one
            return asyncio.run(self.ocr_async(image, **kwargs))
        else:
            # We're in an async context, create task
            return asyncio.create_task(self.ocr_async(image, **kwargs))
    
    async def ocr_batch_async(
        self,
        images: List[np.ndarray],
        model_version: str = "v5",
        **kwargs
    ) -> List[List[List[Any]]]:
        """
        Batch OCR processing with model version selection
        """
        if not self._initialized:
            await self.initialize()
        
        self.logger.info("Starting batch OCR", batch_size=len(images), model_version=model_version)
        
        # Process all images concurrently
        tasks = [
            self.ocr_async(img, model_version=model_version, **kwargs)
            for img in images
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch item {i} failed", error=str(result))
                processed_results.append([[]])  # Empty result for failed item
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def process_base64_image(
        self,
        base64_image: str,
        model_version: str = "v5",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process base64 encoded image and return structured results
        """
        if not base64_image:
            raise InferenceError("Empty base64 image data")
        
        # Decode base64 to OpenCV image
        image = base64_to_cv2(base64_image)
        if image is None:
            raise InferenceError("Failed to decode base64 image")
        
        # Get full OCR result with metadata
        result = await self.ocr_engine.process_ocr(
            image=image,
            model_version=model_version,
            **kwargs
        )
        
        return result
    
    async def switch_model(self, model_version: str) -> Dict[str, Any]:
        """
        Switch default model version and preload it
        """
        if model_version not in config.model_configs:
            raise ModelNotFoundError(f"Unknown model version: {model_version}")
        
        self.logger.info("Switching default model", from_version=self._default_model_version, to_version=model_version)
        
        # Preload the new model
        await self.model_manager.preload_model(model_version)
        
        # Update default
        old_version = self._default_model_version
        self._default_model_version = model_version
        
        return {
            "success": True,
            "previous_version": old_version,
            "current_version": model_version,
            "message": f"Switched to model version {model_version}"
        }
    
    async def get_model_info(self, model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about loaded models
        """
        if model_version:
            # Get specific model info
            sessions = await self.model_manager.get_model_sessions(model_version)
            return {
                "model_version": model_version,
                "detector": await sessions["det"].get_model_info(),
                "recognizer": await sessions["rec"].get_model_info(),
                "classifier": await sessions["cls"].get_model_info(),
            }
        else:
            # Get overall system info
            stats = await self.ocr_engine.get_stats()
            stats["default_model_version"] = self._default_model_version
            stats["available_models"] = list(config.model_configs.keys())
            return stats
    
    async def get_available_models(self) -> List[str]:
        """
        Get list of available model versions
        """
        return list(config.model_configs.keys())
    
    async def cleanup(self) -> None:
        """
        Clean up resources
        """
        self.logger.info("Cleaning up Modern ONNX OCR")
        
        if self.ocr_engine:
            await self.ocr_engine.cleanup()
        
        if self.model_manager:
            await self.model_manager.clear_cache()
        
        self._initialized = False
        
        self.logger.info("Modern ONNX OCR cleanup completed")


# Legacy compatibility alias
ONNXPaddleOCR = ModernONNXOCR