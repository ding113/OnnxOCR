"""
Async OCR Engine - orchestrates the complete OCR pipeline
"""

import asyncio
import time
import copy
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import structlog

from ..core import SmartModelManager, config
from ..core.exceptions import InferenceError, ModelNotFoundError
from ..utils.image_ops import get_rotate_crop_image, get_minarea_rect_crop
from .async_detector import AsyncTextDetector
from .async_recognizer import AsyncTextRecognizer
from .async_classifier import AsyncTextClassifier

logger = structlog.get_logger()


class AsyncOCREngine:
    """
    Modern async OCR engine with features:
    - Dynamic model switching (det/rec) with shared classifier
    - Fully async pipeline with concurrent processing
    - Intelligent batching and resource management
    - Comprehensive performance monitoring
    - Memory optimization and cleanup
    """
    
    def __init__(self, model_manager: Optional[SmartModelManager] = None):
        self.model_manager = model_manager or SmartModelManager()
        self.logger = logger.bind(component="AsyncOCREngine")
        
        # Performance tracking
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "model_switches": 0
        }
        
        # Current state
        self._current_model_version = None
        self._last_used_sessions = None
    
    async def process_ocr(
        self,
        image: np.ndarray,
        model_version: str = "v5",
        use_angle_cls: bool = True,
        drop_score: float = 0.5,
        det_limit_side_len: float = 960
    ) -> Dict[str, Any]:
        """
        Process OCR on an image with specified model version
        
        Args:
            image: Input image as numpy array
            model_version: Model version to use ('v4', 'v5', 'v5-server')
            use_angle_cls: Whether to use angle classification
            drop_score: Confidence threshold for filtering results
            det_limit_side_len: Detection limit side length
            
        Returns:
            Dictionary containing OCR results and metadata
        """
        if image is None or image.size == 0:
            raise InferenceError("Invalid input image")
        
        start_time = time.time()
        
        try:
            self._stats["total_requests"] += 1
            
            # Track model switches
            if self._current_model_version != model_version:
                self._stats["model_switches"] += 1
                self._current_model_version = model_version
                self.logger.info(
                    "Switching model version", 
                    from_version=self._current_model_version,
                    to_version=model_version
                )
            
            # Get model sessions (det/rec/cls)
            sessions = await self.model_manager.get_model_sessions(model_version)
            detector = sessions["det"]
            recognizer = sessions["rec"]  
            classifier = sessions["cls"]
            
            self._last_used_sessions = sessions
            
            # Step 1: Text Detection
            detection_start = time.time()
            dt_boxes, det_metadata = await detector.detect_text(image)
            detection_time = time.time() - detection_start
            
            if dt_boxes is None or len(dt_boxes) == 0:
                return self._create_empty_result(
                    model_version=model_version,
                    processing_time=time.time() - start_time,
                    message="No text detected",
                    image_info=self._get_image_info(image),
                    det_metadata=det_metadata
                )
            
            # Step 2: Image Cropping
            crop_start = time.time()
            img_crop_list = await self._crop_text_regions(image, dt_boxes)
            crop_time = time.time() - crop_start
            
            if not img_crop_list:
                return self._create_empty_result(
                    model_version=model_version,
                    processing_time=time.time() - start_time,
                    message="Failed to crop text regions",
                    image_info=self._get_image_info(image)
                )
            
            # Step 3: Angle Classification (if enabled)
            cls_time = 0.0
            cls_results = []
            
            if use_angle_cls:
                cls_start = time.time()
                img_crop_list, cls_results = await classifier.classify_angles(img_crop_list)
                cls_time = time.time() - cls_start
            
            # Step 4: Text Recognition
            rec_start = time.time()
            rec_results = await recognizer.recognize_text(img_crop_list)
            rec_time = time.time() - rec_start
            
            # Step 5: Post-processing and filtering
            postprocess_start = time.time()
            filtered_results = self._filter_and_format_results(
                dt_boxes, rec_results, cls_results, drop_score
            )
            postprocess_time = time.time() - postprocess_start
            
            total_time = time.time() - start_time
            
            # Update statistics
            self._stats["successful_requests"] += 1
            self._stats["total_processing_time"] += total_time
            
            # Create comprehensive result
            result = {
                "success": True,
                "results": filtered_results,
                "model_version": model_version,
                "processing_time": total_time,
                "image_info": self._get_image_info(image),
                "model_info": {
                    "detector": await detector.get_model_info(),
                    "recognizer": await recognizer.get_model_info(),
                    "classifier": await classifier.get_model_info() if use_angle_cls else None
                },
                "performance_metrics": {
                    "detection_time": detection_time,
                    "cropping_time": crop_time,
                    "classification_time": cls_time,
                    "recognition_time": rec_time,
                    "postprocessing_time": postprocess_time,
                    "total_time": total_time,
                    "detected_boxes": len(dt_boxes),
                    "recognized_texts": len(filtered_results),
                    "use_angle_cls": use_angle_cls
                },
                "metadata": {
                    "detection": det_metadata,
                    "classification": cls_results if use_angle_cls else [],
                    "drop_score": drop_score,
                    "det_limit_side_len": det_limit_side_len
                }
            }
            
            self.logger.info(
                "OCR processing completed",
                model_version=model_version,
                detected_boxes=len(dt_boxes),
                recognized_texts=len(filtered_results),
                processing_time=total_time
            )
            
            return result
            
        except Exception as e:
            self._stats["failed_requests"] += 1
            
            self.logger.error(
                "OCR processing failed",
                model_version=model_version,
                error=str(e),
                processing_time=time.time() - start_time
            )
            
            return {
                "success": False,
                "results": [],
                "model_version": model_version,
                "processing_time": time.time() - start_time,
                "image_info": self._get_image_info(image),
                "error": str(e),
                "error_code": "PROCESSING_ERROR"
            }
    
    async def _crop_text_regions(
        self,
        image: np.ndarray,
        dt_boxes: np.ndarray
    ) -> List[np.ndarray]:
        """
        Crop text regions from the image based on detected boxes
        """
        if dt_boxes is None or len(dt_boxes) == 0:
            return []
        
        img_crop_list = []
        ori_im = image.copy()
        
        # Sort boxes from top to bottom, left to right
        sorted_boxes = self._sort_boxes(dt_boxes)
        
        for box in sorted_boxes:
            tmp_box = copy.deepcopy(box)
            
            # Crop using perspective transform for quadrilateral boxes
            if len(tmp_box) == 4:
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                # Fallback for other box types
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            
            if img_crop is not None and img_crop.size > 0:
                img_crop_list.append(img_crop)
        
        return img_crop_list
    
    def _sort_boxes(self, dt_boxes: np.ndarray) -> List[np.ndarray]:
        """
        Sort text boxes in reading order (top to bottom, left to right)
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)
        
        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if (abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and
                    (_boxes[j + 1][0][0] < _boxes[j][0][0])):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        
        return _boxes
    
    def _filter_and_format_results(
        self,
        dt_boxes: np.ndarray,
        rec_results: List[Tuple[str, float]],
        cls_results: List[Tuple[str, float]],
        drop_score: float
    ) -> List[Dict[str, Any]]:
        """
        Filter and format OCR results
        """
        filtered_results = []
        
        for i, (box, (text, score)) in enumerate(zip(dt_boxes, rec_results)):
            if score >= drop_score:
                result = {
                    "box": box.tolist(),
                    "text": text,
                    "confidence": float(score)
                }
                
                # Add classification info if available
                if i < len(cls_results):
                    cls_label, cls_score = cls_results[i]
                    result["angle"] = {
                        "label": cls_label,
                        "confidence": float(cls_score)
                    }
                
                filtered_results.append(result)
        
        return filtered_results
    
    def _create_empty_result(
        self,
        model_version: str,
        processing_time: float,
        message: str,
        image_info: Dict[str, Any],
        det_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an empty result structure
        """
        return {
            "success": True,
            "results": [],
            "model_version": model_version,
            "processing_time": processing_time,
            "image_info": image_info,
            "message": message,
            "performance_metrics": {
                "detection_metadata": det_metadata
            }
        }
    
    def _get_image_info(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract basic image information
        """
        return {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "size_bytes": image.nbytes,
            "channels": image.shape[2] if len(image.shape) == 3 else 1
        }
    
    async def process_batch(
        self,
        images: List[np.ndarray],
        model_version: str = "v5",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images concurrently
        """
        if not images:
            return []
        
        self.logger.info("Starting batch OCR processing", batch_size=len(images))
        
        # Process all images concurrently
        tasks = [
            self.process_ocr(img, model_version=model_version, **kwargs)
            for img in images
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(
                    "Batch processing failed for image",
                    image_index=i,
                    error=str(result)
                )
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "image_index": i
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics
        """
        cache_info = await self.model_manager.get_cache_info()
        
        avg_processing_time = 0.0
        if self._stats["successful_requests"] > 0:
            avg_processing_time = (
                self._stats["total_processing_time"] / 
                self._stats["successful_requests"]
            )
        
        return {
            "engine_stats": {
                **self._stats,
                "average_processing_time": avg_processing_time,
                "success_rate": (
                    self._stats["successful_requests"] / 
                    max(1, self._stats["total_requests"])
                ),
                "current_model_version": self._current_model_version
            },
            "model_manager_stats": cache_info,
            "config": {
                "max_memory_gb": config.max_memory_gb,
                "thread_pool_size": config.thread_pool_size,
                "model_cache_size": config.model_cache_size
            }
        }
    
    async def cleanup(self):
        """
        Clean up resources
        """
        self.logger.info("Cleaning up OCR engine resources")
        
        if self.model_manager:
            await self.model_manager.clear_cache()
        
        self.logger.info("OCR engine cleanup completed")