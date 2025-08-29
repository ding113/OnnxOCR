"""
Async text detector with modern features
"""

import numpy as np
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..core.async_base import AsyncPredictBase
from ..core.config import ModelConfig
from ..core.exceptions import InferenceError
from ..imaug import transform, create_operators
from ..db_postprocess import DBPostProcess


class AsyncTextDetector(AsyncPredictBase):
    """
    Modern async text detector with features:
    - Fully async inference pipeline
    - Type annotations throughout
    - Structured logging
    - Configurable preprocessing
    - Efficient batch processing
    """
    
    def __init__(
        self,
        model_path: Path,
        model_config: Optional[ModelConfig] = None,
        **kwargs
    ):
        super().__init__(model_path=model_path, **kwargs)
        
        self.model_config = model_config
        
        # Initialize preprocessing operations
        self.preprocess_ops = self._create_preprocess_ops()
        
        # Initialize postprocessing
        self.postprocess_op = self._create_postprocess_op()
        
        self.logger = self.logger.bind(model_type="text_detector")
    
    def _create_preprocess_ops(self) -> List[Dict[str, Any]]:
        """Create preprocessing operation chain"""
        det_limit_side_len = 960
        if self.model_config:
            det_limit_side_len = self.model_config.det_limit_side_len
        
        return create_operators([
            {
                "DetResizeForTest": {
                    "limit_side_len": det_limit_side_len,
                    "limit_type": "max",
                }
            },
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ])
    
    def _create_postprocess_op(self) -> DBPostProcess:
        """Create postprocessing operation"""
        params = {
            "thresh": 0.3,
            "box_thresh": 0.6,
            "max_candidates": 1000,
            "unclip_ratio": 1.5,
            "use_dilation": False,
            "score_mode": "fast",
            "box_type": "quad"
        }
        
        if self.model_config:
            params.update({
                "thresh": self.model_config.det_db_thresh,
                "box_thresh": self.model_config.det_db_box_thresh,
                "unclip_ratio": self.model_config.det_db_unclip_ratio
            })
        
        return DBPostProcess(**params)
    
    async def detect_text(
        self, 
        image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Detect text regions in image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (detected_boxes, metadata)
        """
        if image is None or image.size == 0:
            raise InferenceError("Invalid input image")
        
        try:
            
            # Preprocess image
            start_time = asyncio.get_event_loop().time()
            
            data = {"image": image.copy()}
            processed_data = transform(data, self.preprocess_ops)
            
            
            if processed_data is None:
                return None, {"error": "Preprocessing failed"}
            
            img_tensor, shape_list = processed_data
            if img_tensor is None:
                self.logger.error("[DETECTOR] Image preprocessing failed - returned None")
                return None, {"error": "Image preprocessing returned None"}
            
            preprocess_time = asyncio.get_event_loop().time() - start_time
            
            
            # Prepare input data
            img_batch = np.expand_dims(img_tensor, axis=0)
            shape_batch = np.expand_dims(shape_list, axis=0)
            
            input_data = {}
            for i, name in enumerate(self.input_names):
                if i == 0:  # Image input
                    input_data[name] = img_batch.astype(np.float32)
                elif i == 1 and len(self.input_names) > 1:  # Shape input if exists
                    input_data[name] = shape_batch.astype(np.float32)
                else:
                    # Use image input for additional inputs
                    input_data[name] = img_batch.astype(np.float32)
            
            
            # Run async inference
            inference_start = asyncio.get_event_loop().time()
            outputs = await self.predict_async(input_data)
            inference_time = asyncio.get_event_loop().time() - inference_start
            
            
            # Postprocess results  
            postprocess_start = asyncio.get_event_loop().time()
            
            preds = {"maps": outputs[0]}
            
            
            post_result = self.postprocess_op(preds, shape_batch)
            
            if not post_result or len(post_result) == 0:
                self.logger.debug(
                    "[DETECTOR] Postprocessing returned no results",
                    post_result=post_result,
                    post_result_len=len(post_result) if post_result else 0
                )
                return None, {"error": "No text detected"}
            
            dt_boxes = post_result[0]["points"]
            
            self.logger.debug(
                "[DETECTOR] Postprocessing completed",
                raw_boxes_count=len(dt_boxes) if dt_boxes is not None else 0,
                post_result_keys=list(post_result[0].keys()) if post_result and len(post_result) > 0 else [],
                post_result_type=type(post_result).__name__,
                dt_boxes_type=type(dt_boxes).__name__ if dt_boxes is not None else "None",
                sample_boxes=self._safe_boxes_sample(dt_boxes, 3)
            )
            
            # Filter and validate boxes
            self.logger.debug(
                "[DETECTOR] Starting box filtering and validation",
                raw_boxes_count=len(dt_boxes) if dt_boxes is not None else 0,
                raw_boxes_type=type(dt_boxes).__name__ if dt_boxes is not None else "None",
                image_shape=image.shape
            )
            
            filtered_boxes = self._filter_and_validate_boxes(dt_boxes, image.shape)
            
            self.logger.debug(
                "[DETECTOR] Box filtering completed",
                filtered_boxes_count=len(filtered_boxes) if filtered_boxes is not None else 0,
                filtered_boxes_type=type(filtered_boxes).__name__ if filtered_boxes is not None else "None",
                filtering_ratio=round(len(filtered_boxes) / len(dt_boxes), 3) if dt_boxes is not None and len(dt_boxes) > 0 and filtered_boxes is not None else 0
            )
            
            postprocess_time = asyncio.get_event_loop().time() - postprocess_start
            
            metadata = {
                "num_boxes": len(filtered_boxes) if filtered_boxes is not None else 0,
                "image_shape": image.shape,
                "preprocess_time": preprocess_time,
                "inference_time": inference_time, 
                "postprocess_time": postprocess_time,
                "total_time": preprocess_time + inference_time + postprocess_time
            }
            
            
            return filtered_boxes, metadata
            
        except Exception as e:
            self.logger.error("Text detection failed", error=str(e))
            raise InferenceError(f"Text detection failed: {e}") from e
    
    def _filter_and_validate_boxes(
        self, 
        dt_boxes: np.ndarray,
        image_shape: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """
        Filter and validate detected boxes
        """
        if dt_boxes is None or len(dt_boxes) == 0:
            self.logger.debug("[DETECTOR] No boxes to filter")
            return None
        
        self.logger.debug(
            "[DETECTOR] Input to box filtering",
            dt_boxes_type=type(dt_boxes).__name__,
            dt_boxes_length=len(dt_boxes),
            dt_boxes_shape=getattr(dt_boxes, 'shape', 'N/A'),
            first_box_type=type(dt_boxes[0]).__name__ if len(dt_boxes) > 0 else 'N/A',
            first_box_sample=str(dt_boxes[0])[:100] if len(dt_boxes) > 0 else 'N/A'
        )
        
        img_height, img_width = image_shape[:2]
        valid_boxes = []
        rejected_boxes = {"too_small": 0, "invalid_coords": 0}
        
        self.logger.debug(
            "[DETECTOR] Box validation parameters",
            min_width=3,
            min_height=3,
            image_bounds=(img_width, img_height),
            total_boxes_to_validate=len(dt_boxes)
        )
        
        for i, box in enumerate(dt_boxes):
            try:
                if isinstance(box, list):
                    box = np.array(box)
                
                if i < 3:  # Debug first 3 boxes in detail
                    self.logger.debug(
                        f"[DETECTOR] Processing box {i}",
                        box_type=type(box).__name__,
                        box_shape=getattr(box, 'shape', 'N/A'),
                        box_coords=str(box)[:100]
                    )
                
                # Order points clockwise
                box = self._order_points_clockwise(box)
                
                # Clip to image boundaries
                box = self._clip_box_to_image(box, img_height, img_width)
                
                # Validate box dimensions
                rect_width = int(np.linalg.norm(box[0] - box[1]))
                rect_height = int(np.linalg.norm(box[0] - box[3]))
                
                if rect_width > 3 and rect_height > 3:
                    valid_boxes.append(box)
                    if i < 5:  # Log details for first 5 boxes
                        self.logger.debug(
                            f"[DETECTOR] Box {i}: VALID",
                            width=rect_width,
                            height=rect_height,
                            coords=box.tolist()
                        )
                else:
                    rejected_boxes["too_small"] += 1
                    if i < 5:  # Log details for first 5 rejected boxes
                        self.logger.debug(
                            f"[DETECTOR] Box {i}: REJECTED (too small)",
                            width=rect_width,
                            height=rect_height,
                            coords=box.tolist()
                        )
                        
            except Exception as e:
                rejected_boxes["invalid_coords"] += 1
                self.logger.debug(
                    f"[DETECTOR] Box {i}: ERROR processing",
                    error=str(e),
                    box_type=type(box).__name__,
                    box_data=str(box)[:100]
                )
                continue
        
        self.logger.debug(
            "[DETECTOR] Box validation summary",
            valid_boxes=len(valid_boxes),
            rejected_too_small=rejected_boxes["too_small"],
            rejected_invalid=rejected_boxes["invalid_coords"],
            rejection_rate=round((len(dt_boxes) - len(valid_boxes)) / len(dt_boxes), 3) if len(dt_boxes) > 0 else 0
        )
        
        return np.array(valid_boxes) if valid_boxes else None
    
    def _safe_boxes_sample(self, boxes, count: int = 3):
        """
        Safely get a sample of boxes for debugging, handling different types
        """
        if boxes is None or len(boxes) == 0:
            return []
        
        try:
            sample_boxes = boxes[:count]
            if isinstance(sample_boxes, np.ndarray):
                return sample_boxes.tolist()
            elif isinstance(sample_boxes, list):
                return sample_boxes
            else:
                return [str(box) for box in sample_boxes]
        except Exception as e:
            return [f"Error sampling boxes: {str(e)}"]
    
    def _order_points_clockwise(self, pts: np.ndarray) -> np.ndarray:
        """Order 4 points in clockwise order"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(tmp, axis=1)
        rect[1] = tmp[np.argmin(diff)]  # Top-right
        rect[3] = tmp[np.argmax(diff)]  # Bottom-left
        
        return rect
    
    def _clip_box_to_image(
        self, 
        box: np.ndarray, 
        img_height: int, 
        img_width: int
    ) -> np.ndarray:
        """Clip box coordinates to image boundaries"""
        box[:, 0] = np.clip(box[:, 0], 0, img_width - 1)
        box[:, 1] = np.clip(box[:, 1], 0, img_height - 1)
        return box.astype(np.int32)
    
    async def detect_batch(
        self, 
        images: List[np.ndarray]
    ) -> List[Tuple[Optional[np.ndarray], Dict[str, Any]]]:
        """
        Detect text in multiple images (batch processing)
        """
        if not images:
            return []
        
        self.logger.info("Starting batch text detection", batch_size=len(images))
        
        # Process images concurrently
        tasks = [self.detect_text(img) for img in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(
                    "Batch detection failed for image",
                    image_index=i,
                    error=str(result)
                )
                processed_results.append((None, {"error": str(result)}))
            else:
                processed_results.append(result)
        
        return processed_results