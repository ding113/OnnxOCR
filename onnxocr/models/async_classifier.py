"""
Async text classifier (angle detection) - shared across all model versions
"""

import cv2
import numpy as np
import asyncio
import math
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image

from ..core.async_base import AsyncPredictBase
from ..core.config import ModelConfig
from ..core.exceptions import InferenceError
from ..cls_postprocess import ClsPostProcess


class AsyncTextClassifier(AsyncPredictBase):
    """
    Modern async text classifier (angle detection) with features:
    - Shared across all model versions (v4, v5, v5-server)  
    - Fully async inference pipeline
    - Batch processing optimization
    - Configurable threshold and parameters
    - Angle detection and correction
    """
    
    def __init__(
        self,
        model_path: Path,
        model_config: Optional[ModelConfig] = None,
        **kwargs
    ):
        super().__init__(model_path=model_path, **kwargs)
        
        self.model_config = model_config
        
        # Set up classification parameters
        self.cls_image_shape = [3, 48, 192]
        self.cls_batch_num = 6
        self.cls_thresh = 0.9
        self.label_list = ["0", "180"]
        
        if model_config:
            shape_str = model_config.cls_image_shape
            self.cls_image_shape = [int(v) for v in shape_str.split(",")]
            self.cls_batch_num = model_config.cls_batch_num
            self.cls_thresh = model_config.cls_thresh
        
        # Initialize postprocessing
        self.postprocess_op = ClsPostProcess()
        
        self.logger = self.logger.bind(model_type="text_classifier")
    
    def _resize_norm_img(self, img: np.ndarray) -> np.ndarray:
        """
        Resize and normalize image for classification
        """
        imgC, imgH, imgW = self.cls_image_shape
        h = img.shape[0]
        w = img.shape[1]
        
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        
        if self.cls_image_shape[0] == 1:
            resized_image = resized_image[:, :, 0:1]
        
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        
        # Pad to target width
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        
        return padding_im
    
    async def classify_angles(
        self,
        img_crops: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Tuple[str, float]]]:
        """
        Classify text angles and rotate images if needed
        
        Args:
            img_crops: List of cropped text region images
            
        Returns:
            Tuple of (rotated_images, classification_results)
            classification_results: List of (angle_label, confidence) tuples
        """
        if not img_crops:
            return [], []
        
        try:
            # Process images in batches
            all_rotated_imgs = []
            all_cls_results = []
            
            for i in range(0, len(img_crops), self.cls_batch_num):
                batch_crops = img_crops[i:i + self.cls_batch_num]
                rotated_batch, cls_batch = await self._classify_batch(batch_crops)
                all_rotated_imgs.extend(rotated_batch)
                all_cls_results.extend(cls_batch)
            
            return all_rotated_imgs, all_cls_results
            
        except Exception as e:
            self.logger.error("Angle classification failed", error=str(e))
            raise InferenceError(f"Angle classification failed: {e}") from e
    
    async def _classify_batch(
        self,
        img_crops: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Tuple[str, float]]]:
        """
        Process a batch of cropped images for angle classification
        """
        if not img_crops:
            return [], []
        
        start_time = asyncio.get_event_loop().time()
        
        # Preprocess batch
        norm_img_batch = []
        valid_indices = []
        
        for idx, img in enumerate(img_crops):
            if img is None or img.size == 0:
                continue
            
            try:
                norm_img = self._resize_norm_img(img)
                norm_img_batch.append(norm_img)
                valid_indices.append(idx)
            except Exception as e:
                self.logger.warning(f"Failed to preprocess crop {idx}", error=str(e))
                continue
        
        if not norm_img_batch:
            # Return original images with default results
            return img_crops, [("0", 1.0)] * len(img_crops)
        
        # Stack batch
        input_batch = np.stack(norm_img_batch, axis=0)
        
        preprocess_time = asyncio.get_event_loop().time() - start_time
        
        # Prepare input data
        input_data = {}
        for name in self.input_names:
            input_data[name] = input_batch.astype(np.float32)
        
        # Run async inference
        inference_start = asyncio.get_event_loop().time()
        outputs = await self.predict_async(input_data)
        inference_time = asyncio.get_event_loop().time() - inference_start
        
        # Postprocess results
        postprocess_start = asyncio.get_event_loop().time()
        
        preds = outputs[0]  # Get classification logits
        cls_results = self.postprocess_op(preds, self.label_list)
        
        postprocess_time = asyncio.get_event_loop().time() - postprocess_start
        
        # Process rotation and create results
        rotated_images = []
        classification_results = []
        
        for idx, img in enumerate(img_crops):
            if idx in valid_indices:
                valid_idx_pos = valid_indices.index(idx)
                if valid_idx_pos < len(cls_results):
                    label, score = cls_results[valid_idx_pos]
                    
                    # Rotate image if needed and confidence is high enough
                    if label == "180" and score >= self.cls_thresh:
                        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
                        rotated_images.append(rotated_img)
                    else:
                        rotated_images.append(img)
                    
                    classification_results.append((label, float(score)))
                else:
                    rotated_images.append(img)
                    classification_results.append(("0", 1.0))
            else:
                rotated_images.append(img)
                classification_results.append(("0", 1.0))
        
        self.logger.debug(
            "Batch angle classification completed",
            batch_size=len(img_crops),
            valid_crops=len(valid_indices),
            rotated_count=sum(1 for r in classification_results if r[0] == "180" and r[1] >= self.cls_thresh),
            preprocess_time=preprocess_time,
            inference_time=inference_time,
            postprocess_time=postprocess_time
        )
        
        return rotated_images, classification_results
    
    async def classify_single(
        self,
        img_crop: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[str, float]]:
        """
        Classify angle for a single cropped image
        """
        rotated_imgs, cls_results = await self.classify_angles([img_crop])
        
        if rotated_imgs and cls_results:
            return rotated_imgs[0], cls_results[0]
        else:
            return img_crop, ("0", 1.0)
    
    def is_rotation_needed(self, angle_label: str, confidence: float) -> bool:
        """
        Determine if rotation is needed based on classification result
        """
        return angle_label == "180" and confidence >= self.cls_thresh
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information
        """
        base_info = await super().get_model_info()
        base_info.update({
            "cls_image_shape": self.cls_image_shape,
            "cls_batch_num": self.cls_batch_num,
            "cls_thresh": self.cls_thresh,
            "label_list": self.label_list,
            "supports_batch": True,
            "max_batch_size": self.cls_batch_num,
            "shared_across_versions": True
        })
        return base_info