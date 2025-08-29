"""
Async text recognizer with modern features
"""

import cv2
import numpy as np
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image

from ..core.async_base import AsyncPredictBase
from ..core.config import ModelConfig
from ..core.exceptions import InferenceError
from ..rec_postprocess import CTCLabelDecode


class AsyncTextRecognizer(AsyncPredictBase):
    """
    Modern async text recognizer with features:
    - Fully async inference pipeline
    - Batch processing optimization
    - Configurable image preprocessing
    - Support for multiple algorithms
    - Structured logging and metrics
    """
    
    def __init__(
        self,
        model_path: Path,
        model_config: Optional[ModelConfig] = None,
        **kwargs
    ):
        super().__init__(model_path=model_path, **kwargs)
        
        self.model_config = model_config
        
        # Set up recognition parameters
        self.rec_image_shape = [3, 48, 320]
        self.rec_batch_num = 6
        self.rec_algorithm = "SVTR_LCNet"
        
        self.logger = self.logger.bind(model_type="text_recognizer")
        
        
        if model_config:
            shape_str = model_config.rec_image_shape
            self.rec_image_shape = [int(v) for v in shape_str.split(",")]
            self.rec_batch_num = model_config.rec_batch_num
            
        
        # Initialize postprocessing
        self.postprocess_op = self._create_postprocess_op()
        
        self.logger.info(
            "[RECOGNIZER] Text recognizer initialized",
            final_image_shape=self.rec_image_shape,
            final_batch_num=self.rec_batch_num,
            algorithm=self.rec_algorithm,
            postprocess_created=self.postprocess_op is not None
        )
    
    def _create_postprocess_op(self) -> CTCLabelDecode:
        """Create postprocessing operation for text decoding"""
        dict_path = None
        use_space_char = True
        
        if self.model_config:
            dict_path = str(self.model_config.dict_path)
        
        
        postprocess_op = CTCLabelDecode(
            character_dict_path=dict_path,
            use_space_char=use_space_char
        )
        
        
        return postprocess_op
    
    def _resize_norm_img(
        self, 
        img: np.ndarray, 
        max_wh_ratio: float
    ) -> np.ndarray:
        """
        Resize and normalize image for recognition
        """
        imgC, imgH, imgW = self.rec_image_shape
        
        # Handle special algorithms
        if self.rec_algorithm in ["NRTR", "ViTSTR"]:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_pil = Image.fromarray(np.uint8(img))
            
            if self.rec_algorithm == "ViTSTR":
                img = image_pil.resize([imgW, imgH], Image.BICUBIC)
            else:
                img = image_pil.resize([imgW, imgH], Image.LANCZOS)
            
            img = np.array(img)
            norm_img = np.expand_dims(img, -1)
            norm_img = norm_img.transpose((2, 0, 1))
            
            if self.rec_algorithm == "ViTSTR":
                norm_img = norm_img.astype(np.float32) / 255.0
            else:
                norm_img = norm_img.astype(np.float32) / 128.0 - 1.0
            
            return norm_img
        
        elif self.rec_algorithm == "RFL":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_CUBIC)
            resized_image = resized_image.astype("float32")
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
            resized_image -= 0.5
            resized_image /= 0.5
            return resized_image
        
        # Standard algorithm (SVTR_LCNet, CRNN, etc.)
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        
        if self.rec_algorithm == "RARE":
            imgW = 32
        
        h, w = img.shape[:2]
        ratio = w / float(h)
        
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        
        # Pad to target width
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        
        return padding_im
    
    async def recognize_text(
        self,
        img_crops: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        """
        Recognize text from cropped images
        
        Args:
            img_crops: List of cropped text region images
            
        Returns:
            List of (text, confidence) tuples
        """
        if not img_crops:
            return []
        
        try:
            # Calculate max width-height ratio
            max_wh_ratio = 0
            for img in img_crops:
                if img is not None and img.size > 0:
                    wh_ratio = img.shape[1] / img.shape[0]
                    max_wh_ratio = max(max_wh_ratio, wh_ratio)
            
            max_wh_ratio = max(max_wh_ratio, 1.0)
            
            # Process images in batches
            results = []
            for i in range(0, len(img_crops), self.rec_batch_num):
                batch_crops = img_crops[i:i + self.rec_batch_num]
                batch_results = await self._recognize_batch(batch_crops, max_wh_ratio)
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            self.logger.error("Text recognition failed", error=str(e))
            raise InferenceError(f"Text recognition failed: {e}") from e
    
    async def _recognize_batch(
        self,
        img_crops: List[np.ndarray],
        max_wh_ratio: float
    ) -> List[Tuple[str, float]]:
        """
        Process a batch of cropped images
        """
        if not img_crops:
            return []
        
        start_time = asyncio.get_event_loop().time()
        
        
        # Preprocess batch
        norm_img_batch = []
        valid_indices = []
        preprocess_failures = 0
        
        for idx, img in enumerate(img_crops):
            if img is None or img.size == 0:
                preprocess_failures += 1
                continue
            
            try:
                norm_img = self._resize_norm_img(img, max_wh_ratio)
                norm_img_batch.append(norm_img)
                valid_indices.append(idx)
                
            except Exception as e:
                preprocess_failures += 1
                self.logger.error(
                    f"[RECOGNIZER] Failed to preprocess crop {idx}",
                    error=str(e),
                    img_shape=img.shape if img is not None else None
                )
                continue
        
        if not norm_img_batch:
            self.logger.error(
                "[RECOGNIZER] No valid images after preprocessing",
                total_crops=len(img_crops),
                preprocess_failures=preprocess_failures
            )
            return [("", 0.0)] * len(img_crops)
        
        
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
        
        preds = outputs[0]  # Get logits
        
        
        rec_results = self.postprocess_op(preds)
        
        postprocess_time = asyncio.get_event_loop().time() - postprocess_start
        
        
        # Map results back to original indices
        results = [("", 0.0)] * len(img_crops)
        successful_recognitions = 0
        
        for i, valid_idx in enumerate(valid_indices):
            if i < len(rec_results):
                text, conf = rec_results[i]
                results[valid_idx] = (text, float(conf))
                
                if text.strip():  # Non-empty text
                    successful_recognitions += 1
                
        
        self.logger.info(
            "[RECOGNIZER] Batch recognition completed",
            batch_size=len(img_crops),
            valid_crops=len(valid_indices),
            successful_recognitions=successful_recognitions,
            success_rate=round(successful_recognitions / len(img_crops), 3) if len(img_crops) > 0 else 0,
            preprocess_time=round(preprocess_time, 4),
            inference_time=round(inference_time, 4),
            postprocess_time=round(postprocess_time, 4),
            total_time=round(asyncio.get_event_loop().time() - start_time, 4)
        )
        
        return results
    
    async def recognize_single(
        self,
        img_crop: np.ndarray
    ) -> Tuple[str, float]:
        """
        Recognize text from a single cropped image
        """
        results = await self.recognize_text([img_crop])
        return results[0] if results else ("", 0.0)
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information
        """
        base_info = await super().get_model_info()
        base_info.update({
            "rec_image_shape": self.rec_image_shape,
            "rec_batch_num": self.rec_batch_num,
            "rec_algorithm": self.rec_algorithm,
            "supports_batch": True,
            "max_batch_size": self.rec_batch_num
        })
        return base_info


# Import math for legacy compatibility
import math