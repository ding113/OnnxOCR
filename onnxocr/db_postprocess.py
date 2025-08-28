"""
DB (Differentiable Binarization) postprocessing for text detection
DB后处理模块用于文本检测
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import math
from shapely.geometry import Polygon
import pyclipper


class DBPostProcess:
    """
    DB postprocessing for text detection
    文本检测DB后处理
    """
    
    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.5,
        use_dilation: bool = False,
        score_mode: str = "fast",
        box_type: str = "quad"
    ):
        """
        Initialize DB postprocessing
        
        Args:
            thresh: Binary threshold for segmentation map
            box_thresh: Box score threshold for filtering
            max_candidates: Maximum number of candidate boxes
            unclip_ratio: Unclipping ratio for box expansion
            use_dilation: Whether to use morphological dilation
            score_mode: Scoring mode ('fast' or 'slow')
            box_type: Output box type ('quad' or 'poly')
        """
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.use_dilation = use_dilation
        self.score_mode = score_mode
        self.box_type = box_type
        self.min_size = 3
        self.dilation_kernel = np.array([[1, 1], [1, 1]])
    
    def __call__(
        self,
        outs_dict: Dict[str, np.ndarray],
        shape_list: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Process detection outputs to get bounding boxes
        
        Args:
            outs_dict: Model outputs with 'maps' key containing prediction maps
            shape_list: Shape information for each image in batch
            
        Returns:
            List of detection results, each containing 'points' key with boxes
        """
        pred = outs_dict['maps']
        if isinstance(pred, list):
            pred = pred[0]
        
        pred = pred[:, 0, :, :]  # Remove channel dimension
        
        batch_size = pred.shape[0]
        results = []
        
        for batch_idx in range(batch_size):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_idx]
            
            # Get current prediction map
            bitmap = pred[batch_idx]
            
            # Convert to binary mask
            if self.use_dilation:
                mask = cv2.dilate(
                    np.array(bitmap > self.thresh, dtype=np.uint8),
                    self.dilation_kernel
                )
            else:
                mask = bitmap > self.thresh
            
            # Find boxes from mask
            boxes, scores = self._boxes_from_bitmap(
                bitmap, mask, src_w, src_h, ratio_w, ratio_h
            )
            
            results.append({
                'points': boxes,
                'scores': scores
            })
        
        return results
    
    def _boxes_from_bitmap(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
        dest_width: float,
        dest_height: float,
        ratio_w: float,
        ratio_h: float
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract boxes from prediction bitmap
        
        Args:
            pred: Prediction probability map
            bitmap: Binary mask
            dest_width: Target width
            dest_height: Target height  
            ratio_w: Width scaling ratio
            ratio_h: Height scaling ratio
            
        Returns:
            Tuple of (boxes, scores)
        """
        height, width = bitmap.shape
        
        # Find contours
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        num_contours = min(len(contours), self.max_candidates)
        boxes = []
        scores = []
        
        for i in range(num_contours):
            contour = contours[i]
            points, sside = self._get_mini_boxes(contour.reshape((-1, 1, 2)))
            
            if sside < self.min_size:
                continue
                
            points = np.array(points)
            
            if self.score_mode == "fast":
                score = self._box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self._box_score_slow(pred, contour)
                
            if score < self.box_thresh:
                continue
            
            # Unclip the box
            box = self._unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self._get_mini_boxes(box)
            
            if sside < self.min_size + 2:
                continue
            
            box = np.array(box)
            
            if not isinstance(dest_width, int):
                dest_width = int(dest_width)
            if not isinstance(dest_height, int):
                dest_height = int(dest_height)
            
            # Scale back to original image size
            box[:, 0] = np.clip(
                np.round(box[:, 0] / ratio_w), 0, dest_width
            )
            box[:, 1] = np.clip(
                np.round(box[:, 1] / ratio_h), 0, dest_height  
            )
            
            boxes.append(box.astype(np.int16))
            scores.append(score)
        
        return boxes, scores
    
    def _get_mini_boxes(self, contour: np.ndarray) -> Tuple[List[List[float]], float]:
        """
        Get minimum area rectangle from contour
        
        Args:
            contour: Input contour points
            
        Returns:
            Tuple of (box_points, min_side_length)
        """
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        
        box = [
            points[index_1], points[index_2], 
            points[index_3], points[index_4]
        ]
        
        # Calculate minimum side length
        side_1 = np.linalg.norm(np.array(box[0]) - np.array(box[1]))
        side_2 = np.linalg.norm(np.array(box[1]) - np.array(box[2]))
        
        return box, min(side_1, side_2)
    
    def _box_score_fast(self, bitmap: np.ndarray, _box: np.ndarray) -> float:
        """
        Fast box scoring using mean of pixels inside box
        
        Args:
            bitmap: Prediction probability map
            _box: Box coordinates
            
        Returns:
            Box confidence score
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)
        
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
    
    def _box_score_slow(self, bitmap: np.ndarray, contour: np.ndarray) -> float:
        """
        Slow box scoring using precise contour mask
        
        Args:
            bitmap: Prediction probability map  
            contour: Contour points
            
        Returns:
            Box confidence score
        """
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = contour.reshape((-1, 2))
        
        xmin = np.clip(np.floor(contour[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(contour[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(contour[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(contour[:, 1].max()).astype(np.int32), 0, h - 1)
        
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        
        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin
        
        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
    
    def _unclip(self, box: np.ndarray, unclip_ratio: float) -> np.ndarray:
        """
        Unclip the box to expand it
        
        Args:
            box: Box coordinates
            unclip_ratio: Expansion ratio
            
        Returns:
            Expanded box coordinates
        """
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        
        if len(expanded) == 0:
            return box
        else:
            expanded = np.array(expanded[0])
            return expanded