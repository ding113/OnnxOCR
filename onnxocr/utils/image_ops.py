"""
Image processing operations for OCR
"""

import base64
import numpy as np
import cv2
from typing import Optional


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> Optional[np.ndarray]:
    """
    Crop and rotate image based on text detection points
    
    Args:
        img: Input image
        points: 4 corner points of the text region
        
    Returns:
        Cropped and rotated image
    """
    assert len(points) == 4, "shape of points must be 4*2"
    
    try:
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]), 
                np.linalg.norm(points[2] - points[3])
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]), 
                np.linalg.norm(points[1] - points[2])
            )
        )
        
        pts_std = np.float32([
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ])
        
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        
        return dst_img
        
    except Exception:
        return None


def get_minarea_rect_crop(img: np.ndarray, points: np.ndarray) -> Optional[np.ndarray]:
    """
    Crop image using minimum area rectangle
    
    Args:
        img: Input image
        points: Corner points of the text region
        
    Returns:
        Cropped image
    """
    try:
        bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_a, index_b, index_c, index_d = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_a = 0
            index_d = 1
        else:
            index_a = 1
            index_d = 0
        if points[3][1] > points[2][1]:
            index_b = 2
            index_c = 3
        else:
            index_b = 3
            index_c = 2

        box = [points[index_a], points[index_b], points[index_c], points[index_d]]
        crop_img = get_rotate_crop_image(img, np.array(box))
        return crop_img
        
    except Exception:
        return None


def base64_to_cv2(b64str: str) -> Optional[np.ndarray]:
    """
    Convert base64 string to OpenCV image
    
    Args:
        b64str: Base64 encoded image string
        
    Returns:
        OpenCV image array or None if conversion fails
    """
    try:
        data = base64.b64decode(b64str.encode("utf8"))
        data = np.frombuffer(data, np.uint8)
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return data
    except Exception:
        return None


def validate_image(img: np.ndarray) -> bool:
    """
    Validate that an image array is valid for processing
    
    Args:
        img: Image array to validate
        
    Returns:
        True if image is valid, False otherwise
    """
    if img is None:
        return False
    
    if not isinstance(img, np.ndarray):
        return False
    
    if img.size == 0:
        return False
    
    if len(img.shape) not in [2, 3]:
        return False
    
    if len(img.shape) == 3 and img.shape[2] not in [1, 3, 4]:
        return False
    
    return True


def normalize_image_for_display(img: np.ndarray) -> np.ndarray:
    """
    Normalize image for display purposes
    
    Args:
        img: Input image
        
    Returns:
        Normalized image
    """
    if img.dtype != np.uint8:
        # Convert to uint8 range
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img