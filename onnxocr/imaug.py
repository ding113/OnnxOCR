"""
Modern image augmentation and preprocessing for OCR
现代化图像增强和预处理模块
"""

import cv2
import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


def transform(data: Dict[str, Any], ops: List[Any]) -> Dict[str, Any]:
    """
    Apply a series of transformations to the input data
    对输入数据应用一系列变换
    """
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_config_list: List[Dict[str, Any]]) -> List[Any]:
    """
    Create operators from configuration list
    从配置列表创建操作符
    """
    ops = []
    for config in op_config_list:
        op_name = list(config.keys())[0]
        op_params = config[op_name] if config[op_name] else {}
        
        # Create operator based on name
        if op_name == 'DetResizeForTest':
            ops.append(DetResizeForTest(**op_params))
        elif op_name == 'NormalizeImage':
            ops.append(NormalizeImage(**op_params))
        elif op_name == 'ToCHWImage':
            ops.append(ToCHWImage(**op_params))
        elif op_name == 'KeepKeys':
            ops.append(KeepKeys(**op_params))
        elif op_name == 'RecResizeImg':
            ops.append(RecResizeImg(**op_params))
        else:
            # Default pass-through operator
            ops.append(lambda x: x)
    
    return ops


class DetResizeForTest:
    """Detection model resize for testing"""
    
    def __init__(self, **kwargs):
        self.limit_side_len = kwargs.get('limit_side_len', 960)
        self.limit_type = kwargs.get('limit_type', 'max')

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        img = data['image']
        src_h, src_w, _ = img.shape
        
        if self.limit_type == 'max':
            if max(src_h, src_w) > self.limit_side_len:
                if src_h > src_w:
                    ratio = float(self.limit_side_len) / src_h
                else:
                    ratio = float(self.limit_side_len) / src_w
            else:
                ratio = 1.0
        else:
            ratio = 1.0
        
        resize_h = int(src_h * ratio)
        resize_w = int(src_w * ratio)
        
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)
        
        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except Exception as e:
            print(f"Error in resize: {e}")
            return None
        
        ratio_h = resize_h / src_h
        ratio_w = resize_w / src_w
        
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data


class NormalizeImage:
    """Normalize image"""
    
    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        if isinstance(mean, str):
            mean = eval(mean)
        if isinstance(std, str):
            std = eval(std)
        
        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        
        # Handle scale parameter
        if scale is None:
            scale = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
        elif isinstance(scale, (int, float)):
            scale = [scale, scale, scale]
        elif not isinstance(scale, (list, tuple)):
            scale = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
        self.scale = np.array(scale).reshape(shape).astype('float32')
        
        # Handle mean parameter  
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        elif isinstance(mean, (int, float)):
            mean = [mean, mean, mean]
        elif not isinstance(mean, (list, tuple)):
            mean = [0.485, 0.456, 0.406]
        self.mean = np.array(mean).reshape(shape).astype('float32')
        
        # Handle std parameter
        if std is None:
            std = [0.229, 0.224, 0.225]
        elif isinstance(std, (int, float)):
            std = [std, std, std]
        elif not isinstance(std, (list, tuple)):
            std = [0.229, 0.224, 0.225]
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        img = data['image']
        img = (img.astype('float32') * self.scale - self.mean) / self.std
        data['image'] = img
        return data


class ToCHWImage:
    """Convert image to CHW format"""
    
    def __init__(self, **kwargs):
        pass

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        img = data['image']
        img = img.transpose((2, 0, 1))
        data['image'] = img
        return data


class KeepKeys:
    """Keep specified keys in data"""
    
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class RecResizeImg:
    """Recognition model image resize"""
    
    def __init__(self, image_shape=None, infer_mode=False, character_type='ch', **kwargs):
        self.image_shape = image_shape or [3, 48, 320]
        self.infer_mode = infer_mode
        self.character_type = character_type

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        img = data['image']
        
        if self.infer_mode and self.character_type == 'ch':
            norm_img = self.resize_norm_img_chinese(img, self.image_shape)
        else:
            norm_img = self.resize_norm_img(img, self.image_shape)
        
        data['image'] = norm_img
        return data
    
    def resize_norm_img(self, img, image_shape):
        imgC, imgH, imgW = image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        
        if image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        
        resized_image -= 0.5
        resized_image /= 0.5
        
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
    
    def resize_norm_img_chinese(self, img, image_shape):
        imgC, imgH, imgW = image_shape
        max_wh_ratio = imgW / imgH
        h, w = img.shape[:2]
        ratio = w / float(h)
        max_wh_ratio = max(max_wh_ratio, ratio)
        
        imgW = int((imgH * max_wh_ratio))
        if imgW % 32 != 0:
            imgW = int((imgW // 32 + 1) * 32)
        
        w = int(imgH * ratio)
        if w > imgW:
            w = imgW
        
        img = cv2.resize(img, (w, imgH))
        img = img.astype('float32')
        img = img / 255.0
        img = img[np.newaxis, :]
        
        return img