"""
Core module for modern ONNX OCR system
"""

from .async_base import AsyncPredictBase, create_predictor
from .config import ModelConfig, OCRConfig, OCRRequest, OCRResponse, ModelSwitchRequest, config
from .downloader import ModelDownloader
from .model_manager import SmartModelManager
from .exceptions import (
    OCRError, 
    ModelLoadError, 
    ModelNotFoundError, 
    InferenceError,
    DownloadError,
    ValidationError
)

__all__ = [
    "AsyncPredictBase",
    "create_predictor", 
    "ModelConfig", 
    "OCRConfig",
    "OCRRequest",
    "OCRResponse", 
    "ModelSwitchRequest",
    "config",
    "ModelDownloader",
    "SmartModelManager",
    "OCRError",
    "ModelLoadError", 
    "ModelNotFoundError",
    "InferenceError",
    "DownloadError",
    "ValidationError"
]