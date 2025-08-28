"""
Structured exception definitions for ONNX OCR system
"""

from typing import Optional, Dict, Any


class OCRError(Exception):
    """Base exception for all OCR-related errors"""
    
    def __init__(
        self, 
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "GENERAL_ERROR"
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context
        }


class ModelLoadError(OCRError):
    """Raised when model loading fails"""
    
    def __init__(self, message: str, model_path: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="MODEL_LOAD_ERROR",
            context={"model_path": model_path} if model_path else {}
        )


class ModelNotFoundError(OCRError):
    """Raised when a requested model is not found"""
    
    def __init__(self, message: str, model_version: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="MODEL_NOT_FOUND",
            context={"model_version": model_version} if model_version else {}
        )


class InferenceError(OCRError):
    """Raised when model inference fails"""
    
    def __init__(self, message: str, model_info: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="INFERENCE_ERROR",
            context={"model_info": model_info} if model_info else {}
        )


class DownloadError(OCRError):
    """Raised when model download fails"""
    
    def __init__(
        self, 
        message: str, 
        url: Optional[str] = None,
        status_code: Optional[int] = None
    ):
        context = {}
        if url:
            context["url"] = url
        if status_code:
            context["status_code"] = status_code
            
        super().__init__(
            message=message,
            error_code="DOWNLOAD_ERROR",
            context=context
        )


class ValidationError(OCRError):
    """Raised when model or data validation fails"""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None
    ):
        context = {}
        if field:
            context["field"] = field
        if expected is not None:
            context["expected"] = expected
        if actual is not None:
            context["actual"] = actual
            
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            context=context
        )