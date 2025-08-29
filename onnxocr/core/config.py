"""
Modern configuration system using Pydantic
"""

import os
from pathlib import Path
from typing import Dict, Literal, Optional, Any
from pydantic import BaseModel, Field, validator, computed_field
from pydantic_settings import BaseSettings

# Get the onnxocr module directory
MODULE_DIR = Path(__file__).resolve().parent.parent


class ModelConfig(BaseModel):
    """Configuration for a specific model version"""
    
    version: Literal["v4", "v5", "v5-server"] = Field(
        default="v5-server",
        description="Model version identifier"
    )
    
    det_path: Path = Field(
        description="Path to detection model"
    )
    
    rec_path: Path = Field(
        description="Path to recognition model"
    )
    
    # Shared components (same for all versions)
    cls_path: Path = Field(
        default=MODULE_DIR / "models/ppocrv5/cls/cls.onnx",
        description="Path to classification model (shared)"
    )
    
    dict_path: Path = Field(
        default=MODULE_DIR / "models/ppocrv5/ppocrv5_dict.txt",
        description="Path to character dictionary (shared)"
    )
    
    # Model parameters
    det_limit_side_len: float = Field(default=960, description="Detection limit side length")
    det_db_thresh: float = Field(default=0.3, ge=0.0, le=1.0, description="DB threshold")
    det_db_box_thresh: float = Field(default=0.6, ge=0.0, le=1.0, description="DB box threshold")
    det_db_unclip_ratio: float = Field(default=1.5, description="Unclip ratio")
    
    rec_image_shape: str = Field(default="3, 48, 320", description="Recognition image shape")
    rec_batch_num: int = Field(default=6, ge=1, description="Recognition batch number")
    
    cls_image_shape: str = Field(default="3, 48, 192", description="Classification image shape")
    cls_batch_num: int = Field(default=6, ge=1, description="Classification batch number")
    cls_thresh: float = Field(default=0.9, ge=0.0, le=1.0, description="Classification threshold")
    
    @validator('det_path', 'rec_path', 'cls_path', 'dict_path')
    def validate_paths(cls, v):
        """Validate that model paths exist"""
        if v and not Path(v).exists():
            # Don't raise error yet - model might need to be downloaded
            pass
        return Path(v)


class OCRConfig(BaseSettings):
    """Main OCR system configuration with environment variable support"""
    
    # Model settings
    default_model_version: Literal["v4", "v5", "v5-server"] = Field(
        default="v5-server",
        description="Default model version to use"
    )
    
    use_gpu: bool = Field(
        default=False, 
        description="Whether to use GPU acceleration"
    )
    
    use_angle_cls: bool = Field(
        default=True,
        description="Whether to use angle classification"
    )
    
    drop_score: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0,
        description="Confidence threshold for filtering results"
    )
    
    # Performance settings
    max_memory_gb: int = Field(
        default=8, 
        ge=1, 
        le=32,
        description="Maximum memory allocation in GB"
    )
    
    thread_pool_size: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Thread pool size for ONNX inference"
    )
    
    model_cache_size: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of models to keep in cache"
    )
    
    # Download settings
    download_timeout: int = Field(
        default=300,
        ge=60,
        le=1800,
        description="Download timeout in seconds"
    )
    
    download_retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retry attempts for downloads"
    )
    
    # Directory settings
    models_dir: Path = Field(
        default=MODULE_DIR / "models",
        description="Base directory for model files"
    )
    
    cache_dir: Path = Field(
        default=MODULE_DIR / ".cache",
        description="Cache directory for temporary files"
    )
    
    @computed_field
    @property
    def model_configs(self) -> Dict[str, ModelConfig]:
        """Get all model configurations"""
        return {
            "v4": ModelConfig(
                version="v4",
                det_path=self.models_dir / "ppocrv4/det/det.onnx",
                rec_path=self.models_dir / "ppocrv4/rec/rec.onnx"
            ),
            "v5": ModelConfig(
                version="v5", 
                det_path=self.models_dir / "ppocrv5/det/det.onnx",
                rec_path=self.models_dir / "ppocrv5/rec/rec.onnx"
            ),
            "v5-server": ModelConfig(
                version="v5-server",
                det_path=self.models_dir / "ppocrv5-server/det.onnx", 
                rec_path=self.models_dir / "ppocrv5-server/rec.onnx"
            )
        }
    
    @computed_field 
    @property
    def download_urls(self) -> Dict[str, str]:
        """URLs for downloading missing models"""
        return {
            "v5-server-det": "https://box.ygxz.in/f/d/MgCk/det.onnx",
            "v5-server-rec": "https://box.ygxz.in/f/d/Z9cl/rec.onnx"
        }
    
    def get_model_config(self, version: str) -> ModelConfig:
        """Get configuration for a specific model version"""
        if version not in self.model_configs:
            raise ValueError(f"Unknown model version: {version}")
        return self.model_configs[version]
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure model version directories exist
        for version in ["ppocrv4", "ppocrv5", "ppocrv5-server"]:
            version_dir = self.models_dir / version
            version_dir.mkdir(parents=True, exist_ok=True)
            
            for model_type in ["det", "rec", "cls"]:
                model_dir = version_dir / model_type
                model_dir.mkdir(parents=True, exist_ok=True)
    
    class Config:
        """Pydantic configuration"""
        env_prefix = "ONNX_OCR_"
        case_sensitive = False
        # Allow reading from environment variables
        env_file = ".env"


# Global configuration instance
config = OCRConfig()


class OCRRequest(BaseModel):
    """Enhanced OCR request model with model version support"""
    
    image: str = Field(
        ..., 
        description="Base64 encoded image data",
        min_length=1
    )
    
    model_version: Literal["v4", "v5", "v5-server"] = Field(
        default="v5-server",
        description="Model version to use for OCR processing"
    )
    
    use_angle_cls: bool = Field(
        default=True,
        description="Whether to use angle classification"
    )
    
    drop_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Raw confidence threshold for filtering results (internal model range)"
    )
    
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="User-friendly confidence threshold (0-1 normalized range). If provided, overrides drop_score."
    )
    
    det_limit_side_len: float = Field(
        default=960,
        gt=0,
        description="Maximum side length for detection"
    )
    
    batch_optimize: bool = Field(
        default=False,
        description="Enable batch processing optimization"
    )
    
    @validator('image')
    def validate_base64_image(cls, v):
        """Validate base64 image format"""
        import base64
        try:
            # Try to decode to validate format
            base64.b64decode(v)
        except Exception:
            raise ValueError("Invalid base64 image data")
        return v


class OCRResponse(BaseModel):
    """Enhanced OCR response model"""
    
    success: bool = Field(..., description="Whether OCR processing succeeded")
    results: list = Field(default=[], description="OCR detection and recognition results") 
    model_version: str = Field(..., description="Model version used")
    processing_time: float = Field(..., description="Processing time in seconds")
    image_info: Dict[str, Any] = Field(..., description="Input image information")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    
    # Error information (if success=False)
    error: Optional[str] = Field(None, description="Error message if processing failed")
    error_code: Optional[str] = Field(None, description="Error code if processing failed")


class ModelSwitchRequest(BaseModel):
    """Model switch request"""
    
    model_version: Literal["v4", "v5", "v5-server"] = Field(
        ...,
        description="Target model version to switch to"
    )