"""
Smart model manager with async loading, caching, and dynamic switching
"""

import asyncio
import time
import weakref
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Set
from collections import OrderedDict
import structlog

from .async_base import AsyncPredictBase
from .config import config, ModelConfig
from .downloader import ModelDownloader  
from .exceptions import ModelNotFoundError, ModelLoadError

logger = structlog.get_logger()


class ModelSession:
    """Represents a loaded ONNX model session with metadata"""
    
    def __init__(
        self,
        model_type: str,  # 'det' or 'rec'
        session: AsyncPredictBase,
        model_config: ModelConfig,
        load_time: float
    ):
        self.model_type = model_type
        self.session = session
        self.model_config = model_config
        self.load_time = load_time
        self.last_used = time.time()
        self.use_count = 0
    
    def mark_used(self):
        """Mark this session as recently used"""
        self.last_used = time.time()
        self.use_count += 1


class SmartModelManager:
    """
    Intelligent model manager with features:
    - Async model loading and unloading
    - LRU cache with memory management
    - Automatic download of missing models
    - Dynamic model switching (det/rec only, cls shared)
    - Thread-safe operations
    """
    
    def __init__(self, max_cache_size: Optional[int] = None):
        self.max_cache_size = max_cache_size or config.model_cache_size
        self.downloader = ModelDownloader(
            timeout=config.download_timeout,
            retry_attempts=config.download_retry_attempts
        )
        
        # Cache for loaded model sessions
        # Key: (model_version, model_type) -> ModelSession
        self._model_cache: OrderedDict[Tuple[str, str], ModelSession] = OrderedDict()
        
        # Shared classifier session (same for all versions)
        self._shared_cls_session: Optional[ModelSession] = None
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        
        # Track which models are currently being loaded
        self._loading_models: Set[Tuple[str, str]] = set()
        
        self.logger = logger.bind(component="SmartModelManager")
        
        # Ensure directories exist
        config.ensure_directories()
    
    async def get_model_sessions(
        self, 
        model_version: str
    ) -> Dict[str, AsyncPredictBase]:
        """
        Get loaded model sessions for a specific version
        
        Args:
            model_version: Model version ('v4', 'v5', 'v5-server')
            
        Returns:
            Dict containing 'det', 'rec', and 'cls' sessions
        """
        async with self._lock:
            model_config = config.get_model_config(model_version)
            
            # Ensure models are available (download if needed)
            await self._ensure_models_available(model_version)
            
            # Get or load detection and recognition models
            det_session = await self._get_or_load_model(model_version, "det")
            rec_session = await self._get_or_load_model(model_version, "rec") 
            
            # Get or load shared classifier (only once)
            cls_session = await self._get_or_load_shared_cls()
            
            return {
                "det": det_session.session,
                "rec": rec_session.session,
                "cls": cls_session.session
            }
    
    async def _ensure_models_available(self, model_version: str) -> None:
        """
        Ensure model files are available, download if missing
        """
        model_config = config.get_model_config(model_version)
        
        # Check and download missing models
        missing_models = []
        
        if not model_config.det_path.exists():
            missing_models.append(("det", model_config.det_path))
        
        if not model_config.rec_path.exists():
            missing_models.append(("rec", model_config.rec_path))
        
        if missing_models:
            self.logger.info(
                "Missing model files detected, starting download",
                model_version=model_version,
                missing_count=len(missing_models)
            )
            
            for model_type, model_path in missing_models:
                await self._download_missing_model(model_version, model_type, model_path)
    
    async def _download_missing_model(
        self, 
        model_version: str, 
        model_type: str, 
        target_path: Path
    ) -> None:
        """
        Download a missing model file
        """
        # Only v5-server models need to be downloaded
        if model_version == "v5-server":
            download_key = f"v5-server-{model_type}"
            if download_key in config.download_urls:
                url = config.download_urls[download_key]
                
                self.logger.info(
                    "Downloading missing model",
                    model_version=model_version,
                    model_type=model_type,
                    url=url
                )
                
                progress_callback = self.downloader.create_progress_callback(
                    f"{model_version} {model_type} model"
                )
                
                success = await self.downloader.download_model(
                    url=url,
                    target_path=target_path,
                    progress_callback=progress_callback,
                    verify_integrity=True
                )
                
                if not success:
                    raise ModelLoadError(
                        f"Failed to download {model_version} {model_type} model",
                        str(target_path)
                    )
            else:
                raise ModelNotFoundError(
                    f"No download URL available for {model_version} {model_type}",
                    model_version
                )
        else:
            raise ModelNotFoundError(
                f"Model file not found and cannot be downloaded: {target_path}",
                model_version
            )
    
    async def _get_or_load_model(
        self, 
        model_version: str, 
        model_type: str
    ) -> ModelSession:
        """
        Get cached model session or load new one
        """
        cache_key = (model_version, model_type)
        
        # Check if model is in cache
        if cache_key in self._model_cache:
            session = self._model_cache[cache_key]
            session.mark_used()
            # Move to end (most recently used)
            self._model_cache.move_to_end(cache_key)
            return session
        
        # Check if model is currently being loaded
        if cache_key in self._loading_models:
            self.logger.info("Model is already being loaded, waiting...")
            while cache_key in self._loading_models:
                await asyncio.sleep(0.1)
            # Should be in cache now
            return self._model_cache[cache_key]
        
        # Load new model
        self._loading_models.add(cache_key)
        
        try:
            # Import model classes here to avoid circular imports
            from ..models import AsyncTextDetector, AsyncTextRecognizer, AsyncTextClassifier
            
            model_config = config.get_model_config(model_version)
            
            if model_type == "det":
                model_path = model_config.det_path
                predictor_class = AsyncTextDetector
            elif model_type == "rec":  
                model_path = model_config.rec_path
                predictor_class = AsyncTextRecognizer
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.logger.info(
                "Loading model",
                model_version=model_version,
                model_type=model_type,
                model_path=str(model_path),
                model_exists=model_path.exists(),
                model_size_mb=round(model_path.stat().st_size / 1024 / 1024, 2) if model_path.exists() else None
            )
            
            
            start_time = time.time()
            
            # Create and initialize predictor
            if model_type == "rec":
                predictor = predictor_class(
                    model_path=model_path,
                    model_config=model_config,  # Pass model config to recognizer
                    use_gpu=config.use_gpu,
                    thread_pool_size=config.thread_pool_size
                )
            else:
                predictor = predictor_class(
                    model_path=model_path,
                    use_gpu=config.use_gpu,
                    thread_pool_size=config.thread_pool_size
                )
            
            await predictor.initialize()
            
            load_time = time.time() - start_time
            
            # Create session wrapper
            session = ModelSession(
                model_type=model_type,
                session=predictor,
                model_config=model_config,
                load_time=load_time
            )
            
            # Add to cache
            self._model_cache[cache_key] = session
            
            # Ensure cache size limit
            await self._enforce_cache_limit()
            
            self.logger.info(
                "Model loaded successfully",
                model_version=model_version,
                model_type=model_type,
                load_time=load_time,
                cache_size=len(self._model_cache)
            )
            
            return session
            
        finally:
            self._loading_models.discard(cache_key)
    
    async def _get_or_load_shared_cls(self) -> ModelSession:
        """
        Get or load shared classifier model (same for all versions)
        """
        if self._shared_cls_session is not None:
            self._shared_cls_session.mark_used()
            return self._shared_cls_session
        
        # Import model classes here to avoid circular imports
        from ..models import AsyncTextClassifier
        
        # Load shared classifier
        cls_path = config.get_model_config("v5").cls_path  # Use v5 cls (same for all)
        
        self.logger.info("Loading shared classifier model", model_path=str(cls_path))
        
        start_time = time.time()
        
        predictor = AsyncTextClassifier(
            model_path=cls_path,
            use_gpu=config.use_gpu,
            thread_pool_size=config.thread_pool_size
        )
        
        await predictor.initialize()
        
        load_time = time.time() - start_time
        
        self._shared_cls_session = ModelSession(
            model_type="cls",
            session=predictor,
            model_config=config.get_model_config("v5"),  # Use v5 config
            load_time=load_time
        )
        
        self.logger.info("Shared classifier loaded successfully", load_time=load_time)
        
        return self._shared_cls_session
    
    async def _enforce_cache_limit(self) -> None:
        """
        Enforce cache size limit by removing least recently used models
        """
        while len(self._model_cache) > self.max_cache_size:
            # Remove least recently used (first item in OrderedDict)
            cache_key, session = self._model_cache.popitem(last=False)
            
            self.logger.info(
                "Evicting model from cache",
                model_version=cache_key[0],
                model_type=cache_key[1],
                use_count=session.use_count
            )
            
            # Clean up model session
            try:
                await session.session.cleanup()
            except Exception as e:
                self.logger.warning("Error during model cleanup", error=str(e))
    
    async def preload_model(self, model_version: str) -> None:
        """
        Preload models for a specific version
        """
        self.logger.info("Preloading models", model_version=model_version)
        await self.get_model_sessions(model_version)
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about current cache state
        """
        async with self._lock:
            cache_info = {
                "cache_size": len(self._model_cache),
                "max_cache_size": self.max_cache_size,
                "loaded_models": [],
                "shared_cls_loaded": self._shared_cls_session is not None,
                "currently_loading": len(self._loading_models)
            }
            
            for (version, model_type), session in self._model_cache.items():
                cache_info["loaded_models"].append({
                    "version": version,
                    "type": model_type,
                    "load_time": session.load_time,
                    "use_count": session.use_count,
                    "last_used": session.last_used
                })
            
            return cache_info
    
    async def clear_cache(self) -> None:
        """
        Clear all cached models
        """
        async with self._lock:
            self.logger.info("Clearing model cache")
            
            # Clean up all cached models
            for session in self._model_cache.values():
                try:
                    await session.session.cleanup()
                except Exception as e:
                    self.logger.warning("Error during cache cleanup", error=str(e))
            
            self._model_cache.clear()
            
            # Clean up shared classifier
            if self._shared_cls_session:
                try:
                    await self._shared_cls_session.session.cleanup()
                except Exception as e:
                    self.logger.warning("Error cleaning shared classifier", error=str(e))
                self._shared_cls_session = None
            
            self.logger.info("Model cache cleared")