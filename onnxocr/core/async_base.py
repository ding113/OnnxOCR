"""
Async ONNX prediction base class with modern Python features
"""

import asyncio
import threading
import onnxruntime
import structlog
from typing import Dict, List, Optional, Any, Tuple
from contextlib import asynccontextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .exceptions import ModelLoadError, InferenceError

logger = structlog.get_logger()


class AsyncPredictBase:
    """
    Modern async base class for ONNX model prediction
    
    Features:
    - Fully async model loading and inference
    - Type annotations throughout
    - Context manager support for resource cleanup
    - Thread pool optimization for CPU inference
    - Structured logging integration
    - Proper exception handling
    """
    
    def __init__(
        self, 
        model_path: Optional[Path] = None,
        use_gpu: bool = False,
        thread_pool_size: int = 4
    ):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.thread_pool_size = thread_pool_size
        
        self.session: Optional[onnxruntime.InferenceSession] = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        
        # Thread pool for CPU-bound ONNX inference
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._is_initialized = False
        
        self.logger = logger.bind(
            model_class=self.__class__.__name__,
            model_path=str(model_path) if model_path else None
        )
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        await self.cleanup()
    
    async def initialize(self) -> None:
        """
        Initialize the ONNX session and thread pool
        """
        if self._is_initialized:
            return
            
        try:
            self.logger.info("Initializing ONNX model", model_path=str(self.model_path))
            
            # Create thread pool for CPU inference
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self.thread_pool_size,
                thread_name_prefix=f"{self.__class__.__name__}-pool"
            )
            
            # Load ONNX session in thread pool to avoid blocking
            self.session = await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                self._load_onnx_session,
                self.model_path
            )
            
            # Get input/output names
            self.input_names = [node.name for node in self.session.get_inputs()]
            self.output_names = [node.name for node in self.session.get_outputs()]
            
            self._is_initialized = True
            
            self.logger.info(
                "ONNX model initialized successfully",
                input_names=self.input_names,
                output_names=self.output_names,
                providers=self.session.get_providers()
            )
            
        except Exception as e:
            self.logger.error("Failed to initialize ONNX model", error=str(e))
            raise ModelLoadError(f"Failed to load model {self.model_path}: {e}") from e
    
    def _load_onnx_session(self, model_path: Path) -> onnxruntime.InferenceSession:
        """
        Load ONNX session (runs in thread pool)
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Configure providers
        if self.use_gpu:
            providers = [
                ('CUDAExecutionProvider', {
                    "cudnn_conv_algo_search": "DEFAULT"
                }),
                'CPUExecutionProvider'
            ]
        else:
            providers = ['CPUExecutionProvider']
        
        # Create session with optimizations
        sess_options = onnxruntime.SessionOptions()
        sess_options.inter_op_num_threads = self.thread_pool_size
        sess_options.intra_op_num_threads = self.thread_pool_size
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        
        return onnxruntime.InferenceSession(
            str(model_path), 
            sess_options=sess_options,
            providers=providers
        )
    
    async def predict_async(
        self, 
        input_data: Dict[str, Any]
    ) -> List[Any]:
        """
        Perform async inference
        
        Args:
            input_data: Dictionary mapping input names to numpy arrays
            
        Returns:
            List of output arrays from the model
        """
        if not self._is_initialized:
            await self.initialize()
        
        if self.session is None:
            raise InferenceError("Model not initialized")
        
        
        try:
            # Run inference in thread pool to avoid blocking event loop
            import time
            inference_start = time.time()
            
            outputs = await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                self._run_inference,
                input_data
            )
            
            inference_time = time.time() - inference_start
            
            
            return outputs
            
        except Exception as e:
            self.logger.error("[ONNX] Inference failed", error=str(e), model_path=str(self.model_path))
            raise InferenceError(f"Prediction failed: {e}") from e
    
    def _run_inference(self, input_data: Dict[str, Any]) -> List[Any]:
        """
        Run ONNX inference (executes in thread pool)
        """
        try:
            result = self.session.run(self.output_names, input_data)
            
            return result
            
        except Exception as e:
            self.logger.error(
                "[ONNX] Raw inference execution failed",
                error=str(e),
                error_type=type(e).__name__,
                input_keys=list(input_data.keys()),
                expected_inputs=self.input_names
            )
            raise
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata information
        """
        if not self._is_initialized:
            await self.initialize()
            
        return {
            "model_path": str(self.model_path),
            "input_names": self.input_names,
            "output_names": self.output_names,
            "providers": self.session.get_providers() if self.session else [],
            "use_gpu": self.use_gpu,
            "thread_pool_size": self.thread_pool_size,
            "is_initialized": self._is_initialized
        }
    
    async def cleanup(self) -> None:
        """
        Clean up resources
        """
        self.logger.info("Cleaning up model resources")
        
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
        
        if self.session:
            # ONNX runtime session cleanup happens automatically
            self.session = None
        
        self._is_initialized = False
        
        self.logger.info("Model resources cleaned up")
    
    def __del__(self):
        """
        Ensure cleanup on deletion
        """
        if self._is_initialized:
            # Note: We can't await in __del__, so we just shutdown synchronously
            if self._thread_pool:
                self._thread_pool.shutdown(wait=False)


@asynccontextmanager
async def create_predictor(
    predictor_class: type,
    model_path: Path,
    **kwargs
) -> AsyncPredictBase:
    """
    Context manager factory for creating predictors with automatic cleanup
    
    Usage:
        async with create_predictor(MyPredictor, model_path) as predictor:
            result = await predictor.predict_async(input_data)
    """
    predictor = predictor_class(model_path=model_path, **kwargs)
    try:
        await predictor.initialize()
        yield predictor
    finally:
        await predictor.cleanup()