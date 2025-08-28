"""
Classification postprocessing for text angle detection
文本角度检测分类后处理模块
"""

import numpy as np
from typing import List, Tuple, Any, Optional


class ClsPostProcess:
    """
    Classification postprocessing for text angle detection
    文本角度检测分类后处理
    """
    
    def __init__(self):
        """Initialize classification postprocessing"""
        # Angle labels: ['0', '180'] for 0 and 180 degree rotation
        self.label_list = ['0', '180']
    
    def __call__(
        self, 
        preds: np.ndarray, 
        label_list: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Process classification predictions
        
        Args:
            preds: Classification predictions (logits) with shape [batch_size, num_classes]
            label_list: List of class labels (optional, uses default if None)
            
        Returns:
            List of (label, confidence) tuples for each prediction
        """
        if label_list is None:
            label_list = self.label_list
        
        # Handle different input formats
        if isinstance(preds, list):
            preds = np.array(preds)
        
        # Ensure 2D array
        if len(preds.shape) == 1:
            preds = preds.reshape(1, -1)
        
        # Apply softmax to get probabilities
        pred_probs = self._softmax(preds)
        
        # Get predicted classes and their probabilities
        pred_idxs = np.argmax(pred_probs, axis=1)
        max_probs = np.max(pred_probs, axis=1)
        
        results = []
        for idx, prob in zip(pred_idxs, max_probs):
            if 0 <= idx < len(label_list):
                label = label_list[idx]
            else:
                # Fallback for out-of-bounds indices
                label = '0'
                
            results.append((label, float(prob)))
        
        return results
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Apply softmax activation function
        
        Args:
            x: Input logits
            
        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)