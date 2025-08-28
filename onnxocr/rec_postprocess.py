"""
Recognition postprocessing with CTC (Connectionist Temporal Classification) decoding
文本识别CTC解码后处理模块
"""

import numpy as np
from typing import List, Tuple, Optional, Any
from pathlib import Path


class CTCLabelDecode:
    """
    CTC label decoder for text recognition
    文本识别CTC标签解码器
    """
    
    def __init__(
        self,
        character_dict_path: Optional[str] = None,
        use_space_char: bool = True
    ):
        """
        Initialize CTC label decoder
        
        Args:
            character_dict_path: Path to character dictionary file
            use_space_char: Whether to include space character
        """
        self.use_space_char = use_space_char
        self.character_str = ""
        
        # Load character dictionary
        if character_dict_path is not None and Path(character_dict_path).exists():
            self.character_str = self._load_char_dict(character_dict_path)
        else:
            # Default character set (basic Latin + space)
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        # Add space character if requested
        if self.use_space_char:
            self.character_str = " " + self.character_str
        
        # Create character list (blank token at index 0)
        self.character = ["blank"] + list(self.character_str)
        
        # Create character-to-index mapping
        self.dict = {char: idx for idx, char in enumerate(self.character)}
    
    def _load_char_dict(self, dict_path: str) -> str:
        """
        Load character dictionary from file
        
        Args:
            dict_path: Path to dictionary file
            
        Returns:
            Character string
        """
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Extract characters, skip empty lines and strip whitespace
            chars = []
            for line in lines:
                line = line.strip()
                if line:  # Skip empty lines
                    chars.append(line)
            
            return ''.join(chars)
            
        except Exception as e:
            print(f"Warning: Failed to load dictionary from {dict_path}: {e}")
            # Fallback to basic character set
            return "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    def __call__(self, preds: np.ndarray, label: Optional[Any] = None) -> List[Tuple[str, float]]:
        """
        Decode CTC predictions to text
        
        Args:
            preds: CTC predictions with shape [batch_size, seq_len, num_classes]
            label: Ground truth labels (not used in inference, kept for compatibility)
            
        Returns:
            List of (text, confidence) tuples
        """
        if isinstance(preds, list):
            preds = np.array(preds)
        
        # Ensure 3D array: [batch_size, seq_len, num_classes]
        if len(preds.shape) == 2:
            preds = preds.reshape(1, preds.shape[0], preds.shape[1])
        
        # Apply softmax to get probabilities
        preds_prob = self._softmax(preds)
        
        # Decode each sequence in the batch
        results = []
        for batch_idx in range(preds_prob.shape[0]):
            pred_prob = preds_prob[batch_idx]  # [seq_len, num_classes]
            
            # CTC greedy decoding
            text, confidence = self._ctc_greedy_decode(pred_prob)
            results.append((text, confidence))
        
        return results
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Apply softmax activation
        
        Args:
            x: Input logits
            
        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)
    
    def _ctc_greedy_decode(self, pred_prob: np.ndarray) -> Tuple[str, float]:
        """
        CTC greedy decoding algorithm
        
        Args:
            pred_prob: Prediction probabilities [seq_len, num_classes]
            
        Returns:
            Tuple of (decoded_text, confidence)
        """
        # Get most probable character at each time step
        pred_indices = np.argmax(pred_prob, axis=1)
        confidence_scores = np.max(pred_prob, axis=1)
        
        # CTC decoding: remove blanks and consecutive duplicates
        decoded_chars = []
        decoded_probs = []
        
        prev_idx = -1
        for idx, prob in zip(pred_indices, confidence_scores):
            # Skip blank tokens (index 0)
            if idx == 0:
                continue
            
            # Skip consecutive duplicates
            if idx == prev_idx:
                continue
            
            # Convert index to character
            if 0 < idx < len(self.character):
                decoded_chars.append(self.character[idx])
                decoded_probs.append(prob)
                prev_idx = idx
        
        # Join characters to form text
        text = ''.join(decoded_chars)
        
        # Calculate average confidence
        if decoded_probs:
            confidence = float(np.mean(decoded_probs))
        else:
            confidence = 0.0
        
        return text, confidence
    
    def add_special_char(self, dict_character: List[str]) -> List[str]:
        """
        Add special characters to dictionary (for compatibility)
        
        Args:
            dict_character: Character list
            
        Returns:
            Updated character list with special characters
        """
        dict_character = ["blank"] + dict_character
        return dict_character
    
    def get_ignored_tokens(self) -> List[int]:
        """
        Get list of ignored token indices (blank tokens)
        
        Returns:
            List of ignored token indices
        """
        return [0]  # Blank token at index 0