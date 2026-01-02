"""
Utility functions for the Network Risk Path Detection project.
"""

import os
import random
import numpy as np
import torch
from typing import Optional

import config


def set_all_seeds(seed: int = config.RANDOM_SEED):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the best available device for computation.
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)


def print_gpu_memory():
    """Print GPU memory usage if available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory: {allocated:.1f} MB allocated, {cached:.1f} MB cached")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def normalize_to_range(values: np.ndarray, new_min: float = 0, new_max: float = 1) -> np.ndarray:
    """
    Normalize values to a specific range.
    
    Args:
        values: Array of values
        new_min: New minimum value
        new_max: New maximum value
        
    Returns:
        Normalized array
    """
    old_min = values.min()
    old_max = values.max()
    
    if old_max - old_min == 0:
        return np.full_like(values, (new_min + new_max) / 2)
    
    return (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
