"""
Utility functions for Sintellix
"""

import torch
import numpy as np
from typing import Optional, Tuple
import logging


def setup_logging(level: str = "INFO"):
    """
    Setup logging configuration

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get torch device

    Args:
        device: Device string ("cuda", "cpu", or None for auto)

    Returns:
        torch.device object
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def count_parameters(model) -> int:
    """
    Count total number of parameters in model

    Args:
        model: PyTorch model

    Returns:
        Total parameter count
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seed_everything(seed: int = 42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def format_bytes(bytes_val: int) -> str:
    """
    Format bytes to human-readable string

    Args:
        bytes_val: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"
