"""
Sintellix - Neural Network Framework with HOT Architecture

A PyTorch-like framework for building and training neural networks
with Higher Order Thought (HOT) architecture.
"""

__version__ = "0.1.0"

# Import core components
from .core import (
    NeuronModel,
    NeuronConfig,
    Neuron,
)

# Import training utilities
from .training import (
    Trainer,
    TrainingConfig,
)

# Import model management
from .models import (
    ModelManager,
    download_model,
)

__all__ = [
    # Core
    "NeuronModel",
    "NeuronConfig",
    "Neuron",
    # Training
    "Trainer",
    "TrainingConfig",
    # Models
    "ModelManager",
    "download_model",
]
