# Sintellix Python API

PyTorch-like Python interface for Sintellix Neural Network Framework with HOT (Higher Order Thought) Architecture.

## Features

- **PyTorch-like API**: Familiar interface for PyTorch users
- **CUDA Acceleration**: High-performance CUDA kernels for all operations
- **Advanced Architecture**: Multi-head attention, SSM, RWKV, temporal attention, DDPM, and more
- **Tiered Storage**: Automatic GPU→RAM→Disk memory management
- **VQ-GAN Codec**: Semantic encoding/decoding with vector quantization
- **Easy Training**: Built-in trainer with checkpointing and logging

## Installation

```bash
pip install sintellix
```

Or install from source:

```bash
git clone https://github.com/sintellix/sintellix.git
cd sintellix/python
pip install -e .
```

## Quick Start

### Basic Usage

```python
from sintellix import NeuronModel, NeuronConfig

# Create model
config = NeuronConfig(dim=256, grid_size=(32, 32, 32))
model = NeuronModel(config)
model.initialize()

# Forward pass
import torch
input_tensor = torch.randn(1, 256, 256)
output = model(input_tensor)
```

### Training

```python
from sintellix import Trainer, TrainingConfig

# Create trainer
train_config = TrainingConfig(
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)
trainer = Trainer(model, train_config)

# Train
trainer.train(train_loader, val_loader)
```

### Model Management

```python
from sintellix import download_model

# Download pretrained models
e5_path = download_model("e5-large")
vqgan_path = download_model("vqgan-codebook")
```

## Configuration

```python
config = NeuronConfig(
    dim=256,                    # Neuron dimension
    num_heads=8,                # Attention heads
    grid_size=(32, 32, 32),     # Neuron grid
    temporal_frames=8,          # Temporal history
    enable_multi_head=True,     # Enable modules
    enable_ssm=True,
    enable_rwkv=True,
    gpu_cache_size_mb=4096,     # Storage config
    ram_cache_size_mb=16384,
)
```

## License

MIT License - see [LICENSE](../LICENSE) for details
