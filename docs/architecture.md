# Sintellix Architecture

## Overview

Sintellix is a CUDA-based neural network framework featuring a novel 3D grid neuron architecture with semantic encoding/decoding capabilities. This document describes the core architectural components and their interactions.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  (Python/C++ API, Training Scripts, Inference Pipelines)    │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                   Sintellix Core API                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ NeuronModel  │  │ ConfigManager│  │  KFEManager  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                  Neuron Grid Layer                           │
│  ┌────────────────────────────────────────────────────┐    │
│  │  3D Grid (32×32×32 neurons)                        │    │
│  │  - Multi-head Attention (8 heads)                  │    │
│  │  - Temporal Attention (8 frames)                   │    │
│  │  - Global Aggregation (sparse)                     │    │
│  │  - FXAA-like Auxiliary Layer                       │    │
│  └────────────────────────────────────────────────────┘    │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                  Semantic Codec Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Encoder    │  │  VQ-GAN      │  │   Decoder    │     │
│  │ (E5/CLIP)    │  │  Quantizer   │  │ (Transformer)│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                  Storage & Memory Layer                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │ GPU Cache  │→ │ RAM Cache  │→ │ Disk Cache │           │
│  │  (8GB)     │  │  (32GB)    │  │ (Unlimited)│           │
│  └────────────┘  └────────────┘  └────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Neuron Class (`src/core/neuron.cu`)

The fundamental computational unit in Sintellix. Each neuron maintains:

- **State Vector**: High-dimensional representation (128/256/512/1024-dim)
- **Multi-head Attention**: 8 attention heads for enhanced representation learning
- **Temporal Context**: 8-frame history for temporal modeling
- **Adam Optimizer**: Per-neuron adaptive learning rates with momentum

**Key Features:**
- Forward propagation with multi-head self-attention
- Backward propagation with gradient computation
- Adam optimizer with gradient clipping
- Adaptive noise filtering using EMA thresholds

**CUDA Kernels:**
- `multi_head_attention_kernel`: Parallel multi-head attention computation
- `temporal_attention_kernel`: Temporal context aggregation
- `adam_update_kernel`: Parameter updates with Adam optimizer
- `gradient_clip_kernel`: Gradient clipping for stability

### 2. NeuronModel Class (`src/core/neuron_model.cu`)

Manages the entire 3D grid of neurons and orchestrates training/inference.

**Responsibilities:**
- Initialize and manage 32,768 neurons (32³ grid)
- Coordinate forward/backward passes across all neurons
- Manage global aggregation for cross-neuron communication
- Handle model serialization and checkpointing

**Key Methods:**
```cpp
void forward(const double* input, double* output);
void backward(const double* grad_output, double* grad_input);
void update_parameters(float learning_rate);
bool save_state(const std::string& path);
bool load_state(const std::string& path);
```

### 3. Knowledge Feature Encoding (KFE) Manager

Manages semantic knowledge storage and retrieval:

- **Slot-based Storage**: Up to 10,000 knowledge feature slots
- **Hash-based Lookup**: Fast O(1) knowledge retrieval
- **Compression**: Zstd compression for efficient storage
- **Persistence**: Protobuf serialization for state management

### 4. Tiered Storage System (`src/storage/tiered_storage.cu`)

Hierarchical caching system for large-scale models:

**Tier 1 - GPU Memory:**
- Fastest access (microseconds)
- Limited capacity (~8GB default)
- Hot data: frequently accessed neurons

**Tier 2 - System RAM:**
- Fast access (milliseconds)
- Medium capacity (~32GB default)
- Warm data: occasionally accessed neurons

**Tier 3 - Disk Storage:**
- Slower access (seconds)
- Unlimited capacity
- Cold data: rarely accessed neurons

**Eviction Policy:**
- LRU (Least Recently Used) for GPU → RAM
- LRU for RAM → Disk
- Automatic promotion on access

### 5. VQ-GAN Codec (`src/codec/vqgan.cu`)

Vector-quantized GAN for semantic compression:

**Encoder:**
- Converts multi-modal inputs to semantic space
- E5-Large for text (1024-dim)
- CLIP for images (1024-dim)
- Wav2Vec2 for audio (1024-dim)

**Quantizer:**
- Codebook size: 8192 entries
- Quantization dimension: 1024
- Commitment loss weight: 0.25

**Decoder:**
- Autoregressive transformer
- Reconstructs original modality from codes
- Cross-attention for multi-modal fusion

## Data Flow

### Training Pipeline

```
1. Input Data (Text/Image/Audio)
   ↓
2. Encoder → Semantic Embedding (1024-dim)
   ↓
3. VQ-GAN Quantizer → Discrete Codes
   ↓
4. NeuronModel.forward()
   - Multi-head attention across neurons
   - Temporal attention over history
   - Global aggregation (sparse)
   - FXAA auxiliary prediction
   ↓
5. Loss Computation
   - Reconstruction loss
   - VQ commitment loss
   - Auxiliary losses
   ↓
6. NeuronModel.backward()
   - Gradient computation
   - Backpropagation through attention
   ↓
7. NeuronModel.update_parameters()
   - Adam optimizer updates
   - Gradient clipping
   - Noise filtering
   ↓
8. Checkpoint (every N epochs)
   - Protobuf serialization
   - Zstd compression
   - Disk persistence
```

### Inference Pipeline

```
1. Input Data
   ↓
2. Encoder → Semantic Embedding
   ↓
3. VQ-GAN Quantizer → Codes
   ↓
4. NeuronModel.forward() (no gradient)
   ↓
5. VQ-GAN Decoder → Output
```

## Memory Management

### GPU Memory Layout

```
┌─────────────────────────────────────┐
│  Neuron States (32³ × dim × 8 bytes)│  ~256MB (dim=256)
├─────────────────────────────────────┤
│  Attention Weights (8 heads)        │  ~128MB
├─────────────────────────────────────┤
│  Temporal History (8 frames)        │  ~2GB
├─────────────────────────────────────┤
│  Gradients & Optimizer States       │  ~512MB
├─────────────────────────────────────┤
│  VQ-GAN Codebook                    │  ~64MB
├─────────────────────────────────────┤
│  Working Memory (kernels)           │  ~1GB
└─────────────────────────────────────┘
Total: ~4GB (dim=256, batch=32)
```

### Memory Optimization Strategies

1. **Gradient Checkpointing**: Recompute activations during backward pass
2. **Mixed Precision**: FP16 for forward, FP32 for gradients
3. **Sparse Attention**: Only compute attention for nearby neurons
4. **Tiered Storage**: Offload cold neurons to RAM/Disk

## Parallelization Strategy

### CUDA Grid Configuration

- **Block Size**: 256 threads (optimal for modern GPUs)
- **Grid Size**: Dynamically calculated based on neuron count
- **Shared Memory**: Used for attention weight reduction
- **Warp-level Primitives**: Shuffle operations for efficient reduction

### Multi-GPU Support (Planned)

- **Data Parallelism**: Split batch across GPUs
- **Model Parallelism**: Split neuron grid across GPUs
- **Pipeline Parallelism**: Split layers across GPUs
- **NCCL Communication**: Efficient gradient synchronization

## Configuration System

### JSON Configuration

```json
{
  "neuron": {
    "dim": 256,
    "num_heads": 8,
    "grid_size": [32, 32, 32],
    "temporal_frames": 8
  },
  "optimizer": {
    "learning_rate": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "gradient_clip": 1.0
  },
  "modules": {
    "enable_multi_head": true,
    "enable_global_aggregation": true,
    "enable_noise_filter": true,
    "enable_temporal_attention": true,
    "enable_fxaa_layer": true
  },
  "storage": {
    "gpu_cache_size_mb": 8192,
    "ram_cache_size_mb": 32768,
    "disk_cache_path": "/tmp/sintellix_cache"
  }
}
```

### Protobuf Configuration

Binary format for efficient serialization:
- `proto/neuron_config.proto`: Configuration schema
- `proto/model_state.proto`: Model state schema

## Performance Characteristics

### Computational Complexity

- **Forward Pass**: O(N² × H × D) where N=neurons, H=heads, D=dim
- **Backward Pass**: O(N² × H × D)
- **Memory**: O(N × D + N² × H) for attention weights

### Optimization Techniques

1. **Fast Math**: CUDA `--use_fast_math` for 2-3x speedup
2. **Kernel Fusion**: Combine operations to reduce memory bandwidth
3. **Asynchronous Execution**: Overlap compute and memory transfers
4. **Persistent Kernels**: Keep kernels resident for repeated calls

## Extensibility

### Adding Custom Modules

1. Implement module in `src/core/custom_module.cu`
2. Add configuration in `proto/neuron_config.proto`
3. Integrate in `NeuronModel::forward()`
4. Add Python bindings in `python/bindings.cpp`

### Custom Loss Functions

1. Implement loss in `src/training/custom_loss.cu`
2. Compute gradients in backward pass
3. Register loss in training configuration

## References

- Multi-head Attention: Vaswani et al., "Attention Is All You Need" (2017)
- VQ-GAN: Esser et al., "Taming Transformers" (2021)
- Adam Optimizer: Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014)
- Tiered Storage: Inspired by OS virtual memory systems
