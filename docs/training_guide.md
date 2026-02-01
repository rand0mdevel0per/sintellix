# Training Guide

## Overview

This guide covers training Sintellix models from scratch, fine-tuning, and optimization strategies.

## Basic Training Loop

### Python Training Script

```python
import sintellix
import numpy as np

# Load configuration
config_mgr = sintellix.ConfigManager()
config_mgr.load_from_json("config.json")
config = config_mgr.get_config()

# Initialize model
model = sintellix.NeuronModel(config)
model.initialize()

# Training parameters
epochs = 100
batch_size = 32
learning_rate = 0.001

# Training loop
for epoch in range(epochs):
    epoch_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model.forward(inputs)

        # Compute loss
        loss = compute_loss(outputs, targets)
        epoch_loss += loss

        # Compute gradients
        grad_outputs = compute_gradients(loss, outputs)

        # Backward pass
        grad_inputs = model.backward(grad_outputs)

        # Update parameters
        model.update_parameters(learning_rate)

    # Log progress
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        model.save_state(f"checkpoint_epoch_{epoch+1}.pb")
```

## Loss Functions

### Reconstruction Loss

```python
def reconstruction_loss(output, target):
    """MSE loss for reconstruction"""
    return np.mean((output - target) ** 2)
```

### VQ Commitment Loss

```python
def vq_commitment_loss(embeddings, quantized, beta=0.25):
    """VQ-GAN commitment loss"""
    return beta * np.mean((embeddings - quantized) ** 2)
```

### Combined Loss

```python
def compute_total_loss(output, target, embeddings, quantized):
    recon_loss = reconstruction_loss(output, target)
    commit_loss = vq_commitment_loss(embeddings, quantized)
    return recon_loss + commit_loss
```

## Optimization Strategies

### Learning Rate Scheduling

```python
def get_learning_rate(epoch, initial_lr=0.001):
    """Cosine annealing schedule"""
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))

# In training loop
lr = get_learning_rate(epoch)
model.update_parameters(lr)
```

### Gradient Clipping

Gradient clipping is built into the model. Configure in `config.json`:

```json
{
  "optimizer": {
    "gradient_clip": 1.0
  }
}
```

### Mixed Precision Training

For faster training with lower memory usage:

```python
# Use FP16 for forward pass, FP32 for gradients
inputs_fp16 = inputs.astype(np.float16)
outputs = model.forward(inputs_fp16)
```

## Checkpointing

### Save Checkpoints

```python
# Save every N epochs
if epoch % checkpoint_interval == 0:
    checkpoint_path = f"checkpoints/model_epoch_{epoch}.pb"
    model.save_state(checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
```

### Resume Training

```python
# Load checkpoint
model.load_state("checkpoints/model_epoch_50.pb")
print("Resumed from epoch 50")

# Continue training
for epoch in range(50, 100):
    # Training loop...
```

## Monitoring Training

### Track Metrics

```python
import json

metrics = {
    "epoch": [],
    "loss": [],
    "learning_rate": []
}

for epoch in range(epochs):
    # Training...

    metrics["epoch"].append(epoch)
    metrics["loss"].append(avg_loss)
    metrics["learning_rate"].append(lr)

    # Save metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
```

### Visualize Training

```python
import matplotlib.pyplot as plt

plt.plot(metrics["epoch"], metrics["loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("training_loss.png")
```

## Advanced Techniques

### Data Augmentation

```python
def augment_data(inputs):
    """Apply random augmentation"""
    # Add noise
    noise = np.random.randn(*inputs.shape) * 0.01
    inputs_aug = inputs + noise

    # Random scaling
    scale = np.random.uniform(0.9, 1.1)
    inputs_aug *= scale

    return inputs_aug

# In training loop
inputs_aug = augment_data(inputs)
outputs = model.forward(inputs_aug)
```

### Early Stopping

```python
best_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(epochs):
    # Training...

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        model.save_state("best_model.pb")
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

### Distributed Training (Planned)

Multi-GPU training support is planned for future releases:

```python
# Future API
model = sintellix.NeuronModel(config, devices=[0, 1, 2, 3])
model.initialize()
```

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Range | Default | Impact |
|-----------|-------|---------|--------|
| learning_rate | 1e-5 to 1e-2 | 0.001 | Convergence speed |
| dim | 128/256/512/1024 | 256 | Model capacity |
| num_heads | 4/8/16 | 8 | Attention quality |
| gradient_clip | 0.5 to 2.0 | 1.0 | Training stability |

### Grid Search Example

```python
learning_rates = [0.0001, 0.001, 0.01]
dims = [128, 256, 512]

best_config = None
best_loss = float('inf')

for lr in learning_rates:
    for dim in dims:
        # Update config
        config.set_dim(dim)

        # Train model
        model = sintellix.NeuronModel(config)
        model.initialize()

        final_loss = train(model, lr)

        if final_loss < best_loss:
            best_loss = final_loss
            best_config = (lr, dim)

print(f"Best config: lr={best_config[0]}, dim={best_config[1]}")
```

## Troubleshooting

### Loss Not Decreasing

- Check learning rate (try reducing by 10x)
- Verify gradients are flowing (check gradient norms)
- Ensure data is normalized
- Try simpler configuration (reduce dim, disable modules)

### Training Instability

- Reduce learning rate
- Increase gradient clipping
- Check for NaN values in data
- Enable noise filtering in config

### Out of Memory

- Reduce batch size
- Reduce dim parameter
- Reduce gpu_cache_size_mb
- Enable disk caching

## Best Practices

1. **Start Small**: Begin with dim=128 to verify training works
2. **Monitor Closely**: Track loss, gradients, and learning rate
3. **Save Often**: Checkpoint every few epochs
4. **Validate Regularly**: Use validation set to check overfitting
5. **Log Everything**: Save metrics for later analysis

## Example: Complete Training Script

See `examples/train.py` for a complete training script with all features:
- Data loading
- Training loop
- Validation
- Checkpointing
- Metrics logging
- Visualization
