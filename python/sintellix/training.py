"""
Training utilities for Sintellix Neural Network Framework
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import json
from tqdm import tqdm


class TrainingConfig:
    """
    Configuration for training

    Args:
        learning_rate: Learning rate (default: 0.001)
        batch_size: Batch size (default: 32)
        epochs: Number of epochs (default: 100)
        save_interval: Save checkpoint every N epochs (default: 10)
        log_interval: Log metrics every N steps (default: 100)
        checkpoint_dir: Directory to save checkpoints (default: "./checkpoints")
        use_amp: Use automatic mixed precision (default: False)
        gradient_clip: Gradient clipping value (default: 1.0)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        save_interval: int = 10,
        log_interval: int = 100,
        checkpoint_dir: str = "./checkpoints",
        use_amp: bool = False,
        gradient_clip: float = 1.0,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.use_amp = use_amp
        self.gradient_clip = gradient_clip

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "save_interval": self.save_interval,
            "log_interval": self.log_interval,
            "checkpoint_dir": str(self.checkpoint_dir),
            "use_amp": self.use_amp,
            "gradient_clip": self.gradient_clip,
        }

    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class Trainer:
    """
    Trainer for Sintellix Neural Network Model

    Args:
        model: NeuronModel instance
        config: TrainingConfig instance
        loss_fn: Loss function (default: MSE)
        device: Device to use (default: "cuda" if available)

    Example:
        >>> from sintellix import NeuronModel, NeuronConfig, Trainer, TrainingConfig
        >>> model = NeuronModel(NeuronConfig())
        >>> model.initialize()
        >>> trainer = Trainer(model, TrainingConfig())
        >>> trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        model,
        config: TrainingConfig,
        loss_fn: Optional[Callable] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Default loss function: MSE for reconstruction
        self.loss_fn = loss_fn or nn.MSELoss()

        # Move model to device
        self.model = self.model.to(self.device)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Metrics history
        self.train_losses = []
        self.val_losses = []

    def save_checkpoint(self, path: Optional[str] = None, is_best: bool = False):
        """
        Save training checkpoint

        Args:
            path: Checkpoint path (default: auto-generated)
            is_best: Whether this is the best model so far
        """
        if path is None:
            path = self.config.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_config': self.model.config.to_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")

        # Save best model separately
        if is_best:
            best_path = self.config.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Best model saved to: {best_path}")

    def load_checkpoint(self, path: str):
        """
        Load training checkpoint

        Args:
            path: Checkpoint path
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Checkpoint loaded from: {path}")
        print(f"Resuming from epoch {self.current_epoch}")

    def train_epoch(self, train_loader) -> float:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)

            # Compute loss
            loss = self.loss_fn(outputs, targets)

            # Backward pass
            grad_output = torch.autograd.grad(loss, outputs, retain_graph=True)[0]
            grad_input = self.model.backward(grad_output)

            # Update parameters
            self.model.update_parameters(self.config.learning_rate)

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Log
            if self.global_step % self.config.log_interval == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        return total_loss / num_batches

    def validate(self, val_loader) -> float:
        """
        Validate the model

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = self.loss_fn(outputs, targets)

                # Update metrics
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, train_loader, val_loader=None):
        """
        Train the model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        print(f"Starting training for {self.config.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Batch size: {self.config.batch_size}")

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")

                # Check if best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"New best validation loss: {val_loss:.4f}")
            else:
                is_best = False

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(is_best=is_best)

        print("Training completed!")

