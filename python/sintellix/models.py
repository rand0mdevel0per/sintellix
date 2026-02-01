"""
Model management utilities for Sintellix

Handles downloading and caching of pretrained models (CLIP, VQ-GAN)
"""

import os
from pathlib import Path
from typing import Optional, Dict
import json
import hashlib
from urllib.request import urlretrieve
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for model downloads"""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class ModelManager:
    """
    Manages model downloads and caching

    Models are cached in ~/.cache/sintellix/ by default
    """

    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sintellix"

    # Model URLs (placeholder - will be updated with actual URLs)
    MODEL_URLS = {
        "clip": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/model.onnx",
        "vqgan-codebook": "https://huggingface.co/sintellix/vqgan/resolve/main/codebook.bin",
    }

    # Model checksums for verification
    MODEL_CHECKSUMS = {
        "clip": "placeholder_checksum",
        "vqgan-codebook": "placeholder_checksum",
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize ModelManager

        Args:
            cache_dir: Directory to cache models (default: ~/.cache/sintellix)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load cache metadata
        self.metadata_path = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def verify_model(self, model_name: str) -> bool:
        """
        Verify model integrity using checksum

        Args:
            model_name: Name of the model

        Returns:
            True if model is valid
        """
        model_path = self.get_model_path(model_name)
        if not model_path.exists():
            return False

        # Compute checksum
        checksum = self._compute_checksum(model_path)

        # Compare with expected checksum
        expected = self.MODEL_CHECKSUMS.get(model_name)
        if expected == "placeholder_checksum":
            # Skip verification for placeholder checksums
            return True

        return checksum == expected

    def get_model_path(self, model_name: str) -> Path:
        """
        Get path to cached model

        Args:
            model_name: Name of the model

        Returns:
            Path to model file
        """
        if model_name == "clip":
            return self.cache_dir / "clip.onnx"
        elif model_name == "vqgan-codebook":
            return self.cache_dir / "vqgan-codebook.bin"
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def download_model(self, model_name: str, force: bool = False) -> Path:
        """
        Download model if not cached

        Args:
            model_name: Name of the model
            force: Force re-download even if cached

        Returns:
            Path to downloaded model
        """
        model_path = self.get_model_path(model_name)

        # Check if already cached and valid
        if not force and model_path.exists():
            if self.verify_model(model_name):
                print(f"Model '{model_name}' already cached at: {model_path}")
                return model_path
            else:
                print(f"Model '{model_name}' cache corrupted, re-downloading...")

        # Get download URL
        url = self.MODEL_URLS.get(model_name)
        if not url:
            raise ValueError(f"Unknown model: {model_name}")

        # Download with progress bar
        print(f"Downloading model '{model_name}' from: {url}")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=model_name) as t:
            urlretrieve(url, model_path, reporthook=t.update_to)

        # Verify downloaded model
        if not self.verify_model(model_name):
            model_path.unlink()
            raise RuntimeError(f"Downloaded model '{model_name}' failed verification")

        # Update metadata
        self.metadata[model_name] = {
            "path": str(model_path),
            "url": url,
            "checksum": self._compute_checksum(model_path),
        }
        self._save_metadata()

        print(f"Model '{model_name}' downloaded successfully to: {model_path}")
        return model_path


# Global model manager instance
_global_manager = None


def get_model_manager() -> ModelManager:
    """Get global ModelManager instance"""
    global _global_manager
    if _global_manager is None:
        _global_manager = ModelManager()
    return _global_manager


def download_model(model_name: str, force: bool = False, cache_dir: Optional[str] = None) -> Path:
    """
    Download a pretrained model

    Args:
        model_name: Name of the model ("clip" or "vqgan-codebook")
        force: Force re-download even if cached
        cache_dir: Custom cache directory (default: ~/.cache/sintellix)

    Returns:
        Path to downloaded model

    Example:
        >>> from sintellix import download_model
        >>> clip_path = download_model("clip")
        >>> vqgan_path = download_model("vqgan-codebook")
    """
    if cache_dir:
        manager = ModelManager(cache_dir)
    else:
        manager = get_model_manager()

    return manager.download_model(model_name, force=force)

