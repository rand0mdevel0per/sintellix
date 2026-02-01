"""
Setup script for Sintellix Python package
"""

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
from pathlib import Path

# Read version from __init__.py
version = {}
with open("sintellix/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)

# Read long description from README
long_description = ""
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

# Dependencies
install_requires = [
    "torch>=2.0.0",
    "numpy>=1.20.0",
    "protobuf>=3.20.0",
    "tqdm>=4.60.0",
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
    ],
    "docs": [
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=1.0.0",
    ],
}

setup(
    name="sintellix",
    version=version.get("__version__", "0.1.0"),
    author="Sintellix Team",
    author_email="contact@sintellix.ai",
    description="Neural Network Framework with HOT Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sintellix/sintellix",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    zip_safe=False,
)
