# Getting Started with Sintellix

## Installation

### Prerequisites

Before installing Sintellix, ensure you have:

- **CUDA Toolkit 11.8+**: [Download from NVIDIA](https://developer.nvidia.com/cuda-toolkit)
- **CMake 3.18+**: [Download from cmake.org](https://cmake.org/download/)
- **Visual Studio 2022** (Windows) or **GCC 9+** (Linux)
- **Python 3.8+** (for Python bindings)

### Windows Installation

1. **Install CUDA Toolkit**
   ```bash
   # Verify installation
   nvcc --version
   ```

2. **Clone Repository**
   ```bash
   git clone https://github.com/sintellix/sintellix.git
   cd sintellix
   ```

3. **Install Dependencies with vcpkg**
   ```bash
   # Open Visual Studio Developer Command Prompt
   vcpkg install
   ```

4. **Build with CMake**
   ```bash
   mkdir build
   cd build
   cmake .. -DCMAKE_TOOLCHAIN_FILE=E:/sintellix/vcpkg_installed/vcpkg/scripts/buildsystems/vcpkg.cmake
   cmake --build . --config Release
   ```

5. **Install Python Bindings** (Optional)
   ```bash
   cd python
   pip install -e .
   ```

### Linux Installation

1. **Install Dependencies**
   ```bash
   sudo apt-get update
   sudo apt-get install cmake build-essential libprotobuf-dev protobuf-compiler
   ```

2. **Build from Source**
   ```bash
   git clone https://github.com/sintellix/sintellix.git
   cd sintellix
   mkdir build && cd build
   cmake .. -DBUILD_PYTHON=ON
   make -j$(nproc)
   sudo make install
   ```

## First Steps

### 1. Create Configuration File

Create `config.json`:

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
    "disk_cache_path": "./cache"
  }
}
```

### 2. Basic Python Example

```python
import sintellix
import numpy as np

# Load configuration
config_mgr = sintellix.ConfigManager()
config_mgr.load_from_json("config.json")
config = config_mgr.get_config()

# Create and initialize model
model = sintellix.NeuronModel(config)
if not model.initialize():
    print("Failed to initialize model")
    exit(1)

# Prepare input data
batch_size = 32
dim = 256
input_data = np.random.randn(batch_size, dim).astype(np.float64)

# Forward pass
output = model.forward(input_data)
print(f"Output shape: {output.shape}")

# Save model
model.save_state("model_checkpoint.pb")
print("Model saved successfully")
```

### 3. Basic C++ Example

```cpp
#include <sintellix/core/neuron_model.cuh>
#include <sintellix/core/config.hpp>
#include <iostream>
#include <vector>

int main() {
    // Load configuration
    sintellix::ConfigManager config_mgr;
    if (!config_mgr.loadFromJson("config.json")) {
        std::cerr << "Failed to load config" << std::endl;
        return 1;
    }
    auto config = config_mgr.getConfig();

    // Create and initialize model
    sintellix::NeuronModel model(config);
    if (!model.initialize()) {
        std::cerr << "Failed to initialize model" << std::endl;
        return 1;
    }

    // Prepare input data
    int batch_size = 32;
    int dim = config.dim();
    std::vector<double> input(batch_size * dim);
    std::vector<double> output(batch_size * dim);

    // Forward pass
    model.forward(input.data(), output.data());
    std::cout << "Forward pass completed" << std::endl;

    // Save model
    if (model.save_state("model_checkpoint.pb")) {
        std::cout << "Model saved successfully" << std::endl;
    }

    return 0;
}
```

## Next Steps

- **Training**: See [Training Guide](training_guide.md) for detailed training instructions
- **API Reference**: See [API Reference](api_reference.md) for complete API documentation
- **Architecture**: See [Architecture](architecture.md) for system design details
- **Examples**: Check the `examples/` directory for more use cases

## Common Issues

### CUDA Out of Memory

Reduce the `dim` parameter or `gpu_cache_size_mb` in configuration:

```json
{
  "neuron": {
    "dim": 128
  },
  "storage": {
    "gpu_cache_size_mb": 4096
  }
}
```

### Slow Performance

Enable fast math and check GPU utilization:

```bash
nvidia-smi
```

Ensure your GPU is being used and not throttled.

### Import Error (Python)

Ensure the library is in your Python path:

```python
import sys
sys.path.append('/path/to/sintellix/build/python')
import sintellix
```

## Support

- **GitHub Issues**: [Report bugs](https://github.com/sintellix/sintellix/issues)
- **Documentation**: [Full docs](https://sintellix.readthedocs.io)
- **Examples**: [Example code](https://github.com/sintellix/sintellix/tree/main/examples)
