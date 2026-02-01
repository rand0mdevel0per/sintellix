# Sintellix API Reference

## Python API

### ConfigManager

Configuration management for Sintellix models.

```python
class ConfigManager:
    def __init__(self)
    def load_from_json(self, path: str) -> bool
    def load_from_proto(self, path: str) -> bool
    def save_to_json(self, path: str) -> bool
    def save_to_proto(self, path: str) -> bool
    def get_config(self) -> NeuronConfig
    @staticmethod
    def create_default() -> NeuronConfig
```

**Example:**
```python
import sintellix

config_mgr = sintellix.ConfigManager()
config_mgr.load_from_json("config.json")
config = config_mgr.get_config()
```

### NeuronModel

Main model class for training and inference.

```python
class NeuronModel:
    def __init__(self, config: NeuronConfig)
    def initialize(self) -> bool
    def forward(self, input: np.ndarray) -> np.ndarray
    def backward(self, grad_output: np.ndarray) -> np.ndarray
    def update_parameters(self, learning_rate: float) -> None
    def save_state(self, path: str) -> bool
    def load_state(self, path: str) -> bool
    def get_kfe_manager(self) -> KFEManager
```

**Parameters:**
- `config`: NeuronConfig object with model configuration
- `input`: Input array of shape (batch_size, dim)
- `grad_output`: Gradient array of shape (batch_size, dim)
- `learning_rate`: Learning rate for parameter updates
- `path`: File path for model state serialization

**Returns:**
- `forward()`: Output array of shape (batch_size, dim)
- `backward()`: Gradient input array of shape (batch_size, dim)
- `initialize()`, `save_state()`, `load_state()`: Success boolean

**Example:**
```python
model = sintellix.NeuronModel(config)
model.initialize()

# Training
output = model.forward(input_data)
grad_input = model.backward(grad_output)
model.update_parameters(0.001)

# Save/Load
model.save_state("checkpoint.pb")
model.load_state("checkpoint.pb")
```

### KFEManager

Knowledge Feature Encoding manager for semantic storage.

```python
class KFEManager:
    def __init__(self, max_slots: int = 10000)
    def has_kfe(self, key: str) -> bool
    def get_slot_count(self) -> int
```

**Parameters:**
- `max_slots`: Maximum number of knowledge feature slots
- `key`: String key for knowledge lookup

**Example:**
```python
kfe_mgr = model.get_kfe_manager()
print(f"Total slots: {kfe_mgr.get_slot_count()}")
if kfe_mgr.has_kfe("concept_123"):
    print("Knowledge exists")
```

## C++ API

### ConfigManager

```cpp
namespace sintellix {

class ConfigManager {
public:
    ConfigManager();
    ~ConfigManager();

    bool loadFromJson(const std::string& path);
    bool loadFromProto(const std::string& path);
    bool saveToJson(const std::string& path) const;
    bool saveToProto(const std::string& path) const;

    const NeuronConfig& getConfig() const;
    NeuronConfig& getConfig();

    static NeuronConfig createDefault();
};

}
```

### NeuronModel

```cpp
namespace sintellix {

class NeuronModel {
public:
    explicit NeuronModel(const NeuronConfig& config);
    ~NeuronModel();

    bool initialize();

    void forward(const double* input, double* output);
    void backward(const double* grad_output, double* grad_input);
    void update_parameters(float learning_rate);

    bool save_state(const std::string& path);
    bool load_state(const std::string& path);

    KFEManager& get_kfe_manager();

private:
    NeuronConfig config_;
    std::vector<std::unique_ptr<Neuron>> neurons_;
    std::unique_ptr<TieredStorageManager> storage_manager_;
    KFEManager kfe_manager_;
};

}
```

### Neuron

```cpp
namespace sintellix {

class Neuron {
public:
    Neuron(int dim, int num_heads, int temporal_frames);
    ~Neuron();

    void forward(const double* input, double* output);
    void backward(const double* grad_output, double* grad_input);
    void update_parameters(float learning_rate);

    void set_temporal_context(const std::vector<double*>& history);
    void enable_global_aggregation(bool enable);
    void enable_noise_filter(bool enable);

private:
    int dim_;
    int num_heads_;
    int temporal_frames_;

    // GPU memory
    double* d_state_;
    double* d_gradients_;
    double* d_m_;  // Adam first moment
    double* d_v_;  // Adam second moment
};

}
```

## Configuration Schema

### NeuronConfig

```cpp
message NeuronConfig {
    int32 dim = 1;                    // Neuron dimension (128/256/512/1024)
    int32 num_heads = 2;              // Number of attention heads
    repeated int32 grid_size = 3;     // Grid dimensions [x, y, z]
    int32 temporal_frames = 4;        // Temporal context frames

    OptimizerConfig optimizer = 5;
    ModuleConfig modules = 6;
    StorageConfig storage = 7;
}

message OptimizerConfig {
    float learning_rate = 1;
    float beta1 = 2;
    float beta2 = 3;
    float epsilon = 4;
    float gradient_clip = 5;
}

message ModuleConfig {
    bool enable_multi_head = 1;
    bool enable_global_aggregation = 2;
    bool enable_noise_filter = 3;
    bool enable_temporal_attention = 4;
    bool enable_fxaa_layer = 5;
}

message StorageConfig {
    int32 gpu_cache_size_mb = 1;
    int32 ram_cache_size_mb = 2;
    string disk_cache_path = 3;
}
```

## Error Handling

### Python Exceptions

```python
try:
    model = sintellix.NeuronModel(config)
    model.initialize()
except RuntimeError as e:
    print(f"Initialization failed: {e}")

try:
    model.save_state("checkpoint.pb")
except IOError as e:
    print(f"Save failed: {e}")
```

### C++ Error Codes

```cpp
// Boolean return values indicate success/failure
if (!model.initialize()) {
    std::cerr << "Initialization failed" << std::endl;
    return -1;
}

if (!model.save_state("checkpoint.pb")) {
    std::cerr << "Save failed" << std::endl;
    return -1;
}
```

## Thread Safety

- **NeuronModel**: Not thread-safe. Use separate instances per thread.
- **ConfigManager**: Thread-safe for read operations after initialization.
- **KFEManager**: Thread-safe with internal locking.

## Memory Management

### Python

Memory is automatically managed through pybind11 smart pointers. Models are freed when Python objects are garbage collected.

### C++

Use RAII principles. NeuronModel destructor automatically frees GPU memory:

```cpp
{
    sintellix::NeuronModel model(config);
    model.initialize();
    // ... use model ...
}  // GPU memory automatically freed here
```

## Performance Tips

1. **Batch Processing**: Process multiple samples together for better GPU utilization
2. **Persistent Models**: Reuse model instances instead of recreating
3. **Async Operations**: Use CUDA streams for overlapping compute and I/O
4. **Memory Pinning**: Use pinned memory for faster CPU-GPU transfers

```python
# Good: Batch processing
outputs = model.forward(batch_data)  # Shape: (32, 256)

# Bad: Individual samples
for sample in data:
    output = model.forward(sample)  # Inefficient
```
