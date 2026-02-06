# ToTf

A Cross-Library Compatible Library for PyTorch and TensorFlow that provides advanced features for ease of use, which are not available directly in the base frameworks.

## Features

### 🎯 TrainingMonitor
Real-time training progress tracking with automatic logging:
- **Progress bars** with `tqdm` integration
- **CSV logging** with timestamps and running averages (Keras-style)
- **Resource monitoring** - RAM and VRAM usage tracking
- **Crash-resistant** - auto-flush to prevent data loss
- **Flexible** - works with any DataLoader or iterable

### 🔍 SmartSummary
Advanced model analysis with intelligent insights (UNIQUE features vs torchsummary/torchinfo):
- **Bottleneck detection** - automatically identifies optimization opportunities
- **Gradient tracking** - reveals training instabilities and vanishing/exploding gradients
- **Comprehensive analysis** - layer shapes, parameters, and memory usage
- **Export capabilities** - save to files or export as dictionaries
- **Complex architectures** - works with nested models, residual connections, etc.
- **Cross-framework support** - Available for both PyTorch and TensorFlow/Keras

## Installation

```bash
pip install -r requirements.txt
```

## Table of Contents
- [Quick Start](#quick-start)
- [TrainingMonitor Guide](#trainingmonitor-guide)
- [SmartSummary Guide](#smartsummary-guide)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Comparison with Alternatives](#comparison-with-alternatives)

---

## Quick Start

### PyTorch

#### TrainingMonitor (PyTorch)

```python
from ToTf.pytorch import TrainingMonitor

for epoch in range(epochs):
    monitor = TrainingMonitor(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch in monitor:
        loss = train_step(batch)
        monitor.log({'loss': loss.item()})
```

#### SmartSummary (PyTorch)

```python
from ToTf.pytorch import SmartSummary

# Basic analysis
model = YourModel()
summary = SmartSummary(model, input_size=(3, 224, 224))
summary.show()

# Find bottlenecks
bottlenecks = summary.get_bottlenecks(top_n=5)
for bn in bottlenecks:
    print(f"{bn['layer']}: {', '.join(bn['reasons'])}")

# Track gradients for debugging
summary = SmartSummary(model, input_size=(3, 224, 224), track_gradients=True)
summary.show()
```

### TensorFlow/Keras

#### SmartSummary (TensorFlow)

```python
from ToTf.tenf import SmartSummary

# Basic analysis
model = tf.keras.Sequential([...])
summary = SmartSummary(model, input_shape=(224, 224, 3))
summary.show()

# Find bottlenecks
bottlenecks = summary.get_bottlenecks(top_n=5)
for bn in bottlenecks:
    print(f"{bn['layer']}: {', '.join(bn['reasons'])}")

# Track gradients for debugging
summary = SmartSummary(model, input_shape=(224, 224, 3), track_gradients=True)
summary.show()
```

---

## Framework Differences

### SmartSummary: PyTorch vs TensorFlow

Both implementations provide the same core features, but with framework-specific adaptations:

| Aspect | PyTorch | TensorFlow/Keras |
|--------|---------|------------------|
| **Import** | `from ToTf.pytorch import SmartSummary` | `from ToTf.tenf import SmartSummary` |
| **Input parameter** | `input_size=(3, 224, 224)` (channels first) | `input_shape=(224, 224, 3)` (channels last) |
| **Model type** | `torch.nn.Module` | `tf.keras.Model` |
| **Device param** | `device='cuda'` ✓ | Not needed (TF auto-manages) |
| **Gradient tracking** | Uses backward hooks | Uses `tf.GradientTape` |
| **Multi-input** | Tuple of sizes | List of shapes |
| **Extra methods** | - | `compare_with_keras_summary()` |

**Key differences:**
- **Shape convention**: PyTorch uses channels-first (C, H, W), TensorFlow uses channels-last (H, W, C)
- **Automatic builds**: TensorFlow models build automatically on first forward pass
- **Multi-input models**: TensorFlow has first-class support with `input_shape=[(shape1), (shape2)]`

---

## TrainingMonitor Guide

### Basic Usage

```python
from ToTf import TrainingMonitor

for epoch in range(epochs):
    monitor = TrainingMonitor(train_loader, desc=f"Epoch {epoch+1}", log_file="train.csv")
    
    for batch in monitor:
        loss = train_step(batch)
        monitor.log({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})
```

### Features

**Automatic CSV Logging:**
```csv
timestamp,step,loss,lr,RAM_pct,VRAM_gb
2026-02-06 10:30:15,0,0.6931,0.001,45.2,2.14
2026-02-06 10:30:16,1,0.6523,0.001,45.4,2.14
```

**Running Averages:**  
Metrics are automatically averaged across steps (Keras-style), so displayed values represent cumulative running averages.

**Resource Monitoring:**  
Automatically tracks RAM usage (%) and VRAM usage (GB if CUDA available).

---

## SmartSummary Guide

### Common Patterns

#### PyTorch

**1. Quick Model Analysis**
```python
from ToTf.pytorch import SmartSummary

summary = SmartSummary(model, input_size=(3, 224, 224))
summary.show()
```

**2. Find Bottlenecks**
```python
bottlenecks = summary.get_bottlenecks(top_n=5)
for bn in bottlenecks:
    print(f"⚠️ {bn['layer']}: Score {bn['score']:.1f}")
    print(f"   Issues: {', '.join(bn['reasons'])}")
    print(f"   Parameters: {bn['params']:,}")
```

**3. Debug Training Issues**
```python
summary = SmartSummary(model, input_size=(3, 224, 224), track_gradients=True)
summary.show()  # Shows gradient variance/mean/max - useful for finding vanishing/exploding gradients
```

**4. Export Analysis**
```python
summary.save_to_file("model_analysis.txt")
data = summary.to_dict()  # For programmatic access
```

**5. CUDA Models**
```python
summary = SmartSummary(model, input_size=(3, 224, 224), device='cuda')
```

**6. Without Forward Pass (Fast)**
```python
summary = SmartSummary(model)  # Just count parameters, no shape inference
```

#### TensorFlow/Keras

**1. Quick Model Analysis**
```python
from ToTf.tenf import SmartSummary

summary = SmartSummary(model, input_shape=(224, 224, 3))
summary.show()
```

**2. Find Bottlenecks**
```python
bottlenecks = summary.get_bottlenecks(top_n=5)
for bn in bottlenecks:
    print(f"⚠️ {bn['layer']}: Score {bn['score']:.1f}")
    print(f"   Issues: {', '.join(bn['reasons'])}")
    print(f"   Parameters: {bn['params']:,}")
```

**3. Debug Training Issues**
```python
summary = SmartSummary(model, input_shape=(224, 224, 3), track_gradients=True)
summary.show()  # Shows gradient variance/mean/max
```

**4. Multi-Input Models**
```python
# For models with multiple inputs
summary = SmartSummary(model, input_shape=[(224, 224, 3), (100,)])
summary.show()
```

**5. Compare with Keras Summary**
```python
summary.compare_with_keras_summary()  # Shows both summaries
```

**6. Export Analysis**
```python
summary.save_to_file("model_analysis.txt")
data = summary.to_dict()  # For programmatic access
```

### Bottleneck Detection

SmartSummary identifies bottlenecks based on:
- **High parameters**: Layers with >10% of total model parameters
- **High gradient variance**: Indicates training instability (when tracking enabled)
- **Large outputs**: Intermediate tensors >10MB

Each bottleneck gets a score - higher scores indicate more critical optimization opportunities.

---

## API Reference

### TrainingMonitor

```python
TrainingMonitor(iterable, desc="Training", log_file="train_log.csv")
```

**Parameters:**
- `iterable` - DataLoader or any iterable to monitor
- `desc` (str) - Description for progress bar (default: "Training")
- `log_file` (str) - CSV file path (default: "train_log.csv")

**Methods:**
- `log(metrics: Dict[str, float])` - Log metrics and update running averages

**CSV Columns:**
- `timestamp` - ISO format timestamp
- `step` - Global step counter
- `<metric>` - Your logged metrics (running average)
- `RAM_pct` - RAM usage percentage
- `VRAM_gb` - GPU memory in GB (0 if no CUDA)

---

### SmartSummary

#### PyTorch API

```python
SmartSummary(model, input_size=None, batch_size=1, device='cpu', track_gradients=False)
```

**Parameters:**
- `model` (nn.Module) - PyTorch model to analyze
- `input_size` (Tuple, optional) - Input shape excluding batch, e.g., `(3, 224, 224)`
- `batch_size` (int) - Batch size for inference (default: 1)
- `device` (str) - Device: 'cpu' or 'cuda' (default: 'cpu')
- `track_gradients` (bool) - Track gradient statistics (default: False, slower)

#### TensorFlow/Keras API

```python
SmartSummary(model, input_shape=None, batch_size=1, track_gradients=False)
```

**Parameters:**
- `model` (keras.Model) - TensorFlow/Keras model to analyze
- `input_shape` (Tuple or List[Tuple], optional) - Input shape excluding batch, e.g., `(224, 224, 3)` for single input or `[(224, 224, 3), (100,)]` for multi-input
- `batch_size` (int) - Batch size for inference (default: 1)
- `track_gradients` (bool) - Track gradient statistics (default: False, slower)

#### Common Methods (Both Frameworks)

**Methods:**
- `show(show_bottlenecks=True)` - Display formatted summary table
- `get_bottlenecks(top_n=5)` - Get list of bottleneck layers
- `to_dict()` - Export complete analysis as dictionary
- `save_to_file(filename)` - Save summary to text file (UTF-8)

**TensorFlow-specific Methods:**
- `compare_with_keras_summary()` - Show both Keras built-in summary and SmartSummary side-by-side

**Bottleneck Dictionary Keys:**
- `layer` - Layer identifier
- `layer_name` - Layer class name
- `score` - Bottleneck score (higher = more critical)
- `reasons` - List of issues (e.g., "High params (91.4%)")
- `params` - Parameter count
- `output_shape` - Output tensor shape

---

## Comparison with Alternatives

### SmartSummary vs Other Tools

| Feature | **SmartSummary** | torchsummary | torchinfo | TF model.summary() |
|---------|:----------------:|:------------:|:---------:|:------------------:|
| Basic layer info | ✓ | ✓ | ✓ | ✓ |
| **Bottleneck detection** | **✓** | ✗ | ✗ | ✗ |
| **Gradient tracking** | **✓** | ✗ | ✗ | ✗ |
| **Optimization insights** | **✓** | ✗ | ✗ | ✗ |
| Memory estimation | ✓ | ✓ | ✓ | ✗ |
| Export to file/dict | ✓ | ✗ | ✓ | ✗ |
| Complex models | ✓ | Limited | ✓ | Limited |
| **PyTorch support** | **✓** | ✓ | ✓ | ✗ |
| **TensorFlow support** | **✓** | ✗ | ✗ | ✓ |

**SmartSummary is the ONLY tool with automatic bottleneck detection and gradient tracking!**

---

## Examples

### PyTorch Examples
- [example_usage.py](example_usage.py) - TrainingMonitor examples  
- [example_smartsummary.py](example_smartsummary.py) - SmartSummary examples (6 detailed scenarios)

### TensorFlow Examples
- [example_smartsummary_tf.py](example_smartsummary_tf.py) - SmartSummary examples for TensorFlow/Keras (8 detailed scenarios including multi-input models, gradient tracking, and MobileNetV2 analysis)

---

## Testing

Run the test suites to verify everything works:

```bash
# Test TrainingMonitor (PyTorch)
python test/test_monitor.py
python test/test_integration.py

# Test SmartSummary (PyTorch)
python test/test_smartsummary.py

# Test SmartSummary (TensorFlow)
python test/test_smartsummary_tf.py
```

All tests pass with 100% success rate ✓

---

## Backend Detection

ToTf automatically detects whether PyTorch or TensorFlow is installed:

```python
from ToTf import get_backend
print(get_backend())  # Returns 'torch' or 'tensorflow'
```

---

## Project Structure

```
ToTf/
├── __init__.py                     # Package exports with auto backend detection
├── backend.py                      # Backend detection (PyTorch/TensorFlow)
├── requirements.txt                # Dependencies
├── setup.py                        # Package config
├── pytorch/
│   ├── __init__.py
│   ├── trainingmonitor.py         # TrainingMonitor implementation
│   ├── smartsummary.py            # SmartSummary implementation (PyTorch)
│   └── README.md                  # PyTorch module docs
├── tenf/
│   ├── __init__.py
│   ├── smartsummary.py            # SmartSummary implementation (TensorFlow)
│   └── README.md                  # TensorFlow module docs
├── example_usage.py               # TrainingMonitor examples
├── example_smartsummary.py        # SmartSummary examples (PyTorch)
├── example_smartsummary_tf.py     # SmartSummary examples (TensorFlow)
├── test/
│   ├── test_monitor.py            # TrainingMonitor tests
│   ├── test_integration.py        # Integration tests
│   └── test_smartsummary.py       # SmartSummary tests
└── README.md                      # This file
```

---

## License

See [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting PRs.

---

**Version:** 0.1.0  
**Status:** Production Ready ✓
