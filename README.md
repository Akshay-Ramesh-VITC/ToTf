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
- **Loss curve analysis** - diagnose training dynamics (convergence, overfitting, divergence, plateau detection)
- **Comprehensive analysis** - layer shapes, parameters, and memory usage
- **Export capabilities** - save to files or export as dictionaries
- **Complex architectures** - works with nested models, residual connections, etc.
- **Cross-framework support** - Available for both PyTorch and TensorFlow/Keras
### ✨ What's New (Added NEW!)

- **Top-level dynamic dispatcher (`ToTf`)**: a centralized backend dispatcher now exposes core components from the package root so users can `from ToTf import SmartSummary, ModelView, draw_graph, get_component` instead of importing backend-specific subpackages.
- **`get_component(name, backend_name=...)`**: load utilities or components dynamically from the detected or selected backend (replaces importing `ToTf.pytorch.*` or `ToTf.tenf.*` directly).
- **`draw_graph()` dispatcher**: call `ToTf.draw_graph(...)` and the package will route the call to the active backend's implementation (convenience wrapper for `ModelView` rendering).
- **Examples updated**: example scripts were updated to use the top-level `ToTf` API and include a small `sys.path` guideline to run examples from `ToTf/examples/` when not installed as a package.

- **Dry-run init & gradient checks** - runs a lightweight forward/backward pass to flag zero or abnormally large initial gradients and suspicious initialization scales (Xavier/He heuristics)
- **Receptive Field bookkeeping** - per-layer RF/jump/start propagation supporting dilations and asymmetric kernels across branches; emits warnings when inputs to merges have mismatched RF metadata
- **Precise activation memory profiling** - optional `keep_activations=True` stores captured output tensors and uses actual tensor sizes for a more accurate memory profile. PyTorch also supports `keep_activations_strong=True` with `max_saved_activation_bytes` to retain tensors for deeper analysis while avoiding OOM.
- **Configurable thresholds** - constructor options control sensitivity: `grad_large_threshold`, `grad_zero_tol`, `param_ratio_bottleneck`, `activation_bottleneck_mb`, and init/std heuristics (`init_std_warn_multiply`, `init_std_warn_min_mult`).

### 🛠️ Utility Functions
Framework-agnostic "missing" functions that save time and improve code quality:

#### Auto-Shape Flattener
- **`lazy_flatten(tensor)`** - Automatically flatten tensors without manual size calculation
- **`get_flatten_size(shape)`** - Calculate flattened dimensions for Conv->Linear/Dense transitions
- **Use case**: Eliminates error-prone manual calculations when transitioning from convolutional to fully connected layers

#### Normalized Cross-Correlation (NCC) Loss
- **`loss_ncc(y_true, y_pred)`** - NCC loss function for medical imaging and registration
- **`ncc_score(y_true, y_pred)`** - NCC similarity metric (higher is better)
- **`NCCLoss()`** - Keras-compatible NCC loss class (TensorFlow)
- **Use case**: Medical imaging (ACDC, ADNI datasets), robust to intensity variations, better than MSE for registration tasks

#### Learning Rate Finder
- **`find_lr(model, optimizer, ...)`** - Find optimal learning rate automatically
- **`LRFinder`** - Full-featured LR range test class with plotting
- **Use case**: Automatically discover the best learning rate before training (fast.ai style), avoid manual tuning

#### Cross-Framework Reproducibility
- **`seed_everything(seed=42, deterministic=False, benchmark=False, verbose=False)`** - Seeds Python, NumPy, PyTorch (CPU & CUDA), and TensorFlow in a single call. Optionally sets deterministic algorithms and cuBLAS workspace config.
- **Use case**: Guaranteed reproducible experiments and benchmarks across backends without repeating 10 lines of boilerplate.

#### Tensor Health & Diagnostics
- **`inspect_tensor(tensor, name=None, print_info=True, quantiles=True)`** - One-line diagnostic inspector for PyTorch, TensorFlow, or NumPy tensors.
- **Information provided**: Shape, dtype, device, memory MB, element count, NaN/Inf detection, zero sparsity %, min, max, mean, std, and quantiles (p25, p50, p75).
- **Use case**: Rapid sanity-checking and debugging of activations, embeddings, and gradients.

#### Transfer Learning & Fine-Tuning
- **`freeze_layers(model, up_to=None, layer_names=None, layer_types=None, regex=None)`** - Freeze model parameters/layers (supports both `torch.nn.Module` and `tf.keras.Model`).
- **`unfreeze_layers(model, last_n=None, layer_names=None, layer_types=None, regex=None)`** - Selectively unfreeze layers (e.g., `last_n=1` for classification heads).
- **`freeze_batchnorm(model, freeze_eval=True)`** - Lock BatchNorm affine parameters and freeze running statistics (eval mode) during fine-tuning.
- **Use case**: Effortless backbone freezing and transfer learning across both frameworks.

#### Focal Loss for Class Imbalance
- **`loss_focal(y_true, y_pred, alpha=0.25, gamma=2.0)`** - Focal loss function for addressing extreme class imbalance in object detection and dense classification.
- **`FocalLoss(alpha=0.25, gamma=2.0)`** - PyTorch and Keras-compatible Focal Loss class.
- **Use case**: One-stage detectors, fraud/rare disease detection where easy negatives overwhelm the gradients.

#### Dice Loss & IoU Segmentation Metrics
- **`loss_dice(y_true, y_pred, smooth=1e-6)`** - Soft Dice loss for image segmentation.
- **`DiceLoss(smooth=1e-6)`** - PyTorch and Keras-compatible Dice Loss class.
- **`dice_score(y_true, y_pred)`** / **`iou_score(y_true, y_pred)`** - Fast Dice Similarity Coefficient and Jaccard Index metrics.
- **Use case**: Biomedical and satellite image segmentation with foreground-background class imbalance.

#### Numerical Anomaly Tracking (First-NaN / Inf Pinpointing)
- **`AnomalyDetector(model, check_inputs=True, check_outputs=True, check_gradients=True)`** - Non-intrusive hook monitor that intercepts the exact layer and tensor where NaN or Inf first originates.
- **`check_anomalies(tensor, name=None)`** - Quick boolean assertion check for tensors.
- **Use case**: Instantly identifies exploding gradients, numerical instability, or invalid layer inputs without hours of manual debugging.

#### Publication-Ready Model Profiling & Benchmarking
- **`benchmark_model(model, input_shape=..., num_runs=50, warmup=10)`** - Complete model complexity profiler calculating total/trainable params, memory (MB), estimated FLOPs & MACs, latency percentiles (mean, std, p50, p95, p99), throughput (FPS), plus ready-to-paste **Markdown** and **LaTeX** tables.
- **Use case**: Generate camera-ready complexity and performance tables for research papers, GitHub READMEs, and technical reports.

#### Direct In-Memory Weight Transfer (PyTorch <-> TensorFlow)
- **`transfer_weights(source_model, target_model, layer_mapping=None, verbose=True)`** - Direct zero-disk-overhead in-memory weight mapper between PyTorch and TensorFlow/Keras models.
- **Automated features**: Automatically handles spatial weight transpositions (PyTorch `[out, in, H, W]` vs Keras `[H, W, in, out]`), Linear/Dense transpositions (`[out, in]` vs `[in, out]`), and aligns BatchNorm gamma, beta, mean, and var.
- **Use case**: Instant cross-framework model porting, rapid paper replication, and comparing inference speeds between PyTorch and TensorFlow without cumbersome ONNX exports.

### 📊 ModelView (PyTorch & TensorFlow)
Publication-quality neural network architecture diagrams for both frameworks:
- **PyTorch support** - Wraps torchview for comprehensive PyTorch model visualization
- **TensorFlow support** - Native implementation for Keras models
- **High-resolution outputs** - PNG, PDF, SVG for research papers (300-600 DPI)
- **Automatic graph layout** - Beautiful visualizations with minimal configuration
- **Comprehensive annotations** - Layer types, shapes, parameter counts
- **Complex architectures** - Residual connections, multi-input/output, branching
- **Customizable styling** - Colors, fonts, layouts for different aesthetics
- **Export formats** - JSON summaries, text tables, visual diagrams
- **Unified API** - Same interface for both PyTorch and TensorFlow
- **Use case**: Generate publication-ready architecture diagrams for research papers, presentations, and documentation

## Installation

```bash
pip install ToTf
```

### System Requirements for ModelView

ModelView requires Graphviz and torchview for rendering diagrams:

```bash
# Install Python packages
pip install graphviz torchview

# Install system Graphviz
# Ubuntu/Debian:
sudo apt-get install graphviz

# macOS:
brew install graphviz

# Windows (via conda):
conda install -c conda-forge graphviz
```

## Documentation

📚 **Detailed Guides:**
- **[ModelView Quick Start - PyTorch](./docs/QUICKSTART_MODELVIEW_PYTORCH.md)** - Get started with PyTorch diagrams in 5 minutes
- **[ModelView Quick Start - TensorFlow](./docs/QUICKSTART_MODELVIEW.md)** - Get started with TensorFlow diagrams in 5 minutes
- **[Loss Curve Analysis Guide](./docs/LOSS_CURVE_ANALYSIS.md)** - Diagnose training dynamics and detect overfitting
- **[PyTorch ModelView Implementation](./docs/PYTORCH_MODELVIEW_IMPLEMENTATION.md)** - PyTorch-specific implementation details
- **[ModelView Implementation Details](./docs/MODELVIEW_IMPLEMENTATION_SUMMARY.md)** - Technical specifications and features
- **[Utilities Implementation](./docs/UTILITIES_IMPLEMENTATION_SUMMARY.md)** - Detailed utility functions documentation
- **[TensorFlow Implementation](./docs/TENSORFLOW_IMPLEMENTATION_SUMMARY.md)** - TensorFlow-specific features and design

📖 **Examples:**
- `example_modelview_pytorch.py` - PyTorch ModelView examples (6+ comprehensive examples)
- `example_modelview_tf.py` - TensorFlow ModelView examples (8+ comprehensive examples)
- `example_smartsummary.py` / `example_smartsummary_tf.py` - SmartSummary usage
- `example_loss_analysis_pytorch.py` - Loss curve analysis examples (PyTorch)
- `example_loss_analysis_tf.py` - Loss curve analysis examples (TensorFlow)
- `example_utils_pytorch.py` / `example_utils_tf.py` - Utility functions

🧪 **Verification:**
- Run `python verify_modelview_pytorch.py` to test PyTorch ModelView installation
- Run `python verify_modelview.py` to test TensorFlow ModelView installation
- Run `pytest test/` to run full test suite

## ONNX Export & Inference

ToTf provides convenient ONNX export and inference wrappers for both PyTorch and TensorFlow models.

Examples:

- Export a PyTorch model to ONNX and run inference using ONNX Runtime:

```python
from ToTf import toONNX, fromONNX
import torch

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 4, 3),
    torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool2d((1, 1)),
    torch.nn.Flatten(),
    torch.nn.Linear(4, 10)
)

dummy = torch.randn(1, 1, 8, 8)
onnx_path = toONNX(model, dummy, "model_pytorch.onnx")

# Run ONNX inference (returns numpy array)
out = fromONNX(onnx_path, dummy.numpy())
print(out.shape)
```

- Export a Keras model to ONNX (requires `tf2onnx`) and run inference:

```python
from ToTf import toONNX, fromONNX
import numpy as np
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(8,8,1)),
    tf.keras.layers.Conv2D(4,3,activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)
])

dummy = np.random.randn(1,8,8,1).astype(np.float32)
onnx_path = toONNX(model, "model_tf.onnx", example_input=dummy)
out = fromONNX(onnx_path, dummy)
print(out.shape)
```

Notes:
- The PyTorch `toONNX` uses `torch.onnx.export` (opset default 11). Pass `opset_version` or `dynamic_axes` as needed.
- The TensorFlow `toONNX` uses `tf2onnx` and requires it to be installed (`tf2onnx` is listed in the project requirements).
- `fromONNX` uses `onnxruntime` for inference; you can pass custom `providers` if needed.

## Table of Contents
- [Installation](#installation)
- [Documentation](#documentation)
- [Quick Start](#quick-start)
- [TrainingMonitor Guide](#trainingmonitor-guide)
- [SmartSummary Guide](#smartsummary-guide)
- [ModelView Guide](#modelview-guide)
- [Utility Functions Guide](#utility-functions-guide)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [API Reference](#api-reference)
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
    - `keep_activations_strong` (opt-in): retain strong references to activations; use with `max_saved_activation_bytes` to limit memory
    - `grad_large_threshold`, `grad_zero_tol`: control gradient anomaly detection
    - `param_ratio_bottleneck`, `activation_bottleneck_mb`: control bottleneck detection thresholds
summary.show()
```

#### Utility Functions (PyTorch)

```python
from ToTf.pytorch import lazy_flatten, loss_ncc, find_lr

# 1. Auto-flatten Conv output
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.fc = nn.Linear(get_flatten_size((64, 30, 30)), 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = lazy_flatten(x)  # No manual calculation needed!
        return self.fc(x)

# 2. NCC loss for medical imaging
loss = loss_ncc(ground_truth, prediction)  # Robust to intensity variations

# 3. Find optimal learning rate
best_lr = find_lr(model, optimizer, criterion, train_loader)
print(f"Use learning rate: {best_lr}")
```

#### ModelView (PyTorch)

```python
from ToTf.pytorch import ModelView, draw_graph

# Quick visualization
model = MyModel()
draw_graph(model, input_size=(3, 224, 224), save_path='model.png')

# Advanced usage with customization
view = ModelView(model, input_size=(3, 224, 224))
view.show()  # Text summary powered by torchview

# High-resolution PNG for papers (300 DPI)
view.render('architecture.png', dpi=300, show_shapes=True)

# PDF for LaTeX documents
view.render('architecture.pdf', format='pdf')

# SVG for perfect scaling
view.render('architecture.svg', format='svg')

# Horizontal layout for wide figures
view.render('architecture_wide.png', rankdir='LR', dpi=600)

# Advanced torchview features
view_detailed = ModelView(
    model, 
    input_size=(3, 224, 224),
    depth=4,  # Show nested modules  
    expand_nested=True,  # Expand Sequential blocks
    hide_inner_tensors=False  # Show all tensors
)
view_detailed.render('detailed_architecture.png', dpi=300)

# Export summary as JSON
view.save_summary_json('model_summary.json')
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
    - `param_ratio_bottleneck`, `activation_bottleneck_mb`: control bottleneck detection thresholds
    - `grad_large_threshold`, `init_std_warn_multiply`, `init_std_warn_min_mult`: control dry-run init/gradient heuristics
summary.show()
```

#### ModelView (TensorFlow)

```python
from ToTf.tenf import ModelView, draw_graph

# Quick visualization
model = tf.keras.Sequential([...])
draw_graph(model, input_shape=(224, 224, 3), save_path='model.png')

# Advanced usage with customization
view = ModelView(model, input_shape=(224, 224, 3))
view.show()  # Text summary

# High-resolution PNG for papers (300 DPI)
view.render('architecture.png', dpi=300, show_shapes=True, show_params=True)

# PDF for LaTeX documents
view.render('architecture.pdf', format='pdf')

# SVG for perfect scaling
view.render('architecture.svg', format='svg')

# Horizontal layout for wide figures
view.render('architecture_wide.png', rankdir='LR', dpi=600)

# Export summary as JSON
view.save_summary_json('model_summary.json')
```

#### Utility Functions (TensorFlow)

```python
from ToTf.tenf import lazy_flatten, NCCLoss, find_lr

# 1. Auto-flatten Conv output
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = keras.layers.Conv2D(64, 3)
        self.fc = keras.layers.Dense(10)
    
    def call(self, x):
        x = self.conv(x)
        x = lazy_flatten(x)  # Automatic flattening!
        return self.fc(x)

# 2. NCC loss for medical imaging
model.compile(optimizer='adam', loss=NCCLoss())  # Keras-compatible!

# 3. Find optimal learning rate
best_lr = find_lr(model, keras.losses.CategoricalCrossentropy(), train_dataset)
print(f"Use learning rate: {best_lr}")

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

**7. Analyze Training Loss Curves (PyTorch)**
```python
# During training loop, collect losses
train_losses = []
val_losses = []

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate_epoch(model, val_loader)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Analyze training dynamics
summary = SmartSummary(model, input_size=(3, 224, 224))
result = summary.analyze_loss_curve(train_losses, val_losses)
# Detects: overfitting, divergence, oscillation, plateau, convergence
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

**7. Analyze Training Loss Curves**
```python
# Collect losses during training
train_losses = [2.5, 2.1, 1.8, 1.6, 1.5, 1.45, 1.42, 1.41]
val_losses = [2.6, 2.2, 1.9, 1.7, 1.65, 1.62, 1.60, 1.59]

# Analyze training dynamics
result = summary.analyze_loss_curve(train_losses, val_losses)
# Automatically detects: overfitting, divergence, oscillation, plateau, convergence
# Provides actionable recommendations for improvement

# Programmatic use (no printing)
result = summary.analyze_loss_curve(train_losses, val_losses, verbose=False)
if result['status'] == 'Overfitting Detected':
    print("Early stopping recommended!")
```

### Loss Curve Analysis

SmartSummary can analyze your training loss curves to diagnose training health:

**Detects:**
- ✅ **Healthy Convergence** - Loss decreasing and stabilizing properly
- ⚠️ **Overfitting** - Training loss decreasing while validation loss increases
- ⚠️ **Divergence** - Loss increasing (training instability)
- 📊 **Plateau** - Loss stopped improving
- 📈 **Oscillation** - High variance in loss values
- 🐌 **Slow Convergence** - Training progressing too slowly

**Provides:**
- Trend analysis (slope, improvement percentage)
- Stability metrics (coefficient of variation)
- Actionable recommendations (learning rate adjustments, regularization, etc.)

**Quick Example:**
```python
# During training
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20)

# Analyze the training
result = summary.analyze_loss_curve(
    train_losses=history.history['loss'],
    val_losses=history.history['val_loss']
)

print(f"Status: {result['status']}")
print(f"Improvement: {result['metrics']['improvement_percent']:.2f}%")
for rec in result['recommendations']:
    print(f"  • {rec}")
```

See **[Loss Curve Analysis Documentation](./docs/LOSS_CURVE_ANALYSIS.md)** for detailed examples and integration patterns.

### Bottleneck Detection

SmartSummary identifies bottlenecks based on:
- **High parameters**: Layers with >10% of total model parameters
- **High gradient variance**: Indicates training instability (when tracking enabled)
- **Large outputs**: Intermediate tensors >10MB

Each bottleneck gets a score - higher scores indicate more critical optimization opportunities.

---

## ModelView Guide

### Overview

ModelView generates **publication-quality neural network architecture diagrams** for TensorFlow/Keras models, similar to torchview for PyTorch. Perfect for research papers, presentations, and documentation.

**Key Features:**
- 🎨 High-resolution outputs (300-600 DPI) suitable for academic papers
- 📄 Multiple formats: PNG, PDF, SVG
- 🎯 Automatic graph layout with beautiful styling
- 📊 Shows layer types, shapes, and parameter counts
- 🔀 Handles complex architectures (residual, multi-input/output, branching)
- ⚙️ Customizable styling (colors, fonts, layouts)

### Quick Start

```python
from ToTf.tenf import ModelView, draw_graph
import tensorflow as tf

# Create your model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Quickest way - one-liner
draw_graph(model, input_shape=(28, 28, 1), save_path='model.png')
```

### Basic Usage

```python
from ToTf.tenf import ModelView

# Initialize ModelView
view = ModelView(model, input_shape=(28, 28, 1))

# Show text summary
view.show()

# Render high-resolution PNG
view.render('model.png', dpi=300)
```

### Output Formats

```python
# PNG - for presentations and quick viewing
view.render('architecture.png', format='png', dpi=300)

# PDF - for LaTeX documents (vector graphics)
view.render('architecture.pdf', format='pdf')

# SVG - for perfect scaling in any size
view.render('architecture.svg', format='svg')
```

### Layout Options

```python
# Vertical layout (top-to-bottom) - default
view.render('model_vertical.png', rankdir='TB')

# Horizontal layout (left-to-right) - for wide figures
view.render('model_horizontal.png', rankdir='LR')
```

### Customization

```python
# Control what to display
view.render(
    'model_custom.png',
    show_shapes=True,        # Display tensor shapes
    show_layer_names=True,   # Display layer names
    show_params=True,        # Display parameter counts
    dpi=600                  # Extra high resolution
)

# Custom node styling
custom_node_style = {
    'shape': 'box',
    'style': 'rounded,filled',
    'fillcolor': '#f0f0f0',
    'fontname': 'Helvetica',
    'fontsize': '11',
    'color': '#333333',
    'penwidth': '2'
}

# Custom edge styling
custom_edge_style = {
    'color': '#666666',
    'penwidth': '2',
    'arrowsize': '1.0'
}

view.render(
    'model_styled.png',
    node_style=custom_node_style,
    edge_style=custom_edge_style
)
```

### Complex Architectures

**Multi-Input Models:**
```python
# Model with text and image inputs
text_input = keras.Input(shape=(100,), name='text')
image_input = keras.Input(shape=(64, 64, 3), name='image')
# ... build model ...

model = keras.Model(inputs=[text_input, image_input], outputs=outputs)

# Visualize - provide list of input shapes
view = ModelView(model, input_shape=[(100,), (64, 64, 3)])
view.render('multimodal.png', rankdir='LR', dpi=300)
```

**Residual Networks:**
```python
# ResNet-like architecture with skip connections
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(64, 3, padding='same')(inputs)
residual = x
x = layers.Conv2D(64, 3, padding='same')(x)
x = layers.Add()([x, residual])  # Skip connection
# ... more layers ...

model = keras.Model(inputs=inputs, outputs=outputs)
view = ModelView(model, input_shape=(32, 32, 3))
view.render('resnet.png', dpi=400)
```

**Attention/Transformer Models:**
```python
# Transformer with attention mechanism
inputs = keras.Input(shape=(50, 128))
attention = layers.MultiHeadAttention(num_heads=8, key_dim=32)(inputs, inputs)
# ... build transformer ...

view = ModelView(model, input_shape=(50, 128))
view.render('transformer.png', rankdir='TB', dpi=300)
```

### Export and Analysis

```python
# Get summary as dictionary
summary_dict = view.get_summary_dict()
print(f"Total layers: {summary_dict['num_layers']}")
print(f"Total parameters: {summary_dict['total_parameters']:,}")

# Save summary as JSON
view.save_summary_json('model_architecture.json')

# Text-based summary with connections
view.show(detailed=True)
```

### Best Practices for Publications

#### For Research Papers (LaTeX)

```python
# Use PDF format for LaTeX - perfect quality at any zoom
view.render('paper_figure.pdf', format='pdf', dpi=300)

# Or high-DPI PNG if journal requires raster
view.render('paper_figure.png', format='png', dpi=600)
```

#### For Presentations

```python
# Standard resolution PNG
view.render('presentation_arch.png', dpi=300, rankdir='LR')
```

#### For Documentation/Web

```python
# SVG for responsive scaling
view.render('docs_architecture.svg', format='svg')

# Or moderate PNG
view.render('docs_architecture.png', dpi=150)
```

### Installation Requirements

ModelView requires Graphviz:

```bash
# Python package
pip install graphviz

# System graphviz (required!)
# Ubuntu/Debian:
sudo apt-get install graphviz

# macOS:
brew install graphviz

# Windows:
# Download from https://graphviz.org/download/
# Add to PATH after installation
```

### Examples

See `example_modelview_tf.py` for comprehensive examples including:
- Simple Sequential models
- CNN architectures
- ResNet with skip connections
- Multi-input models
- Transformer/Attention models
- Autoencoders
- Custom styling

Run examples:
```bash
cd ToTf
python example_modelview_tf.py
```

All generated diagrams will be in the `outputs/` directory.

---

## Utility Functions Guide

### Auto-Shape Flattener

**Problem**: Transitioning from convolutional layers to dense/linear layers requires calculating exact flattened sizes, which is error-prone.

**Solution**: `lazy_flatten()` and `get_flatten_size()` handle this automatically.

#### PyTorch Example

```python
from ToTf.pytorch import lazy_flatten, get_flatten_size
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)
        
        # Calculate flatten size automatically
        # Input: 32x32 -> Conv: 30x30 -> Pool: 15x15 -> Conv: 13x13 -> Pool: 6x6
        flat_size = get_flatten_size((64, 6, 6))  # 2304
        self.fc = nn.Linear(flat_size, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = lazy_flatten(x)  # Automatically flattens to [batch, 2304]
        return self.fc(x)
```

#### TensorFlow Example

```python
from ToTf.tenf import lazy_flatten, get_flatten_size
from tensorflow import keras

class MyCNN(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(32, 3)
        self.conv2 = keras.layers.Conv2D(64, 3)
        self.pool = keras.layers.MaxPooling2D(2)
        
        # Calculate flatten size automatically (channels-last format)
        flat_size = get_flatten_size((6, 6, 64))  # 2304
        self.fc = keras.layers.Dense(10)
    
    def call(self, x):
        x = self.pool(tf.nn.relu(self.conv1(x)))
        x = self.pool(tf.nn.relu(self.conv2(x)))
        x = lazy_flatten(x)  # Automatically flattens
        return self.fc(x)
```

---

### Normalized Cross-Correlation (NCC) Loss

**Problem**: Medical imaging tasks (registration, segmentation) need losses robust to intensity variations. MSE fails when images have different brightness/contrast.

**Solution**: NCC loss measures structural similarity, invariant to linear intensity transformations.

#### When to Use NCC

- ✅ Medical image registration (MRI, CT scans)
- ✅ Image alignment tasks
- ✅ When images have varying intensity/contrast
- ✅ ACDC, ADNI, and similar medical datasets
- ❌ Classification tasks (use CrossEntropy)
- ❌ When exact pixel values matter

#### PyTorch Example

```python
from ToTf.pytorch import loss_ncc, ncc_score
import torch.nn as nn

# Medical image segmentation model
model = MySegmentationModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    for images, masks in dataloader:
        optimizer.zero_grad()
        
        predictions = model(images)
        loss = loss_ncc(masks, predictions)  # NCC loss
        
        loss.backward()
        optimizer.step()
        
        # Track similarity score (higher is better)
        score = ncc_score(masks, predictions)
        print(f"NCC Score: {score.item():.4f}")
```

#### TensorFlow Example

```python
from ToTf.tenf import NCCLoss, ncc_score

# Method 1: Use in model.compile()
model = MySegmentationModel()
model.compile(
    optimizer='adam',
    loss=NCCLoss(),  # Keras-compatible!
    metrics=['mse']  # Can track MSE for comparison
)

model.fit(x_train, y_train, epochs=10)

# Method 2: Use as function
for images, masks in dataset:
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_ncc(masks, predictions)
        score = ncc_score(masks, predictions)
```

**NCC Characteristics:**
- Returns 0 when images are identical
- Range: 0 to 2 (lower is better for loss)
- Score range: -1 to 1 (higher is better)
- Invariant to linear intensity scaling: `loss_ncc(img, img*2) ≈ 0`

---

### Learning Rate Finder

**Problem**: Finding the optimal learning rate requires trial and error. Too high = divergence, too low = slow training.

**Solution**: LR Finder runs a short test with exponentially increasing LRs to find the "sweet spot".

#### PyTorch Example

```python
from ToTf.pytorch import find_lr, LRFinder
import torch.nn as nn

model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Placeholder LR
criterion = nn.CrossEntropyLoss()

# Method 1: Quick and easy
best_lr = find_lr(model, optimizer, criterion, train_loader)
print(f"Suggested LR: {best_lr}")

# Now use the found LR
optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)

# Method 2: Full control
lr_finder = LRFinder(model, optimizer, criterion, device='cuda')
lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=10, num_iter=100)
lr_finder.plot()  # Shows loss vs LR curve
best_lr = lr_finder.get_best_lr()
```

#### TensorFlow Example

```python
from ToTf.tenf import find_lr, LRFinder
from tensorflow import keras

model = MyModel()
loss_fn = keras.losses.SparseCategoricalCrossentropy()

# Method 1: Quick and easy
best_lr = find_lr(model, loss_fn, train_dataset)
print(f"Suggested LR: {best_lr}")

# Now compile with optimal LR
model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_lr), loss=loss_fn)

# Method 2: Full control
lr_finder = LRFinder(model, loss_fn)
lr_finder.range_test(train_dataset, start_lr=1e-7, end_lr=10, num_iter=100)
lr_finder.plot()  # Shows loss vs LR curve
best_lr = lr_finder.get_best_lr()
```

**How it Works:**
1. Starts with a very small LR (e.g., 1e-7)
2. Trains for a few iterations, gradually increasing LR
3. Records loss at each LR
4. Finds the LR where loss decreases fastest (steepest gradient)
5. Stops if loss starts diverging

**Best LR Selection:**
- The tool suggests the LR at the steepest negative gradient
- This is typically 1/10th to 1/3 of the LR where loss starts increasing
- Always visually inspect the plot for confirmation

---

### Cross-Framework Reproducibility (`seed_everything`)

**Problem**: Achieving full determinism and reproducibility across PyTorch, TensorFlow, NumPy, and Python usually requires 10+ lines of framework-specific boilerplate, device flags, and cuBLAS configurations.

**Solution**: `seed_everything()` configures all random number generators, backend states, and hardware determinism flags in a single call.

```python
from ToTf import seed_everything

# Basic reproducible seeding (Python, NumPy, PyTorch CPU/CUDA, TensorFlow)
seed_everything(42)

# Strict deterministic execution for scientific benchmarking
seed_everything(42, deterministic=True, verbose=True)
```

---

### Tensor Health & Diagnostics (`inspect_tensor`)

**Problem**: When debugging unexpected model outputs, vanishing gradients, or shape mismatches, printing a tensor only displays raw numbers or truncated floats, requiring multiple manual `.shape`, `.dtype`, `torch.isnan()`, `.mean()`, and `.std()` calls.

**Solution**: `inspect_tensor()` provides a comprehensive one-line diagnostic health profile for PyTorch, TensorFlow, and NumPy tensors.

```python
from ToTf import inspect_tensor
import torch

x = torch.randn(32, 64, 128)
stats = inspect_tensor(x, name="encoder_activations")
```

**Output:**
```
======================================================================
[TENSOR HEALTH CHECK] 'encoder_activations'
======================================================================
Framework:   PyTorch
Shape:       (32, 64, 128)
Dtype:       torch.float32
Device:      cpu
Memory:      1.000 MB
Elements:    262,144
NaNs:        0 (0.0000%) [OK]
Infs:        0 (0.0000%) [OK]
Zeros:       0 (0.0000%)
Min / Max:   -4.521849 / 4.678121
Mean / Std:  -0.000142 / 0.999812
Quantiles:   p25=-0.674912, p50=0.000185, p75=0.674391
======================================================================
```

---

### Transfer Learning & Fine-Tuning Toolkit

**Problem**: Freezing backbones or fine-tuning heads in PyTorch requires navigating parameter generators and setting `param.requires_grad = False`. In TensorFlow/Keras, it requires setting `layer.trainable = False`. Furthermore, PyTorch `BatchNorm` layers continue updating running mean and variance in `.train()` mode even if affine parameters are frozen!

**Solution**: Framework-agnostic `freeze_layers()`, `unfreeze_layers()`, and `freeze_batchnorm()`.

```python
from ToTf import freeze_layers, unfreeze_layers, freeze_batchnorm

# Freeze backbone layers up to index 8, or matching a name pattern
freeze_layers(model, up_to=8)
freeze_layers(model, regex=r"(conv|block[1-3])")

# Unfreeze only the final classifier head
unfreeze_layers(model, last_n=1)

# Lock BatchNorm weights AND running statistics during fine-tuning
freeze_batchnorm(model, freeze_eval=True)
```

---

### Focal Loss for Class Imbalance

**Problem**: Standard Cross-Entropy is overwhelmed by vast numbers of easily classified background/negative examples in tasks like object detection, anomaly detection, or medical triage.

**Solution**: `loss_focal()` and `FocalLoss` dynamically scale the cross-entropy loss, down-weighting easy examples to focus gradient updates on hard, confusing examples.

```python
from ToTf import loss_focal, FocalLoss

# Functional usage
loss = loss_focal(y_true, y_pred, alpha=0.25, gamma=2.0)

# PyTorch Module / Keras Loss Layer usage
criterion = FocalLoss(alpha=0.25, gamma=2.0)
loss = criterion(y_true, y_pred)
```

---

### Dice Loss & Segmentation Metrics

**Problem**: Pixel-wise cross-entropy struggles with severe foreground-background imbalance in semantic/biomedical segmentation (e.g. small tumors or lesions covering <1% of the image).

**Solution**: Differentiable soft `loss_dice()` / `DiceLoss`, alongside `dice_score()` and `iou_score()` (Jaccard Index) evaluation metrics.

```python
from ToTf import loss_dice, DiceLoss, dice_score, iou_score

# Differentiable loss for backpropagation
loss = loss_dice(masks_true, masks_pred, smooth=1e-6)

# Validation evaluation metrics
dsc = dice_score(masks_true, masks_pred)  # Dice Similarity Coefficient
iou = iou_score(masks_true, masks_pred)   # Jaccard Index (IoU)
print(f"Validation Dice: {dsc:.4f} | IoU: {iou:.4f}")
```

---

### Numerical Anomaly Tracking (`AnomalyDetector`)

**Problem**: When NaN or Inf values appear in a deep network, PyTorch and TensorFlow typically do not fail until the loss or backward pass diverges, leaving you with zero visibility into which layer or tensor first went unstable.

**Solution**: `AnomalyDetector` attaches non-intrusive hooks to all layers, recording the exact layer name, stage (input, output, or backward gradient), tensor shape, and NaN/Inf count where instability first occurs.

```python
from ToTf import AnomalyDetector, check_anomalies

# Fast single-tensor check
has_anomaly = check_anomalies(tensor, name="latent_embedding")

# Deep layer-by-layer anomaly interceptor
detector = AnomalyDetector(model, check_inputs=True, check_outputs=True, check_gradients=True)

# Run forward & backward passes normally
output = model(inputs)
loss = criterion(output, targets)
loss.backward()

# Inspect any detected anomalies
if detector.has_anomalies():
    print(detector.summary())
    for anomaly in detector.anomalies:
        print(f"Layer '{anomaly['layer_name']}' failed at {anomaly['stage']} with {anomaly['nan_count']} NaNs!")

# Clean up hooks after debugging
detector.remove_hooks()
```

---

### Publication-Ready Model Profiling (`benchmark_model`)

**Problem**: Writing research papers and technical documentation requires reporting parameter counts, memory footprint, FLOPs/MACs, latency percentiles, and throughput. Setting up CUDA warmup, timing loops, percentiles, and converting results into Markdown and LaTeX tables is repetitive and error-prone.

**Solution**: `benchmark_model()` benchmarks models with warmup, calculates parameter and computational complexity, and outputs copy-paste ready Markdown and LaTeX tables.

```python
from ToTf import benchmark_model

report = benchmark_model(
    model,
    input_shape=(8, 3, 64, 64),
    num_runs=50,
    warmup=10,
    device="cpu",      # or 'cuda'
    print_table=True
)

# Access computed metrics
print(f"FLOPs: {report['flops_g']:.3f} G | Throughput: {report['fps']:.1f} FPS")

# Camera-ready LaTeX table for Overleaf / paper submission:
print(report['latex_table'])

# Markdown table for GitHub / Notion documentation:
print(report['markdown_table'])
```

---

### Direct In-Memory Weight Transfer (`transfer_weights`)

**Problem**: Porting a trained PyTorch model to TensorFlow/Keras (or vice-versa) usually requires saving to ONNX, writing custom conversion scripts, or manually transposing convolution weights and mapping BatchNorm parameters.

**Solution**: `transfer_weights()` transfers compatible parameters directly in-memory between PyTorch and TensorFlow models without disk I/O, automatically handling convolution kernel transpositions (`[out, in, H, W]` vs `[H, W, in, out]`), dense layer transpositions (`[out, in]` vs `[in, out]`), and aligning BatchNorm weights, biases, running means, and running variances.

```python
from ToTf import transfer_weights

# PyTorch -> TensorFlow / Keras
transfer_weights(torch_model, keras_model, verbose=True)

# TensorFlow / Keras -> PyTorch
transfer_weights(keras_model, torch_model, verbose=True)

# Optional custom layer name mapping for non-sequential architectures:
transfer_weights(
    source_model,
    target_model,
    layer_mapping={"resnet_conv1": "features.0", "classifier": "head"}
)
```

---

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

### Utility Functions

#### lazy_flatten

```python
# PyTorch
lazy_flatten(tensor: torch.Tensor, start_dim: int = 1) -> torch.Tensor

# TensorFlow
lazy_flatten(tensor: tf.Tensor, start_dim: int = 1) -> tf.Tensor
```

**Parameters:**
- `tensor`: Input tensor to flatten
- `start_dim`: Dimension to start flattening from (default: 1, preserves batch)

**Returns:** Flattened tensor

---

#### get_flatten_size

```python
get_flatten_size(input_shape: Tuple[int, ...]) -> int
```

**Parameters:**
- `input_shape`: Shape of tensor (excluding batch dimension)

**Returns:** Total flattened size

**Example:**
- PyTorch: `get_flatten_size((64, 7, 7))` → 3136
- TensorFlow: `get_flatten_size((7, 7, 64))` → 3136

---

#### loss_ncc

```python
# PyTorch
loss_ncc(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor

# TensorFlow
@tf.function
loss_ncc(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float = 1e-8) -> tf.Tensor
```

**Parameters:**
- `y_true`: Ground truth tensor
- `y_pred`: Predicted tensor
- `eps`: Small constant for numerical stability (default: 1e-8)

**Returns:** NCC loss value (range: 0 to 2, lower is better)

---

#### ncc_score

```python
ncc_score(y_true, y_pred, eps: float = 1e-8)
```

**Parameters:** Same as `loss_ncc`

**Returns:** NCC similarity score (range: -1 to 1, higher is better)

**Note:** `ncc_score = 1.0 - loss_ncc`

---

#### NCCLoss (TensorFlow only)

```python
NCCLoss(eps: float = 1e-8, name: str = "ncc_loss")
```

Keras-compatible NCC Loss class that can be used with `model.compile()`.

**Example:**
```python
model.compile(optimizer='adam', loss=NCCLoss())
```

---

#### find_lr

```python
# PyTorch
find_lr(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: Callable,
    dataloader: DataLoader,
    device: str = 'cpu',
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    num_iter: int = 100,
    plot: bool = True
) -> float

# TensorFlow
find_lr(
    model: keras.Model,
    loss_fn: Loss,
    dataset: tf.data.Dataset,
    optimizer: Optional[Optimizer] = None,
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    num_iter: int = 100,
    plot: bool = True
) -> float
```

**Parameters:**
- `model`: Model to analyze
- `optimizer/loss_fn`: Optimizer (PyTorch) or loss function (TensorFlow)
- `dataloader/dataset`: Training data
- `start_lr`: Starting LR (default: 1e-7)
- `end_lr`: Ending LR (default: 10.0)
- `num_iter`: Number of iterations (default: 100)
- `plot`: Show plot (default: True)

**Returns:** Suggested optimal learning rate

---

#### LRFinder

```python
# PyTorch
LRFinder(model, optimizer, criterion, device='cpu')

# TensorFlow
LRFinder(model, loss_fn, optimizer=None)
```

**Methods:**
- `range_test(dataloader/dataset, start_lr, end_lr, num_iter, ...)`: Run LR range test
- `plot(skip_start, skip_end, log_lr, show_best, save_path)`: Plot results
- `get_best_lr()`: Get suggested learning rate

---

#### seed_everything

```python
seed_everything(seed: int = 42, deterministic: bool = False, benchmark: bool = False, verbose: bool = False)
```

**Parameters:**
- `seed` (int): Master seed value (default: 42).
- `deterministic` (bool): Enable deterministic algorithms across PyTorch & TensorFlow (may impact performance).
- `benchmark` (bool): PyTorch cuDNN benchmark setting.
- `verbose` (bool): Print seeding confirmation status.

---

#### inspect_tensor

```python
inspect_tensor(tensor: Any, name: Optional[str] = None, print_info: bool = True, quantiles: bool = True) -> Dict[str, Any]
```

**Parameters:**
- `tensor`: Input PyTorch tensor, TensorFlow tensor, or NumPy array.
- `name` (str, optional): Custom display name for debugging.
- `print_info` (bool): Print formatted summary table to console.
- `quantiles` (bool): Compute 25th, 50th (median), and 75th percentiles.

**Returns:** Dictionary with keys `shape`, `dtype`, `device`, `numel`, `memory_mb`, `has_nan`, `nan_count`, `has_inf`, `inf_count`, `zero_pct`, `min`, `max`, `mean`, `std`, and optional `quantiles`.

---

#### freeze_layers / unfreeze_layers / freeze_batchnorm

```python
freeze_layers(model: Any, up_to: Optional[int] = None, layer_names: Optional[List[str]] = None, layer_types: Optional[Tuple[Type, ...]] = None, regex: Optional[str] = None) -> int

unfreeze_layers(model: Any, last_n: Optional[int] = None, layer_names: Optional[List[str]] = None, layer_types: Optional[Tuple[Type, ...]] = None, regex: Optional[str] = None) -> int

freeze_batchnorm(model: Any, freeze_eval: bool = True) -> int
```

**Parameters:**
- `model`: PyTorch `nn.Module` or TensorFlow `tf.keras.Model`.
- `up_to`: Index up to which layers should be frozen (exclusive).
- `last_n`: Count of trailing layers to unfreeze.
- `layer_names`: Explicit list of layer names to match.
- `layer_types`: Tuple of layer classes/types to match (e.g. `(nn.Conv2d, nn.Linear)`).
- `regex`: Regex pattern matching layer names.
- `freeze_eval` (bool): For PyTorch, force BatchNorm modules into eval mode and prevent running statistics from updating.

**Returns:** Count of modified layers/parameters.

---

#### loss_focal / FocalLoss

```python
loss_focal(y_true: Any, y_pred: Any, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean') -> Any

FocalLoss(alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean')
```

**Parameters:**
- `alpha` (float): Weighting factor for rare/positive class (default: 0.25).
- `gamma` (float): Focusing parameter for hard examples (default: 2.0).
- `reduction` (str): `'mean'`, `'sum'`, or `'none'` (default: `'mean'`).

---

#### loss_dice / DiceLoss / dice_score / iou_score

```python
loss_dice(y_true: Any, y_pred: Any, smooth: float = 1e-6) -> Any
DiceLoss(smooth: float = 1e-6)

dice_score(y_true: Any, y_pred: Any, smooth: float = 1e-6) -> float
iou_score(y_true: Any, y_pred: Any, smooth: float = 1e-6) -> float
```

**Parameters:**
- `y_true`, `y_pred`: Ground truth masks and predicted probability/activation masks.
- `smooth` (float): Smoothing epsilon to prevent division by zero (default: 1e-6).

---

#### AnomalyDetector / check_anomalies

```python
check_anomalies(tensor: Any, name: Optional[str] = None) -> bool

AnomalyDetector(model: Any, check_inputs: bool = True, check_outputs: bool = True, check_gradients: bool = True)
```

**Methods:**
- `has_anomalies() -> bool`: Returns True if any NaN or Inf was captured.
- `summary() -> str`: Formatted report of all anomalies encountered.
- `reset()`: Clear captured anomaly history.
- `remove_hooks()`: Detach all hooks from the model.

---

#### benchmark_model

```python
benchmark_model(
    model: Any,
    input_shape: Tuple[int, ...],
    num_runs: int = 50,
    warmup: int = 10,
    device: str = 'cpu',
    print_table: bool = True
) -> Dict[str, Any]
```

**Parameters:**
- `model`: PyTorch `nn.Module` or TensorFlow `tf.keras.Model`.
- `input_shape`: Input tensor shape including or excluding batch (e.g. `(8, 3, 224, 224)`).
- `num_runs` (int): Timed evaluation runs (default: 50).
- `warmup` (int): Untimed warmup iterations to initialize caches/CUDA kernels (default: 10).
- `device` (str): `'cpu'` or `'cuda'` (default: `'cpu'`).
- `print_table` (bool): Pretty-print console summary.

**Returns:** Dictionary with `total_params`, `trainable_params`, `param_memory_mb`, `macs`, `flops`, `latency_mean_ms`, `latency_std_ms`, `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms`, `fps`, `markdown_table`, and `latex_table`.

---

#### transfer_weights

```python
transfer_weights(
    source_model: Any,
    target_model: Any,
    layer_mapping: Optional[Dict[str, str]] = None,
    verbose: bool = True
) -> Dict[str, int]
```

**Parameters:**
- `source_model`: Source model (PyTorch `nn.Module` or TensorFlow `tf.keras.Model`).
- `target_model`: Target model (PyTorch `nn.Module` or TensorFlow `tf.keras.Model`).
- `layer_mapping` (dict, optional): Custom mapping from source layer names to target layer names.
- `verbose` (bool): Print layer mapping and transfer details.

**Returns:** Dictionary with `transferred_layers` and `transferred_params` counts.

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
- [example_utils_pytorch.py](example_utils_pytorch.py) - Utility functions examples (5 scenarios: lazy_flatten, NCC loss, LR finder, complete pipeline)

### TensorFlow Examples
- [example_smartsummary_tf.py](example_smartsummary_tf.py) - SmartSummary examples for TensorFlow/Keras (8 detailed scenarios including multi-input models, gradient tracking, and MobileNetV2 analysis)
- [example_utils_tf.py](example_utils_tf.py) - Utility functions examples (6 scenarios: lazy_flatten, NCC loss with Keras, LR finder, Functional API usage)

### Cross-Framework & Research Examples
- [examples/example_phase1_qol.py](examples/example_phase1_qol.py) - Phase 1: Cross-framework seeding, tensor health check, layer and BatchNorm freezing
- [examples/example_phase2_diagnostics_losses.py](examples/example_phase2_diagnostics_losses.py) - Phase 2: Focal Loss, Dice Loss & IoU metrics, and AnomalyDetector for pinpointing first-NaN/Inf origins
- [examples/example_phase3_benchmarking_transfer.py](examples/example_phase3_benchmarking_transfer.py) - Phase 3: Paper-ready benchmark profiling (FLOPs/latency/Markdown/LaTeX tables) and direct in-memory PyTorch <-> TensorFlow weight transfer

---

## Testing

Run the test suites to verify everything works:

```bash
# Test Phase 1 Quality-of-Life & Transfer Learning Utilities
pytest test/test_phase1_utils.py -v

# Test Phase 2 Advanced Diagnostics & Losses
pytest test/test_phase2_utils.py -v

# Test Phase 3 Benchmarking & Weight Transfer
pytest test/test_phase3_utils.py -v

# Run full test suite across all modules (119+ tests)
pytest test/ -v
```

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
├── backend_manager.py              # Dynamic component dispatching & registration
├── requirements.txt                # Dependencies
├── setup.py                        # Package config
│
├── Documentation/
│   ├── README.md                   # This file (main documentation)
│   ├── QUICKSTART_MODELVIEW.md     # ModelView quick start guide
│   ├── MODELVIEW_IMPLEMENTATION_SUMMARY.md  # ModelView technical details
│   ├── UTILITIES_IMPLEMENTATION_SUMMARY.md  # Utility functions documentation
│   └── TENSORFLOW_IMPLEMENTATION_SUMMARY.md # TensorFlow implementation notes
│
├── pytorch/
│   ├── __init__.py
│   ├── trainingmonitor.py         # TrainingMonitor implementation
│   ├── smartsummary.py            # SmartSummary implementation (PyTorch)
│   └── utils.py                   # Utilities (seed, inspect, freeze, focal, dice, anomaly, benchmark, transfer)
│
├── tenf/
│   ├── __init__.py
│   ├── smartsummary.py            # SmartSummary implementation (TensorFlow)
│   ├── modelview.py               # ModelView for architecture diagrams
│   └── utils.py                   # Utilities (seed, inspect, freeze, focal, dice, anomaly, benchmark, transfer)
│
├── examples/
│   ├── example_usage.py           # TrainingMonitor examples
│   ├── example_smartsummary.py    # SmartSummary examples (PyTorch)
│   ├── example_smartsummary_tf.py # SmartSummary examples (TensorFlow)
│   ├── example_modelview_tf.py    # ModelView examples (8+ examples)
│   ├── example_phase1_qol.py      # Seeding, inspect_tensor, layer & BN freezing
│   ├── example_phase2_diagnostics_losses.py # Focal loss, Dice, IoU, AnomalyDetector
│   ├── example_phase3_benchmarking_transfer.py # Model benchmarking & weight transfer
│   └── demo_complex_architectures.py # Complex architecture demos
│
├── test/
│   ├── test_monitor.py            # TrainingMonitor tests
│   ├── test_integration.py        # Integration tests
│   ├── test_smartsummary.py       # SmartSummary tests (PyTorch)
│   ├── test_smartsummary_tf.py    # SmartSummary tests (TensorFlow)
│   ├── test_modelview_tf.py       # ModelView tests (41 tests, 9 classes)
│   ├── test_utils_pytorch.py      # Utility functions tests (PyTorch)
│   ├── test_utils_tf.py           # Utility functions tests (TensorFlow)
│   ├── test_utils_integration.py  # Cross-framework integration tests
│   ├── test_phase1_utils.py       # Phase 1 tests (19 tests)
│   ├── test_phase2_utils.py       # Phase 2 tests (6 tests)
│   └── test_phase3_utils.py       # Phase 3 tests (4 tests)
│
└── verify_modelview.py            # Quick ModelView verification script
```

---

## License

See [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting PRs.

```bash
# Run all tests
pytest test/ -v
```

---

## What's New

**v0.3.0 - Research & Production Deep Learning Toolkit (Phases 1, 2, & 3)** 🚀
- ✅ **Cross-Framework Reproducibility**: `seed_everything()` sets unified seeds, determinism flags, and hardware workspace configs for Python, NumPy, PyTorch (CPU/CUDA), and TensorFlow.
- ✅ **Tensor Health & Diagnostics**: `inspect_tensor()` provides instant one-line profiling (shape, dtype, memory, NaNs, Infs, zero sparsity, quantiles).
- ✅ **Transfer Learning Toolkit**: `freeze_layers()`, `unfreeze_layers()`, and `freeze_batchnorm()` for seamless backbone locking across PyTorch and Keras.
- ✅ **Focal Loss**: `loss_focal()` and `FocalLoss` for imbalanced dense classification and object detection.
- ✅ **Dice Loss & IoU**: `loss_dice()`, `DiceLoss`, `dice_score()`, and `iou_score()` for segmentation tasks.
- ✅ **First-NaN/Inf Anomaly Interceptor**: `AnomalyDetector` and `check_anomalies()` track exactly which layer and forward/backward stage first produces numerical instability.
- ✅ **Camera-Ready Model Benchmarking**: `benchmark_model()` measures FLOPs, MACs, latency percentiles (p50, p95, p99), and throughput (FPS), auto-generating ready-to-publish Markdown and LaTeX tables.
- ✅ **Direct In-Memory Weight Transfer**: `transfer_weights()` transfers weights between PyTorch and TensorFlow/Keras without disk I/O or ONNX overhead, automatically handling 2D conv kernel and dense weight matrix transpositions.

**v0.2.3 ONNX Support**
- ✅ **toONNX** and **fromONNX** have been added to support ONNX
**v0.2.2 Connection Extraction Fix** 🔧
- ✅ **Fixed connection extraction** - Edges now properly displayed in complex architectures
- ✅ **Improved graph visualization** - Parallel branches, cross-connections, and merge points now visible
- ✅ **Keras API compatibility** - Updated to work with latest TensorFlow/Keras node structure
- ✅ **Verified with 41 tests** - All architectures correctly visualized

**v0.2.1 - Enhanced Testing**
- ✅ **Extended test coverage** - Added 9 new tests for complex architectures
- ✅ **Multiple branches & cross-connections** - Inception, DenseNet, DAG structures
- ✅ **Advanced topologies** - Parallel branches, skip connections, multi-output
- ✅ **Demo examples** - 5 complex architecture visualization demos
- ✅ **Test suite expanded** - 41 total tests

**v0.2.0 - ModelView Release**
- ✅ **ModelView for TensorFlow** - Publication-quality architecture diagrams
- ✅ High-resolution outputs (PNG, PDF, SVG) at 300-600 DPI
- ✅ Support for complex architectures (ResNet, multi-input, attention)
- ✅ Full documentation and 8+ examples
- ✅ Cleaner layer labels (just type names, no redundant info)

**Previous Features:**
- SmartSummary with bottleneck detection
- TrainingMonitor with progress tracking
- Utility functions (NCC loss, LR finder, auto-flatten)


