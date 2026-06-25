# PyTorch Module for ToTf

This module contains PyTorch-specific implementations.

## Modules

### TrainingMonitor

A comprehensive training monitor that integrates seamlessly with PyTorch training loops.

### SmartSummary

Advanced model analysis tool that goes beyond basic `model.summary()` with bottleneck detection and gradient tracking.

---

## TrainingMonitor

A comprehensive training monitor that integrates seamlessly with PyTorch training loops.

### Features
- Real-time progress bars with metric display
- Automatic CSV logging with timestamps
- Running averages for all metrics
- RAM and VRAM monitoring
- Crash-resistant (auto-flush)

### Usage Example

```python
from ToTf import TrainingMonitor

# Wrap your DataLoader
epochs = 5

for epoch in range(epochs):
    monitor = TrainingMonitor(
        train_loader,
        desc=f"Epoch {epoch + 1}",
        log_file=f"train_log_epoch_{epoch + 1}.csv"
    )
    
    # Iterate through batches
    for batch in monitor:
        # Your training logic here
        loss = training_step(batch)
        
        # Log metrics
        monitor.log({
            'loss': loss.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
```

### CSV Output Format

The log file contains:
- `timestamp`: Time of logging
- `step`: Current step number  
- `<metric_name>`: Your logged metrics (running average)
- `RAM_pct`: RAM usage percentage
- `VRAM_gb`: GPU memory usage in GB

### Notes

- The monitor automatically tracks running averages of all metrics
- Metrics are flushed to disk after each log call for crash safety
- VRAM is only logged when CUDA is available
- Compatible with any PyTorch DataLoader or iterable

---

## SmartSummary

Advanced model summary with bottleneck detection and gradient analysis.

### Features
- **Comprehensive Analysis**: Shows layer types, shapes, and parameters
- **Bottleneck Detection**: Identifies layers that may slow down training
- **Gradient Tracking**: Monitors gradient variance to find unstable layers
- **Memory Estimation**: Calculates memory usage per layer
- **Export Options**: Save to file or export as dictionary
- **Works with Complex Models**: Handles nested architectures and residual connections

### Usage Example

```python
from ToTf import SmartSummary
import torch.nn as nn

# Create your model
model = YourModel()

# Basic analysis
summary = SmartSummary(model, input_size=(3, 224, 224))
summary.show()

# With gradient tracking
summary = SmartSummary(
    model, 
    input_size=(3, 224, 224),
    track_gradients=True  # Requires backward pass
)
summary.show()

# Get bottlenecks programmatically
bottlenecks = summary.get_bottlenecks(top_n=5)
for bn in bottlenecks:
    print(f"Bottleneck: {bn['layer']}")
    print(f"  Score: {bn['score']:.2f}")
    print(f"  Reasons: {', '.join(bn['reasons'])}")
    print(f"  Parameters: {bn['params']:,}")

# Export analysis
summary.save_to_file("model_analysis.txt")
data = summary.to_dict()
```

### Parameters

- `model` (nn.Module): PyTorch model to analyze
- `input_size` (Tuple, optional): Input tensor shape excluding batch dimension
  - Example: `(3, 224, 224)` for RGB images
  - If omitted, only parameter counting is performed
- `batch_size` (int): Batch size for shape inference (default: 1)
- `device` (str): Device for analysis - 'cpu' or 'cuda' (default: 'cpu')
- `track_gradients` (bool): Whether to track gradient statistics (default: False)
  - Requires a forward and backward pass
  - Useful for identifying training instabilities

### Methods

- `show(show_bottlenecks=True)`: Display formatted summary table
- `get_bottlenecks(top_n=5)`: Get list of bottleneck layers
- `to_dict()`: Export complete analysis as dictionary
- `save_to_file(filename)`: Save summary to text file

### Bottleneck Detection

SmartSummary identifies bottlenecks based on:

1. **Parameter Count**: Layers with >10% of total parameters
2. **Gradient Variance**: High variance indicates potential instability
3. **Output Size**: Large intermediate tensors (>10MB)

Each bottleneck is scored and ranked. Higher scores indicate more critical bottlenecks.

### Output Format

The summary table shows:
- **Layer (type)**: Layer class name
- **Output Shape**: Tensor dimensions after this layer
- **Param #**: Number of parameters in this layer
- **Trainable**: Whether layer has trainable parameters (✓/✗)
- **Gradient Stats** (if tracking enabled): Variance, mean, and max

### Comparison with Other Tools

| Feature | SmartSummary | torchsummary | torchinfo | model.summary() |
|---------|--------------|--------------|-----------|-----------------|
| Basic layer info | ✓ | ✓ | ✓ | ✓ |
| Bottleneck detection | ✓ | ✗ | ✗ | ✗ |
| Gradient tracking | ✓ | ✗ | ✗ | ✗ |
| Memory estimation | ✓ | ✓ | ✓ | ✗ |
| Export to file/dict | ✓ | ✗ | ✓ | ✗ |
| Complex models | ✓ | Limited | ✓ | Limited |
| PyTorch native | ✓ | ✓ | ✓ | N/A (TF) |

### Advanced Example

```python
# Compare different architectures
models = {
    "ResNet18": resnet18(),
    "VGG16": vgg16(),
    "EfficientNet": efficientnet_b0()
}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Analyzing {name}")
    print(f"{'='*50}")
    
    summary = SmartSummary(model, input_size=(3, 224, 224))
    bottlenecks = summary.get_bottlenecks(top_n=3)
    
    print(f"Total parameters: {summary.total_params:,}")
    print(f"Top bottleneck: {bottlenecks[0]['layer']}")
    print(f"  Score: {bottlenecks[0]['score']:.2f}")
```

### Notes

- The monitor automatically tracks running averages of all metrics
- Metrics are flushed to disk after each log call for crash safety
- VRAM is only logged when CUDA is available
- Compatible with any PyTorch DataLoader or iterable
- For very large models, omit `input_size` to skip forward pass
- Gradient tracking adds overhead - use during model design phase
- Works with models containing skip connections, attention mechanisms, etc.
