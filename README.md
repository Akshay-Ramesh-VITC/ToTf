# ToTf

A Cross-Library Compatible Library for adding more features for ease of use, which are not available directly in PyTorch or TensorFlow.

## Features

### TrainingMonitor
A powerful training monitor that provides:
- **Real-time progress tracking** with `tqdm` integration
- **Automatic metrics logging** to CSV files
- **Running averages** of metrics (Keras-style)
- **Resource monitoring** (RAM and VRAM usage)
- **Crash-resistant logging** with auto-flush

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from ToTf import TrainingMonitor

# Your training loop
for epoch in range(epochs):
    monitor = TrainingMonitor(
        train_loader, 
        desc=f"Epoch {epoch + 1}", 
        log_file="training.csv"
    )
    
    for batch in monitor:
        # Your training code
        loss = train_step(batch)
        
        # Log metrics
        monitor.log({'loss': loss.item()})
```

## API Reference

### TrainingMonitor

**Parameters:**
- `iterable`: The data loader or iterable to monitor
- `desc` (str): Description shown in progress bar (default: "Training")
- `log_file` (str): CSV file path for logging (default: "train_log.csv")

**Methods:**
- `log(metrics: Dict[str, float])`: Log metrics and update running averages

**Logged Data:**
- Timestamp
- Step number
- All logged metrics (with running averages)
- RAM usage percentage
- VRAM usage in GB (if CUDA available)

## Example

See [example_usage.py](example_usage.py) for a complete training example.

## Backend Detection

ToTf automatically detects whether PyTorch or TensorFlow is installed and uses the appropriate backend.

```python
from ToTf import get_backend
print(get_backend())  # Returns 'torch' or 'tensorflow'
```

## License

See [LICENSE](LICENSE) file for details.
