# PyTorch Module for ToTf

This module contains PyTorch-specific implementations.

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
