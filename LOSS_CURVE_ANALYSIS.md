# Loss Curve Analysis Feature

## Overview

The `analyze_loss_curve()` method has been added to SmartSummary in both TensorFlow and PyTorch implementations. This feature provides intelligent analysis of training dynamics by examining loss curves and identifying patterns that indicate training health.

## Features

The loss curve analyzer can detect and diagnose:

- **✅ Convergence**: Healthy decreasing and stabilizing loss
- **⚠️ Divergence**: Increasing loss indicating training instability
- **📈 Oscillation**: High variance suggesting learning rate or batch size issues
- **📊 Plateau**: Loss has stopped improving
- **🔴 Overfitting**: Training loss decreasing while validation loss increases
- **🐌 Slow Convergence**: Training progressing but very slowly

## Usage

### Basic Usage

```python
# PyTorch
from pytorch.smartsummary import SmartSummary

model = YourModel()
summary = SmartSummary(model, input_size=(3, 224, 224))

# Training loop collects losses
train_losses = [2.5, 2.1, 1.8, 1.6, 1.5, 1.45, 1.42, 1.41]
val_losses = [2.6, 2.2, 1.9, 1.7, 1.65, 1.62, 1.60, 1.59]

# Analyze the loss curve
result = summary.analyze_loss_curve(train_losses, val_losses)
```

```python
# TensorFlow
from tenf.smartsummary import SmartSummary

model = YourKerasModel()
summary = SmartSummary(model, input_shape=(224, 224, 3))

# From Keras training history
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)

result = summary.analyze_loss_curve(
    train_losses=history.history['loss'],
    val_losses=history.history['val_loss']
)
```

### Parameters

- `train_losses` (List[float]): **Required**. List of training loss values over epochs
- `val_losses` (List[float], optional): Validation loss values over epochs
- `window_size` (int, default=5): Moving average window for smoothing
- `verbose` (bool, default=True): Whether to print detailed analysis

### Return Value

Returns a dictionary with:

```python
{
    'status': str,              # Overall training status
    'trend': str,               # Direction of loss movement
    'stability': str,           # Measure of oscillation
    'recommendations': List[str],  # Actionable suggestions
    'metrics': {                # Detailed statistics
        'n_epochs': int,
        'initial_loss': float,
        'final_loss': float,
        'improvement_percent': float,
        'trend_slope': float,
        'coefficient_of_variation': float,
        # ... more metrics
    }
}
```

## Examples

### Example 1: Detecting Overfitting

```python
train_losses = [2.5, 2.0, 1.6, 1.3, 1.1, 0.9, 0.75, 0.62]
val_losses = [2.6, 2.1, 1.8, 1.7, 1.75, 1.85, 1.95, 2.1]

result = summary.analyze_loss_curve(train_losses, val_losses)
# Output: Status = "Overfitting Detected"
# Recommendations include: early stopping, regularization, more data
```

### Example 2: Detecting Divergence

```python
train_losses = [2.5, 2.8, 3.2, 3.9, 4.8, 6.1, 8.2, 11.5]

result = summary.analyze_loss_curve(train_losses)
# Output: Status = "Diverging"
# Recommendations include: reduce learning rate, gradient clipping, check data
```

### Example 3: Programmatic Use (No Printing)

```python
result = summary.analyze_loss_curve(train_losses, val_losses, verbose=False)

if result['status'] == 'Overfitting Detected':
    print("Stopping training early!")
    break

if result['metrics']['improvement_percent'] < 1.0:
    print("Minimal improvement - consider adjusting hyperparameters")
```

## Interpretation Guide

### Status Messages

| Status | Meaning | Action Required |
|--------|---------|-----------------|
| **Converging Well** | Training is healthy | Continue training, monitor validation |
| **Converging Slowly** | Progress is minimal | Consider increasing learning rate |
| **Diverging** | Loss increasing | Reduce LR, check gradients, verify data |
| **Oscillating** | High variance | Reduce LR, increase batch size |
| **Plateau** | No improvement | Lower LR, check model capacity |
| **Overfitting Detected** | Val loss increasing | Use early stopping, add regularization |

### Metrics Explained

- **improvement_percent**: How much loss decreased from start to end (higher is better)
- **trend_slope**: Rate of loss change (negative = decreasing, positive = increasing)
- **coefficient_of_variation**: Loss stability (lower = more stable, <5% is excellent)
- **recent_slope**: Trend in last 30% of training (shows current direction)

## Integration Examples

### With PyTorch Training Loop

```python
model = MyModel()
summary = SmartSummary(model, input_size=(3, 224, 224))
optimizer = torch.optim.Adam(model.parameters())

train_losses = []
val_losses = []

for epoch in range(50):
    # Training
    train_loss = train_epoch(model, train_loader, optimizer)
    train_losses.append(train_loss)
    
    # Validation
    val_loss = validate_epoch(model, val_loader)
    val_losses.append(val_loss)
    
    # Analyze every 5 epochs
    if (epoch + 1) % 5 == 0:
        result = summary.analyze_loss_curve(train_losses, val_losses, verbose=False)
        print(f"Epoch {epoch+1}: {result['status']}")
        
        # Early stopping on overfitting
        if result['status'] == 'Overfitting Detected':
            print("Overfitting detected - stopping training")
            break
```

### With TensorFlow/Keras Callback

```python
class LossAnalysisCallback(keras.callbacks.Callback):
    def __init__(self, summary):
        self.summary = summary
        self.train_losses = []
        self.val_losses = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.train_losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        
        if (epoch + 1) % 5 == 0:
            result = self.summary.analyze_loss_curve(
                self.train_losses, 
                self.val_losses, 
                verbose=False
            )
            print(f"\n=== Loss Analysis (Epoch {epoch+1}) ===")
            print(f"Status: {result['status']}")
            print(f"Improvement: {result['metrics']['improvement_percent']:.2f}%")

# Use the callback
model = create_model()
summary = SmartSummary(model, input_shape=(224, 224, 3))
callback = LossAnalysisCallback(summary)

model.fit(x_train, y_train, 
          validation_data=(x_val, y_val),
          epochs=50,
          callbacks=[callback])
```

## Test the Feature

Run the example scripts to see all scenarios in action:

```bash
# PyTorch examples
python example_loss_analysis_pytorch.py

# TensorFlow examples
python example_loss_analysis_tf.py
```

## Technical Details

### Detection Algorithms

1. **Divergence Detection**: Uses linear regression slope and improvement percentage
2. **Overfitting Detection**: Compares recent trends in train vs validation losses
3. **Oscillation Detection**: Uses coefficient of variation (CV = std/mean × 100)
4. **Plateau Detection**: Checks if recent slope is near zero
5. **Convergence Quality**: Combines improvement rate with stability metrics

### Smoothing

Uses moving average with configurable window size to reduce noise in loss analysis. Default window is 5 epochs.

### Thresholds

- Divergence: slope > 0.001 or improvement < -5%
- Plateau: |recent_slope| < 0.0001 and epochs > 10
- High oscillation: CV > 20%
- Good convergence: improvement > 20% and |recent_slope| < 0.01

## Benefits

1. **Automated Diagnostics**: Instantly understand what's happening in training
2. **Actionable Recommendations**: Get specific suggestions for improvement
3. **Early Problem Detection**: Catch issues before wasting compute time
4. **Reproducible Analysis**: Consistent evaluation across experiments
5. **Framework Agnostic**: Same API for PyTorch and TensorFlow

## Future Enhancements

Potential additions:
- Learning rate recommendation based on loss curve shape
- Visualization of loss curves with annotations
- Comparison of multiple training runs
- Integration with TensorBoard/Weights & Biases
- Automatic hyperparameter suggestions
