# ToTf TensorFlow Module

TensorFlow/Keras implementation of advanced training utilities.

## SmartSummary

Advanced model summary with bottleneck detection for TensorFlow/Keras models.

### Features

- **Comprehensive Layer Analysis**: Detailed parameter counts, shapes, and memory usage
- **Bottleneck Detection**: Automatically identifies layers that may need optimization
- **Gradient Tracking**: Monitor gradient statistics to detect training issues
- **Memory Estimation**: Track output sizes and total memory footprint
- **Export Capabilities**: Save summaries or export as dictionaries
- **Keras Compatible**: Works with any `tf.keras.Model`

### Quick Start

```python
import tensorflow as tf
from ToTf.tenf import SmartSummary

# Create your model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Analyze the model
summary = SmartSummary(model, input_shape=(224, 224, 3))
summary.show()
```

### Advanced Usage

#### Gradient Tracking

Monitor gradient statistics during training to identify unstable layers:

```python
summary = SmartSummary(
    model, 
    input_shape=(224, 224, 3),
    track_gradients=True  # Enable gradient analysis
)
summary.show()
```

#### Bottleneck Detection

Get detailed bottleneck analysis:

```python
bottlenecks = summary.get_bottlenecks(top_n=3)
for bn in bottlenecks:
    print(f"Layer: {bn['layer']}")
    print(f"Reasons: {bn['reasons']}")
    print(f"Score: {bn['score']}")
```

#### Export Summary

```python
# Export as dictionary
data = summary.to_dict()

# Save to file
summary.save_to_file("my_model_analysis.txt")
```

#### Compare with Keras Summary

```python
summary.compare_with_keras_summary()
```

### Multi-Input Models

SmartSummary works with multi-input models too:

```python
input1 = tf.keras.layers.Input(shape=(224, 224, 3))
input2 = tf.keras.layers.Input(shape=(100,))

x1 = tf.keras.layers.Conv2D(32, 3)(input1)
x1 = tf.keras.layers.Flatten()(x1)

x2 = tf.keras.layers.Dense(64)(input2)

combined = tf.keras.layers.concatenate([x1, x2])
output = tf.keras.layers.Dense(10)(combined)

model = tf.keras.Model(inputs=[input1, input2], outputs=output)

summary = SmartSummary(model, input_shape=[(224, 224, 3), (100,)])
summary.show()
```

### When to Use SmartSummary vs Keras Summary

Use **SmartSummary** when you need:
- Bottleneck detection and optimization hints
- Gradient variance analysis
- Memory usage estimates
- Detailed parameter breakdowns
- Export and programmatic access to summary data

Use **Keras built-in summary** when you just need:
- Quick layer overview
- Basic parameter counts
- Standard Keras output format

### API Reference

#### SmartSummary

```python
SmartSummary(
    model: keras.Model,
    input_shape: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
    batch_size: int = 1,
    track_gradients: bool = False
)
```

**Parameters:**
- `model`: TensorFlow/Keras model to analyze
- `input_shape`: Input tensor shape (excluding batch dimension). Can be a list for multi-input models
- `batch_size`: Batch size for shape inference
- `track_gradients`: Whether to track gradient statistics

**Methods:**
- `show(show_bottlenecks=True)`: Display formatted summary
- `get_bottlenecks(top_n=5)`: Get list of bottleneck layers
- `to_dict()`: Export summary as dictionary
- `save_to_file(filename)`: Save summary to text file
- `compare_with_keras_summary()`: Show both Keras and Smart summaries

### Requirements

- TensorFlow >= 2.13.0
- NumPy >= 1.24.0

### Differences from PyTorch Version

The TensorFlow implementation maintains feature parity with the PyTorch version while adapting to TensorFlow's paradigms:

- Uses `tf.keras.Model` instead of `torch.nn.Module`
- Uses `tf.GradientTape` for gradient tracking instead of backward hooks
- Layer iteration uses `model.layers` instead of `named_modules()`
- Shape format follows TensorFlow conventions (channels last by default)

### Examples

See `example_smartsummary_tf.py` for comprehensive examples.
