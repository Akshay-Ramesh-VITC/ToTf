# ModelView Advanced Features (Torchview-like)

## Overview

ModelView has been enhanced to support complex neural network architectures similar to [torchview](https://github.com/mert-kurttutan/torchview) for PyTorch. The new features provide fine-grained control over visualization, computation graph tracking, and support for complex architectures.

## New Features

### 1. **Computation Graph Tracking**
- Tracks tensors and operations in addition to layers
- Maintains parent-child relationships between nodes
- Supports complex DAG structures (Directed Acyclic Graphs)

### 2. **Advanced Node Types**
- **TensorNode**: Represents tensors in the computation
- **LayerNode**: Represents Keras layers/modules
- **OperationNode**: Represents operations (Add, Concatenate, etc.)

### 3. **Visualization Control Options**

#### `hide_inner_tensors` (bool, default=True)
Controls whether intermediate tensors are shown.
- `True`: Only show input/output tensors (cleaner visualization)
- `False`: Show all intermediate tensors (detailed computation flow)

#### `hide_module_functions` (bool, default=True)
Controls whether operations inside layers arevisible.
- `True`: Hide operations inside modules (hide Conv2D operations, etc.)
- `False`: Show all operations (detailed layer internals)

#### `expand_nested` (bool, default=False)
Controls how nested models are displayed.
- `True`: Expand nested models with dashed borders
- `False`: Show nested models as single blocks

#### `depth` (int, default=3)
Controls how deep to traverse the module hierarchy.
- Lower values: Show high-level architecture
- Higher values: Show more detailed nested structures
- Useful for limiting complexity in deep models

#### `roll` (bool, default=False)
Controls recursive module visualization.
- `True`: Roll/unroll recursive structures (useful for RNNs)
- `False`: Show each occurrence separately

### 4. **Enhanced Visualization Methods**

#### `render_advanced()`
New rendering method that uses the computation graph:
```python
view.render_advanced(
    filename='model.png',
    format='png',          # 'png', 'pdf', 'svg'
    rankdir='TB',          # 'TB', 'LR', 'BT', 'RL'
    show_shapes=True,
    show_layer_names=False,
    show_params=True,
    dpi=300,
    cleanup=True
)
```

## Usage Examples

### Basic Usage

```python
from tenf.modelview import ModelView
import tensorflow as tf

model = tf.keras.Sequential([...])

# Create ModelView with default settings
view = ModelView(model, input_shape=(224, 224, 3))

# Render using advanced features
view.render_advanced('model.png', show_shapes=True)
```

### Hide Inner Tensors (Clean View)

```python
# Show only input/output tensors
view = ModelView(
    model,
    input_shape=(224, 224, 3),
    hide_inner_tensors=True,  # Clean view
    depth=3
)
view.render_advanced('model_clean.png')
```

### Show All Tensors (Detailed View)

```python
# Show all intermediate tensors
view = ModelView(
    model,
    input_shape=(224, 224, 3),
    hide_inner_tensors=False,  # Show all tensors
    depth=3
)
view.render_advanced('model_detailed.png')
```

### Expand Nested Models

```python
# Show nested models with dashed borders
view = ModelView(
    model,
    input_shape=(224, 224, 3),
    expand_nested=True,  # Expand nested modules
    depth=5
)
view.render_advanced('model_nested.png')
```

### Control Depth

```python
# Shallow view (depth=1)
shallow_view = ModelView(model, input_shape=(224, 224, 3), depth=1)
shallow_view.render_advanced('model_shallow.png')

# Deep view (depth=5)
deep_view = ModelView(model, input_shape=(224, 224, 3), depth=5)
deep_view.render_advanced('model_deep.png')
```

### Show Module Operations

```python
# Show operations inside layers
view = ModelView(
    model,
    input_shape=(224, 224, 3),
    hide_module_functions=False,  # Show operations
    depth=3
)
view.render_advanced('model_with_ops.png')
```

## Complex Architecture Support

### ResNet (Skip Connections)

```python
# ResNet with skip connections
def create_resnet_block(inputs, filters):
    x = layers.Conv2D(filters, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection
    x = layers.Add()([x, inputs])
    x = layers.Activation('relu')(x)
    return x

# Visualize
view = ModelView(resnet_model, input_shape=(32, 32, 3))
view.render_advanced('resnet.png', show_shapes=True)
```

### Multi-Input Models

```python
# Model with multiple inputs
image_input = keras.Input(shape=(32, 32, 3))
meta_input = keras.Input(shape=(10,))

x1 = layers.Conv2D(32, 3)(image_input)
x1 = layers.GlobalAveragePooling2D()(x1)

x2 = layers.Dense(32)(meta_input)

combined = layers.Concatenate()([x1, x2])
outputs = layers.Dense(10)(combined)

model = keras.Model([image_input, meta_input], outputs)

# Visualize with multiple input shapes
view = ModelView(model, input_shape=[(32, 32, 3), (10,)])
view.render_advanced('multi_input.png')
```

### Inception-style Modules

```python
# Parallel branches that concatenate
def inception_module(inputs, filters):
    branch1 = layers.Conv2D(filters, 1)(inputs)
    branch2 = layers.Conv2D(filters, 3, padding='same')(inputs)
    branch3 = layers.MaxPooling2D(3, strides=1, padding='same')(inputs)
    
    return layers.Concatenate()([branch1, branch2, branch3])

# Visualize
view = ModelView(inception_model, input_shape=(32, 32, 3))
view.render_advanced('inception.png', depth=2)
```

### Nested Models

```python
# Reusable submodel
block = keras.Sequential([...], name='FeatureBlock')

# Main model using submodel
inputs = keras.Input(shape=(32,))
x = block(inputs)
x = layers.Dense(16)(x)
x = block(x)  # Reuse
outputs = layers.Dense(10)(x)

model = keras.Model(inputs, outputs)

# Visualize with nested expansion
view = ModelView(model, input_shape=(32,), expand_nested=True, depth=4)
view.render_advanced('nested.png')
```

## Output Formats

### PNG (Raster)
```python
view.render_advanced('model.png', format='png', dpi=300)
```

### PDF (Vector - Publication Quality)
```python
view.render_advanced('model.pdf', format='pdf')
```

### SVG (Vector - Web)
```python
view.render_advanced('model.svg', format='svg')
```

## Comparison with Torchview

| Feature | Torchview (PyTorch) | ModelView (TensorFlow) |
|---------|---------------------|------------------------|
| Computation graph tracking | ✅ | ✅ |
| Hide inner tensors | ✅ (`hide_inner_tensors`) | ✅ (`hide_inner_tensors`) |
| Hide module functions | ✅ (`hide_module_functions`) | ✅ (`hide_module_functions`) |
| Expand nested | ✅ (`expand_nested`) | ✅ (`expand_nested`) |
| Depth control | ✅ (`depth`) | ✅ (`depth`) |
| Roll recursive | ✅ (`roll`) | ✅ (`roll`) |
| Skip connections (ResNet) | ✅ | ✅ |
| Multi-input/output | ✅ | ✅ |
| Show tensor shapes | ✅ | ✅ |
| Multiple formats (PNG/PDF/SVG) | ✅ | ✅ |
| Meta tensor support | ✅ | ⚠️ (TF doesn't have direct equivalent) |

## API Reference

### ModelView Constructor

```python
ModelView(
    model: keras.Model,
    input_shape: Optional[Union[Tuple, List[Tuple]]] = None,
    batch_size: int = 1,
    device: str = "CPU",
    depth: int = 3,
    expand_nested: bool = False,
    hide_inner_tensors: bool = True,
    hide_module_functions: bool = True,
    roll: bool = False,
    show_shapes: bool = True
)
```

**Parameters:**
- `model`: TensorFlow/Keras model to visualize
- `input_shape`: Input shape(s) excluding batch dimension
- `batch_size`: Batch size for shape inference
- `device`: 'CPU' or 'GPU'
- `depth`: Maximum depth for nested models (default: 3)
- `expand_nested`: Show nested models with dashed borders
- `hide_inner_tensors`: Hide intermediate tensors (show only I/O)
- `hide_module_functions`: Hide operations inside layers
- `roll`: Roll recursive modules
- `show_shapes`: Show tensor shapes

### render_advanced()

```python
view.render_advanced(
    filename: str,
    format: Optional[str] = None,
    rankdir: str = 'TB',
    show_shapes: bool = True,
    show_layer_names: bool = False,
    show_params: bool = True,
    dpi: int = 300,
    cleanup: bool = True
) -> str
```

**Parameters:**
- `filename`: Output file path
- `format`: 'png', 'pdf', 'svg' (auto-detected from filename)
- `rankdir`: Graph direction ('TB', 'LR', 'BT', 'RL')
- `show_shapes`: Display tensor shapes
- `show_layer_names`: Display layer instance names
- `show_params`: Display parameter counts
- `dpi`: Resolution for raster formats
- `cleanup`: Remove intermediate files

**Returns:** Path to rendered file

## Complete Example

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tenf.modelview import ModelView

# Create a complex model with skip connections
def create_model():
    inputs = keras.Input(shape=(32, 32, 3))
    
    # Initial conv
    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # ResNet block
    shortcut = x
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])  # Skip connection
    x = layers.Activation('relu')(x)
    
    # Classification
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Create model
model = create_model()

# Visualize with different settings
# 1. Clean view (default)
view1 = ModelView(model, input_shape=(32, 32, 3))
view1.render_advanced('model_clean.png', show_shapes=True)

# 2. Detailed view (show all tensors)
view2 = ModelView(
    model, 
    input_shape=(32, 32, 3),
    hide_inner_tensors=False
)
view2.render_advanced('model_detailed.png', show_shapes=True)

# 3. With operations
view3 = ModelView(
    model,
    input_shape=(32, 32, 3),
    hide_module_functions=False
)
view3.render_advanced('model_with_ops.png', show_shapes=True)

# 4. PDF for publication
view4 = ModelView(model, input_shape=(32, 32, 3))
view4.render_advanced('model.pdf', format='pdf', show_shapes=True)
```

## Running Examples

To run the comprehensive examples:

```bash
python example_modelview_advanced.py
```

This will create various visualizations in the `outputs/` directory demonstrating all the advanced features.

## Requirements

- TensorFlow >= 2.0
- graphviz (Python package): `pip install graphviz`
- Graphviz (system): 
  - Ubuntu/Debian: `sudo apt-get install graphviz`
  - macOS: `brew install graphviz`
  - Windows: Download from https://graphviz.org/download/

## Notes

- The `render_advanced()` method uses the new computation graph tracking
- The legacy `render()` method is still available for backward compatibility
- For very large models, consider using `depth` to limit visualization complexity
- Use `hide_inner_tensors=True` for cleaner, publication-ready diagrams
- Use `hide_inner_tensors=False` for debugging and understanding data flow

## Future Enhancements

- [ ] Support for attention mechanism visualization
- [ ] Graph neural network (GNN) visualization
- [ ] Interactive HTML visualizations
- [ ] Automatic layout optimization for large graphs
- [ ] Support for tf.function decorated models
