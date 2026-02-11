# Quick Start Guide - ModelView for TensorFlow

## Installation

### 1. Install Python Dependencies

```bash
cd ToTf
pip install -r requirements.txt
```

This installs:
- tensorflow>=2.13.0
- graphviz>=0.20.0 (Python package)
- Pillow>=10.0.0
- Other dependencies

### 2. Install System Graphviz (Required for Rendering)

**Windows:**
1. Download from: https://graphviz.org/download/
2. Run installer
3. Add to PATH:
   - During installation, check "Add Graphviz to PATH"
   - OR manually add: `C:\Program Files\Graphviz\bin` to PATH
4. Restart terminal/IDE

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install graphviz
```

**macOS:**
```bash
brew install graphviz
```

### 3. Verify Installation

```bash
python verify_modelview.py
```

Expected output: `5/6 tests passed` (6/6 if graphviz is properly installed)

## Quick Usage

### Method 1: One-Liner (Quickest)

```python
from ToTf.tenf import draw_graph
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Generate diagram instantly
draw_graph(model, input_shape=(100,), save_path='my_model.png')
```

### Method 2: Full Control

```python
from ToTf.tenf import ModelView
import tensorflow as tf

# Create your model
model = tf.keras.Sequential([...])

# Create ModelView
view = ModelView(model, input_shape=(100,))

# Show text summary
view.show()

# Render high-quality diagram
view.render('model_architecture.png', dpi=300)
```

### Method 3: Research Paper Quality

```python
from ToTf.tenf import ModelView

# Your research model
model = build_your_research_model()

view = ModelView(model, input_shape=(224, 224, 3))

# High-resolution PNG (300 DPI)
view.render('paper_figure1.png', dpi=300, show_shapes=True)

# Vector PDF for LaTeX (perfect quality at any zoom)
view.render('paper_figure1.pdf', format='pdf')

# Scalable SVG
view.render('paper_figure1.svg', format='svg')
```

## Run Examples

```bash
# Run all examples (creates diagrams in outputs/ directory)
python example_modelview_tf.py
```

Examples include:
- Simple MLP
- CNN for image classification
- ResNet with skip connections
- Multi-input multimodal model
- Transformer/Attention model
- Autoencoder
- Custom styling

## Run Tests

```bash
# Run comprehensive test suite
pytest test/test_modelview_tf.py -v

# Or without pytest:
python -m pytest test/test_modelview_tf.py -v
```

## Common Use Cases

### For CNN Models

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

view = ModelView(model, input_shape=(28, 28, 1))
view.render('cnn_architecture.png', rankdir='TB', dpi=300)
```

### For Multi-Input Models

```python
text_input = tf.keras.Input(shape=(100,), name='text')
image_input = tf.keras.Input(shape=(64, 64, 3), name='image')
# ... build model ...

model = tf.keras.Model(inputs=[text_input, image_input], outputs=outputs)

view = ModelView(model, input_shape=[(100,), (64, 64, 3)])
view.render('multimodal.png', rankdir='LR', dpi=300)
```

### For ResNet/Skip Connections

```python
# ResNet-like model
inputs = tf.keras.Input(shape=(32, 32, 3))
# ... build with skip connections ...

view = ModelView(model, input_shape=(32, 32, 3))
view.render('resnet.png', dpi=400)
```

## Troubleshooting

### "graphviz.backend.execute.ExecutableNotFound"

**Problem:** System graphviz is not installed or not in PATH

**Solution:**
1. Install system graphviz (see installation section above)
2. Restart terminal/IDE
3. Verify: `dot -V` should show graphviz version

### "No module named 'ToTf'"

**Problem:** Python can't find the module

**Solution:**
```python
import sys
sys.path.insert(0, '/path/to/ToTf')  # Add ToTf directory to path
from tenf.modelview import ModelView
```

Or run from ToTf directory:
```bash
cd ToTf
python your_script.py
```

### "Module 'tensorflow' has no attribute..."

**Problem:** TensorFlow version too old

**Solution:**
```bash
pip install --upgrade tensorflow>=2.13.0
```

## Output Formats

| Format | Use Case | Command |
|--------|----------|---------|
| PNG | Presentations, quick viewing | `view.render('model.png', dpi=300)` |
| PDF | LaTeX papers, print | `view.render('model.pdf', format='pdf')` |
| SVG | Web, perfect scaling | `view.render('model.svg', format='svg')` |

## Layout Options

| Direction | Description | Command |
|-----------|-------------|---------|
| TB | Top-to-bottom (default) | `view.render('model.png', rankdir='TB')` |
| LR | Left-to-right | `view.render('model.png', rankdir='LR')` |

## DPI Recommendations

| Use Case | DPI | Command |
|----------|-----|---------|
| Quick preview | 72-150 | `dpi=100` |
| Presentations | 150-300 | `dpi=200` |
| Papers/Print | 300-600 | `dpi=400` |
| High-quality print | 600+ | `dpi=600` |

## Next Steps

1. ✅ Install dependencies
2. ✅ Run verification: `python verify_modelview.py`
3. ✅ Try examples: `python example_modelview_tf.py`
4. ✅ Create diagrams for your models!
5. ✅ Use in your research papers

## Support

For issues or questions:
1. Check this guide
2. See examples: `example_modelview_tf.py`
3. Read comprehensive docs: See ModelView section in `README.md`
4. Run tests to verify setup: `python verify_modelview.py`

## Features Summary

✅ Multiple output formats (PNG, PDF, SVG)
✅ High-resolution (300-600 DPI)
✅ Publication-ready quality
✅ Complex architecture support
✅ Customizable styling
✅ Multiple layouts
✅ Comprehensive annotations
✅ Easy to use

Enjoy creating beautiful architecture diagrams! 🎨
