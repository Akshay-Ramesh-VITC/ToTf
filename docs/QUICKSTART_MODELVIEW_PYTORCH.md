# Quick Start Guide - ModelView for PyTorch

## Installation

### 1. Install Python Dependencies

```bash
cd ToTf
pip install -r requirements.txt
```

This installs:
- torch>=2.0.0
- torchview>=0.2.6 (for PyTorch ModelView)
- graphviz>=0.20.0 (Python package)
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
python verify_modelview_pytorch.py
```

Expected output: All tests pass ✓

## Quick Usage

### Method 1: One-Liner (Quickest)

```python
from ToTf.pytorch import draw_graph
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleModel()

# Generate diagram instantly
draw_graph(model, input_size=(100,), save_path='my_model.png')
```

### Method 2: Full Control

```python
from ToTf.pytorch import ModelView
import torch.nn as nn

# Create your model
model = SimpleModel()

# Create ModelView
view = ModelView(model, input_size=(100,))

# Show text summary (powered by torchview)
view.show()

# Render high-quality diagram
view.render('model_architecture.png', dpi=300)
```

### Method 3: Research Paper Quality

```python
from ToTf.pytorch import ModelView

# Your research model
model = build_your_research_model()

view = ModelView(model, input_size=(3, 224, 224))

# High-resolution PNG (300 DPI)
view.render('paper_figure1.png', dpi=300, show_shapes=True)

# Vector PDF for LaTeX (perfect quality at any zoom)
view.render('paper_figure1.pdf', format='pdf')

# Scalable SVG
view.render('paper_figure1.svg', format='svg')
```

## Advanced Usage

### Show Detailed Architecture

```python
# Show all intermediate tensors and nested modules
view = ModelView(
    model,
    input_size=(3, 224, 224),
    depth=4,                    # Deeper module hierarchy
    expand_nested=True,         # Expand Sequential blocks
    hide_inner_tensors=False    # Show all tensors (not just I/O)
)
view.render('detailed_architecture.png', dpi=600)
```

### Multi-Input Models

```python
import torch

# For models with multiple inputs, use input_data
text_input = torch.randint(0, 10000, (2, 100))
image_input = torch.randn(2, 3, 64, 64)
numerical_input = torch.randn(2, 20)

view = ModelView(
    multimodal_model,
    input_data=(text_input, image_input, numerical_input)
)
view.render('multimodal_architecture.png', rankdir='LR')
```

### Custom Layout and Styling

```python
view = ModelView(model, input_size=(3, 224, 224))

# Horizontal layout (good for wide architectures)
view.render('wide_model.png', rankdir='LR', dpi=300)

# Vertical layout (good for deep networks)
view.render('deep_model.png', rankdir='TB', dpi=300)

# Very high resolution for publications
view.render('publication.png', dpi=600)
```

### Export Summary Data

```python
view = ModelView(model, input_size=(3, 224, 224))

# Get summary as dictionary
summary = view.get_summary_dict()
print(f"Model: {summary['model_name']}")
print(f"Parameters: {summary['total_parameters']:,}")

# Save summary as JSON
view.save_summary_json('model_summary.json')
```

## Complete Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from ToTf.pytorch import ModelView

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Create and visualize
model = CNN()
view = ModelView(model, input_size=(3, 32, 32))

# Print summary
view.show()

# Save multiple formats
view.render('cnn_architecture.png', dpi=300)      # For presentations
view.render('cnn_architecture.pdf', format='pdf') # For papers
view.render('cnn_architecture.svg', format='svg') # For web

print("✓ Architecture diagrams saved!")
```

## Comparison: PyTorch vs TensorFlow

The API is nearly identical across frameworks:

```python
# PyTorch
from ToTf.pytorch import ModelView
view = ModelView(model, input_size=(3, 224, 224))

# TensorFlow
from ToTf.tenf import ModelView
view = ModelView(model, input_shape=(224, 224, 3))
```

**Key differences:**
- PyTorch uses `input_size` (channels-first: C×H×W)
- TensorFlow uses `input_shape` (channels-last: H×W×C)
- PyTorch uses `input_data` for multi-input; TensorFlow uses list of shapes

## Troubleshooting

### "No module named 'torchview'"
```bash
pip install torchview
```

### Rendering fails
Make sure system Graphviz is installed:
- Windows: Download from https://graphviz.org/download/
- Ubuntu: `sudo apt-get install graphviz`
- macOS: `brew install graphviz`

### "Font fallback" warning
This is normal and doesn't affect functionality. The diagram will still render correctly.

### Multi-input model errors
Use `input_data` instead of `input_size` for models with multiple inputs:
```python
view = ModelView(model, input_data=(input1, input2, input3))
```

## Next Steps

1. **Run examples**: `python example_modelview_pytorch.py`
2. **Read documentation**: [PYTORCH_MODELVIEW_IMPLEMENTATION.md](PYTORCH_MODELVIEW_IMPLEMENTATION.md)
3. **Explore torchview features**: ModelView exposes all torchview options
4. **Combine with SmartSummary**: Analyze and visualize your models

## Tips

✅ **Use high DPI for publications**: `dpi=600` for journals
✅ **Try different layouts**: `rankdir='LR'` vs `rankdir='TB'`
✅ **Export to PDF/SVG**: Vector formats scale perfectly
✅ **Show detailed view for debugging**: `hide_inner_tensors=False`
✅ **Keep it simple for presentations**: `hide_inner_tensors=True`

---

**That's it!** You're ready to create beautiful architecture diagrams for your PyTorch models. 🚀
