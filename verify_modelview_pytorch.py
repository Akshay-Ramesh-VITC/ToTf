"""
Quick verification script for PyTorch ModelView

Tests basic functionality to ensure torchview integration works correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path to import ToTf
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

def test_pytorch_modelview():
    """Quick test of PyTorch ModelView"""
    print("Testing PyTorch ModelView...")
    
    try:
        from pytorch import ModelView, draw_graph
        print("✓ ModelView imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ModelView: {e}")
        return False
    
    # Test 1: Simple model
    try:
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 5)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        model = SimpleModel()
        view = ModelView(model, input_size=(10,))
        print("✓ ModelView created for simple model")
        
    except Exception as e:
        print(f"✗ Failed to create ModelView: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Show summary
    try:
        view.show()
        print("✓ Summary displayed")
    except Exception as e:
        print(f"✗ Failed to show summary: {e}")
        return False
    
    # Test 3: Get summary dict
    try:
        summary = view.get_summary_dict()
        print(f"✓ Summary dict retrieved: {summary['model_name']}")
    except Exception as e:
        print(f"✗ Failed to get summary dict: {e}")
        return False
    
    # Test 4: Render to file
    try:
        import os
        os.makedirs('test_outputs', exist_ok=True)
        view.render('test_outputs/pytorch_test.png', dpi=150)
        print("✓ Rendered to PNG")
    except Exception as e:
        print(f"✗ Failed to render: {e}")
        print("  Note: This might fail if graphviz is not installed on the system")
        # Don't return False here as it might be a system dependency issue
    
    # Test 5: draw_graph function
    try:
        draw_graph(model, input_size=(10,))
        print("✓ draw_graph function works")
    except Exception as e:
        print(f"✗ draw_graph failed: {e}")
        return False
    
    # Test 6: CNN model
    try:
        class CNNModel(nn.Module):
            def __init__(self):
                super(CNNModel, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, 3)
                self.pool = nn.MaxPool2d(2)
                self.conv2 = nn.Conv2d(16, 32, 3)
                self.fc1 = nn.Linear(32 * 5 * 5, 10)
            
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 32 * 5 * 5)
                x = self.fc1(x)
                return x
        
        cnn_model = CNNModel()
        cnn_view = ModelView(cnn_model, input_size=(3, 28, 28))
        print("✓ ModelView created for CNN model")
        
    except Exception as e:
        print(f"✗ Failed to create CNN ModelView: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    print("=" * 60)
    print("PyTorch ModelView Verification")
    print("=" * 60)
    print()
    
    success = test_pytorch_modelview()
    
    print()
    print("=" * 60)
    if success:
        print("✓ PyTorch ModelView verification PASSED")
        print()
        print("Next steps:")
        print("  1. Install torchview: pip install torchview")
        print("  2. Run examples: python example_modelview_pytorch.py")
        print("  3. Make sure graphviz is installed for rendering")
    else:
        print("✗ PyTorch ModelView verification FAILED")
        print()
        print("Troubleshooting:")
        print("  1. Install torchview: pip install torchview")
        print("  2. Check torch installation: pip install torch")
        print("  3. For rendering, install graphviz system package")
    print("=" * 60)


if __name__ == "__main__":
    main()
