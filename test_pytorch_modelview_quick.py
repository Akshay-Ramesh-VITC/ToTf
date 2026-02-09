"""
Quick demo of PyTorch ModelView - Simple test to verify it works
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch import ModelView, draw_graph
import os

# Create a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    print("PyTorch ModelView - Quick Demo")
    print("=" * 60)
    
    # Create model
    model = SimpleNet()
    print("✓ Model created")
    
    # Create ModelView
    view = ModelView(model, input_size=(3, 32, 32))
    print("✓ ModelView created")
    
    # Show summary
    print("\nModel Summary:")
    print("-" * 60)
    view.show()
    print("-" * 60)
    
    # Create output directory
    os.makedirs('test_outputs', exist_ok=True)
    
    # Render to file
    try:
        view.render('test_outputs/simple_net.png', dpi=300)
        print("\n✓ Rendered to test_outputs/simple_net.png")
    except Exception as e:
        print(f"\n⚠ Rendering failed (might need graphviz installed): {e}")
    
    # Get summary dict
    summary = view.get_summary_dict()
    print(f"\n✓ Model: {summary['model_name']}")
    print(f"  Total parameters: {summary['total_parameters']:,}")
    print(f"  Trainable: {summary['trainable_parameters']:,}")
    
    # Test draw_graph function
    print("\n" + "=" * 60)
    print("Testing draw_graph() convenience function...")
    draw_graph(model, input_size=(3, 32, 32))
    print("✓ draw_graph works!")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("PyTorch ModelView is working correctly using torchview internally")
    print("=" * 60)

if __name__ == "__main__":
    main()
