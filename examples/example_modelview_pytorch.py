"""
Example Usage: PyTorch ModelView - Generate Publication-Quality Architecture Diagrams

This example demonstrates how to use ModelView to generate high-quality
neural network architecture diagrams for PyTorch models, using torchview internally.

Features demonstrated:
1. Simple Sequential models
2. Complex CNN architectures
3. ResNet-like architectures with skip connections
4. Multi-input models
5. Custom styling and layouts
6. Different output formats (PNG, PDF, SVG)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Import ModelView
from ToTf.pytorch import ModelView, draw_graph


def example1_simple_sequential():
    """Example 1: Simple Sequential MLP"""
    print("\n=== Example 1: Simple Sequential Model ===")
    
    class SimpleMLP(nn.Module):
        def __init__(self):
            super(SimpleMLP, self).__init__()
            self.fc1 = nn.Linear(784, 128)
            self.dropout1 = nn.Dropout(0.2)
            self.fc2 = nn.Linear(128, 64)
            self.dropout2 = nn.Dropout(0.2)
            self.fc3 = nn.Linear(64, 10)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            x = F.softmax(self.fc3(x), dim=1)
            return x
    
    model = SimpleMLP()
    
    # Quick visualization
    view = ModelView(model, input_size=(784,))
    view.show()
    
    # Save as PNG for presentations
    view.render('outputs/pytorch_simple_mlp.png', dpi=300)
    print("✓ Saved to outputs/pytorch_simple_mlp.png")
    
    # Save as PDF for LaTeX papers
    view.render('outputs/pytorch_simple_mlp.pdf', format='pdf')
    print("✓ Saved to outputs/pytorch_simple_mlp.pdf")
    
    return model


def example2_cnn_architecture():
    """Example 2: CNN for Image Classification"""
    print("\n=== Example 2: CNN Architecture ===")
    
    class MNIST_CNN(nn.Module):
        def __init__(self):
            super(MNIST_CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(2)
            
            self.conv2 = nn.Conv2d(32, 64, 3)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(2)
            
            self.conv3 = nn.Conv2d(64, 128, 3)
            self.bn3 = nn.BatchNorm2d(128)
            
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(128, 128)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            
            x = F.relu(self.bn3(self.conv3(x)))
            
            x = self.gap(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.softmax(self.fc2(x), dim=1)
            return x
    
    model = MNIST_CNN()
    
    view = ModelView(model, input_size=(1, 28, 28))
    view.show(detailed=True)
    
    # Horizontal layout (left-to-right) for wide figures
    view.render('outputs/pytorch_cnn_horizontal.png', rankdir='LR', dpi=300)
    print("✓ Saved horizontal layout")
    
    # Vertical layout (top-to-bottom) for narrow figures
    view.render('outputs/pytorch_cnn_vertical.png', rankdir='TB', dpi=300)
    print("✓ Saved vertical layout")
    
    # SVG for perfect scaling
    view.render('outputs/pytorch_cnn_scalable.svg', format='svg')
    print("✓ Saved SVG (scalable vector graphics)")
    
    return model


def example3_resnet_block():
    """Example 3: ResNet-like Architecture with Skip Connections"""
    print("\n=== Example 3: ResNet with Skip Connections ===")
    
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            # Residual connection adjustment if needed
            self.residual_conv = None
            if in_channels != out_channels:
                self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        
        def forward(self, x):
            residual = x
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
            
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + residual
            out = F.relu(out)
            return out
    
    class MiniResNet(nn.Module):
        def __init__(self):
            super(MiniResNet, self).__init__()
            self.stem_conv = nn.Conv2d(3, 32, 3, padding=1)
            self.stem_bn = nn.BatchNorm2d(32)
            
            self.block1 = ResidualBlock(32, 32)
            self.block2 = ResidualBlock(32, 64)
            self.block3 = ResidualBlock(64, 128)
            
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(128, 256)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, 10)
        
        def forward(self, x):
            x = F.relu(self.stem_bn(self.stem_conv(x)))
            
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.softmax(self.fc2(x), dim=1)
            return x
    
    model = MiniResNet()
    
    view = ModelView(model, input_size=(3, 32, 32), depth=4, expand_nested=True)
    view.show(detailed=True)
    
    # High-DPI PNG for papers
    view.render('outputs/pytorch_resnet_architecture.png', dpi=600, show_params=True)
    print("✓ Saved high-resolution ResNet architecture")
    
    # Export summary as JSON for documentation
    view.save_summary_json('outputs/pytorch_resnet_summary.json')
    print("✓ Saved architecture summary as JSON")
    
    return model


def example4_multi_input_model():
    """Example 4: Multi-Input Model (e.g., for multimodal learning)"""
    print("\n=== Example 4: Multi-Input Model ===")
    
    class MultimodalModel(nn.Module):
        def __init__(self):
            super(MultimodalModel, self).__init__()
            # Text branch
            self.text_embedding = nn.Embedding(10000, 128)
            self.text_lstm = nn.LSTM(128, 64, batch_first=True)
            self.text_dense = nn.Linear(64, 32)
            
            # Image branch
            self.img_conv1 = nn.Conv2d(3, 32, 3)
            self.img_pool1 = nn.MaxPool2d(2)
            self.img_conv2 = nn.Conv2d(32, 64, 3)
            self.img_gap = nn.AdaptiveAvgPool2d(1)
            self.img_dense = nn.Linear(64, 32)
            
            # Numerical features branch
            self.num_dense = nn.Linear(20, 32)
            
            # Merged layers
            self.merged_dense1 = nn.Linear(96, 64)  # 32 + 32 + 32
            self.merged_dropout = nn.Dropout(0.3)
            self.merged_dense2 = nn.Linear(64, 32)
            self.output = nn.Linear(32, 1)
        
        def forward(self, text, image, numerical):
            # Text branch
            x1 = self.text_embedding(text)
            x1, _ = self.text_lstm(x1)
            x1 = x1[:, -1, :]  # Take last output
            x1 = F.relu(self.text_dense(x1))
            
            # Image branch
            x2 = F.relu(self.img_conv1(image))
            x2 = self.img_pool1(x2)
            x2 = F.relu(self.img_conv2(x2))
            x2 = self.img_gap(x2)
            x2 = x2.view(x2.size(0), -1)
            x2 = F.relu(self.img_dense(x2))
            
            # Numerical branch
            x3 = F.relu(self.num_dense(numerical))
            
            # Merge
            merged = torch.cat([x1, x2, x3], dim=1)
            x = F.relu(self.merged_dense1(merged))
            x = self.merged_dropout(x)
            x = F.relu(self.merged_dense2(x))
            x = torch.sigmoid(self.output(x))
            return x
    
    model = MultimodalModel()
    
    # For multi-input models, we need to use input_data
    text_input = torch.randint(0, 10000, (2, 100))  # (batch, seq_len)
    image_input = torch.randn(2, 3, 64, 64)  # (batch, channels, height, width)
    num_input = torch.randn(2, 20)  # (batch, features)
    
    view = ModelView(
        model,
        input_data=(text_input, image_input, num_input),
        depth=4
    )
    view.show(detailed=True)
    
    # Left-to-right layout works better for wide architectures
    view.render('outputs/pytorch_multimodal_model.png', rankdir='LR', dpi=300)
    print("✓ Saved multimodal architecture")
    
    return model


def example5_quick_draw_graph():
    """Example 5: Quick visualization with draw_graph()"""
    print("\n=== Example 5: Quick Draw Graph Function ===")
    
    # Simple one-liner visualization
    model = nn.Sequential(
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
        nn.Softmax(dim=1)
    )
    
    # One-liner to save
    draw_graph(model, input_size=(100,), save_path='outputs/pytorch_quick_model.png')
    print("✓ Quick visualization saved")
    
    return model


def example6_advanced_features():
    """Example 6: Advanced torchview features"""
    print("\n=== Example 6: Advanced Features ===")
    
    class AdvancedCNN(nn.Module):
        def __init__(self):
            super(AdvancedCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 8 * 8, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 10)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    model = AdvancedCNN()
    
    # Show all intermediate tensors
    view1 = ModelView(
        model,
        input_size=(3, 32, 32),
        hide_inner_tensors=False,  # Show all tensors
        depth=5
    )
    view1.render('outputs/pytorch_detailed_tensors.png', dpi=300)
    print("✓ Saved detailed view with all tensors")
    
    # Clean view (only input/output)
    view2 = ModelView(
        model,
        input_size=(3, 32, 32),
        hide_inner_tensors=True,  # Clean view
        expand_nested=True,  # Expand Sequential blocks
        depth=3
    )
    view2.render('outputs/pytorch_clean_view.png', dpi=300)
    print("✓ Saved clean view")
    
    return model


def main():
    """Run all examples"""
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    print("=" * 70)
    print("PyTorch ModelView Examples")
    print("Using torchview internally for comprehensive visualization")
    print("=" * 70)
    
    try:
        example1_simple_sequential()
        example2_cnn_architecture()
        example3_resnet_block()
        example4_multi_input_model()
        example5_quick_draw_graph()
        example6_advanced_features()
        
        print("\n" + "=" * 70)
        print("✓ All examples completed successfully!")
        print("Check the 'outputs/' directory for generated diagrams")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
