"""
SmartSummary Usage Examples

This demonstrates various ways to use SmartSummary for model analysis
"""

# Import PyTorch first
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ToTf import SmartSummary


# Example 1: Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Example 2: VGG-like model with bottleneck
class VGGLike(nn.Module):
    def __init__(self):
        super(VGGLike, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # This will be a bottleneck - huge FC layer
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),  # Bottleneck!
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # Another bottleneck!
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def example_1_basic_usage():
    """Basic usage of SmartSummary"""
    print("\n" + "="*100)
    print("EXAMPLE 1: Basic Usage")
    print("="*100)
    
    model = SimpleCNN()
    
    # Create summary with input size
    summary = SmartSummary(model, input_size=(3, 32, 32))
    
    # Display the summary
    summary.show()


def example_2_bottleneck_detection():
    """Detect bottlenecks in VGG-like model"""
    print("\n" + "="*100)
    print("EXAMPLE 2: Bottleneck Detection in VGG-like Model")
    print("="*100)
    
    model = VGGLike()
    
    # Create summary
    summary = SmartSummary(model, input_size=(3, 32, 32))
    
    # Get bottlenecks programmatically
    bottlenecks = summary.get_bottlenecks(top_n=5)
    
    print("\n🔍 Bottleneck Analysis:")
    print("-" * 100)
    
    for i, bn in enumerate(bottlenecks, 1):
        print(f"\n{i}. Layer: {bn['layer']} ({bn['layer_name']})")
        print(f"   Bottleneck Score: {bn['score']:.2f}")
        print(f"   Parameters: {bn['params']:,}")
        print(f"   Output Shape: {bn['output_shape']}")
        print(f"   Issues: {', '.join(bn['reasons'])}")
    
    print("\n💡 Recommendation:")
    if bottlenecks:
        top_bn = bottlenecks[0]
        if 'High params' in str(top_bn['reasons']):
            print("   - Consider reducing the size of fully connected layers")
            print("   - Use Global Average Pooling instead of large FC layers")
            print("   - Apply model compression techniques")


def example_3_gradient_tracking():
    """Track gradient variance to find unstable layers"""
    print("\n" + "="*100)
    print("EXAMPLE 3: Gradient Variance Tracking")
    print("="*100)
    
    model = SimpleCNN()
    
    # Enable gradient tracking
    summary = SmartSummary(
        model, 
        input_size=(3, 32, 32),
        track_gradients=True  # This runs a forward + backward pass
    )
    
    print("\n📊 Layers with Gradient Statistics:")
    print("-" * 100)
    
    for layer, stats in list(summary.gradient_stats.items())[:5]:
        print(f"\n{layer}:")
        print(f"  Gradient Variance: {stats['grad_variance']:.2e}")
        print(f"  Gradient Mean: {stats['grad_mean']:.2e}")
        print(f"  Gradient Max: {stats['grad_max']:.2e}")
        
        # Check for potential issues
        if stats['grad_variance'] > 10.0:
            print("  ⚠️  WARNING: High gradient variance - potential instability")
        if abs(stats['grad_mean']) < 1e-7:
            print("  ⚠️  WARNING: Very small gradients - potential vanishing gradient")
    
    # Show full summary with bottlenecks
    summary.show(show_bottlenecks=True)


def example_4_model_comparison():
    """Compare different model architectures"""
    print("\n" + "="*100)
    print("EXAMPLE 4: Model Architecture Comparison")
    print("="*100)
    
    models = {
        "SimpleCNN": SimpleCNN(),
        "VGG-like": VGGLike()
    }
    
    results = []
    
    for name, model in models.items():
        summary = SmartSummary(model, input_size=(3, 32, 32))
        bottlenecks = summary.get_bottlenecks(top_n=1)
        
        results.append({
            "name": name,
            "total_params": summary.total_params,
            "trainable_params": summary.trainable_params,
            "output_size_mb": summary.total_output_size / (1024**2),
            "worst_bottleneck": bottlenecks[0]['score'] if bottlenecks else 0
        })
    
    print("\n📋 Model Comparison Table:")
    print("-" * 100)
    print(f"{'Model':<20} {'Total Params':<15} {'Trainable':<15} {'Output (MB)':<15} {'Max Bottleneck':<15}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['name']:<20} {r['total_params']:<15,} {r['trainable_params']:<15,} "
              f"{r['output_size_mb']:<15.2f} {r['worst_bottleneck']:<15.2f}")
    
    print("\n💡 Analysis:")
    best_params = min(results, key=lambda x: x['total_params'])
    best_bottleneck = min(results, key=lambda x: x['worst_bottleneck'])
    
    print(f"   - Most parameter-efficient: {best_params['name']}")
    print(f"   - Least bottlenecks: {best_bottleneck['name']}")


def example_5_export_summary():
    """Export summary to file and dictionary"""
    print("\n" + "="*100)
    print("EXAMPLE 5: Export Summary")
    print("="*100)
    
    model = VGGLike()
    summary = SmartSummary(model, input_size=(3, 32, 32))
    
    # Export to dictionary
    summary_dict = summary.to_dict()
    
    print("\n📦 Summary exported to dictionary:")
    print(f"   - Number of layers: {len(summary_dict['layers'])}")
    print(f"   - Total parameters: {summary_dict['total_params']:,}")
    print(f"   - Bottlenecks detected: {len(summary_dict['bottlenecks'])}")
    
    # Export to file
    filename = "model_analysis.txt"
    summary.save_to_file(filename)
    
    # Show file content preview
    with open(filename, 'r') as f:
        lines = f.readlines()
        print(f"\n📄 Summary saved to '{filename}' ({len(lines)} lines)")
        print("\nFirst 10 lines:")
        for line in lines[:10]:
            print(f"   {line.rstrip()}")
    
    # Cleanup
    os.remove(filename)
    print(f"\n✓ File removed after demo")


def example_6_without_forward_pass():
    """Analyze model without forward pass (useful for very large models)"""
    print("\n" + "="*100)
    print("EXAMPLE 6: Analysis Without Forward Pass")
    print("="*100)
    
    model = VGGLike()
    
    # Don't provide input_size - won't do forward pass
    summary = SmartSummary(model)
    
    print("\n📊 Parameter Analysis (no forward pass):")
    print(f"   Total parameters: {summary.total_params:,}")
    print(f"   Trainable parameters: {summary.trainable_params:,}")
    print(f"   Non-trainable parameters: {summary.total_params - summary.trainable_params:,}")
    
    print("\n💡 Note: Shape information not available without forward pass")
    print("   Use input_size parameter for shape inference")


def main():
    """Run all examples"""
    print("\n" + "="*100)
    print(f"{'SmartSummary - Advanced Model Analysis Examples':^100}")
    print("="*100)
    
    example_1_basic_usage()
    example_2_bottleneck_detection()
    example_4_model_comparison()
    example_5_export_summary()
    example_6_without_forward_pass()
    example_3_gradient_tracking()  # Run last as it's slower
    
    print("\n" + "="*100)
    print(f"{'All Examples Completed! ✓':^100}")
    print("="*100)
    
    print("\n🎯 Key Takeaways:")
    print("   1. SmartSummary provides detailed model analysis beyond basic summaries")
    print("   2. Bottleneck detection helps identify optimization opportunities")
    print("   3. Gradient tracking reveals training instabilities")
    print("   4. Export functionality enables automated analysis workflows")
    print("   5. Works with any PyTorch model architecture")
    
    print("\n📚 For more information, see the documentation in pytorch/README.md")


if __name__ == "__main__":
    main()
