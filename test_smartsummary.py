"""
Test suite for SmartSummary functionality
"""

# Import PyTorch first to avoid conflicts
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ToTf import SmartSummary


class SimpleConvNet(nn.Module):
    """Simple CNN for testing"""
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for testing nested structures"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ComplexNet(nn.Module):
    """More complex network with residual connections"""
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        
        # Input: 64x64 -> after conv1(stride=2): 32x32 -> after pool(stride=2): 16x16
        # So flattened size is 64 * 16 * 16 = 16384
        self.fc = nn.Linear(64 * 16 * 16, 1000)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_basic_summary():
    """Test basic SmartSummary functionality"""
    print("\n" + "="*100)
    print("TEST 1: Basic Summary")
    print("="*100)
    
    model = SimpleConvNet()
    summary = SmartSummary(model, input_size=(3, 32, 32), batch_size=2)
    summary.show(show_bottlenecks=False)
    
    assert summary.total_params > 0, "Total params should be > 0"
    assert summary.trainable_params > 0, "Trainable params should be > 0"
    assert len(summary.summary_data) > 0, "Should have layer information"
    
    print("\n✓ Basic summary test PASSED")


def test_bottleneck_detection():
    """Test bottleneck detection"""
    print("\n" + "="*100)
    print("TEST 2: Bottleneck Detection")
    print("="*100)
    
    model = SimpleConvNet()
    summary = SmartSummary(model, input_size=(3, 32, 32))
    
    bottlenecks = summary.get_bottlenecks(top_n=3)
    
    print(f"\nFound {len(bottlenecks)} bottlenecks:")
    for i, bn in enumerate(bottlenecks, 1):
        print(f"  {i}. {bn['layer']} - Score: {bn['score']:.2f}")
        print(f"     Reasons: {', '.join(bn['reasons'])}")
    
    assert len(bottlenecks) > 0, "Should detect at least one bottleneck"
    assert all('score' in bn for bn in bottlenecks), "Each bottleneck should have a score"
    
    print("\n✓ Bottleneck detection test PASSED")


def test_gradient_tracking():
    """Test gradient variance tracking"""
    print("\n" + "="*100)
    print("TEST 3: Gradient Tracking")
    print("="*100)
    
    model = SimpleConvNet()
    summary = SmartSummary(
        model, 
        input_size=(3, 32, 32), 
        batch_size=2,
        track_gradients=True
    )
    
    print(f"\nGradient statistics captured for {len(summary.gradient_stats)} layers")
    
    # Show some gradient stats
    for i, (layer, stats) in enumerate(list(summary.gradient_stats.items())[:3]):
        print(f"\n{layer}:")
        print(f"  Variance: {stats['grad_variance']:.2e}")
        print(f"  Mean: {stats['grad_mean']:.2e}")
        print(f"  Max: {stats['grad_max']:.2e}")
    
    assert len(summary.gradient_stats) > 0, "Should have gradient statistics"
    
    # Now show with bottlenecks including gradient info
    print("\n" + "-"*100)
    summary.show(show_bottlenecks=True)
    
    print("\n✓ Gradient tracking test PASSED")


def test_complex_model():
    """Test with complex nested model"""
    print("\n" + "="*100)
    print("TEST 4: Complex Nested Model")
    print("="*100)
    
    model = ComplexNet()
    summary = SmartSummary(model, input_size=(3, 64, 64))
    summary.show()
    
    assert summary.total_params > 0, "Complex model should have parameters"
    
    print("\n✓ Complex model test PASSED")


def test_without_input_size():
    """Test summary without forward pass"""
    print("\n" + "="*100)
    print("TEST 5: Summary Without Input Size")
    print("="*100)
    
    model = SimpleConvNet()
    summary = SmartSummary(model)  # No input_size
    
    print(f"\nTotal parameters: {summary.total_params:,}")
    print(f"Trainable parameters: {summary.trainable_params:,}")
    
    assert summary.total_params > 0, "Should count params even without forward pass"
    
    print("\n✓ Summary without input size test PASSED")


def test_export_functionality():
    """Test export to dict and file"""
    print("\n" + "="*100)
    print("TEST 6: Export Functionality")
    print("="*100)
    
    model = SimpleConvNet()
    summary = SmartSummary(model, input_size=(3, 32, 32))
    
    # Test to_dict
    summary_dict = summary.to_dict()
    assert "layers" in summary_dict, "Dictionary should contain layers"
    assert "total_params" in summary_dict, "Dictionary should contain total_params"
    assert "bottlenecks" in summary_dict, "Dictionary should contain bottlenecks"
    
    print(f"Summary dictionary keys: {list(summary_dict.keys())}")
    print(f"Number of layers: {len(summary_dict['layers'])}")
    print(f"Number of bottlenecks: {len(summary_dict['bottlenecks'])}")
    
    # Test save_to_file
    test_file = "test_summary.txt"
    summary.save_to_file(test_file)
    
    assert os.path.exists(test_file), "Summary file should be created"
    
    # Read and show first few lines
    with open(test_file, 'r') as f:
        lines = f.readlines()
        print(f"\nSummary file created with {len(lines)} lines")
        print("First 5 lines:")
        for line in lines[:5]:
            print(f"  {line.rstrip()}")
    
    # Cleanup
    os.remove(test_file)
    
    print("\n✓ Export functionality test PASSED")


def test_cuda_compatibility():
    """Test CUDA device compatibility"""
    print("\n" + "="*100)
    print("TEST 7: CUDA Compatibility")
    print("="*100)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    device = "cuda" if cuda_available else "cpu"
    model = SimpleConvNet()
    
    summary = SmartSummary(model, input_size=(3, 32, 32), device=device)
    
    print(f"Summary created successfully on {device}")
    assert summary.total_params > 0, "Should work on any device"
    
    print("\n✓ CUDA compatibility test PASSED")


def run_all_tests():
    """Run all SmartSummary tests"""
    print("\n" + "="*100)
    print(f"{'SmartSummary Test Suite':^100}")
    print("="*100)
    
    try:
        test_basic_summary()
        test_without_input_size()
        test_bottleneck_detection()
        test_complex_model()
        test_export_functionality()
        test_cuda_compatibility()
        test_gradient_tracking()  # Run this last as it's slower
        
        print("\n" + "="*100)
        print(f"{'ALL SMARTSUMMARY TESTS PASSED! ✓':^100}")
        print("="*100)
        print("\nSmartSummary is working correctly and ready to use!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
