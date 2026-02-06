"""
Simple test to verify TrainingMonitor works correctly
"""

# Import PyTorch first before our module
import torch

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ToTf import TrainingMonitor, get_backend


def test_basic_functionality():
    """Test basic TrainingMonitor functionality"""
    print(f"Detected backend: {get_backend()}")
    
    # Create dummy data
    data = list(range(10))
    
    # Test training monitor
    monitor = TrainingMonitor(data, desc="Testing", log_file="test_log.csv")
    
    for i, item in enumerate(monitor):
        # Simulate some metrics
        monitor.log({
            'loss': 1.0 / (i + 1),  # Decreasing loss
            'accuracy': i / 10.0     # Increasing accuracy
        })
    
    print("\n✓ Basic functionality test passed!")
    print("Check test_log.csv for logged data")


def test_cuda_detection():
    """Test CUDA detection"""
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("✓ CUDA detection test passed!")


if __name__ == "__main__":
    print("Running ToTf Tests...")
    print("=" * 50)
    
    test_basic_functionality()
    test_cuda_detection()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
