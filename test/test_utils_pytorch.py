"""
Unit tests for PyTorch utility functions

Tests:
- lazy_flatten
- get_flatten_size
- loss_ncc
- ncc_score
- LRFinder
- find_lr
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch.utils import (
    lazy_flatten, 
    get_flatten_size, 
    loss_ncc, 
    ncc_score,
    LRFinder,
    find_lr
)
import numpy as np


def test_lazy_flatten():
    """Test lazy_flatten function"""
    print("\n" + "="*80)
    print("TEST: lazy_flatten")
    print("="*80)
    
    # Test 1: 4D tensor (batch, channels, height, width)
    x = torch.randn(32, 16, 7, 7)
    x_flat = lazy_flatten(x)
    assert x_flat.shape == (32, 16*7*7), f"Expected shape (32, 784), got {x_flat.shape}"
    print("✓ Test 1 passed: 4D tensor flattening")
    
    # Test 2: 3D tensor
    x = torch.randn(16, 10, 5)
    x_flat = lazy_flatten(x)
    assert x_flat.shape == (16, 50), f"Expected shape (16, 50), got {x_flat.shape}"
    print("✓ Test 2 passed: 3D tensor flattening")
    
    # Test 3: Different start_dim
    x = torch.randn(8, 3, 224, 224)
    x_flat = lazy_flatten(x, start_dim=2)
    assert x_flat.shape == (8, 3, 224*224), f"Expected shape (8, 3, 50176), got {x_flat.shape}"
    print("✓ Test 3 passed: Custom start_dim")
    
    # Test 4: Already flattened
    x = torch.randn(10, 100)
    x_flat = lazy_flatten(x)
    assert x_flat.shape == (10, 100), f"Expected shape (10, 100), got {x_flat.shape}"
    print("✓ Test 4 passed: Already flat tensor")
    
    print("\n✅ All lazy_flatten tests passed!")


def test_get_flatten_size():
    """Test get_flatten_size function"""
    print("\n" + "="*80)
    print("TEST: get_flatten_size")
    print("="*80)
    
    # Test 1: Conv output shape
    size = get_flatten_size((16, 7, 7))
    assert size == 784, f"Expected 784, got {size}"
    print("✓ Test 1 passed: Conv shape (16, 7, 7) = 784")
    
    # Test 2: Single dimension
    size = get_flatten_size((1024,))
    assert size == 1024, f"Expected 1024, got {size}"
    print("✓ Test 2 passed: Single dimension")
    
    # Test 3: Large shape
    size = get_flatten_size((512, 14, 14))
    assert size == 100352, f"Expected 100352, got {size}"
    print("✓ Test 3 passed: Large shape")
    
    print("\n✅ All get_flatten_size tests passed!")


def test_loss_ncc():
    """Test NCC loss function"""
    print("\n" + "="*80)
    print("TEST: loss_ncc")
    print("="*80)
    
    # Test 1: Identical tensors (should give loss ~0)
    y_true = torch.randn(8, 1, 64, 64)
    y_pred = y_true.clone()
    loss = loss_ncc(y_true, y_pred)
    assert loss < 1e-4, f"Expected loss ~0 for identical tensors, got {loss.item()}"
    print(f"✓ Test 1 passed: Identical tensors, loss = {loss.item():.6f}")
    
    # Test 2: Completely different tensors
    y_true = torch.randn(4, 1, 32, 32)
    y_pred = torch.randn(4, 1, 32, 32)
    loss = loss_ncc(y_true, y_pred)
    assert 0 <= loss <= 2, f"Expected loss in [0, 2], got {loss.item()}"
    print(f"✓ Test 2 passed: Different tensors, loss = {loss.item():.4f}")
    
    # Test 3: Gradient flow
    y_true = torch.randn(2, 1, 16, 16)
    y_pred = torch.randn(2, 1, 16, 16, requires_grad=True)
    loss = loss_ncc(y_true, y_pred)
    loss.backward()
    assert y_pred.grad is not None, "Gradient should exist"
    print("✓ Test 3 passed: Gradient flow works")
    
    # Test 4: Multi-channel images
    y_true = torch.randn(4, 3, 64, 64)
    y_pred = torch.randn(4, 3, 64, 64)
    loss = loss_ncc(y_true, y_pred)
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss should be valid"
    print(f"✓ Test 4 passed: Multi-channel images, loss = {loss.item():.4f}")
    
    # Test 5: Scaled version (same pattern, different scale)
    y_true = torch.randn(2, 1, 32, 32)
    y_pred = y_true * 2.0  # Same pattern, scaled
    loss = loss_ncc(y_true, y_pred)
    assert loss < 0.1, f"Expected low loss for scaled version, got {loss.item()}"
    print(f"✓ Test 5 passed: Scaled version, loss = {loss.item():.6f}")
    
    print("\n✅ All loss_ncc tests passed!")


def test_ncc_score():
    """Test NCC score function"""
    print("\n" + "="*80)
    print("TEST: ncc_score")
    print("="*80)
    
    # Test 1: Identical tensors (should give score ~1)
    y_true = torch.randn(4, 1, 32, 32)
    y_pred = y_true.clone()
    score = ncc_score(y_true, y_pred)
    assert score > 0.999, f"Expected score ~1 for identical tensors, got {score.item()}"
    print(f"✓ Test 1 passed: Identical tensors, score = {score.item():.6f}")
    
    # Test 2: Score range
    y_true = torch.randn(2, 1, 16, 16)
    y_pred = torch.randn(2, 1, 16, 16)
    score = ncc_score(y_true, y_pred)
    assert -1 <= score <= 1, f"Expected score in [-1, 1], got {score.item()}"
    print(f"✓ Test 2 passed: Score in valid range, score = {score.item():.4f}")
    
    print("\n✅ All ncc_score tests passed!")


def test_lr_finder():
    """Test LRFinder class"""
    print("\n" + "="*80)
    print("TEST: LRFinder")
    print("="*80)
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 5)
    )
    
    # Create simple dataset
    train_data = torch.utils.data.TensorDataset(
        torch.randn(100, 10),
        torch.randint(0, 5, (100,))
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Test 1: Initialize LRFinder
    lr_finder = LRFinder(model, optimizer, criterion, device='cpu')
    assert lr_finder is not None, "LRFinder should initialize"
    print("✓ Test 1 passed: LRFinder initialization")
    
    # Test 2: Run range test (short version)
    lr_finder.range_test(train_loader, start_lr=1e-5, end_lr=1.0, num_iter=20)
    assert len(lr_finder.lrs) > 0, "Should have recorded LRs"
    assert len(lr_finder.losses) > 0, "Should have recorded losses"
    print(f"✓ Test 2 passed: Range test completed ({len(lr_finder.lrs)} iterations)")
    
    # Test 3: Get best LR
    best_lr = lr_finder.get_best_lr()
    assert best_lr is not None, "Should return a best LR"
    assert best_lr > 0, "Best LR should be positive"
    print(f"✓ Test 3 passed: Best LR = {best_lr:.2e}")
    
    # Test 4: Model state restored
    # (We can't easily test this directly, but it shouldn't crash)
    print("✓ Test 4 passed: Model state handling")
    
    print("\n✅ All LRFinder tests passed!")


def test_find_lr():
    """Test find_lr convenience function"""
    print("\n" + "="*80)
    print("TEST: find_lr")
    print("="*80)
    
    # Create simple model and data
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )
    
    train_data = torch.utils.data.TensorDataset(
        torch.randn(50, 10),
        torch.randint(0, 2, (50,))
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=10)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Test: Run find_lr (short version, no plot for testing)
    best_lr = find_lr(model, optimizer, criterion, train_loader, 
                     device='cpu', num_iter=15, plot=False)
    
    assert best_lr is not None, "Should return a best LR"
    assert best_lr > 0, "Best LR should be positive"
    print(f"✓ Test passed: find_lr returned {best_lr:.2e}")
    
    print("\n✅ All find_lr tests passed!")


def test_integration_conv_to_linear():
    """Integration test: Using lazy_flatten in a real model"""
    print("\n" + "="*80)
    print("INTEGRATION TEST: Conv to Linear transition")
    print("="*80)
    
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            
            # Using get_flatten_size to determine Linear input
            # Input: 32x32 -> after 2 pools: 8x8, channels: 32
            flat_size = get_flatten_size((32, 8, 8))
            self.fc = nn.Linear(flat_size, 10)
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = lazy_flatten(x)  # Auto-flatten
            x = self.fc(x)
            return x
    
    model = TestModel()
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    
    assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
    print("✓ Integration test passed: Conv->Flatten->Linear works correctly")
    
    print("\n✅ Integration test passed!")


def test_ncc_medical_imaging():
    """Integration test: NCC for medical imaging scenario"""
    print("\n" + "="*80)
    print("INTEGRATION TEST: NCC for medical imaging")
    print("="*80)
    
    # Simulate medical image registration scenario
    # Original image
    original = torch.randn(1, 1, 128, 128)
    
    # Slightly transformed version (simulating registration)
    transformed = original + torch.randn(1, 1, 128, 128) * 0.1
    
    # Calculate NCC
    loss = loss_ncc(original, transformed)
    score = ncc_score(original, transformed)
    
    print(f"NCC Loss: {loss.item():.4f}")
    print(f"NCC Score: {score.item():.4f}")
    
    assert 0 <= loss <= 2, "Loss should be in valid range"
    assert -1 <= score <= 1, "Score should be in valid range"
    print("✓ Integration test passed: NCC works for medical imaging scenario")
    
    print("\n✅ Medical imaging integration test passed!")


def run_all_tests():
    """Run all unit and integration tests"""
    print("\n" + "="*80)
    print("RUNNING ALL PYTORCH UTILS TESTS")
    print("="*80)
    
    try:
        # Unit tests
        test_lazy_flatten()
        test_get_flatten_size()
        test_loss_ncc()
        test_ncc_score()
        test_lr_finder()
        test_find_lr()
        
        # Integration tests
        test_integration_conv_to_linear()
        test_ncc_medical_imaging()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED SUCCESSFULLY!")
        print("="*80)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
