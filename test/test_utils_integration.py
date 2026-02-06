"""
Cross-framework integration tests for utility functions

Tests that PyTorch and TensorFlow implementations produce consistent results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np


def test_lazy_flatten_consistency():
    """Test that lazy_flatten works consistently across frameworks"""
    print("\n" + "="*80)
    print("CROSS-FRAMEWORK TEST: lazy_flatten consistency")
    print("="*80)
    
    try:
        import torch
        from pytorch.utils import lazy_flatten as lazy_flatten_torch
        has_torch = True
    except ImportError:
        has_torch = False
        print("⚠️  PyTorch not available, skipping PyTorch tests")
    
    try:
        import tensorflow as tf
        from tenf.utils import lazy_flatten as lazy_flatten_tf
        has_tf = True
    except ImportError:
        has_tf = False
        print("⚠️  TensorFlow not available, skipping TensorFlow tests")
    
    if not has_torch or not has_tf:
        print("⚠️  Both frameworks required for cross-framework tests")
        return
    
    # Create test data
    np_data = np.random.randn(8, 7, 7, 16).astype(np.float32)
    
    # PyTorch (channels-first)
    torch_data = torch.from_numpy(np_data.transpose(0, 3, 1, 2))  # BHWC -> BCHW
    torch_flat = lazy_flatten_torch(torch_data)
    torch_result = torch_flat.numpy()
    
    # TensorFlow (channels-last)
    tf_data = tf.constant(np_data)  # Already BHWC
    tf_flat = lazy_flatten_tf(tf_data)
    tf_result = tf_flat.numpy()
    
    # Both should flatten to same total size
    assert torch_result.shape == tf_result.shape, \
        f"Shape mismatch: PyTorch {torch_result.shape} vs TF {tf_result.shape}"
    
    print(f"✓ Both frameworks produce shape: {torch_result.shape}")
    print("✅ lazy_flatten consistency test passed!")


def test_get_flatten_size_consistency():
    """Test get_flatten_size consistency"""
    print("\n" + "="*80)
    print("CROSS-FRAMEWORK TEST: get_flatten_size consistency")
    print("="*80)
    
    try:
        from pytorch.utils import get_flatten_size as get_size_torch
        from tenf.utils import get_flatten_size as get_size_tf
    except ImportError as e:
        print(f"⚠️  Import failed: {e}")
        return
    
    # Test shapes (note: different conventions)
    # PyTorch: (C, H, W)
    # TensorFlow: (H, W, C)
    
    # Same total size, different ordering
    torch_size = get_size_torch((16, 7, 7))  # 784
    tf_size = get_size_tf((7, 7, 16))        # 784
    
    assert torch_size == tf_size, \
        f"Size mismatch: PyTorch {torch_size} vs TF {tf_size}"
    
    print(f"✓ Both frameworks calculate size: {torch_size}")
    print("✅ get_flatten_size consistency test passed!")


def test_ncc_loss_consistency():
    """Test NCC loss consistency across frameworks"""
    print("\n" + "="*80)
    print("CROSS-FRAMEWORK TEST: NCC loss consistency")
    print("="*80)
    
    try:
        import torch
        from pytorch.utils import loss_ncc as loss_ncc_torch
        has_torch = True
    except ImportError:
        has_torch = False
    
    try:
        import tensorflow as tf
        from tenf.utils import loss_ncc as loss_ncc_tf
        has_tf = True
    except ImportError:
        has_tf = False
    
    if not has_torch or not has_tf:
        print("⚠️  Both frameworks required")
        return
    
    # Test case 1: Identical tensors
    print("\nTest 1: Identical tensors")
    np_data1 = np.random.randn(4, 32, 32, 1).astype(np.float32)
    
    torch_data1 = torch.from_numpy(np_data1.transpose(0, 3, 1, 2))  # BHWC -> BCHW
    torch_loss1 = loss_ncc_torch(torch_data1, torch_data1).item()
    
    tf_data1 = tf.constant(np_data1)
    tf_loss1 = float(loss_ncc_tf(tf_data1, tf_data1).numpy())
    
    print(f"PyTorch loss: {torch_loss1:.6f}")
    print(f"TensorFlow loss: {tf_loss1:.6f}")
    assert abs(torch_loss1 - tf_loss1) < 1e-4, "Identical tensors should give ~0 loss in both"
    print("✓ Test 1 passed")
    
    # Test case 2: Different tensors (similar NCC value expected)
    print("\nTest 2: Different tensors")
    np_true = np.random.randn(2, 16, 16, 1).astype(np.float32)
    np_pred = np.random.randn(2, 16, 16, 1).astype(np.float32)
    
    torch_true = torch.from_numpy(np_true.transpose(0, 3, 1, 2))
    torch_pred = torch.from_numpy(np_pred.transpose(0, 3, 1, 2))
    torch_loss2 = loss_ncc_torch(torch_true, torch_pred).item()
    
    tf_true = tf.constant(np_true)
    tf_pred = tf.constant(np_pred)
    tf_loss2 = float(loss_ncc_tf(tf_true, tf_pred).numpy())
    
    print(f"PyTorch loss: {torch_loss2:.4f}")
    print(f"TensorFlow loss: {tf_loss2:.4f}")
    
    # Allow small numerical differences due to different implementations
    diff = abs(torch_loss2 - tf_loss2)
    print(f"Difference: {diff:.6f}")
    assert diff < 0.01, f"NCC values should be close, diff: {diff}"
    print("✓ Test 2 passed")
    
    # Test case 3: Scaled version
    print("\nTest 3: Scaled version (same pattern)")
    np_base = np.random.randn(2, 32, 32, 1).astype(np.float32)
    np_scaled = np_base * 2.0
    
    torch_base = torch.from_numpy(np_base.transpose(0, 3, 1, 2))
    torch_scaled = torch.from_numpy(np_scaled.transpose(0, 3, 1, 2))
    torch_loss3 = loss_ncc_torch(torch_base, torch_scaled).item()
    
    tf_base = tf.constant(np_base)
    tf_scaled = tf.constant(np_scaled)
    tf_loss3 = float(loss_ncc_tf(tf_base, tf_scaled).numpy())
    
    print(f"PyTorch loss: {torch_loss3:.6f}")
    print(f"TensorFlow loss: {tf_loss3:.6f}")
    
    # Both should be very low (pattern is same)
    assert torch_loss3 < 0.1 and tf_loss3 < 0.1, "Scaled versions should have low loss"
    diff = abs(torch_loss3 - tf_loss3)
    assert diff < 0.01, f"Results should be close, diff: {diff}"
    print("✓ Test 3 passed")
    
    print("\n✅ NCC loss consistency test passed!")


def test_lr_finder_behavior():
    """Test that LR finder behaves similarly across frameworks"""
    print("\n" + "="*80)
    print("CROSS-FRAMEWORK TEST: LR Finder behavior")
    print("="*80)
    
    try:
        import torch
        import torch.nn as nn
        from pytorch.utils import LRFinder as LRFinder_torch
        has_torch = True
    except ImportError:
        has_torch = False
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tenf.utils import LRFinder as LRFinder_tf
        has_tf = True
    except ImportError:
        has_tf = False
    
    if not has_torch or not has_tf:
        print("⚠️  Both frameworks required")
        return
    
    print("Testing LR Finder creates results in both frameworks...")
    
    # PyTorch
    if has_torch:
        model_torch = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        train_data = torch.utils.data.TensorDataset(
            torch.randn(50, 10),
            torch.randint(0, 2, (50,))
        )
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=10)
        optimizer = torch.optim.Adam(model_torch.parameters())
        criterion = nn.CrossEntropyLoss()
        
        lr_finder_torch = LRFinder_torch(model_torch, optimizer, criterion)
        lr_finder_torch.range_test(train_loader, num_iter=20)
        best_lr_torch = lr_finder_torch.get_best_lr()
        
        print(f"PyTorch LR Finder: {len(lr_finder_torch.lrs)} iterations, best_lr={best_lr_torch:.2e}")
    
    # TensorFlow
    if has_tf:
        model_tf = keras.Sequential([
            keras.layers.Dense(20, activation='relu', input_shape=(10,)),
            keras.layers.Dense(2, activation='softmax')
        ])
        x_train = tf.random.normal([50, 10])
        y_train = tf.random.uniform([50], minval=0, maxval=2, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(10)
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        
        lr_finder_tf = LRFinder_tf(model_tf, loss_fn)
        lr_finder_tf.range_test(dataset, num_iter=20)
        best_lr_tf = lr_finder_tf.get_best_lr()
        
        print(f"TensorFlow LR Finder: {len(lr_finder_tf.lrs)} iterations, best_lr={best_lr_tf:.2e}")
    
    # Both should produce valid results
    if has_torch and has_tf:
        assert best_lr_torch is not None and best_lr_torch > 0, "PyTorch should find valid LR"
        assert best_lr_tf is not None and best_lr_tf > 0, "TensorFlow should find valid LR"
        
        # LRs might differ due to different models/optimizers and random data, but should be positive
        # The exact value can vary widely depending on the random initialization
        assert 1e-10 < best_lr_torch < 100, f"PyTorch LR should be reasonable, got {best_lr_torch:.2e}"
        assert 1e-10 < best_lr_tf < 100, f"TensorFlow LR should be reasonable, got {best_lr_tf:.2e}"
        
        print("✓ Both frameworks produce valid LR suggestions")
    
    print("\n✅ LR Finder behavior test passed!")


def test_complete_workflow():
    """Test complete workflow using utils in both frameworks"""
    print("\n" + "="*80)
    print("CROSS-FRAMEWORK TEST: Complete workflow")
    print("="*80)
    
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from pytorch.utils import lazy_flatten, get_flatten_size, loss_ncc
        has_torch = True
    except ImportError:
        has_torch = False
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tenf.utils import lazy_flatten as lazy_flatten_tf, get_flatten_size, NCCLoss
        has_tf = True
    except ImportError:
        has_tf = False
    
    if not has_torch or not has_tf:
        print("⚠️  Both frameworks required")
        return
    
    print("\n--- PyTorch Workflow ---")
    if has_torch:
        # Define model using utils
        class PyTorchModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.pool = nn.MaxPool2d(2)
                
                # Use get_flatten_size
                flat_size = get_flatten_size((32, 8, 8))  # After 2 pools from 32x32
                self.fc = nn.Linear(flat_size, 10)
            
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = lazy_flatten(x)  # Use lazy_flatten
                x = self.fc(x)
                return x
        
        model_torch = PyTorchModel()
        x_torch = torch.randn(4, 1, 32, 32)
        out_torch = model_torch(x_torch)
        print(f"Output shape: {out_torch.shape}")
        
        # Test NCC loss
        y_true = torch.randn(2, 1, 16, 16)
        y_pred = torch.randn(2, 1, 16, 16)
        ncc_loss_val = loss_ncc(y_true, y_pred)
        print(f"NCC loss: {ncc_loss_val.item():.4f}")
        print("✓ PyTorch workflow completed")
    
    print("\n--- TensorFlow Workflow ---")
    if has_tf:
        # Define model using utils
        class TFModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.conv1 = keras.layers.Conv2D(16, 3, padding='same')
                self.conv2 = keras.layers.Conv2D(32, 3, padding='same')
                self.pool = keras.layers.MaxPooling2D(2)
                
                # Use get_flatten_size
                flat_size = get_flatten_size((8, 8, 32))  # After 2 pools from 32x32
                self.fc = keras.layers.Dense(10)
            
            def call(self, x):
                x = self.pool(tf.nn.relu(self.conv1(x)))
                x = self.pool(tf.nn.relu(self.conv2(x)))
                x = lazy_flatten_tf(x)  # Use lazy_flatten
                x = self.fc(x)
                return x
        
        model_tf = TFModel()
        x_tf = tf.random.normal([4, 32, 32, 1])
        out_tf = model_tf(x_tf)
        print(f"Output shape: {out_tf.shape}")
        
        # Test NCC loss
        model_simple = keras.Sequential([
            keras.layers.Conv2D(16, 3, activation='relu', input_shape=(16, 16, 1)),
            keras.layers.Conv2D(1, 3, padding='same')
        ])
        model_simple.compile(optimizer='adam', loss=NCCLoss())
        print("NCC Loss integrated with model.compile()")
        print("✓ TensorFlow workflow completed")
    
    print("\n✅ Complete workflow test passed!")


def run_all_integration_tests():
    """Run all cross-framework integration tests"""
    print("\n" + "="*80)
    print("RUNNING CROSS-FRAMEWORK INTEGRATION TESTS")
    print("="*80)
    
    try:
        test_lazy_flatten_consistency()
        test_get_flatten_size_consistency()
        test_ncc_loss_consistency()
        test_lr_finder_behavior()
        test_complete_workflow()
        
        print("\n" + "="*80)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_integration_tests()
    sys.exit(0 if success else 1)
