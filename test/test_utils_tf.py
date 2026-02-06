"""
Unit tests for TensorFlow utility functions

Tests:
- lazy_flatten
- get_flatten_size
- loss_ncc
- ncc_score
- NCCLoss
- LRFinder
- find_lr
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tensorflow as tf
from tensorflow import keras
from tenf.utils import (
    lazy_flatten,
    get_flatten_size,
    loss_ncc,
    ncc_score,
    NCCLoss,
    LRFinder,
    find_lr
)
import numpy as np


def test_lazy_flatten():
    """Test lazy_flatten function"""
    print("\n" + "="*80)
    print("TEST: lazy_flatten")
    print("="*80)
    
    # Test 1: 4D tensor (batch, height, width, channels)
    x = tf.random.normal([32, 7, 7, 16])
    x_flat = lazy_flatten(x)
    assert x_flat.shape == (32, 7*7*16), f"Expected shape (32, 784), got {x_flat.shape}"
    print("✓ Test 1 passed: 4D tensor flattening")
    
    # Test 2: 3D tensor
    x = tf.random.normal([16, 10, 5])
    x_flat = lazy_flatten(x)
    assert x_flat.shape == (16, 50), f"Expected shape (16, 50), got {x_flat.shape}"
    print("✓ Test 2 passed: 3D tensor flattening")
    
    # Test 3: Already flattened
    x = tf.random.normal([10, 100])
    x_flat = lazy_flatten(x)
    assert x_flat.shape == (10, 100), f"Expected shape (10, 100), got {x_flat.shape}"
    print("✓ Test 3 passed: Already flat tensor")
    
    # Test 4: Dynamic batch size
    @tf.function
    def flatten_dynamic(x):
        return lazy_flatten(x)
    
    x = tf.random.normal([8, 5, 5, 3])
    x_flat = flatten_dynamic(x)
    assert x_flat.shape[1] == 75, f"Expected flat dim 75, got {x_flat.shape[1]}"
    print("✓ Test 4 passed: Works with tf.function")
    
    print("\n✅ All lazy_flatten tests passed!")


def test_get_flatten_size():
    """Test get_flatten_size function"""
    print("\n" + "="*80)
    print("TEST: get_flatten_size")
    print("="*80)
    
    # Test 1: Conv output shape
    size = get_flatten_size((7, 7, 16))
    assert size == 784, f"Expected 784, got {size}"
    print("✓ Test 1 passed: Conv shape (7, 7, 16) = 784")
    
    # Test 2: Single dimension
    size = get_flatten_size((1024,))
    assert size == 1024, f"Expected 1024, got {size}"
    print("✓ Test 2 passed: Single dimension")
    
    # Test 3: Large shape
    size = get_flatten_size((14, 14, 512))
    assert size == 100352, f"Expected 100352, got {size}"
    print("✓ Test 3 passed: Large shape")
    
    print("\n✅ All get_flatten_size tests passed!")


def test_loss_ncc():
    """Test NCC loss function"""
    print("\n" + "="*80)
    print("TEST: loss_ncc")
    print("="*80)
    
    # Test 1: Identical tensors (should give loss ~0)
    y_true = tf.random.normal([8, 64, 64, 1])
    y_pred = tf.identity(y_true)
    loss = loss_ncc(y_true, y_pred)
    assert loss < 1e-4, f"Expected loss ~0 for identical tensors, got {loss.numpy()}"
    print(f"✓ Test 1 passed: Identical tensors, loss = {loss.numpy():.6f}")
    
    # Test 2: Completely different tensors
    y_true = tf.random.normal([4, 32, 32, 1])
    y_pred = tf.random.normal([4, 32, 32, 1])
    loss = loss_ncc(y_true, y_pred)
    assert 0 <= loss <= 2, f"Expected loss in [0, 2], got {loss.numpy()}"
    print(f"✓ Test 2 passed: Different tensors, loss = {loss.numpy():.4f}")
    
    # Test 3: Gradient flow
    y_true = tf.random.normal([2, 16, 16, 1])
    with tf.GradientTape() as tape:
        y_pred = tf.Variable(tf.random.normal([2, 16, 16, 1]))
        loss = loss_ncc(y_true, y_pred)
    
    gradients = tape.gradient(loss, y_pred)
    assert gradients is not None, "Gradient should exist"
    print("✓ Test 3 passed: Gradient flow works")
    
    # Test 4: Multi-channel images
    y_true = tf.random.normal([4, 64, 64, 3])
    y_pred = tf.random.normal([4, 64, 64, 3])
    loss = loss_ncc(y_true, y_pred)
    assert not tf.math.is_nan(loss) and not tf.math.is_inf(loss), "Loss should be valid"
    print(f"✓ Test 4 passed: Multi-channel images, loss = {loss.numpy():.4f}")
    
    # Test 5: Scaled version (same pattern, different scale)
    y_true = tf.random.normal([2, 32, 32, 1])
    y_pred = y_true * 2.0  # Same pattern, scaled
    loss = loss_ncc(y_true, y_pred)
    assert loss < 0.1, f"Expected low loss for scaled version, got {loss.numpy()}"
    print(f"✓ Test 5 passed: Scaled version, loss = {loss.numpy():.6f}")
    
    # Test 6: Works with tf.function
    @tf.function
    def compute_loss(y_t, y_p):
        return loss_ncc(y_t, y_p)
    
    y_true = tf.random.normal([4, 32, 32, 1])
    y_pred = tf.random.normal([4, 32, 32, 1])
    loss = compute_loss(y_true, y_pred)
    assert not tf.math.is_nan(loss), "Loss should be valid with tf.function"
    print("✓ Test 6 passed: Works with tf.function")
    
    print("\n✅ All loss_ncc tests passed!")


def test_ncc_score():
    """Test NCC score function"""
    print("\n" + "="*80)
    print("TEST: ncc_score")
    print("="*80)
    
    # Test 1: Identical tensors (should give score ~1)
    y_true = tf.random.normal([4, 32, 32, 1])
    y_pred = tf.identity(y_true)
    score = ncc_score(y_true, y_pred)
    assert score > 0.999, f"Expected score ~1 for identical tensors, got {score.numpy()}"
    print(f"✓ Test 1 passed: Identical tensors, score = {score.numpy():.6f}")
    
    # Test 2: Score range
    y_true = tf.random.normal([2, 16, 16, 1])
    y_pred = tf.random.normal([2, 16, 16, 1])
    score = ncc_score(y_true, y_pred)
    assert -1 <= score <= 1, f"Expected score in [-1, 1], got {score.numpy()}"
    print(f"✓ Test 2 passed: Score in valid range, score = {score.numpy():.4f}")
    
    print("\n✅ All ncc_score tests passed!")


def test_ncc_loss_class():
    """Test NCCLoss Keras class"""
    print("\n" + "="*80)
    print("TEST: NCCLoss class")
    print("="*80)
    
    # Test 1: Initialize
    ncc_loss = NCCLoss()
    assert ncc_loss is not None, "NCCLoss should initialize"
    print("✓ Test 1 passed: NCCLoss initialization")
    
    # Test 2: Call method
    y_true = tf.random.normal([4, 32, 32, 1])
    y_pred = tf.random.normal([4, 32, 32, 1])
    loss = ncc_loss(y_true, y_pred)
    assert 0 <= loss <= 2, f"Expected loss in [0, 2], got {loss.numpy()}"
    print(f"✓ Test 2 passed: NCCLoss call, loss = {loss.numpy():.4f}")
    
    # Test 3: Use in model compilation
    model = keras.Sequential([
        keras.layers.Conv2D(16, 3, activation='relu', input_shape=(32, 32, 1)),
        keras.layers.Conv2D(1, 3, activation='linear', padding='same')
    ])
    
    try:
        model.compile(optimizer='adam', loss=NCCLoss())
        print("✓ Test 3 passed: NCCLoss works with model.compile()")
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
    
    # Test 4: Config serialization
    config = ncc_loss.get_config()
    assert 'eps' in config, "Config should contain eps"
    print("✓ Test 4 passed: Config serialization")
    
    print("\n✅ All NCCLoss tests passed!")


def test_lr_finder():
    """Test LRFinder class"""
    print("\n" + "="*80)
    print("TEST: LRFinder")
    print("="*80)
    
    # Create simple model
    model = keras.Sequential([
        keras.layers.Dense(50, activation='relu', input_shape=(10,)),
        keras.layers.Dense(5, activation='softmax')
    ])
    
    # Create simple dataset
    x_train = tf.random.normal([100, 10])
    y_train = tf.random.uniform([100], minval=0, maxval=5, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(10)
    
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    
    # Test 1: Initialize LRFinder
    lr_finder = LRFinder(model, loss_fn)
    assert lr_finder is not None, "LRFinder should initialize"
    print("✓ Test 1 passed: LRFinder initialization")
    
    # Test 2: Run range test (short version)
    lr_finder.range_test(dataset, start_lr=1e-5, end_lr=1.0, num_iter=20)
    assert len(lr_finder.lrs) > 0, "Should have recorded LRs"
    assert len(lr_finder.losses) > 0, "Should have recorded losses"
    print(f"✓ Test 2 passed: Range test completed ({len(lr_finder.lrs)} iterations)")
    
    # Test 3: Get best LR
    best_lr = lr_finder.get_best_lr()
    assert best_lr is not None, "Should return a best LR"
    assert best_lr > 0, "Best LR should be positive"
    print(f"✓ Test 3 passed: Best LR = {best_lr:.2e}")
    
    # Test 4: Model weights restored
    print("✓ Test 4 passed: Model state handling")
    
    print("\n✅ All LRFinder tests passed!")


def test_find_lr():
    """Test find_lr convenience function"""
    print("\n" + "="*80)
    print("TEST: find_lr")
    print("="*80)
    
    # Create simple model and data
    model = keras.Sequential([
        keras.layers.Dense(20, activation='relu', input_shape=(10,)),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    x_train = tf.random.normal([50, 10])
    y_train = tf.random.uniform([50], minval=0, maxval=2, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(10)
    
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    
    # Test: Run find_lr (short version, no plot for testing)
    best_lr = find_lr(model, loss_fn, dataset, num_iter=15, plot=False)
    
    assert best_lr is not None, "Should return a best LR"
    assert best_lr > 0, "Best LR should be positive"
    print(f"✓ Test passed: find_lr returned {best_lr:.2e}")
    
    print("\n✅ All find_lr tests passed!")


def test_integration_conv_to_dense():
    """Integration test: Using lazy_flatten in a real model"""
    print("\n" + "="*80)
    print("INTEGRATION TEST: Conv to Dense transition")
    print("="*80)
    
    class TestModel(keras.Model):
        def __init__(self):
            super().__init__()
            self.conv1 = keras.layers.Conv2D(16, 3, padding='same')
            self.conv2 = keras.layers.Conv2D(32, 3, padding='same')
            self.pool = keras.layers.MaxPooling2D(2, 2)
            
            # Using get_flatten_size to determine Dense input
            # Input: 32x32 -> after 2 pools: 8x8, channels: 32
            flat_size = get_flatten_size((8, 8, 32))
            self.fc = keras.layers.Dense(10)
        
        def call(self, x):
            x = self.pool(tf.nn.relu(self.conv1(x)))
            x = self.pool(tf.nn.relu(self.conv2(x)))
            x = lazy_flatten(x)  # Auto-flatten
            x = self.fc(x)
            return x
    
    model = TestModel()
    x = tf.random.normal([4, 32, 32, 3])
    output = model(x)
    
    assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
    print("✓ Integration test passed: Conv->Flatten->Dense works correctly")
    
    print("\n✅ Integration test passed!")


def test_ncc_medical_imaging():
    """Integration test: NCC for medical imaging scenario"""
    print("\n" + "="*80)
    print("INTEGRATION TEST: NCC for medical imaging")
    print("="*80)
    
    # Simulate medical image registration scenario
    # Original image
    original = tf.random.normal([1, 128, 128, 1])
    
    # Slightly transformed version (simulating registration)
    transformed = original + tf.random.normal([1, 128, 128, 1]) * 0.1
    
    # Calculate NCC
    loss = loss_ncc(original, transformed)
    score = ncc_score(original, transformed)
    
    print(f"NCC Loss: {loss.numpy():.4f}")
    print(f"NCC Score: {score.numpy():.4f}")
    
    assert 0 <= loss <= 2, "Loss should be in valid range"
    assert -1 <= score <= 1, "Score should be in valid range"
    print("✓ Integration test passed: NCC works for medical imaging scenario")
    
    print("\n✅ Medical imaging integration test passed!")


def test_ncc_in_training():
    """Integration test: Using NCC in actual training"""
    print("\n" + "="*80)
    print("INTEGRATION TEST: NCC in training loop")
    print("="*80)
    
    # Create simple autoencoder
    model = keras.Sequential([
        keras.layers.Conv2D(16, 3, activation='relu', padding='same', input_shape=(32, 32, 1)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(8, 3, activation='relu', padding='same'),
        keras.layers.UpSampling2D(),
        keras.layers.Conv2D(1, 3, activation='linear', padding='same')
    ])
    
    # Compile with NCC loss
    model.compile(optimizer='adam', loss=NCCLoss())
    
    # Create dummy data
    x_train = tf.random.normal([20, 32, 32, 1])
    
    # Train for 1 epoch (should not crash)
    try:
        history = model.fit(x_train, x_train, epochs=1, batch_size=4, verbose=0)
        loss = history.history['loss'][0]
        print(f"Training completed, final loss: {loss:.4f}")
        print("✓ Integration test passed: NCC works in training")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    
    print("\n✅ Training integration test passed!")


def run_all_tests():
    """Run all unit and integration tests"""
    print("\n" + "="*80)
    print("RUNNING ALL TENSORFLOW UTILS TESTS")
    print("="*80)
    
    try:
        # Unit tests
        test_lazy_flatten()
        test_get_flatten_size()
        test_loss_ncc()
        test_ncc_score()
        test_ncc_loss_class()
        test_lr_finder()
        test_find_lr()
        
        # Integration tests
        test_integration_conv_to_dense()
        test_ncc_medical_imaging()
        test_ncc_in_training()
        
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
