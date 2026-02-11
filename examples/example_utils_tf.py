"""
Example usage of utility functions for TensorFlow/Keras

Demonstrates:
1. lazy_flatten - Auto-shape flattener
2. NCCLoss - Normalized Cross-Correlation loss
3. find_lr - Learning Rate Finder
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import tensorflow as tf
from tensorflow import keras
from tenf.utils import (
    lazy_flatten,
    get_flatten_size,
    loss_ncc,
    ncc_score,
    NCCLoss,
    find_lr,
    LRFinder
)
import numpy as np


def example_1_lazy_flatten():
    """Example 1: Using lazy_flatten to simplify Conv->Dense transition"""
    print("\n" + "="*80)
    print("Example 1: Auto-Shape Flattener (lazy_flatten)")
    print("="*80)
    
    class SimpleCNN(keras.Model):
        """CNN without manual size calculation"""
        def __init__(self):
            super().__init__()
            self.conv1 = keras.layers.Conv2D(32, 3, padding='same')
            self.conv2 = keras.layers.Conv2D(64, 3, padding='same')
            self.pool = keras.layers.MaxPooling2D(2, 2)
            
            # Calculate flattened size automatically
            # Input: 32x32 -> after 2 pools: 8x8, channels: 64
            flat_size = get_flatten_size((8, 8, 64))
            print(f"Calculated flatten size: {flat_size}")
            
            self.fc1 = keras.layers.Dense(128)
            self.fc2 = keras.layers.Dense(10)
        
        def call(self, x):
            x = self.pool(tf.nn.relu(self.conv1(x)))
            x = self.pool(tf.nn.relu(self.conv2(x)))
            x = lazy_flatten(x)  # No need to calculate 8*8*64!
            x = tf.nn.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleCNN()
    x = tf.random.normal([4, 32, 32, 3])
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("\n✓ Example 1 completed!")


def example_2_ncc_loss():
    """Example 2: Using NCC loss for image similarity"""
    print("\n" + "="*80)
    print("Example 2: Normalized Cross-Correlation Loss")
    print("="*80)
    
    # Simulate medical image registration task
    print("\nScenario: Medical Image Registration")
    print("Comparing original vs. registered images")
    
    # Original image
    original = tf.random.normal([1, 128, 128, 1])
    
    # Case 1: Perfect registration (identical)
    perfect = tf.identity(original)
    ncc_loss_perfect = loss_ncc(original, perfect)
    ncc_score_perfect = ncc_score(original, perfect)
    print(f"\nPerfect registration:")
    print(f"  NCC Loss: {ncc_loss_perfect.numpy():.6f} (lower is better)")
    print(f"  NCC Score: {ncc_score_perfect.numpy():.6f} (higher is better)")
    
    # Case 2: Good registration (slight noise)
    good = original + tf.random.normal([1, 128, 128, 1]) * 0.1
    ncc_loss_good = loss_ncc(original, good)
    ncc_score_good = ncc_score(original, good)
    print(f"\nGood registration (slight noise):")
    print(f"  NCC Loss: {ncc_loss_good.numpy():.6f}")
    print(f"  NCC Score: {ncc_score_good.numpy():.6f}")
    
    # Case 3: Poor registration (random image)
    poor = tf.random.normal([1, 128, 128, 1])
    ncc_loss_poor = loss_ncc(original, poor)
    ncc_score_poor = ncc_score(original, poor)
    print(f"\nPoor registration (random):")
    print(f"  NCC Loss: {ncc_loss_poor.numpy():.6f}")
    print(f"  NCC Score: {ncc_score_poor.numpy():.6f}")
    
    # Case 4: Scale invariance (NCC's advantage over MSE)
    scaled = original * 2.0
    ncc_loss_scaled = loss_ncc(original, scaled)
    mse_loss = tf.reduce_mean(tf.square(original - scaled))
    print(f"\nScale invariance test (2x brightness):")
    print(f"  NCC Loss: {ncc_loss_scaled.numpy():.6f} (should be low)")
    print(f"  MSE Loss: {mse_loss.numpy():.6f} (will be high)")
    print("  → NCC is robust to intensity scaling!")
    
    print("\n✓ Example 2 completed!")


def example_3_ncc_in_training():
    """Example 3: Training a model with NCC loss"""
    print("\n" + "="*80)
    print("Example 3: Training with NCC Loss")
    print("="*80)
    
    # Simple autoencoder for image reconstruction
    autoencoder = keras.Sequential([
        # Encoder
        keras.layers.Conv2D(16, 3, activation='relu', padding='same', input_shape=(32, 32, 1)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(8, 3, activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2, 2),
        
        # Decoder
        keras.layers.Conv2DTranspose(8, 3, strides=2, activation='relu', padding='same'),
        keras.layers.Conv2DTranspose(16, 3, strides=2, activation='relu', padding='same'),
        keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')
    ])
    
    # Compile with NCC loss
    autoencoder.compile(
        optimizer='adam',
        loss=NCCLoss(),  # Using NCCLoss class!
        metrics=['mse']  # Can still track MSE for comparison
    )
    
    # Create dummy dataset
    images = tf.random.normal([100, 32, 32, 1])
    
    # Training
    print("\nTraining for 3 epochs with NCC loss...")
    history = autoencoder.fit(
        images, images,  # Reconstruct same image
        epochs=3,
        batch_size=10,
        verbose=0
    )
    
    for epoch, (ncc_loss, mse) in enumerate(zip(history.history['loss'], 
                                                 history.history['mse'])):
        print(f"Epoch {epoch+1}/3, NCC Loss: {ncc_loss:.4f}, MSE: {mse:.4f}")
    
    print("\n✓ Example 3 completed!")


def example_4_lr_finder():
    """Example 4: Using Learning Rate Finder"""
    print("\n" + "="*80)
    print("Example 4: Learning Rate Finder")
    print("="*80)
    
    # Create a simple classification model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Create dummy MNIST-like dataset
    x_train = tf.random.normal([500, 28, 28, 1])
    y_train = tf.random.uniform([500], minval=0, maxval=10, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    
    # Method 1: Using LRFinder class
    print("\nMethod 1: Using LRFinder class")
    lr_finder = LRFinder(model, loss_fn)
    lr_finder.range_test(dataset, start_lr=1e-6, end_lr=1.0, num_iter=50)
    
    best_lr = lr_finder.get_best_lr()
    print(f"\n✓ Suggested learning rate: {best_lr:.2e}")
    
    # Uncomment to show plot:
    # lr_finder.plot()
    
    # Method 2: Using convenience function
    print("\nMethod 2: Using find_lr() convenience function")
    
    # Recreate model (LR finder modifies weights)
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    best_lr_2 = find_lr(
        model, loss_fn, dataset,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=50,
        plot=False  # Set to True to see plot
    )
    print(f"\n✓ Suggested learning rate: {best_lr_2:.2e}")
    
    # Now use the best LR for actual training
    print(f"\n💡 Now you can compile model with lr={best_lr:.2e}")
    print(f"   model.compile(optimizer=keras.optimizers.Adam(lr={best_lr:.2e}), ...)")
    
    print("\n✓ Example 4 completed!")


def example_5_complete_pipeline():
    """Example 5: Complete training pipeline using all utilities"""
    print("\n" + "="*80)
    print("Example 5: Complete Pipeline with All Utilities")
    print("="*80)
    
    # Define model using lazy_flatten and get_flatten_size
    class MedicalSegmentationModel(keras.Model):
        def __init__(self):
            super().__init__()
            # Encoder
            self.enc1 = keras.layers.Conv2D(32, 3, padding='same')
            self.enc2 = keras.layers.Conv2D(64, 3, padding='same')
            self.pool = keras.layers.MaxPooling2D(2, 2)
            
            # Bottleneck
            flat_size = get_flatten_size((16, 16, 64))  # After one pool
            self.fc1 = keras.layers.Dense(512)
            
            # Decoder (simplified)
            self.fc2 = keras.layers.Dense(flat_size)
            self.dec1 = keras.layers.Conv2DTranspose(32, 2, strides=2)
            self.dec2 = keras.layers.Conv2D(1, 3, padding='same')
        
        def call(self, x):
            # Encode
            x1 = tf.nn.relu(self.enc1(x))
            x2 = self.pool(tf.nn.relu(self.enc2(x1)))
            
            # Bottleneck
            x_flat = lazy_flatten(x2)  # Using lazy_flatten!
            x_fc = tf.nn.relu(self.fc1(x_flat))
            
            # Decode
            x_unfold = self.fc2(x_fc)
            x_unfold = tf.reshape(x_unfold, [-1, 16, 16, 64])
            x_dec = tf.nn.relu(self.dec1(x_unfold))
            x_out = tf.nn.sigmoid(self.dec2(x_dec))
            
            return x_out
    
    model = MedicalSegmentationModel()
    
    # Create dummy medical imaging dataset
    images = tf.random.normal([200, 32, 32, 1])
    masks = tf.cast(tf.random.uniform([200, 32, 32, 1]) > 0.5, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((images, masks)).batch(16)
    
    # Step 1: Find optimal learning rate
    print("\n📊 Step 1: Finding optimal learning rate...")
    
    best_lr = find_lr(
        model, NCCLoss(), dataset,
        num_iter=30, plot=False
    )
    print(f"   Found LR: {best_lr:.2e}")
    
    # Step 2: Train with optimal LR and NCC loss
    print(f"\n📚 Step 2: Training with LR={best_lr:.2e} and NCC loss...")
    
    # Recreate model
    model = MedicalSegmentationModel()
    
    # Compile with NCC loss and optimal LR
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best_lr),
        loss=NCCLoss(),
        metrics=[keras.metrics.MeanSquaredError(name='mse')]
    )
    
    # Train
    history = model.fit(
        dataset,
        epochs=2,  # Just 2 epochs for demo
        verbose=0
    )
    
    for epoch, (ncc_loss, mse) in enumerate(zip(history.history['loss'],
                                                 history.history['mse'])):
        print(f"   Epoch {epoch+1}: NCC Loss={ncc_loss:.4f}, MSE={mse:.4f}")
    
    # Step 3: Evaluate
    print(f"\n📈 Step 3: Evaluating model...")
    test_images = tf.random.normal([10, 32, 32, 1])
    test_masks = tf.cast(tf.random.uniform([10, 32, 32, 1]) > 0.5, tf.float32)
    
    predictions = model(test_images)
    test_ncc_loss = loss_ncc(test_masks, predictions)
    test_ncc_score = ncc_score(test_masks, predictions)
    
    print(f"   Test NCC Loss: {test_ncc_loss.numpy():.4f}")
    print(f"   Test NCC Score: {test_ncc_score.numpy():.4f}")
    
    print("\n✓ Complete pipeline executed successfully!")
    print("\n💡 This pipeline used:")
    print("   - get_flatten_size() to calculate layer sizes")
    print("   - lazy_flatten() to simplify Conv->Dense transition")
    print("   - find_lr() to find optimal learning rate")
    print("   - NCCLoss() for medical image segmentation")
    print("   - loss_ncc() and ncc_score() for evaluation")
    
    print("\n✓ Example 5 completed!")


def example_6_custom_model_with_flatten():
    """Example 6: Using lazy_flatten in Functional API"""
    print("\n" + "="*80)
    print("Example 6: lazy_flatten with Functional API")
    print("="*80)
    
    # Using Functional API
    inputs = keras.layers.Input(shape=(64, 64, 3))
    
    # Convolutional layers
    x = keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)
    
    # Use Lambda layer with lazy_flatten
    x = keras.layers.Lambda(lambda t: lazy_flatten(t))(x)
    
    # Dense layers
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Test
    test_input = tf.random.normal([4, 64, 64, 3])
    test_output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print("\n✓ lazy_flatten works seamlessly with Functional API!")
    
    print("\n✓ Example 6 completed!")


def run_all_examples():
    """Run all examples"""
    print("\n" + "="*80)
    print("TENSORFLOW UTILITY FUNCTIONS - EXAMPLES")
    print("="*80)
    
    example_1_lazy_flatten()
    example_2_ncc_loss()
    example_3_ncc_in_training()
    example_4_lr_finder()
    example_5_complete_pipeline()
    example_6_custom_model_with_flatten()
    
    print("\n" + "="*80)
    print("✅ ALL EXAMPLES COMPLETED!")
    print("="*80)
    print("\n📖 Summary:")
    print("   1. lazy_flatten - Simplifies Conv->Dense transitions")
    print("   2. NCCLoss - Robust loss for medical imaging (Keras-compatible)")
    print("   3. find_lr - Finds optimal learning rate automatically")
    print("\n💡 These utilities save time and improve model performance!")


if __name__ == "__main__":
    run_all_examples()
