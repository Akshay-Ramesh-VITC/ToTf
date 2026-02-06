"""
Test SmartSummary for TensorFlow/Keras

Run this to verify the TensorFlow SmartSummary implementation works correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import tensorflow as tf
    from tensorflow import keras
    from tenf import SmartSummary
    
    print("TensorFlow version:", tf.__version__)
    print("="*80)
    
    # Test 1: Simple Sequential Model
    print("\n[TEST 1] Simple Sequential Model")
    print("-"*80)
    model1 = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(100,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    summary1 = SmartSummary(model1, input_shape=(100,))
    summary1.show()
    print("✓ Test 1 passed\n")
    
    # Test 2: CNN Model
    print("\n[TEST 2] CNN Model")
    print("-"*80)
    model2 = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=(64, 64, 3)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    summary2 = SmartSummary(model2, input_shape=(64, 64, 3))
    bottlenecks = summary2.get_bottlenecks()
    print(f"\nFound {len(bottlenecks)} bottleneck(s)")
    print("✓ Test 2 passed\n")
    
    # Test 3: Gradient Tracking
    print("\n[TEST 3] Gradient Tracking")
    print("-"*80)
    model3 = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(50,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(5, activation='softmax')
    ])
    
    summary3 = SmartSummary(model3, input_shape=(50,), track_gradients=True)
    summary3.show()
    print("✓ Test 3 passed\n")
    
    # Test 4: Export to Dict and File
    print("\n[TEST 4] Export to Dict and File")
    print("-"*80)
    data = summary2.to_dict()
    print(f"Exported {len(data['layers'])} layers")
    print(f"Total params: {data['total_params']:,}")
    print(f"Trainable params: {data['trainable_params']:,}")
    
    summary2.save_to_file("test_tf_summary.txt")
    if os.path.exists("test_tf_summary.txt"):
        print("✓ File saved successfully")
        # Cleanup
        os.remove("test_tf_summary.txt")
    print("✓ Test 4 passed\n")
    
    # Test 5: Multi-Input Model
    print("\n[TEST 5] Multi-Input Model")
    print("-"*80)
    input1 = keras.layers.Input(shape=(32, 32, 3))
    input2 = keras.layers.Input(shape=(10,))
    
    x1 = keras.layers.Conv2D(16, 3)(input1)
    x1 = keras.layers.Flatten()(x1)
    
    x2 = keras.layers.Dense(32)(input2)
    
    combined = keras.layers.concatenate([x1, x2])
    output = keras.layers.Dense(5)(combined)
    
    model5 = keras.Model(inputs=[input1, input2], outputs=output)
    
    summary5 = SmartSummary(model5, input_shape=[(32, 32, 3), (10,)])
    summary5.show()
    print("✓ Test 5 passed\n")
    
    # Test 6: Compare with Keras Summary
    print("\n[TEST 6] Compare with Keras Summary")
    print("-"*80)
    summary1.compare_with_keras_summary()
    print("✓ Test 6 passed\n")
    
    print("="*80)
    print("✅ All tests passed successfully!")
    print("="*80)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install TensorFlow: pip install tensorflow>=2.13.0")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
