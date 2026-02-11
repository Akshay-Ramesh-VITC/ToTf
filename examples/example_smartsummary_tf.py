"""
Example usage of SmartSummary for TensorFlow/Keras models

This script demonstrates various features of the SmartSummary tool.
"""

import tensorflow as tf
from tensorflow import keras
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tenf import SmartSummary


def example_1_simple_cnn():
    """Example 1: Basic CNN model analysis"""
    print("\n" + "="*80)
    print("Example 1: Simple CNN Model")
    print("="*80)
    
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, 3, activation='relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    summary = SmartSummary(model, input_shape=(224, 224, 3))
    summary.show()


def example_2_with_gradients():
    """Example 2: Model analysis with gradient tracking"""
    print("\n" + "="*80)
    print("Example 2: Model with Gradient Tracking")
    print("="*80)
    
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    summary = SmartSummary(
        model, 
        input_shape=(784,),
        track_gradients=True  # Enable gradient analysis
    )
    summary.show()


def example_3_resnet_style():
    """Example 3: ResNet-style model with skip connections"""
    print("\n" + "="*80)
    print("Example 3: ResNet-Style Model")
    print("="*80)
    
    inputs = keras.layers.Input(shape=(224, 224, 3))
    
    # Initial conv
    x = keras.layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual block
    shortcut = x
    x = keras.layers.Conv2D(64, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(64, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    
    # Global pooling and classifier
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    outputs = keras.layers.Dense(1000, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    summary = SmartSummary(model, input_shape=(224, 224, 3))
    summary.show()


def example_4_multi_input():
    """Example 4: Multi-input model"""
    print("\n" + "="*80)
    print("Example 4: Multi-Input Model")
    print("="*80)
    
    # Image input
    image_input = keras.layers.Input(shape=(224, 224, 3), name='image')
    x1 = keras.layers.Conv2D(32, 3, activation='relu')(image_input)
    x1 = keras.layers.GlobalAveragePooling2D()(x1)
    
    # Tabular input
    tabular_input = keras.layers.Input(shape=(100,), name='tabular')
    x2 = keras.layers.Dense(64, activation='relu')(tabular_input)
    
    # Combine
    combined = keras.layers.concatenate([x1, x2])
    x = keras.layers.Dense(128, activation='relu')(combined)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs=[image_input, tabular_input], outputs=outputs)
    
    summary = SmartSummary(
        model, 
        input_shape=[(224, 224, 3), (100,)]
    )
    summary.show()


def example_5_bottleneck_analysis():
    """Example 5: Detailed bottleneck analysis"""
    print("\n" + "="*80)
    print("Example 5: Bottleneck Analysis")
    print("="*80)
    
    # Create a model with clear bottlenecks
    model = keras.Sequential([
        keras.layers.Dense(2048, activation='relu', input_shape=(1000,)),  # Large layer
        keras.layers.Dense(2048, activation='relu'),  # Large layer
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    summary = SmartSummary(model, input_shape=(1000,))
    
    # Get bottlenecks programmatically
    bottlenecks = summary.get_bottlenecks(top_n=3)
    
    print("\n" + "="*80)
    print("Bottleneck Details:")
    print("="*80)
    for i, bn in enumerate(bottlenecks, 1):
        print(f"\n{i}. {bn['layer']}")
        print(f"   Type: {bn['layer_name']}")
        print(f"   Score: {bn['score']:.2f}")
        print(f"   Reasons: {', '.join(bn['reasons'])}")
        print(f"   Parameters: {bn['params']:,}")


def example_6_export_and_save():
    """Example 6: Export summary data and save to file"""
    print("\n" + "="*80)
    print("Example 6: Export and Save")
    print("="*80)
    
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=(64, 64, 3)),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    summary = SmartSummary(model, input_shape=(64, 64, 3))
    
    # Export as dictionary
    data = summary.to_dict()
    print(f"\nExported {len(data['layers'])} layers")
    print(f"Total parameters: {data['total_params']:,}")
    print(f"Bottlenecks found: {len(data['bottlenecks'])}")
    
    # Save to file
    summary.save_to_file("tensorflow_model_summary.txt")
    print("\n✓ Summary saved to tensorflow_model_summary.txt")


def example_7_compare_with_keras():
    """Example 7: Compare with Keras built-in summary"""
    print("\n" + "="*80)
    print("Example 7: Comparison with Keras Summary")
    print("="*80)
    
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(100,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    summary = SmartSummary(model, input_shape=(100,))
    summary.compare_with_keras_summary()


def example_8_mobilenet():
    """Example 8: Analyze a pre-trained MobileNetV2"""
    print("\n" + "="*80)
    print("Example 8: Pre-trained MobileNetV2 Analysis")
    print("="*80)
    
    # Load pre-trained model
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=True,
        weights=None  # Use None to avoid downloading weights for demo
    )
    
    summary = SmartSummary(base_model, input_shape=(224, 224, 3))
    summary.show()
    
    # Show top 10 bottlenecks
    print("\nTop 10 Bottleneck Layers:")
    bottlenecks = summary.get_bottlenecks(top_n=10)
    for i, bn in enumerate(bottlenecks, 1):
        print(f"{i}. {bn['layer']}: Score = {bn['score']:.2f}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SmartSummary for TensorFlow/Keras - Examples")
    print("="*80)
    
    # Run examples
    example_1_simple_cnn()
    example_2_with_gradients()
    example_3_resnet_style()
    example_4_multi_input()
    example_5_bottleneck_analysis()
    example_6_export_and_save()
    example_7_compare_with_keras()
    
    # Uncomment to run MobileNet example (may take longer)
    # example_8_mobilenet()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)
