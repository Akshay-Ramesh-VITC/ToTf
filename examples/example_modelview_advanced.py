"""
Example demonstrating ModelView's torchview-like features for complex architectures

This example shows how to use the advanced visualization features:
- hide_inner_tensors: Control visibility of intermediate tensors
- hide_module_functions: Control visibility of operations inside layers
- expand_nested: Show nested models with dashed borders
- depth: Control how deep to show in module hierarchy
- roll: Roll/unroll recursive structures
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tenf.modelview import ModelView, draw_graph


def create_simple_model():
    """Simple sequential model for basic testing"""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(100,), name='dense_1'),
        layers.Dropout(0.5, name='dropout'),
        layers.Dense(32, activation='relu', name='dense_2'),
        layers.Dense(10, activation='softmax', name='output')
    ], name='SimpleModel')
    return model


def create_resnet_block(inputs, filters, stride=1, name='resnet_block'):
    """Create a ResNet-style residual block"""
    # Main path
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', name=f'{name}_conv1')(inputs)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Activation('relu', name=f'{name}_relu1')(x)
    
    x = layers.Conv2D(filters, 3, padding='same', name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    
    # Shortcut path
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', name=f'{name}_shortcut')(inputs)
        shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)
    else:
        shortcut = inputs
    
    # Add shortcut
    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.Activation('relu', name=f'{name}_relu2')(x)
    
    return x


def create_resnet_model():
    """Create a mini ResNet-style model with skip connections"""
    inputs = keras.Input(shape=(32, 32, 3), name='input')
    
    x = layers.Conv2D(32, 3, padding='same', name='initial_conv')(inputs)
    x = layers.BatchNormalization(name='initial_bn')(x)
    x = layers.Activation('relu', name='initial_relu')(x)
    
    # ResNet blocks
    x = create_resnet_block(x, 32, name='block1')
    x = create_resnet_block(x, 64, stride=2, name='block2')
    x = create_resnet_block(x, 64, name='block3')
    
    # Classification head
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    outputs = layers.Dense(10, activation='softmax', name='classifier')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='MiniResNet')
    return model


def create_multi_input_model():
    """Create a model with multiple inputs"""
    # Image input
    image_input = keras.Input(shape=(32, 32, 3), name='image_input')
    x1 = layers.Conv2D(32, 3, activation='relu', name='conv1')(image_input)
    x1 = layers.MaxPooling2D(2, name='pool1')(x1)
    x1 = layers.Conv2D(64, 3, activation='relu', name='conv2')(x1)
    x1 = layers.GlobalAveragePooling2D(name='gap')(x1)
    
    # Metadata input
    meta_input = keras.Input(shape=(10,), name='metadata_input')
    x2 = layers.Dense(32, activation='relu', name='meta_dense')(meta_input)
    
    # Concatenate
    combined = layers.Concatenate(name='concat')([x1, x2])
    x = layers.Dense(64, activation='relu', name='combined_dense')(combined)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = keras.Model(inputs=[image_input, meta_input], outputs=outputs, name='MultiInputModel')
    return model


def create_inception_module(inputs, filters, name='inception'):
    """Create an Inception-style module with parallel branches"""
    # 1x1 conv branch
    branch1 = layers.Conv2D(filters, 1, padding='same', activation='relu', name=f'{name}_1x1')(inputs)
    
    # 3x3 conv branch
    branch2 = layers.Conv2D(filters, 1, padding='same', activation='relu', name=f'{name}_3x3_reduce')(inputs)
    branch2 = layers.Conv2D(filters, 3, padding='same', activation='relu', name=f'{name}_3x3')(branch2)
    
    # 5x5 conv branch
    branch3 = layers.Conv2D(filters, 1, padding='same', activation='relu', name=f'{name}_5x5_reduce')(inputs)
    branch3 = layers.Conv2D(filters, 5, padding='same', activation='relu', name=f'{name}_5x5')(branch3)
    
    # Max pooling branch
    branch4 = layers.MaxPooling2D(3, strides=1, padding='same', name=f'{name}_pool')(inputs)
    branch4 = layers.Conv2D(filters, 1, padding='same', activation='relu', name=f'{name}_pool_proj')(branch4)
    
    # Concatenate all branches
    output = layers.Concatenate(name=f'{name}_concat')([branch1, branch2, branch3, branch4])
    return output


def create_inception_model():
    """Create a model with Inception-style modules"""
    inputs = keras.Input(shape=(32, 32, 3), name='input')
    
    x = layers.Conv2D(32, 3, padding='same', activation='relu', name='initial_conv')(inputs)
    x = create_inception_module(x, 16, name='inception1')
    x = layers.MaxPooling2D(2, name='pool')(x)
    x = create_inception_module(x, 32, name='inception2')
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    outputs = layers.Dense(10, activation='softmax', name='classifier')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='MiniInception')
    return model


def create_nested_model():
    """Create a model with nested submodels"""
    # Create a reusable block as a separate model
    block_input = keras.Input(shape=(32,))
    x = layers.Dense(64, activation='relu')(block_input)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    feature_block = keras.Model(block_input, x, name='FeatureBlock')
    
    # Main model using the block
    main_input = keras.Input(shape=(32,), name='input')
    x = feature_block(main_input)
    x = layers.Dense(32, activation='relu', name='dense_1')(x)  # Changed to 32 to match block input
    x = feature_block(x)  # Reuse the block
    outputs = layers.Dense(10, activation='softmax', name='output')(x)
    
    model = keras.Model(main_input, outputs, name='NestedModel')
    return model


def demonstrate_basic_usage():
    """Demonstrate basic ModelView usage"""
    print("=" * 80)
    print("1. Basic Sequential Model")
    print("=" * 80)
    
    model = create_simple_model()
    view = ModelView(model, input_shape=(100,))
    
    # Show text summary
    view.show()
    
    # Render visualization (using legacy method)
    output = view.render('outputs/simple_model.png', show_shapes=True, show_params=True)
    print(f"\nVisualization saved to: {output}\n")


def demonstrate_advanced_features():
    """Demonstrate advanced torchview-like features"""
    print("=" * 80)
    print("2. ResNet with Advanced Features")
    print("=" * 80)
    
    model = create_resnet_model()
    
    # Example 1: Default view (hide inner tensors, hide module functions)
    print("\n2a. Default view (hide inner tensors & operations)")
    view1 = ModelView(
        model, 
        input_shape=(32, 32, 3),
        hide_inner_tensors=True,
        hide_module_functions=True,
        depth=3
    )
    output1 = view1.render_advanced('outputs/resnet_default.png', show_shapes=True)
    print(f"Saved to: {output1}")
    
    # Example 2: Show all tensors
    print("\n2b. Show all intermediate tensors")
    view2 = ModelView(
        model,
        input_shape=(32, 32, 3),
        hide_inner_tensors=False,  # Show all tensors
        hide_module_functions=True,
        depth=3
    )
    output2 = view2.render_advanced('outputs/resnet_all_tensors.png', show_shapes=True)
    print(f"Saved to: {output2}")
    
    # Example 3: Show operations inside modules
    print("\n2c. Show operations inside modules")
    view3 = ModelView(
        model,
        input_shape=(32, 32, 3),
        hide_inner_tensors=True,
        hide_module_functions=False,  # Show operations
        depth=3
    )
    output3 = view3.render_advanced('outputs/resnet_with_ops.png', show_shapes=True)
    print(f"Saved to: {output3}")


def demonstrate_depth_control():
    """Demonstrate depth control for nested models"""
    print("=" * 80)
    print("3. Nested Model with Depth Control")
    print("=" * 80)
    
    model = create_nested_model()
    
    # Depth = 1 (shallow)
    print("\n3a. Depth = 1 (shallow view)")
    view1 = ModelView(model, input_shape=(32,), depth=1)
    output1 = view1.render_advanced('outputs/nested_depth1.png')
    print(f"Saved to: {output1}")
    
    # Depth = 3 (deeper)
    print("\n3b. Depth = 3 (deeper view)")
    view2 = ModelView(model, input_shape=(32,), depth=3, expand_nested=True)
    output2 = view2.render_advanced('outputs/nested_depth3.png')
    print(f"Saved to: {output2}")


def demonstrate_expand_nested():
    """Demonstrate expand_nested with dashed borders"""
    print("=" * 80)
    print("4. Expand Nested Modules")
    print("=" * 80)
    
    model = create_nested_model()
    
    # Without expand_nested
    print("\n4a. Without expand_nested")
    view1 = ModelView(model, input_shape=(32,), expand_nested=False, depth=2)
    output1 = view1.render_advanced('outputs/nested_collapsed.png')
    print(f"Saved to: {output1}")
    
    # With expand_nested (shows nested models with dashed borders)
    print("\n4b. With expand_nested (dashed borders)")
    view2 = ModelView(model, input_shape=(32,), expand_nested=True, depth=3)
    output2 = view2.render_advanced('outputs/nested_expanded.png')
    print(f"Saved to: {output2}")


def demonstrate_complex_architectures():
    """Demonstrate handling of complex architectures"""
    print("=" * 80)
    print("5. Complex Architectures")
    print("=" * 80)
    
    # Multi-input model
    print("\n5a. Multi-input model")
    model1 = create_multi_input_model()
    view1 = ModelView(model1, input_shape=[(32, 32, 3), (10,)])
    output1 = view1.render_advanced('outputs/multi_input.png', show_shapes=True)
    print(f"Saved to: {output1}")
    
    # Inception-style parallel branches
    print("\n5b. Inception-style parallel branches")
    model2 = create_inception_model()
    view2 = ModelView(model2, input_shape=(32, 32, 3), depth=2)
    output2 = view2.render_advanced('outputs/inception.png', show_shapes=True)
    print(f"Saved to: {output2}")


def demonstrate_different_formats():
    """Demonstrate different output formats"""
    print("=" * 80)
    print("6. Different Output Formats")
    print("=" * 80)
    
    model = create_simple_model()
    view = ModelView(model, input_shape=(100,))
    
    # PNG (default)
    print("\n6a. PNG format (300 DPI)")
    output1 = view.render_advanced('outputs/format_example.png', dpi=300)
    print(f"Saved to: {output1}")
    
    # PDF
    print("\n6b. PDF format (vector)")
    output2 = view.render_advanced('outputs/format_example.pdf', format='pdf')
    print(f"Saved to: {output2}")
    
    # SVG
    print("\n6c. SVG format (vector)")
    output3 = view.render_advanced('outputs/format_example.svg', format='svg')
    print(f"Saved to: {output3}")


def main():
    """Run all demonstrations"""
    import os
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    print("\n" + "="*80)
    print("ModelView Advanced Features Demonstration")  
    print("Torchview-like capabilities for TensorFlow/Keras")
    print("="*80 + "\n")
    
    try:
        demonstrate_basic_usage()
        demonstrate_advanced_features()
        demonstrate_depth_control()
        demonstrate_expand_nested()
        demonstrate_complex_architectures()
        demonstrate_different_formats()
        
        print("\n" + "="*80)
        print("All demonstrations completed successfully!")
        print("Check the 'outputs' directory for generated visualizations.")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
