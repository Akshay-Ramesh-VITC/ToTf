"""
Demo script to visualize complex architectures with multiple branches and cross-connections
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tenf import ModelView

def create_parallel_branches_demo():
    """Create and visualize a model with parallel branches"""
    print("Creating parallel branches model...")
    
    inputs = layers.Input(shape=(32,), name='input')
    
    # Branch 1
    branch1 = layers.Dense(64, activation='relu', name='branch1_dense1')(inputs)
    branch1 = layers.Dense(32, activation='relu', name='branch1_dense2')(branch1)
    
    # Branch 2
    branch2 = layers.Dense(64, activation='relu', name='branch2_dense1')(inputs)
    branch2 = layers.Dense(32, activation='relu', name='branch2_dense2')(branch2)
    
    # Merge branches
    merged = layers.Concatenate(name='merge_branches')([branch1, branch2])
    outputs = layers.Dense(10, activation='softmax', name='output')(merged)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='ParallelBranches')
    
    # Visualize
    view = ModelView(model, input_shape=(32,))
    view.render('parallel_branches_demo.png', show_layer_names=True, dpi=300)
    print("✓ Saved to parallel_branches_demo.png")
    view.show()
    

def create_inception_demo():
    """Create and visualize an Inception-like module"""
    print("\nCreating Inception-like module...")
    
    inputs = layers.Input(shape=(28, 28, 64))
    
    # 1x1 convolution branch
    conv1x1 = layers.Conv2D(32, 1, padding='same', activation='relu', name='conv1x1')(inputs)
    
    # 3x3 convolution branch
    conv3x3 = layers.Conv2D(32, 1, padding='same', activation='relu', name='conv3x3_reduce')(inputs)
    conv3x3 = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv3x3')(conv3x3)
    
    # 5x5 convolution branch
    conv5x5 = layers.Conv2D(16, 1, padding='same', activation='relu', name='conv5x5_reduce')(inputs)
    conv5x5 = layers.Conv2D(32, 5, padding='same', activation='relu', name='conv5x5')(conv5x5)
    
    # Max pooling branch
    pool = layers.MaxPooling2D(3, strides=1, padding='same', name='maxpool')(inputs)
    pool = layers.Conv2D(32, 1, padding='same', activation='relu', name='pool_proj')(pool)
    
    # Concatenate all branches
    concat = layers.Concatenate(name='inception_concat')([conv1x1, conv3x3, conv5x5, pool])
    
    model = keras.Model(inputs=inputs, outputs=concat, name='InceptionModule')
    
    # Visualize
    view = ModelView(model, input_shape=(28, 28, 64))
    view.render('inception_module_demo.png', show_layer_names=True, dpi=300)
    print("✓ Saved to inception_module_demo.png")
    view.show()


def create_densenet_demo():
    """Create and visualize a DenseNet-like block"""
    print("\nCreating DenseNet-like block...")
    
    inputs = layers.Input(shape=(32, 32, 64), name='input')
    
    # Layer 1
    x1 = layers.Conv2D(32, 3, padding='same', activation='relu', name='conv1')(inputs)
    
    # Layer 2 (connected to input and x1)
    concat1 = layers.Concatenate(name='concat1')([inputs, x1])
    x2 = layers.Conv2D(32, 3, padding='same', activation='relu', name='conv2')(concat1)
    
    # Layer 3 (connected to input, x1, and x2)
    concat2 = layers.Concatenate(name='concat2')([inputs, x1, x2])
    x3 = layers.Conv2D(32, 3, padding='same', activation='relu', name='conv3')(concat2)
    
    # Layer 4 (connected to all previous)
    concat3 = layers.Concatenate(name='concat3')([inputs, x1, x2, x3])
    x4 = layers.Conv2D(64, 1, activation='relu', name='transition')(concat3)
    
    model = keras.Model(inputs=inputs, outputs=x4, name='DenseBlock')
    
    # Visualize
    view = ModelView(model, input_shape=(32, 32, 64))
    view.render('densenet_block_demo.png', show_layer_names=True, dpi=300)
    print("✓ Saved to densenet_block_demo.png")
    view.show()


def create_complex_dag_demo():
    """Create and visualize a complex DAG structure"""
    print("\nCreating complex DAG...")
    
    inputs = layers.Input(shape=(100,), name='input')
    
    # Level 1: 3 parallel branches
    l1a = layers.Dense(64, activation='relu', name='l1a')(inputs)
    l1b = layers.Dense(64, activation='relu', name='l1b')(inputs)
    l1c = layers.Dense(64, activation='relu', name='l1c')(inputs)
    
    # Level 2: Cross connections
    l2a = layers.Dense(64, activation='relu', name='l2a')(
        layers.Concatenate(name='merge_l1a_l1b')([l1a, l1b])
    )
    l2b = layers.Dense(64, activation='relu', name='l2b')(
        layers.Concatenate(name='merge_l1b_l1c')([l1b, l1c])
    )
    l2c = layers.Dense(64, activation='relu', name='l2c')(
        layers.Concatenate(name='merge_l1a_l1c')([l1a, l1c])
    )
    
    # Level 3: Merge all paths
    l3 = layers.Concatenate(name='merge_all_l2')([l2a, l2b, l2c])
    l3 = layers.Dense(128, activation='relu', name='l3_dense')(l3)
    
    # Add skip from input to level 3
    input_proj = layers.Dense(128, activation='relu', name='input_projection')(inputs)
    l3_with_skip = layers.Add(name='final_skip')([l3, input_proj])
    
    outputs = layers.Dense(10, activation='softmax', name='output')(l3_with_skip)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='ComplexDAG')
    
    # Visualize
    view = ModelView(model, input_shape=(100,))
    view.render('complex_dag_demo.png', show_layer_names=True, dpi=300, rankdir='TB')
    print("✓ Saved to complex_dag_demo.png")
    view.show()


def create_multi_output_demo():
    """Create and visualize multi-output model with branches"""
    print("\nCreating multi-output multi-branch model...")
    
    inputs = layers.Input(shape=(128,), name='input')
    
    # Shared bottom layers
    shared = layers.Dense(256, activation='relu', name='shared1')(inputs)
    shared = layers.Dense(128, activation='relu', name='shared2')(shared)
    
    # Branch for output 1
    branch1 = layers.Dense(64, activation='relu', name='branch1_dense1')(shared)
    branch1 = layers.Dense(32, activation='relu', name='branch1_dense2')(branch1)
    output1 = layers.Dense(5, activation='softmax', name='output1')(branch1)
    
    # Branch for output 2
    branch2 = layers.Dense(64, activation='relu', name='branch2_dense1')(shared)
    branch2 = layers.Dense(32, activation='relu', name='branch2_dense2')(branch2)
    output2 = layers.Dense(3, activation='softmax', name='output2')(branch2)
    
    # Branch for output 3 - uses both previous branches
    branch3 = layers.Concatenate(name='branch3_merge')([branch1, branch2])
    branch3 = layers.Dense(64, activation='relu', name='branch3_dense')(branch3)
    output3 = layers.Dense(10, activation='sigmoid', name='output3')(branch3)
    
    model = keras.Model(
        inputs=inputs,
        outputs=[output1, output2, output3],
        name='MultiOutputMultiBranch'
    )
    
    # Visualize
    view = ModelView(model, input_shape=(128,))
    view.render('multi_output_demo.png', show_layer_names=True, dpi=300)
    print("✓ Saved to multi_output_demo.png")
    view.show()


if __name__ == '__main__':
    print("=" * 70)
    print("ModelView - Complex Architecture Visualization Demo")
    print("=" * 70)
    
    try:
        import graphviz
        print("✓ Graphviz is available\n")
    except ImportError:
        print("✗ Graphviz not installed. Please install: pip install graphviz")
        print("  And system graphviz: conda install -c conda-forge graphviz")
        exit(1)
    
    # Run all demos
    create_parallel_branches_demo()
    create_inception_demo()
    create_densenet_demo()
    create_complex_dag_demo()
    create_multi_output_demo()
    
    print("\n" + "=" * 70)
    print("✓ All demos completed successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - parallel_branches_demo.png")
    print("  - inception_module_demo.png")
    print("  - densenet_block_demo.png")
    print("  - complex_dag_demo.png")
    print("  - multi_output_demo.png")
    print("\nAll diagrams are publication-quality (300 DPI) PNG files.")
