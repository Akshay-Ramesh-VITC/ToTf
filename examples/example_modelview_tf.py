"""
Example Usage: TensorFlow ModelView - Generate Publication-Quality Architecture Diagrams

This example demonstrates how to use ModelView to generate high-quality
neural network architecture diagrams suitable for research papers.

Features demonstrated:
1. Simple Sequential models
2. Complex CNN architectures
3. Residual networks
4. Multi-input models
5. Custom styling and layouts
6. Different output formats (PNG, PDF, SVG)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# Import ModelView
from ToTf.tenf import ModelView, draw_graph


def example1_simple_sequential():
    """Example 1: Simple Sequential MLP"""
    print("\n=== Example 1: Simple Sequential Model ===")
    
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ], name='SimpleMLP')
    
    # Quick visualization
    view = ModelView(model, input_shape=(784,))
    view.show()
    
    # Save as PNG for presentations
    view.render('outputs/simple_mlp.png', dpi=300)
    print("✓ Saved to outputs/simple_mlp.png")
    
    # Save as PDF for LaTeX papers
    view.render('outputs/simple_mlp.pdf', format='pdf')
    print("✓ Saved to outputs/simple_mlp.pdf")
    
    return model


def example2_cnn_architecture():
    """Example 2: CNN for Image Classification"""
    print("\n=== Example 2: CNN Architecture ===")
    
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(64, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(128, 3, activation='relu'),
        layers.BatchNormalization(),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ], name='MNIST_CNN')
    
    view = ModelView(model, input_shape=(28, 28, 1))
    view.show(detailed=True)
    
    # Horizontal layout (left-to-right) for wide figures
    view.render('outputs/cnn_horizontal.png', rankdir='LR', dpi=300)
    print("✓ Saved horizontal layout")
    
    # Vertical layout (top-to-bottom) for narrow figures
    view.render('outputs/cnn_vertical.png', rankdir='TB', dpi=300)
    print("✓ Saved vertical layout")
    
    # SVG for perfect scaling
    view.render('outputs/cnn_scalable.svg', format='svg')
    print("✓ Saved SVG (scalable vector graphics)")
    
    return model


def example3_resnet_block():
    """Example 3: ResNet-like Architecture with Skip Connections"""
    print("\n=== Example 3: ResNet with Skip Connections ===")
    
    def residual_block(x, filters, name_prefix):
        """Create a residual block"""
        residual = x
        
        # Adjust residual if needed
        if x.shape[-1] != filters:
            residual = layers.Conv2D(
                filters, 1, padding='same',
                name=f'{name_prefix}_residual_conv'
            )(residual)
        
        # Main path
        x = layers.Conv2D(
            filters, 3, padding='same', activation='relu',
            name=f'{name_prefix}_conv1'
        )(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        
        x = layers.Conv2D(
            filters, 3, padding='same',
            name=f'{name_prefix}_conv2'
        )(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
        
        # Add skip connection
        x = layers.Add(name=f'{name_prefix}_add')([x, residual])
        x = layers.Activation('relu', name=f'{name_prefix}_relu')(x)
        
        return x
    
    # Build ResNet-like model
    inputs = keras.Input(shape=(32, 32, 3), name='input_image')
    
    x = layers.Conv2D(32, 3, padding='same', activation='relu', name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    
    x = residual_block(x, 32, 'block1')
    x = residual_block(x, 64, 'block2')
    x = residual_block(x, 128, 'block3')
    
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(10, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='MiniResNet')
    
    view = ModelView(model, input_shape=(32, 32, 3))
    view.show(detailed=True)
    
    # High-DPI PNG for papers
    view.render('outputs/resnet_architecture.png', dpi=600, show_params=True)
    print("✓ Saved high-resolution ResNet architecture")
    
    # Export summary as JSON for documentation
    view.save_summary_json('outputs/resnet_summary.json')
    print("✓ Saved architecture summary as JSON")
    
    return model


def example4_multi_input_model():
    """Example 4: Multi-Input Model (e.g., for multimodal learning)"""
    print("\n=== Example 4: Multi-Input Model ===")
    
    # Text input branch
    text_input = keras.Input(shape=(100,), name='text_input')
    x1 = layers.Embedding(10000, 128, name='text_embedding')(text_input)
    x1 = layers.LSTM(64, name='text_lstm')(x1)
    x1 = layers.Dense(32, activation='relu', name='text_dense')(x1)
    
    # Image input branch
    image_input = keras.Input(shape=(64, 64, 3), name='image_input')
    x2 = layers.Conv2D(32, 3, activation='relu', name='img_conv1')(image_input)
    x2 = layers.MaxPooling2D(2, name='img_pool1')(x2)
    x2 = layers.Conv2D(64, 3, activation='relu', name='img_conv2')(x2)
    x2 = layers.GlobalAveragePooling2D(name='img_gap')(x2)
    x2 = layers.Dense(32, activation='relu', name='img_dense')(x2)
    
    # Numerical features input
    num_input = keras.Input(shape=(20,), name='numerical_input')
    x3 = layers.Dense(32, activation='relu', name='num_dense')(num_input)
    
    # Merge all branches
    merged = layers.Concatenate(name='concatenate')([x1, x2, x3])
    x = layers.Dense(64, activation='relu', name='merged_dense1')(merged)
    x = layers.Dropout(0.3, name='merged_dropout')(x)
    x = layers.Dense(32, activation='relu', name='merged_dense2')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = keras.Model(
        inputs=[text_input, image_input, num_input],
        outputs=outputs,
        name='MultimodalModel'
    )
    
    view = ModelView(
        model,
        input_shape=[(100,), (64, 64, 3), (20,)]
    )
    view.show(detailed=True)
    
    # Left-to-right layout works better for wide architectures
    view.render('outputs/multimodal_model.png', rankdir='LR', dpi=300)
    print("✓ Saved multimodal architecture")
    
    return model


def example5_custom_styling():
    """Example 5: Custom Styling for Different Aesthetics"""
    print("\n=== Example 5: Custom Styling ===")
    
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='softmax')
    ], name='StyledCNN')
    
    view = ModelView(model, input_shape=(32, 32, 3))
    
    # Custom node styling (professional look)
    custom_node_style = {
        'shape': 'box',
        'style': 'rounded,filled',
        'fillcolor': '#f0f0f0',
        'fontname': 'Helvetica',
        'fontsize': '11',
        'color': '#333333',
        'penwidth': '2'
    }
    
    # Custom edge styling
    custom_edge_style = {
        'color': '#666666',
        'penwidth': '2',
        'arrowsize': '1.0'
    }
    
    view.render(
        'outputs/custom_styled.png',
        node_style=custom_node_style,
        edge_style=custom_edge_style,
        dpi=300
    )
    print("✓ Saved custom-styled architecture")
    
    return model


def example6_convenience_function():
    """Example 6: Using the Convenience draw_graph Function"""
    print("\n=== Example 6: Quick Visualization with draw_graph ===")
    
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(100,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    # Quick visualization without creating ModelView object
    draw_graph(
        model,
        input_shape=(100,),
        save_path='outputs/quick_viz.png',
        dpi=300
    )
    print("✓ Quick visualization saved")


def example7_attention_transformer():
    """Example 7: Transformer/Attention Architecture"""
    print("\n=== Example 7: Transformer Architecture ===")
    
    # Simple transformer encoder block
    inputs = keras.Input(shape=(50, 128), name='sequence_input')
    
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(
        num_heads=8,
        key_dim=32,
        name='multi_head_attention'
    )(inputs, inputs)
    
    # Add & Norm
    x = layers.Add(name='add_attention')([inputs, attention_output])
    x = layers.LayerNormalization(name='norm_attention')(x)
    
    # Feed-forward network
    ff = layers.Dense(512, activation='relu', name='ff_dense1')(x)
    ff = layers.Dropout(0.1, name='ff_dropout')(ff)
    ff = layers.Dense(128, name='ff_dense2')(ff)
    
    # Add & Norm
    x = layers.Add(name='add_ff')([x, ff])
    x = layers.LayerNormalization(name='norm_ff')(x)
    
    # Output
    x = layers.GlobalAveragePooling1D(name='global_pool')(x)
    outputs = layers.Dense(10, activation='softmax', name='classification')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='TransformerEncoder')
    
    view = ModelView(model, input_shape=(50, 128))
    view.show(detailed=True)
    
    view.render('outputs/transformer.png', rankdir='TB', dpi=400)
    print("✓ Saved transformer architecture")
    
    return model


def example8_autoencoder():
    """Example 8: Autoencoder Architecture"""
    print("\n=== Example 8: Autoencoder ===")
    
    # Encoder
    encoder_input = keras.Input(shape=(28, 28, 1), name='input')
    x = layers.Conv2D(32, 3, activation='relu', padding='same', name='enc_conv1')(encoder_input)
    x = layers.MaxPooling2D(2, name='enc_pool1')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same', name='enc_conv2')(x)
    x = layers.MaxPooling2D(2, name='enc_pool2')(x)
    x = layers.Flatten(name='enc_flatten')(x)
    latent = layers.Dense(32, activation='relu', name='latent')(x)
    
    # Decoder
    x = layers.Dense(7 * 7 * 64, activation='relu', name='dec_dense')(latent)
    x = layers.Reshape((7, 7, 64), name='dec_reshape')(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same', name='dec_conv1')(x)
    x = layers.UpSampling2D(2, name='dec_upsample1')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', padding='same', name='dec_conv2')(x)
    x = layers.UpSampling2D(2, name='dec_upsample2')(x)
    decoder_output = layers.Conv2D(1, 3, activation='sigmoid', padding='same', name='output')(x)
    
    model = keras.Model(inputs=encoder_input, outputs=decoder_output, name='Autoencoder')
    
    view = ModelView(model, input_shape=(28, 28, 1))
    view.show()
    
    view.render('outputs/autoencoder.png', rankdir='TB', dpi=300)
    print("✓ Saved autoencoder architecture")
    
    return model


def main():
    """Run all examples"""
    print("=" * 80)
    print("TensorFlow ModelView Examples")
    print("Publication-Quality Neural Network Architecture Diagrams")
    print("=" * 80)
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    try:
        # Run all examples
        example1_simple_sequential()
        example2_cnn_architecture()
        example3_resnet_block()
        example4_multi_input_model()
        example5_custom_styling()
        example6_convenience_function()
        example7_attention_transformer()
        example8_autoencoder()
        
        print("\n" + "=" * 80)
        print("✓ All examples completed successfully!")
        print("✓ Check the 'outputs/' directory for generated diagrams")
        print("=" * 80)
        
    except ImportError as e:
        print(f"\n⚠ Error: {e}")
        print("\nTo use ModelView, install required dependencies:")
        print("  pip install graphviz pillow")
        print("\nAlso install system graphviz:")
        print("  Ubuntu: sudo apt-get install graphviz")
        print("  macOS: brew install graphviz")
        print("  Windows: Download from https://graphviz.org/download/")


if __name__ == '__main__':
    main()
