"""Test script to understand Keras connection API"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create a simple model with branches
inputs = layers.Input(shape=(32,))
x1 = layers.Dense(64, name='d1')(inputs)
x2 = layers.Dense(64, name='d2')(inputs)
merged = layers.Concatenate(name='concat')([x1, x2])
output = layers.Dense(10, name='output')(merged)

model = keras.Model(inputs=inputs, outputs=output)

print('=== Layers ===')
for l in model.layers:
    print(f'{l.name}: {type(l).__name__}')

print('\n=== Testing connection extraction ===')
for layer in model.layers:
    print(f'\nLayer: {layer.name}')
    if hasattr(layer, '_inbound_nodes'):
        print(f'  Number of inbound nodes: {len(layer._inbound_nodes)}')
        for i, node in enumerate(layer._inbound_nodes):
            print(f'  Node {i}:')
            print(f'    Type: {type(node).__name__}')
            
            # Try different attributes
            if hasattr(node, 'inbound_layers'):
                print(f'    Has inbound_layers: {[l.name for l in node.inbound_layers]}')
            
            if hasattr(node, 'parent_nodes'):
                print(f'    Has parent_nodes: {len(node.parent_nodes)}')
                for j, pn in enumerate(node.parent_nodes):
                    print(f'      Parent node {j}: {type(pn).__name__}')
                    if hasattr(pn, 'operation'):
                        print(f'        Operation: {pn.operation.name if hasattr(pn.operation, "name") else type(pn.operation).__name__}')
                    if hasattr(pn, 'layer'):
                        print(f'        Has layer: {pn.layer.name}')
                    elif hasattr(pn, 'operation'):
                        print(f'        Operation is the layer!')
            
            # Check keras_inputs
            if hasattr(node, 'keras_inputs'):
                print(f'    Has keras_inputs: {type(node.keras_inputs)}')
                if isinstance(node.keras_inputs, list):
                    for ki in node.keras_inputs:
                        if hasattr(ki, '_keras_history'):
                            hist = ki._keras_history
                            print(f'      Keras history: layer={hist[0].name if hasattr(hist[0], "name") else hist[0]}')
