"""Quick visualization test to see if connections are shown"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tenf import ModelView

# Create a simple parallel branch model
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

# Create view
view = ModelView(model, input_shape=(32,))

print(f"Number of layers: {len(view.layer_info)}")
print(f"Number of connections: {len(view.connections)}")
print("\nConnections:")
for src, dst in view.connections:
    src_name = view.layer_info[src]['name']
    dst_name = view.layer_info[dst]['name']
    print(f"  {src_name} -> {dst_name}")

# Render
output_path = 'test_connections.png' 
result = view.render(output_path, show_layer_names=True, dpi=300)
print(f"\nRendered to: {result}")
