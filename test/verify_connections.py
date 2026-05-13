"""Verification script to confirm connections are extracted correctly"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tenf import ModelView

print("=" * 70)
print("Connection Extraction Verification")
print("=" * 70)

# Test 1: Simple parallel branches
print("\n1. Parallel Branches Model")
inputs = layers.Input(shape=(32,))
b1 = layers.Dense(64, name='b1')(inputs)
b2 = layers.Dense(64, name='b2')(inputs)
merged = layers.Concatenate(name='concat')([b1, b2])
output = layers.Dense(10, name='out')(merged)
model = keras.Model(inputs=inputs, outputs=output)

view = ModelView(model, input_shape=(32,))
print(f"   Layers: {len(view.layer_info)}")
print(f"   Connections extracted: {len(view.connections)}")
print(f"   Expected: 5 connections (input->b1, input->b2, b1->concat, b2->concat, concat->out)")
assert len(view.connections) == 5, f"Expected 5 connections, got {len(view.connections)}"
print("   ✓ PASS - Parallel branches correctly extracted")

# Test 2: DenseNet-style connections
print("\n2. DenseNet-Style Dense Connections")
inputs = layers.Input(shape=(32,))
x1 = layers.Dense(32, name='x1')(inputs)
c1 = layers.Concatenate(name='c1')([inputs, x1])
x2 = layers.Dense(32, name='x2')(c1)
c2 = layers.Concatenate(name='c2')([inputs, x1, x2])
model = keras.Model(inputs=inputs, outputs=c2)

view = ModelView(model, input_shape=(32,))
print(f"   Layers: {len(view.layer_info)}")
print(f"   Connections extracted: {len(view.connections)}")
print(f"   Expected: 7 connections")
# input->x1, x1->c1, input->c1, c1->x2, x2->c2, input->c2, x1->c2
assert len(view.connections) == 7, f"Expected 7 connections, got {len(view.connections)}"
print("   ✓ PASS - Dense connections correctly extracted")

# Test 3: Multi-path model
print("\n3. Complex Multi-Path Model")
inputs = layers.Input(shape=(64,))
p1 = layers.Dense(32, name='p1')(inputs)
p2 = layers.Dense(32, name='p2')(inputs)
m1 = layers.Concatenate(name='m1')([p1, p2])
shared = layers.Dense(64, name='shared')(m1)
p3 = layers.Dense(32, name='p3')(shared)
p4 = layers.Dense(32, name='p4')(shared)
c1 = layers.Concatenate(name='cross1')([p3, p2])
c2 = layers.Concatenate(name='cross2')([p4, p1])
final = layers.Concatenate(name='final')([c1, c2])
model = keras.Model(inputs=inputs, outputs=final)

view = ModelView(model, input_shape=(64,))
print(f"   Layers: {len(view.layer_info)}")
print(f"   Connections extracted: {len(view.connections)}")
for src, dst in view.connections:
    print(f"      {view.layer_info[src]['name']} -> {view.layer_info[dst]['name']}")
print(f"   Expected: ~13-14 connections")
assert len(view.connections) >= 13, f"Expected at least 13 connections, got {len(view.connections)}"
print("   ✓ PASS - Complex multi-path connections correctly extracted")

# Test 4: Multi-input model
print("\n4. Multi-Input Model")
in1 = layers.Input(shape=(32,), name='in1')
in2 = layers.Input(shape=(16,), name='in2')
x1 = layers.Dense(64, name='x1')(in1)
x2 = layers.Dense(64, name='x2')(in2)
merged = layers.Concatenate(name='merge')([x1, x2])
out = layers.Dense(10, name='out')(merged)
model = keras.Model(inputs=[in1, in2], outputs=out)

view = ModelView(model, input_shape=[(32,), (16,)])
print(f"   Layers: {len(view.layer_info)}")
print(f"   Connections extracted: {len(view.connections)}")
print(f"   Expected: 5 connections")
assert len(view.connections) == 5, f"Expected 5 connections, got {len(view.connections)}"
print("   ✓ PASS - Multi-input connections correctly extracted")

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED - Connections are correctly extracted!")
print("=" * 70)
print("\nConnection extraction is working as expected.")
print("All complex branching, merging, and cross-connection patterns")
print("are being properly identified and will be visualized in diagrams.")
