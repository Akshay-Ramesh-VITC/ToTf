"""
Example demonstration of ToTf Phase 1 QoL & Transfer Learning Utilities:
- seed_everything: Cross-framework deterministic reproducibility
- inspect_tensor: Diagnostic tensor health & statistics inspector
- freeze_layers / unfreeze_layers / freeze_batchnorm: Selective freezing for fine-tuning
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras

import ToTf
from ToTf import (
    seed_everything,
    inspect_tensor,
    freeze_layers,
    unfreeze_layers,
    freeze_batchnorm,
)


def demo_seed_everything():
    print("\n" + "=" * 70)
    print("1. DEMO: seed_everything")
    print("=" * 70)

    # Seed Python, NumPy, PyTorch, and TensorFlow in a single call
    seed = seed_everything(42, deterministic=True, verbose=True)

    th_sample = torch.randn(3).numpy()
    tf_sample = tf.random.normal([3]).numpy()

    print(f"PyTorch sample with seed {seed}: {th_sample}")
    print(f"TensorFlow sample with seed {seed}: {tf_sample}")


def demo_inspect_tensor():
    print("\n" + "=" * 70)
    print("2. DEMO: inspect_tensor")
    print("=" * 70)

    # Inspect a PyTorch activation tensor with some zeros
    act_torch = torch.relu(torch.randn(16, 64, 14, 14))
    print("\n--- Inspecting PyTorch Tensor ---")
    inspect_tensor(act_torch, name="torch_relu_features")

    # Inspect a PyTorch tensor with anomalous NaN / Inf values
    noisy = torch.tensor([1.2, float('nan'), -0.5, float('inf'), 0.0, 3.4])
    print("\n--- Inspecting Tensor with Anomalies (NaN/Inf) ---")
    inspect_tensor(noisy, name="noisy_gradients")

    # Inspect a TensorFlow tensor
    tf_tensor = tf.random.uniform([32, 128], minval=-1.0, maxval=1.0)
    print("\n--- Inspecting TensorFlow Tensor ---")
    inspect_tensor(tf_tensor, name="tf_embeddings")


def demo_pytorch_transfer_learning():
    print("\n" + "=" * 70)
    print("3. DEMO: PyTorch Transfer Learning & Layer Freezing")
    print("=" * 70)

    # Define a simple feature extractor + classifier
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 10),
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initial total trainable parameters: {total_params:,}")

    # Step 1: Freeze all layers first
    frozen_count = freeze_layers(model)
    print(f"Frozen all {frozen_count:,} parameters.")

    # Step 2: Ensure BatchNorm statistics and affine weights are frozen
    bn_frozen = freeze_batchnorm(model, freeze_eval=True)
    print(f"Frozen {bn_frozen} BatchNorm layers (affine params locked, eval mode enforced).")

    # Step 3: Unfreeze only the classification head
    unfrozen = unfreeze_layers(model, last_n=1)
    trainable_final = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Unfrozen {unfrozen:,} parameters in the head. Final trainable: {trainable_final:,}")


def demo_tensorflow_transfer_learning():
    print("\n" + "=" * 70)
    print("4. DEMO: TensorFlow/Keras Transfer Learning & Layer Freezing")
    print("=" * 70)

    model = keras.Sequential([
        keras.layers.Input(shape=(32, 32, 3)),
        keras.layers.Conv2D(16, 3, padding='same', name='conv1'),
        keras.layers.BatchNormalization(name='bn1'),
        keras.layers.ReLU(name='relu1'),
        keras.layers.Conv2D(32, 3, padding='same', name='conv2'),
        keras.layers.BatchNormalization(name='bn2'),
        keras.layers.Flatten(name='flatten'),
        keras.layers.Dense(10, name='head'),
    ])

    print("Initial layer trainability:")
    for l in model.layers:
        print(f"  {l.name}: trainable={l.trainable}")

    # Freeze up to conv2
    print("\nFreezing up to 'conv2':")
    freeze_layers(model, up_to='conv2')
    for l in model.layers:
        print(f"  {l.name}: trainable={l.trainable}")

    # Freeze BatchNorm
    print("\nFreezing BatchNorm layers:")
    freeze_batchnorm(model)
    for l in model.layers:
        print(f"  {l.name}: trainable={l.trainable}")

    # Unfreeze only the head
    print("\nUnfreezing classification head:")
    unfreeze_layers(model, last_n=1)
    for l in model.layers:
        print(f"  {l.name}: trainable={l.trainable}")


if __name__ == "__main__":
    demo_seed_everything()
    demo_inspect_tensor()
    demo_pytorch_transfer_learning()
    demo_tensorflow_transfer_learning()
    print("\n[SUCCESS] Phase 1 demonstrations completed successfully!")
