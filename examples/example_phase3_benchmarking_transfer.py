"""
Example demonstration of ToTf Phase 3 Research Utilities:
- benchmark_model: FLOPs, MACs, latency percentiles, FPS, and paper-ready tables
- transfer_weights: Direct in-memory parameter transfer between PyTorch and TensorFlow
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras

import ToTf
from ToTf import benchmark_model, transfer_weights


def demo_benchmarking():
    print("\n" + "=" * 70)
    print("1. DEMO: Model Profiler & Paper-Ready Benchmark")
    print("=" * 70)

    # 1. PyTorch Model Benchmark
    torch_model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10),
    )

    print("Benchmarking PyTorch Model:")
    report = benchmark_model(
        torch_model,
        input_size=(3, 32, 32),
        warmup=5,
        repetition=25,
        batch_size=8,
        print_table=True,
    )

    print("\n--- Markdown Table (Ready for GitHub/Notion) ---")
    print(report["markdown_table"])

    print("\n--- LaTeX Table (Ready for Overleaf/Paper) ---")
    print(report["latex_table"])


def demo_weight_transfer():
    print("\n" + "=" * 70)
    print("2. DEMO: In-Memory Weight Transfer (PyTorch <-> TensorFlow)")
    print("=" * 70)

    # Step 1: Create a PyTorch model and initialize weights
    torch_model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 8 * 8, 4),
    )

    # Step 2: Create equivalent Keras architecture
    keras_model = keras.Sequential([
        keras.layers.Input(shape=(8, 8, 3)),
        keras.layers.Conv2D(16, kernel_size=3, padding='same', name="conv1"),
        keras.layers.BatchNormalization(name="bn1"),
        keras.layers.ReLU(name="relu1"),
        keras.layers.Flatten(name="flatten"),
        keras.layers.Dense(4, name="head"),
    ])

    print("Initial Keras weights before transfer:")
    initial_w_norm = sum(torch.as_tensor(w.numpy()).norm().item() for w in keras_model.weights)
    print(f"  Keras weights L2 norm sum: {initial_w_norm:.4f}")

    # Step 3: Transfer weights PyTorch -> TensorFlow
    print("\nExecuting transfer_weights(torch_model, keras_model)...")
    res = transfer_weights(torch_model, keras_model, verbose=True)

    print(f"\nSuccessfully transferred {res['transferred_parameters']:,} parameters across {res['transferred_layers']} layers.")

    # Step 4: Transfer back TensorFlow -> PyTorch
    print("\nExecuting transfer_weights(keras_model, torch_model)...")
    res_back = transfer_weights(keras_model, torch_model, verbose=True)
    print(f"Transferred back: {res_back['transferred_parameters']:,} parameters.")


if __name__ == "__main__":
    demo_benchmarking()
    demo_weight_transfer()
    print("\n[SUCCESS] Phase 3 demonstrations completed successfully!")
