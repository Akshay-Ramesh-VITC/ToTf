"""
Unit and integration tests for Phase 3:
- benchmark_model (performance, parameters, FLOPs, tables)
- transfer_weights (PyTorch <-> TensorFlow in-memory parameter transfer)
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras

import ToTf
from ToTf import benchmark_model, transfer_weights


# =========================================================================
# 1. benchmark_model Tests
# =========================================================================

def test_benchmark_model_pytorch():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Linear(16, 10),
    )

    res = benchmark_model(
        model,
        input_size=(3, 16, 16),
        warmup=3,
        repetition=10,
        batch_size=2,
        print_table=False,
    )

    assert res["total_params"] > 0
    assert res["trainable_params"] == res["total_params"]
    assert res["mean_latency_ms"] > 0.0
    assert res["median_latency_ms"] > 0.0
    assert res["fps"] > 0.0
    assert "markdown_table" in res
    assert "latex_table" in res
    assert "| Model |" in res["markdown_table"]
    assert r"\begin{table}" in res["latex_table"]


def test_benchmark_model_tensorflow():
    model = keras.Sequential([
        keras.layers.Input(shape=(16, 16, 3)),
        keras.layers.Conv2D(16, 3, padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(10),
    ])

    res = benchmark_model(
        model,
        input_size=(16, 16, 3),
        warmup=3,
        repetition=10,
        batch_size=1,
        print_table=False,
    )

    assert res["total_params"] > 0
    assert res["fps"] > 0.0
    assert "markdown_table" in res


# =========================================================================
# 2. transfer_weights Tests (Torch <-> TF)
# =========================================================================

def test_transfer_weights_pytorch_to_tensorflow():
    # Build PyTorch model
    torch_model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=True),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 5, bias=True),
    )

    # Build equivalent Keras model
    keras_model = keras.Sequential([
        keras.layers.Input(shape=(4, 4, 3)),
        keras.layers.Conv2D(8, kernel_size=3, padding='same', name="conv1"),
        keras.layers.BatchNormalization(name="bn1"),
        keras.layers.ReLU(name="relu1"),
        keras.layers.Flatten(name="flatten"),
        keras.layers.Dense(5, name="dense1"),
    ])

    # Transfer from Torch to Keras
    res = transfer_weights(torch_model, keras_model, verbose=False)

    assert res["transferred_parameters"] > 0
    assert res["transferred_layers"] >= 3  # Conv, BN, Linear/Dense

    # Verify weight transposition for Conv2D:
    # PyTorch: (8, 3, 3, 3) -> TF: (3, 3, 3, 8)
    torch_conv_w = torch_model[0].weight.detach().cpu().numpy()
    keras_conv_w = keras_model.get_layer("conv1").get_weights()[0]
    expected_tf_w = np.transpose(torch_conv_w, (2, 3, 1, 0))
    assert np.allclose(keras_conv_w, expected_tf_w, atol=1e-5)

    # Verify weight transposition for Dense:
    # PyTorch Linear weight: (5, 128) -> TF Dense kernel: (128, 5)
    torch_linear_w = torch_model[4].weight.detach().cpu().numpy()
    keras_dense_w = keras_model.get_layer("dense1").get_weights()[0]
    expected_tf_dense_w = np.transpose(torch_linear_w, (1, 0))
    assert np.allclose(keras_dense_w, expected_tf_dense_w, atol=1e-5)


def test_transfer_weights_tensorflow_to_pytorch():
    # Keras model with known weights
    keras_model = keras.Sequential([
        keras.layers.Input(shape=(4, 4, 3)),
        keras.layers.Conv2D(8, 3, padding='same', name="c1"),
        keras.layers.Flatten(),
        keras.layers.Dense(4, name="d1"),
    ])

    # Initialize PyTorch model
    torch_model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 4),
    )

    # Transfer TF -> Torch
    res = transfer_weights(keras_model, torch_model, verbose=False)
    assert res["transferred_parameters"] > 0

    # Verify Conv2D weights transferred back to PyTorch format (Out, In, H, W)
    tf_conv_w = keras_model.get_layer("c1").get_weights()[0]
    expected_torch_conv_w = np.transpose(tf_conv_w, (3, 2, 0, 1))
    assert np.allclose(torch_model[0].weight.detach().cpu().numpy(), expected_torch_conv_w, atol=1e-5)

    # Verify Dense weights transferred back to PyTorch Linear format (Out, In)
    tf_dense_w = keras_model.get_layer("d1").get_weights()[0]
    expected_torch_linear_w = np.transpose(tf_dense_w, (1, 0))
    assert np.allclose(torch_model[2].weight.detach().cpu().numpy(), expected_torch_linear_w, atol=1e-5)
