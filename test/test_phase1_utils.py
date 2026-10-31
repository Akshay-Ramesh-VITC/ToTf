"""
Unit and integration tests for Phase 1 features:
- seed_everything
- inspect_tensor
- freeze_layers, unfreeze_layers, freeze_batchnorm
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
from ToTf import (
    seed_everything,
    inspect_tensor,
    freeze_layers,
    unfreeze_layers,
    freeze_batchnorm,
)


class TorchToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.head = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.flatten(1)
        return self.head(x)


def create_tf_toy_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(32, 32, 3)),
        keras.layers.Conv2D(8, 3, padding='same', name='conv1'),
        keras.layers.BatchNormalization(name='bn1'),
        keras.layers.ReLU(name='relu1'),
        keras.layers.Conv2D(16, 3, padding='same', name='conv2'),
        keras.layers.BatchNormalization(name='bn2'),
        keras.layers.Flatten(name='flatten'),
        keras.layers.Dense(10, name='head'),
    ])
    return model


# =========================================================================
# 1. seed_everything tests
# =========================================================================

def test_seed_everything_determinism():
    # Run 1
    seed_everything(1234)
    np_val1 = np.random.rand(5)
    th_val1 = torch.randn(5).numpy()
    tf_val1 = tf.random.normal([5]).numpy()

    # Run 2 with same seed
    seed_val = seed_everything(1234, deterministic=True, benchmark=False)
    assert seed_val == 1234
    np_val2 = np.random.rand(5)
    th_val2 = torch.randn(5).numpy()
    tf_val2 = tf.random.normal([5]).numpy()

    assert np.allclose(np_val1, np_val2), "NumPy seed failed to reproduce"
    assert np.allclose(th_val1, th_val2), "PyTorch seed failed to reproduce"
    assert np.allclose(tf_val1, tf_val2), "TensorFlow seed failed to reproduce"


# =========================================================================
# 2. inspect_tensor tests
# =========================================================================

def test_inspect_tensor_pytorch():
    # Normal tensor
    t = torch.tensor([[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]], dtype=torch.float32)
    info = inspect_tensor(t, name="test_th", print_info=True)

    assert info["framework"] == "torch"
    assert info["shape"] == (2, 3)
    assert info["dtype"] == "float32"
    assert info["numel"] == 6
    assert info["zeros_count"] == 2
    assert abs(info["sparsity_pct"] - 33.333) < 0.1
    assert info["min"] == 0.0
    assert info["max"] == 5.0
    assert info["has_nan"] is False
    assert info["has_inf"] is False
    assert "p50" in info["quantiles"]


def test_inspect_tensor_pytorch_nan_inf():
    # Tensor with NaN and Inf
    t = torch.tensor([1.0, float('nan'), float('inf'), -float('inf'), 0.0])
    info = inspect_tensor(t, name="nan_inf_th", print_info=False)

    assert info["has_nan"] is True
    assert info["nan_count"] == 1
    assert info["has_inf"] is True
    assert info["inf_count"] == 2
    assert info["min"] == 0.0
    assert info["max"] == 1.0


def test_inspect_tensor_tensorflow():
    # TF Tensor
    t = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    info = inspect_tensor(t, name="test_tf", print_info=True)

    assert info["framework"] == "tensorflow"
    assert info["shape"] == (2, 2)
    assert info["dtype"] == "float32"
    assert info["numel"] == 4
    assert info["min"] == 1.0
    assert info["max"] == 4.0
    assert info["has_nan"] is False


def test_inspect_tensor_tf_variable():
    v = tf.Variable([0.0, 10.0, 20.0], dtype=tf.float32)
    info = inspect_tensor(v, name="var_tf", print_info=False)

    assert info["framework"] == "tensorflow"
    assert info["numel"] == 3
    assert info["zeros_count"] == 1


def test_inspect_tensor_numpy_and_list():
    arr = np.array([10, 20, 30], dtype=np.int32)
    info = inspect_tensor(arr, name="np_arr", print_info=False)
    assert info["framework"] == "numpy"
    assert info["numel"] == 3
    assert info["dtype"] == "int32"

    py_list = [1.0, 2.0, 3.0]
    info_list = inspect_tensor(py_list, name="py_list", print_info=False)
    assert info_list["numel"] == 3


# =========================================================================
# 3. PyTorch freeze / unfreeze / freeze_batchnorm tests
# =========================================================================

def test_pytorch_freeze_and_unfreeze_all():
    model = TorchToyNet()
    total_params = sum(p.numel() for p in model.parameters())

    # Freeze all
    frozen = freeze_layers(model)
    assert frozen == total_params
    assert all(not p.requires_grad for p in model.parameters())

    # Unfreeze all
    unfrozen = unfreeze_layers(model)
    assert unfrozen == total_params
    assert all(p.requires_grad for p in model.parameters())


def test_pytorch_freeze_up_to_int():
    model = TorchToyNet()
    # Children in order: conv1, bn1, conv2, bn2, head
    # Freeze first 2: conv1 and bn1
    freeze_layers(model, up_to=2)

    assert all(not p.requires_grad for p in model.conv1.parameters())
    assert all(not p.requires_grad for p in model.bn1.parameters())
    assert all(p.requires_grad for p in model.conv2.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_pytorch_freeze_up_to_str():
    model = TorchToyNet()
    freeze_layers(model, up_to='conv2')

    assert all(not p.requires_grad for p in model.conv1.parameters())
    assert all(not p.requires_grad for p in model.bn1.parameters())
    assert all(not p.requires_grad for p in model.conv2.parameters())
    # head should still be trainable
    assert all(p.requires_grad for p in model.head.parameters())


def test_pytorch_unfreeze_last_n():
    model = TorchToyNet()
    freeze_layers(model)  # freeze everything
    assert all(not p.requires_grad for p in model.parameters())

    # Unfreeze only the classification head
    unfreeze_layers(model, last_n=1)
    assert all(p.requires_grad for p in model.head.parameters())
    assert all(not p.requires_grad for p in model.conv1.parameters())


def test_pytorch_freeze_by_types_and_regex():
    model = TorchToyNet()
    # Freeze only Conv2d
    freeze_layers(model, layer_types=(nn.Conv2d,))
    assert all(not p.requires_grad for p in model.conv1.parameters())
    assert all(not p.requires_grad for p in model.conv2.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    assert all(p.requires_grad for p in model.bn1.parameters())

    # Unfreeze with regex
    unfreeze_layers(model, regex=r'conv1')
    assert all(p.requires_grad for p in model.conv1.parameters())
    assert all(not p.requires_grad for p in model.conv2.parameters())


def test_pytorch_freeze_batchnorm():
    model = TorchToyNet()
    count = freeze_batchnorm(model, freeze_eval=True)
    assert count == 2  # bn1 and bn2

    # Affine parameters frozen
    assert model.bn1.weight.requires_grad is False
    assert model.bn1.bias.requires_grad is False
    assert model.bn2.weight.requires_grad is False
    assert model.bn2.bias.requires_grad is False

    # Calling model.train() should keep BN in eval mode because of pre-hook
    model.train()
    dummy_input = torch.randn(2, 3, 4, 4)
    model(dummy_input)
    assert model.bn1.training is False
    assert model.bn2.training is False
    assert model.conv1.training is True


# =========================================================================
# 4. TensorFlow freeze / unfreeze / freeze_batchnorm tests
# =========================================================================

def test_tf_freeze_and_unfreeze_all():
    model = create_tf_toy_model()

    # Freeze all layers
    frozen = freeze_layers(model)
    assert frozen == len(model.layers)
    assert all(not l.trainable for l in model.layers)

    # Unfreeze all layers
    unfrozen = unfreeze_layers(model)
    assert unfrozen == len(model.layers)
    assert all(l.trainable for l in model.layers)


def test_tf_freeze_up_to_int():
    model = create_tf_toy_model()
    # Freeze first 3 layers (conv1, bn1, relu1)
    freeze_layers(model, up_to=3)
    assert model.layers[0].trainable is False
    assert model.layers[1].trainable is False
    assert model.layers[2].trainable is False
    assert model.layers[3].trainable is True
    assert model.get_layer('head').trainable is True


def test_tf_freeze_up_to_str():
    model = create_tf_toy_model()
    freeze_layers(model, up_to='conv2')

    assert model.get_layer('conv1').trainable is False
    assert model.get_layer('bn1').trainable is False
    assert model.get_layer('conv2').trainable is False
    assert model.get_layer('head').trainable is True


def test_tf_unfreeze_last_n():
    model = create_tf_toy_model()
    freeze_layers(model)  # freeze all
    assert all(not l.trainable for l in model.layers)

    unfreeze_layers(model, last_n=1)
    assert model.get_layer('head').trainable is True
    assert model.get_layer('conv1').trainable is False


def test_tf_freeze_by_types_and_regex():
    model = create_tf_toy_model()
    freeze_layers(model, layer_types=(keras.layers.Conv2D,))
    assert model.get_layer('conv1').trainable is False
    assert model.get_layer('conv2').trainable is False
    assert model.get_layer('head').trainable is True

    unfreeze_layers(model, regex=r'conv1')
    assert model.get_layer('conv1').trainable is True
    assert model.get_layer('conv2').trainable is False


def test_tf_freeze_batchnorm():
    model = create_tf_toy_model()
    count = freeze_batchnorm(model)
    assert count == 2  # bn1 and bn2
    assert model.get_layer('bn1').trainable is False
    assert model.get_layer('bn2').trainable is False
    assert model.get_layer('conv1').trainable is True
    assert model.get_layer('head').trainable is True


# =========================================================================
# 5. Dynamic Dispatch via ToTf top-level
# =========================================================================

def test_totf_dispatcher_integration():
    # 1. seed_everything from top-level
    s = ToTf.seed_everything(999)
    assert s == 999

    # 2. inspect_tensor with torch
    t_th = torch.randn(4, 4)
    info_th = ToTf.inspect_tensor(t_th, name="disp_th", print_info=False)
    assert info_th["framework"] == "torch"

    # 3. inspect_tensor with tf
    t_tf = tf.ones([3, 3])
    info_tf = ToTf.inspect_tensor(t_tf, name="disp_tf", print_info=False)
    assert info_tf["framework"] == "tensorflow"

    # 4. freeze_layers on PyTorch model
    m_th = TorchToyNet()
    assert all(p.requires_grad for p in m_th.parameters())
    ToTf.freeze_layers(m_th)
    assert all(not p.requires_grad for p in m_th.parameters())
    ToTf.unfreeze_layers(m_th)
    assert all(p.requires_grad for p in m_th.parameters())

    # 5. freeze_layers on Keras model
    m_tf = create_tf_toy_model()
    assert all(l.trainable for l in m_tf.layers)
    ToTf.freeze_layers(m_tf)
    assert all(not l.trainable for l in m_tf.layers)
    ToTf.unfreeze_layers(m_tf)
    assert all(l.trainable for l in m_tf.layers)
