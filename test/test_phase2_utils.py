"""
Unit and integration tests for Phase 2:
- loss_focal and FocalLoss (PyTorch & TensorFlow)
- loss_dice, DiceLoss, dice_score, iou_score (PyTorch & TensorFlow)
- AnomalyDetector and check_anomalies (PyTorch & TensorFlow)
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
    loss_focal,
    FocalLoss,
    loss_dice,
    DiceLoss,
    dice_score,
    iou_score,
    AnomalyDetector,
    check_anomalies,
)


# =========================================================================
# 1. Focal Loss Tests (PyTorch & TensorFlow)
# =========================================================================

def test_focal_loss_pytorch():
    # Binary case
    y_true = torch.tensor([1.0, 0.0, 1.0, 1.0])
    y_pred_probs = torch.tensor([0.9, 0.1, 0.8, 0.4])
    loss = loss_focal(y_true, y_pred_probs, alpha=0.25, gamma=2.0)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)

    # From logits
    logits = torch.tensor([2.0, -2.0, 1.5, -0.5], requires_grad=True)
    loss_logits = loss_focal(y_true, logits, from_logits=True)
    loss_logits.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))

    # Multiclass case (1D class indices vs (N, C) logits)
    mc_targets = torch.tensor([0, 2])
    mc_logits = torch.randn(2, 4)
    mc_loss = loss_focal(mc_targets, mc_logits, from_logits=True)
    assert mc_loss.item() > 0.0

    # FocalLoss module
    criterion = FocalLoss(alpha=0.5, gamma=1.0, from_logits=True)
    mod_loss = criterion(logits, y_true)
    assert mod_loss.item() > 0.0


def test_focal_loss_tensorflow():
    # Binary case
    y_true = tf.constant([1.0, 0.0, 1.0, 1.0])
    y_pred_probs = tf.constant([0.9, 0.1, 0.8, 0.4])
    loss = loss_focal(y_true, y_pred_probs, alpha=0.25, gamma=2.0)
    assert float(loss.numpy()) > 0.0

    # From logits
    logits = tf.constant([2.0, -2.0, 1.5, -0.5])
    loss_logits = loss_focal(y_true, logits, from_logits=True)
    assert float(loss_logits.numpy()) > 0.0

    # Multiclass case
    mc_targets = tf.constant([0, 2])
    mc_logits = tf.random.normal([2, 4])
    mc_loss = loss_focal(mc_targets, mc_logits, from_logits=True)
    assert float(mc_loss.numpy()) > 0.0

    # Keras FocalLoss class
    crit = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True, backend_name="tensorflow")
    res = crit(y_true, logits)
    assert float(res.numpy()) > 0.0
    cfg = crit.get_config()
    assert cfg["alpha"] == 0.25


# =========================================================================
# 2. Dice Loss & IoU Tests (PyTorch & TensorFlow)
# =========================================================================

def test_dice_and_iou_pytorch():
    # Perfect match: prediction matches ground truth -> Dice loss ~ 0, Dice ~ 1, IoU ~ 1
    y_true = torch.ones(2, 1, 8, 8)
    y_pred = torch.ones(2, 1, 8, 8)

    loss = loss_dice(y_true, y_pred)
    assert loss.item() < 0.01

    d_score = dice_score(y_true, y_pred)
    assert abs(d_score.item() - 1.0) < 0.01

    j_score = iou_score(y_true, y_pred)
    assert abs(j_score.item() - 1.0) < 0.01

    # Completely disjoint masks: Dice ~ 0, IoU ~ 0, Loss ~ 1
    y_disjoint = torch.zeros(2, 1, 8, 8)
    loss_dis = loss_dice(y_true, y_disjoint)
    assert loss_dis.item() > 0.99

    # Partial overlap: mathematical property IoU <= Dice
    y_half = torch.zeros(2, 1, 8, 8)
    y_half[:, :, :4, :] = 1.0
    ds = dice_score(y_true, y_half)
    js = iou_score(y_true, y_half)
    assert js.item() <= ds.item()

    # DiceLoss module
    dice_mod = DiceLoss(from_logits=True)
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    l_mod = dice_mod(logits, y_true)
    l_mod.backward()
    assert logits.grad is not None


def test_dice_and_iou_tensorflow():
    y_true = tf.ones([2, 8, 8, 1])
    y_pred = tf.ones([2, 8, 8, 1])

    loss = loss_dice(y_true, y_pred)
    assert float(loss.numpy()) < 0.01

    d_score = dice_score(y_true, y_pred)
    assert abs(float(d_score.numpy()) - 1.0) < 0.01

    j_score = iou_score(y_true, y_pred)
    assert abs(float(j_score.numpy()) - 1.0) < 0.01

    # Keras DiceLoss class
    crit = DiceLoss(from_logits=False, backend_name="tensorflow")
    res = crit(y_true, y_pred)
    assert float(res.numpy()) < 0.01
    cfg = crit.get_config()
    assert "smooth" in cfg


# =========================================================================
# 3. AnomalyDetector Tests (PyTorch & TensorFlow)
# =========================================================================

class TorchBuggyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.fc1(x)
        # Deliberately introduce NaN
        x = x / 0.0  # Infs
        x = x - float('inf')  # NaNs
        return self.fc2(x)


def test_anomaly_detector_pytorch():
    # 1. Clean model
    clean_model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    clean_x = torch.randn(2, 4)
    with AnomalyDetector(clean_model) as detector:
        out = clean_model(clean_x)
        loss = out.sum()
        loss.backward()
    assert not detector.has_anomalies

    # 2. Model with NaNs
    buggy_model = TorchBuggyNet()
    with AnomalyDetector(buggy_model) as detector:
        _ = buggy_model(clean_x)
    assert detector.has_anomalies
    assert len(detector.anomalies) > 0
    # First detected anomaly should point to forward_output or fc2 input
    first = detector.anomalies[0]
    assert first["nan_count"] > 0 or first["inf_count"] > 0

    # 3. check_anomalies helper
    anoms = check_anomalies(buggy_model, clean_x)
    assert len(anoms) > 0

    # 4. raise_error flag
    with pytest.raises(FloatingPointError):
        with AnomalyDetector(buggy_model, raise_error=True):
            _ = buggy_model(clean_x)


def test_anomaly_detector_tensorflow():
    model = keras.Sequential([
        keras.layers.Input(shape=(4,)),
        keras.layers.Dense(4, name="dense1"),
        keras.layers.Dense(2, name="dense2"),
    ])

    clean_x = tf.random.normal([2, 4])
    with AnomalyDetector(model) as detector:
        _ = model(clean_x)
    assert not detector.has_anomalies

    # Input with NaN
    nan_x = tf.constant([[1.0, float('nan'), 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    with AnomalyDetector(model) as detector:
        _ = model(nan_x)
    assert detector.has_anomalies
    assert detector.anomalies[0]["nan_count"] > 0

    # check_anomalies
    anoms = check_anomalies(model, nan_x)
    assert len(anoms) > 0
