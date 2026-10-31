"""
Example demonstration of ToTf Phase 2 Utilities:
- loss_focal / FocalLoss: Imbalanced classification loss (Lin et al.)
- loss_dice / DiceLoss / dice_score / iou_score: Segmentation losses and metrics
- AnomalyDetector / check_anomalies: Instant first-NaN / Inf anomaly detection
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import tensorflow as tf

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


def demo_focal_loss():
    print("\n" + "=" * 70)
    print("1. DEMO: Focal Loss (Class Imbalance)")
    print("=" * 70)

    # Simulated rare disease detection: 95% negative, 5% positive
    y_true = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    # Model predictions (probabilities)
    y_pred = torch.tensor([0.05, 0.02, 0.10, 0.70, 0.01, 0.04, 0.02, 0.03])

    loss = loss_focal(y_true, y_pred, alpha=0.25, gamma=2.0)
    print(f"Computed Focal Loss (PyTorch): {loss.item():.6f}")

    # TensorFlow execution
    tf_true = tf.constant([0.0, 0.0, 1.0, 0.0])
    tf_pred = tf.constant([0.05, 0.02, 0.85, 0.01])
    tf_loss = loss_focal(tf_true, tf_pred, alpha=0.25, gamma=2.0)
    print(f"Computed Focal Loss (TensorFlow): {float(tf_loss.numpy()):.6f}")


def demo_dice_and_iou():
    print("\n" + "=" * 70)
    print("2. DEMO: Dice Loss & IoU Segmentation Metrics")
    print("=" * 70)

    # Simulated 2D segmentation masks (Batch=1, Channels=1, 16x16)
    target_mask = torch.zeros(1, 1, 16, 16)
    target_mask[:, :, 4:12, 4:12] = 1.0  # Center lesion box (8x8 = 64 pixels)

    # Predicted mask with slight offset
    pred_probs = torch.zeros(1, 1, 16, 16)
    pred_probs[:, :, 5:13, 5:13] = 0.90  # Shifted by 1 pixel

    loss = loss_dice(target_mask, pred_probs)
    dsc = dice_score(target_mask, pred_probs, threshold=0.5)
    iou = iou_score(target_mask, pred_probs, threshold=0.5)

    print(f"Dice Loss:               {loss.item():.4f} (lower is better, 0=perfect)")
    print(f"Dice Similarity (DSC):   {dsc.item():.4f} (higher is better, 1=perfect)")
    print(f"Intersection over Union: {iou.item():.4f} (Jaccard index <= DSC)")


def demo_anomaly_detector():
    print("\n" + "=" * 70)
    print("3. DEMO: AnomalyDetector (Pinpointing NaN / Inf origins)")
    print("=" * 70)

    # Deep network where layer 2 produces an explosive NaN/Inf
    class ProblematicNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
            )
            self.unstable_layer = nn.Linear(16, 16)
            self.head = nn.Linear(16, 2)

        def forward(self, x):
            x = self.backbone(x)
            # Simulate sudden division by zero / overflow inside custom operator
            x = x / 0.0
            x = self.unstable_layer(x)
            return self.head(x)

    model = ProblematicNetwork()
    x = torch.randn(4, 8)

    print("Running forward pass with AnomalyDetector attached...")
    with AnomalyDetector(model) as detector:
        _ = model(x)

    # Print formatted diagnostic report
    detector.print_report()


if __name__ == "__main__":
    demo_focal_loss()
    demo_dice_and_iou()
    demo_anomaly_detector()
    print("\n[SUCCESS] Phase 2 demonstrations completed successfully!")
