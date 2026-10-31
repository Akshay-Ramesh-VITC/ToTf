"""
ToTf - A Cross-Library Compatible Library for PyTorch and TensorFlow
Provides additional features for ease of use not directly available in Torch or TF
"""

from .backend import get_backend, _BACKEND
from .backend_manager import SmartSummary, TrainingMonitor, ModelView, draw_graph, get_component
from .backend_manager import toONNX, fromONNX
from .backend_manager import (
    seed_everything,
    inspect_tensor,
    freeze_layers,
    unfreeze_layers,
    freeze_batchnorm,
    loss_focal,
    FocalLoss,
    loss_dice,
    DiceLoss,
    dice_score,
    iou_score,
    AnomalyDetector,
    check_anomalies,
    benchmark_model,
    transfer_weights,
)

__version__ = "0.2.5"
__author__ = "Akshay"
__all__ = [
    "get_backend",
    "TrainingMonitor",
    "SmartSummary",
    "ModelView",
    "draw_graph",
    "get_component",
    "toONNX",
    "fromONNX",
    "seed_everything",
    "inspect_tensor",
    "freeze_layers",
    "unfreeze_layers",
    "freeze_batchnorm",
    "loss_focal",
    "FocalLoss",
    "loss_dice",
    "DiceLoss",
    "dice_score",
    "iou_score",
    "AnomalyDetector",
    "check_anomalies",
    "benchmark_model",
    "transfer_weights",
]

# Use backend_manager factories for dynamic dispatch to framework implementations
