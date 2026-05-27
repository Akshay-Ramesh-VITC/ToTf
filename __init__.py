"""
ToTf - A Cross-Library Compatible Library for PyTorch and TensorFlow
Provides additional features for ease of use not directly available in Torch or TF
"""

from .backend import get_backend, _BACKEND
from .backend_manager import SmartSummary, TrainingMonitor, ModelView, draw_graph, get_component

__version__ = "0.1.2"
__author__ = "Akshay"
__all__ = ["get_backend", "TrainingMonitor", "SmartSummary", "ModelView", "draw_graph", "get_component"]

# Use backend_manager factories for dynamic dispatch to framework implementations
