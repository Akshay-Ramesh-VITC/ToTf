"""
ToTf - A Cross-Library Compatible Library for PyTorch and TensorFlow
Provides additional features for ease of use not directly available in Torch or TF
"""

from .backend import get_backend, _BACKEND

__version__ = "0.1.0"
__author__ = "Your Name"
__all__ = ["get_backend", "TrainingMonitor", "SmartSummary"]

# Import framework-specific modules based on detected backend
if _BACKEND == "torch":
    from .pytorch.trainingmonitor import TrainingMonitor
    from .pytorch.smartsummary import SmartSummary
elif _BACKEND == "tensorflow":
    from .tenf.smartsummary import SmartSummary
    # TrainingMonitor for TensorFlow to be implemented
else:
    raise ImportError("No compatible backend found. Install PyTorch or TensorFlow.")
