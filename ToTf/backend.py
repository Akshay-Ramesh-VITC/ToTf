import importlib.util
from typing import Optional

_BACKEND = None

if importlib.util.find_spec("torch"):
    _BACKEND = "torch"
elif importlib.util.find_spec("tensorflow"):
    _BACKEND = "tensorflow"
else:
    raise ImportError("Neither PyTorch nor TensorFlow is installed. Please install one of them.")


def get_backend() -> str:
    """Returns the detected backend: 'torch' or 'tensorflow'"""
    return _BACKEND


def is_mps_available() -> bool:
    """Returns True if Apple Silicon Metal Performance Shaders (MPS) is available in PyTorch."""
    try:
        import torch
        if hasattr(torch.backends, "mps"):
            is_built = getattr(torch.backends.mps, "is_built", lambda: True)()
            return is_built and torch.backends.mps.is_available()
        return False
    except Exception:
        return False


def is_cuda_available() -> bool:
    """Returns True if CUDA GPU acceleration is available in PyTorch or TensorFlow."""
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    try:
        import tensorflow as tf
        if len(tf.config.list_physical_devices("GPU")) > 0:
            return True
    except Exception:
        pass
    return False


def get_device(preferred: Optional[str] = None) -> str:
    """
    Returns the optimal compute device string ('cuda', 'mps', or 'cpu').

    Supports Apple Silicon Metal (MPS) acceleration out of the box.

    Args:
        preferred: Optional preferred device string (e.g. 'auto', 'cuda', 'mps', 'cpu').
                   If None or 'auto', automatically detects the best accelerator:
                   CUDA -> Apple Metal (MPS) -> CPU.

    Returns:
        Device string ('cuda', 'mps', or 'cpu' / preferred device).
    """
    if preferred is not None and preferred != "auto":
        return str(preferred)
    if is_cuda_available():
        return "cuda"
    if is_mps_available():
        return "mps"
    return "cpu"
