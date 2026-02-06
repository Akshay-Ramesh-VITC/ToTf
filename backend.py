import importlib.util

_BACKEND = None

if importlib.util.find_spec("torch"):
    _BACKEND = "torch"
elif importlib.util.find_spec("tensorflow"):
    _BACKEND = "tensorflow"
else:
    raise ImportError("Neither PyTorch nor TensorFlow is installed. Please install one of them.")

def get_backend():
    """Returns the detected backend: 'torch' or 'tensorflow'"""
    return _BACKEND