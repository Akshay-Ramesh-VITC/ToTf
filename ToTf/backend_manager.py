"""
Backend Manager / Dispatcher

Provides a unified API that dynamically routes calls to the appropriate
framework implementation (PyTorch or TensorFlow) based on the provided
model/tensor types or the detected default backend.

Usage:
    from ToTf import SmartSummary
    ss = SmartSummary(model_or_args...)

This file avoids heavy imports at module import time and only imports
framework modules lazily when needed.
"""
from typing import Any, Optional, Callable, Dict
import importlib
import inspect

from . import backend as _backend_pkg


def _import_pytorch_module(name: str):
    try:
        return importlib.import_module(f"ToTf.pytorch.{name}")
    except Exception:
        return None


def _import_tensorflow_module(name: str):
    try:
        return importlib.import_module(f"ToTf.tenf.{name}")
    except Exception:
        return None


def _is_torch_tensor(obj: Any) -> bool:
    try:
        import torch
        return isinstance(obj, torch.Tensor) or isinstance(obj, torch.nn.Module)
    except Exception:
        return False


def _is_tf_tensor(obj: Any) -> bool:
    try:
        import tensorflow as tf
        return isinstance(obj, (tf.Tensor, tf.Variable, tf.keras.Model, tf.keras.layers.Layer))
    except Exception:
        return False


def _choose_backend_from_obj(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if _is_torch_tensor(obj):
        return "torch"
    if _is_tf_tensor(obj):
        return "tensorflow"
    return None


def get_default_backend() -> str:
    return _backend_pkg.get_backend()


def _load_backend_impl(backend_name: str, component: str):
    if backend_name == "torch":
        mod = _import_pytorch_module(component)
        return mod
    elif backend_name == "tensorflow":
        mod = _import_tensorflow_module(component)
        return mod
    return None


def _select_backend_for_args(*args, **kwargs) -> str:
    # Try to pick a backend from any args/kwargs by checking types
    for a in list(args) + list(kwargs.values()):
        b = _choose_backend_from_obj(a)
        if b:
            return b
    # fallback to detected default
    return get_default_backend()


def SmartSummary(*args, **kwargs):
    """Factory that returns a SmartSummary instance from the appropriate backend.

    Selection order:
    - If first positional arg is a model object (torch.nn.Module or tf.keras.Model)
      we pick the backend accordingly.
    - Otherwise we inspect all args/kwargs for recognizable types (tensors or models).
    - Fallback to the globally detected backend.
    """
    backend_name = None
    if args:
        backend_name = _choose_backend_from_obj(args[0])
    if backend_name is None:
        backend_name = _select_backend_for_args(*args, **kwargs)

    impl_mod = _load_backend_impl(backend_name, "smartsummary")
    if impl_mod is None:
        raise ImportError(f"SmartSummary backend '{backend_name}' not available")

    # The implementation exposes SmartSummary class
    SmartSummaryCls = getattr(impl_mod, "SmartSummary", None)
    if SmartSummaryCls is None:
        raise ImportError(f"SmartSummary class not found in backend module {impl_mod}")

    return SmartSummaryCls(*args, **kwargs)


def TrainingMonitor(*args, **kwargs):
    backend_name = _select_backend_for_args(*args, **kwargs)
    impl_mod = _load_backend_impl(backend_name, "trainingmonitor")
    if impl_mod is None:
        raise ImportError(f"TrainingMonitor backend '{backend_name}' not available")
    TM = getattr(impl_mod, "TrainingMonitor", None)
    if TM is None:
        raise ImportError(f"TrainingMonitor class not found in backend module {impl_mod}")
    return TM(*args, **kwargs)


def ModelView(*args, **kwargs):
    backend_name = _select_backend_for_args(*args, **kwargs)
    impl_mod = _load_backend_impl(backend_name, "modelview")
    if impl_mod is None:
        raise ImportError(f"ModelView backend '{backend_name}' not available")
    MV = getattr(impl_mod, "ModelView", None)
    if MV is None:
        raise ImportError(f"ModelView class not found in backend module {impl_mod}")
    return MV(*args, **kwargs)


def draw_graph(*args, **kwargs):
    """Dispatch to the backend-specific convenience draw_graph function.

    This will load the backend's `modelview` component and call its
    `draw_graph` function. Selection of backend follows the same rules as
    other factories (inspect args for model/tensor types, then fallback to
    detected default).
    """
    backend_name = _select_backend_for_args(*args, **kwargs)
    impl_mod = _load_backend_impl(backend_name, "modelview")
    if impl_mod is None:
        raise ImportError(f"draw_graph backend '{backend_name}' not available")
    dg = getattr(impl_mod, "draw_graph", None)
    if dg is None:
        raise ImportError(f"draw_graph function not found in backend module {impl_mod}")
    return dg(*args, **kwargs)


# Allow users to register custom backend modules (advanced)
_CUSTOM: Dict[str, Dict[str, Any]] = {}


def register_backend(name: str, components: Dict[str, Any]):
    _CUSTOM[name] = components


def get_backend_for_obj(obj: Any) -> str:
    b = _choose_backend_from_obj(obj)
    if b:
        return b
    return get_default_backend()


def get_component(component: str, backend_name: Optional[str] = None):
    """Return the backend-specific component module.

    Args:
        component: name of the component module (e.g. 'utils', 'smartsummary')
        backend_name: 'torch' or 'tensorflow'. If None, uses detected default.

    Returns:
        Module object or raises ImportError if unavailable.
    """
    if backend_name is None:
        backend_name = get_default_backend()

    mod = _load_backend_impl(backend_name, component)
    if mod is None:
        raise ImportError(f"Component '{component}' not available for backend '{backend_name}'")
    return mod


def toONNX(*args, backend_name: Optional[str] = None, **kwargs):
    """
    Dispatch to backend-specific toONNX implementation.

    The function will try to infer backend from the first positional argument
    (model) if available, otherwise use `backend_name` or the default detected backend.
    """
    # Try to infer backend from args
    b = None
    if args:
        b = _choose_backend_from_obj(args[0])
    if b is None:
        b = backend_name or get_default_backend()

    impl_mod = _load_backend_impl(b, "utils")
    if impl_mod is None:
        raise ImportError(f"toONNX utils not available for backend '{b}'")

    fn = getattr(impl_mod, "toONNX", None)
    if fn is None:
        raise ImportError(f"toONNX not implemented in backend utils for '{b}'")

    return fn(*args, **kwargs)


def fromONNX(onnx_path: str, *args, backend_name: Optional[str] = None, **kwargs):
    """
    Dispatch to a backend-agnostic ONNX inference wrapper. This currently uses onnxruntime
    under the hood from each backend utils module for consistency.
    """
    # backend_name is less important here; both backends use onnxruntime wrapper
    b = backend_name or get_default_backend()
    impl_mod = _load_backend_impl(b, "utils")
    if impl_mod is None:
        raise ImportError(f"fromONNX utils not available for backend '{b}'")

    fn = getattr(impl_mod, "fromONNX", None)
    if fn is None:
        raise ImportError(f"fromONNX not implemented in backend utils for '{b}'")

    return fn(onnx_path, *args, **kwargs)


def seed_everything(
    seed: int = 42,
    deterministic: bool = False,
    benchmark: bool = False,
    verbose: bool = False,
    backend_name: Optional[str] = None,
) -> int:
    """
    Sets seeds across Python, NumPy, PyTorch, and TensorFlow for reproducibility.
    """
    b = backend_name or get_default_backend()
    impl_mod = _load_backend_impl(b, "utils")
    if impl_mod is not None and hasattr(impl_mod, "seed_everything"):
        return impl_mod.seed_everything(seed=seed, deterministic=deterministic, benchmark=benchmark, verbose=verbose)
    # Fallback to direct seeding
    import os
    import random
    import numpy as np
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    return seed


def inspect_tensor(
    tensor: Any,
    name: Optional[str] = None,
    print_info: bool = True,
    quantiles: bool = True,
    backend_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Inspects a tensor (Torch, TF, or NumPy) and returns detailed statistics and health metrics.
    """
    b = _choose_backend_from_obj(tensor) or backend_name or get_default_backend()
    impl_mod = _load_backend_impl(b, "utils")
    if impl_mod is None or not hasattr(impl_mod, "inspect_tensor"):
        raise ImportError(f"inspect_tensor utils not available for backend '{b}'")
    return impl_mod.inspect_tensor(tensor, name=name, print_info=print_info, quantiles=quantiles)


def freeze_layers(model: Any, *args, backend_name: Optional[str] = None, **kwargs) -> int:
    """
    Freeze layers or parameters in a PyTorch or TensorFlow model for transfer learning.
    """
    b = _choose_backend_from_obj(model) or backend_name or get_default_backend()
    impl_mod = _load_backend_impl(b, "utils")
    if impl_mod is None or not hasattr(impl_mod, "freeze_layers"):
        raise ImportError(f"freeze_layers utils not available for backend '{b}'")
    return impl_mod.freeze_layers(model, *args, **kwargs)


def unfreeze_layers(model: Any, *args, backend_name: Optional[str] = None, **kwargs) -> int:
    """
    Unfreeze layers or parameters in a PyTorch or TensorFlow model.
    """
    b = _choose_backend_from_obj(model) or backend_name or get_default_backend()
    impl_mod = _load_backend_impl(b, "utils")
    if impl_mod is None or not hasattr(impl_mod, "unfreeze_layers"):
        raise ImportError(f"unfreeze_layers utils not available for backend '{b}'")
    return impl_mod.unfreeze_layers(model, *args, **kwargs)


def freeze_batchnorm(model: Any, *args, backend_name: Optional[str] = None, **kwargs) -> int:
    """
    Freeze BatchNorm layers in a PyTorch or TensorFlow model.
    """
    b = _choose_backend_from_obj(model) or backend_name or get_default_backend()
    impl_mod = _load_backend_impl(b, "utils")
    if impl_mod is None or not hasattr(impl_mod, "freeze_batchnorm"):
        raise ImportError(f"freeze_batchnorm utils not available for backend '{b}'")
    return impl_mod.freeze_batchnorm(model, *args, **kwargs)


def loss_focal(y_true: Any, y_pred: Any, *args, backend_name: Optional[str] = None, **kwargs):
    """
    Dispatch to framework-specific Focal Loss.
    """
    b = _choose_backend_from_obj(y_pred) or _choose_backend_from_obj(y_true) or backend_name or get_default_backend()
    impl_mod = _load_backend_impl(b, "utils")
    if impl_mod is None or not hasattr(impl_mod, "loss_focal"):
        raise ImportError(f"loss_focal not available for backend '{b}'")
    return impl_mod.loss_focal(y_true, y_pred, *args, **kwargs)


def FocalLoss(*args, backend_name: Optional[str] = None, **kwargs):
    """
    Factory creating a FocalLoss module (PyTorch nn.Module or Keras Loss).
    """
    b = backend_name or get_default_backend()
    impl_mod = _load_backend_impl(b, "utils")
    if impl_mod is None or not hasattr(impl_mod, "FocalLoss"):
        raise ImportError(f"FocalLoss not available for backend '{b}'")
    return impl_mod.FocalLoss(*args, **kwargs)


def loss_dice(y_true: Any, y_pred: Any, *args, backend_name: Optional[str] = None, **kwargs):
    """
    Dispatch to framework-specific Dice Loss.
    """
    b = _choose_backend_from_obj(y_pred) or _choose_backend_from_obj(y_true) or backend_name or get_default_backend()
    impl_mod = _load_backend_impl(b, "utils")
    if impl_mod is None or not hasattr(impl_mod, "loss_dice"):
        raise ImportError(f"loss_dice not available for backend '{b}'")
    return impl_mod.loss_dice(y_true, y_pred, *args, **kwargs)


def DiceLoss(*args, backend_name: Optional[str] = None, **kwargs):
    """
    Factory creating a DiceLoss module (PyTorch nn.Module or Keras Loss).
    """
    b = backend_name or get_default_backend()
    impl_mod = _load_backend_impl(b, "utils")
    if impl_mod is None or not hasattr(impl_mod, "DiceLoss"):
        raise ImportError(f"DiceLoss not available for backend '{b}'")
    return impl_mod.DiceLoss(*args, **kwargs)


def dice_score(y_true: Any, y_pred: Any, *args, backend_name: Optional[str] = None, **kwargs):
    """
    Dispatch to framework-specific Dice similarity score.
    """
    b = _choose_backend_from_obj(y_pred) or _choose_backend_from_obj(y_true) or backend_name or get_default_backend()
    impl_mod = _load_backend_impl(b, "utils")
    if impl_mod is None or not hasattr(impl_mod, "dice_score"):
        raise ImportError(f"dice_score not available for backend '{b}'")
    return impl_mod.dice_score(y_true, y_pred, *args, **kwargs)


def iou_score(y_true: Any, y_pred: Any, *args, backend_name: Optional[str] = None, **kwargs):
    """
    Dispatch to framework-specific IoU / Jaccard similarity score.
    """
    b = _choose_backend_from_obj(y_pred) or _choose_backend_from_obj(y_true) or backend_name or get_default_backend()
    impl_mod = _load_backend_impl(b, "utils")
    if impl_mod is None or not hasattr(impl_mod, "iou_score"):
        raise ImportError(f"iou_score not available for backend '{b}'")
    return impl_mod.iou_score(y_true, y_pred, *args, **kwargs)


def AnomalyDetector(model: Any, *args, backend_name: Optional[str] = None, **kwargs):
    """
    Create an AnomalyDetector instance attached to the model.
    """
    b = _choose_backend_from_obj(model) or backend_name or get_default_backend()
    impl_mod = _load_backend_impl(b, "utils")
    if impl_mod is None or not hasattr(impl_mod, "AnomalyDetector"):
        raise ImportError(f"AnomalyDetector not available for backend '{b}'")
    return impl_mod.AnomalyDetector(model, *args, **kwargs)


def check_anomalies(model: Any, sample_input: Any, *args, backend_name: Optional[str] = None, **kwargs):
    """
    Run forward pass and return any NaN/Inf anomalies detected.
    """
    b = _choose_backend_from_obj(model) or backend_name or get_default_backend()
    impl_mod = _load_backend_impl(b, "utils")
    if impl_mod is None or not hasattr(impl_mod, "check_anomalies"):
        raise ImportError(f"check_anomalies not available for backend '{b}'")
    return impl_mod.check_anomalies(model, sample_input, *args, **kwargs)


def benchmark_model(model: Any, *args, backend_name: Optional[str] = None, **kwargs):
    """
    Run full performance and complexity benchmark on a PyTorch or TensorFlow model.
    """
    b = _choose_backend_from_obj(model) or backend_name or get_default_backend()
    impl_mod = _load_backend_impl(b, "utils")
    if impl_mod is None or not hasattr(impl_mod, "benchmark_model"):
        raise ImportError(f"benchmark_model not available for backend '{b}'")
    return impl_mod.benchmark_model(model, *args, **kwargs)


def transfer_weights(source_model: Any, target_model: Any, *args, backend_name: Optional[str] = None, **kwargs):
    """
    Direct in-memory weight transfer between PyTorch and TensorFlow models.
    """
    b = _choose_backend_from_obj(source_model) or _choose_backend_from_obj(target_model) or backend_name or get_default_backend()
    impl_mod = _load_backend_impl(b, "utils")
    if impl_mod is None or not hasattr(impl_mod, "transfer_weights"):
        raise ImportError(f"transfer_weights not available for backend '{b}'")
    return impl_mod.transfer_weights(source_model, target_model, *args, **kwargs)
