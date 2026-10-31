"""
Utility functions for TensorFlow/Keras - Framework-agnostic "missing" functions

Provides:
- lazy_flatten: Auto-shape flattener for transitioning from Conv to Dense layers
- loss_ncc: Normalized Cross-Correlation loss (for medical imaging)
- find_lr: Learning Rate Finder (LR range test)
- seed_everything: Cross-framework reproducible seed setter
- inspect_tensor: Diagnostic tensor statistics, memory, and health inspector
- freeze_layers / unfreeze_layers: Selective layer freezing for transfer learning
- freeze_batchnorm: Freeze BatchNormalization parameters and running statistics
"""

import tensorflow as tf
from tensorflow import keras
from typing import Optional, Tuple, Callable, Dict, Any, List, Union
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import re
import os
import random
import time
import onnx
import onnxruntime as ort
try:
    import tf2onnx
except Exception:
    tf2onnx = None


def toONNX(model: keras.Model, filepath: str, example_input: Optional[np.ndarray] = None,
           opset: int = 11, input_signature: Optional[list] = None) -> str:
    """Export a Keras model to ONNX using tf2onnx.

    Args:
        model: Keras model instance (built)
        filepath: Path to write ONNX model
        example_input: Example numpy input to infer signature (batch dim allowed)
        opset: ONNX opset version
        input_signature: Optional list of `tf.TensorSpec` objects

    Returns:
        The filepath written.
    """
    if tf2onnx is None:
        raise ImportError("tf2onnx is required to export Keras models to ONNX")

    # Infer input signature if not provided
    if input_signature is None:
        if example_input is None:
            raise ValueError("Either example_input or input_signature must be provided")
        arr = example_input
        if isinstance(arr, tf.Tensor):
            arr = arr.numpy()
        # allow variable batch dimension
        shape = list(arr.shape)
        shape[0] = None
        spec = [tf.TensorSpec(shape, dtype=arr.dtype, name="input")]
    else:
        spec = input_signature

    try:
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset)
    except Exception:
        # Fallback: create a concrete function and convert from function
        fn = tf.function(model)
        model_proto, _ = tf2onnx.convert.from_function(fn, input_signature=spec, opset=opset)
    onnx.save_model(model_proto, filepath)
    return filepath


def fromONNX(onnx_path: str, inputs, providers: Optional[list] = None):
    """Run inference with an ONNX model using ONNX Runtime.

    Args:
        onnx_path: Path to ONNX model
        inputs: numpy array for single input or dict{name: np.array}
        providers: Optional list of ORT providers

    Returns:
        Numpy array or list of outputs
    """
    # Use only available providers to avoid noisy warnings when optional providers
    # are not installed in the environment.
    available = ort.get_available_providers()
    sess = ort.InferenceSession(onnx_path, providers=(providers if providers is not None else available) or None)
    input_meta = sess.get_inputs()
    if isinstance(inputs, dict):
        input_map = inputs
    else:
        # single ndarray -> map to first input name
        if len(input_meta) == 0:
            raise RuntimeError("ONNX model has no inputs")
        name = input_meta[0].name
        # if TF model expects NHWC, ensure inputs shape matches
        input_map = {name: inputs}

    outs = sess.run(None, input_map)
    if len(outs) == 1:
        return outs[0]
    return outs


def lazy_flatten(tensor: tf.Tensor, start_dim: int = 1) -> tf.Tensor:
    """
    Auto-shape flattener that automatically calculates correct dimensions.
    
    Eliminates the need to manually calculate flattened sizes when transitioning
    from convolutional layers to dense layers.
    
    Args:
        tensor: Input tensor to flatten
        start_dim: Dimension to start flattening from (default: 1, preserves batch)
    
    Returns:
        Flattened tensor
    
    Example:
        >>> x = tf.random.normal([32, 7, 7, 16])  # batch=32, H=7, W=7, channels=16
        >>> x_flat = lazy_flatten(x)              # Shape: [32, 784]
        >>> # No need to calculate 7*7*16 manually!
    """
    batch_size = tf.shape(tensor)[0]
    flattened = tf.reshape(tensor, [batch_size, -1])
    return flattened


def get_flatten_size(input_shape: Tuple[int, ...]) -> int:
    """
    Calculate the flattened size given an input shape.
    
    Useful for determining the input size of the first Dense layer after Conv layers.
    
    Args:
        input_shape: Shape of the tensor (excluding batch dimension)
                    e.g., (7, 7, 16) for Conv output
    
    Returns:
        Total flattened size
    
    Example:
        >>> size = get_flatten_size((7, 7, 16))
        >>> print(size)  # 784
        >>> # Now you can use: keras.layers.Dense(128, input_shape=(size,))
    """
    return int(np.prod(input_shape))


@tf.function
def loss_ncc(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float = 1e-8) -> tf.Tensor:
    """
    Normalized Cross-Correlation (NCC) loss function.
    
    Commonly used in medical image registration and similarity measurement.
    NCC is robust to intensity variations and widely used in ACDC, ADNI datasets.
    
    NCC = 1 - [ sum((y_true - mean_true) * (y_pred - mean_pred)) / 
                (sqrt(sum((y_true - mean_true)^2)) * sqrt(sum((y_pred - mean_pred)^2))) ]
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        eps: Small constant for numerical stability (default: 1e-8)
    
    Returns:
        NCC loss value (lower is better, range: 0 to 2)
    
    Example:
        >>> y_true = tf.random.normal([8, 64, 64, 1])
        >>> y_pred = model(x)
        >>> loss = loss_ncc(y_true, y_pred)
    
    Note:
        - Returns 0 when images are identical
        - Returns 2 when images are completely anti-correlated
        - More robust than MSE for images with intensity variations
        - Decorated with @tf.function for performance
    """
    # Get batch size
    batch_size = tf.shape(y_true)[0]
    
    # Flatten spatial dimensions
    y_true_flat = tf.reshape(y_true, [batch_size, -1])
    y_pred_flat = tf.reshape(y_pred, [batch_size, -1])
    
    # Calculate means
    mean_true = tf.reduce_mean(y_true_flat, axis=1, keepdims=True)
    mean_pred = tf.reduce_mean(y_pred_flat, axis=1, keepdims=True)
    
    # Center the tensors
    y_true_centered = y_true_flat - mean_true
    y_pred_centered = y_pred_flat - mean_pred
    
    # Calculate cross-correlation
    numerator = tf.reduce_sum(y_true_centered * y_pred_centered, axis=1)
    
    # Calculate standard deviations
    std_true = tf.sqrt(tf.reduce_sum(y_true_centered ** 2, axis=1) + eps)
    std_pred = tf.sqrt(tf.reduce_sum(y_pred_centered ** 2, axis=1) + eps)
    
    # NCC coefficient
    ncc = numerator / (std_true * std_pred + eps)
    
    # Return as loss (1 - NCC), averaged over batch
    return tf.reduce_mean(1.0 - ncc)


@tf.function
def ncc_score(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float = 1e-8) -> tf.Tensor:
    """
    Normalized Cross-Correlation score (similarity metric).
    
    Returns the NCC coefficient itself (higher is better, range: -1 to 1).
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        eps: Small constant for numerical stability
    
    Returns:
        NCC score (higher is better, 1 = perfect match)
    """
    return 1.0 - loss_ncc(y_true, y_pred, eps)


class NCCLoss(keras.losses.Loss):
    """
    Keras-compatible NCC Loss class.
    
    Can be used directly as a loss function in model.compile().
    
    Example:
        >>> model.compile(optimizer='adam', loss=NCCLoss())
        >>> model.fit(x_train, y_train, epochs=10)
    """
    
    def __init__(self, eps: float = 1e-8, name: str = "ncc_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.eps = eps
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return loss_ncc(y_true, y_pred, self.eps)
    
    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


class LRFinder:
    """
    Learning Rate Finder for TensorFlow/Keras models.
    
    Implements the LR range test popularized by fast.ai to find optimal learning rates.
    Runs the model for a few iterations with exponentially increasing learning rates
    and plots the loss to identify the "knee" of the curve.
    
    Example:
        >>> lr_finder = LRFinder(model, loss_fn=keras.losses.CategoricalCrossentropy())
        >>> lr_finder.range_test(train_dataset, start_lr=1e-7, end_lr=10, num_iter=100)
        >>> lr_finder.plot()
        >>> best_lr = lr_finder.get_best_lr()
        >>> print(f"Suggested LR: {best_lr}")
    """
    
    def __init__(
        self,
        model: keras.Model,
        loss_fn: Union[keras.losses.Loss, Callable],
        optimizer: Optional[keras.optimizers.Optimizer] = None
    ):
        """
        Initialize LR Finder.
        
        Args:
            model: Keras model to train
            loss_fn: Loss function (Keras Loss or callable)
            optimizer: Optimizer instance (optional, will create Adam if not provided)
        """
        self.model = model
        self.loss_fn = loss_fn
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        else:
            self.optimizer = optimizer
        
        # Store original weights
        self.original_weights = [w.numpy() for w in model.trainable_weights]
        
        # Results
        self.lrs: List[float] = []
        self.losses: List[float] = []
        self.best_lr: Optional[float] = None
    
    def range_test(
        self,
        dataset: tf.data.Dataset,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_iter: int = 100,
        smooth_f: float = 0.05,
        diverge_th: float = 5.0
    ):
        """
        Perform LR range test.
        
        Args:
            dataset: Training dataset (tf.data.Dataset)
            start_lr: Starting learning rate (default: 1e-7)
            end_lr: Ending learning rate (default: 10.0)
            num_iter: Number of iterations to run (default: 100)
            smooth_f: Loss smoothing factor (default: 0.05)
            diverge_th: Stop if loss > diverge_th * best_loss (default: 5.0)
        """
        # Reset to original weights
        for w, w_orig in zip(self.model.trainable_weights, self.original_weights):
            w.assign(w_orig)
        
        # Calculate LR multiplier
        lr_mult = (end_lr / start_lr) ** (1.0 / num_iter)
        
        # Initialize
        lr = start_lr
        self.optimizer.learning_rate.assign(lr)
        
        avg_loss = 0.0
        best_loss = float('inf')
        
        self.lrs = []
        self.losses = []
        
        # Create iterator
        dataset_iter = iter(dataset)
        
        with tqdm(range(num_iter), desc="LR Finder") as pbar:
            for iteration in pbar:
                # Get batch
                try:
                    batch = next(dataset_iter)
                except StopIteration:
                    dataset_iter = iter(dataset)
                    batch = next(dataset_iter)
                
                # Unpack batch
                if isinstance(batch, tuple):
                    inputs, targets = batch
                else:
                    inputs = batch
                    targets = None
                
                # Forward and backward pass
                with tf.GradientTape() as tape:
                    predictions = self.model(inputs, training=True)
                    
                    if targets is not None:
                        loss = self.loss_fn(targets, predictions)
                    else:
                        loss = self.loss_fn(predictions)
                
                # Check for nan/inf
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    break
                
                # Compute gradients and update
                gradients = tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
                
                # Smooth loss
                loss_val = float(loss.numpy())
                if iteration == 0:
                    avg_loss = loss_val
                else:
                    avg_loss = smooth_f * loss_val + (1 - smooth_f) * avg_loss
                
                # Record
                self.lrs.append(lr)
                self.losses.append(avg_loss)
                
                # Update best loss
                if avg_loss < best_loss:
                    best_loss = avg_loss
                
                # Check for divergence
                if avg_loss > diverge_th * best_loss:
                    print(f"\nStopping early - loss diverged at LR={lr:.2e}")
                    break
                
                # Update progress bar
                pbar.set_postfix({'lr': f'{lr:.2e}', 'loss': f'{avg_loss:.4f}'})
                
                # Increase learning rate
                lr *= lr_mult
                self.optimizer.learning_rate.assign(lr)
        
        # Restore original weights
        for w, w_orig in zip(self.model.trainable_weights, self.original_weights):
            w.assign(w_orig)
        
        # Calculate best LR
        self.best_lr = self._calculate_best_lr()
    
    def _calculate_best_lr(self) -> Optional[float]:
        """
        Calculate the best learning rate based on the steepest gradient.
        
        Returns the LR where the loss decrease is steepest (before divergence).
        """
        if len(self.losses) < 2:
            return None
        
        # Calculate gradients of loss
        losses_array = np.array(self.losses)
        lrs_array = np.array(self.lrs)
        
        # Find minimum loss and its index
        min_loss_idx = np.argmin(losses_array)
        
        # Calculate gradient (rate of change)
        gradients = np.gradient(losses_array)
        
        # Find steepest negative gradient before minimum
        if min_loss_idx > 10:
            steepest_idx = np.argmin(gradients[:min_loss_idx])
            return float(lrs_array[steepest_idx])
        else:
            # Return LR at 1/10th of minimum if not enough data
            return float(lrs_array[min_loss_idx] / 10.0)
    
    def plot(
        self,
        skip_start: int = 10,
        skip_end: int = 5,
        log_lr: bool = True,
        show_best: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Plot the learning rate vs loss curve.
        
        Args:
            skip_start: Skip first N points (default: 10)
            skip_end: Skip last N points (default: 5)
            log_lr: Use log scale for LR axis (default: True)
            show_best: Mark the suggested best LR (default: True)
            save_path: Path to save the plot (optional)
        """
        if len(self.lrs) == 0:
            print("No data to plot. Run range_test() first.")
            return
        
        # Prepare data
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses, linewidth=2)
        
        if log_lr:
            plt.xscale('log')
        
        # Mark best LR
        if show_best and self.best_lr is not None:
            plt.axvline(x=self.best_lr, color='r', linestyle='--', 
                       label=f'Suggested LR: {self.best_lr:.2e}')
            plt.legend()
        
        plt.xlabel('Learning Rate', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Learning Rate Finder', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def get_best_lr(self) -> Optional[float]:
        """
        Get the suggested best learning rate.
        
        Returns:
            Suggested learning rate or None if range_test not run
        """
        return self.best_lr


    def toONNX(
        model: keras.Model,
        filepath: str,
        example_input: Optional[Union[np.ndarray, tf.Tensor]] = None,
        opset: int = 11,
        input_signature: Optional[list] = None,
    ):
        """
        Export a Keras model to ONNX using tf2onnx.

        Args:
            model: Keras model
            filepath: path where to save ONNX model
            example_input: optional example input used to build signature
            opset: ONNX opset version
            input_signature: optional tf.TensorSpec list

        Returns:
            filepath
        """
        if tf2onnx is None:
            raise ImportError("tf2onnx is required to export TensorFlow models to ONNX")

        # Build input signature from example input if provided
        if input_signature is None and example_input is not None:
            if isinstance(example_input, np.ndarray):
                input_signature = [tf.TensorSpec(example_input.shape, dtype=tf.as_dtype(example_input.dtype))]
            elif isinstance(example_input, tf.Tensor):
                input_signature = [tf.TensorSpec(example_input.shape, dtype=example_input.dtype)]

        # Use tf2onnx converter
        try:
            spec = None
            if input_signature is not None:
                spec = input_signature

            model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset)
            onnx.save_model(model_proto, filepath)
        except Exception as e:
            raise

        return filepath


    def fromONNX(onnx_path: str, inputs, providers: Optional[list] = None):
        """
        Run inference on an ONNX model using onnxruntime (tf wrapper convenience).

        Args:
            onnx_path: path to onnx file
            inputs: single input (tf.Tensor/ndarray) or dict mapping input names to arrays
            providers: optional ORT providers list

        Returns:
            numpy array or list of numpy outputs
        """
        sess_options = ort.SessionOptions()
        if providers is None:
            providers = None

        sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers or ort.get_all_providers())

        input_map = {}
        ort_inputs = sess.get_inputs()
        if isinstance(inputs, dict):
            for k, v in inputs.items():
                if isinstance(v, tf.Tensor):
                    v = v.numpy()
                input_map[k] = np.asarray(v)
        else:
            v = inputs
            if isinstance(v, tf.Tensor):
                v = v.numpy()
            input_map[ort_inputs[0].name] = np.asarray(v)

        out = sess.run(None, input_map)
        if len(out) == 1:
            return out[0]
        return out


def find_lr(
    model: keras.Model,
    loss_fn: Union[keras.losses.Loss, Callable],
    dataset: tf.data.Dataset,
    optimizer: Optional[keras.optimizers.Optimizer] = None,
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    num_iter: int = 100,
    plot: bool = True
) -> float:
    """
    Convenience function to find optimal learning rate.
    
    Args:
        model: Keras model
        loss_fn: Loss function
        dataset: Training dataset (tf.data.Dataset)
        optimizer: Optimizer instance (optional)
        start_lr: Starting learning rate (default: 1e-7)
        end_lr: Ending learning rate (default: 10.0)
        num_iter: Number of iterations (default: 100)
        plot: Whether to show plot (default: True)
    
    Returns:
        Suggested optimal learning rate
    
    Example:
        >>> best_lr = find_lr(model, keras.losses.SparseCategoricalCrossentropy(),
        ...                   train_dataset)
        >>> print(f"Use learning rate: {best_lr}")
        >>> # Now create optimizer with best_lr
        >>> optimizer = keras.optimizers.Adam(learning_rate=best_lr)
    """
    lr_finder = LRFinder(model, loss_fn, optimizer)
    lr_finder.range_test(dataset, start_lr, end_lr, num_iter)
    
    if plot:
        lr_finder.plot()
    
    best_lr = lr_finder.get_best_lr()
    
    if best_lr is None:
        print("Warning: Could not determine best LR. Using default 1e-3")
        return 1e-3
    
    return best_lr


def seed_everything(
    seed: int = 42,
    deterministic: bool = False,
    benchmark: bool = False,
    verbose: bool = False,
) -> int:
    """
    Sets seeds across Python, NumPy, TensorFlow, and PyTorch for reproducibility.

    Args:
        seed: The integer seed value (default: 42).
        deterministic: If True, instructs TensorFlow to enable op determinism.
        benchmark: If True and deterministic is False, sets torch cudnn.benchmark if torch is installed.
        verbose: If True, prints messages about which backends were seeded.

    Returns:
        The integer seed used.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    seeded = ["python", "numpy"]

    tf.random.set_seed(seed)
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    seeded.append("tensorflow")

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        elif benchmark:
            torch.backends.cudnn.benchmark = True
        seeded.append("torch")
    except ImportError:
        pass

    if verbose:
        print(f"[ToTf] Seeded {', '.join(seeded)} with seed={seed} (deterministic={deterministic})")

    return seed


def inspect_tensor(
    tensor: Any,
    name: Optional[str] = None,
    print_info: bool = True,
    quantiles: bool = True,
) -> Dict[str, Any]:
    """
    Inspects a tensor (TensorFlow, PyTorch, or NumPy) and provides a comprehensive diagnostic
    summary including shape, dtype, device, memory footprint, statistics, NaN/Inf detection,
    and quantiles.

    Args:
        tensor: Tensor, variable, or array to inspect.
        name: Optional identifier or label for the tensor (e.g. 'layer1_act').
        print_info: If True, prints formatted summary.
        quantiles: If True, computes 25th, 50th (median), and 75th percentiles.

    Returns:
        Dict containing all measured metrics.
    """
    framework = "numpy"
    device_str = "cpu"
    min_val = None
    max_val = None
    mean_val = None
    std_val = None
    zero_count = 0
    sparsity_pct = 0.0
    has_nan = False
    nan_count = 0
    has_inf = False
    inf_count = 0
    q_dict: Dict[str, float] = {}

    if isinstance(tensor, (tf.Tensor, tf.Variable)):
        framework = "tensorflow"
        shape = tuple(tensor.shape.as_list()) if hasattr(tensor.shape, "as_list") else tuple(tensor.shape)
        dtype_str = tensor.dtype.name
        device_str = getattr(tensor, "device", "cpu") or "cpu"
        try:
            arr = tensor.numpy()
        except Exception:
            arr = np.asarray(tensor)
        numel = arr.size
        itemsize = arr.itemsize

        if numel > 0:
            if np.issubdtype(arr.dtype, np.floating):
                isnan = np.isnan(arr)
                nan_count = int(np.sum(isnan))
                has_nan = nan_count > 0

                isinf = np.isinf(arr)
                inf_count = int(np.sum(isinf))
                has_inf = inf_count > 0

                zero_count = int(np.sum(arr == 0))
                sparsity_pct = (zero_count / numel) * 100.0

                valid = arr[~(isnan | isinf)]
                if valid.size > 0:
                    min_val = float(np.min(valid))
                    max_val = float(np.max(valid))
                    mean_val = float(np.mean(valid))
                    std_val = float(np.std(valid)) if valid.size > 1 else 0.0
                    if quantiles:
                        q = np.percentile(valid, [25, 50, 75])
                        q_dict = {"p25": float(q[0]), "p50": float(q[1]), "p75": float(q[2])}
            elif np.issubdtype(arr.dtype, np.number):
                zero_count = int(np.sum(arr == 0))
                sparsity_pct = (zero_count / numel) * 100.0
                min_val = float(np.min(arr))
                max_val = float(np.max(arr))
                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr)) if numel > 1 else 0.0
                if quantiles:
                    q = np.percentile(arr, [25, 50, 75])
                    q_dict = {"p25": float(q[0]), "p50": float(q[1]), "p75": float(q[2])}
    else:
        is_torch = False
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                is_torch = True
        except ImportError:
            pass

        if is_torch:
            framework = "torch"
            shape = tuple(tensor.shape)
            dtype_str = str(tensor.dtype).replace("torch.", "")
            device_str = str(tensor.device)
            numel = tensor.numel()
            itemsize = tensor.element_size()

            if numel > 0:
                if tensor.is_floating_point() or tensor.is_complex():
                    isnan = torch.isnan(tensor)
                    nan_count = int(torch.sum(isnan).item())
                    has_nan = nan_count > 0

                    isinf = torch.isinf(tensor)
                    inf_count = int(torch.sum(isinf).item())
                    has_inf = inf_count > 0

                    zero_count = int(torch.sum(tensor == 0).item())
                    sparsity_pct = (zero_count / numel) * 100.0

                    valid_mask = ~(isnan | isinf)
                    valid_count = int(torch.sum(valid_mask).item())
                    if valid_count > 0:
                        valid_vals = tensor[valid_mask].float()
                        min_val = float(torch.min(valid_vals).item())
                        max_val = float(torch.max(valid_vals).item())
                        mean_val = float(torch.mean(valid_vals).item())
                        std_val = float(torch.std(valid_vals).item()) if valid_count > 1 else 0.0
                        if quantiles:
                            q = torch.quantile(valid_vals, torch.tensor([0.25, 0.50, 0.75], device=valid_vals.device))
                            q_dict = {
                                "p25": float(q[0].item()),
                                "p50": float(q[1].item()),
                                "p75": float(q[2].item()),
                            }
                else:
                    zero_count = int(torch.sum(tensor == 0).item())
                    sparsity_pct = (zero_count / numel) * 100.0
                    t_float = tensor.float()
                    min_val = float(torch.min(t_float).item())
                    max_val = float(torch.max(t_float).item())
                    mean_val = float(torch.mean(t_float).item())
                    std_val = float(torch.std(t_float).item()) if numel > 1 else 0.0
                    if quantiles:
                        q = torch.quantile(t_float, torch.tensor([0.25, 0.50, 0.75], device=t_float.device))
                        q_dict = {
                            "p25": float(q[0].item()),
                            "p50": float(q[1].item()),
                            "p75": float(q[2].item()),
                        }
        else:
            arr = np.asarray(tensor)
            framework = "numpy"
            shape = arr.shape
            dtype_str = str(arr.dtype)
            device_str = "cpu"
            numel = arr.size
            itemsize = arr.itemsize

            if numel > 0:
                if np.issubdtype(arr.dtype, np.floating):
                    isnan = np.isnan(arr)
                    nan_count = int(np.sum(isnan))
                    has_nan = nan_count > 0

                    isinf = np.isinf(arr)
                    inf_count = int(np.sum(isinf))
                    has_inf = inf_count > 0

                    zero_count = int(np.sum(arr == 0))
                    sparsity_pct = (zero_count / numel) * 100.0

                    valid = arr[~(isnan | isinf)]
                    if valid.size > 0:
                        min_val = float(np.min(valid))
                        max_val = float(np.max(valid))
                        mean_val = float(np.mean(valid))
                        std_val = float(np.std(valid)) if valid.size > 1 else 0.0
                        if quantiles:
                            q = np.percentile(valid, [25, 50, 75])
                            q_dict = {"p25": float(q[0]), "p50": float(q[1]), "p75": float(q[2])}
                elif np.issubdtype(arr.dtype, np.number):
                    zero_count = int(np.sum(arr == 0))
                    sparsity_pct = (zero_count / numel) * 100.0
                    min_val = float(np.min(arr))
                    max_val = float(np.max(arr))
                    mean_val = float(np.mean(arr))
                    std_val = float(np.std(arr)) if numel > 1 else 0.0
                    if quantiles:
                        q = np.percentile(arr, [25, 50, 75])
                        q_dict = {"p25": float(q[0]), "p50": float(q[1]), "p75": float(q[2])}

    memory_mb = (numel * itemsize) / (1024 * 1024)

    tag = f"Tensor: {name}" if name else "Tensor"
    line1 = f"[{tag}] Framework: {framework} | Shape: {shape} | Dtype: {dtype_str} | Device: {device_str} | Mem: {memory_mb:.4f} MB ({numel:,} elements)"
    if min_val is not None:
        line2 = f"  Stats: Min={min_val:.4f} | Mean={mean_val:.4f} | Max={max_val:.4f} | Std={std_val:.4f} | Zeros={zero_count} ({sparsity_pct:.1f}%)"
    else:
        line2 = "  Stats: Empty or Non-numeric"
    line3 = f"  Health: NaNs={nan_count} | Infs={inf_count}"
    if q_dict:
        line3 += f" | Quantiles: [p25={q_dict['p25']:.4f}, p50={q_dict['p50']:.4f}, p75={q_dict['p75']:.4f}]"

    summary_text = f"{line1}\n{line2}\n{line3}"
    if print_info:
        print(summary_text)

    return {
        "name": name,
        "framework": framework,
        "shape": shape,
        "dtype": dtype_str,
        "device": device_str,
        "numel": numel,
        "memory_mb": memory_mb,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "zeros_count": zero_count,
        "sparsity_pct": sparsity_pct,
        "has_nan": has_nan,
        "nan_count": nan_count,
        "has_inf": has_inf,
        "inf_count": inf_count,
        "quantiles": q_dict,
        "summary": summary_text,
    }


def freeze_layers(
    model: keras.Model,
    up_to: Optional[Union[str, int]] = None,
    layer_names: Optional[List[str]] = None,
    layer_types: Optional[Tuple[type, ...]] = None,
    regex: Optional[str] = None,
) -> int:
    """
    Freeze layers in a Keras/TensorFlow model for transfer learning or fine-tuning.

    Args:
        model: Keras Model instance.
        up_to: Either an int (freeze the first N layers in model.layers)
               or a str (freeze all layers up to and including the layer named `up_to`).
        layer_names: List of layer name substrings to freeze.
        layer_types: Tuple of layer types to freeze (e.g. (layers.Conv2D, layers.Dense)).
        regex: Regular expression string matched against layer names.

    Returns:
        Number of layers frozen.
    """
    compiled_regex = re.compile(regex) if regex else None
    layers_to_freeze = set()

    # 1. up_to
    if up_to is not None:
        if isinstance(up_to, int):
            for l in model.layers[:up_to]:
                layers_to_freeze.add(l)
        elif isinstance(up_to, str):
            for l in model.layers:
                layers_to_freeze.add(l)
                if l.name == up_to or l.name.endswith(f"/{up_to}"):
                    break

    # 2. layer_types
    if layer_types is not None:
        for l in model.layers:
            if isinstance(l, layer_types):
                layers_to_freeze.add(l)

    # 3. layer_names
    if layer_names is not None:
        for l in model.layers:
            if any(target in l.name for target in layer_names):
                layers_to_freeze.add(l)

    # 4. regex
    if compiled_regex is not None:
        for l in model.layers:
            if compiled_regex.search(l.name):
                layers_to_freeze.add(l)

    # 5. Default: freeze all layers
    if up_to is None and layer_names is None and layer_types is None and regex is None:
        for l in model.layers:
            layers_to_freeze.add(l)

    frozen_count = 0
    for l in layers_to_freeze:
        if l.trainable:
            l.trainable = False
            frozen_count += 1

    return frozen_count


def unfreeze_layers(
    model: keras.Model,
    last_n: Optional[int] = None,
    layer_names: Optional[List[str]] = None,
    layer_types: Optional[Tuple[type, ...]] = None,
    regex: Optional[str] = None,
) -> int:
    """
    Unfreeze layers in a Keras/TensorFlow model.

    Args:
        model: Keras Model instance.
        last_n: If provided, unfreezes the last N layers in model.layers (e.g. classification head).
        layer_names: List of layer name substrings to unfreeze.
        layer_types: Tuple of layer types to unfreeze.
        regex: Regular expression string matched against layer names.

    Returns:
        Number of layers unfrozen.
    """
    compiled_regex = re.compile(regex) if regex else None
    layers_to_unfreeze = set()

    if last_n is not None:
        target = model.layers[-last_n:] if last_n > 0 else []
        for l in target:
            layers_to_unfreeze.add(l)

    if layer_types is not None:
        for l in model.layers:
            if isinstance(l, layer_types):
                layers_to_unfreeze.add(l)

    if layer_names is not None:
        for l in model.layers:
            if any(target in l.name for target in layer_names):
                layers_to_unfreeze.add(l)

    if compiled_regex is not None:
        for l in model.layers:
            if compiled_regex.search(l.name):
                layers_to_unfreeze.add(l)

    if last_n is None and layer_names is None and layer_types is None and regex is None:
        for l in model.layers:
            layers_to_unfreeze.add(l)

    unfrozen_count = 0
    for l in layers_to_unfreeze:
        if not l.trainable:
            l.trainable = True
            unfrozen_count += 1

    return unfrozen_count


def freeze_batchnorm(model: keras.Model) -> int:
    """
    Freeze all BatchNormalization layers in a Keras model.
    In Keras, setting layer.trainable = False on BatchNormalization freezes both
    affine weights and prevents updates to moving_mean and moving_variance during training.

    Args:
        model: Keras Model instance.

    Returns:
        Number of BatchNormalization layers frozen.
    """
    bn_types = (
        keras.layers.BatchNormalization,
        tf.keras.layers.BatchNormalization,
    )
    count = 0
    for l in model.layers:
        if isinstance(l, bn_types):
            if l.trainable:
                l.trainable = False
                count += 1
        elif hasattr(l, "layers"):
            count += freeze_batchnorm(l)
    return count


# =========================================================================
# Phase 2: Domain Losses, Metrics & Anomaly Diagnostics
# =========================================================================

def loss_focal(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
    from_logits: bool = False,
    eps: float = 1e-8,
) -> tf.Tensor:
    """
    Focal Loss for addressing class imbalance in TensorFlow / Keras (Lin et al., ICCV 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Supports binary/multilabel and multiclass classification.

    Args:
        y_true: Ground truth tensor (one-hot or probabilities matching y_pred shape, or class indices).
        y_pred: Predicted probabilities or unnormalized logits.
        alpha: Weighting factor for rare class (default: 0.25).
        gamma: Focusing parameter (default: 2.0).
        reduction: 'mean', 'sum', or 'none'.
        from_logits: Whether y_pred contains raw logits or probabilities.
        eps: Small constant for numerical stability.

    Returns:
        Focal loss tensor.
    """
    try:
        import torch
        if isinstance(y_true, torch.Tensor) or isinstance(y_pred, torch.Tensor):
            from ToTf.pytorch.utils import loss_focal as th_loss_focal
            return th_loss_focal(y_true, y_pred, alpha=alpha, gamma=gamma, reduction=reduction, from_logits=from_logits, eps=eps)
    except ImportError:
        pass

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # If y_true is 1D class indices and y_pred is (*, C)
    if len(y_true.shape) == len(y_pred.shape) - 1:
        num_classes = y_pred.shape[-1]
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)

    if from_logits:
        prob = tf.sigmoid(y_pred)
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    else:
        prob = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        bce = -(y_true * tf.math.log(prob + eps) + (1.0 - y_true) * tf.math.log(1.0 - prob + eps))

    p_t = prob * y_true + (1.0 - prob) * (1.0 - y_true)
    alpha_factor = alpha * y_true + (1.0 - alpha) * (1.0 - y_true)
    loss = alpha_factor * tf.pow(1.0 - p_t, gamma) * bce

    if reduction == "mean":
        return tf.reduce_mean(loss)
    elif reduction == "sum":
        return tf.reduce_sum(loss)
    return loss


class FocalLoss(keras.losses.Loss):
    """
    Keras Loss class wrapper for Focal Loss.
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        from_logits: bool = False,
        name: str = "focal_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.reduction_type = reduction

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return loss_focal(
            y_true=y_true,
            y_pred=y_pred,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction_type,
            from_logits=self.from_logits,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "reduction_type": self.reduction_type,
            "from_logits": self.from_logits,
        })
        return config


def loss_dice(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smooth: float = 1e-6,
    from_logits: bool = False,
    reduction: str = "mean",
) -> tf.Tensor:
    """
    Dice Loss for image and volume segmentation in TensorFlow / Keras.

    Dice = (2 * |X n Y| + smooth) / (|X| + |Y| + smooth)
    Loss = 1.0 - Dice

    Args:
        y_true: Ground truth binary/probability masks.
        y_pred: Predicted probabilities or logits.
        smooth: Smoothing factor to prevent division by zero.
        from_logits: If True, applies sigmoid to y_pred.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Dice loss value (range [0, 1]).
    """
    try:
        import torch
        if isinstance(y_true, torch.Tensor) or isinstance(y_pred, torch.Tensor):
            from ToTf.pytorch.utils import loss_dice as th_loss_dice
            return th_loss_dice(y_true, y_pred, smooth=smooth, from_logits=from_logits, reduction=reduction)
    except ImportError:
        pass

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if from_logits:
        y_pred = tf.sigmoid(y_pred)

    batch_size = tf.shape(y_pred)[0]
    pred_flat = tf.reshape(y_pred, [batch_size, -1])
    true_flat = tf.reshape(y_true, [batch_size, -1])

    intersection = tf.reduce_sum(pred_flat * true_flat, axis=1)
    cardinality = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    loss = 1.0 - dice

    if reduction == "mean":
        return tf.reduce_mean(loss)
    elif reduction == "sum":
        return tf.reduce_sum(loss)
    return loss


class DiceLoss(keras.losses.Loss):
    """
    Keras Loss class wrapper for Dice Loss.
    """
    def __init__(
        self,
        smooth: float = 1e-6,
        from_logits: bool = False,
        reduction: str = "mean",
        name: str = "dice_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.from_logits = from_logits
        self.reduction_type = reduction

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return loss_dice(
            y_true=y_true,
            y_pred=y_pred,
            smooth=self.smooth,
            from_logits=self.from_logits,
            reduction=self.reduction_type,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "smooth": self.smooth,
            "from_logits": self.from_logits,
            "reduction_type": self.reduction_type,
        })
        return config


def dice_score(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    threshold: Optional[float] = 0.5,
    smooth: float = 1e-6,
) -> tf.Tensor:
    """
    Dice Similarity Coefficient (DSC) evaluation metric in TensorFlow.
    """
    y_true = tf.cast(y_true, tf.float32)
    if threshold is not None:
        y_pred = tf.cast(y_pred >= threshold, tf.float32)
    else:
        y_pred = tf.cast(y_pred, tf.float32)

    batch_size = tf.shape(y_pred)[0]
    pred_flat = tf.reshape(y_pred, [batch_size, -1])
    true_flat = tf.reshape(y_true, [batch_size, -1])

    intersection = tf.reduce_sum(pred_flat * true_flat, axis=1)
    cardinality = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return tf.reduce_mean(dice)


def iou_score(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    threshold: Optional[float] = 0.5,
    smooth: float = 1e-6,
) -> tf.Tensor:
    """
    Intersection over Union (IoU / Jaccard Index) evaluation metric in TensorFlow.
    """
    y_true = tf.cast(y_true, tf.float32)
    if threshold is not None:
        y_pred = tf.cast(y_pred >= threshold, tf.float32)
    else:
        y_pred = tf.cast(y_pred, tf.float32)

    batch_size = tf.shape(y_pred)[0]
    pred_flat = tf.reshape(y_pred, [batch_size, -1])
    true_flat = tf.reshape(y_true, [batch_size, -1])

    intersection = tf.reduce_sum(pred_flat * true_flat, axis=1)
    union = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)


class AnomalyDetector:
    """
    Context manager to monitor Keras models for NaNs or Infs during layer execution.
    """
    def __init__(self, model: keras.Model, raise_error: bool = False, **kwargs):
        self.model = model
        self.raise_error = raise_error
        self.anomalies: List[Dict[str, Any]] = []
        self._orig_calls = {}
        self.is_active = False

    def attach(self):
        if self.is_active:
            return self
        self.anomalies = []
        for layer in self.model.layers:
            orig_call = layer.call
            self._orig_calls[layer] = orig_call

            def make_call(l_name, l_type, fn):
                def wrapped_call(*args, **kwargs):
                    for idx, a in enumerate(args):
                        if isinstance(a, tf.Tensor) and tf.is_tensor(a):
                            try:
                                arr = a.numpy() if hasattr(a, 'numpy') else np.asarray(a)
                                nan_c = int(np.sum(np.isnan(arr)))
                                inf_c = int(np.sum(np.isinf(arr)))
                                if nan_c > 0 or inf_c > 0:
                                    rec = {
                                        "stage": "forward_input",
                                        "layer": l_name,
                                        "type": l_type,
                                        "arg_idx": idx,
                                        "nan_count": nan_c,
                                        "inf_count": inf_c,
                                        "shape": tuple(a.shape),
                                    }
                                    self.anomalies.append(rec)
                                    if self.raise_error:
                                        raise FloatingPointError(f"NaN/Inf detected in {l_name} input: {rec}")
                            except Exception:
                                pass

                    out = fn(*args, **kwargs)

                    outs = [out] if isinstance(out, tf.Tensor) else (out if isinstance(out, (list, tuple)) else [])
                    for idx, o in enumerate(outs):
                        if isinstance(o, tf.Tensor) and tf.is_tensor(o):
                            try:
                                arr = o.numpy() if hasattr(o, 'numpy') else np.asarray(o)
                                nan_c = int(np.sum(np.isnan(arr)))
                                inf_c = int(np.sum(np.isinf(arr)))
                                if nan_c > 0 or inf_c > 0:
                                    rec = {
                                        "stage": "forward_output",
                                        "layer": l_name,
                                        "type": l_type,
                                        "arg_idx": idx,
                                        "nan_count": nan_c,
                                        "inf_count": inf_c,
                                        "shape": tuple(o.shape),
                                    }
                                    self.anomalies.append(rec)
                                    if self.raise_error:
                                        raise FloatingPointError(f"NaN/Inf detected in {l_name} output: {rec}")
                            except Exception:
                                pass
                    return out
                return wrapped_call

            layer.call = make_call(layer.name, type(layer).__name__, orig_call)

        self.is_active = True
        return self

    def detach(self):
        for layer, orig_call in self._orig_calls.items():
            layer.call = orig_call
        self._orig_calls.clear()
        self.is_active = False

    def __enter__(self):
        return self.attach()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.detach()

    @property
    def has_anomalies(self) -> bool:
        return len(self.anomalies) > 0

    def print_report(self):
        if not self.has_anomalies:
            print("[AnomalyDetector] No NaN/Inf anomalies detected.")
            return
        print(f"\n[AnomalyDetector] Found {len(self.anomalies)} anomaly event(s):")
        for i, a in enumerate(self.anomalies, 1):
            print(f"  [{i}] Stage: {a['stage']} | Layer: '{a['layer']}' ({a['type']}) | "
                  f"Shape: {a['shape']} | NaNs: {a['nan_count']} | Infs: {a['inf_count']}")


def check_anomalies(model: keras.Model, sample_input: Any) -> List[Dict[str, Any]]:
    """
    Runs a forward pass through Keras model and returns any NaN/Inf anomalies.
    """
    with AnomalyDetector(model, raise_error=False) as detector:
        _ = model(sample_input, training=False)
    return detector.anomalies


# =========================================================================
# Phase 3: Research Benchmarking & Cross-Framework Weight Transfer
# =========================================================================

def benchmark_model(
    model: Any,
    input_size: Optional[Tuple[int, ...]] = None,
    input_data: Optional[Any] = None,
    device: Optional[str] = None,
    warmup: int = 10,
    repetition: int = 50,
    batch_size: int = 1,
    print_table: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive performance and complexity benchmark for TensorFlow and PyTorch models.
    """
    is_tf_model = isinstance(model, (keras.Model, keras.layers.Layer))
    is_torch_model = False
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            is_torch_model = True
    except ImportError:
        torch = None

    if not is_torch_model and not is_tf_model:
        raise TypeError(f"Unsupported model type: {type(model)}")

    # 1. Parameter counting
    total_params = 0
    trainable_params = 0
    param_bytes = 0
    estimated_macs = 0

    if is_tf_model:
        total_params = int(sum(np.prod(w.shape) for w in model.weights))
        trainable_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
        non_trainable_params = total_params - trainable_params
        param_memory_mb = (total_params * 4) / (1024 * 1024)
        for layer in model.layers:
            w_c = sum(int(np.prod(w.shape)) for w in layer.weights)
            if 'conv' in layer.name.lower():
                estimated_macs += w_c * 14 * 14
            else:
                estimated_macs += w_c
    else:
        for p in model.parameters():
            total_params += p.numel()
            param_bytes += p.numel() * p.element_size()
            if p.requires_grad:
                trainable_params += p.numel()
        non_trainable_params = total_params - trainable_params
        param_memory_mb = param_bytes / (1024 * 1024)
        for m in model.modules():
            if hasattr(m, "weight") and m.weight is not None:
                if 'Conv' in type(m).__name__:
                    estimated_macs += m.weight.numel() * 14 * 14
                elif 'Linear' in type(m).__name__:
                    estimated_macs += m.weight.numel()

    estimated_flops = estimated_macs * 2

    # 2. Timing
    target_device = device
    if is_tf_model:
        if input_data is not None:
            x = tf.convert_to_tensor(input_data)
        elif input_size is not None:
            shape = (batch_size, *input_size)
            x = tf.random.normal(shape)
        else:
            raise ValueError("Either input_size or input_data must be provided.")

        for _ in range(warmup):
            _ = model(x, training=False)

        latencies = []
        for _ in range(repetition):
            t0 = time.perf_counter()
            _ = model(x, training=False)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
    else:
        if target_device is None:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        model_device = torch.device(target_device)
        model = model.to(model_device)
        was_train = model.training
        model.eval()

        if input_data is not None:
            x = torch.as_tensor(input_data, device=model_device)
        elif input_size is not None:
            shape = (batch_size, *input_size)
            x = torch.randn(*shape, device=model_device)
        else:
            raise ValueError("Either input_size or input_data must be provided.")

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)
                if target_device == "cuda":
                    torch.cuda.synchronize()

            latencies = []
            if target_device == "cuda":
                for _ in range(repetition):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    _ = model(x)
                    end_event.record()
                    torch.cuda.synchronize()
                    latencies.append(start_event.elapsed_time(end_event))
            else:
                for _ in range(repetition):
                    t0 = time.perf_counter()
                    _ = model(x)
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)

        model.train(was_train)

    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    std_lat = float(np.std(lat_arr))
    median_lat = float(np.median(lat_arr))
    p90_lat = float(np.percentile(lat_arr, 90))
    p95_lat = float(np.percentile(lat_arr, 95))
    p99_lat = float(np.percentile(lat_arr, 99))
    fps = float((batch_size * 1000.0) / mean_lat) if mean_lat > 0 else 0.0

    model_name = getattr(model, "name", None) or getattr(model, "__name__", None) or type(model).__name__

    markdown_table = (
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| Model | {model_name} |\n"
        f"| Total Parameters | {total_params:,} ({total_params/1e6:.2f} M) |\n"
        f"| Trainable Parameters | {trainable_params:,} |\n"
        f"| Parameter Memory | {param_memory_mb:.2f} MB |\n"
        f"| Estimated MACs | {estimated_macs/1e9:.3f} G |\n"
        f"| Estimated FLOPs | {estimated_flops/1e9:.3f} G |\n"
        f"| Latency (Mean +/- Std) | {mean_lat:.2f} +/- {std_lat:.2f} ms |\n"
        f"| Latency (P50 / P95 / P99) | {median_lat:.2f} / {p95_lat:.2f} / {p99_lat:.2f} ms |\n"
        f"| Throughput | {fps:.1f} FPS (batch={batch_size}) |"
    )

    latex_table = (
        r"\begin{table}[h]" "\n"
        r"\centering" "\n"
        r"\begin{tabular}{lcccc}" "\n"
        r"\hline" "\n"
        r"Model & Params (M) & FLOPs (G) & Latency (ms) & Throughput (FPS) \\" "\n"
        r"\hline" "\n"
        f"{model_name} & {total_params/1e6:.2f} & {estimated_flops/1e9:.2f} & {median_lat:.2f} & {fps:.1f} \\\\" "\n"
        r"\hline" "\n"
        r"\end{tabular}" "\n"
        r"\caption{Model Performance and Complexity Benchmark}" "\n"
        r"\end{table}"
    )

    if print_table:
        print("\n" + "=" * 65)
        print(f"BENCHMARK REPORT: {model_name}")
        print("=" * 65)
        print(f"Parameters:     {total_params:,} ({total_params/1e6:.2f}M) [Trainable: {trainable_params:,}]")
        print(f"Memory:         {param_memory_mb:.2f} MB")
        print(f"Estimated FLOPs:{estimated_flops/1e9:.3f} G (MACs: {estimated_macs/1e9:.3f} G)")
        print(f"Latency (mean): {mean_lat:.2f} ms +/- {std_lat:.2f} ms")
        print(f"Latency (p50):  {median_lat:.2f} ms | p95: {p95_lat:.2f} ms | p99: {p99_lat:.2f} ms")
        print(f"Throughput:     {fps:.1f} FPS (batch_size={batch_size})")
        print("=" * 65)

    return {
        "model_name": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory_mb,
        "estimated_macs": estimated_macs,
        "estimated_flops": estimated_flops,
        "mean_latency_ms": mean_lat,
        "std_latency_ms": std_lat,
        "median_latency_ms": median_lat,
        "p90_latency_ms": p90_lat,
        "p95_latency_ms": p95_lat,
        "p99_latency_ms": p99_lat,
        "fps": fps,
        "latencies_ms": latencies,
        "markdown_table": markdown_table,
        "latex_table": latex_table,
    }


def transfer_weights(
    source_model: Any,
    target_model: Any,
    name_mapping: Optional[Dict[str, str]] = None,
    strict: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Direct in-memory weight transfer between TensorFlow/Keras and PyTorch models.
    """
    is_source_torch = False
    is_target_torch = False
    try:
        import torch
        if isinstance(source_model, torch.nn.Module):
            is_source_torch = True
        if isinstance(target_model, torch.nn.Module):
            is_target_torch = True
    except ImportError:
        torch = None

    source_items = []
    if is_source_torch:
        for name, mod in source_model.named_modules():
            m_type = type(mod).__name__
            if 'Conv' in m_type:
                source_items.append({
                    "name": name,
                    "type": "conv",
                    "weight": mod.weight.detach().cpu().numpy(),
                    "bias": mod.bias.detach().cpu().numpy() if mod.bias is not None else None,
                })
            elif 'Linear' in m_type:
                source_items.append({
                    "name": name,
                    "type": "linear",
                    "weight": mod.weight.detach().cpu().numpy(),
                    "bias": mod.bias.detach().cpu().numpy() if mod.bias is not None else None,
                })
            elif 'BatchNorm' in m_type:
                source_items.append({
                    "name": name,
                    "type": "bn",
                    "weight": mod.weight.detach().cpu().numpy() if mod.weight is not None else None,
                    "bias": mod.bias.detach().cpu().numpy() if mod.bias is not None else None,
                    "running_mean": mod.running_mean.detach().cpu().numpy() if mod.running_mean is not None else None,
                    "running_var": mod.running_var.detach().cpu().numpy() if mod.running_var is not None else None,
                })
    else:
        for layer in source_model.layers:
            l_type = type(layer).__name__.lower()
            if "conv" in l_type and len(layer.weights) > 0:
                weights = layer.get_weights()
                source_items.append({
                    "name": layer.name,
                    "type": "conv",
                    "weight": weights[0],
                    "bias": weights[1] if len(weights) > 1 else None,
                })
            elif "dense" in l_type and len(layer.weights) > 0:
                weights = layer.get_weights()
                source_items.append({
                    "name": layer.name,
                    "type": "linear",
                    "weight": weights[0],
                    "bias": weights[1] if len(weights) > 1 else None,
                })
            elif ("bn" in l_type or "batchnorm" in l_type or "batch_norm" in l_type) and len(layer.weights) > 0:
                weights = layer.get_weights()
                source_items.append({
                    "name": layer.name,
                    "type": "bn",
                    "weight": weights[0] if len(weights) > 0 else None,
                    "bias": weights[1] if len(weights) > 1 else None,
                    "running_mean": weights[2] if len(weights) > 2 else None,
                    "running_var": weights[3] if len(weights) > 3 else None,
                })

    target_items = []
    if is_target_torch:
        for name, mod in target_model.named_modules():
            m_type = type(mod).__name__
            if 'Conv' in m_type:
                target_items.append({"name": name, "type": "conv", "obj": mod})
            elif 'Linear' in m_type:
                target_items.append({"name": name, "type": "linear", "obj": mod})
            elif 'BatchNorm' in m_type:
                target_items.append({"name": name, "type": "bn", "obj": mod})
    else:
        for layer in target_model.layers:
            l_type = type(layer).__name__.lower()
            if "conv" in l_type and len(layer.weights) > 0:
                target_items.append({"name": layer.name, "type": "conv", "obj": layer})
            elif "dense" in l_type and len(layer.weights) > 0:
                target_items.append({"name": layer.name, "type": "linear", "obj": layer})
            elif ("bn" in l_type or "batchnorm" in l_type or "batch_norm" in l_type) and len(layer.weights) > 0:
                target_items.append({"name": layer.name, "type": "bn", "obj": layer})

    transferred_count = 0
    matched_pairs = []
    unmatched_src = [s["name"] for s in source_items]
    unmatched_tgt = [t["name"] for t in target_items]

    pairs_to_transfer = []
    if name_mapping:
        for s in source_items:
            tgt_name = name_mapping.get(s["name"])
            if tgt_name:
                for t in target_items:
                    if t["name"] == tgt_name:
                        pairs_to_transfer.append((s, t))
                        break
    else:
        s_idx = 0
        t_idx = 0
        while s_idx < len(source_items) and t_idx < len(target_items):
            s = source_items[s_idx]
            t = target_items[t_idx]
            if s["type"] == t["type"]:
                pairs_to_transfer.append((s, t))
                s_idx += 1
                t_idx += 1
            elif s_idx < t_idx:
                s_idx += 1
            else:
                t_idx += 1

    for s, t in pairs_to_transfer:
        s_w = s["weight"]
        s_b = s["bias"]
        s_type = s["type"]

        if is_source_torch and not is_target_torch:
            # PyTorch -> TensorFlow
            if s_type == "conv":
                if s_w.ndim == 4:
                    w_tf = np.transpose(s_w, (2, 3, 1, 0))
                elif s_w.ndim == 3:
                    w_tf = np.transpose(s_w, (2, 1, 0))
                else:
                    w_tf = s_w
                weights_to_set = [w_tf]
                if s_b is not None:
                    weights_to_set.append(s_b)
                t["obj"].set_weights(weights_to_set)
                transferred_count += w_tf.size + (s_b.size if s_b is not None else 0)

            elif s_type == "linear":
                w_tf = np.transpose(s_w, (1, 0))
                weights_to_set = [w_tf]
                if s_b is not None:
                    weights_to_set.append(s_b)
                t["obj"].set_weights(weights_to_set)
                transferred_count += w_tf.size + (s_b.size if s_b is not None else 0)

            elif s_type == "bn":
                weights_to_set = [s["weight"], s["bias"], s["running_mean"], s["running_var"]]
                valid_weights = [w for w in weights_to_set if w is not None]
                t["obj"].set_weights(valid_weights)
                transferred_count += sum(w.size for w in valid_weights)

        elif not is_source_torch and is_target_torch:
            # TensorFlow -> PyTorch
            if s_type == "conv":
                if s_w.ndim == 4:
                    w_th = np.transpose(s_w, (3, 2, 0, 1))
                elif s_w.ndim == 3:
                    w_th = np.transpose(s_w, (2, 1, 0))
                else:
                    w_th = s_w
                with torch.no_grad():
                    t["obj"].weight.copy_(torch.from_numpy(w_th))
                    if s_b is not None and t["obj"].bias is not None:
                        t["obj"].bias.copy_(torch.from_numpy(s_b))
                transferred_count += w_th.size + (s_b.size if s_b is not None else 0)

            elif s_type == "linear":
                w_th = np.transpose(s_w, (1, 0))
                with torch.no_grad():
                    t["obj"].weight.copy_(torch.from_numpy(w_th))
                    if s_b is not None and t["obj"].bias is not None:
                        t["obj"].bias.copy_(torch.from_numpy(s_b))
                transferred_count += w_th.size + (s_b.size if s_b is not None else 0)

            elif s_type == "bn":
                with torch.no_grad():
                    if s["weight"] is not None and t["obj"].weight is not None:
                        t["obj"].weight.copy_(torch.from_numpy(s["weight"]))
                    if s["bias"] is not None and t["obj"].bias is not None:
                        t["obj"].bias.copy_(torch.from_numpy(s["bias"]))
                    if s["running_mean"] is not None and t["obj"].running_mean is not None:
                        t["obj"].running_mean.copy_(torch.from_numpy(s["running_mean"]))
                    if s["running_var"] is not None and t["obj"].running_var is not None:
                        t["obj"].running_var.copy_(torch.from_numpy(s["running_var"]))
                transferred_count += sum(w.size for w in [s["weight"], s["bias"], s["running_mean"], s["running_var"]] if w is not None)

        else:
            if is_target_torch:
                with torch.no_grad():
                    t["obj"].weight.copy_(torch.from_numpy(s_w))
                    if s_b is not None and t["obj"].bias is not None:
                        t["obj"].bias.copy_(torch.from_numpy(s_b))
            else:
                t["obj"].set_weights([s_w] + ([s_b] if s_b is not None else []))
            transferred_count += s_w.size + (s_b.size if s_b is not None else 0)

        matched_pairs.append((s["name"], t["name"]))
        if s["name"] in unmatched_src:
            unmatched_src.remove(s["name"])
        if t["name"] in unmatched_tgt:
            unmatched_tgt.remove(t["name"])

    if strict and (unmatched_src or unmatched_tgt):
        raise ValueError(f"Strict transfer failed. Unmatched source: {unmatched_src}, Unmatched target: {unmatched_tgt}")

    if verbose:
        print(f"[ToTf.transfer_weights] Transferred {len(matched_pairs)} layers ({transferred_count:,} parameters).")
        for s_n, t_n in matched_pairs:
            print(f"  {s_n} -> {t_n}")

    return {
        "transferred_parameters": transferred_count,
        "transferred_layers": len(matched_pairs),
        "matched_pairs": matched_pairs,
        "unmatched_source": unmatched_src,
        "unmatched_target": unmatched_tgt,
    }
