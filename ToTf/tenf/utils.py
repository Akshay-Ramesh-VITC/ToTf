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
