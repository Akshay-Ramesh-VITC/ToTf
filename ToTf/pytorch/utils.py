"""
Utility functions for PyTorch - Framework-agnostic "missing" functions

Provides:
- lazy_flatten: Auto-shape flattener for transitioning from Conv to Linear layers
- loss_ncc: Normalized Cross-Correlation loss (for medical imaging)
- find_lr: Learning Rate Finder (LR range test)
- seed_everything: Cross-framework reproducible seed setter
- inspect_tensor: Diagnostic tensor statistics, memory, and health inspector
- freeze_layers / unfreeze_layers: Selective layer/parameter freezing for transfer learning
- freeze_batchnorm: Freeze BatchNorm parameters and running statistics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def lazy_flatten(tensor: torch.Tensor, start_dim: int = 1) -> torch.Tensor:
    """
    Auto-shape flattener that automatically calculates correct dimensions.
    
    Eliminates the need to manually calculate flattened sizes when transitioning
    from convolutional layers to fully connected layers.
    
    Args:
        tensor: Input tensor to flatten
        start_dim: Dimension to start flattening from (default: 1, preserves batch)
    
    Returns:
        Flattened tensor
    
    Example:
        >>> x = torch.randn(32, 16, 7, 7)  # batch=32, channels=16, H=7, W=7
        >>> x_flat = lazy_flatten(x)       # Shape: [32, 784]
        >>> # No need to calculate 16*7*7 manually!
    """
    return torch.flatten(tensor, start_dim=start_dim)


def get_flatten_size(input_shape: Tuple[int, ...]) -> int:
    """
    Calculate the flattened size given an input shape.
    
    Useful for determining the input size of the first Linear layer after Conv layers.
    
    Args:
        input_shape: Shape of the tensor (excluding batch dimension)
                    e.g., (16, 7, 7) for Conv output
    
    Returns:
        Total flattened size
    
    Example:
        >>> size = get_flatten_size((16, 7, 7))
        >>> print(size)  # 784
        >>> # Now you can use: nn.Linear(size, 128)
    """
    return np.prod(input_shape)


def loss_ncc(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
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
        >>> y_true = torch.randn(8, 1, 64, 64)
        >>> y_pred = model(x)
        >>> loss = loss_ncc(y_true, y_pred)
        >>> loss.backward()
    
    Note:
        - Returns 0 when images are identical
        - Returns 2 when images are completely anti-correlated
        - More robust than MSE for images with intensity variations
    """
    # Flatten spatial dimensions
    y_true_flat = y_true.view(y_true.size(0), -1)
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    
    # Calculate means
    mean_true = torch.mean(y_true_flat, dim=1, keepdim=True)
    mean_pred = torch.mean(y_pred_flat, dim=1, keepdim=True)
    
    # Center the tensors
    y_true_centered = y_true_flat - mean_true
    y_pred_centered = y_pred_flat - mean_pred
    
    # Calculate cross-correlation
    numerator = torch.sum(y_true_centered * y_pred_centered, dim=1)
    
    # Calculate standard deviations
    std_true = torch.sqrt(torch.sum(y_true_centered ** 2, dim=1) + eps)
    std_pred = torch.sqrt(torch.sum(y_pred_centered ** 2, dim=1) + eps)
    
    # NCC coefficient
    ncc = numerator / (std_true * std_pred + eps)
    
    # Return as loss (1 - NCC), averaged over batch
    return torch.mean(1.0 - ncc)


def ncc_score(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
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


class LRFinder:
    """
    Learning Rate Finder for PyTorch models.
    
    Implements the LR range test popularized by fast.ai to find optimal learning rates.
    Runs the model for a few iterations with exponentially increasing learning rates
    and plots the loss to identify the "knee" of the curve.
    
    Example:
        >>> lr_finder = LRFinder(model, optimizer, criterion, device='cuda')
        >>> lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=10, num_iter=100)
        >>> lr_finder.plot()
        >>> best_lr = lr_finder.get_best_lr()
        >>> print(f"Suggested LR: {best_lr}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        device: str = 'cpu'
    ):
        """
        Initialize LR Finder.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer instance (will be modified during search)
            criterion: Loss function
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Store original state
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())
        
        # Results
        self.lrs: List[float] = []
        self.losses: List[float] = []
        self.best_lr: Optional[float] = None
    
    def range_test(
        self,
        dataloader: torch.utils.data.DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_iter: int = 100,
        smooth_f: float = 0.05,
        diverge_th: float = 5.0
    ):
        """
        Perform LR range test.
        
        Args:
            dataloader: Training data loader
            start_lr: Starting learning rate (default: 1e-7)
            end_lr: Ending learning rate (default: 10.0)
            num_iter: Number of iterations to run (default: 100)
            smooth_f: Loss smoothing factor (default: 0.05)
            diverge_th: Stop if loss > diverge_th * best_loss (default: 5.0)
        """
        # Reset to original state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        
        self.model.to(self.device)
        self.model.train()
        
        # Calculate LR multiplier
        lr_mult = (end_lr / start_lr) ** (1.0 / num_iter)
        
        # Initialize
        lr = start_lr
        self._set_lr(lr)
        
        avg_loss = 0.0
        best_loss = float('inf')
        batch_iter = iter(dataloader)
        
        self.lrs = []
        self.losses = []
        
        with tqdm(range(num_iter), desc="LR Finder") as pbar:
            for iteration in pbar:
                # Get batch
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(dataloader)
                    batch = next(batch_iter)
                
                # Unpack batch
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = None
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Calculate loss
                if targets is not None:
                    loss = self.criterion(outputs, targets)
                else:
                    loss = self.criterion(outputs)
                
                # Check for nan/inf
                if torch.isnan(loss) or torch.isinf(loss):
                    break
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Smooth loss
                loss_val = loss.item()
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
                self._set_lr(lr)
        
        # Restore original state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        
        # Calculate best LR
        self.best_lr = self._calculate_best_lr()
    
    def _set_lr(self, lr: float):
        """Set learning rate for all param groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
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
        # (where loss is decreasing fastest)
        if min_loss_idx > 10:
            steepest_idx = np.argmin(gradients[:min_loss_idx])
            return lrs_array[steepest_idx]
        else:
            # Return LR at 1/10th of minimum if not enough data
            return lrs_array[min_loss_idx] / 10.0
    
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


def find_lr(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    num_iter: int = 100,
    plot: bool = True
) -> float:
    """
    Convenience function to find optimal learning rate.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        criterion: Loss function
        dataloader: Training data loader
        device: Device to run on ('cpu' or 'cuda')
        start_lr: Starting learning rate (default: 1e-7)
        end_lr: Ending learning rate (default: 10.0)
        num_iter: Number of iterations (default: 100)
        plot: Whether to show plot (default: True)
    
    Returns:
        Suggested optimal learning rate
    
    Example:
        >>> best_lr = find_lr(model, optimizer, nn.CrossEntropyLoss(), 
        ...                   train_loader, device='cuda')
        >>> print(f"Use learning rate: {best_lr}")
        >>> # Now create optimizer with best_lr
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    """
    lr_finder = LRFinder(model, optimizer, criterion, device)
    lr_finder.range_test(dataloader, start_lr, end_lr, num_iter)
    
    if plot:
        lr_finder.plot()
    
    best_lr = lr_finder.get_best_lr()
    
    if best_lr is None:
        print("Warning: Could not determine best LR. Using default 1e-3")
        return 1e-3
    
    return best_lr


def toONNX(
    model: nn.Module,
    dummy_input,
    filepath: str,
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
    opset_version: int = 11,
    dynamic_axes: Optional[dict] = None,
    verbose: bool = False,
):
    """
    Export a PyTorch `nn.Module` to ONNX format.

    Args:
        model: PyTorch model to export
        dummy_input: example input (torch.Tensor or tuple of tensors)
        filepath: output path for the .onnx file
        input_names: optional list of input names
        output_names: optional list of output names
        opset_version: ONNX opset version
        dynamic_axes: dynamic axes mapping for inputs/outputs
        verbose: pass verbose flag to torch.onnx.export

    Returns:
        The filepath where the ONNX model was saved.
    """
    model_cpu = model
    was_training = model_cpu.training
    model_cpu.eval()

    # Default names
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    # Export
    torch.onnx.export(
        model_cpu,
        dummy_input,
        filepath,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
    )

    # restore training state
    model_cpu.train(was_training)
    return filepath


def fromONNX(onnx_path: str, inputs, providers: Optional[list] = None):
    """
    Run inference on an ONNX model using onnxruntime.

    Args:
        onnx_path: path to the ONNX model file
        inputs: single input (tensor/ndarray) or dict mapping input names to data
        providers: optional list of ONNXRuntime providers

    Returns:
        List of numpy outputs from the ONNX model (or single ndarray if one output)
    """
    sess_options = ort.SessionOptions()
    if providers is None:
        # default providers order
        providers = None

    # Use only available providers to avoid noisy warnings when optional providers
    # are not installed in the environment.
    available = ort.get_available_providers()
    sess = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=(providers if providers is not None else available) or None,
    )

    # Build input dict
    input_map = {}
    ort_inputs = sess.get_inputs()
    if isinstance(inputs, dict):
        for k, v in inputs.items():
            if hasattr(v, 'cpu'):
                v = v.cpu().detach().numpy()
            input_map[k] = np.asarray(v)
    else:
        # single input -> map to first input name
        v = inputs
        if hasattr(v, 'cpu'):
            v = v.cpu().detach().numpy()
        input_map[ort_inputs[0].name] = np.asarray(v)

    out = sess.run(None, input_map)
    if len(out) == 1:
        return out[0]
    return out


def seed_everything(
    seed: int = 42,
    deterministic: bool = False,
    benchmark: bool = False,
    verbose: bool = False,
) -> int:
    """
    Sets seeds across Python, NumPy, PyTorch (CPU & CUDA), and TensorFlow for reproducibility.

    Args:
        seed: The integer seed value (default: 42).
        deterministic: If True, instructs PyTorch to use deterministic algorithms and sets
                       cuBLAS workspace determinism.
        benchmark: If True and deterministic is False, sets torch.backends.cudnn.benchmark = True.
        verbose: If True, prints which backends were seeded.

    Returns:
        The integer seed used.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    seeded = ["python", "numpy"]

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

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        if deterministic:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                pass
        seeded.append("tensorflow")
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
    Inspects a tensor (PyTorch, TensorFlow, or NumPy) and provides a comprehensive diagnostic
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

    if isinstance(tensor, torch.Tensor):
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
        # Check TensorFlow if available
        is_tf = False
        try:
            import tensorflow as tf
            if isinstance(tensor, (tf.Tensor, tf.Variable)):
                is_tf = True
        except ImportError:
            pass

        if is_tf:
            framework = "tensorflow"
            shape = tuple(tensor.shape.as_list()) if hasattr(tensor.shape, "as_list") else tuple(tensor.shape)
            dtype_str = tensor.dtype.name
            device_str = getattr(tensor, "device", "cpu") or "cpu"
            try:
                arr = tensor.numpy()
            except Exception:
                arr = np.asarray(tensor)
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
    model: nn.Module,
    up_to: Optional[Union[str, int]] = None,
    layer_names: Optional[List[str]] = None,
    layer_types: Optional[Tuple[type, ...]] = None,
    regex: Optional[str] = None,
) -> int:
    """
    Freeze parameters in a PyTorch model for transfer learning or fine-tuning.

    Args:
        model: PyTorch nn.Module.
        up_to: Either an int (freeze the first N top-level child modules)
               or a str (freeze all child modules up to and including the module named `up_to`).
        layer_names: List of module or parameter name substrings to freeze.
        layer_types: Tuple of layer types to freeze (e.g. (nn.Conv2d, nn.BatchNorm2d)).
        regex: Regular expression string matched against module or parameter names.

    Returns:
        Total number of individual parameters (elements) frozen.
    """
    compiled_regex = re.compile(regex) if regex else None
    params_to_freeze = set()

    # 1. up_to
    if up_to is not None:
        children = list(model.named_children())
        if isinstance(up_to, int):
            target_children = children[:up_to]
            for _, child in target_children:
                for p in child.parameters():
                    params_to_freeze.add(p)
        elif isinstance(up_to, str):
            child_names = [name for name, _ in children]
            if up_to in child_names:
                for name, child in children:
                    for p in child.parameters():
                        params_to_freeze.add(p)
                    if name == up_to:
                        break
            else:
                for name, mod in model.named_modules():
                    if mod is model:
                        continue
                    for p in mod.parameters(recurse=False):
                        params_to_freeze.add(p)
                    if name == up_to or name.endswith(f".{up_to}"):
                        break

    # 2. layer_types
    if layer_types is not None:
        for _, mod in model.named_modules():
            if isinstance(mod, layer_types):
                for p in mod.parameters():
                    params_to_freeze.add(p)

    # 3. layer_names
    if layer_names is not None:
        for name, mod in model.named_modules():
            if any(target in name for target in layer_names):
                for p in mod.parameters():
                    params_to_freeze.add(p)
        for name, p in model.named_parameters():
            if any(target in name for target in layer_names):
                params_to_freeze.add(p)

    # 4. regex
    if compiled_regex is not None:
        for name, mod in model.named_modules():
            if compiled_regex.search(name):
                for p in mod.parameters():
                    params_to_freeze.add(p)
        for name, p in model.named_parameters():
            if compiled_regex.search(name):
                params_to_freeze.add(p)

    # 5. Default: If no filters provided, freeze all parameters
    if up_to is None and layer_names is None and layer_types is None and regex is None:
        for p in model.parameters():
            params_to_freeze.add(p)

    frozen_count = 0
    for p in params_to_freeze:
        if p.requires_grad:
            p.requires_grad = False
            frozen_count += p.numel()

    return frozen_count


def unfreeze_layers(
    model: nn.Module,
    last_n: Optional[int] = None,
    layer_names: Optional[List[str]] = None,
    layer_types: Optional[Tuple[type, ...]] = None,
    regex: Optional[str] = None,
) -> int:
    """
    Unfreeze parameters in a PyTorch model.

    Args:
        model: PyTorch nn.Module.
        last_n: If provided, unfreezes the last N child modules (e.g. classification head).
        layer_names: List of module or parameter name substrings to unfreeze.
        layer_types: Tuple of layer types to unfreeze.
        regex: Regular expression string matched against module or parameter names.

    Returns:
        Total number of individual parameters (elements) unfrozen.
    """
    compiled_regex = re.compile(regex) if regex else None
    params_to_unfreeze = set()

    if last_n is not None:
        children = list(model.named_children())
        target_children = children[-last_n:] if last_n > 0 else []
        for _, child in target_children:
            for p in child.parameters():
                params_to_unfreeze.add(p)

    if layer_types is not None:
        for _, mod in model.named_modules():
            if isinstance(mod, layer_types):
                for p in mod.parameters():
                    params_to_unfreeze.add(p)

    if layer_names is not None:
        for name, mod in model.named_modules():
            if any(target in name for target in layer_names):
                for p in mod.parameters():
                    params_to_unfreeze.add(p)
        for name, p in model.named_parameters():
            if any(target in name for target in layer_names):
                params_to_unfreeze.add(p)

    if compiled_regex is not None:
        for name, mod in model.named_modules():
            if compiled_regex.search(name):
                for p in mod.parameters():
                    params_to_unfreeze.add(p)
        for name, p in model.named_parameters():
            if compiled_regex.search(name):
                params_to_unfreeze.add(p)

    if last_n is None and layer_names is None and layer_types is None and regex is None:
        for p in model.parameters():
            params_to_unfreeze.add(p)

    unfrozen_count = 0
    for p in params_to_unfreeze:
        if not p.requires_grad:
            p.requires_grad = True
            unfrozen_count += p.numel()

    return unfrozen_count


def freeze_batchnorm(
    model: nn.Module,
    freeze_eval: bool = True,
) -> int:
    """
    Freeze all BatchNorm layers in a PyTorch model.
    Freezes affine weights and biases. If freeze_eval is True, keeps BatchNorm
    modules in eval() mode even if model.train() is called.

    Args:
        model: PyTorch nn.Module.
        freeze_eval: If True, attaches a forward pre-hook ensuring BatchNorm stays in eval mode.

    Returns:
        Number of BatchNorm modules frozen.
    """
    bn_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )
    count = 0
    for m in model.modules():
        if isinstance(m, bn_types):
            if hasattr(m, "weight") and m.weight is not None:
                m.weight.requires_grad = False
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = False
            if freeze_eval:
                m.eval()
                if not getattr(m, "_totf_bn_hook_registered", False):
                    def _keep_eval(module, input):
                        module.eval()
                        return None

                    m.register_forward_pre_hook(_keep_eval)
                    m._totf_bn_hook_registered = True
            count += 1
    return count


# =========================================================================
# Phase 2: Domain Losses, Metrics & Anomaly Diagnostics
# =========================================================================

def loss_focal(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
    from_logits: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Focal Loss for addressing extreme foreground-background class imbalance (Lin et al., ICCV 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Supports binary/multilabel classification and multiclass targets.

    Args:
        y_true: Ground truth tensor (same shape as y_pred for binary/multilabel, or 1D class indices).
        y_pred: Predicted probabilities or unnormalized logits.
        alpha: Weighting factor in [0, 1] for rare class (default: 0.25).
        gamma: Focusing parameter >= 0 (default: 2.0).
        reduction: 'mean', 'sum', or 'none'.
        from_logits: Whether y_pred contains raw logits or probabilities.
        eps: Small constant for numerical stability.

    Returns:
        Focal loss tensor.
    """
    try:
        import tensorflow as tf
        if isinstance(y_true, (tf.Tensor, tf.Variable)) or isinstance(y_pred, (tf.Tensor, tf.Variable)):
            from ToTf.tenf.utils import loss_focal as tf_loss_focal
            return tf_loss_focal(y_true, y_pred, alpha=alpha, gamma=gamma, reduction=reduction, from_logits=from_logits, eps=eps)
    except ImportError:
        pass

    # Multiclass case: y_true is 1D class indices and y_pred is (N, C) or (*, C)
    if y_true.dim() == y_pred.dim() - 1:
        if from_logits:
            log_p = F.log_softmax(y_pred, dim=-1)
            p = torch.exp(log_p)
        else:
            p = torch.clamp(y_pred, eps, 1.0)
            log_p = torch.log(p)

        target = y_true.long().unsqueeze(-1)
        p_t = p.gather(-1, target).squeeze(-1)
        log_p_t = log_p.gather(-1, target).squeeze(-1)
        loss = -alpha * ((1.0 - p_t) ** gamma) * log_p_t

    else:
        # Binary or multilabel case: shapes match
        if from_logits:
            prob = torch.sigmoid(y_pred)
            bce = F.binary_cross_entropy_with_logits(y_pred, y_true.float(), reduction="none")
        else:
            prob = torch.clamp(y_pred, eps, 1.0 - eps)
            y_t = y_true.float()
            bce = -(y_t * torch.log(prob + eps) + (1.0 - y_t) * torch.log(1.0 - prob + eps))

        y_true_f = y_true.float()
        p_t = prob * y_true_f + (1.0 - prob) * (1.0 - y_true_f)
        alpha_factor = alpha * y_true_f + (1.0 - alpha) * (1.0 - y_true_f)
        loss = alpha_factor * ((1.0 - p_t) ** gamma) * bce

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class FocalLoss(nn.Module):
    """
    PyTorch nn.Module wrapper for Focal Loss.
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        from_logits: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, arg1: Any, arg2: Any) -> Any:
        try:
            import tensorflow as tf
            if isinstance(arg1, (tf.Tensor, tf.Variable)) or isinstance(arg2, (tf.Tensor, tf.Variable)):
                from ToTf.tenf.utils import loss_focal as tf_loss_focal
                return tf_loss_focal(arg1, arg2, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction, from_logits=self.from_logits)
        except ImportError:
            pass
        return loss_focal(
            y_true=arg2,
            y_pred=arg1,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
            from_logits=self.from_logits,
        )


def loss_dice(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    smooth: float = 1e-6,
    from_logits: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Dice Loss for image and volume segmentation.

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
        import tensorflow as tf
        if isinstance(y_true, (tf.Tensor, tf.Variable)) or isinstance(y_pred, (tf.Tensor, tf.Variable)):
            from ToTf.tenf.utils import loss_dice as tf_loss_dice
            return tf_loss_dice(y_true, y_pred, smooth=smooth, from_logits=from_logits, reduction=reduction)
    except ImportError:
        pass

    if from_logits:
        y_pred = torch.sigmoid(y_pred)

    y_true_f = y_true.float()
    batch_size = y_pred.shape[0]
    pred_flat = y_pred.reshape(batch_size, -1)
    true_flat = y_true_f.reshape(batch_size, -1)

    intersection = torch.sum(pred_flat * true_flat, dim=1)
    cardinality = torch.sum(pred_flat, dim=1) + torch.sum(true_flat, dim=1)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    loss = 1.0 - dice

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class DiceLoss(nn.Module):
    """
    PyTorch nn.Module wrapper for Dice Loss.
    """
    def __init__(self, smooth: float = 1e-6, from_logits: bool = False, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, arg1: Any, arg2: Any) -> Any:
        try:
            import tensorflow as tf
            if isinstance(arg1, (tf.Tensor, tf.Variable)) or isinstance(arg2, (tf.Tensor, tf.Variable)):
                from ToTf.tenf.utils import loss_dice as tf_loss_dice
                return tf_loss_dice(arg1, arg2, smooth=self.smooth, from_logits=self.from_logits, reduction=self.reduction)
        except ImportError:
            pass
        return loss_dice(
            y_true=arg2,
            y_pred=arg1,
            smooth=self.smooth,
            from_logits=self.from_logits,
            reduction=self.reduction,
        )


def dice_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: Optional[float] = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Dice Similarity Coefficient (DSC) evaluation metric.
    """
    try:
        import tensorflow as tf
        if isinstance(y_true, (tf.Tensor, tf.Variable)) or isinstance(y_pred, (tf.Tensor, tf.Variable)):
            from ToTf.tenf.utils import dice_score as tf_dice_score
            return tf_dice_score(y_true, y_pred, threshold=threshold, smooth=smooth)
    except ImportError:
        pass

    y_true_f = y_true.float()
    if threshold is not None:
        y_pred_bin = (y_pred >= threshold).float()
    else:
        y_pred_bin = y_pred.float()

    batch_size = y_pred_bin.shape[0]
    pred_flat = y_pred_bin.reshape(batch_size, -1)
    true_flat = y_true_f.reshape(batch_size, -1)

    intersection = torch.sum(pred_flat * true_flat, dim=1)
    cardinality = torch.sum(pred_flat, dim=1) + torch.sum(true_flat, dim=1)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return dice.mean()


def iou_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: Optional[float] = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Intersection over Union (IoU / Jaccard Index) evaluation metric.
    """
    try:
        import tensorflow as tf
        if isinstance(y_true, (tf.Tensor, tf.Variable)) or isinstance(y_pred, (tf.Tensor, tf.Variable)):
            from ToTf.tenf.utils import iou_score as tf_iou_score
            return tf_iou_score(y_true, y_pred, threshold=threshold, smooth=smooth)
    except ImportError:
        pass

    y_true_f = y_true.float()
    if threshold is not None:
        y_pred_bin = (y_pred >= threshold).float()
    else:
        y_pred_bin = y_pred.float()

    batch_size = y_pred_bin.shape[0]
    pred_flat = y_pred_bin.reshape(batch_size, -1)
    true_flat = y_true_f.reshape(batch_size, -1)

    intersection = torch.sum(pred_flat * true_flat, dim=1)
    union = torch.sum(pred_flat, dim=1) + torch.sum(true_flat, dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


class AnomalyDetector:
    """
    Context manager and monitor for detecting NaNs and Infs during forward and backward passes.
    """
    def __init__(self, model: nn.Module, raise_error: bool = False, track_backward: bool = True):
        self.model = model
        self.raise_error = raise_error
        self.track_backward = track_backward
        self.anomalies: List[Dict[str, Any]] = []
        self._hooks: List[Any] = []
        self.is_active = False

    def attach(self):
        if self.is_active:
            return self
        self.anomalies = []
        for name, module in self.model.named_modules():
            if module is self.model:
                continue

            def make_fwd_hook(mod_name):
                def hook(mod, inputs, output):
                    for idx, inp in enumerate(inputs):
                        if isinstance(inp, torch.Tensor):
                            nan_c = int(torch.isnan(inp).sum().item())
                            inf_c = int(torch.isinf(inp).sum().item())
                            if nan_c > 0 or inf_c > 0:
                                rec = {
                                    "stage": "forward_input",
                                    "layer": mod_name,
                                    "type": type(mod).__name__,
                                    "arg_idx": idx,
                                    "nan_count": nan_c,
                                    "inf_count": inf_c,
                                    "shape": tuple(inp.shape),
                                }
                                self.anomalies.append(rec)
                                if self.raise_error:
                                    raise FloatingPointError(f"NaN/Inf detected in {mod_name} forward input: {rec}")

                    outs = [output] if isinstance(output, torch.Tensor) else (output if isinstance(output, (list, tuple)) else [])
                    for idx, out in enumerate(outs):
                        if isinstance(out, torch.Tensor):
                            nan_c = int(torch.isnan(out).sum().item())
                            inf_c = int(torch.isinf(out).sum().item())
                            if nan_c > 0 or inf_c > 0:
                                rec = {
                                    "stage": "forward_output",
                                    "layer": mod_name,
                                    "type": type(mod).__name__,
                                    "arg_idx": idx,
                                    "nan_count": nan_c,
                                    "inf_count": inf_c,
                                    "shape": tuple(out.shape),
                                }
                                self.anomalies.append(rec)
                                if self.raise_error:
                                    raise FloatingPointError(f"NaN/Inf detected in {mod_name} forward output: {rec}")
                    return None
                return hook

            h1 = module.register_forward_hook(make_fwd_hook(name))
            self._hooks.append(h1)

            if self.track_backward:
                def make_bwd_hook(mod_name):
                    def bwd_hook(mod, grad_input, grad_output):
                        for idx, g in enumerate(grad_output):
                            if isinstance(g, torch.Tensor):
                                nan_c = int(torch.isnan(g).sum().item())
                                inf_c = int(torch.isinf(g).sum().item())
                                if nan_c > 0 or inf_c > 0:
                                    rec = {
                                        "stage": "backward_grad_output",
                                        "layer": mod_name,
                                        "type": type(mod).__name__,
                                        "arg_idx": idx,
                                        "nan_count": nan_c,
                                        "inf_count": inf_c,
                                        "shape": tuple(g.shape),
                                    }
                                    self.anomalies.append(rec)
                                    if self.raise_error:
                                        raise FloatingPointError(f"NaN/Inf detected in {mod_name} gradient: {rec}")
                        return None
                    return bwd_hook

                try:
                    h2 = module.register_full_backward_hook(make_bwd_hook(name))
                    self._hooks.append(h2)
                except Exception:
                    pass

        self.is_active = True
        return self

    def detach(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
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


def check_anomalies(model: nn.Module, sample_input: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Runs a forward pass through model and returns a list of detected NaN/Inf anomalies.
    """
    with AnomalyDetector(model, raise_error=False, track_backward=False) as detector:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(sample_input)
        model.train(was_training)
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
    Comprehensive performance and complexity benchmark for PyTorch and Keras models.

    Measures:
    - Latency (mean, std, median/p50, p90, p95, p99)
    - Throughput (FPS)
    - Total, Trainable, and Non-trainable parameter counts
    - Estimated MACs and FLOPs
    - Model parameter memory footprint
    - Publication-ready LaTeX and Markdown comparison tables

    Args:
        model: PyTorch nn.Module or TensorFlow keras.Model.
        input_size: Shape of single input excluding batch dimension, e.g. (3, 224, 224).
        input_data: Example input tensor/array (if provided, input_size is ignored).
        device: Device to benchmark on ('cpu' or 'cuda').
        warmup: Number of warmup iterations.
        repetition: Number of measured inference iterations.
        batch_size: Batch size for input generation (default: 1).
        print_table: Whether to print formatted ASCII benchmark table.

    Returns:
        Dictionary of all measured metrics and formatted tables.
    """
    is_torch_model = isinstance(model, nn.Module)
    is_tf_model = False
    try:
        import tensorflow as tf
        if isinstance(model, (tf.keras.Model, tf.keras.layers.Layer)):
            is_tf_model = True
    except ImportError:
        pass

    if not is_torch_model and not is_tf_model:
        raise TypeError(f"Unsupported model type: {type(model)}")

    # 1. Parameter counting
    total_params = 0
    trainable_params = 0
    param_bytes = 0
    estimated_macs = 0

    if is_torch_model:
        for p in model.parameters():
            total_params += p.numel()
            param_bytes += p.numel() * p.element_size()
            if p.requires_grad:
                trainable_params += p.numel()
        non_trainable_params = total_params - trainable_params
        param_memory_mb = param_bytes / (1024 * 1024)

        # Estimate MACs
        for m in model.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                estimated_macs += m.weight.numel() * 14 * 14
            elif isinstance(m, nn.Linear):
                estimated_macs += m.weight.numel()
    else:
        # Keras model
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

    estimated_flops = estimated_macs * 2

    # 2. Prepare sample input & measure
    target_device = device
    if is_torch_model:
        if target_device is None:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        model_device = torch.device(target_device)
        model = model.to(model_device)
        was_train = model.training
        model.eval()

        if input_data is not None:
            if isinstance(input_data, torch.Tensor):
                x = input_data.to(model_device)
            else:
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

        latencies: List[float] = []
        with torch.no_grad():
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

    else:
        # TensorFlow model
        import tensorflow as tf
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

    # 3. Compute statistics
    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    std_lat = float(np.std(lat_arr))
    median_lat = float(np.median(lat_arr))
    p90_lat = float(np.percentile(lat_arr, 90))
    p95_lat = float(np.percentile(lat_arr, 95))
    p99_lat = float(np.percentile(lat_arr, 99))
    fps = float((batch_size * 1000.0) / mean_lat) if mean_lat > 0 else 0.0

    # 4. Format tables
    model_name = getattr(model, "__name__", None) or type(model).__name__

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
    Direct in-memory weight transfer between PyTorch and TensorFlow / Keras models.

    Handles framework-specific layout transformations automatically:
    - Conv2D weights: PyTorch [Out, In, H, W] <-> TensorFlow [H, W, In, Out]
    - Linear/Dense weights: PyTorch [Out, In] <-> TensorFlow [In, Out]
    - BatchNorm: PyTorch (weight, bias, running_mean, running_var) <->
                 TensorFlow (gamma, beta, moving_mean, moving_variance)

    Supports bidirectional transfer:
    - PyTorch -> TensorFlow
    - TensorFlow -> PyTorch
    - PyTorch -> PyTorch / TensorFlow -> TensorFlow

    Args:
        source_model: Source model instance (torch.nn.Module or tf.keras.Model).
        target_model: Target model instance to update in-place.
        name_mapping: Optional mapping of source layer/param names to target layer names.
        strict: If True, raises ValueError if any layer cannot be matched.
        verbose: If True, prints transfer log.

    Returns:
        Dictionary with transfer summary, count of transferred parameters, and mapping pairs.
    """
    is_source_torch = isinstance(source_model, nn.Module)
    is_target_torch = isinstance(target_model, nn.Module)

    # 1. Collect parameterized stages from source
    source_items = []
    if is_source_torch:
        for name, mod in source_model.named_modules():
            if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                source_items.append({
                    "name": name,
                    "type": "conv",
                    "weight": mod.weight.detach().cpu().numpy(),
                    "bias": mod.bias.detach().cpu().numpy() if mod.bias is not None else None,
                })
            elif isinstance(mod, nn.Linear):
                source_items.append({
                    "name": name,
                    "type": "linear",
                    "weight": mod.weight.detach().cpu().numpy(),
                    "bias": mod.bias.detach().cpu().numpy() if mod.bias is not None else None,
                })
            elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
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

    # 2. Collect parameterized stages from target
    target_items = []
    if is_target_torch:
        for name, mod in target_model.named_modules():
            if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                target_items.append({"name": name, "type": "conv", "obj": mod})
            elif isinstance(mod, nn.Linear):
                target_items.append({"name": name, "type": "linear", "obj": mod})
            elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
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

    # Map pairs
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

    # Perform transfer
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
            # Same framework
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

