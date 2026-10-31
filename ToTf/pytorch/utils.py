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
