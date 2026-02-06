"""
Utility functions for TensorFlow/Keras - Framework-agnostic "missing" functions

Provides:
- lazy_flatten: Auto-shape flattener for transitioning from Conv to Dense layers
- loss_ncc: Normalized Cross-Correlation loss (for medical imaging)
- find_lr: Learning Rate Finder (LR range test)
"""

import tensorflow as tf
from tensorflow import keras
from typing import Optional, Tuple, Callable, Dict, Any, List, Union
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy


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
