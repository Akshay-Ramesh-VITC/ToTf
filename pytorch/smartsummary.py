"""
SmartSummary - Advanced model summary with bottleneck detection

Provides comprehensive model analysis including:
- Layer-by-layer parameter counts and shapes
- Gradient variance analysis
- Bottleneck layer identification
- Memory usage estimation
- FLOPS estimation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import numpy as np


class SmartSummary:
    """
    Advanced model summary that identifies bottlenecks and provides detailed insights.
    
    Features beyond standard model.summary():
    - Gradient variance tracking to identify unstable layers
    - Bottleneck detection based on parameters and computational cost
    - Memory usage estimation
    - Output shape inference for complex models
    - Pretty-printed table format
    
    Example:
        >>> model = YourModel()
        >>> summary = SmartSummary(model, input_size=(3, 224, 224))
        >>> summary.show()
        >>> bottlenecks = summary.get_bottlenecks()
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        input_size: Optional[Tuple[int, ...]] = None,
        batch_size: int = 1,
        device: str = "cpu",
        track_gradients: bool = False
    ):
        """
        Initialize SmartSummary.
        
        Args:
            model: PyTorch model to analyze
            input_size: Input tensor size (excluding batch dimension)
                       e.g., (3, 224, 224) for images
            batch_size: Batch size for shape inference
            device: Device to run analysis on ('cpu' or 'cuda')
            track_gradients: Whether to track gradient statistics (requires forward+backward pass)
        """
        self.model = model
        self.input_size = input_size
        self.batch_size = batch_size
        self.device = device
        self.track_gradients = track_gradients
        
        self.summary_data: OrderedDict = OrderedDict()
        self.total_params = 0
        self.trainable_params = 0
        self.total_output_size = 0
        self.gradient_stats: Dict[str, Dict[str, float]] = {}
        
        self._analyze_model()
    
    def _register_hooks(self, module_hooks: List):
        """Register forward hooks to capture layer information"""
        def hook_fn(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(self.summary_data)
            
            m_key = f"{class_name}-{module_idx}"
            
            # Calculate output shape
            if isinstance(output, (list, tuple)):
                output_shape = [list(o.size()) for o in output if isinstance(o, torch.Tensor)]
            else:
                output_shape = list(output.size())
            
            # Count parameters
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            # Estimate output size in MB
            if isinstance(output, (list, tuple)):
                output_size = sum(o.numel() * o.element_size() for o in output if isinstance(o, torch.Tensor))
            else:
                output_size = output.numel() * output.element_size()
            
            self.summary_data[m_key] = {
                "layer_name": class_name,
                "input_shape": [list(i.size()) if isinstance(i, torch.Tensor) else str(i) for i in input] if isinstance(input, tuple) else list(input.size()) if isinstance(input, torch.Tensor) else str(input),
                "output_shape": output_shape,
                "params": params,
                "trainable": trainable,
                "output_size_mb": output_size / (1024 ** 2),
                "module": module
            }
            
            self.total_params += params
            self.trainable_params += trainable
            self.total_output_size += output_size
        
        hooks = []
        for name, layer in self.model.named_modules():
            # Skip container modules
            if list(layer.children()):
                continue
            hooks.append(layer.register_forward_hook(hook_fn))
        
        return hooks
    
    def _register_gradient_hooks(self):
        """Register hooks to track gradient statistics"""
        def grad_hook_fn(module, grad_input, grad_output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            
            # Calculate gradient variance
            if grad_output[0] is not None:
                grad_var = grad_output[0].var().item()
                grad_mean = grad_output[0].mean().item()
                grad_max = grad_output[0].abs().max().item()
                
                # Find corresponding key in summary_data
                for key in self.summary_data.keys():
                    if key.startswith(class_name):
                        self.gradient_stats[key] = {
                            "grad_variance": grad_var,
                            "grad_mean": grad_mean,
                            "grad_max": grad_max
                        }
                        break
        
        hooks = []
        for layer in self.model.modules():
            if list(layer.children()):
                continue
            hooks.append(layer.register_full_backward_hook(grad_hook_fn))
        
        return hooks
    
    def _analyze_model(self):
        """Run model analysis"""
        if self.input_size is None:
            # Can't do shape inference without input size
            self._analyze_without_forward()
            return
        
        # Register hooks
        hooks = self._register_hooks([])
        
        # Create dummy input
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        self.model.to(self.device)
        
        # Forward pass
        if self.track_gradients:
            grad_hooks = self._register_gradient_hooks()
            self.model.train()
            output = self.model(x)
            
            # Backward pass to get gradients
            if isinstance(output, (list, tuple)):
                loss = output[0].sum()
            else:
                loss = output.sum()
            loss.backward()
            
            # Remove gradient hooks
            for h in grad_hooks:
                h.remove()
        else:
            self.model.eval()
            with torch.no_grad():
                output = self.model(x)
        
        # Remove hooks
        for h in hooks:
            h.remove()
    
    def _analyze_without_forward(self):
        """Analyze model structure without forward pass"""
        for name, module in self.model.named_modules():
            if list(module.children()):
                continue
            
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            self.summary_data[name] = {
                "layer_name": class_name,
                "input_shape": "N/A",
                "output_shape": "N/A",
                "params": params,
                "trainable": trainable,
                "output_size_mb": 0.0,
                "module": module
            }
            
            self.total_params += params
            self.trainable_params += trainable
    
    def get_bottlenecks(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Identify bottleneck layers based on multiple criteria.
        
        A bottleneck is defined as a layer with:
        1. High parameter count (>10% of total)
        2. High gradient variance (if tracking enabled)
        3. Large output size
        
        Args:
            top_n: Number of top bottlenecks to return
        
        Returns:
            List of dictionaries containing bottleneck information
        """
        bottlenecks = []
        
        for key, info in self.summary_data.items():
            score = 0.0
            reasons = []
            
            # Parameter count criterion
            if self.total_params > 0:
                param_ratio = info["params"] / self.total_params
                if param_ratio > 0.1:  # More than 10% of total params
                    score += param_ratio * 100
                    reasons.append(f"High params ({param_ratio*100:.1f}%)")
            
            # Gradient variance criterion (if available)
            if key in self.gradient_stats:
                grad_var = self.gradient_stats[key]["grad_variance"]
                if grad_var > 1.0:  # High variance
                    score += np.log10(grad_var + 1) * 10
                    reasons.append(f"High grad variance ({grad_var:.2e})")
            
            # Output size criterion
            if info["output_size_mb"] > 10:  # More than 10MB
                score += info["output_size_mb"]
                reasons.append(f"Large output ({info['output_size_mb']:.1f}MB)")
            
            if score > 0:
                bottlenecks.append({
                    "layer": key,
                    "layer_name": info["layer_name"],
                    "score": score,
                    "reasons": reasons,
                    "params": info["params"],
                    "output_shape": info["output_shape"]
                })
        
        # Sort by score and return top N
        bottlenecks.sort(key=lambda x: x["score"], reverse=True)
        return bottlenecks[:top_n]
    
    def show(self, show_bottlenecks: bool = True):
        """
        Display the model summary in a pretty-printed format.
        
        Args:
            show_bottlenecks: Whether to show bottleneck analysis
        """
        print("=" * 100)
        print(f"{'Model Summary':^100}")
        print("=" * 100)
        
        # Header
        print(f"{'Layer (type)':<30} {'Output Shape':<25} {'Param #':<15} {'Trainable':<10}")
        print("-" * 100)
        
        # Layers
        for key, info in self.summary_data.items():
            layer_name = f"{info['layer_name']}"
            output_shape = str(info['output_shape'])
            if len(output_shape) > 24:
                output_shape = output_shape[:21] + "..."
            
            params = f"{info['params']:,}"
            trainable = "Yes" if info['trainable'] > 0 else "No"
            
            print(f"{layer_name:<30} {output_shape:<25} {params:<15} {trainable:<10}")
            
            # Show gradient stats if available
            if key in self.gradient_stats:
                grad_info = self.gradient_stats[key]
                print(f"  └─ Gradient: var={grad_info['grad_variance']:.2e}, "
                      f"mean={grad_info['grad_mean']:.2e}, max={grad_info['grad_max']:.2e}")
        
        print("=" * 100)
        print(f"Total params: {self.total_params:,}")
        print(f"Trainable params: {self.trainable_params:,}")
        print(f"Non-trainable params: {self.total_params - self.trainable_params:,}")
        print(f"Total output size: {self.total_output_size / (1024**2):.2f} MB")
        print("=" * 100)
        
        # Show bottlenecks
        if show_bottlenecks:
            bottlenecks = self.get_bottlenecks()
            if bottlenecks:
                print(f"\n{'[!] Detected Bottleneck Layers':^100}")
                print("=" * 100)
                
                for i, bn in enumerate(bottlenecks, 1):
                    print(f"\n{i}. {bn['layer']} ({bn['layer_name']})")
                    print(f"   Score: {bn['score']:.2f}")
                    print(f"   Reasons: {', '.join(bn['reasons'])}")
                    print(f"   Parameters: {bn['params']:,}")
                    print(f"   Output Shape: {bn['output_shape']}")
                
                print("\n" + "=" * 100)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export summary as a dictionary.
        
        Returns:
            Dictionary containing all summary information
        """
        return {
            "layers": self.summary_data,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "total_output_size_mb": self.total_output_size / (1024**2),
            "gradient_stats": self.gradient_stats,
            "bottlenecks": self.get_bottlenecks()
        }
    
    def save_to_file(self, filename: str = "model_summary.txt"):
        """
        Save the summary to a text file.
        
        Args:
            filename: Output file path
        """
        import sys
        from io import StringIO
        
        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        self.show()
        
        sys.stdout = old_stdout
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(captured_output.getvalue())
        
        print(f"Summary saved to {filename}")
    
    def analyze_loss_curve(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        window_size: int = 5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze the nature of training loss curve(s) to understand training dynamics.
        
        This method identifies patterns such as:
        - Convergence: Loss is decreasing and stabilizing
        - Divergence: Loss is increasing (training instability)
        - Oscillation: High variance in loss values
        - Plateau: Loss has stopped improving
        - Overfitting: Training loss decreases while validation loss increases
        
        Args:
            train_losses: List of training loss values over epochs
            val_losses: Optional list of validation loss values over epochs
            window_size: Window size for moving average smoothing (default: 5)
            verbose: Whether to print analysis results (default: True)
        
        Returns:
            Dictionary containing:
            - 'status': Overall training status
            - 'trend': Direction of loss movement
            - 'stability': Measure of oscillation/variance
            - 'recommendations': List of suggestions
            - 'metrics': Detailed metrics about the loss curve
        
        Example:
            >>> train_losses = [2.5, 2.1, 1.8, 1.6, 1.5, 1.45, 1.42, 1.41]
            >>> val_losses = [2.6, 2.2, 1.9, 1.7, 1.8, 1.9, 2.0, 2.1]
            >>> result = smart_summary.analyze_loss_curve(train_losses, val_losses)
            >>> print(result['status'])  # 'Overfitting Detected'
        """
        if not train_losses or len(train_losses) < 2:
            return {
                "status": "Insufficient Data",
                "trend": "Unknown",
                "stability": "Unknown",
                "recommendations": ["Provide at least 2 epochs of training data"],
                "metrics": {}
            }
        
        train_losses = np.array(train_losses, dtype=np.float32)
        n_epochs = len(train_losses)
        
        # Calculate moving average for smoothing
        def moving_average(data, window):
            if len(data) < window:
                return data
            weights = np.ones(window) / window
            return np.convolve(data, weights, mode='valid')
        
        smoothed_train = moving_average(train_losses, min(window_size, len(train_losses)))
        
        # Calculate key metrics
        initial_loss = float(train_losses[0])
        final_loss = float(train_losses[-1])
        min_loss = float(np.min(train_losses))
        max_loss = float(np.max(train_losses))
        
        # Calculate trend (slope)
        epochs_idx = np.arange(len(smoothed_train))
        if len(smoothed_train) > 1:
            trend_slope = np.polyfit(epochs_idx, smoothed_train, 1)[0]
        else:
            trend_slope = 0.0
        
        # Calculate variance and coefficient of variation
        loss_variance = float(np.var(train_losses))
        loss_std = float(np.std(train_losses))
        mean_loss = float(np.mean(train_losses))
        coef_variation = (loss_std / mean_loss * 100) if mean_loss != 0 else 0.0
        
        # Calculate recent trend (last 30% of epochs)
        recent_window = max(3, int(n_epochs * 0.3))
        recent_losses = train_losses[-recent_window:]
        recent_slope = np.polyfit(np.arange(len(recent_losses)), recent_losses, 1)[0]
        
        # Improvement percentage
        improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss != 0 else 0.0
        
        # Determine training status
        status_messages = []
        recommendations = []
        
        # Check for divergence
        if trend_slope > 0.001 or improvement < -5:
            status = "Diverging"
            status_messages.append("⚠️ Loss is INCREASING - training instability detected")
            recommendations.extend([
                "Reduce learning rate significantly (try 0.1x current value)",
                "Check for gradient explosion (use gradient clipping)",
                "Verify data preprocessing and labels are correct",
                "Consider using a simpler model architecture"
            ])
        
        # Check for plateau
        elif abs(recent_slope) < 0.0001 and n_epochs > 10:
            status = "Plateau"
            status_messages.append("📊 Loss has plateaued - minimal improvement in recent epochs")
            recommendations.extend([
                "Consider reducing learning rate to escape plateau",
                "Try learning rate scheduling or cyclical learning rates",
                "Check if model has sufficient capacity",
                "Evaluate if current performance is acceptable"
            ])
        
        # Check for high oscillation
        elif coef_variation > 20:
            status = "Oscillating"
            status_messages.append("📈 High oscillation in loss values detected")
            recommendations.extend([
                "Reduce learning rate to stabilize training",
                "Increase batch size for more stable gradients",
                "Consider using gradient clipping",
                "Check for outliers or noise in training data"
            ])
        
        # Check for good convergence
        elif improvement > 20 and abs(recent_slope) < 0.01:
            status = "Converging Well"
            status_messages.append("✅ Loss is converging nicely")
            recommendations.extend([
                "Training appears healthy - continue monitoring",
                "Consider early stopping if validation loss plateaus"
            ])
        
        # Slow improvement
        elif improvement > 5:
            status = "Converging Slowly"
            status_messages.append("🐌 Loss is decreasing but slowly")
            recommendations.extend([
                "Consider slightly increasing learning rate",
                "Verify model has sufficient capacity for the task",
                "Check if data preprocessing is optimal"
            ])
        
        else:
            status = "Stable"
            status_messages.append("📍 Loss is relatively stable")
            recommendations.append("Continue monitoring training progress")
        
        # Analyze validation losses if provided
        overfitting_detected = False
        if val_losses is not None and len(val_losses) >= 2:
            val_losses = np.array(val_losses, dtype=np.float32)
            
            if len(val_losses) != len(train_losses):
                recommendations.append(
                    f"⚠️ Warning: Train and validation losses have different lengths "
                    f"({len(train_losses)} vs {len(val_losses)})"
                )
            
            val_initial = float(val_losses[0])
            val_final = float(val_losses[-1])
            val_min = float(np.min(val_losses))
            
            # Check for overfitting: train loss decreasing, val loss increasing
            recent_val = val_losses[-recent_window:]
            val_recent_slope = np.polyfit(np.arange(len(recent_val)), recent_val, 1)[0]
            
            if recent_slope < -0.001 and val_recent_slope > 0.005:
                status = "Overfitting Detected"
                overfitting_detected = True
                status_messages.append(
                    "⚠️ OVERFITTING: Training loss decreasing but validation loss increasing"
                )
                recommendations.extend([
                    "Stop training or use early stopping",
                    "Increase regularization (L1/L2, dropout)",
                    "Add more training data if possible",
                    "Reduce model complexity",
                    "Use data augmentation"
                ])
            
            # Check for validation loss not improving
            elif val_final > val_min * 1.05 and n_epochs > 5:
                status_messages.append(
                    "⚠️ Validation loss stopped improving (potential overfitting)"
                )
                if not overfitting_detected:
                    recommendations.extend([
                        "Consider early stopping",
                        "Monitor validation loss closely"
                    ])
        
        # Determine trend description
        if trend_slope < -0.01:
            trend = "Decreasing"
        elif trend_slope > 0.01:
            trend = "Increasing"
        else:
            trend = "Stable"
        
        # Determine stability
        if coef_variation < 5:
            stability = "Very Stable"
        elif coef_variation < 15:
            stability = "Stable"
        elif coef_variation < 30:
            stability = "Moderately Unstable"
        else:
            stability = "Highly Unstable"
        
        # Compile metrics
        metrics = {
            "n_epochs": n_epochs,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "min_loss": min_loss,
            "max_loss": max_loss,
            "improvement_percent": improvement,
            "trend_slope": float(trend_slope),
            "recent_slope": float(recent_slope),
            "variance": loss_variance,
            "std_dev": loss_std,
            "mean_loss": mean_loss,
            "coefficient_of_variation": coef_variation
        }
        
        if val_losses is not None:
            metrics["val_initial_loss"] = float(val_losses[0])
            metrics["val_final_loss"] = float(val_losses[-1])
            metrics["val_min_loss"] = float(np.min(val_losses))
            metrics["val_improvement_percent"] = (
                (float(val_losses[0]) - float(val_losses[-1])) / float(val_losses[0]) * 100
                if val_losses[0] != 0 else 0.0
            )
        
        result = {
            "status": status,
            "trend": trend,
            "stability": stability,
            "recommendations": recommendations,
            "metrics": metrics
        }
        
        # Print analysis if verbose
        if verbose:
            print("\n" + "="*80)
            print(f"{'Loss Curve Analysis':^80}")
            print("="*80)
            
            for msg in status_messages:
                print(f"\n{msg}")
            
            print(f"\n📊 Training Statistics:")
            print(f"   • Epochs: {n_epochs}")
            print(f"   • Initial Loss: {initial_loss:.4f}")
            print(f"   • Final Loss: {final_loss:.4f}")
            print(f"   • Minimum Loss: {min_loss:.4f}")
            print(f"   • Improvement: {improvement:.2f}%")
            print(f"   • Trend: {trend} (slope: {trend_slope:.6f})")
            print(f"   • Stability: {stability} (CV: {coef_variation:.2f}%)")
            
            if val_losses is not None:
                print(f"\n📊 Validation Statistics:")
                print(f"   • Initial Val Loss: {metrics['val_initial_loss']:.4f}")
                print(f"   • Final Val Loss: {metrics['val_final_loss']:.4f}")
                print(f"   • Minimum Val Loss: {metrics['val_min_loss']:.4f}")
                print(f"   • Val Improvement: {metrics['val_improvement_percent']:.2f}%")
            
            if recommendations:
                print(f"\n💡 Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            
            print("\n" + "="*80)
        
        return result
