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
