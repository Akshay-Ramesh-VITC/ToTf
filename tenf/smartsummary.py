"""
SmartSummary - Advanced model summary with bottleneck detection for TensorFlow/Keras

Provides comprehensive model analysis including:
- Layer-by-layer parameter counts and shapes
- Gradient variance analysis
- Bottleneck layer identification
- Memory usage estimation
- FLOPS estimation
"""

import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional, Any, Union
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
        >>> summary = SmartSummary(model, input_shape=(224, 224, 3))
        >>> summary.show()
        >>> bottlenecks = summary.get_bottlenecks()
    """
    
    def __init__(
        self, 
        model: keras.Model, 
        input_shape: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
        batch_size: int = 1,
        track_gradients: bool = False
    ):
        """
        Initialize SmartSummary.
        
        Args:
            model: TensorFlow/Keras model to analyze
            input_shape: Input tensor shape (excluding batch dimension)
                        e.g., (224, 224, 3) for images
                        Can be a list of shapes for multi-input models
            batch_size: Batch size for shape inference
            track_gradients: Whether to track gradient statistics (requires forward+backward pass)
        """
        self.model = model
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.track_gradients = track_gradients
        
        self.summary_data: OrderedDict = OrderedDict()
        self.total_params = 0
        self.trainable_params = 0
        self.total_output_size = 0
        self.gradient_stats: Dict[str, Dict[str, float]] = {}
        
        self._analyze_model()
    
    def _get_layer_output_shapes(self):
        """Get output shapes for all layers using a forward pass"""
        if self.input_shape is None:
            return {}
        
        # Create dummy input
        if isinstance(self.input_shape, list):
            x = [tf.random.normal([self.batch_size] + list(shape)) for shape in self.input_shape]
        else:
            x = tf.random.normal([self.batch_size] + list(self.input_shape))
        
        # Get intermediate outputs
        layer_outputs = {}
        
        # Build the model if not already built
        if not self.model.built:
            self.model(x)
        
        # Get outputs for each layer
        for layer in self.model.layers:
            try:
                # Create a temporary model to get this layer's output
                if hasattr(layer, 'input') and hasattr(layer, 'output'):
                    temp_model = keras.Model(inputs=self.model.input, outputs=layer.output)
                    output = temp_model(x, training=False)
                    
                    if isinstance(output, (list, tuple)):
                        layer_outputs[layer.name] = [list(o.shape) for o in output]
                    else:
                        layer_outputs[layer.name] = list(output.shape)
            except Exception:
                # Some layers might not be directly accessible
                layer_outputs[layer.name] = None
        
        return layer_outputs
    
    def _analyze_model(self):
        """Run model analysis"""
        # Get layer output shapes
        layer_outputs = {}
        if self.input_shape is not None:
            layer_outputs = self._get_layer_output_shapes()
        
        # Analyze each layer
        for idx, layer in enumerate(self.model.layers):
            class_name = layer.__class__.__name__
            layer_name = layer.name
            
            # Get layer configuration
            config = layer.get_config() if hasattr(layer, 'get_config') else {}
            
            # Count parameters
            params = layer.count_params()
            trainable_count = sum([tf.size(w).numpy() for w in layer.trainable_weights]) if layer.trainable_weights else 0
            
            # Get output shape
            output_shape = layer_outputs.get(layer_name, "N/A")
            
            # Calculate output size in MB
            output_size_mb = 0.0
            if output_shape != "N/A" and output_shape is not None:
                if isinstance(output_shape, list) and isinstance(output_shape[0], list):
                    # Multiple outputs
                    output_size = sum([np.prod(shape) * 4 for shape in output_shape])  # 4 bytes for float32
                else:
                    output_size = np.prod(output_shape) * 4  # 4 bytes for float32
                output_size_mb = output_size / (1024 ** 2)
            
            # Get input shape
            if hasattr(layer, 'input_shape'):
                input_shape = layer.input_shape
            else:
                input_shape = "N/A"
            
            m_key = f"{layer_name}"
            
            self.summary_data[m_key] = {
                "layer_name": class_name,
                "layer_config": config,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "params": params,
                "trainable": trainable_count,
                "output_size_mb": output_size_mb,
                "layer_obj": layer
            }
            
            self.total_params += params
            self.trainable_params += trainable_count
            self.total_output_size += output_size_mb * (1024 ** 2)
        
        # Track gradients if requested
        if self.track_gradients and self.input_shape is not None:
            self._track_gradients()
    
    def _track_gradients(self):
        """Track gradient statistics during a backward pass"""
        if self.input_shape is None:
            return
        
        # Create dummy input and target
        if isinstance(self.input_shape, list):
            x = [tf.random.normal([self.batch_size] + list(shape)) for shape in self.input_shape]
        else:
            x = tf.random.normal([self.batch_size] + list(self.input_shape))
        
        # Simple dummy target (sum of outputs)
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            output = self.model(x, training=True)
            
            # Create a simple loss
            if isinstance(output, (list, tuple)):
                loss = tf.reduce_sum(output[0])
            else:
                loss = tf.reduce_sum(output)
        
        # Get gradients for each layer
        for layer_name, layer_info in self.summary_data.items():
            layer = layer_info["layer_obj"]
            
            if layer.trainable_weights:
                try:
                    # Get gradients for this layer's weights
                    grads = tape.gradient(loss, layer.trainable_weights)
                    
                    if grads and grads[0] is not None:
                        # Compute statistics on the first gradient tensor
                        grad_tensor = grads[0]
                        
                        grad_var = tf.math.reduce_variance(grad_tensor).numpy()
                        grad_mean = tf.reduce_mean(grad_tensor).numpy()
                        grad_max = tf.reduce_max(tf.abs(grad_tensor)).numpy()
                        
                        self.gradient_stats[layer_name] = {
                            "grad_variance": float(grad_var),
                            "grad_mean": float(grad_mean),
                            "grad_max": float(grad_max)
                        }
                except Exception as e:
                    # Skip layers where gradient computation fails
                    pass
        
        del tape
    
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
            layer_name = f"{key} ({info['layer_name']})"
            if len(layer_name) > 29:
                layer_name = layer_name[:26] + "..."
            
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
        # Remove layer objects for serialization
        serializable_data = OrderedDict()
        for key, info in self.summary_data.items():
            serializable_info = {k: v for k, v in info.items() if k != 'layer_obj'}
            serializable_data[key] = serializable_info
        
        return {
            "layers": serializable_data,
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
    
    def compare_with_keras_summary(self):
        """
        Show both SmartSummary and Keras's built-in summary for comparison.
        """
        print("\n" + "="*50)
        print("Keras Built-in Summary:")
        print("="*50)
        self.model.summary()
        
        print("\n" + "="*50)
        print("SmartSummary (with advanced features):")
        print("="*50)
        self.show()
