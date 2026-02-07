"""
ModelView - TensorFlow Model Architecture Visualization

A comprehensive tool for generating publication-quality neural network architecture diagrams
for TensorFlow/Keras models, similar to torchview for PyTorch.

Features:
- High-quality architecture diagrams suitable for research papers
- Multiple output formats (PNG, PDF, SVG)
- Automatic layer shape and parameter annotation
- Support for complex architectures (residual, multi-input/output, branching)
- Customizable styling and layout
- Memory usage and FLOPS estimation display
"""

import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import OrderedDict, defaultdict
import numpy as np
import json
from pathlib import Path

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ModelView:
    """
    Generate publication-quality architecture diagrams for TensorFlow/Keras models.
    
    Features:
    - Automatic graph layout and rendering
    - Layer parameter counts and shapes
    - Tensor flow visualization
    - Multiple output formats
    - Customizable styling
    
    Example:
        >>> model = YourModel()
        >>> view = ModelView(model, input_shape=(224, 224, 3))
        >>> view.render('model_architecture.png')
        >>> # Or with custom styling
        >>> view.render('model.pdf', format='pdf', rankdir='TB', 
        ...             show_shapes=True, show_layer_names=True)
    """
    
    def __init__(
        self,
        model: keras.Model,
        input_shape: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
        batch_size: int = 1,
        device: str = "CPU",
        depth: Optional[int] = None,
        expand_nested: bool = True
    ):
        """
        Initialize ModelView.
        
        Args:
            model: TensorFlow/Keras model to visualize
            input_shape: Input tensor shape(s) excluding batch dimension
                        e.g., (224, 224, 3) for images
                        Can be list of shapes for multi-input models
            batch_size: Batch size for shape inference
            device: Device for computation ('CPU' or 'GPU')
            depth: Maximum depth for nested models (None = unlimited)
            expand_nested: Whether to expand nested/functional models
        """
        self.model = model
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.device = device
        self.depth = depth
        self.expand_nested = expand_nested
        
        self.graph_data: OrderedDict = OrderedDict()
        self.connections: List[Tuple[str, str]] = []
        self.layer_info: Dict[str, Dict[str, Any]] = {}
        self.input_shapes: Dict[str, List[int]] = {}
        self.output_shapes: Dict[str, List[int]] = {}
        
        self._analyze_model()
    
    def _analyze_model(self):
        """Analyze model structure and extract graph information"""
        # Build model if not already built
        if self.input_shape is not None:
            if isinstance(self.input_shape, list):
                dummy_inputs = [
                    tf.random.normal([self.batch_size] + list(shape)) 
                    for shape in self.input_shape
                ]
            else:
                dummy_inputs = tf.random.normal([self.batch_size] + list(self.input_shape))
            
            if not self.model.built:
                self.model(dummy_inputs, training=False)
        
        # Extract layer information
        self._extract_layers()
        
        # Extract connections
        self._extract_connections()
        
        # Calculate shapes
        if self.input_shape is not None:
            self._infer_shapes(dummy_inputs)
    
    def _extract_layers(self):
        """Extract all layers from the model"""
        layer_count = defaultdict(int)
        
        for idx, layer in enumerate(self.model.layers):
            layer_type = layer.__class__.__name__
            layer_count[layer_type] += 1
            
            layer_id = f"{layer.name}_{idx}"
            
            # Count parameters
            trainable_params = sum([tf.size(w).numpy() for w in layer.trainable_weights])
            non_trainable_params = sum([tf.size(w).numpy() for w in layer.non_trainable_weights])
            total_params = trainable_params + non_trainable_params
            
            # Get layer config
            config = layer.get_config()
            
            self.layer_info[layer_id] = {
                'name': layer.name,
                'type': layer_type,
                'trainable_params': int(trainable_params),
                'non_trainable_params': int(non_trainable_params),
                'total_params': int(total_params),
                'config': config,
                'layer_obj': layer
            }
    
    def _extract_connections(self):
        """Extract connections between layers"""
        # For sequential models, connections are linear
        if isinstance(self.model, keras.Sequential):
            layer_ids = list(self.layer_info.keys())
            for i in range(len(layer_ids) - 1):
                self.connections.append((layer_ids[i], layer_ids[i + 1]))
        else:
            # For functional/subclassed models, use keras layer connectivity
            for layer in self.model.layers:
                if hasattr(layer, '_inbound_nodes') and len(layer._inbound_nodes) > 0:
                    for node in layer._inbound_nodes:
                        # Handle different TensorFlow/Keras versions
                        if hasattr(node, 'inbound_layers'):
                            inbound_layers_list = node.inbound_layers
                        elif hasattr(node, 'parent_nodes'):
                            # Older API
                            inbound_layers_list = [n.layer for n in node.parent_nodes if hasattr(n, 'layer')]
                        elif hasattr(node, 'call_args'):
                            # Extract from call args (newer Keras 3.x)
                            inbound_layers_list = []
                            if hasattr(node, 'parent_nodes'):
                                for pn in node.parent_nodes:
                                    if hasattr(pn, 'layer'):
                                        inbound_layers_list.append(pn.layer)
                        else:
                            # Try to get from node's operation
                            inbound_layers_list = []
                            if hasattr(node, 'operation') and hasattr(node.operation, '_inbound_nodes'):
                                for n in node.operation._inbound_nodes:
                                    if hasattr(n, 'layer'):
                                        inbound_layers_list.append(n.layer)
                        
                        for inbound_layer in inbound_layers_list:
                            # Find layer IDs
                            src_id = None
                            dst_id = None
                            
                            for lid, linfo in self.layer_info.items():
                                if linfo['layer_obj'] == inbound_layer:
                                    src_id = lid
                                if linfo['layer_obj'] == layer:
                                    dst_id = lid
                            
                            if src_id and dst_id and src_id != dst_id:
                                self.connections.append((src_id, dst_id))
    
    def _infer_shapes(self, dummy_inputs):
        """Infer input and output shapes for each layer"""
        # For Sequential models, get outputs by building intermediate models
        if isinstance(self.model, keras.Sequential):
            # Run a forward pass to get all layer outputs
            try:
                # Get outputs for all layers
                layer_outputs = []
                x = dummy_inputs
                for layer in self.model.layers:
                    x = layer(x, training=False)
                    layer_outputs.append(x)
                
                # Map outputs to layer_ids
                for idx, (layer_id, layer_info) in enumerate(self.layer_info.items()):
                    if idx < len(layer_outputs):
                        output = layer_outputs[idx]
                        if isinstance(output, (list, tuple)):
                            self.output_shapes[layer_id] = [list(o.shape) for o in output]
                        else:
                            self.output_shapes[layer_id] = list(output.shape)
            except Exception as e:
                # Fallback to None if inference fails
                for layer_id in self.layer_info.keys():
                    if layer_id not in self.output_shapes:
                        self.output_shapes[layer_id] = None
        else:
            # For Functional/Subclassed models, use layer connectivity
            for layer_id, layer_info in self.layer_info.items():
                layer = layer_info['layer_obj']
                
                try:
                    # Create a model that outputs this layer's output
                    if hasattr(layer, 'input') and hasattr(layer, 'output'):
                        temp_model = keras.Model(
                            inputs=self.model.input,
                            outputs=layer.output
                        )
                        output = temp_model(dummy_inputs, training=False)
                        
                        if isinstance(output, (list, tuple)):
                            self.output_shapes[layer_id] = [list(o.shape) for o in output]
                        else:
                            self.output_shapes[layer_id] = list(output.shape)
                        
                        # Input shape
                        if isinstance(layer.input, (list, tuple)):
                            self.input_shapes[layer_id] = [list(i.shape) for i in layer.input]
                        else:
                            self.input_shapes[layer_id] = list(layer.input.shape)
                except Exception as e:
                    # Some layers might not be directly accessible
                    self.output_shapes[layer_id] = None
                    self.input_shapes[layer_id] = None
    
    def _format_shape(self, shape: Union[List[int], List[List[int]], None]) -> str:
        """Format shape for display"""
        if shape is None:
            return "?"
        
        if isinstance(shape[0], list):
            # Multiple shapes
            return "\n".join([str(tuple(s)) for s in shape])
        else:
            # Single shape - remove batch dimension
            if len(shape) > 1:
                return str(tuple(shape[1:]))
            return str(tuple(shape))
    
    def _format_params(self, params: int) -> str:
        """Format parameter count for display"""
        if params == 0:
            return "0"
        elif params < 1000:
            return str(params)
        elif params < 1_000_000:
            return f"{params / 1000:.1f}K"
        else:
            return f"{params / 1_000_000:.1f}M"
    
    def _create_graphviz_graph(
        self,
        rankdir: str = 'TB',
        show_shapes: bool = True,
        show_layer_names: bool = False,
        show_params: bool = True,
        node_style: Optional[Dict[str, str]] = None,
        edge_style: Optional[Dict[str, str]] = None
    ) -> 'graphviz.Digraph':
        """Create a Graphviz graph representing the model"""
        if not GRAPHVIZ_AVAILABLE:
            raise ImportError(
                "Graphviz is required for rendering. "
                "Install with: pip install graphviz"
            )
        
        # Default styles
        default_node_style = {
            'shape': 'box',
            'style': 'rounded,filled',
            'fillcolor': '#E8F4F8',
            'fontname': 'Arial',
            'fontsize': '10',
            'margin': '0.2,0.1'
        }
        default_edge_style = {
            'color': '#5C6BC0',
            'penwidth': '1.5',
            'arrowsize': '0.7'
        }
        
        node_style = {**default_node_style, **(node_style or {})}
        edge_style = {**default_edge_style, **(edge_style or {})}
        
        # Create graph
        graph = graphviz.Digraph(
            name='model_architecture',
            format='png',
            graph_attr={
                'rankdir': rankdir,
                'bgcolor': 'white',
                'splines': 'ortho',
                'nodesep': '0.5',
                'ranksep': '0.8',
                'fontname': 'Arial',
                'fontsize': '12'
            }
        )
        
        # Add input node
        input_label = "Input"
        if show_shapes and self.input_shape is not None:
            if isinstance(self.input_shape, list):
                shape_str = "\n".join([str(tuple(s)) for s in self.input_shape])
            else:
                shape_str = str(tuple(self.input_shape))
            input_label += f"\n{shape_str}"
        
        # Create input node attributes without conflicting keys
        input_attrs = {k: v for k, v in node_style.items() if k not in ['shape', 'fillcolor']}
        input_attrs['shape'] = 'ellipse'
        input_attrs['fillcolor'] = '#C8E6C9'
        
        graph.node('input', label=input_label, **input_attrs)
        
        # Add layer nodes
        for layer_id, layer_info in self.layer_info.items():
            label_parts = []
            
            # Show layer type (always) and optionally the instance name
            if show_layer_names:
                label_parts.append(f"{layer_info['name']}: {layer_info['type']}")
            else:
                label_parts.append(f"{layer_info['type']}")
            
            if show_shapes and layer_id in self.output_shapes:
                shape_str = self._format_shape(self.output_shapes[layer_id])
                label_parts.append(f"Shape: {shape_str}")
            
            if show_params and layer_info['total_params'] > 0:
                params_str = self._format_params(layer_info['total_params'])
                label_parts.append(f"Params: {params_str}")
            
            label = "\n".join(label_parts)
            
            # Color code by layer type
            color = self._get_layer_color(layer_info['type'])
            
            # Create node attributes without conflicting fillcolor
            layer_attrs = {k: v for k, v in node_style.items() if k != 'fillcolor'}
            layer_attrs['fillcolor'] = color
            
            graph.node(layer_id, label=label, **layer_attrs)
        
        # Add output node
        output_attrs = {k: v for k, v in node_style.items() if k not in ['shape', 'fillcolor']}
        output_attrs['shape'] = 'ellipse'
        output_attrs['fillcolor'] = '#FFCDD2'
        
        graph.node('output', label="Output", **output_attrs)
        
        # Add edges
        if len(self.layer_info) > 0:
            first_layer = list(self.layer_info.keys())[0]
            graph.edge('input', first_layer, **edge_style)
        
        for src, dst in self.connections:
            graph.edge(src, dst, **edge_style)
        
        if len(self.layer_info) > 0:
            last_layer = list(self.layer_info.keys())[-1]
            graph.edge(last_layer, 'output', **edge_style)
        
        return graph
    
    def _get_layer_color(self, layer_type: str) -> str:
        """Get color for layer type"""
        color_map = {
            'Dense': '#FFE0B2',
            'Conv2D': '#B2DFDB',
            'Conv3D': '#B2DFDB',
            'MaxPooling2D': '#F8BBD0',
            'AveragePooling2D': '#F8BBD0',
            'GlobalMaxPooling2D': '#F8BBD0',
            'GlobalAveragePooling2D': '#F8BBD0',
            'Dropout': '#E1BEE7',
            'BatchNormalization': '#FFF9C4',
            'LayerNormalization': '#FFF9C4',
            'Flatten': '#DCEDC8',
            'Reshape': '#DCEDC8',
            'LSTM': '#BBDEFB',
            'GRU': '#BBDEFB',
            'SimpleRNN': '#BBDEFB',
            'Embedding': '#FFE082',
            'Concatenate': '#FFCCBC',
            'Add': '#FFCCBC',
            'Multiply': '#FFCCBC',
            'Activation': '#E1BEE7',
            'LeakyReLU': '#E1BEE7',
            'ReLU': '#E1BEE7',
        }
        return color_map.get(layer_type, '#E8F4F8')
    
    def render(
        self,
        filename: str,
        format: Optional[str] = None,
        rankdir: str = 'TB',
        show_shapes: bool = True,
        show_layer_names: bool = False,
        show_params: bool = True,
        dpi: int = 300,
        node_style: Optional[Dict[str, str]] = None,
        edge_style: Optional[Dict[str, str]] = None,
        cleanup: bool = True
    ) -> str:
        """
        Render the model architecture diagram to a file.
        
        Args:
            filename: Output file path
            format: Output format ('png', 'pdf', 'svg'). If None, inferred from filename
            rankdir: Graph direction ('TB'=top-to-bottom, 'LR'=left-to-right)
            show_shapes: Whether to display tensor shapes
            show_layer_names: Whether to display layer names
            show_params: Whether to display parameter counts
            dpi: Resolution for raster formats (PNG)
            node_style: Custom node styling (dict of graphviz attributes)
            edge_style: Custom edge styling (dict of graphviz attributes)
            cleanup: Whether to remove intermediate files
        
        Returns:
            Path to the rendered file
        """
        if not GRAPHVIZ_AVAILABLE:
            raise ImportError(
                "Graphviz is required for rendering. "
                "Install with: pip install graphviz\n"
                "Also install system graphviz: "
                "Ubuntu: sudo apt-get install graphviz\n"
                "macOS: brew install graphviz\n"
                "Windows: Download from https://graphviz.org/download/"
            )
        
        # Infer format from filename if not provided
        if format is None:
            format = Path(filename).suffix[1:].lower()
            if not format:
                format = 'png'
        
        # Create graph
        graph = self._create_graphviz_graph(
            rankdir=rankdir,
            show_shapes=show_shapes,
            show_layer_names=show_layer_names,
            show_params=show_params,
            node_style=node_style,
            edge_style=edge_style
        )
        
        # Set format and DPI
        graph.format = format
        graph.graph_attr['dpi'] = str(dpi)
        
        # Render
        output_path = Path(filename).with_suffix('')
        graph.render(str(output_path), cleanup=cleanup)
        
        result_path = f"{output_path}.{format}"
        return result_path
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """
        Get a dictionary summary of the model architecture.
        
        Returns:
            Dictionary containing layer info, connections, and statistics
        """
        total_params = sum(info['total_params'] for info in self.layer_info.values())
        trainable_params = sum(info['trainable_params'] for info in self.layer_info.values())
        
        return {
            'model_name': self.model.name,
            'num_layers': len(self.layer_info),
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(total_params - trainable_params),
            'layers': [
                {
                    'id': layer_id,
                    'name': info['name'],
                    'type': info['type'],
                    'params': info['total_params'],
                    'output_shape': self._format_shape(self.output_shapes.get(layer_id))
                }
                for layer_id, info in self.layer_info.items()
            ],
            'connections': [
                {'from': src, 'to': dst}
                for src, dst in self.connections
            ]
        }
    
    def save_summary_json(self, filename: str):
        """Save model summary as JSON file"""
        summary = self.get_summary_dict()
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def show(self, detailed: bool = False):
        """
        Print a text-based summary of the model architecture.
        
        Args:
            detailed: Whether to show detailed information
        """
        print("=" * 80)
        print(f"Model: {self.model.name}")
        print("=" * 80)
        
        total_params = sum(info['total_params'] for info in self.layer_info.values())
        trainable_params = sum(info['trainable_params'] for info in self.layer_info.values())
        
        print(f"Total Layers: {len(self.layer_info)}")
        print(f"Total Parameters: {self._format_params(total_params)} ({total_params:,})")
        print(f"Trainable Parameters: {self._format_params(trainable_params)} ({trainable_params:,})")
        print(f"Non-trainable Parameters: {self._format_params(total_params - trainable_params)}")
        print("=" * 80)
        
        # Layer table
        print(f"{'Layer (type)':<30} {'Output Shape':<20} {'Params':<15}")
        print("-" * 80)
        
        for layer_id, info in self.layer_info.items():
            layer_label = f"{info['name']} ({info['type']})"
            output_shape = self._format_shape(self.output_shapes.get(layer_id))
            params = self._format_params(info['total_params'])
            
            print(f"{layer_label:<30} {output_shape:<20} {params:<15}")
        
        print("=" * 80)
        
        if detailed:
            print("\nConnections:")
            for src, dst in self.connections:
                src_name = self.layer_info[src]['name']
                dst_name = self.layer_info[dst]['name']
                print(f"  {src_name} -> {dst_name}")
            print("=" * 80)


def draw_graph(
    model: keras.Model,
    input_shape: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
    save_path: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """
    Convenience function to quickly visualize a model.
    
    Args:
        model: TensorFlow/Keras model to visualize
        input_shape: Input shape(s) excluding batch dimension
        save_path: Path to save the visualization. If None, only returns ModelView object
        **kwargs: Additional arguments passed to ModelView.render()
    
    Returns:
        Path to saved file if save_path is provided, else None
        
    Example:
        >>> model = tf.keras.Sequential([...])
        >>> draw_graph(model, input_shape=(224, 224, 3), save_path='model.png')
    """
    view = ModelView(model, input_shape=input_shape)
    
    if save_path:
        return view.render(save_path, **kwargs)
    else:
        view.show()
        return None
