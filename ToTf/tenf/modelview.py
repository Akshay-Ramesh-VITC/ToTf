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
- Computation graph tracking with tensor operations
- Advanced visualization controls (hide_inner_tensors, hide_module_functions, etc.)
- Depth control for nested models
- Expand nested modules with dashed borders
- Roll/unroll recursive structures
"""

import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from collections import OrderedDict, defaultdict
import numpy as np
import json
from pathlib import Path
import hashlib

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

from .computation_nodes import (
    TensorNode, LayerNode, OperationNode, NodeContainer,
    BaseNode, COMPUTATION_NODE, get_node_color
)


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
        depth: int = 3,
        expand_nested: bool = False,
        hide_inner_tensors: bool = True,
        hide_module_functions: bool = True,
        roll: bool = False,
        show_shapes: bool = True
    ):
        """
        Initialize ModelView with torchview-like features.
        
        Args:
            model: TensorFlow/Keras model to visualize
            input_shape: Input tensor shape(s) excluding batch dimension
                        e.g., (224, 224, 3) for images
                        Can be list of shapes for multi-input models
            batch_size: Batch size for shape inference
            device: Device for computation ('CPU' or 'GPU')
            depth: Maximum depth for nested models (default: 3)
                  Controls how deep to show in module hierarchy
            expand_nested: Whether to expand nested models with dashed borders
            hide_inner_tensors: If True, only show input/output tensors
                               If False, show all intermediate tensors
            hide_module_functions: If True, hide operations inside layers
                                  If False, show all operations
            roll: If True, roll recursive modules (useful for RNNs)
            show_shapes: Whether to show tensor shapes in visualization
        """
        self.model = model
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.device = device
        self.depth = depth
        self.expand_nested = expand_nested
        self.hide_inner_tensors = hide_inner_tensors
        self.hide_module_functions = hide_module_functions
        self.roll = roll
        self.show_shapes = show_shapes
        
        # Computation graph data structures
        self.graph_data: OrderedDict = OrderedDict()
        self.connections: List[Tuple[str, str]] = []
        self.layer_info: Dict[str, Dict[str, Any]] = {}
        self.input_shapes: Dict[str, List[int]] = {}
        self.output_shapes: Dict[str, List[int]] = {}
        
        # Node tracking
        self.tensor_nodes: Dict[str, TensorNode] = {}
        self.layer_nodes: Dict[str, LayerNode] = {}
        self.operation_nodes: Dict[str, OperationNode] = {}
        self.all_nodes: Dict[str, COMPUTATION_NODE] = {}
        
        # Graph structure
        self.input_nodes: List[TensorNode] = []
        self.output_nodes: List[TensorNode] = []
        self.edge_list: List[Tuple[COMPUTATION_NODE, COMPUTATION_NODE]] = []
        
        # For tracking node IDs and subgraphs
        self.node_id_map: Dict[str, int] = {}
        self.running_node_id: int = 0
        self.subgraph_map: Dict[str, int] = {}
        self.running_subgraph_id: int = 0
        
        self._analyze_model()
    
    def _analyze_model(self):
        """Analyze model structure and build computation graph"""
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
        else:
            dummy_inputs = None
        
        # Build computation graph from layers and connections
        self._build_computation_graph(dummy_inputs)
        
        # Legacy support: also populate layer_info for backward compatibility
        self._extract_layers()
        self._extract_connections()
        if dummy_inputs is not None:
            self._infer_shapes(dummy_inputs)
    
    def _build_computation_graph(self, dummy_inputs):
        """Build computation graph using TensorFlow's layer connectivity"""
        # Create input nodes
        if dummy_inputs is not None:
            if isinstance(dummy_inputs, list):
                for idx, inp in enumerate(dummy_inputs):
                    node_id = f"input_{idx}"
                    tensor_node = TensorNode(
                        node_id=node_id,
                        name=f"Input_{idx}",
                        shape=tuple(inp.shape),
                        dtype=inp.dtype,
                        is_input=True,
                        depth=0
                    )
                    self.tensor_nodes[node_id] = tensor_node
                    self.all_nodes[node_id] = tensor_node
                    self.input_nodes.append(tensor_node)
            else:
                node_id = "input_0"
                tensor_node = TensorNode(
                    node_id=node_id,
                    name="Input",
                    shape=tuple(dummy_inputs.shape),
                    dtype=dummy_inputs.dtype,
                    is_input=True,
                    depth=0
                )
                self.tensor_nodes[node_id] = tensor_node
                self.all_nodes[node_id] = tensor_node
                self.input_nodes.append(tensor_node)
        
        # Process layers and build graph
        self._process_layers_recursive(self.model, parent_depth=0)
        
        # Identify output tensors
        self._identify_output_nodes()
    
    def _process_layers_recursive(self, model_or_layer, parent_depth=0):
        """Recursively process layers to build computation graph"""
        if parent_depth > self.depth:
            return
        
        layers_to_process = []
        
        if hasattr(model_or_layer, 'layers'):
            layers_to_process = model_or_layer.layers
        else:
            layers_to_process = [model_or_layer]
        
        for idx, layer in enumerate(layers_to_process):
            # Create layer node
            layer_id = f"{layer.name}_{id(layer)}"
            current_depth = parent_depth
            
            # Check if this is a nested model
            is_nested = hasattr(layer, 'layers') and len(layer.layers) > 0
            
            if is_nested and self.expand_nested and current_depth < self.depth:
                # Process nested model recursively
                self._process_layers_recursive(layer, parent_depth=current_depth + 1)
            
            # Create layer node
            layer_node = LayerNode(
                node_id=layer_id,
                name=layer.name,
                layer_type=layer.__class__.__name__,
                layer_obj=layer,
                depth=current_depth
            )
            
            self.layer_nodes[layer_id] = layer_node
            self.all_nodes[layer_id] = layer_node
            
            # Get input and output shapes if possible
            try:
                if hasattr(layer, 'input') and layer.input is not None:
                    if isinstance(layer.input, list):
                        layer_node.input_shape = [tuple(inp.shape) for inp in layer.input]
                    else:
                        layer_node.input_shape = tuple(layer.input.shape)
                
                if hasattr(layer, 'output') and layer.output is not None:
                    if isinstance(layer.output, list):
                        layer_node.output_shape = [tuple(out.shape) for out in layer.output]
                    else:
                        layer_node.output_shape = tuple(layer.output.shape)
            except:
                pass
            
            # Build connections from inbound nodes
            if hasattr(layer, '_inbound_nodes') and len(layer._inbound_nodes) > 0:
                for node in layer._inbound_nodes:
                    if hasattr(node, 'parent_nodes'):
                        for parent_node in node.parent_nodes:
                            if hasattr(parent_node, 'operation'):
                                parent_layer = parent_node.operation
                                parent_id = f"{parent_layer.name}_{id(parent_layer)}"
                                
                                # Add edge
                                if parent_id in self.all_nodes:
                                    parent_layer_node = self.all_nodes[parent_id]
                                    self.edge_list.append((parent_layer_node, layer_node))
                                    layer_node.add_parent(parent_layer_node)
                                    parent_layer_node.add_child(layer_node)
    
    def _identify_output_nodes(self):
        """Identify output nodes (leaf nodes in the graph)"""
        # Create a list of nodes to avoid dictionary change during iteration
        nodes_to_check = list(self.all_nodes.items())
        
        for node_id, node in nodes_to_check:
            if isinstance(node, (LayerNode, TensorNode)) and node.is_leaf():
                if isinstance(node, LayerNode):
                    # Create output tensor node
                    output_id = f"output_{node_id}"
                    output_node = TensorNode(
                        node_id=output_id,
                        name=f"Output_{node.name}",
                        shape=node.output_shape if hasattr(node, 'output_shape') else None,
                        is_output=True,
                        depth=node.depth
                    )
                    self.tensor_nodes[output_id] = output_node
                    self.all_nodes[output_id] = output_node
                    self.output_nodes.append(output_node)
                    
                    # Connect layer to output
                    self.edge_list.append((node, output_node))
                    output_node.add_parent(node)
                    node.add_child(output_node)
                elif node.is_output:
                    self.output_nodes.append(node)
    
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
                        # Get parent nodes (inbound connections)
                        if hasattr(node, 'parent_nodes'):
                            inbound_layers_list = []
                            for parent_node in node.parent_nodes:
                                # The operation attribute contains the actual layer
                                if hasattr(parent_node, 'operation'):
                                    inbound_layers_list.append(parent_node.operation)
                        else:
                            # Fallback for older Keras versions
                            inbound_layers_list = []
                        
                        for inbound_layer in inbound_layers_list:
                            # Find layer IDs by matching layer objects
                            src_id = None
                            dst_id = None
                            
                            for lid, linfo in self.layer_info.items():
                                if linfo['layer_obj'] == inbound_layer:
                                    src_id = lid
                                if linfo['layer_obj'] == layer:
                                    dst_id = lid
                            
                            if src_id and dst_id and src_id != dst_id:
                                conn = (src_id, dst_id)
                                # Avoid duplicate connections
                                if conn not in self.connections:
                                    self.connections.append(conn)
    
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


    # ========== Advanced Visualization Methods (torchview-like) ==========
    
    def _is_node_visible(self, node: COMPUTATION_NODE) -> bool:
        """Determine if a node should be visible based on visualization settings"""
        # Always show input/output nodes
        if isinstance(node, TensorNode):
            if node.is_input or node.is_output:
                return True
            # Hide inner tensors if requested
            if self.hide_inner_tensors:
                return False
            return True
        
        elif isinstance(node, LayerNode):
            # Check depth
            if node.depth > self.depth:
                return False
            
            # Show container modules at the right depth
            if node.depth == self.depth:
                return True
            
            # Show based on expand_nested
            if self.expand_nested:
                return True
            
            return node.depth <= self.depth
        
        elif isinstance(node, OperationNode):
            # Hide operations inside modules if requested
            if self.hide_module_functions:
                return False
            return node.depth <= self.depth
        
        return True
    
    def _get_node_label_html(
        self, 
        node: COMPUTATION_NODE,
        show_shapes: bool,
        show_layer_names: bool,
        show_params: bool
    ) -> str:
        """Generate HTML label for a node (torchview style)"""
        if isinstance(node, TensorNode):
            if show_shapes and node.shape:
                shape_str = self._format_shape(list(node.shape))
                return f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">'\
                       f'<TR><TD>{node.name}<BR/>depth:{node.depth}</TD></TR>'\
                       f'<TR><TD>Shape: {shape_str}</TD></TR>'\
                       f'</TABLE>>'
            return f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">'\
                   f'<TR><TD>{node.name}<BR/>depth:{node.depth}</TD></TR>'\
                   f'</TABLE>>'
        
        elif isinstance(node, LayerNode):
            rows = []
            if show_layer_names:
                rows.append(f'<TR><TD>{node.name}: {node.layer_type}</TD></TR>')
            else:
                rows.append(f'<TR><TD>{node.layer_type}</TD></TR>')
            
            if show_shapes:
                if node.input_shape:
                    input_str = self._format_shape(node.input_shape)
                    rows.append(f'<TR><TD>Input: {input_str}</TD></TR>')
                if node.output_shape:
                    output_str = self._format_shape(node.output_shape)
                    rows.append(f'<TR><TD>Output: {output_str}</TD></TR>')
            
            if show_params and node.total_params > 0:
                params_str = self._format_params(node.total_params)
                rows.append(f'<TR><TD>Params: {params_str}</TD></TR>')
            
            rows.append(f'<TR><TD>depth:{node.depth}</TD></TR>')
            
            return f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">'\
                   f'{"".join(rows)}'\
                   f'</TABLE>>'
        
        elif isinstance(node, OperationNode):
            return f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">'\
                   f'<TR><TD>{node.op_type}<BR/>depth:{node.depth}</TD></TR>'\
                   f'</TABLE>>'
        
        return node.name
    
    def _create_advanced_graph(
        self,
        rankdir: str = 'TB',
        show_shapes: bool = True,
        show_layer_names: bool = False,
        show_params: bool = True
    ) -> 'graphviz.Digraph':
        """Create advanced computation graph visualization (torchview-like)"""
        if not GRAPHVIZ_AVAILABLE:
            raise ImportError("Graphviz is required for rendering.")
        
        # Create graph with torchview-like styling
        graph = graphviz.Digraph(
            name='computation_graph',
            format='png',
            engine='dot',
            graph_attr={
                'rankdir': rankdir,
                'bgcolor': 'white',
                'ordering': 'in',
                'ranksep': '0.1',
                'fontname': 'Arial',
                'fontsize': '12'
            },
            node_attr={
                'style': 'filled',
                'shape': 'plaintext',
                'align': 'left',
                'fontsize': '10',
                'height': '0.2',
                'fontname': 'Arial',
                'margin': '0'
            },
            edge_attr={
                'fontsize': '10'
            }
        )
        
        # Add visible nodes
        added_nodes: Set[str] = set()
        for node_id, node in self.all_nodes.items():
            if not self._is_node_visible(node):
                continue
            
            if node_id not in self.node_id_map:
                self.node_id_map[node_id] = self.running_node_id
                self.running_node_id += 1
            
            gv_id = str(self.node_id_map[node_id])
            label = self._get_node_label_html(
                node, show_shapes, show_layer_names, show_params
            )
            color = get_node_color(node)
            
            # Handle nested modules with dashed borders
            style = 'filled'
            if isinstance(node, LayerNode) and node.is_container and self.expand_nested:
                style = 'filled,dashed'
            
            graph.node(gv_id, label=label, fillcolor=color, style=style)
            added_nodes.add(gv_id)
        
        # Add edges
        added_edges: Set[Tuple[int, int]] = set()
        for src_node, dst_node in self.edge_list:
            if not self._is_node_visible(src_node) or not self._is_node_visible(dst_node):
                continue
            
            if src_node.node_id not in self.node_id_map or dst_node.node_id not in self.node_id_map:
                continue
            
            src_id = str(self.node_id_map[src_node.node_id])
            dst_id = str(self.node_id_map[dst_node.node_id])
            
            if src_id not in added_nodes or dst_id not in added_nodes:
                continue
            
            edge_key = (int(src_id), int(dst_id))
            if edge_key in added_edges:
                continue
            
            added_edges.add(edge_key)
            graph.edge(src_id, dst_id)
        
        return graph
    
    def render_advanced(
        self,
        filename: str,
        format: Optional[str] = None,
        rankdir: str = 'TB',
        show_shapes: bool = True,
        show_layer_names: bool = False,
        show_params: bool = True,
        dpi: int = 300,
        cleanup: bool = True
    ) -> str:
        """
        Render advanced computation graph visualization (torchview-like).
        
        This uses the computation graph with advanced features like
        hide_inner_tensors, hide_module_functions, expand_nested, etc.
        
        Args:
            filename: Output file path
            format: Output format ('png', 'pdf', 'svg')
            rankdir: Graph direction ('TB', 'LR', 'BT', 'RL')
            show_shapes: Whether to display tensor shapes
            show_layer_names: Whether to display layer instance names
            show_params: Whether to display parameter counts
            dpi: Resolution for raster formats
            cleanup: Whether to remove intermediate files
        
        Returns:
            Path to the rendered file
        """
        if format is None:
            format = Path(filename).suffix[1:].lower()
            if not format:
                format = 'png'
        
        graph = self._create_advanced_graph(
            rankdir=rankdir,
            show_shapes=show_shapes,
            show_layer_names=show_layer_names,
            show_params=show_params
        )
        
        graph.format = format
        graph.graph_attr['dpi'] = str(dpi)
        
        output_path = Path(filename).with_suffix('')
        graph.render(str(output_path), cleanup=cleanup)
        
        result_path = f"{output_path}.{format}"
        return result_path


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
