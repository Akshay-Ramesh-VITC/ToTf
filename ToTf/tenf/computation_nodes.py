"""
Computation Node Classes for TensorFlow ModelView

Defines node types used in the computation graph:
- TensorNode: Represents tensors in the computation
- LayerNode: Represents Keras layers/modules  
- OperationNode: Represents TensorFlow operations (Add, Concat, etc.)
"""

from typing import List, Optional, Set, Dict, Any, Union
from collections import OrderedDict
import tensorflow as tf


class NodeContainer:
    """Container for managing multiple nodes"""
    
    def __init__(self, nodes: Optional[List['BaseNode']] = None):
        self.nodes: List['BaseNode'] = nodes or []
    
    def __iter__(self):
        return iter(self.nodes)
    
    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, idx):
        return self.nodes[idx]
    
    def append(self, node: 'BaseNode'):
        self.nodes.append(node)
    
    def extend(self, nodes: List['BaseNode']):
        self.nodes.extend(nodes)


class BaseNode:
    """Base class for all computation graph nodes"""
    
    def __init__(
        self,
        node_id: str,
        name: str,
        depth: int = 0,
        parent_hierarchy: Optional[Dict[int, 'BaseNode']] = None
    ):
        self.node_id = node_id
        self.name = name
        self.depth = depth
        self.parent_hierarchy = parent_hierarchy or {}
        
        self.parents: List['BaseNode'] = []
        self.children: List['BaseNode'] = []
        
        self.input_shape: Optional[Union[tuple, List[tuple]]] = None
        self.output_shape: Optional[Union[tuple, List[tuple]]] = None
    
    def add_parent(self, node: 'BaseNode'):
        """Add a parent node (incoming connection)"""
        if node not in self.parents:
            self.parents.append(node)
    
    def add_child(self, node: 'BaseNode'):
        """Add a child node (outgoing connection)"""
        if node not in self.children:
            self.children.append(node)
    
    def is_root(self) -> bool:
        """Check if this is a root node (no parents)"""
        return len(self.parents) == 0
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)"""
        return len(self.children) == 0
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, depth={self.depth})"


class TensorNode(BaseNode):
    """Node representing a tensor in the computation graph"""
    
    def __init__(
        self,
        node_id: str,
        name: str,
        shape: Optional[tuple] = None,
        dtype: Optional[tf.DType] = None,
        depth: int = 0,
        parent_hierarchy: Optional[Dict[int, BaseNode]] = None,
        is_input: bool = False,
        is_output: bool = False
    ):
        super().__init__(node_id, name, depth, parent_hierarchy)
        self.shape = shape
        self.dtype = dtype
        self.is_input = is_input
        self.is_output = is_output
        self.output_shape = shape
        
        # For tracking auxiliary tensors (intermediate computations)
        self.is_aux = False
        self.main_node: Optional['TensorNode'] = None


class LayerNode(BaseNode):
    """Node representing a Keras layer/module"""
    
    def __init__(
        self,
        node_id: str,
        name: str,
        layer_type: str,
        layer_obj: Optional[tf.keras.layers.Layer] = None,
        depth: int = 0,
        parent_hierarchy: Optional[Dict[int, BaseNode]] = None
    ):
        super().__init__(node_id, name, depth, parent_hierarchy)
        self.layer_type = layer_type
        self.layer_obj = layer_obj
        
        self.trainable_params = 0
        self.non_trainable_params = 0
        self.total_params = 0
        
        # Check if this is a container layer (has sublayers)
        self.is_container = False
        if layer_obj is not None:
            try:
                self.trainable_params = sum([tf.size(w).numpy() for w in layer_obj.trainable_weights])
                self.non_trainable_params = sum([tf.size(w).numpy() for w in layer_obj.non_trainable_weights])
                self.total_params = self.trainable_params + self.non_trainable_params
                
                # Check if container (has sublayers)
                if hasattr(layer_obj, 'layers') and len(layer_obj.layers) > 0:
                    self.is_container = True
            except:
                pass
        
        # For tracking output tensors from this layer
        self.output_nodes: List[TensorNode] = []


class OperationNode(BaseNode):
    """Node representing a TensorFlow operation (Add, Concatenate, etc.)"""
    
    def __init__(
        self,
        node_id: str,
        name: str,
        op_type: str,
        depth: int = 0,
        parent_hierarchy: Optional[Dict[int, BaseNode]] = None
    ):
        super().__init__(node_id, name, depth, parent_hierarchy)
        self.op_type = op_type
        
        # For tracking output tensors from this operation
        self.output_nodes: List[TensorNode] = []


# Type alias for any computation node
COMPUTATION_NODE = Union[TensorNode, LayerNode, OperationNode]


def get_node_color(node: COMPUTATION_NODE) -> str:
    """Get the display color for a node based on its type"""
    if isinstance(node, TensorNode):
        if node.is_input or node.is_output:
            return "lightyellow"
        return "lightyellow"
    elif isinstance(node, LayerNode):
        return "darkseagreen1"
    elif isinstance(node, OperationNode):
        return "aliceblue"
    return "white"
