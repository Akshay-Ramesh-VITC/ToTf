"""
ModelView - PyTorch Model Architecture Visualization

A wrapper around torchview for generating publication-quality neural network 
architecture diagrams for PyTorch models with a unified API similar to the 
TensorFlow ModelView.

Features:
- High-quality architecture diagrams suitable for research papers
- Multiple output formats (PNG, PDF, SVG)
- Automatic layer shape and parameter annotation
- Support for complex architectures (residual, multi-input/output, branching)
- Customizable styling and layout
- Leverages torchview internally for comprehensive PyTorch support
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json

try:
    from torchview import draw_graph as torchview_draw_graph
    TORCHVIEW_AVAILABLE = True
except ImportError:
    TORCHVIEW_AVAILABLE = False
    torchview_draw_graph = None


class ModelView:
    """
    Generate publication-quality architecture diagrams for PyTorch models.
    
    This class wraps torchview to provide a consistent API similar to the 
    TensorFlow ModelView, making it easy to switch between frameworks while
    maintaining the same visualization workflow.
    
    Features:
    - Automatic graph layout and rendering
    - Layer parameter counts and shapes
    - Tensor flow visualization
    - Multiple output formats
    - Customizable styling
    
    Example:
        >>> model = YourModel()
        >>> view = ModelView(model, input_size=(3, 224, 224))
        >>> view.render('model_architecture.png')
        >>> # Or with custom styling
        >>> view.render('model.pdf', format='pdf', rankdir='TB', 
        ...             show_shapes=True, show_layer_names=True)
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_size: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
        input_data: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        batch_size: int = 1,
        device: str = "cpu",
        depth: int = 3,
        expand_nested: bool = False,
        hide_inner_tensors: bool = True,
        hide_module_functions: bool = True,
        roll: bool = False,
        show_shapes: bool = True,
        dtypes: Optional[List[torch.dtype]] = None,
        **kwargs
    ):
        """
        Initialize ModelView with torchview integration.
        
        Args:
            model: PyTorch model (nn.Module) to visualize
            input_size: Input tensor size(s) excluding batch dimension
                       e.g., (3, 224, 224) for images
                       Can be list of sizes for multi-input models
            input_data: Optional actual input tensor(s) instead of input_size
            batch_size: Batch size for shape inference (default: 1)
            device: Device for computation ('cpu' or 'cuda')
            depth: Maximum depth for nested models (default: 3)
                  Controls how deep to show in module hierarchy
            expand_nested: Whether to expand nested models with dashed borders
            hide_inner_tensors: If True, only show input/output tensors
                               If False, show all intermediate tensors
            hide_module_functions: If True, hide operations inside layers
                                  If False, show all operations
            roll: If True, roll recursive modules (useful for RNNs)
            show_shapes: Whether to show tensor shapes in visualization
            dtypes: Optional list of dtypes for each input
            **kwargs: Additional arguments to pass to torchview
        """
        if not TORCHVIEW_AVAILABLE:
            raise ImportError(
                "torchview is required for PyTorch ModelView. "
                "Install with: pip install torchview"
            )
        
        self.model = model
        self.input_size = input_size
        self.input_data = input_data
        self.batch_size = batch_size
        self.device = device
        self.depth = depth
        self.expand_nested = expand_nested
        self.hide_inner_tensors = hide_inner_tensors
        self.hide_module_functions = hide_module_functions
        self.roll = roll
        self.show_shapes = show_shapes
        self.dtypes = dtypes
        self.kwargs = kwargs
        
        # Store the torchview graph object
        self._graph = None
        self._build_graph()
    
    def _build_graph(self):
        """Build the computation graph using torchview"""
        # Prepare input data
        if self.input_data is not None:
            input_arg = self.input_data
        elif self.input_size is not None:
            # Convert input_size to proper format
            if isinstance(self.input_size, list):
                input_arg = [tuple([self.batch_size] + list(size)) for size in self.input_size]
            else:
                input_arg = tuple([self.batch_size] + list(self.input_size))
        else:
            raise ValueError("Either input_size or input_data must be provided")
        
        # Build graph using torchview
        try:
            self._graph = torchview_draw_graph(
                model=self.model,
                input_size=input_arg if self.input_data is None else None,
                input_data=self.input_data,
                device=self.device,
                depth=self.depth,
                expand_nested=self.expand_nested,
                hide_inner_tensors=self.hide_inner_tensors,
                hide_module_functions=self.hide_module_functions,
                roll=self.roll,
                dtypes=self.dtypes,
                **self.kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build computation graph: {e}")
    
    def render(
        self,
        filename: str,
        format: Optional[str] = None,
        rankdir: str = 'TB',
        show_shapes: bool = True,
        show_layer_names: bool = False,
        show_params: bool = True,
        dpi: int = 300,
        cleanup: bool = True,
        **kwargs
    ) -> str:
        """
        Render the model architecture diagram to a file.
        
        Args:
            filename: Output file path
            format: Output format ('png', 'pdf', 'svg'). If None, inferred from filename
            rankdir: Graph direction ('TB'=top-to-bottom, 'LR'=left-to-right,
                                     'BT'=bottom-to-top, 'RL'=right-to-left)
            show_shapes: Whether to display tensor shapes (always enabled in torchview)
            show_layer_names: Whether to display layer names (always shown in torchview)
            show_params: Whether to display parameter counts (always shown in torchview)
            dpi: Resolution for raster formats (PNG)
            cleanup: Whether to remove intermediate files
            **kwargs: Additional rendering arguments
        
        Returns:
            Path to the rendered file
        """
        if self._graph is None:
            raise RuntimeError("Graph not built. Call _build_graph() first.")
        
        # Infer format from filename if not provided
        if format is None:
            format = Path(filename).suffix[1:].lower()
            if not format:
                format = 'png'
        
        # Configure graph visualization settings
        graph_viz = self._graph.visual_graph
        graph_viz.graph_attr['rankdir'] = rankdir
        graph_viz.graph_attr['dpi'] = str(dpi)
        graph_viz.format = format
        
        # Render the graph
        output_path = Path(filename).with_suffix('')
        try:
            graph_viz.render(str(output_path), cleanup=cleanup)
            result_path = f"{output_path}.{format}"
            return result_path
        except Exception as e:
            raise RuntimeError(f"Failed to render graph: {e}")
    
    def render_advanced(
        self,
        filename: str,
        format: Optional[str] = None,
        rankdir: str = 'TB',
        show_shapes: bool = True,
        show_layer_names: bool = True,
        show_params: bool = True,
        dpi: int = 300,
        cleanup: bool = True,
        **kwargs
    ) -> str:
        """
        Render advanced computation graph visualization (alias for render).
        
        This method provides the same functionality as render() but with a name
        consistent with the TensorFlow ModelView API.
        
        Args:
            filename: Output file path
            format: Output format ('png', 'pdf', 'svg')
            rankdir: Graph direction
            show_shapes: Whether to display tensor shapes
            show_layer_names: Whether to display layer names
            show_params: Whether to display parameter counts
            dpi: Resolution for raster formats
            cleanup: Whether to remove intermediate files
            **kwargs: Additional rendering arguments
        
        Returns:
            Path to the rendered file
        """
        return self.render(
            filename=filename,
            format=format,
            rankdir=rankdir,
            show_shapes=show_shapes,
            show_layer_names=show_layer_names,
            show_params=show_params,
            dpi=dpi,
            cleanup=cleanup,
            **kwargs
        )
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """
        Get a dictionary summary of the model architecture.
        
        Returns:
            Dictionary containing layer info, shapes, and parameter statistics
        """
        if self._graph is None:
            raise RuntimeError("Graph not built.")
        
        # Extract information from torchview graph
        total_params = 0
        trainable_params = 0
        layers = []
        
        # Parse the graph nodes
        for node in self._graph.edge_list:
            if hasattr(node, 'num_params'):
                total_params += node.num_params.get('total', 0)
                trainable_params += node.num_params.get('trainable', 0)
        
        return {
            'model_name': self.model.__class__.__name__,
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(total_params - trainable_params),
            'input_size': self.input_size,
            'device': self.device,
            'depth': self.depth,
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
            detailed: Whether to show detailed information (not used in torchview)
        """
        if self._graph is None:
            raise RuntimeError("Graph not built.")
        
        print(self._graph)
    
    def export_svg(self, filename: str) -> str:
        """Export as SVG (vector format for publications)"""
        return self.render(filename, format='svg')
    
    def export_pdf(self, filename: str) -> str:
        """Export as PDF (vector format for papers)"""
        return self.render(filename, format='pdf')
    
    def export_png(self, filename: str, dpi: int = 300) -> str:
        """Export as high-resolution PNG"""
        return self.render(filename, format='png', dpi=dpi)
    
    @property
    def visual_graph(self):
        """Get the underlying torchview visual graph"""
        if self._graph is None:
            raise RuntimeError("Graph not built.")
        return self._graph.visual_graph


def draw_graph(
    model: nn.Module,
    input_size: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
    input_data: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    save_path: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """
    Convenience function to quickly visualize a PyTorch model.
    
    Args:
        model: PyTorch model (nn.Module) to visualize
        input_size: Input size(s) excluding batch dimension
        input_data: Optional actual input tensor(s) instead of input_size
        save_path: Path to save the visualization. If None, only prints summary
        **kwargs: Additional arguments passed to ModelView.render()
    
    Returns:
        Path to saved file if save_path is provided, else None
        
    Example:
        >>> model = MyModel()
        >>> draw_graph(model, input_size=(3, 224, 224), save_path='model.png')
    """
    view = ModelView(model, input_size=input_size, input_data=input_data)
    
    if save_path:
        return view.render(save_path, **kwargs)
    else:
        view.show()
        return None
