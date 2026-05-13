"""
Quick test to verify ModelView torchview-like features work correctly
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tenf.modelview import ModelView
from tenf.computation_nodes import TensorNode, LayerNode, OperationNode


def test_imports():
    """Test that all classes can be imported"""
    print("Testing imports...")
    assert TensorNode is not None
    assert LayerNode is not None
    assert OperationNode is not None
    assert ModelView is not None
    print("✓ All imports successful")


def test_simple_model():
    """Test with a simple sequential model"""
    print("\nTesting simple sequential model...")
    
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(100,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    # Test with default settings
    view = ModelView(model, input_shape=(100,))
    
    assert view.model is not None
    assert len(view.input_nodes) > 0
    print(f"✓ Created ModelView with {len(view.all_nodes)} nodes")
    print(f"  - Input nodes: {len(view.input_nodes)}")
    print(f"  - Layer nodes: {len(view.layer_nodes)}")
    print(f"  - Tensor nodes: {len(view.tensor_nodes)}")
    print(f"  - Edges: {len(view.edge_list)}")


def test_resnet_model():
    """Test with a model containing skip connections"""
    print("\nTesting ResNet-style model with skip connections...")
    
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Skip connection
    shortcut = x
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Test with different settings
    view1 = ModelView(model, input_shape=(32, 32, 3), hide_inner_tensors=True)
    view2 = ModelView(model, input_shape=(32, 32, 3), hide_inner_tensors=False)
    
    print(f"✓ ResNet model created successfully")
    print(f"  - With hide_inner_tensors=True: {len([n for n in view1.all_nodes.values() if view1._is_node_visible(n)])} visible nodes")
    print(f"  - With hide_inner_tensors=False: {len([n for n in view2.all_nodes.values() if view2._is_node_visible(n)])} visible nodes")


def test_multi_input_model():
    """Test with multi-input model"""
    print("\nTesting multi-input model...")
    
    input1 = keras.Input(shape=(32,))
    input2 = keras.Input(shape=(16,))
    
    x1 = layers.Dense(64)(input1)
    x2 = layers.Dense(64)(input2)
    
    combined = layers.Concatenate()([x1, x2])
    outputs = layers.Dense(10)(combined)
    
    model = keras.Model([input1, input2], outputs)
    
    view = ModelView(model, input_shape=[(32,), (16,)])
    
    assert len(view.input_nodes) >= 2, f"Expected at least 2 input nodes, got {len(view.input_nodes)}"
    print(f"✓ Multi-input model created with {len(view.input_nodes)} input nodes")


def test_visualization_options():
    """Test different visualization options"""
    print("\nTesting visualization options...")
    
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(10,)),
        layers.Dense(10, activation='softmax')
    ])
    
    # Test different depth settings
    view_depth1 = ModelView(model, input_shape=(10,), depth=1)
    view_depth3 = ModelView(model, input_shape=(10,), depth=3)
    
    # Test expand_nested
    view_expanded = ModelView(model, input_shape=(10,), expand_nested=True)
    view_collapsed = ModelView(model, input_shape=(10,), expand_nested=False)
    
    print(f"✓ All visualization option combinations work")
    print(f"  - depth=1: {len(view_depth1.all_nodes)} nodes")
    print(f"  - depth=3: {len(view_depth3.all_nodes)} nodes")
    print(f"  - expand_nested=True: {len(view_expanded.all_nodes)} nodes")
    print(f"  - expand_nested=False: {len(view_collapsed.all_nodes)} nodes")


def test_nested_model():
    """Test with nested models"""
    print("\nTesting nested model...")
    
    # Create a submodel
    submodel_input = keras.Input(shape=(32,))
    x = layers.Dense(64, activation='relu')(submodel_input)
    x = layers.Dense(32, activation='relu')(x)
    submodel = keras.Model(submodel_input, x, name='SubModel')
    
    # Create main model using submodel
    main_input = keras.Input(shape=(32,))
    x = submodel(main_input)
    outputs = layers.Dense(10, activation='softmax')(x)
    main_model = keras.Model(main_input, outputs, name='MainModel')
    
    # Test with different depth and expand_nested settings
    view1 = ModelView(main_model, input_shape=(32,), depth=1, expand_nested=False)
    view2 = ModelView(main_model, input_shape=(32,), depth=5, expand_nested=True)
    
    print(f"✓ Nested model handled correctly")
    print(f"  - Shallow (depth=1, expand=False): {len(view1.all_nodes)} nodes")
    print(f"  - Deep (depth=5, expand=True): {len(view2.all_nodes)} nodes")


def test_graph_creation():
    """Test that graphviz graph can be created"""
    print("\nTesting graphviz graph creation...")
    
    try:
        import graphviz
        
        model = keras.Sequential([
            layers.Dense(32, input_shape=(10,)),
            layers.Dense(10)
        ])
        
        view = ModelView(model, input_shape=(10,))
        
        # Try creating advanced graph
        graph = view._create_advanced_graph(show_shapes=True)
        
        assert graph is not None
        print(f"✓ Graphviz graph created successfully")
        print(f"  - Graph has {len(graph.body)} nodes/edges")
        
    except ImportError:
        print("⚠ Graphviz not installed, skipping graph creation test")


def run_all_tests():
    """Run all tests"""
    print("="*80)
    print("ModelView Torchview-like Features - Quick Test Suite")
    print("="*80)
    
    try:
        test_imports()
        test_simple_model()
        test_resnet_model()
        test_multi_input_model()
        test_visualization_options()
        test_nested_model()
        test_graph_creation()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
