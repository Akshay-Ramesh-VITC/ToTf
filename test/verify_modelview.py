"""
Quick Verification Script for ModelView

This script tests the ModelView implementation with a simple model
to ensure everything works correctly before running comprehensive tests.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_basic_import():
    """Test that ModelView can be imported"""
    print("Testing import...")
    try:
        from tenf.modelview import ModelView, draw_graph
        print("✓ Import successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_simple_model():
    """Test with a simple Sequential model"""
    print("\nTesting simple Sequential model...")
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tenf.modelview import ModelView
        
        # Create simple model
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(100,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        # Create ModelView
        view = ModelView(model, input_shape=(100,))
        
        # Check basic attributes
        assert len(view.layer_info) == 3, "Should have 3 layers"
        assert len(view.connections) == 2, "Should have 2 connections"
        
        print(f"✓ Model analyzed: {len(view.layer_info)} layers")
        
        # Test summary
        summary = view.get_summary_dict()
        assert 'total_parameters' in summary
        print(f"✓ Summary generated: {summary['total_parameters']:,} parameters")
        
        # Test show method
        print("\n" + "="*60)
        view.show()
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cnn_model():
    """Test with a CNN model"""
    print("\nTesting CNN model...")
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tenf.modelview import ModelView
        
        # Create CNN model
        model = keras.Sequential([
            keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        # Create ModelView
        view = ModelView(model, input_shape=(28, 28, 1))
        
        print(f"✓ CNN analyzed: {len(view.layer_info)} layers")
        
        # Check shapes were inferred
        assert len(view.output_shapes) > 0, "Should have output shapes"
        
        print("✓ Shape inference successful")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_functional_model():
    """Test with Functional API model"""
    print("\nTesting Functional API model...")
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tenf.modelview import ModelView
        
        # Create functional model with skip connection
        inputs = keras.Input(shape=(32, 32, 3))
        x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
        
        # Residual block
        residual = x
        x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = keras.layers.Add()([x, residual])
        
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Create ModelView
        view = ModelView(model, input_shape=(32, 32, 3))
        
        print(f"✓ Functional model analyzed: {len(view.layer_info)} layers")
        
        # Should have Add layer for residual connection
        has_add = any('Add' in info['type'] for info in view.layer_info.values())
        assert has_add, "Should detect Add layer"
        
        print("✓ Skip connection detected")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rendering():
    """Test rendering capability"""
    print("\nTesting rendering...")
    
    # Check if graphviz is available
    try:
        import graphviz
        graphviz_available = True
    except ImportError:
        graphviz_available = False
        print("⚠ Graphviz not installed - skipping render tests")
        print("  Install with: pip install graphviz")
        print("  Also install system graphviz")
        return True
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tenf.modelview import ModelView
        import tempfile
        
        # Create simple model
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(100,)),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        view = ModelView(model, input_shape=(100,))
        
        # Try rendering to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_model.png')
            result = view.render(output_path, dpi=150)
            
            if os.path.exists(result):
                print(f"✓ Rendering successful: {result}")
                file_size = os.path.getsize(result)
                print(f"  File size: {file_size:,} bytes")
                return True
            else:
                print(f"✗ Render file not created: {result}")
                return False
                
    except Exception as e:
        print(f"✗ Rendering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_input():
    """Test multi-input model"""
    print("\nTesting multi-input model...")
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tenf.modelview import ModelView
        
        # Create multi-input model
        input1 = keras.Input(shape=(100,), name='text')
        input2 = keras.Input(shape=(28, 28, 1), name='image')
        
        x1 = keras.layers.Dense(32, activation='relu')(input1)
        x2 = keras.layers.Flatten()(input2)
        x2 = keras.layers.Dense(32, activation='relu')(x2)
        
        merged = keras.layers.Concatenate()([x1, x2])
        outputs = keras.layers.Dense(10, activation='softmax')(merged)
        
        model = keras.Model(inputs=[input1, input2], outputs=outputs)
        
        # Create ModelView with multiple input shapes
        view = ModelView(model, input_shape=[(100,), (28, 28, 1)])
        
        print(f"✓ Multi-input model analyzed: {len(view.layer_info)} layers")
        
        # Should have Concatenate layer
        has_concat = any('Concatenate' in info['type'] for info in view.layer_info.values())
        assert has_concat, "Should detect Concatenate layer"
        
        print("✓ Multi-input architecture detected")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests"""
    print("="*80)
    print("ModelView Verification Tests")
    print("="*80)
    
    tests = [
        ("Import", test_basic_import),
        ("Simple Model", test_simple_model),
        ("CNN Model", test_cnn_model),
        ("Functional Model", test_functional_model),
        ("Multi-Input Model", test_multi_input),
        ("Rendering", test_rendering),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("="*80)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All verification tests passed! ModelView is working correctly.")
        print("\nNext steps:")
        print("  1. Run full test suite: python -m pytest test/test_modelview_tf.py -v")
        print("  2. Try examples: python example_modelview_tf.py")
        print("  3. Generate diagrams for your own models!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
