"""
Comprehensive test suite for TensorFlow ModelView

Tests cover:
- Basic Sequential models
- Complex Functional API models
- Multi-input/multi-output models
- Residual/skip connections
- Various layer types
- Output format validation
- Shape inference
- Parameter counting
- Graph structure validation
"""

import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import json
from pathlib import Path
import tempfile
import shutil

# Import the module to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tenf.modelview import ModelView, draw_graph


class TestModelViewBasic:
    """Test basic functionality of ModelView"""
    
    def test_import(self):
        """Test that ModelView can be imported"""
        from tenf.modelview import ModelView, draw_graph
        assert ModelView is not None
        assert draw_graph is not None
    
    def test_sequential_model_initialization(self):
        """Test ModelView with a simple Sequential model"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(100,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        view = ModelView(model, input_shape=(100,))
        
        assert view is not None
        assert len(view.layer_info) == 3
        assert view.model == model
    
    def test_functional_model_initialization(self):
        """Test ModelView with Functional API model"""
        inputs = keras.Input(shape=(224, 224, 3))
        x = layers.Conv2D(32, 3, activation='relu')(inputs)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(10, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        view = ModelView(model, input_shape=(224, 224, 3))
        
        assert view is not None
        assert len(view.layer_info) > 0
    
    def test_layer_info_extraction(self):
        """Test that layer information is correctly extracted"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(100,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        view = ModelView(model, input_shape=(100,))
        
        # Check layer info structure
        for layer_id, info in view.layer_info.items():
            assert 'name' in info
            assert 'type' in info
            assert 'total_params' in info
            assert 'trainable_params' in info
            assert 'non_trainable_params' in info
    
    def test_parameter_counting(self):
        """Test that parameters are counted correctly"""
        model = keras.Sequential([
            layers.Dense(64, input_shape=(100,)),  # 100*64 + 64 = 6464
            layers.Dense(10)  # 64*10 + 10 = 650
        ])
        
        view = ModelView(model, input_shape=(100,))
        
        total_params = sum(info['total_params'] for info in view.layer_info.values())
        expected_params = 6464 + 650
        
        assert total_params == expected_params
    
    def test_shape_inference(self):
        """Test that output shapes are correctly inferred"""
        model = keras.Sequential([
            layers.Dense(64, input_shape=(100,)),
            layers.Dense(32),
            layers.Dense(10)
        ])
        
        view = ModelView(model, input_shape=(100,))
        
        # Check that shapes were inferred
        assert len(view.output_shapes) > 0
        
        # Check that at least some shapes are not None
        non_none_shapes = [s for s in view.output_shapes.values() if s is not None]
        assert len(non_none_shapes) > 0
        
        # For shapes that were inferred, verify they have correct structure
        for shape in non_none_shapes:
            if isinstance(shape, list) and len(shape) > 0:
                assert isinstance(shape[-1], int)


class TestModelViewArchitectures:
    """Test ModelView with various complex architectures"""
    
    def test_cnn_model(self):
        """Test with a CNN model"""
        model = keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        view = ModelView(model, input_shape=(28, 28, 1))
        
        assert len(view.layer_info) == 8
        assert any('Conv2D' in info['type'] for info in view.layer_info.values())
        assert any('MaxPooling2D' in info['type'] for info in view.layer_info.values())
        assert any('Flatten' in info['type'] for info in view.layer_info.values())
    
    def test_residual_model(self):
        """Test with a model containing residual connections"""
        inputs = keras.Input(shape=(32, 32, 3))
        
        # First conv block
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        
        # Residual block
        residual = x
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        
        # Output
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(10, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        view = ModelView(model, input_shape=(32, 32, 3))
        
        assert len(view.layer_info) > 0
        assert any('Add' in info['type'] for info in view.layer_info.values())
    
    def test_multi_input_model(self):
        """Test with multi-input model"""
        # Text input
        text_input = keras.Input(shape=(100,), name='text')
        x1 = layers.Embedding(1000, 64)(text_input)
        x1 = layers.LSTM(32)(x1)
        
        # Image input
        image_input = keras.Input(shape=(28, 28, 1), name='image')
        x2 = layers.Conv2D(32, 3, activation='relu')(image_input)
        x2 = layers.GlobalMaxPooling2D()(x2)
        
        # Concatenate
        combined = layers.Concatenate()([x1, x2])
        outputs = layers.Dense(10, activation='softmax')(combined)
        
        model = keras.Model(inputs=[text_input, image_input], outputs=outputs)
        view = ModelView(model, input_shape=[(100,), (28, 28, 1)])
        
        assert len(view.layer_info) > 0
        assert any('Concatenate' in info['type'] for info in view.layer_info.values())
    
    def test_multi_output_model(self):
        """Test with multi-output model"""
        inputs = keras.Input(shape=(100,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        
        # Two separate outputs
        output1 = layers.Dense(10, activation='softmax', name='classification')(x)
        output2 = layers.Dense(1, activation='sigmoid', name='regression')(x)
        
        model = keras.Model(inputs=inputs, outputs=[output1, output2])
        view = ModelView(model, input_shape=(100,))
        
        assert len(view.layer_info) > 0
    
    def test_rnn_model(self):
        """Test with RNN/LSTM model"""
        model = keras.Sequential([
            layers.Embedding(1000, 128, input_length=50),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(32),
            layers.Dense(10, activation='softmax')
        ])
        
        view = ModelView(model, input_shape=(50,))
        
        assert any('LSTM' in info['type'] for info in view.layer_info.values())
        assert any('Embedding' in info['type'] for info in view.layer_info.values())


class TestModelViewRendering:
    """Test rendering capabilities"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(100,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        return model
    
    def test_render_png(self, simple_model, temp_dir):
        """Test rendering to PNG format"""
        try:
            import graphviz
        except ImportError:
            pytest.skip("graphviz not installed")
        
        view = ModelView(simple_model, input_shape=(100,))
        output_path = os.path.join(temp_dir, 'model.png')
        
        result = view.render(output_path, format='png')
        
        assert os.path.exists(result)
        assert result.endswith('.png')
    
    def test_render_pdf(self, simple_model, temp_dir):
        """Test rendering to PDF format"""
        try:
            import graphviz
        except ImportError:
            pytest.skip("graphviz not installed")
        
        view = ModelView(simple_model, input_shape=(100,))
        output_path = os.path.join(temp_dir, 'model.pdf')
        
        result = view.render(output_path, format='pdf')
        
        assert os.path.exists(result)
        assert result.endswith('.pdf')
    
    def test_render_svg(self, simple_model, temp_dir):
        """Test rendering to SVG format"""
        try:
            import graphviz
        except ImportError:
            pytest.skip("graphviz not installed")
        
        view = ModelView(simple_model, input_shape=(100,))
        output_path = os.path.join(temp_dir, 'model.svg')
        
        result = view.render(output_path, format='svg')
        
        assert os.path.exists(result)
        assert result.endswith('.svg')
    
    def test_render_with_options(self, simple_model, temp_dir):
        """Test rendering with various options"""
        try:
            import graphviz
        except ImportError:
            pytest.skip("graphviz not installed")
        
        view = ModelView(simple_model, input_shape=(100,))
        output_path = os.path.join(temp_dir, 'model_custom.png')
        
        result = view.render(
            output_path,
            rankdir='LR',
            show_shapes=True,
            show_layer_names=True,
            show_params=True,
            dpi=150
        )
        
        assert os.path.exists(result)
    
    def test_render_without_graphviz(self, simple_model, temp_dir, monkeypatch):
        """Test that appropriate error is raised without graphviz"""
        # Temporarily make graphviz unavailable
        import tenf.modelview
        monkeypatch.setattr(tenf.modelview, 'GRAPHVIZ_AVAILABLE', False)
        
        view = ModelView(simple_model, input_shape=(100,))
        output_path = os.path.join(temp_dir, 'model.png')
        
        with pytest.raises(ImportError, match="Graphviz is required"):
            view.render(output_path)


class TestModelViewOutput:
    """Test various output formats and summaries"""
    
    def test_get_summary_dict(self):
        """Test getting summary as dictionary"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(100,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        view = ModelView(model, input_shape=(100,))
        summary = view.get_summary_dict()
        
        assert 'model_name' in summary
        assert 'num_layers' in summary
        assert 'total_parameters' in summary
        assert 'trainable_parameters' in summary
        assert 'layers' in summary
        assert 'connections' in summary
        
        assert summary['num_layers'] == 3
        assert isinstance(summary['total_parameters'], int)
    
    def test_save_summary_json(self, tmp_path):
        """Test saving summary as JSON"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(100,)),
            layers.Dense(10, activation='softmax')
        ])
        
        view = ModelView(model, input_shape=(100,))
        json_path = tmp_path / 'summary.json'
        
        view.save_summary_json(str(json_path))
        
        assert json_path.exists()
        
        # Load and verify JSON
        with open(json_path) as f:
            data = json.load(f)
        
        assert 'model_name' in data
        assert 'num_layers' in data
    
    def test_show_method(self, capsys):
        """Test the show() method prints correctly"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(100,)),
            layers.Dense(10, activation='softmax')
        ])
        
        view = ModelView(model, input_shape=(100,))
        view.show()
        
        captured = capsys.readouterr()
        assert 'Model:' in captured.out
        assert 'Total Layers:' in captured.out
        assert 'Total Parameters:' in captured.out
    
    def test_show_detailed(self, capsys):
        """Test the show() method with detailed=True"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(100,)),
            layers.Dense(10, activation='softmax')
        ])
        
        view = ModelView(model, input_shape=(100,))
        view.show(detailed=True)
        
        captured = capsys.readouterr()
        assert 'Connections:' in captured.out


class TestDrawGraphFunction:
    """Test the convenience draw_graph function"""
    
    def test_draw_graph_basic(self):
        """Test basic draw_graph functionality"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(100,)),
            layers.Dense(10, activation='softmax')
        ])
        
        # Should not raise an error
        draw_graph(model, input_shape=(100,))
    
    def test_draw_graph_with_save(self, tmp_path):
        """Test draw_graph with save_path"""
        try:
            import graphviz
        except ImportError:
            pytest.skip("graphviz not installed")
        
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(100,)),
            layers.Dense(10, activation='softmax')
        ])
        
        save_path = tmp_path / 'model.png'
        result = draw_graph(model, input_shape=(100,), save_path=str(save_path))
        
        assert result is not None
        assert os.path.exists(result)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_model_without_input_shape(self):
        """Test ModelView without providing input_shape"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        # Should still work, just won't have shape inference
        view = ModelView(model, input_shape=None)
        assert view is not None
    
    def test_empty_model(self):
        """Test with an empty model"""
        model = keras.Sequential()
        
        view = ModelView(model, input_shape=None)
        assert len(view.layer_info) == 0
    
    def test_format_params_helper(self):
        """Test parameter formatting helper"""
        model = keras.Sequential([
            layers.Dense(64, input_shape=(100,))
        ])
        
        view = ModelView(model, input_shape=(100,))
        
        # Test various parameter counts
        assert view._format_params(0) == "0"
        assert view._format_params(500) == "500"
        assert 'K' in view._format_params(5000)
        assert 'M' in view._format_params(5000000)
    
    def test_connections_sequential(self):
        """Test that connections are correctly identified in sequential models"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(100,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        view = ModelView(model, input_shape=(100,))
        
        # Should have n-1 connections for n layers
        assert len(view.connections) == len(view.layer_info) - 1


class TestComplexModels:
    """Test with realistic complex models"""
    
    def test_resnet_like_model(self):
        """Test with a ResNet-like architecture"""
        def residual_block(x, filters):
            residual = x
            
            # If dimensions change, adjust residual with 1x1 conv
            if x.shape[-1] != filters:
                residual = layers.Conv2D(filters, 1, padding='same')(residual)
            
            x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.Add()([x, residual])
            x = layers.Activation('relu')(x)
            return x
        
        inputs = keras.Input(shape=(32, 32, 3))
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
        x = residual_block(x, 32)
        x = residual_block(x, 64)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(10, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        view = ModelView(model, input_shape=(32, 32, 3))
        
        assert len(view.layer_info) > 10
    
    def test_inception_like_model(self):
        """Test with an Inception-like architecture with parallel branches"""
        inputs = keras.Input(shape=(28, 28, 1))
        
        # Branch 1
        branch1 = layers.Conv2D(32, 1, padding='same', activation='relu')(inputs)
        
        # Branch 2
        branch2 = layers.Conv2D(32, 1, padding='same', activation='relu')(inputs)
        branch2 = layers.Conv2D(32, 3, padding='same', activation='relu')(branch2)
        
        # Branch 3
        branch3 = layers.Conv2D(32, 1, padding='same', activation='relu')(inputs)
        branch3 = layers.Conv2D(32, 5, padding='same', activation='relu')(branch3)
        
        # Concatenate branches
        x = layers.Concatenate()([branch1, branch2, branch3])
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(10, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        view = ModelView(model, input_shape=(28, 28, 1))
        
        assert any('Concatenate' in info['type'] for info in view.layer_info.values())
    
    def test_attention_model(self):
        """Test with a model containing attention mechanisms"""
        inputs = keras.Input(shape=(50, 128))
        
        # Self-attention
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
        attention = layers.Add()([inputs, attention])
        attention = layers.LayerNormalization()(attention)
        
        # Feed-forward
        ff = layers.Dense(256, activation='relu')(attention)
        ff = layers.Dense(128)(ff)
        ff = layers.Add()([attention, ff])
        ff = layers.LayerNormalization()(ff)
        
        # Output
        x = layers.GlobalAveragePooling1D()(ff)
        outputs = layers.Dense(10, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        view = ModelView(model, input_shape=(50, 128))
        
        assert any('MultiHeadAttention' in info['type'] for info in view.layer_info.values())


class TestPaperReadyOutput:
    """Test that outputs are suitable for research papers"""
    
    def test_high_dpi_rendering(self, tmp_path):
        """Test rendering at high DPI for publication quality"""
        try:
            import graphviz
        except ImportError:
            pytest.skip("graphviz not installed")
        
        model = keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(10, activation='softmax')
        ])
        
        view = ModelView(model, input_shape=(28, 28, 1))
        output_path = tmp_path / 'high_dpi_model.png'
        
        result = view.render(str(output_path), dpi=300)
        
        assert os.path.exists(result)
    
    def test_pdf_output_for_latex(self, tmp_path):
        """Test PDF output suitable for LaTeX documents"""
        try:
            import graphviz
        except ImportError:
            pytest.skip("graphviz not installed")
        
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(100,)),
            layers.Dense(10, activation='softmax')
        ])
        
        view = ModelView(model, input_shape=(100,))
        output_path = tmp_path / 'paper_model.pdf'
        
        result = view.render(str(output_path), format='pdf')
        
        assert os.path.exists(result)
        assert result.endswith('.pdf')
    
    def test_svg_for_scalability(self, tmp_path):
        """Test SVG output for perfect scaling"""
        try:
            import graphviz
        except ImportError:
            pytest.skip("graphviz not installed")
        
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(100,)),
            layers.Dense(10, activation='softmax')
        ])
        
        view = ModelView(model, input_shape=(100,))
        output_path = tmp_path / 'scalable_model.svg'
        
        result = view.render(str(output_path), format='svg')
        
        assert os.path.exists(result)
        assert result.endswith('.svg')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
