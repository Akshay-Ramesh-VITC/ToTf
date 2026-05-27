import pytest

# PyTorch tests
try:
    import torch
    import torch.nn as nn
    from ToTf.pytorch.smartsummary import SmartSummary as PT_SmartSummary
    has_torch = True
except Exception:
    has_torch = False

# TensorFlow tests
try:
    import tensorflow as tf
    from ToTf.tenf.smartsummary import SmartSummary as TF_SmartSummary
    has_tf = True
except Exception:
    has_tf = False


@pytest.mark.skipif(not has_torch, reason="PyTorch not available")
def test_pytorch_smartsummary_features():
    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
            self.fc = nn.Linear(16 * 16 * 16, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = SmallCNN()
    ss = PT_SmartSummary(
        model,
        input_size=(3, 32, 32),
        batch_size=1,
        device='cpu',
        track_gradients=True,
        keep_activations=True,
        grad_large_threshold=1e2,
    )

    d = ss.to_dict()
    # basic fields
    assert 'layers' in d
    assert 'gradient_stats' in d
    assert 'receptive_field' in d
    assert 'memory_profile' in d
    # receptive field entries present for conv layers
    rf_keys = [k for k in d['receptive_field'].keys()]
    assert any('Conv2d' in k or 'conv' in k for k in rf_keys)


@pytest.mark.skipif(not has_tf, reason="TensorFlow not available")
def test_tf_smartsummary_features():
    inputs = tf.keras.layers.Input(shape=(28, 28, 3))
    x = tf.keras.layers.Conv2D(8, 3, padding='same')(inputs)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(16, 3, padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    ss = TF_SmartSummary(model, input_shape=(28, 28, 3), batch_size=1, track_gradients=True, keep_activations=True)
    d = ss.to_dict()

    assert 'layers' in d
    assert 'gradient_stats' in d
    assert 'receptive_field' in d
    assert 'memory_profile' in d
    # check receptive field numeric values
    for v in d['receptive_field'].values():
        assert 'rf' in v and 'jump' in v and 'start' in v
    # memory_profile and saved activations
    assert isinstance(d.get('memory_profile'), dict)
    assert 'activation_bytes_estimate' in d['memory_profile']
