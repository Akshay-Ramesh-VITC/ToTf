import os
import tempfile
import numpy as np
import pytest

# skip if tensorflow not available
pytest.importorskip("tensorflow")
try:
    import tf2onnx
except Exception:
    tf2onnx = None

from ToTf import toONNX, fromONNX

import tensorflow as tf


def test_tensorflow_to_from_onnx(tmp_path):
    if tf2onnx is None:
        pytest.skip("tf2onnx not installed")

    # simple model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(8, 8, 1)),
        tf.keras.layers.Conv2D(4, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10)
    ])

    # build weights by calling model
    dummy = np.random.randn(1, 8, 8, 1).astype(np.float32)
    _ = model(dummy)

    onnx_path = tmp_path / "tf_model.onnx"
    toONNX(model, str(onnx_path), example_input=dummy)

    assert os.path.exists(str(onnx_path))

    out = fromONNX(str(onnx_path), dummy)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 1 and out.shape[1] == 10
