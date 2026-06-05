import os
import tempfile
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from ToTf import toONNX, fromONNX


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, kernel_size=3)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(4, 10)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_pytorch_to_from_onnx(tmp_path):
    model = TinyNet()
    model.eval()
    dummy = torch.randn(1, 1, 8, 8)

    onnx_path = tmp_path / "tmp_model.onnx"
    # export
    toONNX(model, dummy, str(onnx_path))

    assert os.path.exists(str(onnx_path)), "ONNX file was not created"

    # run inference
    inp = dummy.numpy()
    out = fromONNX(str(onnx_path), inp)
    assert isinstance(out, np.ndarray), "fromONNX should return numpy array"
    assert out.shape[0] == 1 and out.shape[1] == 10
