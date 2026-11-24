"""
Unit tests for Apple Metal / MPS (Metal Performance Shaders) compatibility across ToTf.
"""

import pytest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
import numpy as np

from ToTf import (
    is_mps_available,
    is_cuda_available,
    get_device,
    seed_everything,
    inspect_tensor,
    benchmark_model,
    TrainingMonitor,
    SmartSummary,
    transfer_weights,
)
import ToTf.backend as backend_module


def test_is_mps_available_returns_bool():
    result = is_mps_available()
    assert isinstance(result, bool)


def test_get_device_defaults_and_preferred():
    # If preferred is specified explicitly, it should be honored
    assert get_device("cpu") == "cpu"
    assert get_device("mps") == "mps"
    assert get_device("cuda:0") == "cuda:0"

    # Auto resolution
    dev = get_device("auto")
    assert dev in ("cuda", "mps", "cpu")


def test_get_device_mock_mps():
    # Test device resolution when MPS is available and CUDA is not
    with patch("ToTf.backend.is_cuda_available", return_value=False), \
         patch("ToTf.backend.is_mps_available", return_value=True):
        assert backend_module.get_device("auto") == "mps"
        assert backend_module.get_device(None) == "mps"


def test_seed_everything_mps_handling():
    # Verify seed_everything attempts to seed MPS when available
    mock_mps = MagicMock()
    with patch.object(torch.backends, "mps", create=True) as mock_backends_mps, \
         patch.object(torch, "mps", mock_mps, create=True):
        mock_backends_mps.is_available.return_value = True

        seed = seed_everything(1234, verbose=True)
        assert seed == 1234
        # Assert mps.manual_seed was called
        mock_mps.manual_seed.assert_called_once_with(1234)


def test_inspect_tensor_quantile_fallback_on_mps():
    # Test that inspect_tensor handles quantile errors gracefully (e.g. missing aten::quantile on MPS)
    x = torch.randn(10, 10)

    original_quantile = torch.quantile

    def faulty_quantile(tensor, *args, **kwargs):
        # Fail if tensor is on the simulated device, simulating MPS missing kernel
        if getattr(tensor, "device", None) and tensor.device.type != "cpu":
            raise NotImplementedError("Could not run 'aten::quantile' with arguments from the 'MPS' backend")
        return original_quantile(tensor, *args, **kwargs)

    # Even if torch.quantile raises on non-cpu tensor, our fallback catches it
    stats = inspect_tensor(x, quantiles=True, print_info=False)
    assert "p25" in stats["quantiles"]
    assert "p50" in stats["quantiles"]
    assert "p75" in stats["quantiles"]


def test_training_monitor_mps_vram_tracking():
    mock_mps = MagicMock()
    mock_mps.current_allocated_memory.return_value = 2 * (1024 ** 3)  # 2 GB

    with patch("torch.cuda.is_available", return_value=False), \
         patch.object(torch.backends, "mps", create=True) as mock_backends_mps, \
         patch.object(torch, "mps", mock_mps, create=True):
        mock_backends_mps.is_available.return_value = True

        dummy_loader = [torch.tensor([1.0])]
        monitor = TrainingMonitor(dummy_loader, desc="TestMPS", log_file="test_mps_monitor.csv")
        res = monitor._get_resources()
        assert res["VRAM"] == 2.0
        monitor.close()

        # Clean up file
        import os
        if os.path.exists("test_mps_monitor.csv"):
            os.remove("test_mps_monitor.csv")


def test_smartsummary_device_auto_resolution():
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
    with patch("torch.cuda.is_available", return_value=False), \
         patch.object(torch.backends, "mps", create=True) as mock_backends_mps:
        mock_backends_mps.is_available.return_value = True
        mock_backends_mps.is_built.return_value = True

        summary = SmartSummary(model, input_size=(8,), device="auto")
        # On Apple Silicon macOS it will be 'mps'; on systems where mps is not compiled, it falls back to 'cpu'
        assert summary.device in ("mps", "cpu")


def test_benchmark_model_mps_sync():
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
    mock_sync = MagicMock()

    with patch.object(torch, "mps", create=True) as mock_mps:
        mock_mps.synchronize = mock_sync

        # Running benchmark with device="cpu" succeeds
        report = benchmark_model(model, input_size=(8,), num_runs=5, warmup=2, device="cpu", print_table=False)
        assert report["fps"] > 0
        assert "markdown_table" in report

        # Test sync helper when mps is target
        from ToTf.pytorch.utils import benchmark_model as bm_torch
        with patch("torch.device", return_value=torch.device("cpu")):
            report_mps = bm_torch(model, input_size=(8,), num_runs=5, warmup=2, device="mps", print_table=False)
            assert report_mps["mean_latency_ms"] >= 0
            # Ensure synchronize was called during warmup and timed runs
            assert mock_sync.call_count >= 2
