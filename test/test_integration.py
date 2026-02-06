"""
Comprehensive integration test for ToTf TrainingMonitor
Tests all major functionality including edge cases
"""

# Import PyTorch first before our module 
import torch
import torch.nn as nn

import sys
import os
import csv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ToTf import TrainingMonitor, get_backend


def test_running_averages():
    """Test that running averages are calculated correctly"""
    print("\n" + "="*60)
    print("TEST 1: Running Average Calculation")
    print("="*60)
    
    data = list(range(5))
    monitor = TrainingMonitor(data, desc="Average Test", log_file="test_avg.csv")
    
    expected_losses = []
    for i, item in enumerate(monitor):
        loss_value = float(i + 1)  # 1, 2, 3, 4, 5
        monitor.log({'loss': loss_value})
        
        # Calculate expected running average
        expected_avg = sum(range(1, i + 2)) / (i + 1)
        expected_losses.append(expected_avg)
    
    # Read the CSV and verify
    with open("test_avg.csv", 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    print(f"Total rows logged: {len(rows)}")
    for i, row in enumerate(rows):
        actual_avg = float(row['loss'])
        expected_avg = expected_losses[i]
        print(f"Step {i}: Expected={expected_avg:.4f}, Actual={actual_avg:.4f}, Match={abs(actual_avg - expected_avg) < 0.0001}")
        assert abs(actual_avg - expected_avg) < 0.0001, f"Running average mismatch at step {i}"
    
    print("✓ Running average test PASSED")
    os.remove("test_avg.csv")


def test_multiple_metrics():
    """Test logging multiple metrics simultaneously"""
    print("\n" + "="*60)
    print("TEST 2: Multiple Metrics")
    print("="*60)
    
    data = list(range(3))
    monitor = TrainingMonitor(data, desc="Multi-Metric Test", log_file="test_multi.csv")
    
    for i, item in enumerate(monitor):
        monitor.log({
            'loss': float(i + 1),
            'accuracy': float(i * 10),
            'lr': 0.001
        })
    
    # Verify all metrics are logged
    with open("test_multi.csv", 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    print(f"Columns: {rows[0].keys()}")
    assert 'loss' in rows[0], "Loss not logged"
    assert 'accuracy' in rows[0], "Accuracy not logged"
    assert 'lr' in rows[0], "Learning rate not logged"
    assert 'RAM_pct' in rows[0], "RAM not logged"
    assert 'VRAM_gb' in rows[0], "VRAM not logged"
    
    print("✓ Multiple metrics test PASSED")
    os.remove("test_multi.csv")


def test_resource_monitoring():
    """Test RAM and VRAM monitoring"""
    print("\n" + "="*60)
    print("TEST 3: Resource Monitoring")
    print("="*60)
    
    data = list(range(2))
    monitor = TrainingMonitor(data, desc="Resource Test", log_file="test_resource.csv")
    
    for item in monitor:
        monitor.log({'dummy': 1.0})
    
    with open("test_resource.csv", 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    for row in rows:
        ram = float(row['RAM_pct'])
        vram = float(row['VRAM_gb'])
        
        assert 0 <= ram <= 100, f"Invalid RAM value: {ram}"
        assert vram >= 0, f"Invalid VRAM value: {vram}"
        
        print(f"RAM: {ram}%, VRAM: {vram}GB")
    
    print("✓ Resource monitoring test PASSED")
    os.remove("test_resource.csv")


def test_backend_detection():
    """Test backend detection"""
    print("\n" + "="*60)
    print("TEST 4: Backend Detection")
    print("="*60)
    
    backend = get_backend()
    print(f"Detected backend: {backend}")
    assert backend in ['torch', 'tensorflow'], f"Invalid backend: {backend}"
    assert backend == 'torch', "Expected PyTorch backend"
    
    print("✓ Backend detection test PASSED")


def test_cuda_compatibility():
    """Test CUDA compatibility"""
    print("\n" + "="*60)
    print("TEST 5: CUDA Compatibility")
    print("="*60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Test that monitor works regardless of CUDA
    data = list(range(2))
    monitor = TrainingMonitor(data, desc="CUDA Test", log_file="test_cuda.csv")
    
    for item in monitor:
        monitor.log({'loss': 1.0})
    
    print("✓ CUDA compatibility test PASSED")
    os.remove("test_cuda.csv")


def test_empty_iteration():
    """Test with empty iterable"""
    print("\n" + "="*60)
    print("TEST 6: Empty Iteration")
    print("="*60)
    
    data = []
    monitor = TrainingMonitor(data, desc="Empty Test", log_file="test_empty.csv")
    
    count = 0
    for item in monitor:
        count += 1
    
    assert count == 0, "Empty iterable should not iterate"
    print("✓ Empty iteration test PASSED")
    
    # Clean up if file exists
    if os.path.exists("test_empty.csv"):
        os.remove("test_empty.csv")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print(" ToTf Integration Tests")
    print("="*70)
    
    try:
        test_backend_detection()
        test_running_averages()
        test_multiple_metrics()
        test_resource_monitoring()
        test_cuda_compatibility()
        test_empty_iteration()
        
        print("\n" + "="*70)
        print(" ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nThe TrainingMonitor implementation is working correctly!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
