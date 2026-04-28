"""
Model efficiency analysis utilities.

This module provides functions for measuring model efficiency including:
- FLOPs counting
- Parameter counting
- Latency benchmarking
- Comprehensive efficiency reporting
"""

import time
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn


def count_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: Optional[torch.device] = None
) -> float:
    """
    Count the number of FLOPs (floating point operations) for a model.

    Uses the thop library if available, otherwise falls back to a
    manual estimation based on parameter count.

    Args:
        model: PyTorch model.
        input_size: Input tensor size as (batch, channels, height, width).
        device: Device to run the model on.

    Returns:
        Number of GFLOPs (billions of FLOPs).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    try:
        from thop import profile
        with torch.no_grad():
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = flops / 1e9
    except ImportError:
        total_params = sum(p.numel() for p in model.parameters())
        gflops = total_params * 2 / 1e9

    return gflops


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.

    Returns:
        Total number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def measure_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    n_runs: int = 100,
    warmup_runs: int = 10,
    device: Optional[Union[str, torch.device]] = None
) -> float:
    """
    Measure inference latency of a model.

    Performs warmup runs followed by timed runs to get accurate
    latency measurements.

    Args:
        model: PyTorch model.
        input_size: Input tensor size as (batch, channels, height, width).
        n_runs: Number of timed inference runs.
        warmup_runs: Number of warmup runs before timing.
        device: Device to run inference on.

    Returns:
        Average latency in milliseconds per frame.
    """
    if device is None:
        device = "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(input_size).to(device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)

    avg_latency = sum(times) / len(times)
    return avg_latency


def efficiency_report(
    models_dict: Dict[str, nn.Module],
    accuracy_dict: Optional[Dict[str, float]] = None,
    f1_dict: Optional[Dict[str, float]] = None,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu"
) -> pd.DataFrame:
    """
    Generate a comprehensive efficiency report for multiple models.

    Args:
        models_dict: Dictionary mapping architecture names to models.
        accuracy_dict: Optional dictionary mapping arch names to accuracy.
        f1_dict: Optional dictionary mapping arch names to F1 scores.
        input_size: Input tensor size for FLOPs and latency measurement.
        device: Device to run benchmarks on.

    Returns:
        DataFrame with columns: Architecture, Accuracy, Macro_F1,
        FLOPs_G, Params_M, Latency_ms
    """
    device_obj = torch.device(device)

    results = []
    for arch_name, model in models_dict.items():
        model = model.to(device_obj)
        model.eval()

        flops = count_flops(model, input_size, device_obj)
        params = count_params(model) / 1e6
        latency = measure_latency(model, input_size, n_runs=50, device=device_obj)

        accuracy = accuracy_dict.get(arch_name, 0.0) if accuracy_dict else 0.0
        f1 = f1_dict.get(arch_name, 0.0) if f1_dict else 0.0

        results.append({
            "Architecture": arch_name,
            "Accuracy": f"{accuracy*100:.2f}%",
            "Macro_F1": f"{f1*100:.2f}%",
            "FLOPs_G": f"{flops:.2f}",
            "Params_M": f"{params:.2f}",
            "Latency_ms": f"{latency:.2f}",
        })

    df = pd.DataFrame(results)
    return df


def get_model_summary(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Get a summary of model characteristics.

    Args:
        model: PyTorch model.
        input_size: Input tensor size.
        device: Device for benchmarking.

    Returns:
        Dictionary with model summary information.
    """
    device_obj = torch.device(device)
    model = model.to(device_obj)

    total_params = count_params(model, trainable_only=False)
    trainable_params = count_params(model, trainable_only=True)
    flops = count_flops(model, input_size, device_obj)
    latency = measure_latency(model, input_size, n_runs=50, device=device_obj)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_M": total_params / 1e6,
        "trainable_params_M": trainable_params / 1e6,
        "flops_G": flops,
        "latency_ms": latency,
        "fps": 1000 / latency if latency > 0 else 0,
    }


def print_efficiency_summary(
    model: nn.Module,
    model_name: str = "Model",
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu"
) -> None:
    """
    Print a formatted efficiency summary for a model.

    Args:
        model: PyTorch model.
        model_name: Name of the model for display.
        input_size: Input tensor size.
        device: Device for benchmarking.
    """
    summary = get_model_summary(model, input_size, device)

    print(f"\n{'='*50}")
    print(f"  {model_name} - Efficiency Summary")
    print(f"{'='*50}")
    print(f"  Total Parameters:     {summary['total_params']:,}")
    print(f"  Trainable Parameters: {summary['trainable_params']:,}")
    print(f"  Parameters (M):       {summary['total_params_M']:.2f}")
    print(f"  FLOPs (G):            {summary['flops_G']:.2f}")
    print(f"  Latency (ms):         {summary['latency_ms']:.2f}")
    print(f"  Throughput (FPS):     {summary['fps']:.1f}")
    print(f"{'='*50}\n")


def compare_models_efficiency(
    models_dict: Dict[str, nn.Module],
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu"
) -> pd.DataFrame:
    """
    Compare efficiency metrics across multiple models.

    Args:
        models_dict: Dictionary mapping model names to models.
        input_size: Input tensor size.
        device: Device for benchmarking.

    Returns:
        DataFrame with efficiency comparison.
    """
    results = []
    for name, model in models_dict.items():
        summary = get_model_summary(model, input_size, device)
        results.append({
            "Model": name,
            "Params (M)": f"{summary['total_params_M']:.2f}",
            "FLOPs (G)": f"{summary['flops_G']:.2f}",
            "Latency (ms)": f"{summary['latency_ms']:.2f}",
            "FPS": f"{summary['fps']:.1f}",
        })

    return pd.DataFrame(results)
