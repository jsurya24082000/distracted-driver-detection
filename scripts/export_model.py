#!/usr/bin/env python
"""
Model export script for Distracted Driver Detection.

This script exports trained models to ONNX and TorchScript formats
for deployment in production environments.

Usage:
    python scripts/export_model.py --arch efficientnet_b0 --checkpoint outputs/checkpoints/best_efficientnet_b0.pth
    python scripts/export_model.py --arch resnet50 --checkpoint ... --export_dir outputs/exports
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from models.model_factory import get_model
from utils import ensure_dir, get_device, load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export distracted driver detection models"
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["efficientnet_b0", "mobilenet_v3_small", "resnet50"],
        help="Model architecture to export",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default="outputs/exports",
        help="Directory to save exported models",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--no_onnx",
        action="store_true",
        help="Skip ONNX export",
    )
    parser.add_argument(
        "--no_torchscript",
        action="store_true",
        help="Skip TorchScript export",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Input image size (default: 224)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version (default: 11)",
    )
    return parser.parse_args()


def export_to_onnx(
    model: nn.Module,
    export_path: Path,
    img_size: int = 224,
    opset_version: int = 11,
    device: torch.device = torch.device("cpu"),
) -> bool:
    """
    Export model to ONNX format.

    Args:
        model: PyTorch model.
        export_path: Path to save the ONNX model.
        img_size: Input image size.
        opset_version: ONNX opset version.
        device: Device for export.

    Returns:
        True if export successful, False otherwise.
    """
    print(f"\nExporting to ONNX: {export_path}")

    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(export_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        print(f"  ONNX export successful!")

        try:
            import onnx
            import onnxruntime as ort
            import numpy as np

            onnx_model = onnx.load(str(export_path))
            onnx.checker.check_model(onnx_model)
            print(f"  ONNX model validation: PASSED")

            ort_session = ort.InferenceSession(str(export_path))

            input_name = ort_session.get_inputs()[0].name
            input_shape = ort_session.get_inputs()[0].shape
            output_name = ort_session.get_outputs()[0].name
            output_shape = ort_session.get_outputs()[0].shape

            print(f"  Input name: {input_name}, shape: {input_shape}")
            print(f"  Output name: {output_name}, shape: {output_shape}")

            test_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
            ort_outputs = ort_session.run(None, {input_name: test_input})
            print(f"  ONNX Runtime inference test: PASSED")
            print(f"  Output shape: {ort_outputs[0].shape}")

        except ImportError:
            print("  Warning: onnx or onnxruntime not installed, skipping validation")
        except Exception as e:
            print(f"  Warning: ONNX validation failed: {e}")

        return True

    except Exception as e:
        print(f"  Error: ONNX export failed: {e}")
        return False


def export_to_torchscript(
    model: nn.Module,
    export_path: Path,
    img_size: int = 224,
    device: torch.device = torch.device("cpu"),
    use_trace: bool = True,
) -> bool:
    """
    Export model to TorchScript format.

    Args:
        model: PyTorch model.
        export_path: Path to save the TorchScript model.
        img_size: Input image size.
        device: Device for export.
        use_trace: Use tracing (True) or scripting (False).

    Returns:
        True if export successful, False otherwise.
    """
    print(f"\nExporting to TorchScript: {export_path}")

    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    try:
        if use_trace:
            scripted_model = torch.jit.trace(model, dummy_input)
        else:
            scripted_model = torch.jit.script(model)

        scripted_model.save(str(export_path))
        print(f"  TorchScript export successful!")

        loaded_model = torch.jit.load(str(export_path))
        loaded_model.eval()

        with torch.no_grad():
            test_output = loaded_model(dummy_input)
            print(f"  TorchScript inference test: PASSED")
            print(f"  Output shape: {test_output.shape}")

        return True

    except Exception as e:
        print(f"  Error: TorchScript export failed: {e}")
        return False


def main():
    """Main export function."""
    args = parse_args()

    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config
    config = load_config(config_path)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    export_dir = Path(args.export_dir)
    if not export_dir.is_absolute():
        export_dir = project_root / export_dir
    ensure_dir(export_dir)

    device = torch.device("cpu")

    print(f"\n{'='*60}")
    print(f"MODEL EXPORT: {args.arch}")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Export directory: {export_dir}")
    print(f"Image size: {args.img_size}")

    print(f"\nLoading model: {args.arch}")
    model = get_model(
        arch_name=args.arch,
        num_classes=config["data"]["num_classes"],
        pretrained=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")

    results = {"architecture": args.arch, "exports": {}}

    if not args.no_onnx:
        onnx_path = export_dir / f"{args.arch}.onnx"
        success = export_to_onnx(
            model=model,
            export_path=onnx_path,
            img_size=args.img_size,
            opset_version=args.opset,
            device=device,
        )
        results["exports"]["onnx"] = {
            "path": str(onnx_path),
            "success": success,
        }

    if not args.no_torchscript:
        ts_path = export_dir / f"{args.arch}.pt"
        success = export_to_torchscript(
            model=model,
            export_path=ts_path,
            img_size=args.img_size,
            device=device,
        )
        results["exports"]["torchscript"] = {
            "path": str(ts_path),
            "success": success,
        }

    print(f"\n{'='*60}")
    print("EXPORT SUMMARY")
    print(f"{'='*60}")

    for format_name, info in results["exports"].items():
        status = "SUCCESS" if info["success"] else "FAILED"
        print(f"  {format_name.upper()}: {status}")
        if info["success"]:
            print(f"    Path: {info['path']}")

    print(f"\nExported models saved to: {export_dir}")


if __name__ == "__main__":
    main()
