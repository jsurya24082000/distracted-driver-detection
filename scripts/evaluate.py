#!/usr/bin/env python
"""
Evaluation script for Distracted Driver Detection models.

This script provides a CLI interface for evaluating trained models
on the State Farm dataset and optionally on the AUC dataset for
domain generalization testing.

Usage:
    python scripts/evaluate.py --arch efficientnet_b0 --checkpoint outputs/checkpoints/best_efficientnet_b0.pth
    python scripts/evaluate.py --arch efficientnet_b0 --checkpoint ... --auc_test
    python scripts/evaluate.py --arch efficientnet_b0 --checkpoint ... --domain_gap
    python scripts/evaluate.py --arch efficientnet_b0 --checkpoint ... --cam_quality
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from data.dataset import AUCDataset, DriverDataset
from data.transforms import val_transforms
from evaluation.efficiency import efficiency_report, get_model_summary
from evaluation.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_per_class_metrics,
    print_metrics_summary,
)
from evaluation.domain_generalization import DomainGeneralizationEvaluator
from evaluation.cam_quality import CAMQualityEvaluator
from models.model_factory import get_model, get_target_layer
from utils import (
    ensure_dir,
    get_device,
    get_short_class_names,
    load_config,
    save_json,
    set_seed,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate distracted driver detection models"
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["efficientnet_b0", "mobilenet_v3_small", "resnet50"],
        help="Model architecture to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory from config",
    )
    parser.add_argument(
        "--auc_test",
        action="store_true",
        help="Also run zero-shot evaluation on AUC dataset",
    )
    parser.add_argument(
        "--auc_dir",
        type=str,
        default=None,
        help="Path to AUC dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--domain_gap",
        action="store_true",
        help="Run domain generalization analysis with detailed gap report",
    )
    parser.add_argument(
        "--cam_quality",
        action="store_true",
        help="Run CAM quality evaluation with Pointing Game metric",
    )
    parser.add_argument(
        "--n_cam_samples",
        type=int,
        default=50,
        help="Number of samples per class for CAM quality evaluation",
    )
    return parser.parse_args()


def evaluate_on_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dataset_name: str = "Dataset",
) -> dict:
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model.
        dataloader: DataLoader for the dataset.
        device: Device to run evaluation on.
        dataset_name: Name of the dataset for logging.

    Returns:
        Dictionary with predictions, labels, and metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    print(f"\nEvaluating on {dataset_name}...")

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch[0], batch[1]
            images = images.to(device)

            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    metrics = compute_metrics(all_labels, all_preds, num_classes=10)

    return {
        "predictions": all_preds,
        "labels": all_labels,
        "metrics": metrics,
    }


def main():
    """Main evaluation function."""
    args = parse_args()

    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config
    config = load_config(config_path)

    set_seed(config["training"]["seed"])

    device = get_device()
    print(f"\nUsing device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print(f"\nLoading model: {args.arch}")
    model = get_model(
        arch_name=args.arch,
        num_classes=config["data"]["num_classes"],
        pretrained=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Checkpoint val accuracy: {checkpoint.get('val_accuracy', 0)*100:.2f}%")

    data_dir = Path(args.data_dir) if args.data_dir else Path(config["data"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    output_dir = Path(args.output_dir) if args.output_dir else project_root / "outputs" / "results"
    ensure_dir(output_dir)

    img_size = config["data"]["img_size"]
    num_workers = config["data"]["num_workers"]

    print("\nLoading State Farm validation dataset...")
    val_dataset = DriverDataset(
        root_dir=data_dir,
        split="val",
        transform=val_transforms(img_size),
        train_ratio=config["data"]["train_split"],
        seed=config["training"]["seed"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"  Validation samples: {len(val_dataset)}")

    sf_results = evaluate_on_dataset(model, val_loader, device, "State Farm Val Set")

    class_names = get_short_class_names()
    print_metrics_summary(sf_results["metrics"], class_names, f"{args.arch} (State Farm)")

    plot_confusion_matrix(
        y_true=sf_results["labels"],
        y_pred=sf_results["predictions"],
        class_names=class_names,
        save_path=output_dir / f"{args.arch}_confusion_matrix.png",
        show=False,
    )

    plot_per_class_metrics(
        sf_results["metrics"],
        class_names,
        save_path=output_dir / f"{args.arch}_per_class_metrics.png",
        show=False,
    )

    all_results = {
        "architecture": args.arch,
        "checkpoint": str(checkpoint_path),
        "state_farm": {
            "accuracy": sf_results["metrics"]["accuracy"],
            "macro_f1": sf_results["metrics"]["macro_f1"],
            "per_class_f1": sf_results["metrics"]["per_class_f1"],
            "per_class_precision": sf_results["metrics"]["per_class_precision"],
            "per_class_recall": sf_results["metrics"]["per_class_recall"],
        },
    }

    if args.auc_test:
        auc_dir = Path(args.auc_dir) if args.auc_dir else Path(config["data"]["auc_dir"])
        if not auc_dir.is_absolute():
            auc_dir = project_root / auc_dir

        if auc_dir.exists():
            print("\n" + "=" * 50)
            print("AUC DATASET EVALUATION (Zero-Shot)")
            print("=" * 50)

            print("\nLoading AUC dataset and computing normalization stats...")
            auc_dataset = AUCDataset(
                root_dir=auc_dir,
                compute_stats=True,
                n_stat_samples=200,
                img_size=img_size,
            )

            if len(auc_dataset) > 0:
                auc_mean, auc_std = auc_dataset.get_normalization_stats()
                print(f"  AUC samples: {len(auc_dataset)}")
                print(f"  Computed mean: {auc_mean}")
                print(f"  Computed std: {auc_std}")

                auc_loader = DataLoader(
                    auc_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )

                auc_results = evaluate_on_dataset(model, auc_loader, device, "AUC Dataset")

                print_metrics_summary(auc_results["metrics"], class_names, f"{args.arch} (AUC)")

                plot_confusion_matrix(
                    y_true=auc_results["labels"],
                    y_pred=auc_results["predictions"],
                    class_names=class_names,
                    save_path=output_dir / f"{args.arch}_auc_confusion_matrix.png",
                    show=False,
                )

                all_results["auc"] = {
                    "accuracy": auc_results["metrics"]["accuracy"],
                    "macro_f1": auc_results["metrics"]["macro_f1"],
                    "per_class_f1": auc_results["metrics"]["per_class_f1"],
                    "normalization_mean": auc_mean,
                    "normalization_std": auc_std,
                }

                print("\n" + "-" * 50)
                print("DOMAIN GENERALIZATION COMPARISON")
                print("-" * 50)
                sf_acc = sf_results["metrics"]["accuracy"] * 100
                auc_acc = auc_results["metrics"]["accuracy"] * 100
                drop = sf_acc - auc_acc
                print(f"  State Farm Accuracy: {sf_acc:.2f}%")
                print(f"  AUC Accuracy:        {auc_acc:.2f}%")
                print(f"  Accuracy Drop:       {drop:.2f}%")
            else:
                print("  Warning: AUC dataset is empty")
        else:
            print(f"\nWarning: AUC dataset directory not found at {auc_dir}")
            print("Skipping AUC evaluation.")

    print("\n" + "=" * 50)
    print("EFFICIENCY METRICS")
    print("=" * 50)

    summary = get_model_summary(model, input_size=(1, 3, img_size, img_size), device=str(device))
    print(f"  Parameters (M):  {summary['total_params_M']:.2f}")
    print(f"  FLOPs (G):       {summary['flops_G']:.2f}")
    print(f"  Latency (ms):    {summary['latency_ms']:.2f}")
    print(f"  Throughput (FPS): {summary['fps']:.1f}")

    all_results["efficiency"] = {
        "params_M": summary["total_params_M"],
        "flops_G": summary["flops_G"],
        "latency_ms": summary["latency_ms"],
        "fps": summary["fps"],
    }

    results_path = output_dir / f"{args.arch}_results.json"
    save_json(all_results, results_path)
    print(f"\nResults saved to: {results_path}")

    # Domain Gap Analysis
    if args.domain_gap:
        auc_dir = Path(args.auc_dir) if args.auc_dir else Path(config["data"]["auc_dir"])
        if not auc_dir.is_absolute():
            auc_dir = project_root / auc_dir

        if auc_dir.exists():
            print("\n" + "=" * 60)
            print("DOMAIN GENERALIZATION ANALYSIS")
            print("=" * 60)

            evaluator = DomainGeneralizationEvaluator(
                model=model,
                device=device,
                source_class_names=class_names
            )

            # Compute AUC normalization stats
            norm_mean, norm_std = evaluator.compute_normalization_stats(
                auc_dir=auc_dir,
                save_path=output_dir / f"{args.arch}_auc_norm_stats.json"
            )

            # Evaluate on source (State Farm)
            source_results = evaluator.evaluate_source(val_loader)

            # Evaluate on target (AUC) with normalization correction
            target_results = evaluator.evaluate_target_zeroshot(
                auc_dir=auc_dir,
                norm_mean=norm_mean,
                norm_std=norm_std,
                img_size=img_size
            )

            # Compute and display domain gap
            gap_analysis = evaluator.compute_domain_gap(source_results, target_results)

            print(f"\n  Source (State Farm) Accuracy: {source_results['accuracy']*100:.2f}%")
            print(f"  Target (AUC Zero-Shot) Accuracy: {target_results['accuracy']*100:.2f}%")
            print(f"  Accuracy Drop: {gap_analysis['overall_accuracy_drop']*100:.2f}pp")
            print(f"\n  Worst 3 Classes (largest gap):")
            for class_name, gap in gap_analysis['worst_3_classes']:
                print(f"    - {class_name}: {gap*100:.1f}pp drop")
            print(f"\n  Gap Hypothesis:\n  {gap_analysis['gap_hypothesis']}")

            # Plot domain gap
            evaluator.plot_domain_gap(
                source_results=source_results,
                target_results=target_results,
                save_path=output_dir / f"{args.arch}_domain_gap.png",
                show=False
            )

            # Generate detailed report
            evaluator.generate_domain_report(
                source_results=source_results,
                target_results=target_results,
                save_path=output_dir / f"{args.arch}_domain_report.json"
            )

            all_results["domain_gap"] = {
                "source_accuracy": source_results["accuracy"],
                "target_accuracy": target_results["accuracy"],
                "accuracy_drop_pp": gap_analysis["overall_accuracy_drop"] * 100,
                "worst_3_classes": gap_analysis["worst_3_classes"],
                "gap_hypothesis": gap_analysis["gap_hypothesis"]
            }
        else:
            print(f"\nWarning: AUC dataset not found at {auc_dir}")
            print("Skipping domain gap analysis.")

    # CAM Quality Evaluation
    if args.cam_quality:
        print("\n" + "=" * 60)
        print("CAM QUALITY EVALUATION (Pointing Game)")
        print("=" * 60)

        target_layer = get_target_layer(model, args.arch)
        cam_method = "eigencam" if "mobilenet" in args.arch.lower() else "gradcam"

        cam_evaluator = CAMQualityEvaluator(
            model=model,
            device=device,
            arch_name=args.arch,
            target_layer=target_layer,
            class_names=class_names
        )

        cam_results = cam_evaluator.evaluate_cam_quality(
            val_loader=val_loader,
            cam_method=cam_method,
            n_samples_per_class=args.n_cam_samples
        )

        print(f"\n  CAM Method: {cam_results['method']}")
        print(f"  Overall Pointing Accuracy: {cam_results['overall_pointing_accuracy']*100:.1f}%")
        print(f"  Random Baseline: 50.0%")
        print(f"  Beats Baseline: {'Yes' if cam_results['overall_pointing_accuracy'] > 0.5 else 'No'}")

        print(f"\n  Per-Class Pointing Accuracy:")
        for class_name, acc in cam_results['per_class_accuracy'].items():
            status = "✓" if acc > 0.5 else "✗"
            print(f"    {status} {class_name}: {acc*100:.1f}%")

        # Plot results
        cam_evaluator.plot_pointing_game_results(
            results_dict={f"{args.arch}_{cam_method}": cam_results},
            save_path=output_dir / f"{args.arch}_pointing_game.png",
            show=False
        )

        # Generate report
        cam_evaluator.generate_quality_report(
            all_arch_results={f"{args.arch}_{cam_method}": cam_results},
            save_path=output_dir / f"{args.arch}_cam_quality_report.json"
        )

        all_results["cam_quality"] = {
            "method": cam_results["method"],
            "overall_pointing_accuracy": cam_results["overall_pointing_accuracy"],
            "per_class_accuracy": cam_results["per_class_accuracy"],
            "beats_random_baseline": cam_results["overall_pointing_accuracy"] > 0.5
        }

    # Save updated results with all analyses
    save_json(all_results, results_path)
    print(f"\nAll results saved to: {results_path}")


if __name__ == "__main__":
    main()
