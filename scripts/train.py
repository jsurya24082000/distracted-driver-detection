#!/usr/bin/env python
"""
Training script for Distracted Driver Detection models.

This script provides a CLI interface for training CNN models on the
State Farm Distracted Driver Detection dataset.

Usage:
    python scripts/train.py --arch efficientnet_b0
    python scripts/train.py --arch all --epochs 30
    python scripts/train.py --arch resnet50 --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from data.dataset import DriverDataset
from data.transforms import train_transforms, val_transforms
from evaluation.efficiency import print_efficiency_summary
from evaluation.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    print_metrics_summary,
)
from models.model_factory import get_model
from training.losses import compute_class_weights, get_loss
from training.trainer import Trainer, create_optimizer, create_scheduler
from utils import (
    ensure_dir,
    get_class_names,
    get_device,
    get_short_class_names,
    load_config,
    save_json,
    set_seed,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train distracted driver detection models"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="efficientnet_b0",
        choices=["efficientnet_b0", "mobilenet_v3_small", "resnet50", "all"],
        help="Model architecture to train (default: efficientnet_b0)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory from config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs from config",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    return parser.parse_args()


def train_single_model(
    arch_name: str,
    config: dict,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    resume_path: str = None,
) -> dict:
    """
    Train a single model architecture.

    Args:
        arch_name: Architecture name.
        config: Configuration dictionary.
        data_dir: Path to dataset directory.
        output_dir: Path to output directory.
        device: Device to train on.
        resume_path: Optional path to checkpoint to resume from.

    Returns:
        Dictionary with training results.
    """
    print(f"\n{'#'*60}")
    print(f"# Training {arch_name}")
    print(f"{'#'*60}")

    img_size = config["data"]["img_size"]
    train_ratio = config["data"]["train_split"]
    num_workers = config["data"]["num_workers"]
    num_classes = config["data"]["num_classes"]
    seed = config["training"]["seed"]

    set_seed(seed)

    print("\nLoading datasets...")
    train_dataset = DriverDataset(
        root_dir=data_dir,
        split="train",
        transform=train_transforms(img_size),
        train_ratio=train_ratio,
        seed=seed,
    )

    val_dataset = DriverDataset(
        root_dir=data_dir,
        split="val",
        transform=val_transforms(img_size),
        train_ratio=train_ratio,
        seed=seed,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("\nInitializing model...")
    model = get_model(
        arch_name=arch_name,
        num_classes=num_classes,
        pretrained=config["models"]["pretrained"],
    )
    model = model.to(device)

    print("\nComputing class weights...")
    class_weights = compute_class_weights(train_dataset, num_classes)
    class_weights = class_weights.to(device)

    criterion = get_loss(
        use_class_weights=config["training"]["use_class_weights"],
        class_weights_tensor=class_weights,
        label_smoothing=config["training"]["label_smoothing"],
        device=device,
    )

    optimizer = create_optimizer(
        model=model,
        optimizer_name=config["training"]["optimizer"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_name=config["training"]["scheduler"],
        t_max=config["training"].get("scheduler_t_max", 20),
    )

    checkpoints_dir = ensure_dir(output_dir / "checkpoints")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        save_dir=checkpoints_dir,
        arch_name=arch_name,
    )

    if resume_path:
        trainer.load_checkpoint(resume_path)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["training"]["epochs"],
    )

    print("\nRunning final evaluation...")
    val_metrics = trainer.validate(val_loader)

    metrics = compute_metrics(
        y_true=val_metrics["labels"],
        y_pred=val_metrics["predictions"],
        num_classes=num_classes,
    )

    print_metrics_summary(metrics, get_short_class_names(), arch_name)

    results_dir = ensure_dir(output_dir / "results")

    plot_training_curves(
        history,
        save_path=results_dir / f"{arch_name}_training_curves.png",
        show=False,
    )

    plot_confusion_matrix(
        y_true=val_metrics["labels"],
        y_pred=val_metrics["predictions"],
        class_names=get_short_class_names(),
        save_path=results_dir / f"{arch_name}_confusion_matrix.png",
        show=False,
    )

    results = {
        "architecture": arch_name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_f1": metrics["per_class_f1"],
        "per_class_precision": metrics["per_class_precision"],
        "per_class_recall": metrics["per_class_recall"],
        "best_epoch": trainer.best_epoch,
        "best_val_acc": trainer.best_val_acc,
    }

    save_json(results, results_dir / f"{arch_name}_results.json")

    print_efficiency_summary(model, arch_name, device=str(device))

    return results


def main():
    """Main training function."""
    args = parse_args()

    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config
    config = load_config(config_path)

    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    data_dir = Path(config["data"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    output_dir = Path(args.output_dir) if args.output_dir else project_root / "outputs"
    ensure_dir(output_dir)

    device = get_device()
    print(f"\nUsing device: {device}")

    if args.arch == "all":
        architectures = config["models"]["architectures"]
    else:
        architectures = [args.arch]

    all_results = {}

    for arch in architectures:
        results = train_single_model(
            arch_name=arch,
            config=config,
            data_dir=data_dir,
            output_dir=output_dir,
            device=device,
            resume_path=args.resume,
        )
        all_results[arch] = results

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"{'Architecture':<25} {'Accuracy':>12} {'Macro F1':>12}")
    print("-" * 60)
    for arch, results in all_results.items():
        print(
            f"{arch:<25} {results['accuracy']*100:>11.2f}% {results['macro_f1']*100:>11.2f}%"
        )
    print("=" * 60)

    save_json(all_results, output_dir / "results" / "all_results_summary.json")

    print(f"\nAll results saved to: {output_dir / 'results'}")
    print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
