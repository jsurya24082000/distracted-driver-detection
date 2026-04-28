"""
Training loop and validation utilities.

This module provides the Trainer class for managing the training process
including epoch loops, validation, checkpointing, and logging.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import f1_score


class Trainer:
    """
    Trainer class for managing model training and validation.

    Handles the training loop, validation, checkpointing, and logging
    of training metrics.

    Attributes:
        model: PyTorch model to train.
        optimizer: Optimizer for updating model weights.
        scheduler: Learning rate scheduler.
        criterion: Loss function.
        device: Device to train on (cuda/mps/cpu).
        save_dir: Directory for saving checkpoints and logs.
        arch_name: Architecture name for checkpoint naming.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        criterion: nn.Module,
        device: torch.device,
        save_dir: Union[str, Path],
        arch_name: str = "model"
    ):
        """
        Initialize the Trainer.

        Args:
            model: PyTorch model to train.
            optimizer: Optimizer for parameter updates.
            scheduler: Learning rate scheduler (can be None).
            criterion: Loss function.
            device: Device for training.
            save_dir: Directory to save checkpoints and history.
            arch_name: Name of the architecture for file naming.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.save_dir = Path(save_dir)
        self.arch_name = arch_name

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "learning_rate": [],
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Dictionary with 'loss' and 'accuracy' keys.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch in pbar:
            images, labels = batch[0], batch[1]
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100. * correct / total:.2f}%"
            })

        avg_loss = total_loss / total
        accuracy = correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Validate the model.

        Args:
            dataloader: Validation data loader.

        Returns:
            Dictionary with keys: loss, accuracy, f1, per_class_f1.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds: List[int] = []
        all_labels: List[int] = []

        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for batch in pbar:
            images, labels = batch[0], batch[1]
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        avg_loss = total_loss / total
        accuracy = correct / total

        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        per_class_f1 = f1_score(all_labels, all_preds, average=None).tolist()

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": macro_f1,
            "per_class_f1": per_class_f1,
            "predictions": all_preds,
            "labels": all_labels,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Saves the best checkpoint based on validation accuracy and
        logs training history.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs to train.

        Returns:
            Training history dictionary.
        """
        print(f"\n{'='*60}")
        print(f"Training {self.arch_name} for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train_metrics = self.train_epoch(train_loader)

            val_metrics = self.validate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["learning_rate"].append(current_lr)

            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.best_epoch = epoch
                self._save_checkpoint(
                    self.save_dir / f"best_{self.arch_name}.pth",
                    epoch,
                    val_metrics
                )

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_metrics['loss']:.3f} | "
                f"Val Loss: {val_metrics['loss']:.3f} | "
                f"Val Acc: {val_metrics['accuracy']*100:.1f}% | "
                f"Val F1: {val_metrics['f1']:.3f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

        self._save_checkpoint(
            self.save_dir / f"final_{self.arch_name}.pth",
            epochs,
            val_metrics
        )

        history_path = self.save_dir / f"history_{self.arch_name}.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best validation accuracy: {self.best_val_acc*100:.2f}% (epoch {self.best_epoch})")
        print(f"Checkpoints saved to: {self.save_dir}")
        print(f"{'='*60}\n")

        return self.history

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Save a model checkpoint.

        Args:
            path: Path to save the checkpoint.
            epoch: Current epoch number.
            metrics: Validation metrics dictionary.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_accuracy": metrics["accuracy"],
            "val_f1": metrics["f1"],
            "val_loss": metrics["loss"],
            "arch_name": self.arch_name,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a model checkpoint.

        Args:
            path: Path to the checkpoint file.

        Returns:
            Checkpoint dictionary with metadata.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Loaded checkpoint from {path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Accuracy: {checkpoint.get('val_accuracy', 0)*100:.2f}%")

        return checkpoint


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4
) -> Optimizer:
    """
    Create an optimizer for the model.

    Args:
        model: PyTorch model.
        optimizer_name: Name of optimizer ('adamw', 'adam', 'sgd').
        learning_rate: Learning rate.
        weight_decay: Weight decay for regularization.

    Returns:
        Configured optimizer.
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = "cosine",
    t_max: int = 20,
    **kwargs
) -> Optional[_LRScheduler]:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_name: Name of scheduler ('cosine', 'step', 'plateau').
        t_max: T_max parameter for cosine annealing.
        **kwargs: Additional scheduler parameters.

    Returns:
        Configured scheduler or None.
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=1e-6
        )
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 10),
            gamma=kwargs.get("gamma", 0.1)
        )
    elif scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            verbose=True
        )
    elif scheduler_name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
