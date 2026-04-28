"""
Evaluation metrics for model performance analysis.

This module provides functions for computing classification metrics
and generating visualizations like confusion matrices and training curves.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int = 10
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Number of classes.

    Returns:
        Dictionary containing:
        - accuracy: Overall accuracy
        - macro_f1: Macro-averaged F1 score
        - per_class_f1: F1 score for each class
        - per_class_precision: Precision for each class
        - per_class_recall: Recall for each class
        - confusion_matrix: Confusion matrix as numpy array
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=range(num_classes), zero_division=0
    ).tolist()

    per_class_precision = precision_score(
        y_true, y_pred, average=None, labels=range(num_classes), zero_division=0
    ).tolist()

    per_class_recall = recall_score(
        y_true, y_pred, average=None, labels=range(num_classes), zero_division=0
    ).tolist()

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    normalize: bool = True,
    figsize: tuple = (12, 10),
    show: bool = True
) -> plt.Figure:
    """
    Plot and optionally save a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names for axis labels.
        save_path: Path to save the figure. If None, figure is not saved.
        normalize: Whether to normalize the confusion matrix by row.
        figsize: Figure size as (width, height).
        show: Whether to display the plot inline.

    Returns:
        Matplotlib Figure object.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    if normalize:
        cm_normalized = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-6)
        cm_display = cm_normalized
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        cm_display = cm
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_training_curves(
    history_dict: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 5),
    show: bool = True
) -> plt.Figure:
    """
    Plot training and validation curves side by side.

    Creates a figure with two subplots:
    1. Training and validation loss over epochs
    2. Validation accuracy and F1 score over epochs

    Args:
        history_dict: Dictionary with keys 'train_loss', 'val_loss',
                      'val_accuracy', 'val_f1'.
        save_path: Path to save the figure. If None, figure is not saved.
        figsize: Figure size as (width, height).
        show: Whether to display the plot inline.

    Returns:
        Matplotlib Figure object.
    """
    epochs = range(1, len(history_dict.get("train_loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    if "train_loss" in history_dict:
        ax1.plot(epochs, history_dict["train_loss"], "b-", label="Train Loss", linewidth=2)
    if "val_loss" in history_dict:
        ax1.plot(epochs, history_dict["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    if "val_accuracy" in history_dict:
        acc_percent = [a * 100 for a in history_dict["val_accuracy"]]
        ax2.plot(epochs, acc_percent, "g-", label="Val Accuracy (%)", linewidth=2)
    if "val_f1" in history_dict:
        f1_percent = [f * 100 for f in history_dict["val_f1"]]
        ax2.plot(epochs, f1_percent, "m-", label="Val F1 (%)", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Score (%)", fontsize=12)
    ax2.set_title("Validation Metrics", fontsize=14)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_per_class_metrics(
    metrics_dict: Dict[str, Any],
    class_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 6),
    show: bool = True
) -> plt.Figure:
    """
    Plot per-class precision, recall, and F1 scores.

    Args:
        metrics_dict: Dictionary from compute_metrics().
        class_names: List of class names.
        save_path: Path to save the figure.
        figsize: Figure size.
        show: Whether to display the plot.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(class_names))
    width = 0.25

    precision = metrics_dict.get("per_class_precision", [0] * len(class_names))
    recall = metrics_dict.get("per_class_recall", [0] * len(class_names))
    f1 = metrics_dict.get("per_class_f1", [0] * len(class_names))

    bars1 = ax.bar(x - width, precision, width, label="Precision", color="#2ecc71")
    bars2 = ax.bar(x, recall, width, label="Recall", color="#3498db")
    bars3 = ax.bar(x + width, f1, width, label="F1 Score", color="#9b59b6")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Metrics", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def print_metrics_summary(
    metrics_dict: Dict[str, Any],
    class_names: Optional[List[str]] = None,
    model_name: str = "Model"
) -> None:
    """
    Print a formatted summary of classification metrics.

    Args:
        metrics_dict: Dictionary from compute_metrics().
        class_names: Optional list of class names.
        model_name: Name of the model for the header.
    """
    print(f"\n{'='*50}")
    print(f"  {model_name} - Evaluation Results")
    print(f"{'='*50}")
    print(f"  Overall Accuracy: {metrics_dict['accuracy']*100:.2f}%")
    print(f"  Macro F1 Score:   {metrics_dict['macro_f1']*100:.2f}%")
    print(f"{'='*50}")

    if class_names is not None and "per_class_f1" in metrics_dict:
        print(f"\n  Per-Class F1 Scores:")
        print(f"  {'-'*40}")
        for i, (name, f1) in enumerate(zip(class_names, metrics_dict["per_class_f1"])):
            print(f"  {name:25s}: {f1*100:6.2f}%")
        print(f"  {'-'*40}")
