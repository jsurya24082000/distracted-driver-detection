"""
Loss functions for training distracted driver detection models.

This module provides loss function utilities including:
- Weighted cross-entropy loss
- Label smoothing
- Class weight computation
"""

from collections import Counter
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset


def get_loss(
    use_class_weights: bool = True,
    class_weights_tensor: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.1,
    device: Optional[torch.device] = None
) -> nn.CrossEntropyLoss:
    """
    Create a cross-entropy loss function with optional class weights and label smoothing.

    Class weights help handle class imbalance by giving more importance to
    underrepresented classes. Label smoothing helps prevent overconfident
    predictions and improves generalization.

    Args:
        use_class_weights: Whether to use class weights.
        class_weights_tensor: Precomputed class weights tensor.
                              Required if use_class_weights is True.
        label_smoothing: Label smoothing factor (0.0 to 1.0).
                         0.0 means no smoothing.
        device: Device to place the weights tensor on.

    Returns:
        Configured CrossEntropyLoss instance.
    """
    weight = None
    if use_class_weights and class_weights_tensor is not None:
        weight = class_weights_tensor
        if device is not None:
            weight = weight.to(device)

    return nn.CrossEntropyLoss(
        weight=weight,
        label_smoothing=label_smoothing
    )


def compute_class_weights(
    dataset: Dataset,
    num_classes: int = 10,
    method: str = "inverse_freq"
) -> torch.Tensor:
    """
    Compute class weights based on label distribution.

    Computes inverse-frequency weights so that underrepresented classes
    receive higher weights during training.

    Args:
        dataset: PyTorch Dataset with get_labels() method or indexable
                 samples returning (image, label, ...).
        num_classes: Number of classes.
        method: Weighting method. Options:
                - "inverse_freq": Weight = 1 / frequency
                - "inverse_sqrt": Weight = 1 / sqrt(frequency)
                - "effective": Effective number of samples weighting

    Returns:
        Normalized weight tensor of shape (num_classes,).
    """
    if hasattr(dataset, "get_labels"):
        labels = dataset.get_labels()
    else:
        labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if isinstance(sample, tuple) and len(sample) >= 2:
                labels.append(sample[1])

    label_counts = Counter(labels)

    counts = torch.zeros(num_classes)
    for label, count in label_counts.items():
        if 0 <= label < num_classes:
            counts[label] = count

    counts = counts + 1e-6

    if method == "inverse_freq":
        weights = 1.0 / counts
    elif method == "inverse_sqrt":
        weights = 1.0 / torch.sqrt(counts)
    elif method == "effective":
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    weights = weights / weights.sum() * num_classes

    return weights


def compute_class_weights_from_labels(
    labels: List[int],
    num_classes: int = 10
) -> torch.Tensor:
    """
    Compute class weights from a list of labels.

    Args:
        labels: List of integer labels.
        num_classes: Number of classes.

    Returns:
        Normalized weight tensor of shape (num_classes,).
    """
    label_counts = Counter(labels)

    counts = torch.zeros(num_classes)
    for label, count in label_counts.items():
        if 0 <= label < num_classes:
            counts[label] = count

    counts = counts + 1e-6

    weights = 1.0 / counts

    weights = weights / weights.sum() * num_classes

    return weights


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focal loss down-weights easy examples and focuses training on hard
    negatives. Useful when there is extreme class imbalance.

    Attributes:
        alpha: Weighting factor for each class.
        gamma: Focusing parameter (higher = more focus on hard examples).
        reduction: Reduction method ('mean', 'sum', or 'none').
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        """
        Initialize FocalLoss.

        Args:
            alpha: Class weights tensor. If None, all classes weighted equally.
            gamma: Focusing parameter. gamma=0 is equivalent to CE loss.
            reduction: How to reduce the loss ('mean', 'sum', 'none').
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Model predictions of shape (N, C).
            targets: Ground truth labels of shape (N,).

        Returns:
            Computed focal loss.
        """
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction="none"
        )

        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha[targets]
            focal_loss = at * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
