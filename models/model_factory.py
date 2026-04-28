"""
Model factory for loading and configuring CNN architectures.

This module provides functions to load pretrained models and configure
them for the distracted driver detection task.

Supported architectures:
- EfficientNet-B0
- MobileNetV3-Small
- ResNet-50
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet50_Weights,
)


def get_model(
    arch_name: str,
    num_classes: int = 10,
    pretrained: bool = True
) -> nn.Module:
    """
    Load and configure a CNN model for classification.

    Loads a pretrained model from torchvision and replaces the final
    classifier head with a new linear layer for the target number of classes.

    Args:
        arch_name: Architecture name. One of:
                   "efficientnet_b0", "mobilenet_v3_small", "resnet50"
        num_classes: Number of output classes (default 10).
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        Configured PyTorch model.

    Raises:
        ValueError: If arch_name is not supported.
    """
    arch_name = arch_name.lower()

    if arch_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif arch_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif arch_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported architecture: {arch_name}. "
            f"Supported: efficientnet_b0, mobilenet_v3_small, resnet50"
        )

    return model


def get_target_layer(model: nn.Module, arch_name: str) -> nn.Module:
    """
    Get the target layer for Grad-CAM visualization.

    Returns the last convolutional layer of the model, which is typically
    the best layer for generating class activation maps.

    Args:
        model: The PyTorch model.
        arch_name: Architecture name to determine the correct layer.

    Returns:
        The target layer module for Grad-CAM.

    Raises:
        ValueError: If arch_name is not supported.
    """
    arch_name = arch_name.lower()

    if arch_name == "efficientnet_b0":
        return model.features[-1]

    elif arch_name == "mobilenet_v3_small":
        return model.features[-1]

    elif arch_name == "resnet50":
        return model.layer4[-1]

    else:
        raise ValueError(
            f"Unsupported architecture: {arch_name}. "
            f"Supported: efficientnet_b0, mobilenet_v3_small, resnet50"
        )


def get_model_info(arch_name: str) -> dict:
    """
    Get information about a model architecture.

    Args:
        arch_name: Architecture name.

    Returns:
        Dictionary with model information including:
        - input_size: Expected input size
        - gradcam_method: Recommended Grad-CAM method
        - description: Brief description
    """
    arch_name = arch_name.lower()

    info = {
        "efficientnet_b0": {
            "input_size": (224, 224),
            "gradcam_method": "gradcam",
            "description": "EfficientNet-B0: Balanced accuracy and efficiency",
        },
        "mobilenet_v3_small": {
            "input_size": (224, 224),
            "gradcam_method": "eigencam",
            "description": "MobileNetV3-Small: Optimized for mobile/edge deployment",
        },
        "resnet50": {
            "input_size": (224, 224),
            "gradcam_method": "gradcam",
            "description": "ResNet-50: Deep residual network with strong performance",
        },
    }

    if arch_name not in info:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    return info[arch_name]


def freeze_backbone(model: nn.Module, arch_name: str) -> nn.Module:
    """
    Freeze the backbone layers of a model for transfer learning.

    Only the classifier head will be trainable after this operation.

    Args:
        model: The PyTorch model.
        arch_name: Architecture name.

    Returns:
        Model with frozen backbone.
    """
    arch_name = arch_name.lower()

    if arch_name == "efficientnet_b0":
        for param in model.features.parameters():
            param.requires_grad = False

    elif arch_name == "mobilenet_v3_small":
        for param in model.features.parameters():
            param.requires_grad = False

    elif arch_name == "resnet50":
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    return model


def unfreeze_model(model: nn.Module) -> nn.Module:
    """
    Unfreeze all layers of a model.

    Args:
        model: The PyTorch model.

    Returns:
        Model with all layers trainable.
    """
    for param in model.parameters():
        param.requires_grad = True
    return model


def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters in a model.

    Args:
        model: The PyTorch model.

    Returns:
        Tuple of (trainable_params, total_params).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
