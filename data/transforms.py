"""
Data augmentation and transformation pipelines using torchvision.

This module provides transformation pipelines for training, validation,
and AUC dataset evaluation.
"""

from typing import List, Optional, Tuple

from torchvision import transforms


def train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Create training augmentation pipeline.

    Applies various augmentations to improve model generalization:
    - RandomResizedCrop for scale invariance
    - HorizontalFlip for left-right invariance
    - Color augmentations for lighting robustness
    - RandomErasing for occlusion robustness

    Args:
        img_size: Target image size (height and width).

    Returns:
        torchvision Compose object with training transforms.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=img_size,
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
        ),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))
    ])


def val_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Create validation/test transformation pipeline.

    Applies minimal transforms for consistent evaluation:
    - Resize to slightly larger than target
    - CenterCrop to target size
    - ImageNet normalization

    Args:
        img_size: Target image size (height and width).

    Returns:
        torchvision Compose object with validation transforms.
    """
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def auc_transforms(
    img_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> transforms.Compose:
    """
    Create transformation pipeline for AUC dataset with custom normalization.

    Similar to validation transforms but uses dataset-specific normalization
    statistics for better domain adaptation.

    Args:
        img_size: Target image size (height and width).
        mean: Dataset-specific mean values for normalization.
              Defaults to ImageNet mean if None.
        std: Dataset-specific std values for normalization.
             Defaults to ImageNet std if None.

    Returns:
        torchvision Compose object with AUC dataset transforms.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def compute_dataset_stats(
    image_paths: List[str],
    img_size: int = 224,
    n_samples: int = 200
) -> Tuple[List[float], List[float]]:
    """
    Compute mean and std statistics from a set of images.

    Useful for computing normalization statistics for new datasets
    like the AUC dataset.

    Args:
        image_paths: List of paths to images.
        img_size: Size to resize images to before computing stats.
        n_samples: Number of samples to use for computation.

    Returns:
        Tuple of (mean, std) as lists of 3 floats (RGB channels).
    """
    import random
    import numpy as np
    from PIL import Image

    if len(image_paths) > n_samples:
        image_paths = random.sample(image_paths, n_samples)

    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    n_pixels = 0

    for path in image_paths:
        try:
            img = Image.open(str(path)).convert("RGB")
            img = img.resize((img_size, img_size))
            img = np.array(img).astype(np.float32) / 255.0

            pixel_sum += img.sum(axis=(0, 1))
            pixel_sq_sum += (img ** 2).sum(axis=(0, 1))
            n_pixels += img.shape[0] * img.shape[1]
        except Exception:
            continue

    mean = pixel_sum / n_pixels
    std = np.sqrt(pixel_sq_sum / n_pixels - mean ** 2)

    return mean.tolist(), std.tolist()
