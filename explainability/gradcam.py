"""
Grad-CAM and EigenCAM visualization utilities.

This module provides functions for generating class activation maps
to explain model predictions using gradient-based methods.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_cam_method(
    method_name: str,
    model: nn.Module,
    target_layer: nn.Module
):
    """
    Get the appropriate CAM method instance.

    For MobileNetV3-Small, automatically switches to EigenCAM since
    standard GradCAM produces poor heatmaps on depthwise separable
    convolutions.

    Args:
        method_name: CAM method name ('gradcam' or 'eigencam').
        model: PyTorch model.
        target_layer: Target layer for CAM computation.

    Returns:
        CAM method instance (GradCAM or EigenCAM).

    Raises:
        ValueError: If method_name is not supported.
    """
    method_name = method_name.lower()

    if method_name == "gradcam":
        return GradCAM(model=model, target_layers=[target_layer])
    elif method_name == "eigencam":
        return EigenCAM(model=model, target_layers=[target_layer])
    else:
        raise ValueError(
            f"Unsupported CAM method: {method_name}. "
            f"Supported: gradcam, eigencam"
        )


def generate_cam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    method: str = "gradcam",
    target_layer: Optional[nn.Module] = None,
    arch_name: Optional[str] = None
) -> np.ndarray:
    """
    Generate a class activation map for an input image.

    Args:
        model: PyTorch model.
        image_tensor: Input image tensor of shape (1, C, H, W) or (C, H, W).
        target_class: Target class index for CAM. If None, uses predicted class.
        method: CAM method ('gradcam' or 'eigencam').
        target_layer: Target layer for CAM. If None, must provide arch_name.
        arch_name: Architecture name to auto-detect target layer.

    Returns:
        Heatmap as numpy array of shape (H, W) with values in [0, 1].
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    if target_layer is None:
        if arch_name is None:
            raise ValueError("Either target_layer or arch_name must be provided")
        from models.model_factory import get_target_layer
        target_layer = get_target_layer(model, arch_name)

    if arch_name is not None and "mobilenet" in arch_name.lower():
        method = "eigencam"

    cam = get_cam_method(method, model, target_layer)

    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]
    else:
        targets = None

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)

    heatmap = grayscale_cam[0, :]

    return heatmap


def overlay_cam_on_image(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay a CAM heatmap on the original image.

    Args:
        image_np: Original image as numpy array (H, W, 3) in RGB format,
                  values in [0, 1] or [0, 255].
        heatmap: CAM heatmap of shape (H, W) with values in [0, 1].
        alpha: Blending factor for the heatmap (0 = image only, 1 = heatmap only).
        colormap: OpenCV colormap to use for the heatmap.

    Returns:
        Overlaid image as numpy array (H, W, 3) in RGB format, values in [0, 1].
    """
    if image_np.max() > 1.0:
        image_np = image_np.astype(np.float32) / 255.0

    if heatmap.shape[:2] != image_np.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))

    overlaid = show_cam_on_image(image_np, heatmap, use_rgb=True, image_weight=1-alpha)

    return overlaid


def denormalize_image(
    image_tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """
    Denormalize an image tensor back to [0, 1] range.

    Args:
        image_tensor: Normalized image tensor of shape (C, H, W) or (1, C, H, W).
        mean: Normalization mean values.
        std: Normalization std values.

    Returns:
        Denormalized image as numpy array (H, W, C) in [0, 1] range.
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)

    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)

    mean = np.array(mean)
    std = np.array(std)
    image_np = image_np * std + mean

    image_np = np.clip(image_np, 0, 1)

    return image_np


def visualize_batch(
    model: nn.Module,
    dataloader: DataLoader,
    arch_name: str,
    method: str,
    target_layer: nn.Module,
    class_names: List[str],
    save_dir: Union[str, Path],
    n_samples: int = 5,
    show: bool = False
) -> None:
    """
    Generate and save Grad-CAM visualizations for each class.

    For each class, picks n_samples correctly classified images,
    generates CAM, and saves an overlaid image grid.

    Args:
        model: PyTorch model.
        dataloader: DataLoader with images and labels.
        arch_name: Architecture name.
        method: CAM method ('gradcam' or 'eigencam').
        target_layer: Target layer for CAM.
        class_names: List of class names.
        save_dir: Directory to save visualizations.
        n_samples: Number of samples per class.
        show: Whether to display plots inline.
    """
    save_dir = Path(save_dir) / arch_name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    if "mobilenet" in arch_name.lower():
        method = "eigencam"

    class_samples = {i: [] for i in range(len(class_names))}

    print(f"Collecting samples for {arch_name}...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Scanning batches"):
            images, labels = batch[0], batch[1]
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = outputs.max(1)

            for i in range(images.size(0)):
                label = labels[i].item()
                pred = preds[i].item()

                if pred == label and len(class_samples[label]) < n_samples:
                    class_samples[label].append(images[i].cpu())

            all_collected = all(
                len(samples) >= n_samples
                for samples in class_samples.values()
            )
            if all_collected:
                break

    print(f"Generating CAM visualizations for {arch_name}...")
    for class_idx, samples in tqdm(class_samples.items(), desc="Processing classes"):
        if len(samples) == 0:
            continue

        n_cols = min(len(samples), n_samples)
        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

        if n_cols == 1:
            axes = axes.reshape(2, 1)

        for i, img_tensor in enumerate(samples[:n_samples]):
            img_np = denormalize_image(img_tensor)

            heatmap = generate_cam(
                model=model,
                image_tensor=img_tensor,
                target_class=class_idx,
                method=method,
                target_layer=target_layer
            )

            overlaid = overlay_cam_on_image(img_np, heatmap, alpha=0.4)

            axes[0, i].imshow(img_np)
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")

            axes[1, i].imshow(overlaid)
            axes[1, i].set_title("Grad-CAM")
            axes[1, i].axis("off")

        class_name = class_names[class_idx] if class_idx < len(class_names) else f"c{class_idx}"
        safe_name = class_name.replace(" ", "_").replace("/", "-")
        fig.suptitle(f"{arch_name} - {class_name}", fontsize=14)
        plt.tight_layout()

        save_path = save_dir / f"class_{class_idx}_{safe_name}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

    print(f"CAM visualizations saved to: {save_dir}")


def compare_models_cam(
    models_dict: dict,
    image_tensor: torch.Tensor,
    target_class: int,
    class_name: str,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Compare CAM visualizations across multiple models for the same image.

    Args:
        models_dict: Dictionary mapping arch_name to (model, target_layer).
        image_tensor: Input image tensor.
        target_class: Target class for CAM.
        class_name: Class name for title.
        save_path: Path to save the figure.
        show: Whether to display the plot.

    Returns:
        Matplotlib Figure object.
    """
    from models.model_factory import get_target_layer

    n_models = len(models_dict)
    fig, axes = plt.subplots(2, n_models + 1, figsize=(4 * (n_models + 1), 8))

    img_np = denormalize_image(image_tensor)

    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    for i, (arch_name, model) in enumerate(models_dict.items(), 1):
        target_layer = get_target_layer(model, arch_name)
        method = "eigencam" if "mobilenet" in arch_name.lower() else "gradcam"

        heatmap = generate_cam(
            model=model,
            image_tensor=image_tensor,
            target_class=target_class,
            method=method,
            target_layer=target_layer
        )

        overlaid = overlay_cam_on_image(img_np, heatmap, alpha=0.4)

        axes[0, i].imshow(heatmap, cmap="jet")
        axes[0, i].set_title(f"{arch_name}\nHeatmap")
        axes[0, i].axis("off")

        axes[1, i].imshow(overlaid)
        axes[1, i].set_title("Overlay")
        axes[1, i].axis("off")

    fig.suptitle(f"CAM Comparison - {class_name}", fontsize=14)
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


def generate_single_visualization(
    model: nn.Module,
    image_path: Union[str, Path],
    arch_name: str,
    class_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    img_size: int = 224,
    show: bool = True
) -> Tuple[np.ndarray, int, float]:
    """
    Generate CAM visualization for a single image file.

    Args:
        model: PyTorch model.
        image_path: Path to the image file.
        arch_name: Architecture name.
        class_names: List of class names.
        save_path: Path to save the visualization.
        img_size: Image size for preprocessing.
        show: Whether to display the plot.

    Returns:
        Tuple of (overlaid_image, predicted_class, confidence).
    """
    from data.transforms import val_transforms
    from models.model_factory import get_target_layer

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = val_transforms(img_size)
    transformed = transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0)

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = probs.max(1)
        pred_class = pred_class.item()
        confidence = confidence.item()

    target_layer = get_target_layer(model, arch_name)
    method = "eigencam" if "mobilenet" in arch_name.lower() else "gradcam"

    heatmap = generate_cam(
        model=model,
        image_tensor=image_tensor,
        target_class=pred_class,
        method=method,
        target_layer=target_layer
    )

    img_np = denormalize_image(image_tensor.squeeze(0))
    overlaid = overlay_cam_on_image(img_np, heatmap, alpha=0.4)

    if save_path is not None or show:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(img_np)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlaid)
        axes[2].set_title(f"Pred: {class_names[pred_class]}\nConf: {confidence:.2%}")
        axes[2].axis("off")

        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return overlaid, pred_class, confidence
