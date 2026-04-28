"""
Domain Generalization Analysis for Distracted Driver Detection.

This module provides tools for evaluating zero-shot domain generalization
performance when transferring models from the State Farm dataset to the
AUC Distracted Driver Dataset.

Key Finding: Identifies which specific distraction classes collapse most
when the model encounters a completely new dataset it was never trained on.
This analysis is underreported in the literature.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score


class DomainGeneralizationEvaluator:
    """
    Evaluator for zero-shot domain generalization analysis.

    Compares model performance on source domain (State Farm) vs target domain
    (AUC dataset) without any retraining, identifying which classes suffer
    the most from domain shift.

    Attributes:
        model: Trained PyTorch model.
        device: Device for inference.
        source_class_names: List of class names from source domain.
        auc_label_map: Mapping from AUC folder names to State Farm class indices.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        source_class_names: List[str],
        auc_label_map: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the DomainGeneralizationEvaluator.

        Args:
            model: Trained PyTorch model.
            device: Device for inference (cuda/mps/cpu).
            source_class_names: List of 10 class names from State Farm.
            auc_label_map: Dict mapping AUC dataset folder names to State Farm
                           class indices. If None, uses default mapping.
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.source_class_names = source_class_names

        if auc_label_map is None:
            self.auc_label_map = {
                "c0": 0, "c1": 1, "c2": 2, "c3": 3, "c4": 4,
                "c5": 5, "c6": 6, "c7": 7, "c8": 8, "c9": 9,
            }
        else:
            self.auc_label_map = auc_label_map

    def compute_normalization_stats(
        self,
        auc_dir: Union[str, Path],
        n_samples: int = 200,
        img_size: int = 224,
        save_path: Optional[Union[str, Path]] = None
    ) -> Tuple[List[float], List[float]]:
        """
        Compute per-channel mean and std from AUC dataset samples.

        This corrects for camera/lighting distribution shift between datasets.

        Args:
            auc_dir: Path to AUC dataset directory.
            n_samples: Number of random samples to use for computation.
            img_size: Size to resize images to before computing stats.
            save_path: Optional path to save stats JSON. If None, saves to
                       outputs/auc_norm_stats.json.

        Returns:
            Tuple of (mean, std) as lists of 3 floats (RGB channels).
        """
        auc_dir = Path(auc_dir)

        image_paths = []
        for class_folder in auc_dir.iterdir():
            if not class_folder.is_dir():
                continue
            for img_path in class_folder.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    image_paths.append(img_path)

        if len(image_paths) == 0:
            raise ValueError(f"No images found in {auc_dir}")

        if len(image_paths) > n_samples:
            random.seed(42)
            image_paths = random.sample(image_paths, n_samples)

        pixel_sum = np.zeros(3)
        pixel_sq_sum = np.zeros(3)
        n_pixels = 0

        for path in tqdm(image_paths, desc="Computing AUC normalization stats"):
            img = cv2.imread(str(path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32) / 255.0

            pixel_sum += img.sum(axis=(0, 1))
            pixel_sq_sum += (img ** 2).sum(axis=(0, 1))
            n_pixels += img.shape[0] * img.shape[1]

        mean = pixel_sum / n_pixels
        std = np.sqrt(pixel_sq_sum / n_pixels - mean ** 2)

        mean_list = mean.tolist()
        std_list = std.tolist()

        if save_path is None:
            save_path = Path("outputs") / "auc_norm_stats.json"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        stats = {
            "mean": mean_list,
            "std": std_list,
            "n_samples": len(image_paths),
            "source": str(auc_dir)
        }
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"AUC normalization stats saved to: {save_path}")
        print(f"  Mean: {mean_list}")
        print(f"  Std: {std_list}")

        return mean_list, std_list

    @torch.no_grad()
    def evaluate_source(self, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model on State Farm validation set.

        Args:
            val_loader: DataLoader for State Farm validation set.

        Returns:
            Dictionary with per_class_f1, accuracy, macro_f1, and dataset name.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        for batch in tqdm(val_loader, desc="Evaluating on State Farm"):
            images, labels = batch[0], batch[1]
            images = images.to(self.device)

            outputs = self.model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        per_class_f1 = f1_score(
            all_labels, all_preds, average=None,
            labels=range(len(self.source_class_names)), zero_division=0
        ).tolist()

        return {
            "dataset": "state_farm",
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1,
            "predictions": all_preds,
            "labels": all_labels
        }

    @torch.no_grad()
    def evaluate_target_zeroshot(
        self,
        auc_dir: Union[str, Path],
        norm_mean: List[float],
        norm_std: List[float],
        img_size: int = 224,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Evaluate model on AUC dataset with zero-shot transfer (no retraining).

        Args:
            auc_dir: Path to AUC dataset directory.
            norm_mean: Normalization mean computed from AUC dataset.
            norm_std: Normalization std computed from AUC dataset.
            img_size: Image size for preprocessing.
            batch_size: Batch size for inference.

        Returns:
            Dictionary with per_class_f1, accuracy, macro_f1, and dataset name.
        """
        from data.dataset import AUCDataset
        from data.transforms import auc_transforms

        auc_dir = Path(auc_dir)

        transform = auc_transforms(img_size, norm_mean, norm_std)

        auc_dataset = AUCDataset(
            root_dir=auc_dir,
            transform=transform,
            compute_stats=False
        )

        if len(auc_dataset) == 0:
            raise ValueError(f"AUC dataset is empty at {auc_dir}")

        auc_loader = DataLoader(
            auc_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        self.model.eval()
        all_preds = []
        all_labels = []

        for batch in tqdm(auc_loader, desc="Evaluating on AUC (zero-shot)"):
            images, labels = batch[0], batch[1]
            images = images.to(self.device)

            outputs = self.model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        per_class_f1 = f1_score(
            all_labels, all_preds, average=None,
            labels=range(len(self.source_class_names)), zero_division=0
        ).tolist()

        return {
            "dataset": "auc_zeroshot",
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1,
            "predictions": all_preds,
            "labels": all_labels
        }

    def compute_domain_gap(
        self,
        source_results: Dict[str, Any],
        target_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute per-class domain gap between source and target results.

        Args:
            source_results: Results from evaluate_source().
            target_results: Results from evaluate_target_zeroshot().

        Returns:
            Dictionary containing:
            - per_class_gap: Dict mapping class_name to gap value
            - overall_accuracy_drop: Float
            - worst_3_classes: List of (class_name, gap) sorted descending
            - best_3_classes: List of (class_name, gap) sorted ascending
            - gap_hypothesis: Auto-generated explanation string
        """
        source_f1 = source_results["per_class_f1"]
        target_f1 = target_results["per_class_f1"]

        per_class_gap = {}
        gaps_with_names = []

        for i, class_name in enumerate(self.source_class_names):
            gap = source_f1[i] - target_f1[i]
            per_class_gap[class_name] = gap
            gaps_with_names.append((class_name, gap))

        gaps_with_names.sort(key=lambda x: x[1], reverse=True)
        worst_3_classes = gaps_with_names[:3]
        best_3_classes = gaps_with_names[-3:][::-1]

        overall_accuracy_drop = source_results["accuracy"] - target_results["accuracy"]
        macro_f1_drop = source_results["macro_f1"] - target_results["macro_f1"]

        gap_hypothesis = self._generate_gap_hypothesis(
            worst_3_classes, best_3_classes, overall_accuracy_drop
        )

        return {
            "per_class_gap": per_class_gap,
            "overall_accuracy_drop": overall_accuracy_drop,
            "macro_f1_drop": macro_f1_drop,
            "worst_3_classes": worst_3_classes,
            "best_3_classes": best_3_classes,
            "gap_hypothesis": gap_hypothesis
        }

    def _generate_gap_hypothesis(
        self,
        worst_3: List[Tuple[str, float]],
        best_3: List[Tuple[str, float]],
        accuracy_drop: float
    ) -> str:
        """Generate an auto-explanation for the domain gap pattern."""
        worst_names = [w[0] for w in worst_3]
        worst_gaps = [w[1] for w in worst_3]

        hand_classes = ["c1", "c3"]
        phone_classes = ["c2", "c4"]
        face_classes = ["c8", "c9"]

        hand_related = sum(1 for w in worst_names if any(h in w for h in hand_classes))
        phone_related = sum(1 for w in worst_names if any(p in w for p in phone_classes))
        face_related = sum(1 for w in worst_names if any(f in w for f in face_classes))

        if hand_related >= 2:
            hypothesis = (
                f"Classes requiring fine-grained hand position recognition "
                f"({', '.join(worst_names[:2])}) show the largest domain gap "
                f"(>{worst_gaps[0]:.2f}), suggesting the model learned "
                f"subject-specific hand appearance rather than generalizable "
                f"gesture features."
            )
        elif phone_related >= 2:
            hypothesis = (
                f"Phone-related classes ({', '.join(worst_names[:2])}) show "
                f"significant domain gap (>{worst_gaps[0]:.2f}), indicating "
                f"sensitivity to phone appearance and positioning variations "
                f"across datasets."
            )
        elif face_related >= 2:
            hypothesis = (
                f"Face-oriented classes ({', '.join(worst_names[:2])}) suffer "
                f"the largest drops (>{worst_gaps[0]:.2f}), suggesting the model "
                f"relies on subject-specific facial features rather than "
                f"generalizable head pose patterns."
            )
        else:
            hypothesis = (
                f"Classes {', '.join(worst_names)} show the largest domain gaps "
                f"(up to {worst_gaps[0]:.2f}), while classes "
                f"{', '.join([b[0] for b in best_3])} transfer more robustly. "
                f"The overall accuracy drop of {accuracy_drop*100:.1f}pp suggests "
                f"significant distribution shift between camera setups and "
                f"subject populations."
            )

        return hypothesis

    def plot_domain_gap(
        self,
        source_results: Dict[str, Any],
        target_results: Dict[str, Any],
        save_path: Union[str, Path],
        show: bool = True
    ) -> plt.Figure:
        """
        Plot grouped bar chart comparing source vs target F1 scores.

        Args:
            source_results: Results from evaluate_source().
            target_results: Results from evaluate_target_zeroshot().
            save_path: Path to save the figure.
            show: Whether to display the plot.

        Returns:
            Matplotlib Figure object.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        source_f1 = [f * 100 for f in source_results["per_class_f1"]]
        target_f1 = [f * 100 for f in target_results["per_class_f1"]]
        gaps = [s - t for s, t in zip(source_f1, target_f1)]

        fig, ax1 = plt.subplots(figsize=(14, 7))

        x = np.arange(len(self.source_class_names))
        width = 0.35

        bars1 = ax1.bar(
            x - width/2, source_f1, width,
            label="State Farm F1", color="#3498db", alpha=0.8
        )
        bars2 = ax1.bar(
            x + width/2, target_f1, width,
            label="AUC Zero-Shot F1", color="#e67e22", alpha=0.8
        )

        ax2 = ax1.twinx()
        line = ax2.plot(
            x, gaps, "r--", linewidth=2, marker="o",
            markersize=6, label="Gap (pp)"
        )
        ax2.set_ylabel("Gap (percentage points)", color="red", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.set_ylim([0, max(gaps) * 1.3 if max(gaps) > 0 else 10])

        gap_with_idx = [(i, g) for i, g in enumerate(gaps)]
        gap_with_idx.sort(key=lambda x: x[1], reverse=True)
        worst_3_idx = [g[0] for g in gap_with_idx[:3]]

        for idx in worst_3_idx:
            ax1.annotate(
                "",
                xy=(idx, target_f1[idx] + 2),
                xytext=(idx, target_f1[idx] + 15),
                arrowprops=dict(arrowstyle="->", color="red", lw=2),
            )
            ax1.text(
                idx, target_f1[idx] + 17,
                f"↓{gaps[idx]:.1f}pp",
                ha="center", fontsize=9, color="red", fontweight="bold"
            )

        ax1.set_xlabel("Class", fontsize=12)
        ax1.set_ylabel("F1 Score (%)", fontsize=12)
        ax1.set_title(
            "Per-Class F1: Source vs Zero-Shot Target Domain",
            fontsize=14, fontweight="bold"
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.source_class_names, rotation=45, ha="right")
        ax1.set_ylim([0, 105])
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax1.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Domain gap plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def generate_domain_report(
        self,
        source_results: Dict[str, Any],
        target_results: Dict[str, Any],
        save_path: Union[str, Path],
        before_norm_acc: Optional[float] = None,
        after_norm_acc: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive domain generalization report.

        Args:
            source_results: Results from evaluate_source().
            target_results: Results from evaluate_target_zeroshot().
            save_path: Path to save the JSON report.
            before_norm_acc: Optional accuracy before normalization correction.
            after_norm_acc: Optional accuracy after normalization correction.

        Returns:
            Report dictionary.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        gap_analysis = self.compute_domain_gap(source_results, target_results)

        per_class_analysis = []
        for i, class_name in enumerate(self.source_class_names):
            source_f1 = source_results["per_class_f1"][i]
            target_f1 = target_results["per_class_f1"][i]
            gap = source_f1 - target_f1

            if gap < 0.05:
                interpretation = "robust"
            elif gap < 0.15:
                interpretation = "moderate_drop"
            elif gap < 0.25:
                interpretation = "significant_drop"
            else:
                interpretation = "collapsed"

            per_class_analysis.append({
                "class_id": i,
                "class_name": class_name,
                "source_f1": round(source_f1, 4),
                "target_f1": round(target_f1, 4),
                "gap": round(gap, 4),
                "interpretation": interpretation
            })

        worst_classes = [w[0] for w in gap_analysis["worst_3_classes"]]
        key_finding = (
            f"Zero-shot transfer from State Farm to AUC dataset results in "
            f"{gap_analysis['overall_accuracy_drop']*100:.1f}pp accuracy drop. "
            f"Classes {', '.join(worst_classes)} show the largest degradation, "
            f"indicating these behaviors are most sensitive to domain shift."
        )

        preprocessing_impact = None
        if before_norm_acc is not None and after_norm_acc is not None:
            preprocessing_impact = {
                "before_norm_correction_acc": round(before_norm_acc, 4),
                "after_norm_correction_acc": round(after_norm_acc, 4),
                "improvement_pp": round((after_norm_acc - before_norm_acc) * 100, 2)
            }

        report = {
            "source_dataset": "State Farm",
            "target_dataset": "AUC (zero-shot)",
            "source_accuracy": round(source_results["accuracy"], 4),
            "target_accuracy": round(target_results["accuracy"], 4),
            "accuracy_drop_pp": round(gap_analysis["overall_accuracy_drop"] * 100, 2),
            "macro_f1_drop": round(gap_analysis["macro_f1_drop"], 4),
            "per_class_analysis": per_class_analysis,
            "key_finding": key_finding,
            "gap_hypothesis": gap_analysis["gap_hypothesis"],
            "preprocessing_impact": preprocessing_impact
        }

        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Domain report saved to: {save_path}")

        return report


def compare_models_domain_gap(
    models_dict: Dict[str, nn.Module],
    val_loader: DataLoader,
    auc_dir: Union[str, Path],
    class_names: List[str],
    device: torch.device,
    save_dir: Union[str, Path]
) -> Dict[str, Dict[str, Any]]:
    """
    Compare domain generalization across multiple model architectures.

    Args:
        models_dict: Dict mapping arch_name to model.
        val_loader: State Farm validation DataLoader.
        auc_dir: Path to AUC dataset.
        class_names: List of class names.
        device: Device for inference.
        save_dir: Directory to save results.

    Returns:
        Dict mapping arch_name to gap analysis results.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for arch_name, model in models_dict.items():
        print(f"\n{'='*50}")
        print(f"Evaluating domain gap for {arch_name}")
        print(f"{'='*50}")

        evaluator = DomainGeneralizationEvaluator(
            model=model,
            device=device,
            source_class_names=class_names
        )

        norm_mean, norm_std = evaluator.compute_normalization_stats(
            auc_dir=auc_dir,
            save_path=save_dir / f"{arch_name}_auc_norm_stats.json"
        )

        source_results = evaluator.evaluate_source(val_loader)
        target_results = evaluator.evaluate_target_zeroshot(
            auc_dir=auc_dir,
            norm_mean=norm_mean,
            norm_std=norm_std
        )

        gap_analysis = evaluator.compute_domain_gap(source_results, target_results)

        evaluator.plot_domain_gap(
            source_results=source_results,
            target_results=target_results,
            save_path=save_dir / f"{arch_name}_domain_gap.png",
            show=False
        )

        evaluator.generate_domain_report(
            source_results=source_results,
            target_results=target_results,
            save_path=save_dir / f"{arch_name}_domain_report.json"
        )

        all_results[arch_name] = {
            "source_results": source_results,
            "target_results": target_results,
            "gap_analysis": gap_analysis
        }

    return all_results
