"""
Custom PyTorch Dataset classes for distracted driver detection.

This module provides dataset classes for:
- State Farm Distracted Driver Detection dataset
- AUC Distracted Driver Dataset (for generalization testing)
"""

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import compute_dataset_stats


class DriverDataset(Dataset):
    """
    PyTorch Dataset for State Farm Distracted Driver Detection.

    Implements subject-aware train/val splitting to prevent data leakage.
    Images are grouped by subject ID and subjects are split between
    train and validation sets.

    Attributes:
        root_dir: Root directory containing the dataset.
        split: Either "train" or "val".
        transform: Albumentations transform pipeline.
        samples: List of (image_path, label) tuples.
        class_names: List of class names (c0-c9).
    """

    CLASS_NAMES = [
        "c0", "c1", "c2", "c3", "c4",
        "c5", "c6", "c7", "c8", "c9"
    ]

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        train_ratio: float = 0.8,
        seed: int = 42
    ):
        """
        Initialize the DriverDataset.

        Args:
            root_dir: Root directory of the State Farm dataset.
                      Should contain imgs/train/ and driver_imgs_list.csv.
            split: Either "train" or "val".
            transform: Albumentations transform pipeline to apply.
            train_ratio: Ratio of subjects to use for training (default 0.8).
            seed: Random seed for reproducible subject splitting.

        Raises:
            ValueError: If split is not "train" or "val".
            FileNotFoundError: If dataset files are not found.
        """
        if split not in ["train", "val"]:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.train_ratio = train_ratio
        self.seed = seed

        self.samples: List[Tuple[Path, int]] = []
        self.class_names = self.CLASS_NAMES

        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load dataset with subject-aware splitting."""
        train_dir = self.root_dir / "imgs" / "train"
        csv_path = self.root_dir / "driver_imgs_list.csv"

        if not train_dir.exists():
            alt_train_dir = self.root_dir / "train"
            if alt_train_dir.exists():
                train_dir = alt_train_dir
            else:
                raise FileNotFoundError(
                    f"Training directory not found at {train_dir} or {alt_train_dir}"
                )

        subject_to_images: Dict[str, List[Tuple[Path, int]]] = {}

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                subject = row["subject"]
                classname = row["classname"]
                img_name = row["img"]

                label = int(classname[1])
                img_path = train_dir / classname / img_name

                if img_path.exists():
                    if subject not in subject_to_images:
                        subject_to_images[subject] = []
                    subject_to_images[subject].append((img_path, label))
        else:
            for class_idx, class_name in enumerate(self.CLASS_NAMES):
                class_dir = train_dir / class_name
                if not class_dir.exists():
                    continue

                for img_path in class_dir.glob("*.jpg"):
                    parts = img_path.stem.split("_")
                    if len(parts) >= 2:
                        subject = parts[0]
                    else:
                        subject = f"unknown_{class_idx}"

                    if subject not in subject_to_images:
                        subject_to_images[subject] = []
                    subject_to_images[subject].append((img_path, class_idx))

        subjects = list(subject_to_images.keys())
        random.seed(self.seed)
        random.shuffle(subjects)

        n_train = int(len(subjects) * self.train_ratio)
        train_subjects = set(subjects[:n_train])
        val_subjects = set(subjects[n_train:])

        target_subjects = train_subjects if self.split == "train" else val_subjects

        for subject in target_subjects:
            self.samples.extend(subject_to_images[subject])

        random.shuffle(self.samples)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (image_tensor, label, image_path_string).
        """
        img_path, label = self.samples[idx]

        image = Image.open(str(img_path)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            from torchvision import transforms
            image = transforms.ToTensor()(image)

        return image, label, str(img_path)

    def get_class_distribution(self) -> Dict[int, int]:
        """
        Get the distribution of classes in the dataset.

        Returns:
            Dictionary mapping class index to count.
        """
        distribution = {i: 0 for i in range(len(self.CLASS_NAMES))}
        for _, label in self.samples:
            distribution[label] += 1
        return distribution

    def get_labels(self) -> List[int]:
        """
        Get all labels in the dataset.

        Returns:
            List of integer labels.
        """
        return [label for _, label in self.samples]


class AUCDataset(Dataset):
    """
    PyTorch Dataset for AUC Distracted Driver Dataset.

    Used for zero-shot domain generalization evaluation.
    Computes dataset-specific normalization statistics.

    Attributes:
        root_dir: Root directory containing the AUC dataset.
        transform: Albumentations transform pipeline.
        samples: List of (image_path, label) tuples.
        mean: Dataset mean for normalization.
        std: Dataset std for normalization.
    """

    AUC_TO_STATEFARM_MAPPING = {
        "c0": 0,
        "c1": 1,
        "c2": 2,
        "c3": 3,
        "c4": 4,
        "c5": 5,
        "c6": 6,
        "c7": 7,
        "c8": 8,
        "c9": 9,
    }

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        compute_stats: bool = True,
        n_stat_samples: int = 200,
        img_size: int = 224
    ):
        """
        Initialize the AUCDataset.

        Args:
            root_dir: Root directory of the AUC dataset.
            transform: Albumentations transform pipeline.
                       If None and compute_stats is True, will create one.
            compute_stats: Whether to compute dataset-specific normalization.
            n_stat_samples: Number of samples for computing statistics.
            img_size: Image size for transforms.
        """
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.samples: List[Tuple[Path, int]] = []
        self.mean: Optional[List[float]] = None
        self.std: Optional[List[float]] = None

        self._load_dataset()

        if compute_stats and len(self.samples) > 0:
            image_paths = [str(p) for p, _ in self.samples]
            self.mean, self.std = compute_dataset_stats(
                image_paths, img_size, n_stat_samples
            )

        if transform is not None:
            self.transform = transform
        elif self.mean is not None and self.std is not None:
            from .transforms import auc_transforms
            self.transform = auc_transforms(img_size, self.mean, self.std)
        else:
            from .transforms import val_transforms
            self.transform = val_transforms(img_size)

    def _load_dataset(self) -> None:
        """Load AUC dataset images and labels."""
        for class_folder in sorted(self.root_dir.iterdir()):
            if not class_folder.is_dir():
                continue

            class_name = class_folder.name.lower()
            if class_name not in self.AUC_TO_STATEFARM_MAPPING:
                if class_name.startswith("c") and len(class_name) == 2:
                    try:
                        label = int(class_name[1])
                    except ValueError:
                        continue
                else:
                    continue
            else:
                label = self.AUC_TO_STATEFARM_MAPPING[class_name]

            for img_path in class_folder.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (image_tensor, label, image_path_string).
        """
        img_path, label = self.samples[idx]

        try:
            image = Image.open(str(img_path)).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)
        else:
            from torchvision import transforms
            image = transforms.ToTensor()(image)

        return image, label, str(img_path)

    def get_normalization_stats(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        Get the computed normalization statistics.

        Returns:
            Tuple of (mean, std) lists or (None, None) if not computed.
        """
        return self.mean, self.std
