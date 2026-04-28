"""
Utility functions for the Distracted Driver Detection project.

This module provides helper functions for reproducibility, device management,
configuration loading, and JSON file operations.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the best available device for computation.

    Returns:
        torch.device: CUDA if available, MPS for Apple Silicon, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def save_json(data: Any, filepath: Union[str, Path]) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save (must be JSON serializable).
        filepath: Path where to save the JSON file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load data from a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Loaded data from the JSON file.

    Raises:
        FileNotFoundError: If the JSON file doesn't exist.
        json.JSONDecodeError: If the JSON file is malformed.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_class_names() -> list:
    """
    Get the list of class names for the distracted driver detection task.

    Returns:
        List of 10 class names (c0-c9).
    """
    return [
        "c0 - safe driving",
        "c1 - texting (right hand)",
        "c2 - talking on phone (right hand)",
        "c3 - texting (left hand)",
        "c4 - talking on phone (left hand)",
        "c5 - operating the radio",
        "c6 - drinking",
        "c7 - reaching behind",
        "c8 - hair and makeup",
        "c9 - talking to passenger",
    ]


def get_short_class_names() -> list:
    """
    Get short class names for visualization.

    Returns:
        List of 10 short class names.
    """
    return [
        "Safe",
        "Text R",
        "Phone R",
        "Text L",
        "Phone L",
        "Radio",
        "Drink",
        "Reach",
        "Makeup",
        "Passenger",
    ]


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory.

    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
