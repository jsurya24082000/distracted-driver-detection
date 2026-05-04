"""
Run Domain Generalization and CAM Quality Analysis
This script evaluates all trained models on novel contributions.
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.transforms import val_transforms
from models.model_factory import get_model

# Class names
CLASS_NAMES = [
    "Safe Driving", "Texting (R)", "Phone (R)", "Texting (L)", "Phone (L)",
    "Radio", "Drinking", "Reaching", "Makeup", "Passenger"
]

def load_model(arch_name, device):
    """Load a trained model."""
    model = get_model(arch_name, num_classes=10, pretrained=False)
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / f"best_{arch_name}.pth"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded {arch_name} checkpoint")
    else:
        print(f"  WARNING: No checkpoint for {arch_name}")
        return None
    
    model = model.to(device)
    model.eval()
    return model


def compute_saliency_map(model, input_tensor, target_class, device):
    """Compute gradient-based saliency map."""
    input_tensor = input_tensor.clone().requires_grad_(True)
    
    output = model(input_tensor)
    model.zero_grad()
    output[0, target_class].backward()
    
    saliency = input_tensor.grad.data.abs()
    saliency = saliency[0].mean(dim=0).cpu().numpy()
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency


def pointing_game_score(saliency_map, bbox):
    """
    Compute Pointing Game score.
    Returns 1 if max saliency point is inside bbox, 0 otherwise.
    
    bbox: (x_min, y_min, x_max, y_max) normalized [0, 1]
    """
    h, w = saliency_map.shape
    max_idx = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    max_y, max_x = max_idx[0] / h, max_idx[1] / w
    
    x_min, y_min, x_max, y_max = bbox
    
    if x_min <= max_x <= x_max and y_min <= max_y <= y_max:
        return 1.0
    return 0.0


# Approximate bounding boxes for each class (normalized coordinates)
# These represent typical regions of interest for each distraction type
CLASS_BBOXES = {
    0: (0.3, 0.2, 0.8, 0.9),   # Safe: steering wheel area
    1: (0.5, 0.4, 0.9, 0.9),   # Texting R: right hand/lap area
    2: (0.5, 0.1, 0.9, 0.6),   # Phone R: right side face/ear
    3: (0.1, 0.4, 0.5, 0.9),   # Texting L: left hand area
    4: (0.1, 0.1, 0.5, 0.6),   # Phone L: left side face/ear
    5: (0.4, 0.3, 0.8, 0.7),   # Radio: center console
    6: (0.3, 0.2, 0.7, 0.7),   # Drinking: center/face area
    7: (0.2, 0.3, 0.8, 0.9),   # Reaching: back area
    8: (0.3, 0.1, 0.7, 0.5),   # Makeup: face/mirror area
    9: (0.1, 0.2, 0.5, 0.8),   # Passenger: right side looking left
}


def evaluate_cam_quality(model, arch_name, data_dir, device, n_samples=50):
    """Evaluate CAM quality using Pointing Game metric."""
    transform = val_transforms(224)
    
    results = {
        "arch": arch_name,
        "per_class_scores": {},
        "per_class_counts": {},
        "overall_score": 0.0,
        "total_samples": 0
    }
    
    total_score = 0
    total_count = 0
    
    for class_idx in range(10):
        class_dir = data_dir / f"c{class_idx}"
        if not class_dir.exists():
            continue
        
        images = list(class_dir.glob("*.jpg"))[:n_samples]
        class_score = 0
        
        for img_path in images:
            # Load and transform image
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(input_tensor)
                pred_class = output.argmax(dim=1).item()
            
            # Only evaluate if prediction is correct
            if pred_class == class_idx:
                # Compute saliency
                saliency = compute_saliency_map(model, input_tensor, class_idx, device)
                
                # Compute pointing game score
                bbox = CLASS_BBOXES[class_idx]
                score = pointing_game_score(saliency, bbox)
                class_score += score
                total_score += score
                total_count += 1
        
        if len(images) > 0:
            results["per_class_scores"][CLASS_NAMES[class_idx]] = class_score / len(images)
            results["per_class_counts"][CLASS_NAMES[class_idx]] = len(images)
    
    results["overall_score"] = total_score / total_count if total_count > 0 else 0
    results["total_samples"] = total_count
    
    return results


def run_cam_quality_analysis():
    """Run CAM quality analysis on all models."""
    print("\n" + "="*60)
    print("  CAM QUALITY EVALUATION (Pointing Game Metric)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    data_dir = PROJECT_ROOT / "data" / "raw" / "state_farm" / "imgs" / "train"
    
    if not data_dir.exists():
        print("ERROR: Training data not found!")
        return None
    
    architectures = ["efficientnet_b0", "mobilenet_v3_small", "resnet50"]
    all_results = {}
    
    for arch in architectures:
        print(f"\nEvaluating {arch}...")
        model = load_model(arch, device)
        if model is None:
            continue
        
        results = evaluate_cam_quality(model, arch, data_dir, device, n_samples=30)
        all_results[arch] = results
        
        print(f"  Overall Pointing Game Score: {results['overall_score']*100:.1f}%")
        print(f"  Samples evaluated: {results['total_samples']}")
    
    # Save results
    output_path = PROJECT_ROOT / "outputs" / "results" / "cam_quality_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Plot comparison
    plot_cam_comparison(all_results)
    
    return all_results


def plot_cam_comparison(results):
    """Plot CAM quality comparison across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall scores
    archs = list(results.keys())
    scores = [results[a]["overall_score"] * 100 for a in archs]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    axes[0].bar(archs, scores, color=colors)
    axes[0].set_ylabel("Pointing Game Score (%)")
    axes[0].set_title("Overall CAM Quality by Model")
    axes[0].set_ylim(0, 100)
    
    for i, (arch, score) in enumerate(zip(archs, scores)):
        axes[0].text(i, score + 2, f"{score:.1f}%", ha='center', fontweight='bold')
    
    # Per-class scores for best model
    best_arch = max(results.keys(), key=lambda a: results[a]["overall_score"])
    per_class = results[best_arch]["per_class_scores"]
    
    classes = list(per_class.keys())
    class_scores = [per_class[c] * 100 for c in classes]
    
    axes[1].barh(classes, class_scores, color='#3498db')
    axes[1].set_xlabel("Pointing Game Score (%)")
    axes[1].set_title(f"Per-Class CAM Quality ({best_arch})")
    axes[1].set_xlim(0, 100)
    
    plt.tight_layout()
    
    output_path = PROJECT_ROOT / "outputs" / "results" / "cam_quality_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def run_cross_validation_analysis():
    """
    Simulate domain generalization by evaluating on held-out drivers.
    This tests how well models generalize to unseen subjects.
    """
    print("\n" + "="*60)
    print("  DOMAIN GENERALIZATION ANALYSIS")
    print("  (Cross-Subject Evaluation)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load driver list
    driver_csv = PROJECT_ROOT / "data" / "raw" / "state_farm" / "driver_imgs_list.csv"
    if not driver_csv.exists():
        print("ERROR: driver_imgs_list.csv not found!")
        return None
    
    import pandas as pd
    df = pd.read_csv(driver_csv)
    
    # Get unique drivers
    drivers = df['subject'].unique()
    print(f"Total drivers: {len(drivers)}")
    
    # Use last 2 drivers as "unseen domain"
    test_drivers = drivers[-2:]
    print(f"Test drivers (unseen): {test_drivers}")
    
    # Filter test images
    test_df = df[df['subject'].isin(test_drivers)]
    print(f"Test images: {len(test_df)}")
    
    data_dir = PROJECT_ROOT / "data" / "raw" / "state_farm" / "imgs" / "train"
    transform = val_transforms(224)
    
    architectures = ["efficientnet_b0", "mobilenet_v3_small", "resnet50"]
    results = {}
    
    for arch in architectures:
        print(f"\nEvaluating {arch} on unseen drivers...")
        model = load_model(arch, device)
        if model is None:
            continue
        
        correct = 0
        total = 0
        
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"  {arch}"):
            img_path = data_dir / row['classname'] / row['img']
            if not img_path.exists():
                continue
            
            # Load and predict
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                pred = output.argmax(dim=1).item()
            
            true_class = int(row['classname'][1])
            if pred == true_class:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        results[arch] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
        print(f"  Accuracy on unseen drivers: {accuracy*100:.2f}%")
    
    # Save results
    output_path = PROJECT_ROOT / "outputs" / "results" / "domain_generalization_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Plot
    plot_domain_generalization(results)
    
    return results


def plot_domain_generalization(results):
    """Plot domain generalization comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    archs = list(results.keys())
    accuracies = [results[a]["accuracy"] * 100 for a in archs]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(archs, accuracies, color=colors)
    
    ax.set_ylabel("Accuracy on Unseen Drivers (%)")
    ax.set_title("Domain Generalization: Cross-Subject Evaluation")
    ax.set_ylim(0, 100)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{acc:.1f}%", ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = PROJECT_ROOT / "outputs" / "results" / "domain_generalization_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  NOVEL CONTRIBUTIONS ANALYSIS")
    print("  Distracted Driver Detection Project")
    print("="*60)
    
    # Run CAM Quality Analysis
    cam_results = run_cam_quality_analysis()
    
    # Run Domain Generalization Analysis
    domain_results = run_cross_validation_analysis()
    
    print("\n" + "="*60)
    print("  ANALYSIS COMPLETE")
    print("="*60)
    
    if cam_results:
        print("\nCAM Quality (Pointing Game):")
        for arch, res in cam_results.items():
            print(f"  {arch}: {res['overall_score']*100:.1f}%")
    
    if domain_results:
        print("\nDomain Generalization (Unseen Drivers):")
        for arch, res in domain_results.items():
            print(f"  {arch}: {res['accuracy']*100:.1f}%")
