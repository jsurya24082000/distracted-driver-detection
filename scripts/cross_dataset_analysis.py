"""
Cross-Dataset Domain Generalization Analysis
Evaluates models trained on State Farm on the Roboflow dataset (different domain)
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

# State Farm classes (training domain)
SF_CLASSES = [
    "Safe Driving", "Texting (R)", "Phone (R)", "Texting (L)", "Phone (L)",
    "Radio", "Drinking", "Reaching", "Makeup", "Passenger"
]

# Roboflow classes (target domain) - from data.yaml
RF_CLASSES = [
    'drinking', 'hair and makeup', 'operating the radio', 
    'reaching behind', 'safe driving', 'talking on the phone', 
    'talking to passenger', 'texting'
]

# Mapping from Roboflow class index to State Farm class index
# Some classes don't have exact matches
RF_TO_SF_MAPPING = {
    0: 6,   # drinking -> Drinking
    1: 8,   # hair and makeup -> Makeup
    2: 5,   # operating the radio -> Radio
    3: 7,   # reaching behind -> Reaching
    4: 0,   # safe driving -> Safe Driving
    5: 2,   # talking on the phone -> Phone (R) (approximate)
    6: 9,   # talking to passenger -> Passenger
    7: 1,   # texting -> Texting (R) (approximate)
}

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


def get_roboflow_labels(labels_dir):
    """Parse YOLO format labels to get class for each image."""
    labels = {}
    for label_file in labels_dir.glob("*.txt"):
        img_name = label_file.stem
        with open(label_file, 'r') as f:
            lines = f.readlines()
            if lines:
                # YOLO format: class_id x_center y_center width height
                # Take the first object's class
                class_id = int(lines[0].split()[0])
                labels[img_name] = class_id
    return labels


def evaluate_on_roboflow(model, arch_name, data_dir, device):
    """Evaluate model on Roboflow dataset."""
    transform = val_transforms(224)
    
    results = {
        "arch": arch_name,
        "total": 0,
        "correct": 0,
        "correct_mapped": 0,
        "per_class_correct": {c: 0 for c in RF_CLASSES},
        "per_class_total": {c: 0 for c in RF_CLASSES},
    }
    
    # Process test set
    for split in ["test", "valid"]:
        images_dir = data_dir / split / "images"
        labels_dir = data_dir / split / "labels"
        
        if not images_dir.exists():
            continue
        
        labels = get_roboflow_labels(labels_dir)
        
        for img_path in tqdm(list(images_dir.glob("*.jpg")), desc=f"  {split}"):
            img_name = img_path.stem
            if img_name not in labels:
                continue
            
            rf_class = labels[img_name]
            sf_class = RF_TO_SF_MAPPING.get(rf_class)
            
            if sf_class is None:
                continue
            
            # Load and predict
            try:
                image = Image.open(img_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    pred = output.argmax(dim=1).item()
                
                results["total"] += 1
                results["per_class_total"][RF_CLASSES[rf_class]] += 1
                
                # Check if prediction matches mapped class
                if pred == sf_class:
                    results["correct_mapped"] += 1
                    results["per_class_correct"][RF_CLASSES[rf_class]] += 1
                    
            except Exception as e:
                continue
    
    # Calculate accuracy
    results["accuracy"] = results["correct_mapped"] / results["total"] if results["total"] > 0 else 0
    
    # Per-class accuracy
    results["per_class_accuracy"] = {}
    for c in RF_CLASSES:
        if results["per_class_total"][c] > 0:
            results["per_class_accuracy"][c] = results["per_class_correct"][c] / results["per_class_total"][c]
        else:
            results["per_class_accuracy"][c] = 0
    
    return results


def run_cross_dataset_analysis():
    """Run cross-dataset domain generalization analysis."""
    print("\n" + "="*60)
    print("  CROSS-DATASET DOMAIN GENERALIZATION")
    print("  Training: State Farm | Testing: Roboflow")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    data_dir = PROJECT_ROOT / "data" / "raw" / "roboflow_dataset"
    
    if not data_dir.exists():
        print("ERROR: Roboflow dataset not found!")
        return None
    
    architectures = ["efficientnet_b0", "mobilenet_v3_small", "resnet50"]
    all_results = {}
    
    for arch in architectures:
        print(f"\nEvaluating {arch} on Roboflow dataset...")
        model = load_model(arch, device)
        if model is None:
            continue
        
        results = evaluate_on_roboflow(model, arch, data_dir, device)
        all_results[arch] = results
        
        print(f"  Cross-Dataset Accuracy: {results['accuracy']*100:.1f}%")
        print(f"  Samples evaluated: {results['total']}")
    
    # Save results
    output_path = PROJECT_ROOT / "outputs" / "results" / "cross_dataset_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Plot comparison
    plot_cross_dataset_results(all_results)
    
    return all_results


def plot_cross_dataset_results(results):
    """Plot cross-dataset domain generalization results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall accuracy comparison
    archs = list(results.keys())
    accuracies = [results[a]["accuracy"] * 100 for a in archs]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = axes[0].bar(archs, accuracies, color=colors)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Cross-Dataset Domain Generalization\n(State Farm → Roboflow)")
    axes[0].set_ylim(0, 100)
    
    for bar, acc in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f"{acc:.1f}%", ha='center', fontweight='bold')
    
    # Per-class accuracy for best model
    best_arch = max(results.keys(), key=lambda a: results[a]["accuracy"])
    per_class = results[best_arch]["per_class_accuracy"]
    
    classes = list(per_class.keys())
    class_accs = [per_class[c] * 100 for c in classes]
    
    axes[1].barh(classes, class_accs, color='#3498db')
    axes[1].set_xlabel("Accuracy (%)")
    axes[1].set_title(f"Per-Class Accuracy on Roboflow ({best_arch})")
    axes[1].set_xlim(0, 100)
    
    plt.tight_layout()
    
    output_path = PROJECT_ROOT / "outputs" / "results" / "cross_dataset_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    results = run_cross_dataset_analysis()
    
    if results:
        print("\n" + "="*60)
        print("  CROSS-DATASET ANALYSIS COMPLETE")
        print("="*60)
        print("\nDomain Generalization Results (State Farm → Roboflow):")
        for arch, res in results.items():
            print(f"  {arch}: {res['accuracy']*100:.1f}%")
        
        print("\nKey Insight: Lower accuracy on Roboflow indicates domain shift")
        print("(different cameras, lighting, subjects, car interiors)")
