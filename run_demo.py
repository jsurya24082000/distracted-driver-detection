"""
Distracted Driver Detection - Quick Demo
Run: python run_demo.py

This script demonstrates the trained model on sample images.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.transforms import val_transforms
from models.model_factory import get_model

# Class names
CLASS_NAMES = [
    "Safe Driving", "Texting (R)", "Phone (R)", "Texting (L)", "Phone (L)",
    "Radio", "Drinking", "Reaching", "Makeup", "Passenger"
]

CLASS_EMOJIS = ["✅", "📱", "📞", "📱", "📞", "📻", "🥤", "🔙", "💄", "💬"]


def load_model(arch_name="efficientnet_b0"):
    """Load the best trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(arch_name, num_classes=10, pretrained=False)
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / f"best_{arch_name}.pth"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ Loaded {arch_name} model")
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("  Please ensure model weights are in outputs/checkpoints/")
        return None, None
    
    model = model.to(device)
    model.eval()
    return model, device


def predict_image(model, device, image_path):
    """Run prediction on a single image."""
    transform = val_transforms(224)
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    return pred_class, confidence, probs[0].cpu().numpy()


def find_sample_images():
    """Find sample images from the dataset."""
    samples = []
    data_dir = PROJECT_ROOT / "data" / "raw" / "state_farm" / "imgs" / "train"
    
    if not data_dir.exists():
        print("Dataset not found. Using placeholder demo.")
        return []
    
    for class_idx in range(10):
        class_dir = data_dir / f"c{class_idx}"
        if class_dir.exists():
            images = list(class_dir.glob("*.jpg"))[:1]
            if images:
                samples.append((images[0], class_idx))
    
    return samples


def run_demo():
    """Run the demonstration."""
    print("\n" + "="*60)
    print("  DISTRACTED DRIVER DETECTION DEMO")
    print("  Group 26 - Jayasurya & Karthikaa")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model, device = load_model("efficientnet_b0")
    
    if model is None:
        return
    
    print(f"Device: {device}")
    
    # Find sample images
    samples = find_sample_images()
    
    if not samples:
        print("\n⚠ No sample images found.")
        print("  To run inference, place images in data/raw/state_farm/imgs/train/")
        print("  Or use the Streamlit app: streamlit run app.py")
        return
    
    print(f"\nRunning inference on {len(samples)} sample images...\n")
    print("-" * 60)
    
    correct = 0
    for img_path, true_class in samples:
        pred_class, confidence, _ = predict_image(model, device, img_path)
        
        is_correct = pred_class == true_class
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        emoji = CLASS_EMOJIS[pred_class]
        
        print(f"{status} True: c{true_class} ({CLASS_NAMES[true_class]})")
        print(f"  Pred: c{pred_class} ({CLASS_NAMES[pred_class]}) {emoji}")
        print(f"  Confidence: {confidence*100:.1f}%")
        print()
    
    print("-" * 60)
    print(f"Accuracy: {correct}/{len(samples)} ({correct/len(samples)*100:.1f}%)")
    print("="*60)
    
    print("\n📌 To try the interactive web demo, run:")
    print("   streamlit run app.py")
    print()


if __name__ == "__main__":
    run_demo()
