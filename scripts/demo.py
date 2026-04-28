"""
Demo script for video recording - Distracted Driver Detection
Run this to show live inference on sample images with Grad-CAM visualization.
"""

import sys
import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.transforms import val_transforms
from models.model_factory import get_model

# Class names
CLASS_NAMES = [
    "c0 - Safe Driving",
    "c1 - Texting (Right)",
    "c2 - Phone (Right)", 
    "c3 - Texting (Left)",
    "c4 - Phone (Left)",
    "c5 - Adjusting Radio",
    "c6 - Drinking",
    "c7 - Reaching Behind",
    "c8 - Hair/Makeup",
    "c9 - Talking to Passenger"
]

def load_model():
    """Load trained EfficientNet-B0 model."""
    print("\n" + "="*60)
    print("  DISTRACTED DRIVER DETECTION - DEMO")
    print("  Group 26: Jayasurya & Karthikaa")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/4] Loading model on {device}...")
    
    model = get_model("efficientnet_b0", num_classes=10, pretrained=False)
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "best_efficientnet_b0.pth"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"      Loaded checkpoint: {checkpoint_path.name}")
        best_acc = checkpoint.get('best_acc', 0)
        if isinstance(best_acc, (int, float)):
            print(f"      Best accuracy: {best_acc:.2f}%")
        else:
            print(f"      Checkpoint loaded successfully")
    else:
        print("      WARNING: No checkpoint found, using random weights")
    
    model = model.to(device)
    model.eval()
    return model, device

def get_sample_images():
    """Get sample images from each class for demo."""
    print("\n[2/4] Finding sample images...")
    
    data_dir = PROJECT_ROOT / "data" / "raw" / "state_farm" / "imgs" / "train"
    samples = []
    
    for i in range(10):
        class_dir = data_dir / f"c{i}"
        if class_dir.exists():
            images = list(class_dir.glob("*.jpg"))[:1]  # Get 1 image per class
            if images:
                samples.append(images[0])
                print(f"      Found sample for {CLASS_NAMES[i][:20]}...")
    
    print(f"      Total samples: {len(samples)}")
    return samples

def run_inference(model, device, image_path, transform):
    """Run inference on a single image."""
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image.resize((224, 224)))
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        start_time = time.time()
        outputs = model(input_tensor)
        inference_time = (time.time() - start_time) * 1000
        
        probs = torch.softmax(outputs, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item() * 100
    
    return pred_class, confidence, inference_time, original_image, input_tensor

def generate_simple_cam(model, input_tensor, pred_class, device):
    """Generate simple CAM-like heatmap using gradients."""
    model.eval()
    input_tensor.requires_grad = True
    
    # Forward pass
    outputs = model(input_tensor)
    
    # Backward pass for target class
    model.zero_grad()
    outputs[0, pred_class].backward()
    
    # Get gradients and create heatmap
    gradients = input_tensor.grad.data.abs()
    heatmap = gradients[0].mean(dim=0).cpu().numpy()
    
    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on image."""
    import cv2
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlaid = np.uint8(alpha * heatmap_colored + (1 - alpha) * image)
    return overlaid

def demo():
    """Main demo function."""
    # Load model
    model, device = load_model()
    transform = val_transforms(224)
    
    # Get samples
    samples = get_sample_images()
    if not samples:
        print("ERROR: No sample images found!")
        return
    
    print("\n[3/4] Running inference with Grad-CAM...")
    print("-" * 60)
    
    # Create figure for visualization
    n_samples = min(6, len(samples))
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
    
    total_time = 0
    correct = 0
    
    for idx, img_path in enumerate(samples[:n_samples]):
        true_class = int(img_path.parent.name[1])
        
        # Run inference
        pred_class, confidence, inf_time, orig_img, input_tensor = run_inference(
            model, device, img_path, transform
        )
        total_time += inf_time
        
        # Check if correct
        is_correct = pred_class == true_class
        if is_correct:
            correct += 1
        
        # Generate attention heatmap
        try:
            heatmap = generate_simple_cam(model, input_tensor.clone(), pred_class, device)
            overlaid = overlay_heatmap(orig_img, heatmap, alpha=0.4)
        except Exception as e:
            overlaid = orig_img
        
        # Print result
        status = "✓" if is_correct else "✗"
        print(f"  {status} Image {idx+1}: Predicted {CLASS_NAMES[pred_class][:25]} "
              f"({confidence:.1f}%) in {inf_time:.1f}ms")
        
        # Plot original
        axes[0, idx].imshow(orig_img)
        axes[0, idx].set_title(f"True: {CLASS_NAMES[true_class][:15]}", fontsize=10)
        axes[0, idx].axis('off')
        
        # Plot Grad-CAM
        axes[1, idx].imshow(overlaid)
        color = 'green' if is_correct else 'red'
        axes[1, idx].set_title(f"Pred: {CLASS_NAMES[pred_class][:15]}\n{confidence:.1f}%", 
                               fontsize=10, color=color)
        axes[1, idx].axis('off')
    
    plt.suptitle("Distracted Driver Detection with Grad-CAM Visualization\n"
                 "Top: Original | Bottom: Grad-CAM Attention", fontsize=14)
    plt.tight_layout()
    
    # Save figure
    output_path = PROJECT_ROOT / "outputs" / "results" / "demo_output.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n      Saved visualization to: {output_path.name}")
    
    # Show summary
    print("\n" + "="*60)
    print("  DEMO SUMMARY")
    print("="*60)
    print(f"  Model: EfficientNet-B0 (4.02M parameters)")
    print(f"  Samples tested: {n_samples}")
    print(f"  Accuracy: {correct}/{n_samples} ({100*correct/n_samples:.1f}%)")
    print(f"  Avg inference time: {total_time/n_samples:.1f}ms")
    print(f"  Throughput: {1000/(total_time/n_samples):.1f} FPS")
    print("="*60)
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    demo()
