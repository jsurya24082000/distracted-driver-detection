"""
Streamlit Web Demo for Distracted Driver Detection
Run with: streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
import torch
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.transforms import val_transforms
from models.model_factory import get_model

# Class names and descriptions
CLASS_INFO = {
    0: ("Safe Driving", "Driver is focused on the road", "✅"),
    1: ("Texting (Right)", "Driver texting with right hand", "⚠️"),
    2: ("Phone Call (Right)", "Driver on phone with right hand", "⚠️"),
    3: ("Texting (Left)", "Driver texting with left hand", "⚠️"),
    4: ("Phone Call (Left)", "Driver on phone with left hand", "⚠️"),
    5: ("Adjusting Radio", "Driver adjusting radio/console", "⚠️"),
    6: ("Drinking", "Driver drinking beverage", "⚠️"),
    7: ("Reaching Behind", "Driver reaching to back seat", "🚨"),
    8: ("Hair/Makeup", "Driver grooming", "⚠️"),
    9: ("Talking to Passenger", "Driver talking to passenger", "⚠️"),
}

@st.cache_resource
def load_model(arch_name="efficientnet_b0"):
    """Load trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(arch_name, num_classes=10, pretrained=False)
    
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / f"best_{arch_name}.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(device)
    model.eval()
    return model, device

def predict(model, device, image):
    """Run prediction on image."""
    transform = val_transforms(224)
    
    # Transform image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    return pred_class, confidence, probs[0].cpu().numpy()

def main():
    st.set_page_config(
        page_title="Distracted Driver Detection",
        page_icon="🚗",
        layout="wide"
    )
    
    # Header
    st.title("🚗 Distracted Driver Detection")
    st.markdown("**Deep Learning System for Real-Time Driver Behavior Classification**")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Model selection
    available_models = []
    for arch in ["efficientnet_b0", "mobilenet_v3_small", "resnet50"]:
        if (PROJECT_ROOT / "outputs" / "checkpoints" / f"best_{arch}.pth").exists():
            available_models.append(arch)
    
    if not available_models:
        st.error("No trained models found! Please train a model first.")
        return
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        available_models,
        format_func=lambda x: {
            "efficientnet_b0": "EfficientNet-B0 (Best Accuracy)",
            "mobilenet_v3_small": "MobileNetV3 (Fastest)",
            "resnet50": "ResNet50 (Largest)"
        }.get(x, x)
    )
    
    # Load model
    with st.spinner(f"Loading {selected_model}..."):
        model, device = load_model(selected_model)
    
    st.sidebar.success(f"✅ Model loaded on {device}")
    
    # Model info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    model_stats = {
        "efficientnet_b0": ("4.02M", "88.1%", "72 FPS"),
        "mobilenet_v3_small": ("1.53M", "85.9%", "136 FPS"),
        "resnet50": ("23.5M", "~87%", "25 FPS")
    }
    params, acc, fps = model_stats.get(selected_model, ("N/A", "N/A", "N/A"))
    st.sidebar.markdown(f"- **Parameters:** {params}")
    st.sidebar.markdown(f"- **Accuracy:** {acc}")
    st.sidebar.markdown(f"- **Speed:** {fps}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a driver image...",
            type=["jpg", "jpeg", "png"],
            help="Upload an image of a driver to classify their behavior"
        )
        
        # Sample images
        st.markdown("---")
        st.markdown("**Or try a sample image:**")
        
        sample_dir = PROJECT_ROOT / "data" / "raw" / "state_farm" / "imgs" / "train"
        if sample_dir.exists():
            sample_cols = st.columns(5)
            for i, col in enumerate(sample_cols):
                class_dir = sample_dir / f"c{i}"
                if class_dir.exists():
                    samples = list(class_dir.glob("*.jpg"))[:1]
                    if samples:
                        with col:
                            if st.button(f"c{i}", key=f"sample_{i}"):
                                uploaded_file = samples[0]
    
    with col2:
        st.header("🔍 Prediction Result")
        
        if uploaded_file is not None:
            # Load image
            if isinstance(uploaded_file, Path):
                image = Image.open(uploaded_file).convert("RGB")
            else:
                image = Image.open(uploaded_file).convert("RGB")
            
            # Display image
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Run prediction
            with st.spinner("Analyzing..."):
                pred_class, confidence, all_probs = predict(model, device, image)
            
            # Get class info
            class_name, description, emoji = CLASS_INFO[pred_class]
            
            # Display result
            if pred_class == 0:
                st.success(f"## {emoji} {class_name}")
            else:
                st.warning(f"## {emoji} {class_name}")
            
            st.markdown(f"**{description}**")
            st.markdown(f"**Confidence:** {confidence*100:.1f}%")
            
            # Progress bar for confidence
            st.progress(confidence)
            
            # Show all probabilities
            st.markdown("---")
            st.markdown("### 📊 All Class Probabilities")
            
            for i, prob in enumerate(all_probs):
                name, _, icon = CLASS_INFO[i]
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(float(prob), text=f"{icon} {name}")
                with col_b:
                    st.markdown(f"**{prob*100:.1f}%**")
        else:
            st.info("👆 Upload an image to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Group 26** | Jayasurya Jayadevan & Karthikaa Mikkilineni | "
        "[GitHub](https://github.com/jsurya24082000/distracted-driver-detection)"
    )

if __name__ == "__main__":
    main()
