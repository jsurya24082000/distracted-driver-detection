"""
Distracted Driver Detection - Hugging Face Spaces Demo
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from huggingface_hub import hf_hub_download
import os

# Class names
CLASS_NAMES = [
    "Safe Driving", "Texting (Right)", "Phone Call (Right)", 
    "Texting (Left)", "Phone Call (Left)", "Operating Radio",
    "Drinking", "Reaching Behind", "Hair/Makeup", "Talking to Passenger"
]

CLASS_DESCRIPTIONS = {
    0: ("✅ Safe", "Driver is focused on the road"),
    1: ("⚠️ Texting (R)", "Driver texting with right hand"),
    2: ("⚠️ Phone (R)", "Driver on phone with right hand"),
    3: ("⚠️ Texting (L)", "Driver texting with left hand"),
    4: ("⚠️ Phone (L)", "Driver on phone with left hand"),
    5: ("⚠️ Radio", "Driver adjusting radio/console"),
    6: ("⚠️ Drinking", "Driver drinking beverage"),
    7: ("🚨 Reaching", "Driver reaching to back seat"),
    8: ("⚠️ Makeup", "Driver grooming"),
    9: ("⚠️ Passenger", "Driver talking to passenger"),
}

# Model cache
model = None
device = None

def get_model():
    """Load EfficientNet-B0 model."""
    global model, device
    
    if model is not None:
        return model, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create EfficientNet-B0 model
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    
    # Try to load weights from Hugging Face Hub
    try:
        weights_path = hf_hub_download(
            repo_id="jsurya24082000/distracted-driver-detection",
            filename="best_efficientnet_b0.pth"
        )
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✓ Loaded model from Hugging Face Hub")
    except Exception as e:
        print(f"Using pretrained weights: {e}")
        # Use pretrained ImageNet weights as fallback
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    
    model = model.to(device)
    model.eval()
    
    return model, device


def get_transform():
    """Get validation transforms."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict(image):
    """Run prediction on uploaded image."""
    if image is None:
        return None, "Please upload an image"
    
    model, device = get_model()
    transform = get_transform()
    
    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    image = image.convert("RGB")
    
    # Transform and predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Create results
    label, description = CLASS_DESCRIPTIONS[pred_class]
    
    # Create probability dict for label output
    prob_dict = {CLASS_NAMES[i]: float(probs[0, i]) for i in range(10)}
    
    result_text = f"**{label}**\n\n{description}\n\nConfidence: {confidence*100:.1f}%"
    
    return prob_dict, result_text


# Create Gradio interface
with gr.Blocks(title="Distracted Driver Detection") as demo:
    gr.Markdown("""
    # 🚗 Distracted Driver Detection
    
    **Group 26** | Jayasurya Jayadevan & Karthikaa Mikkilineni
    
    Upload an image of a driver to classify their behavior into one of 10 categories.
    
    This model was trained on the State Farm Distracted Driver Detection dataset and achieves **88.1% accuracy**.
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Driver Image", type="pil")
            submit_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column():
            output_label = gr.Label(label="Class Probabilities", num_top_classes=5)
            output_text = gr.Markdown(label="Result")
    
    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_label, output_text]
    )
    
    gr.Markdown("""
    ---
    ### Classes:
    - **c0**: Safe Driving ✅
    - **c1-c4**: Phone-related distractions (texting/calling) ⚠️
    - **c5**: Operating Radio ⚠️
    - **c6**: Drinking ⚠️
    - **c7**: Reaching Behind 🚨
    - **c8**: Hair/Makeup ⚠️
    - **c9**: Talking to Passenger ⚠️
    
    [GitHub Repository](https://github.com/jsurya24082000/distracted-driver-detection)
    """)

if __name__ == "__main__":
    demo.launch()
