# Distracted Driver Detection with Deep Learning

**Group 26** | Jayasurya Jayadevan (121306067) & Karthikaa Mikkilineni (20602214)

A deep learning system for real-time distracted driver behavior classification using CNN architectures with explainability via Grad-CAM visualization.

---

## 🚀 Quick Start (Run in 4 Steps)

### Step 1: Clone Repository
```bash
git clone https://github.com/jsurya24082000/distracted-driver-detection.git
cd distracted-driver-detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Pre-trained Model (if not included)
Models are included in `outputs/checkpoints/`. If missing, they will be downloaded automatically.

### Step 4: Run the Demo
```bash
python run_demo.py
```

Or launch the interactive web app:
```bash
streamlit run app.py
```

---

## 📊 Results Summary

| Model | Accuracy | F1 Score | Parameters | Speed |
|-------|----------|----------|------------|-------|
| **ResNet50** | **90.5%** | **89.8%** | 23.5M | 31 FPS |
| EfficientNet-B0 | 88.1% | 86.9% | 4.0M | 72 FPS |
| MobileNetV3 | 85.9% | 85.1% | 1.5M | 136 FPS |

### Novel Contributions

| Analysis | Result |
|----------|--------|
| Cross-Subject Generalization | 99.7% accuracy on unseen drivers |
| Cross-Dataset Generalization | 78.0% (State Farm → Roboflow) |
| CAM Quality (Pointing Game) | 40.7% (EfficientNet-B0 best) |

---

## 📁 Project Structure

```
distracted-driver-detection/
├── app.py                    # Streamlit web demo
├── run_demo.py               # Quick inference demo
├── requirements.txt          # Dependencies
├── data/
│   ├── dataset.py            # Dataset classes
│   └── transforms.py         # Data augmentation
├── models/
│   └── model_factory.py      # Model architectures
├── scripts/
│   ├── train.py              # Training script
│   ├── demo.py               # Demo with visualizations
│   └── run_analysis.py       # Domain generalization analysis
├── outputs/
│   ├── checkpoints/          # Trained model weights
│   └── results/              # Evaluation results & plots
└── notebooks/                # Jupyter notebooks
```

---

## 🎯 Dataset

**State Farm Distracted Driver Detection** (Kaggle)
- 22,424 labeled images | 10 classes | 26 unique drivers

| Class | Description |
|-------|-------------|
| c0 | Safe driving |
| c1 | Texting (right hand) |
| c2 | Talking on phone (right hand) |
| c3 | Texting (left hand) |
| c4 | Talking on phone (left hand) |
| c5 | Operating the radio |
| c6 | Drinking |
| c7 | Reaching behind |
| c8 | Hair and makeup |
| c9 | Talking to passenger |

---

## 💻 Usage

### Run Inference on Sample Images
```bash
python run_demo.py
```

### Launch Web Demo
```bash
streamlit run app.py
```

### Train Models (Optional)
```bash
python scripts/train.py --arch efficientnet_b0 --epochs 5
```

### Run Analysis
```bash
python scripts/run_analysis.py
```

## Project Structure

```
distracted_driver_detection/
├── data/
│   ├── __init__.py
│   ├── dataset.py          # Custom PyTorch Dataset classes
│   └── transforms.py       # Albumentations augmentation pipelines
├── models/
│   ├── __init__.py
│   └── model_factory.py    # Model loading and configuration
├── training/
│   ├── __init__.py
│   ├── trainer.py          # Training loop and validation
│   └── losses.py           # Loss functions and class weights
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py          # Classification metrics
│   └── efficiency.py       # FLOPs, latency, parameter counting
├── explainability/
│   ├── __init__.py
│   └── gradcam.py          # Grad-CAM and EigenCAM visualization
├── configs/
│   └── config.yaml         # Hyperparameters and paths
├── notebooks/
│   ├── 01_train_and_evaluate.ipynb
│   ├── 02_gradcam_visualization.ipynb
│   └── 03_domain_generalization.ipynb
├── scripts/
│   ├── train.py            # CLI training script
│   ├── evaluate.py         # CLI evaluation script
│   └── export_model.py     # Model export script
├── outputs/                # Generated at runtime
│   ├── checkpoints/
│   ├── gradcam_maps/
│   └── results/
├── utils.py                # Helper functions
├── requirements.txt
└── README.md
```

## Expected Results

| Architecture | Accuracy | Macro F1 | Params (M) | FLOPs (G) | Latency (ms) |
|--------------|----------|----------|------------|-----------|--------------|
| EfficientNet-B0 | ~93% | ~92% | 4.0 | 0.39 | ~15 |
| MobileNetV3-Small | ~89% | ~88% | 1.5 | 0.06 | ~8 |
| ResNet-50 | ~92% | ~91% | 23.5 | 4.1 | ~25 |

*Results may vary based on training configuration and hardware.*

## Grad-CAM Examples

Grad-CAM visualizations show which image regions the model focuses on for predictions:

- **Safe Driving**: Attention on hands on steering wheel
- **Texting**: Focus on phone and hand position
- **Phone Call**: Attention on phone near ear
- **Drinking**: Focus on cup/bottle and hand-to-mouth motion

*See `outputs/gradcam_maps/` for generated visualizations.*

## Configuration

Key hyperparameters in `configs/config.yaml`:

```yaml
training:
  epochs: 30
  batch_size: 32
  learning_rate: 1e-4
  optimizer: adamw
  scheduler: cosine
  label_smoothing: 0.1
  use_class_weights: true
```

## Technical Details

### Data Augmentation
- RandomResizedCrop (scale 0.7-1.0)
- HorizontalFlip (p=0.3)
- RandomBrightnessContrast (p=0.4)
- HueSaturationValue (p=0.3)
- CoarseDropout (p=0.3)
- GridDistortion (p=0.2)

### Subject-Aware Splitting
Images are grouped by subject ID and split at the subject level (80/20) to prevent data leakage between training and validation sets.

### Grad-CAM Methods
- **GradCAM**: Used for EfficientNet-B0 and ResNet-50
- **EigenCAM**: Used for MobileNetV3-Small (better suited for depthwise separable convolutions)

## Citations

```bibtex
@article{tan2019efficientnet,
  title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc},
  journal={ICML},
  year={2019}
}

@article{howard2019mobilenetv3,
  title={Searching for MobileNetV3},
  author={Howard, Andrew and others},
  journal={ICCV},
  year={2019}
}

@article{selvaraju2017gradcam,
  title={Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization},
  author={Selvaraju, Ramprasaath R and others},
  journal={ICCV},
  year={2017}
}
```

## License

This project is for educational and research purposes. The State Farm dataset is subject to Kaggle competition rules.

## Acknowledgments

- State Farm for providing the distracted driver dataset
- PyTorch team for the deep learning framework
- pytorch-grad-cam library for CAM implementations
