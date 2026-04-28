# Distracted Driver Behavior Detection using CNNs with Grad-CAM Explainability

A comprehensive deep learning project for classifying distracted driver behavior from in-cabin camera images using three CNN architectures with Grad-CAM visualization for model explainability.

## Project Overview

### Motivation
Distracted driving is a leading cause of road accidents worldwide. This project develops and compares CNN-based classifiers to automatically detect distracted driving behaviors from dashboard camera footage, enabling real-time driver monitoring systems.

### Features
- **Three CNN Architectures**: EfficientNet-B0, MobileNetV3-Small, and ResNet-50
- **Grad-CAM Explainability**: Visual explanations of model predictions
- **Subject-Aware Data Splitting**: Prevents data leakage between train/val sets
- **Comprehensive Evaluation**: Accuracy, F1, confusion matrices, efficiency metrics
- **Model Export**: ONNX and TorchScript formats for deployment
- **Domain Generalization Testing**: Zero-shot evaluation on AUC dataset

### Dataset
**State Farm Distracted Driver Detection** (Kaggle)
- 22,424 labeled images
- 10 classes (c0-c9)
- 26 unique subjects

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

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd distracted_driver_detection
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
1. Download from [Kaggle State Farm Competition](https://www.kaggle.com/c/state-farm-distracted-driver-detection)
2. Extract to `data/raw/state_farm/`
3. Ensure the following structure:
```
data/raw/state_farm/
├── imgs/
│   └── train/
│       ├── c0/
│       ├── c1/
│       └── ...
└── driver_imgs_list.csv
```

## Usage

### Training

Train all three models:
```bash
python scripts/train.py --arch all
```

Train a specific model:
```bash
python scripts/train.py --arch efficientnet_b0 --epochs 30
```

With custom configuration:
```bash
python scripts/train.py --arch resnet50 --config configs/config.yaml --data_dir path/to/data
```

### Evaluation

Evaluate a trained model:
```bash
python scripts/evaluate.py --arch efficientnet_b0 --checkpoint outputs/checkpoints/best_efficientnet_b0.pth
```

With AUC dataset evaluation:
```bash
python scripts/evaluate.py --arch efficientnet_b0 --checkpoint outputs/checkpoints/best_efficientnet_b0.pth --auc_test
```

### Model Export

Export to ONNX and TorchScript:
```bash
python scripts/export_model.py --arch efficientnet_b0 --checkpoint outputs/checkpoints/best_efficientnet_b0.pth
```

### Jupyter Notebooks

1. **01_train_and_evaluate.ipynb**: Complete training pipeline with visualizations
2. **02_gradcam_visualization.ipynb**: Grad-CAM analysis and model comparison
3. **03_domain_generalization.ipynb**: Cross-dataset evaluation

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
