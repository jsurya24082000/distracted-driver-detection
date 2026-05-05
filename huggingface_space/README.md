---
title: Distracted Driver Detection
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Distracted Driver Detection

**Group 26** | Jayasurya Jayadevan & Karthikaa Mikkilineni

A deep learning system for real-time classification of distracted driver behaviors.

## Model Performance

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| EfficientNet-B0 | 88.1% | 86.9% |

## Classes

- c0: Safe Driving
- c1: Texting (Right Hand)
- c2: Phone Call (Right Hand)
- c3: Texting (Left Hand)
- c4: Phone Call (Left Hand)
- c5: Operating Radio
- c6: Drinking
- c7: Reaching Behind
- c8: Hair/Makeup
- c9: Talking to Passenger

## Usage

Upload an image of a driver and the model will classify their behavior.

## Links

- [GitHub Repository](https://github.com/jsurya24082000/distracted-driver-detection)
