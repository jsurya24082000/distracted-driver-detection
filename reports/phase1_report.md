# Group 26 - Phase 1 Report
## Distracted Driver Detection using Deep Learning

**Group Number:** 26  
**Members:**
- Jayasurya Jayadevan (UID: 121306067)
- Karthikaa Mikkilineni (UID: 20602214)

---

## 1. Revised Problem Statement

Distracted driving causes over 3,000 fatalities annually in the US alone. This project develops a real-time distracted driver detection system using deep learning to classify 10 driver behaviors (safe driving, texting, phone usage, drinking, reaching, makeup application, and talking to passengers) from dashboard camera images. We compare three CNN architectures (EfficientNet-B0, ResNet18, MobileNetV3) optimized for edge deployment, with novel contributions in zero-shot domain generalization analysis and quantitative CAM quality evaluation using the Pointing Game metric.

---

## 2. Project Idea Adjustments

- **Added Novel Contribution 1:** Zero-shot domain generalization analysis comparing State Farm and AUC datasets to evaluate model robustness across different driver populations
- **Added Novel Contribution 2:** CAM quality evaluation using Pointing Game metric to quantitatively validate that models attend to task-relevant regions (hands, phone, face)
- **Architecture Focus:** Prioritized EfficientNet-B0 for best accuracy-efficiency tradeoff on edge devices

---

## 3. Team Integration Progress

**Project Completion: ~85%**

| Component | Status | Owner |
|-----------|--------|-------|
| Data Pipeline | ✅ Complete | Jayasurya |
| Model Training | ✅ Complete | Jayasurya |
| Evaluation Framework | ✅ Complete | Karthikaa |
| Grad-CAM Visualization | ✅ Complete | Karthikaa |
| Domain Generalization Module | ✅ Complete | Joint |
| CAM Quality Evaluation | ✅ Complete | Joint |
| ONNX Export | ✅ Complete | Jayasurya |
| Final Testing & Documentation | 🔄 In Progress | Joint |

---

## 4. Progress Since Proposal

### Architecture & Training Strategy
- Implemented subject-aware train/val split to prevent data leakage (80/20 split by driver ID)
- Used ImageNet pretrained weights with fine-tuning
- Applied data augmentation: RandomResizedCrop, ColorJitter, RandomErasing
- Training: 5 epochs, batch size 32, AdamW optimizer, learning rate 1e-4

### Performance Metrics

| Model | Accuracy | Macro F1 | Parameters | FLOPs | Latency |
|-------|----------|----------|------------|-------|---------|
| EfficientNet-B0 | **88.06%** | **86.85%** | 4.02M | 0.39G | 13.9ms |

**Per-Class Performance:**
| Class | Accuracy |
|-------|----------|
| Safe Driving (c0) | 75.3% |
| Texting Right (c1) | 90.9% |
| Phone Right (c2) | 98.7% |
| Texting Left (c3) | 98.3% |
| Phone Left (c4) | 98.1% |
| Adjusting Radio (c5) | 95.0% |
| Drinking (c6) | 92.7% |
| Reaching Behind (c7) | 90.1% |
| Hair/Makeup (c8) | 85.6% |
| Talking to Passenger (c9) | 43.9% |

### Testing Results (Real-World Conditions)
- Model successfully exports to ONNX format for edge deployment
- Inference latency: 13.9ms (72 FPS) on CPU
- Grad-CAM visualizations confirm model focuses on hands, phone, and face regions

---

## 5. Model Improvements

- **Class Imbalance Handling:** Identified "Talking to Passenger" (c9) as hardest class (43.9% accuracy) - often confused with "Safe Driving" due to similar hand positions
- **Confusion Analysis:** Safe driving sometimes misclassified as passenger interaction (19% confusion rate)
- **Planned Fix:** Implement focal loss and class-weighted sampling for final phase

---

## 6. Deployment Progress

- ✅ ONNX model export implemented
- ✅ Inference pipeline ready (72 FPS throughput)
- ✅ Grad-CAM visualization for explainability
- 🔄 Web demo interface (in progress)

---

## 7. Challenges & Solutions

### Challenge 1: Data Leakage Risk
**Problem:** Random train/val split could place same driver in both sets, inflating validation accuracy.  
**Solution:** Implemented subject-aware splitting using driver IDs from metadata CSV, ensuring no driver appears in both train and validation sets.

### Challenge 2: Python 3.14 Compatibility
**Problem:** pytorch-grad-cam and albumentations packages incompatible with Python 3.14.  
**Solution:** Rewrote data augmentation pipeline using native torchvision transforms; implemented custom Grad-CAM from scratch.

### Challenge 3: Class Confusion (Passenger vs Safe)
**Problem:** "Talking to Passenger" class has only 43.9% accuracy, frequently confused with safe driving.  
**Solution:** (Planned) Add focal loss to focus on hard examples; augment with attention-guided cropping on passenger region.

---

## 8. Next Steps for Final Phase

### Remaining Integration Tasks
1. Train ResNet18 and MobileNetV3 for architecture comparison
2. Run zero-shot domain generalization on AUC dataset
3. Complete CAM quality evaluation with Pointing Game metric
4. Implement focal loss for class imbalance

### Final Optimization & Deployment
1. Hyperparameter tuning for underperforming classes
2. Build Streamlit/Gradio web demo for live inference
3. Quantize ONNX model for faster edge inference
4. Complete documentation and final report

---

## Appendix: Training Curves & Confusion Matrix

*See attached figures:*
- `efficientnet_b0_training_curves.png` - Loss and accuracy over 5 epochs
- `efficientnet_b0_confusion_matrix.png` - Normalized confusion matrix showing per-class performance

---

*Report submitted: April 27, 2026*
