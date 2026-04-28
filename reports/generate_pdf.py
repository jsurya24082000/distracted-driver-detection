"""Generate PDF report using fpdf2."""
from fpdf import FPDF
from pathlib import Path

# Paths
REPORT_DIR = Path(__file__).parent
PDF_FILE = REPORT_DIR / "26_phase1.pdf"
RESULTS_DIR = REPORT_DIR.parent / "outputs" / "results"

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(26, 82, 118)
        self.cell(0, 6, 'Group 26 - Phase 1 Report: Distracted Driver Detection', align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(2)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(40, 116, 166)
        self.cell(0, 6, title, new_x='LMARGIN', new_y='NEXT')
        self.ln(1)
    
    def body_text(self, text):
        self.set_font('Helvetica', '', 9)
        self.set_text_color(51, 51, 51)
        self.multi_cell(0, 4.5, text)
        self.ln(1)
    
    def bullet_point(self, text):
        self.set_font('Helvetica', '', 9)
        self.set_text_color(51, 51, 51)
        self.cell(5, 4.5, '-')
        x = self.get_x()
        self.multi_cell(0, 4.5, text)
        self.set_x(10)

# Create PDF
pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Title
pdf.set_font('Helvetica', 'B', 14)
pdf.set_text_color(26, 82, 118)
pdf.cell(0, 8, 'Distracted Driver Detection using Deep Learning', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(3)

# Group Info
pdf.set_font('Helvetica', '', 9)
pdf.set_text_color(51, 51, 51)
pdf.cell(0, 5, 'Group Number: 26', new_x='LMARGIN', new_y='NEXT')
pdf.cell(0, 5, 'Members: Jayasurya Jayadevan (121306067), Karthikaa Mikkilineni (20602214)', new_x='LMARGIN', new_y='NEXT')
pdf.ln(3)

# Section 1
pdf.section_title('1. Revised Problem Statement')
pdf.body_text('Distracted driving causes over 3,000 fatalities annually in the US. This project develops a real-time distracted driver detection system using deep learning to classify 10 driver behaviors (safe driving, texting, phone usage, drinking, reaching, makeup, talking to passengers) from dashboard camera images. We compare three CNN architectures (EfficientNet-B0, ResNet18, MobileNetV3) optimized for edge deployment, with novel contributions in zero-shot domain generalization analysis and quantitative CAM quality evaluation using the Pointing Game metric.')

# Section 2
pdf.section_title('2. Project Adjustments & Team Progress')
pdf.body_text('Added Novel Contribution 1: Zero-shot domain generalization analysis comparing State Farm and AUC datasets. Added Novel Contribution 2: CAM quality evaluation using Pointing Game metric. Project Completion: ~85%')
pdf.ln(1)

# Section 3
pdf.section_title('3. Progress Since Proposal')
pdf.body_text('Architecture: Implemented subject-aware train/val split to prevent data leakage. Used ImageNet pretrained EfficientNet-B0 with fine-tuning. Applied augmentation: RandomResizedCrop, ColorJitter, RandomErasing. Training: 5 epochs, batch 32, AdamW, lr=1e-4.')
pdf.ln(1)

# Performance Table
pdf.set_font('Helvetica', 'B', 9)
pdf.cell(0, 5, 'Performance Metrics:', new_x='LMARGIN', new_y='NEXT')
pdf.set_font('Helvetica', '', 8)

# Table header
pdf.set_fill_color(40, 116, 166)
pdf.set_text_color(255, 255, 255)
pdf.cell(35, 5, 'Model', border=1, fill=True, align='C')
pdf.cell(25, 5, 'Accuracy', border=1, fill=True, align='C')
pdf.cell(25, 5, 'Macro F1', border=1, fill=True, align='C')
pdf.cell(25, 5, 'Params', border=1, fill=True, align='C')
pdf.cell(25, 5, 'Latency', border=1, fill=True, align='C')
pdf.ln()

# Table row
pdf.set_text_color(51, 51, 51)
pdf.cell(35, 5, 'EfficientNet-B0', border=1, align='C')
pdf.cell(25, 5, '88.06%', border=1, align='C')
pdf.cell(25, 5, '86.85%', border=1, align='C')
pdf.cell(25, 5, '4.02M', border=1, align='C')
pdf.cell(25, 5, '13.9ms', border=1, align='C')
pdf.ln(3)

# Per-class table
pdf.set_font('Helvetica', 'B', 9)
pdf.cell(0, 5, 'Per-Class Accuracy:', new_x='LMARGIN', new_y='NEXT')
pdf.set_font('Helvetica', '', 8)

classes = [('Safe', '75.3%'), ('Text R', '90.9%'), ('Phone R', '98.7%'), ('Text L', '98.3%'), ('Phone L', '98.1%')]
classes2 = [('Radio', '95.0%'), ('Drink', '92.7%'), ('Reach', '90.1%'), ('Makeup', '85.6%'), ('Passenger', '43.9%')]

for cls, acc in classes:
    pdf.cell(19, 4, cls, border=1, align='C')
    pdf.cell(15, 4, acc, border=1, align='C')
pdf.ln()
for cls, acc in classes2:
    pdf.cell(19, 4, cls, border=1, align='C')
    pdf.cell(15, 4, acc, border=1, align='C')
pdf.ln(3)

# Section 4
pdf.section_title('4. Model Improvements & Deployment')
pdf.body_text('Identified "Talking to Passenger" as hardest class (43.9%) - confused with Safe Driving. Planned fix: focal loss and class-weighted sampling. Deployment: ONNX export complete, 72 FPS throughput, Grad-CAM visualization ready.')

# Section 5
pdf.section_title('5. Challenges & Solutions')
pdf.bullet_point('Data Leakage Risk: Random split could place same driver in both sets. Solution: Subject-aware splitting using driver IDs.')
pdf.bullet_point('Python 3.14 Compatibility: pytorch-grad-cam incompatible. Solution: Rewrote using native torchvision transforms.')
pdf.bullet_point('Class Confusion: Passenger vs Safe has 19% confusion. Solution: Planned focal loss for hard examples.')
pdf.ln(1)

# Section 6
pdf.section_title('6. Next Steps for Final Phase')
pdf.bullet_point('Train ResNet18 and MobileNetV3 for architecture comparison')
pdf.bullet_point('Run zero-shot domain generalization on AUC dataset')
pdf.bullet_point('Complete CAM quality evaluation with Pointing Game metric')
pdf.bullet_point('Build Streamlit web demo for live inference')
pdf.bullet_point('Quantize ONNX model for faster edge deployment')

# Page 2 - Figures
pdf.add_page()
pdf.section_title('Appendix: Training Results')

# Add images
curves_path = str(RESULTS_DIR / 'efficientnet_b0_training_curves.png')
confusion_path = str(RESULTS_DIR / 'efficientnet_b0_confusion_matrix.png')

pdf.set_font('Helvetica', 'B', 9)
pdf.cell(0, 5, 'Training Curves (Loss & Accuracy over 5 epochs):', new_x='LMARGIN', new_y='NEXT')
pdf.image(curves_path, x=10, w=190)
pdf.ln(5)

pdf.cell(0, 5, 'Confusion Matrix (Normalized):', new_x='LMARGIN', new_y='NEXT')
pdf.image(confusion_path, x=30, w=150)

# Save
pdf.output(PDF_FILE)
print(f"PDF generated: {PDF_FILE}")
