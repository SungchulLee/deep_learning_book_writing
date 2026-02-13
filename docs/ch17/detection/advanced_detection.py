"""
Example 4: Advanced Object Detection Techniques
================================================

Demonstrates production-ready detection techniques.
"""

import torch
import numpy as np
import time
from pathlib import Path

try:
    from ultralytics import YOLO
except:
    print("Install: pip install ultralytics")
    exit()

print("="*70)
print("ADVANCED OBJECT DETECTION TECHNIQUES")
print("="*70)

# Load model
model = YOLO('yolov8n.pt')

# Technique 1: Multi-Scale Inference
print("\n1. Multi-Scale Inference")
print("-"*70)
print("Testing at multiple scales improves accuracy")

from PIL import Image
test_img = Image.new('RGB', (640, 640), (200, 200, 200))

scales = [0.8, 1.0, 1.2]
for scale in scales:
    size = int(640 * scale)
    resized = test_img.resize((size, size))
    results = model(resized, verbose=False)
    print(f"Scale {scale}: {len(results[0].boxes)} detections")

# Technique 2: Model Export
print("\n2. Model Export to ONNX")
print("-"*70)
print("Exporting model to ONNX for production deployment...")

try:
    model.export(format='onnx', dynamic=False, simplify=True)
    print("✓ Model exported to ONNX successfully")
    print("  File: yolov8n.onnx")
except Exception as e:
    print(f"Export failed: {e}")

# Technique 3: Batch Processing
print("\n3. Batch Processing")
print("-"*70)

batch_size = 4
images = [test_img] * batch_size

start = time.time()
results = model(images, verbose=False)
batch_time = time.time() - start

print(f"Processed {batch_size} images in {batch_time*1000:.2f} ms")
print(f"Per-image: {batch_time*1000/batch_size:.2f} ms")

# Technique 4: Confidence Calibration
print("\n4. Confidence Thresholds")
print("-"*70)

thresholds = [0.25, 0.5, 0.75]
for conf in thresholds:
    results = model(test_img, conf=conf, verbose=False)
    print(f"Confidence {conf}: {len(results[0].boxes)} detections")

# Summary
print("\n" + "="*70)
print("ADVANCED TECHNIQUES COMPLETE!")
print("="*70)
print("\nTechniques Covered:")
print("✓ Multi-scale inference for accuracy")
print("✓ ONNX export for deployment")
print("✓ Batch processing for efficiency")
print("✓ Confidence calibration")
print("\nProduction Ready!")
print("="*70)
