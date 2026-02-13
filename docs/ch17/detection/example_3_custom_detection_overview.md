# Example 3: Custom Object Detection

## 🎯 Learning Objectives

- Train YOLO on custom objects
- Data annotation and format conversion
- Creating custom datasets
- Transfer learning for detection
- Model evaluation (mAP, precision, recall)
- Handling class imbalance

## 📋 Overview

Learn to train object detectors on your own custom objects. This example covers the complete pipeline from data preparation to model evaluation.

**Pipeline:**
```
Collect Images → Annotate → Format Data → Train → Evaluate → Deploy
```

## 🗂️ Dataset Structure

YOLO expects this format:
```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       └── img3.jpg
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── img2.txt
    └── val/
        └── img3.txt
```

**Label Format (YOLO):**
```
class_id x_center y_center width height

Example:
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2

All values normalized to [0, 1]
```

## 💻 Running

```bash
python custom_object_detection.py
```

**What it does:**
- Creates synthetic custom dataset
- Trains YOLOv8 from scratch
- Evaluates with mAP
- Saves trained model

## 📊 Evaluation Metrics

**mAP (mean Average Precision):**
- mAP@50: IoU threshold 0.5
- mAP@50:95: Average across IoU 0.5 to 0.95
- **The standard metric for object detection**

**Precision & Recall:**
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

## 🎓 Training Tips

1. **Data Quality > Quantity**: 100 good annotations > 1000 poor ones
2. **Balance Classes**: Each class needs sufficient examples
3. **Augmentation**: YOLO has built-in augmentation
4. **Start Small**: Test with small model first
5. **Monitor mAP**: Best metric for detection

## 🚀 Next Steps

- Apply to your own objects
- Try different model sizes
- Experiment with hyperparameters
- Move to Example 4 for advanced techniques!
