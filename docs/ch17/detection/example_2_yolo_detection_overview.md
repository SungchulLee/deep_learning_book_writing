# Example 2: YOLO Object Detection

## 🎯 Learning Objectives

By completing this example, you will learn:
- YOLO (You Only Look Once) architecture overview
- Using pre-trained YOLOv8 models
- Understanding grid-based detection
- Anchor boxes and their role
- Transfer learning for object detection
- Working with COCO dataset format
- Real-time detection capabilities

## 📋 Overview

This example introduces **YOLO** - one of the most popular object detection architectures. YOLO is known for its speed and accuracy, making it ideal for real-time applications.

**Why YOLO?**
- ⚡ **Fast**: 30-100+ FPS (real-time capable)
- 🎯 **Accurate**: State-of-the-art performance
- 🔧 **Easy to use**: Simple API
- 🚀 **Well-maintained**: Active community
- 📦 **Pre-trained models**: Ready to use

## 🏗️ YOLO Architecture

### The "You Only Look Once" Principle

Unlike two-stage detectors (Faster R-CNN), YOLO makes predictions in a single forward pass:

```
Input Image (640×640)
      ↓
┌─────────────────────────┐
│   Backbone Network      │
│   (CSPDarknet/ResNet)   │
│   Feature Extraction    │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│   Neck                  │
│   (FPN + PAN)           │
│   Multi-scale Features  │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│   Detection Head        │
│   Predict for each cell │
│   - Bounding boxes      │
│   - Class probabilities │
│   - Objectness scores   │
└──────────┬──────────────┘
           ↓
   Detections Output
   [x, y, w, h, conf, class]
```

### Grid-Based Detection

YOLO divides the image into an S×S grid:

```
Image divided into 13×13 grid (for example):

┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─●─┼─┼─┼─┼─┼─┼─┼─┤  ← Dog center in this cell
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤     This cell predicts dog
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘

Each cell predicts:
- B bounding boxes (typically 3)
- Confidence for each box
- Class probabilities (C classes)
```

**Each grid cell predicts:**
- Bounding box coordinates (x, y, w, h)
- Objectness score: P(object) × IoU
- Class probabilities: P(class | object)

### Anchor Boxes

Anchor boxes are pre-defined box shapes:

```
Small Anchor    Medium Anchor    Large Anchor
┌───┐           ┌─────┐          ┌───────┐
│   │           │     │          │       │
└───┘           │     │          │       │
                └─────┘          │       │
                                 └───────┘

YOLO predicts offsets from these anchors:
Predicted Box = Anchor Box + Offsets
```

**Why Anchor Boxes?**
- Handle objects of different aspect ratios
- Faster convergence during training
- Better handling of multiple objects
- Learned from training data (k-means clustering)

## 🎯 YOLOv8 Improvements

YOLOv8 (latest version as of 2024) includes:

1. **Anchor-Free Detection**: No pre-defined anchors needed
2. **Improved Backbone**: Better feature extraction
3. **Multi-Scale Features**: FPN + PAN for different object sizes
4. **Better Loss Functions**: CIoU loss for bounding boxes
5. **Mosaic Augmentation**: Combines 4 images for training

```
YOLOv8 variants:
- YOLOv8n (nano): Fastest, smallest
- YOLOv8s (small): Balanced
- YOLOv8m (medium): Good accuracy
- YOLOv8l (large): Better accuracy
- YOLOv8x (extra large): Best accuracy
```

## 💻 Running the Code

```bash
python yolo_detection.py
```

**Expected Runtime:** 
- First run: 5-10 minutes (downloads pre-trained weights ~50MB)
- Subsequent runs: 1-2 minutes

**What it does:**
- Loads pre-trained YOLOv8 model
- Runs detection on sample images
- Visualizes results with bounding boxes
- Shows confidence scores and class labels

## 📊 Expected Results

You should see:
- Detection of 80 COCO classes (person, car, dog, etc.)
- High confidence scores (>0.5) for clear objects
- Fast inference time (<50ms per image on GPU)
- Accurate bounding boxes

## 🔧 COCO Dataset Classes

YOLO is pre-trained on COCO dataset (80 classes):

**People & Animals:**
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles:**
traffic light, fire hydrant, stop sign, parking meter, bench

**Indoor:**
chair, couch, potted plant, bed, dining table, toilet, tv, laptop,
mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink,
refrigerator, book, clock, vase, scissors, teddy bear, hair drier,
toothbrush

**Food:**
banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza,
donut, cake

**Sports:**
frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove,
skateboard, surfboard, tennis racket

**Accessories:**
bottle, wine glass, cup, fork, knife, spoon, bowl, backpack, umbrella,
handbag, tie, suitcase

## 🎓 Key Concepts

### 1. Single-Stage Detection
```
Image → CNN → Predictions (in one step)

vs Two-Stage:
Image → Region Proposals → Classification (two steps)
```

### 2. Grid Responsibility
Each grid cell is responsible for detecting objects whose center falls in that cell.

### 3. Multiple Predictions Per Cell
Each cell can predict multiple boxes (3-5 typically) to handle:
- Multiple objects in same cell
- Different object sizes/shapes

### 4. Objectness Score
```
Objectness = Confidence that cell contains an object
Final Score = Objectness × Class Probability
```

### 5. Loss Function Components
```
Total Loss = Box Loss + Object Loss + Class Loss

Box Loss: How accurate are coordinates?
Object Loss: Is there an object or not?
Class Loss: What class is the object?
```

## 🚀 Advanced Features

### Transfer Learning
Use pre-trained YOLO and fine-tune on your data:
```python
model = YOLO('yolov8n.pt')  # Load pre-trained
model.train(data='custom.yaml', epochs=100)  # Fine-tune
```

### Different Model Sizes
Trade-off between speed and accuracy:
```python
model = YOLO('yolov8n.pt')  # Nano: fastest
model = YOLO('yolov8s.pt')  # Small
model = YOLO('yolov8m.pt')  # Medium
model = YOLO('yolov8l.pt')  # Large
model = YOLO('yolov8x.pt')  # Extra: most accurate
```

### Inference Options
```python
# Confidence threshold
results = model(image, conf=0.5)

# IoU threshold for NMS
results = model(image, iou=0.45)

# Specific classes only
results = model(image, classes=[0, 2, 3])  # person, car, motorcycle
```

## 📊 Performance Benchmarks

**COCO val2017 (80 classes):**

| Model | mAP@50:95 | FPS (V100) | Params | Size |
|-------|-----------|------------|--------|------|
| YOLOv8n | 37.3% | 238 | 3.2M | 6MB |
| YOLOv8s | 44.9% | 217 | 11.2M | 22MB |
| YOLOv8m | 50.2% | 139 | 25.9M | 52MB |
| YOLOv8l | 52.9% | 110 | 43.7M | 88MB |
| YOLOv8x | 53.9% | 73 | 68.2M | 136MB |

## 🤔 Questions to Think About

1. Why is YOLO called "You Only Look Once"?
2. What's the advantage of grid-based detection?
3. How do anchor boxes help detection?
4. Why multiple predictions per grid cell?
5. What's the trade-off between YOLO and Faster R-CNN?

## 🚀 Next Steps

After understanding this example:
- Try different YOLOv8 model sizes
- Test on your own images
- Experiment with confidence thresholds
- Compare speed vs accuracy trade-offs
- Move to Example 3 for custom training!

## 💡 Extension Ideas

- Batch processing multiple images
- Video object detection
- Real-time webcam detection
- Counting specific objects
- Tracking objects across frames
- Export to ONNX for production

## 📚 YOLO Evolution

- **YOLOv1 (2015)**: Original paper, grid-based detection
- **YOLOv2 (2016)**: Batch normalization, anchor boxes
- **YOLOv3 (2018)**: Multi-scale predictions, better backbone
- **YOLOv4 (2020)**: CSPDarknet, advanced augmentation
- **YOLOv5 (2020)**: PyTorch implementation, easy to use
- **YOLOv6 (2022)**: Industrial applications focus
- **YOLOv7 (2022)**: New training strategies
- **YOLOv8 (2023)**: Anchor-free, best performance

## ⚠️ Common Issues

1. **Low FPS**: Use smaller model (YOLOv8n), lower resolution
2. **False Positives**: Increase confidence threshold
3. **Missed Objects**: Decrease confidence threshold, use larger model
4. **Wrong Classes**: Model trained on COCO only (80 classes)
5. **Poor on Custom Objects**: Need fine-tuning (Example 3)

## 🔗 Useful Resources

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [YOLO Papers](https://arxiv.org/abs/1506.02640)
- [COCO Dataset](https://cocodataset.org/)
- [Model Zoo](https://github.com/ultralytics/ultralytics)
