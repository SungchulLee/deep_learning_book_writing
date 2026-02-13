# Example 1: Basic Object Detection Concepts

## 🎯 Learning Objectives

By completing this example, you will learn:
- What a bounding box is and how to represent it
- Intersection over Union (IoU) calculation
- Non-Maximum Suppression (NMS) algorithm
- Confidence scores and thresholding
- Basic anchor box concepts
- Difference between object detection and classification

## 📋 Overview

This example introduces object detection fundamentals by implementing core algorithms from scratch. We'll build understanding using simple synthetic data before moving to complex architectures.

**Learning Approach:**
- Build components from scratch (no black boxes!)
- Understand IoU deeply
- Implement NMS step-by-step
- Visualize everything

## 🎨 Bounding Box Representations

### Format 1: (x, y, width, height)
```
(x, y) = Top-left corner
width = Box width
height = Box height

Example: [100, 150, 200, 300]
┌────────────────┐
│ (100,150)      │
│  ┌─────────┐   │
│  │  Dog    │   │ height=300
│  │         │   │
│  └─────────┘   │
│    width=200   │
└────────────────┘
```

### Format 2: (x_min, y_min, x_max, y_max)
```
(x_min, y_min) = Top-left corner
(x_max, y_max) = Bottom-right corner

Example: [100, 150, 300, 450]
```

### Format 3: (x_center, y_center, width, height) - YOLO Format
```
(x_center, y_center) = Center of box
width, height = Dimensions

Used by YOLO and many modern detectors
```

## 🔍 Intersection over Union (IoU)

IoU measures how much two boxes overlap:

```
         Box A          Box B       Intersection    Union
        ┌──────┐      ┌──────┐      ┌────┐       ┌────────┐
        │      │      │      │      │    │       │        │
        │  ┌───┼──────┼───┐  │  →   │████│   /   │        │
        │  │   │      │   │  │      │████│       │        │
        └──┼───┘      └───┼──┘      └────┘       └────────┘
           └──────────────┘

IoU = Area(A ∩ B) / Area(A ∪ B) = 0.42
```

**IoU Interpretation:**
- IoU = 0.0: No overlap
- IoU = 0.5: Decent overlap (common threshold)
- IoU = 0.75: Good overlap (strict threshold)
- IoU = 1.0: Perfect match

**Why IoU?**
- Measures both position and size accuracy
- Scale-invariant
- Range [0, 1] - easy to interpret
- Standard metric in object detection

## 🎯 Non-Maximum Suppression (NMS)

NMS removes duplicate detections of the same object:

```
Problem: Multiple overlapping predictions
┌─────┐
│ 0.9 │ ← High confidence
└─────┘
  ┌─────┐
  │ 0.7 │ ← Medium confidence
  └─────┘
    ┌─────┐
    │ 0.6 │ ← Lower confidence
    └─────┘

Solution: Keep only the best one
┌─────┐
│ 0.9 │ ← Keep this!
└─────┘
```

**NMS Algorithm:**
```
1. Sort all detections by confidence (highest first)
2. Pick detection with highest confidence
3. Remove all detections with IoU > threshold (e.g., 0.5)
4. Repeat until no detections remain
```

**Example:**
```
Detections: [
    (box1, confidence=0.9),
    (box2, confidence=0.8),  ← overlaps with box1 (IoU=0.7)
    (box3, confidence=0.7),  ← overlaps with box1 (IoU=0.6)
    (box4, confidence=0.6),  ← no overlap with box1
]

After NMS with threshold=0.5:
[box1, box4]  ← Keep non-overlapping detections
```

## 📊 Confidence Scores

Confidence = Objectness × Classification Probability

```
Objectness: Is there an object? (0-1)
Classification: What class is it? (0-1 per class)

Example:
- Dog detected with 0.95 objectness
- Dog class probability: 0.92
- Confidence = 0.95 × 0.92 = 0.87 (87%)
```

## 🎯 Detection Pipeline

```
Input Image
    ↓
Feature Extraction (CNN)
    ↓
Detection Heads
    ↓
[Multiple Predictions]
├─ Dog: 0.95 @ [100, 100, 200, 200]
├─ Dog: 0.87 @ [105, 98, 198, 202]  ← Duplicate
├─ Cat: 0.82 @ [300, 150, 100, 150]
└─ Car: 0.78 @ [400, 200, 250, 180]
    ↓
Confidence Thresholding (e.g., > 0.5)
    ↓
Non-Maximum Suppression
    ↓
[Final Detections]
├─ Dog: 0.95 @ [100, 100, 200, 200]
└─ Cat: 0.82 @ [300, 150, 100, 150]
```

## 💻 Running the Code

```bash
python basic_object_detection.py
```

**Expected Runtime:** 2-5 minutes
**What it does:**
- Generates synthetic images with objects
- Implements IoU from scratch
- Implements NMS from scratch
- Visualizes detection results

## 📊 Expected Output

You should see:
- Visualization of bounding boxes
- IoU calculation examples
- NMS removing duplicate detections
- Confidence filtering in action

## 🔧 Key Functions Implemented

### 1. `calculate_iou(box1, box2)`
Calculates IoU between two boxes

### 2. `non_maximum_suppression(boxes, scores, iou_threshold)`
Filters overlapping boxes

### 3. `draw_boxes(image, boxes, labels, scores)`
Visualizes detections

### 4. `generate_synthetic_detections()`
Creates sample data for learning

## 🎓 Key Takeaways

1. **Bounding Boxes**: 4 numbers define a rectangle
2. **IoU**: Measures overlap quality (0 = no overlap, 1 = perfect)
3. **NMS**: Removes duplicate detections
4. **Confidence**: How sure the model is
5. **Threshold**: Controls precision vs recall trade-off

## 🤔 Questions to Think About

1. What IoU threshold should you use for NMS?
2. What happens if confidence threshold is too high? Too low?
3. Why do we need NMS? Can't the model learn to output one box?
4. How does IoU handle different sized objects?

## 🚀 Next Steps

After understanding this example:
- Experiment with different IoU thresholds
- Try different confidence thresholds
- Visualize IoU for various box pairs
- Move on to Example 2 for YOLO!

## 💡 Extension Ideas

- Implement different IoU variants (GIoU, DIoU, CIoU)
- Add Soft-NMS (weighted suppression)
- Visualize NMS process step-by-step
- Compare different bounding box formats
- Implement bounding box format conversions

## 📚 Mathematical Details

### IoU Formula
```
IoU = Intersection / Union

where:
Intersection = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) 
             × max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

Union = Area(box1) + Area(box2) - Intersection
```

### NMS Complexity
```
Time: O(n²) where n = number of boxes
Can be optimized with spatial hashing
```

## ⚠️ Common Pitfalls

1. **Wrong coordinate format**: Mixing (x,y,w,h) with (x1,y1,x2,y2)
2. **IoU threshold too strict**: Missing valid detections
3. **IoU threshold too loose**: Keeping duplicates
4. **Confidence too high**: Low recall (missing objects)
5. **Confidence too low**: Low precision (false positives)

## 🎯 Real-World Considerations

- **Speed**: NMS can be slow with many boxes
- **Occlusion**: Overlapping objects are challenging
- **Scale**: Objects at different scales need different handling
- **Classes**: NMS typically applied per-class

These concepts form the foundation for all modern object detectors!
