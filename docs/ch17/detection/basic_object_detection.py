"""
Example 1: Basic Object Detection Concepts
===========================================

This script implements fundamental object detection concepts from scratch:
- Bounding box representations
- Intersection over Union (IoU)
- Non-Maximum Suppression (NMS)
- Confidence thresholding
- Visualization

Understanding these concepts is crucial before using complex architectures.

Author: PyTorch Object Detection Tutorial
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("="*70)
print("BASIC OBJECT DETECTION CONCEPTS")
print("="*70)
print("\nThis example teaches fundamental detection concepts:")
print("1. Bounding boxes and coordinate systems")
print("2. Intersection over Union (IoU)")
print("3. Non-Maximum Suppression (NMS)")
print("4. Confidence scores and thresholding\n")

# ============================================================================
# STEP 1: BOUNDING BOX REPRESENTATION
# ============================================================================
"""
A bounding box is a rectangle that encloses an object.
We'll use the format: [x_min, y_min, x_max, y_max]

where:
- (x_min, y_min) is the top-left corner
- (x_max, y_max) is the bottom-right corner
"""

def convert_bbox_format(bbox, from_format='xyxy', to_format='xywh'):
    """
    Convert bounding box between different formats.
    
    Formats:
    - 'xyxy': [x_min, y_min, x_max, y_max]
    - 'xywh': [x_min, y_min, width, height]
    - 'cxcywh': [x_center, y_center, width, height] (YOLO format)
    
    Args:
        bbox: Bounding box in source format
        from_format: Source format
        to_format: Target format
    
    Returns:
        Bounding box in target format
    """
    x1, y1, x2, y2 = bbox if from_format == 'xyxy' else [0, 0, 0, 0]
    
    if from_format == 'xywh':
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
    elif from_format == 'cxcywh':
        cx, cy, w, h = bbox
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
    
    # Convert to target format
    if to_format == 'xyxy':
        return [x1, y1, x2, y2]
    elif to_format == 'xywh':
        return [x1, y1, x2-x1, y2-y1]
    elif to_format == 'cxcywh':
        return [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]


# Example bounding boxes
print("Step 1: Bounding Box Formats")
print("-" * 70)

example_box_xyxy = [100, 150, 300, 450]  # [x_min, y_min, x_max, y_max]
print(f"Original format (xyxy): {example_box_xyxy}")
print(f"  → Top-left: ({example_box_xyxy[0]}, {example_box_xyxy[1]})")
print(f"  → Bottom-right: ({example_box_xyxy[2]}, {example_box_xyxy[3]})")

example_box_xywh = convert_bbox_format(example_box_xyxy, 'xyxy', 'xywh')
print(f"\nConverted to xywh: {example_box_xywh}")
print(f"  → Position: ({example_box_xywh[0]}, {example_box_xywh[1]})")
print(f"  → Size: {example_box_xywh[2]} × {example_box_xywh[3]}")

example_box_cxcywh = convert_bbox_format(example_box_xyxy, 'xyxy', 'cxcywh')
print(f"\nConverted to cxcywh (YOLO): {example_box_cxcywh}")
print(f"  → Center: ({example_box_cxcywh[0]}, {example_box_cxcywh[1]})")
print(f"  → Size: {example_box_cxcywh[2]} × {example_box_cxcywh[3]}\n")

# ============================================================================
# STEP 2: INTERSECTION OVER UNION (IoU)
# ============================================================================
"""
IoU measures the overlap between two bounding boxes.
It's the ratio of intersection area to union area.

IoU = Area of Overlap / Area of Union
    = Area(A ∩ B) / Area(A ∪ B)

IoU ranges from 0 (no overlap) to 1 (perfect match).
"""

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two boxes.
    
    Args:
        box1: [x_min, y_min, x_max, y_max]
        box2: [x_min, y_min, x_max, y_max]
    
    Returns:
        iou: Float between 0 and 1
    """
    # Extract coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    # The intersection rectangle's coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Calculate width and height of intersection
    # Use max(0, ...) to handle non-overlapping boxes
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    
    intersection_area = inter_width * inter_height
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    # Add small epsilon to avoid division by zero
    iou = intersection_area / (union_area + 1e-6)
    
    return iou


print("\nStep 2: Intersection over Union (IoU)")
print("-" * 70)

# Example IoU calculations
box_a = [100, 100, 200, 200]  # 100x100 box
box_b = [150, 150, 250, 250]  # 100x100 box, partially overlapping

iou = calculate_iou(box_a, box_b)
print(f"Box A: {box_a}")
print(f"Box B: {box_b}")
print(f"IoU: {iou:.4f}")

# More examples
box_c = [100, 100, 200, 200]  # Same as box_a
iou_perfect = calculate_iou(box_a, box_c)
print(f"\nPerfect match IoU: {iou_perfect:.4f} (boxes are identical)")

box_d = [300, 300, 400, 400]  # No overlap
iou_zero = calculate_iou(box_a, box_d)
print(f"No overlap IoU: {iou_zero:.4f} (boxes don't overlap)")

box_e = [120, 120, 180, 180]  # box_e inside box_a
iou_inside = calculate_iou(box_a, box_e)
print(f"One inside another IoU: {iou_inside:.4f}\n")

# ============================================================================
# STEP 3: NON-MAXIMUM SUPPRESSION (NMS)
# ============================================================================
"""
NMS removes duplicate detections of the same object.

Algorithm:
1. Sort all boxes by confidence score (highest first)
2. Pick the box with highest confidence
3. Remove all boxes with IoU > threshold with the picked box
4. Repeat until no boxes remain

This keeps the best detection and removes overlapping duplicates.
"""

def non_maximum_suppression(boxes, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove duplicate detections.
    
    Args:
        boxes: List of bounding boxes [[x_min, y_min, x_max, y_max], ...]
        scores: List of confidence scores for each box
        iou_threshold: IoU threshold for suppression (default 0.5)
    
    Returns:
        keep_indices: Indices of boxes to keep
    """
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Sort boxes by score (highest first)
    sorted_indices = np.argsort(scores)[::-1]
    
    keep_indices = []
    
    while len(sorted_indices) > 0:
        # Pick the box with highest score
        current_idx = sorted_indices[0]
        keep_indices.append(current_idx)
        
        if len(sorted_indices) == 1:
            break
        
        # Get current box
        current_box = boxes[current_idx]
        
        # Calculate IoU with all remaining boxes
        remaining_indices = sorted_indices[1:]
        remaining_boxes = boxes[remaining_indices]
        
        ious = np.array([calculate_iou(current_box, box) for box in remaining_boxes])
        
        # Keep only boxes with IoU below threshold
        keep_mask = ious < iou_threshold
        sorted_indices = remaining_indices[keep_mask]
    
    return keep_indices


print("Step 3: Non-Maximum Suppression (NMS)")
print("-" * 70)

# Example: Multiple overlapping detections of the same object
detections = [
    [100, 100, 200, 200],  # Detection 1
    [105, 105, 205, 205],  # Detection 2 (similar to 1)
    [110, 95, 210, 195],   # Detection 3 (similar to 1)
    [300, 300, 400, 400],  # Detection 4 (different object)
]

confidence_scores = [0.95, 0.88, 0.82, 0.90]

print(f"Before NMS: {len(detections)} detections")
for i, (box, score) in enumerate(zip(detections, confidence_scores)):
    print(f"  Detection {i+1}: {box}, confidence: {score:.2f}")

# Apply NMS
keep_indices = non_maximum_suppression(detections, confidence_scores, iou_threshold=0.5)

print(f"\nAfter NMS: {len(keep_indices)} detections kept")
for idx in keep_indices:
    print(f"  Detection {idx+1}: {detections[idx]}, confidence: {confidence_scores[idx]:.2f}")

print(f"\nRemoved {len(detections) - len(keep_indices)} duplicate detections\n")

# ============================================================================
# STEP 4: CONFIDENCE THRESHOLDING
# ============================================================================
"""
Not all detections are reliable. We filter out low-confidence detections
before applying NMS.

Confidence threshold controls the precision-recall trade-off:
- High threshold (e.g., 0.7): High precision, low recall (miss some objects)
- Low threshold (e.g., 0.3): Low precision, high recall (more false positives)
"""

def filter_by_confidence(boxes, scores, classes, conf_threshold=0.5):
    """
    Filter detections by confidence threshold.
    
    Args:
        boxes: List of bounding boxes
        scores: List of confidence scores
        classes: List of class labels
        conf_threshold: Minimum confidence to keep
    
    Returns:
        Filtered boxes, scores, and classes
    """
    keep_mask = np.array(scores) >= conf_threshold
    
    filtered_boxes = [box for i, box in enumerate(boxes) if keep_mask[i]]
    filtered_scores = [score for i, score in enumerate(scores) if keep_mask[i]]
    filtered_classes = [cls for i, cls in enumerate(classes) if keep_mask[i]]
    
    return filtered_boxes, filtered_scores, filtered_classes


print("Step 4: Confidence Thresholding")
print("-" * 70)

# Example detections with various confidence scores
all_detections = [
    ([100, 100, 200, 200], 0.95, 'dog'),    # High confidence
    ([150, 150, 250, 250], 0.75, 'dog'),    # Medium confidence
    ([300, 100, 400, 200], 0.45, 'cat'),    # Low confidence
    ([350, 300, 450, 400], 0.25, 'car'),    # Very low confidence
]

print("All detections:")
for box, score, cls in all_detections:
    print(f"  {cls}: confidence={score:.2f}, box={box}")

# Apply confidence threshold
conf_threshold = 0.5
boxes = [det[0] for det in all_detections]
scores = [det[1] for det in all_detections]
classes = [det[2] for det in all_detections]

filtered_boxes, filtered_scores, filtered_classes = filter_by_confidence(
    boxes, scores, classes, conf_threshold
)

print(f"\nAfter confidence threshold ({conf_threshold}):")
for box, score, cls in zip(filtered_boxes, filtered_scores, filtered_classes):
    print(f"  {cls}: confidence={score:.2f}, box={box}")

print(f"\nFiltered out {len(all_detections) - len(filtered_boxes)} low-confidence detections\n")

# ============================================================================
# STEP 5: COMPLETE DETECTION PIPELINE
# ============================================================================
"""
Putting it all together:
1. Get predictions from model (boxes, scores, classes)
2. Filter by confidence threshold
3. Apply NMS per class
4. Return final detections
"""

def detection_pipeline(boxes, scores, classes, conf_threshold=0.5, nms_threshold=0.5):
    """
    Complete object detection post-processing pipeline.
    
    Args:
        boxes: List of bounding boxes
        scores: List of confidence scores
        classes: List of class labels
        conf_threshold: Confidence threshold
        nms_threshold: NMS IoU threshold
    
    Returns:
        Final boxes, scores, and classes after filtering and NMS
    """
    # Step 1: Filter by confidence
    boxes, scores, classes = filter_by_confidence(boxes, scores, classes, conf_threshold)
    
    if len(boxes) == 0:
        return [], [], []
    
    # Step 2: Apply NMS per class
    # Group detections by class
    unique_classes = set(classes)
    final_boxes = []
    final_scores = []
    final_classes = []
    
    for cls in unique_classes:
        # Get detections for this class
        class_mask = [c == cls for c in classes]
        class_boxes = [boxes[i] for i in range(len(boxes)) if class_mask[i]]
        class_scores = [scores[i] for i in range(len(scores)) if class_mask[i]]
        
        # Apply NMS
        keep_indices = non_maximum_suppression(class_boxes, class_scores, nms_threshold)
        
        # Add to final results
        for idx in keep_indices:
            final_boxes.append(class_boxes[idx])
            final_scores.append(class_scores[idx])
            final_classes.append(cls)
    
    return final_boxes, final_scores, final_classes


print("Step 5: Complete Detection Pipeline")
print("-" * 70)

# Simulate model predictions (multiple detections per object)
raw_predictions = [
    ([100, 100, 200, 200], 0.95, 'dog'),
    ([105, 105, 205, 205], 0.88, 'dog'),    # Duplicate of dog
    ([110, 95, 210, 195], 0.82, 'dog'),     # Another duplicate
    ([300, 300, 400, 400], 0.90, 'cat'),
    ([305, 305, 405, 405], 0.75, 'cat'),    # Duplicate of cat
    ([500, 100, 600, 200], 0.45, 'car'),    # Low confidence
    ([700, 300, 800, 400], 0.30, 'person'), # Very low confidence
]

print(f"Raw model predictions: {len(raw_predictions)} detections")
for box, score, cls in raw_predictions:
    print(f"  {cls}: {score:.2f}")

# Apply pipeline
boxes = [pred[0] for pred in raw_predictions]
scores = [pred[1] for pred in raw_predictions]
classes = [pred[2] for pred in raw_predictions]

final_boxes, final_scores, final_classes = detection_pipeline(
    boxes, scores, classes,
    conf_threshold=0.5,
    nms_threshold=0.5
)

print(f"\nFinal detections: {len(final_boxes)} objects")
for box, score, cls in zip(final_boxes, final_scores, final_classes):
    print(f"  {cls}: {score:.2f} at {box}")

print(f"\nPipeline removed {len(raw_predictions) - len(final_boxes)} detections")
print(f"  - Confidence filtering: {sum(1 for s in scores if s < 0.5)} detections")
print(f"  - NMS: {len(raw_predictions) - sum(1 for s in scores if s < 0.5) - len(final_boxes)} duplicates\n")

# ============================================================================
# STEP 6: VISUALIZATION
# ============================================================================
"""
Visualizing detections is crucial for understanding and debugging.
We'll create a synthetic image with objects and draw bounding boxes.
"""

def create_synthetic_image(size=(600, 600)):
    """Create a synthetic image with colored rectangles as objects."""
    img = Image.new('RGB', size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw some "objects" (colored rectangles)
    objects = [
        ([100, 100, 200, 200], (255, 100, 100), 'red_square'),
        ([300, 300, 400, 400], (100, 100, 255), 'blue_square'),
        ([450, 100, 550, 180], (100, 255, 100), 'green_rect'),
    ]
    
    for box, color, _ in objects:
        draw.rectangle(box, fill=color, outline=(0, 0, 0), width=2)
    
    return img, objects


def visualize_detections(image, boxes, labels, scores, title="Object Detections"):
    """
    Visualize bounding boxes on image.
    
    Args:
        image: PIL Image or numpy array
        boxes: List of [x_min, y_min, x_max, y_max]
        labels: List of class labels
        scores: List of confidence scores
        title: Plot title
    """
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    
    # Define colors for different classes
    colors = {
        'red_square': 'red',
        'blue_square': 'blue',
        'green_rect': 'green',
        'dog': 'yellow',
        'cat': 'cyan',
        'car': 'magenta',
    }
    
    for box, label, score in zip(boxes, labels, scores):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        
        # Get color for this class
        color = colors.get(label, 'white')
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label with confidence
        label_text = f'{label}: {score:.2f}'
        ax.text(
            x_min, y_min - 5,
            label_text,
            color='white',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7)
        )
    
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('basic_detection_results.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved as 'basic_detection_results.png'")
    plt.close()


print("Step 6: Visualization")
print("-" * 70)

# Create synthetic image
image, ground_truth = create_synthetic_image()

# Simulate detections (with some duplicates and false positives)
detections = [
    ([100, 100, 200, 200], 0.95, 'red_square'),
    ([105, 105, 205, 205], 0.85, 'red_square'),  # Duplicate
    ([300, 300, 400, 400], 0.92, 'blue_square'),
    ([450, 100, 550, 180], 0.88, 'green_rect'),
    ([200, 400, 280, 500], 0.35, 'red_square'),  # False positive
]

print("Creating visualization...")
boxes = [det[0] for det in detections]
scores = [det[1] for det in detections]
classes = [det[2] for det in detections]

# Apply detection pipeline
final_boxes, final_scores, final_classes = detection_pipeline(
    boxes, scores, classes,
    conf_threshold=0.5,
    nms_threshold=0.5
)

visualize_detections(image, final_boxes, final_classes, final_scores,
                    title="Object Detection Results (After NMS)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("BASIC OBJECT DETECTION CONCEPTS - COMPLETE!")
print("="*70)
print("\nKey Takeaways:")
print("1. Bounding Box: 4 numbers define object location")
print("   - Multiple formats: xyxy, xywh, cxcywh")
print("2. IoU: Measures overlap between boxes (0-1 range)")
print("   - Used for evaluation and NMS")
print("3. NMS: Removes duplicate detections")
print("   - Keeps highest confidence, removes high IoU overlaps")
print("4. Confidence: Filters unreliable detections")
print("   - Trade-off between precision and recall")
print("5. Pipeline: Confidence filter → NMS → Final detections")
print("\nCore Metrics:")
print(f"  - Typical confidence threshold: 0.5")
print(f"  - Typical NMS IoU threshold: 0.5")
print(f"  - IoU > 0.5 considered 'good' detection")
print("\nYou now understand the foundations of object detection!")
print("Next: Example 2 - Learn YOLO architecture")
print("="*70)
