"""
Example 2: YOLO Object Detection
=================================

This script demonstrates using pre-trained YOLOv8 for object detection.
YOLO (You Only Look Once) is a fast, accurate single-stage detector
perfect for real-time applications.

Key Concepts:
- Loading pre-trained YOLO models
- Running inference on images
- Understanding YOLO predictions
- Visualization and interpretation
- Model comparison (different sizes)

Author: PyTorch Object Detection Tutorial
Date: 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import time
import os

# Try to import ultralytics (YOLOv8)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠ ultralytics not installed. Installing...")
    print("Run: pip install ultralytics")
    YOLO_AVAILABLE = False

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

print("="*70)
print("YOLO OBJECT DETECTION")
print("="*70)
print("\nThis example demonstrates:")
print("1. Loading pre-trained YOLOv8 models")
print("2. Running object detection on images")
print("3. Understanding YOLO outputs")
print("4. Comparing different model sizes")
print("5. Real-time performance analysis\n")

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cpu':
    print("⚠ GPU not available. Inference will be slower.\n")
else:
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}\n")

# ============================================================================
# STEP 1: CREATE SYNTHETIC TEST IMAGES
# ============================================================================
"""
For demonstration, we'll create synthetic images with simple objects.
In practice, you would use real images.
"""

def create_test_image(size=(640, 640), num_objects=3):
    """Create a synthetic image for testing."""
    img = Image.new('RGB', size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Predefined object types
    object_types = [
        {'shape': 'rectangle', 'color': (255, 100, 100), 'label': 'red_box'},
        {'shape': 'ellipse', 'color': (100, 100, 255), 'label': 'blue_circle'},
        {'shape': 'rectangle', 'color': (100, 255, 100), 'label': 'green_box'},
    ]
    
    objects_drawn = []
    
    for i in range(min(num_objects, len(object_types))):
        obj_type = object_types[i]
        
        # Random position and size
        x = np.random.randint(50, size[0] - 150)
        y = np.random.randint(50, size[1] - 150)
        w = np.random.randint(80, 150)
        h = np.random.randint(80, 150)
        
        bbox = [x, y, x + w, y + h]
        
        if obj_type['shape'] == 'rectangle':
            draw.rectangle(bbox, fill=obj_type['color'], outline=(0, 0, 0), width=3)
        else:
            draw.ellipse(bbox, fill=obj_type['color'], outline=(0, 0, 0), width=3)
        
        objects_drawn.append({
            'bbox': bbox,
            'label': obj_type['label'],
            'shape': obj_type['shape']
        })
    
    return img, objects_drawn


print("Step 1: Creating Test Images")
print("-" * 70)

# Create multiple test images
test_images = []
for i in range(3):
    img, objects = create_test_image()
    test_images.append((img, objects))
    print(f"Created test image {i+1} with {len(objects)} objects")

print()

# ============================================================================
# STEP 2: LOAD PRE-TRAINED YOLO MODEL
# ============================================================================
"""
YOLOv8 comes in different sizes:
- yolov8n (nano): Fastest, smallest (3.2M params)
- yolov8s (small): Balanced (11.2M params)
- yolov8m (medium): Good accuracy (25.9M params)
- yolov8l (large): Better accuracy (43.7M params)
- yolov8x (extra large): Best accuracy (68.2M params)

We'll start with yolov8n for speed.
"""

if not YOLO_AVAILABLE:
    print("Skipping YOLO demo - ultralytics not installed")
    print("Install with: pip install ultralytics")
    exit()

print("Step 2: Loading Pre-trained YOLOv8 Model")
print("-" * 70)

# Load YOLOv8 nano model (will download on first run)
print("Loading YOLOv8n (nano) model...")
print("(First run will download ~6MB model weights)")

model = YOLO('yolov8n.pt')  # Automatically downloads if not present

print(f"✓ Model loaded successfully")
print(f"Model type: {type(model)}")
print(f"Model device: {next(model.model.parameters()).device}")

# Move to GPU if available
if device == 'cuda':
    model.to(device)
    print(f"✓ Model moved to GPU\n")
else:
    print()

# ============================================================================
# STEP 3: UNDERSTAND YOLO PREDICTIONS
# ============================================================================
"""
YOLO outputs for each detection:
- Bounding box: [x_min, y_min, x_max, y_max]
- Confidence: Float between 0 and 1
- Class: Integer class ID
- Class name: String label

Results object contains:
- boxes: All bounding boxes
- names: Class name mapping (dict)
- conf: Confidence scores
- cls: Class indices
"""

print("Step 3: Running Detection and Understanding Output")
print("-" * 70)

# Run detection on first test image
test_img = test_images[0][0]

print("Running inference...")
start_time = time.time()
results = model(test_img, verbose=False)
inference_time = (time.time() - start_time) * 1000

print(f"Inference time: {inference_time:.2f} ms")
print(f"\nResults type: {type(results)}")
print(f"Number of result objects: {len(results)}")

# Extract first result
result = results[0]

print(f"\nDetections found: {len(result.boxes)}")
if len(result.boxes) > 0:
    print("\nDetailed output structure:")
    print(f"  result.boxes: {type(result.boxes)}")
    print(f"  result.boxes.data: Shape {result.boxes.data.shape}")
    print(f"  Each detection: [x1, y1, x2, y2, confidence, class_id]")
    
    # Show first detection details
    if len(result.boxes) > 0:
        first_det = result.boxes[0]
        print(f"\nFirst detection:")
        print(f"  Box coordinates: {first_det.xyxy[0].cpu().numpy()}")
        print(f"  Confidence: {first_det.conf[0].cpu().numpy():.3f}")
        print(f"  Class ID: {int(first_det.cls[0].cpu().numpy())}")
        print(f"  Class name: {result.names[int(first_det.cls[0].cpu().numpy())]}")

print()

# ============================================================================
# STEP 4: COCO CLASSES OVERVIEW
# ============================================================================
"""
YOLOv8 is pre-trained on COCO dataset with 80 classes.
Let's see what classes it can detect.
"""

print("Step 4: COCO Dataset Classes")
print("-" * 70)

# Get class names
class_names = result.names
print(f"Total classes: {len(class_names)}")
print(f"\nSample of COCO classes:")

# Show some common classes
common_classes = [0, 1, 2, 3, 15, 16, 17, 18]  # person, bicycle, car, etc.
for class_id in common_classes:
    print(f"  ID {class_id:2d}: {class_names[class_id]}")

print(f"\nFull class list:")
# Print all classes in columns
class_list = [f"{i:2d}:{name}" for i, name in class_names.items()]
for i in range(0, len(class_list), 4):
    print("  " + "  ".join(class_list[i:i+4]))

print()

# ============================================================================
# STEP 5: DETECTION WITH CONFIDENCE THRESHOLDING
# ============================================================================
"""
We can control detection behavior with parameters:
- conf: Confidence threshold (default 0.25)
- iou: IoU threshold for NMS (default 0.45)
- classes: Specific classes to detect
"""

print("Step 5: Detection with Different Thresholds")
print("-" * 70)

# Test with different confidence thresholds
thresholds = [0.25, 0.5, 0.7]

for conf_thresh in thresholds:
    results = model(test_img, conf=conf_thresh, verbose=False)
    num_detections = len(results[0].boxes)
    print(f"Confidence threshold {conf_thresh:.2f}: {num_detections} detections")

print()

# ============================================================================
# STEP 6: VISUALIZE DETECTIONS
# ============================================================================
"""
YOLO provides built-in visualization, but we'll also create custom
visualizations to better understand the results.
"""

def visualize_yolo_results(image, result, title="YOLO Detections", save_path=None):
    """
    Visualize YOLO detection results.
    
    Args:
        image: PIL Image or numpy array
        result: YOLO result object
        title: Plot title
        save_path: Path to save visualization
    """
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)
    
    # Get detections
    boxes = result.boxes
    
    # Define colors for different classes
    colors = plt.cm.tab20(np.linspace(0, 1, len(result.names)))
    
    for box in boxes:
        # Extract box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        class_name = result.names[cls]
        
        # Get color for this class
        color = colors[cls % len(colors)]
        
        # Draw rectangle
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label_text = f'{class_name}: {conf:.2f}'
        ax.text(
            x1, y1 - 5,
            label_text,
            color='white',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7)
        )
    
    ax.set_title(f"{title} ({len(boxes)} detections)", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.close()


print("Step 6: Creating Visualizations")
print("-" * 70)

# Detect and visualize
results = model(test_img, conf=0.25, verbose=False)
visualize_yolo_results(
    test_img,
    results[0],
    title="YOLOv8 Detections",
    save_path="yolo_detections.png"
)

print()

# ============================================================================
# STEP 7: COMPARE MODEL SIZES
# ============================================================================
"""
Compare different YOLOv8 model sizes:
- Speed (inference time)
- Accuracy (detected objects)
- Model size (parameters)
"""

print("Step 7: Comparing YOLO Model Sizes")
print("-" * 70)

# Available models (comment out to speed up)
model_sizes = ['yolov8n.pt']  # Just nano for demo
# Uncomment to compare multiple:
# model_sizes = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']

comparison_results = []

for model_name in model_sizes:
    print(f"\nTesting {model_name}...")
    
    # Load model
    test_model = YOLO(model_name)
    if device == 'cuda':
        test_model.to(device)
    
    # Warm-up run
    _ = test_model(test_img, verbose=False)
    
    # Timed runs
    times = []
    for _ in range(5):
        start = time.time()
        results = test_model(test_img, verbose=False)
        times.append((time.time() - start) * 1000)
    
    avg_time = np.mean(times)
    num_detections = len(results[0].boxes)
    
    comparison_results.append({
        'model': model_name,
        'time_ms': avg_time,
        'detections': num_detections
    })
    
    print(f"  Average inference time: {avg_time:.2f} ms")
    print(f"  Detections: {num_detections}")

print("\nComparison Summary:")
print("-" * 70)
print(f"{'Model':<15} {'Time (ms)':<15} {'Detections':<15}")
print("-" * 70)
for res in comparison_results:
    print(f"{res['model']:<15} {res['time_ms']:<15.2f} {res['detections']:<15}")

print()

# ============================================================================
# STEP 8: BATCH PROCESSING
# ============================================================================
"""
YOLO can process multiple images efficiently in batches.
"""

print("Step 8: Batch Processing Multiple Images")
print("-" * 70)

# Process all test images
batch_images = [img for img, _ in test_images]

print(f"Processing {len(batch_images)} images in batch...")
start_time = time.time()
batch_results = model(batch_images, verbose=False)
batch_time = (time.time() - start_time) * 1000

print(f"Batch inference time: {batch_time:.2f} ms")
print(f"Average per image: {batch_time / len(batch_images):.2f} ms")

for i, result in enumerate(batch_results):
    print(f"  Image {i+1}: {len(result.boxes)} detections")

print()

# ============================================================================
# STEP 9: ADVANCED FEATURES
# ============================================================================
"""
YOLOv8 supports various advanced features:
- Class filtering
- Region of Interest (ROI)
- Custom confidence/IoU thresholds
- Image augmentation during inference
"""

print("Step 9: Advanced YOLO Features")
print("-" * 70)

# Feature 1: Detect specific classes only
print("Feature 1: Detect specific classes only")
print("  Detecting only persons, cars, and dogs (classes 0, 2, 15)...")
results = model(test_img, classes=[0, 2, 15], verbose=False)
print(f"  Detections: {len(results[0].boxes)}")

# Feature 2: Custom thresholds
print("\nFeature 2: Custom confidence and IoU thresholds")
results_strict = model(test_img, conf=0.7, iou=0.3, verbose=False)
print(f"  High confidence (0.7): {len(results_strict[0].boxes)} detections")

# Feature 3: Get prediction details
print("\nFeature 3: Accessing detailed predictions")
results = model(test_img, verbose=False)
if len(results[0].boxes) > 0:
    print("  Available attributes:")
    print(f"    - boxes.xyxy: Bounding boxes (x1, y1, x2, y2)")
    print(f"    - boxes.xywh: Bounding boxes (x_center, y_center, width, height)")
    print(f"    - boxes.conf: Confidence scores")
    print(f"    - boxes.cls: Class indices")
    print(f"    - names: Class name mapping")

print()

# ============================================================================
# STEP 10: PERFORMANCE ANALYSIS
# ============================================================================
"""
Analyze YOLO performance characteristics.
"""

print("Step 10: Performance Analysis")
print("-" * 70)

# Measure FPS
num_runs = 10
times = []

for i in range(num_runs):
    start = time.time()
    _ = model(test_img, verbose=False)
    times.append(time.time() - start)

avg_time = np.mean(times) * 1000
std_time = np.std(times) * 1000
fps = 1000 / avg_time

print(f"Performance metrics ({num_runs} runs):")
print(f"  Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
print(f"  Estimated FPS: {fps:.1f}")
print(f"  Throughput: {fps * 3600:.0f} images/hour")

print(f"\nModel Information:")
print(f"  Device: {device}")
print(f"  Model: YOLOv8n")
print(f"  Input size: 640×640")

print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*70)
print("YOLO OBJECT DETECTION - COMPLETE!")
print("="*70)
print("\nKey Takeaways:")
print("1. YOLO is a fast single-stage detector")
print("   - One forward pass for all detections")
print("   - Grid-based predictions")
print("2. Pre-trained on COCO (80 classes)")
print("   - Ready to use out-of-the-box")
print("   - Can detect common objects")
print("3. Multiple model sizes available")
print("   - Nano (fastest) to Extra Large (most accurate)")
print("   - Trade-off between speed and accuracy")
print("4. Easy-to-use API")
print("   - Simple inference: model(image)")
print("   - Built-in visualization")
print("   - Configurable thresholds")
print("5. Real-time capable")
print(f"   - {fps:.1f} FPS on {device}")
print("   - Suitable for video processing")
print("\nYOLO Advantages:")
print("  ✓ Fast inference (real-time)")
print("  ✓ High accuracy")
print("  ✓ Easy to use")
print("  ✓ Active community")
print("  ✓ Pre-trained models")
print("\nNext Steps:")
print("  - Try on your own images")
print("  - Test different model sizes")
print("  - Experiment with thresholds")
print("  - Move to Example 3 for custom training!")
print("="*70)
