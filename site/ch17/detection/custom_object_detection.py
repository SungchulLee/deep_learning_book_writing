"""
Example 3: Custom Object Detection Training
============================================

Train YOLO on custom objects with synthetic data.
Demonstrates complete training pipeline.
"""

import os
import yaml
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

try:
    from ultralytics import YOLO
except:
    print("Install ultralytics: pip install ultralytics")
    exit()

print("="*70)
print("CUSTOM OBJECT DETECTION TRAINING")
print("="*70)

# Create synthetic dataset
def create_custom_dataset(root='custom_dataset', num_images=100):
    """Create synthetic dataset in YOLO format."""
    
    # Create directories
    for split in ['train', 'val']:
        Path(f'{root}/images/{split}').mkdir(parents=True, exist_ok=True)
        Path(f'{root}/labels/{split}').mkdir(parents=True, exist_ok=True)
    
    classes = ['circle', 'square', 'triangle']
    
    for split, count in [('train', num_images), ('val', num_images//5)]:
        for i in range(count):
            # Create image
            img = Image.new('RGB', (640, 640), (240, 240, 240))
            draw = ImageDraw.Draw(img)
            
            # Generate objects
            labels = []
            for _ in range(np.random.randint(1, 4)):
                cls = np.random.randint(0, len(classes))
                x = np.random.randint(100, 540)
                y = np.random.randint(100, 540)
                s = np.random.randint(40, 80)
                
                if cls == 0:  # circle
                    draw.ellipse([x-s, y-s, x+s, y+s], fill=(255,100,100))
                elif cls == 1:  # square
                    draw.rectangle([x-s, y-s, x+s, y+s], fill=(100,255,100))
                else:  # triangle
                    draw.polygon([(x, y-s), (x-s, y+s), (x+s, y+s)], fill=(100,100,255))
                
                # YOLO format: class x_center y_center width height (normalized)
                x_center, y_center = x/640, y/640
                width, height = (2*s)/640, (2*s)/640
                labels.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Save image and labels
            img.save(f'{root}/images/{split}/img_{i:04d}.jpg')
            with open(f'{root}/labels/{split}/img_{i:04d}.txt', 'w') as f:
                f.write('\n'.join(labels))
    
    # Create data.yaml
    data_yaml = {
        'path': os.path.abspath(root),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }
    
    with open(f'{root}/data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
    
    return f'{root}/data.yaml'

# Create dataset
print("Creating synthetic custom dataset...")
data_yaml = create_custom_dataset()
print(f"Dataset created: {data_yaml}")
print("  - 100 training images")
print("  - 20 validation images")
print("  - 3 classes: circle, square, triangle\n")

# Load model
print("Loading YOLOv8n model...")
model = YOLO('yolov8n.pt')

# Train
print("\nTraining model...")
print("(This will take a few minutes)")
results = model.train(
    data=data_yaml,
    epochs=10,
    imgsz=640,
    batch=16,
    name='custom_detection',
    verbose=False
)

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nModel saved to: runs/detect/custom_detection/weights/best.pt")
print("\nTo use trained model:")
print("  model = YOLO('runs/detect/custom_detection/weights/best.pt')")
print("  results = model('image.jpg')")
print("="*70)
