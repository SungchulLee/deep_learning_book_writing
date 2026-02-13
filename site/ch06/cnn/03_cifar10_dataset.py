"""
03_cifar10_dataset.py
=====================
CIFAR-10 Dataset Visualization

CIFAR-10 introduces us to color images and real-world object recognition!
This dataset is significantly more challenging than MNIST variants.

CIFAR-10 (Canadian Institute For Advanced Research):
- 60,000 32×32 RGB (color) images
- 10 object classes from the real world
- Natural images with backgrounds
- Variation in pose, lighting, and scale

What you'll learn:
- Working with RGB (3-channel) images
- Handling real-world image variation
- Understanding dataset difficulty scaling
- Normalization for color images

Difficulty: Easy
Estimated Time: 30 minutes

Author: PyTorch CNN Tutorial
Date: November 2025
"""

import matplotlib.pyplot as plt
import cnn_utils as utils

# =============================================================================
# SECTION 1: CIFAR-10 Class Labels
# =============================================================================

print("=" * 70)
print("CIFAR-10 Dataset Exploration")
print("=" * 70)

# Define human-readable labels for CIFAR-10 classes
# These represent common objects and animals
CIFAR10_LABELS = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

print("\nCIFAR-10 Classes:")
print("-" * 40)
for idx, name in CIFAR10_LABELS.items():
    print(f"  Class {idx}: {name}")

print("\nNote: Unlike MNIST, these are REAL photographs!")
print("You'll see:")
print("  • Natural backgrounds")
print("  • Different poses and angles")
print("  • Varying lighting conditions")
print("  • Scale variation")

# =============================================================================
# SECTION 2: Configuration and Data Loading
# =============================================================================

print("\n" + "=" * 70)
print("Loading CIFAR-10 Dataset")
print("=" * 70)

# Parse configuration
cfg = utils.parse_args()

print("\nConfiguration:")
print(f"  Batch size: {cfg.batch_size}")
print(f"  Device: {cfg.device}")

# Set seed
utils.set_seed(seed=cfg.seed)

# Configure data loaders
train_kwargs = {'batch_size': cfg.batch_size, 'shuffle': True}
test_kwargs = {'batch_size': cfg.test_batch_size, 'shuffle': False}

# Load CIFAR-10 dataset
# Key difference: cifar10=True flag
trainloader, testloader = utils.load_data(
    train_kwargs,
    test_kwargs,
    cifar10=True  # This loads CIFAR-10 instead of MNIST
)

print("\nDataset loaded successfully!")
print(f"  Training batches: {len(trainloader)}")
print(f"  Test batches: {len(testloader)}")

# Get sample batch
sample_images, sample_labels = next(iter(trainloader))
print(f"\nBatch shape:")
print(f"  Images: {sample_images.shape}")
print(f"  Labels: {sample_labels.shape}")

print(f"\nImage details:")
print(f"  - Channels: {sample_images.shape[1]} (RGB color)")
print(f"  - Height: {sample_images.shape[2]} pixels")
print(f"  - Width: {sample_images.shape[3]} pixels")
print(f"  - Total pixels per image: {32 * 32 * 3} = 3,072")

# =============================================================================
# SECTION 3: Understanding Color Images
# =============================================================================

print("\n" + "=" * 70)
print("Understanding RGB Color Images")
print("=" * 70)

print("""
RGB Color Representation:
------------------------
• R = Red channel (0-255 or normalized)
• G = Green channel (0-255 or normalized)
• B = Blue channel (0-255 or normalized)

Image tensor structure:
- Shape: (batch_size, 3, height, width)
- Channel 0: Red intensities
- Channel 1: Green intensities
- Channel 2: Blue intensities

Example RGB values:
• Pure Red:    (255, 0, 0)   or (1.0, -1.0, -1.0) normalized
• Pure Green:  (0, 255, 0)   or (-1.0, 1.0, -1.0) normalized
• Pure Blue:   (0, 0, 255)   or (-1.0, -1.0, 1.0) normalized
• White:       (255, 255, 255) or (1.0, 1.0, 1.0) normalized
• Black:       (0, 0, 0)     or (-1.0, -1.0, -1.0) normalized

Our normalization:
- Original: [0, 255] per channel
- After transform: [-1, 1] per channel
- Mean = 0.5, Std = 0.5 for each channel
""")

# Analyze a sample image
sample_img = sample_images[0]
print(f"\nSample image statistics:")
print(f"  Shape: {sample_img.shape}")
for ch, color in enumerate(['Red', 'Green', 'Blue']):
    channel_data = sample_img[ch].cpu().numpy()
    print(f"  {color} channel:")
    print(f"    Min: {channel_data.min():.3f}")
    print(f"    Max: {channel_data.max():.3f}")
    print(f"    Mean: {channel_data.mean():.3f}")

# =============================================================================
# SECTION 4: Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Visualizing CIFAR-10 Images")
print("=" * 70)

# Create 8x8 grid for 64 images
fig, axes = plt.subplots(8, 8, figsize=(12, 12))
fig.suptitle('CIFAR-10 Natural Images Sample', fontsize=16)

# Display first batch
for images, labels in trainloader:
    for ax, image, label in zip(axes.reshape(-1), images, labels):
        # Prepare image for display
        # Convert from (C, H, W) to (H, W, C) for matplotlib
        img_display = image.permute(1, 2, 0).cpu().numpy()
        
        # Denormalize from [-1, 1] to [0, 1]
        img_display = img_display / 2 + 0.5
        
        # Clip to valid range (in case of numerical issues)
        img_display = img_display.clip(0, 1)
        
        # Display with class name
        ax.imshow(img_display)
        ax.axis("off")
        
        class_name = CIFAR10_LABELS[label.item()]
        ax.set_title(class_name, fontsize=8)
    
    break  # Only show first batch

plt.tight_layout()
print("\nDisplaying 64 sample images...")
print("Notice:")
print("  • Images are small (32×32) but contain complex objects")
print("  • Natural backgrounds and varied conditions")
print("  • Some images are blurry or partially occluded")
print("  • Objects can be in different poses and scales")
plt.show()

# =============================================================================
# SECTION 5: Dataset Statistics
# =============================================================================

print("\n" + "=" * 70)
print("Dataset Statistics")
print("=" * 70)

# Count samples per class
class_counts = [0] * 10

for _, labels in trainloader:
    for label in labels:
        class_counts[label.item()] += 1

print("\nClass distribution:")
print(f"{'Class ID':<12} | {'Name':<15} | {'Count':<10} | {'Percentage':<10}")
print("-" * 55)

total = sum(class_counts)
for class_id, count in enumerate(class_counts):
    percentage = 100.0 * count / total
    name = CIFAR10_LABELS[class_id]
    print(f"{class_id:<12} | {name:<15} | {count:<10} | {percentage:.2f}%")

print(f"\nTotal samples: {total}")
print("Dataset is perfectly balanced: 5,000 training images per class!")

# =============================================================================
# SECTION 6: Comparison with Previous Datasets
# =============================================================================

print("\n" + "=" * 70)
print("Comparing CIFAR-10 with MNIST Variants")
print("=" * 70)

comparison = """
┌─────────────────┬──────────────┬───────────────┬──────────────┐
│ Feature         │ MNIST        │ Fashion-MNIST │ CIFAR-10     │
├─────────────────┼──────────────┼───────────────┼──────────────┤
│ Image Size      │ 28×28        │ 28×28         │ 32×32        │
│ Channels        │ 1 (gray)     │ 1 (gray)      │ 3 (RGB)      │
│ Total Pixels    │ 784          │ 784           │ 3,072        │
│ Num Classes     │ 10           │ 10            │ 10           │
│ Train Samples   │ 60,000       │ 60,000        │ 50,000       │
│ Test Samples    │ 10,000       │ 10,000        │ 10,000       │
│ Content         │ Digits       │ Clothing      │ Objects      │
│ Complexity      │ Low          │ Medium        │ High         │
│ Typical Acc.    │ 99%+         │ 90-92%        │ 75-85%       │
└─────────────────┴──────────────┴───────────────┴──────────────┘

Why CIFAR-10 is harder:
1. ✗ Color adds complexity (3× more data per pixel)
2. ✗ Natural images with backgrounds
3. ✗ High intra-class variation (many different cats, dogs, etc.)
4. ✗ Small image size (32×32) means less detail
5. ✗ Varied lighting, poses, and occlusions
6. ✗ Some classes are visually similar (cat vs dog, truck vs automobile)

Required model changes:
• Must handle 3 input channels instead of 1
• Needs more parameters to capture complexity
• Benefits from deeper architectures
• Requires more training epochs
• Often needs data augmentation
"""

print(comparison)

# =============================================================================
# SECTION 7: Challenging Aspects
# =============================================================================

print("\n" + "=" * 70)
print("What Makes CIFAR-10 Challenging?")
print("=" * 70)

challenges = {
    "Low Resolution": "32×32 is tiny! Many details are lost.",
    "Class Similarity": "Cats vs dogs, trucks vs automobiles - subtle differences",
    "Backgrounds": "Objects aren't centered or isolated like MNIST",
    "Pose Variation": "Animals/objects can face any direction",
    "Occlusion": "Objects might be partially hidden",
    "Lighting": "Images taken in different lighting conditions",
    "Scale": "Objects can be large or small in the frame"
}

for challenge, explanation in challenges.items():
    print(f"  • {challenge:20s}: {explanation}")

print("\nThis is why CIFAR-10 is a great dataset:")
print("  ✓ Tests real-world model performance")
print("  ✓ Requires sophisticated architectures")
print("  ✓ Good stepping stone to ImageNet")
print("  ✓ Computational requirements still reasonable")

# =============================================================================
# SUMMARY AND KEY TAKEAWAYS
# =============================================================================

print("\n" + "=" * 70)
print("Summary - What We Learned")
print("=" * 70)

print("""
✅ CIFAR-10 Overview:
   - 60,000 32×32 RGB images of real-world objects
   - 10 classes: animals, vehicles, etc.
   - Significantly more challenging than MNIST variants
   - Requires deeper, more sophisticated CNNs

✅ Color Images:
   - 3 channels (RGB) instead of 1 (grayscale)
   - Each pixel is a 3D vector of color intensities
   - Normalized independently per channel
   - 3× more parameters needed in first layer

✅ Dataset Characteristics:
   - Natural photographs with backgrounds
   - High intra-class variation
   - Low inter-class distinction for some pairs
   - Small image size limits recognizable detail

✅ PyTorch Skills:
   - Loading RGB datasets
   - Handling multi-channel image tensors
   - Channel-wise normalization
   - Proper visualization of color images

Expected Performance:
--------------------
• Simple CNN: 60-70% accuracy
• Deeper CNN: 75-80% accuracy
• State-of-the-art: 95%+ accuracy
• Human performance: ~94% accuracy

Next Steps:
-----------
1. Train on MNIST first: 04_mnist_classifier.py
2. Progress to Fashion-MNIST: 05_fashion_mnist_classifier.py
3. Then tackle CIFAR-10: 06_cifar10_basic.py
4. Advanced CIFAR-10: 07_cifar10_advanced.py

Key Insight:
------------
CIFAR-10's difficulty comes from being REAL photographs
with all the messiness of the real world - perfect for
learning how to build robust computer vision models!

Fun Facts:
----------
• Created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
• Subset of the 80 million tiny images dataset
• Widely used benchmark since 2009
• ImageNet is the "big brother" of CIFAR-10
""")

print("=" * 70)
print("Tutorial Complete! ✓")
print("=" * 70)
