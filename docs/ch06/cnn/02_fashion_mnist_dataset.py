"""
02_fashion_mnist_dataset.py
============================
Fashion-MNIST Dataset Visualization

Fashion-MNIST is a modern replacement for MNIST that's more challenging!
Instead of handwritten digits, it contains grayscale images of clothing items.

Why Fashion-MNIST?
- Same format as MNIST (28x28 grayscale)
- More challenging and realistic
- Better for testing model generalization
- Used in modern research and education

What you'll learn:
- Working with different datasets (same format)
- Understanding class labels for non-digit data
- Comparing dataset difficulty
- Handling categorical data with meaningful names

Difficulty: Easy
Estimated Time: 30 minutes

Author: PyTorch CNN Tutorial
Date: November 2025
"""

import matplotlib.pyplot as plt
import cnn_utils as utils

# =============================================================================
# SECTION 1: Fashion-MNIST Class Labels
# =============================================================================

print("=" * 70)
print("Fashion-MNIST Dataset Exploration")
print("=" * 70)

# Define human-readable labels for Fashion-MNIST classes
# These map from numerical labels (0-9) to clothing item names
FASHION_MNIST_LABELS = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

print("\nFashion-MNIST Classes:")
print("-" * 40)
for idx, name in FASHION_MNIST_LABELS.items():
    print(f"  Class {idx}: {name}")

print("\nNote: These are grayscale images of clothing items,")
print("not the full-color photographs you might expect!")

# =============================================================================
# SECTION 2: Configuration and Data Loading
# =============================================================================

print("\n" + "=" * 70)
print("Loading Fashion-MNIST Dataset")
print("=" * 70)

# Parse configuration
cfg = utils.parse_args()

print("\nConfiguration:")
print(f"  Batch size: {cfg.batch_size}")
print(f"  Device: {cfg.device}")

# Set seed for reproducibility
utils.set_seed(seed=cfg.seed)

# Configure data loaders
train_kwargs = {'batch_size': cfg.batch_size, 'shuffle': True}
test_kwargs = {'batch_size': cfg.test_batch_size, 'shuffle': False}

# Load Fashion-MNIST dataset
# Key difference: fashion_mnist=True flag
trainloader, testloader = utils.load_data(
    train_kwargs, 
    test_kwargs, 
    fashion_mnist=True  # This loads Fashion-MNIST instead of MNIST
)

print("\nDataset loaded successfully!")
print(f"  Training batches: {len(trainloader)}")
print(f"  Test batches: {len(testloader)}")

# Get sample batch
sample_images, sample_labels = next(iter(trainloader))
print(f"\nBatch shape:")
print(f"  Images: {sample_images.shape}")
print(f"  Labels: {sample_labels.shape}")

# =============================================================================
# SECTION 3: Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Visualizing Fashion-MNIST Images")
print("=" * 70)

# Create 8x8 grid for 64 images
fig, axes = plt.subplots(8, 8, figsize=(12, 12))
fig.suptitle('Fashion-MNIST Clothing Items Sample', fontsize=16)

# Display first batch
for images, labels in trainloader:
    for ax, image, label in zip(axes.reshape(-1), images, labels):
        # Prepare image for display
        img_display = image.squeeze().cpu().numpy()
        img_display = img_display / 2 + 0.5  # Denormalize
        
        # Display with class name
        ax.imshow(img_display, cmap="gray")
        ax.axis("off")
        
        # Use descriptive label instead of just number
        class_name = FASHION_MNIST_LABELS[label.item()]
        ax.set_title(class_name, fontsize=8)
    
    break  # Only show first batch

plt.tight_layout()
print("\nDisplaying 64 sample images...")
print("Notice how some items look similar (e.g., Shirt vs T-shirt)")
print("This makes the classification task more challenging!")
plt.show()

# =============================================================================
# SECTION 4: Comparing with MNIST
# =============================================================================

print("\n" + "=" * 70)
print("Fashion-MNIST vs Regular MNIST")
print("=" * 70)

print("""
Similarities:
✓ Same image format: 28×28 grayscale
✓ Same number of classes: 10
✓ Same dataset size: 60,000 train + 10,000 test
✓ Same pixel value range: [0, 255] → normalized to [-1, 1]

Differences:
✗ Content: Clothing items vs handwritten digits
✗ Complexity: More intra-class variation
✗ Inter-class similarity: Some classes look very similar
✗ Difficulty: Generally lower accuracy than MNIST

Why Fashion-MNIST is harder:
1. Shirts vs T-shirts are visually similar
2. Items can have different orientations
3. Textures and patterns vary within classes
4. Edge features are less distinctive

Typical Accuracies:
- MNIST: 99%+ with simple CNN
- Fashion-MNIST: 90-92% with same architecture
""")

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
    name = FASHION_MNIST_LABELS[class_id]
    print(f"{class_id:<12} | {name:<15} | {count:<10} | {percentage:.2f}%")

print(f"\nTotal samples: {total}")
print("The dataset is balanced - each class has equal representation!")

# =============================================================================
# SECTION 6: Visual Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Visual Characteristics Analysis")
print("=" * 70)

# Get samples from each class for comparison
print("\nClass characteristics:")
print("-" * 50)

characteristics = {
    0: "Simple shapes, clear boundaries",
    1: "Vertical rectangular shape",
    2: "Similar to T-shirt but often thicker",
    3: "Longer than shirts, flowing shape",
    4: "Heavy garments, varied textures",
    5: "Open-toe footwear, minimal coverage",
    6: "Very similar to T-shirts!",
    7: "Closed-toe footwear, athletic",
    8: "Varied shapes, often has handles",
    9: "Footwear covering ankles"
}

for class_id, desc in characteristics.items():
    name = FASHION_MNIST_LABELS[class_id]
    print(f"{name:15s} (Class {class_id}): {desc}")

print("\nMost Confusing Pairs:")
print("  • T-shirt/top (0) vs Shirt (6) - Very similar!")
print("  • Pullover (2) vs Coat (4) - Both upper garments")
print("  • Sneaker (7) vs Ankle boot (9) - Both closed footwear")

# =============================================================================
# SUMMARY AND KEY TAKEAWAYS
# =============================================================================

print("\n" + "=" * 70)
print("Summary - What We Learned")
print("=" * 70)

print("""
✅ Fashion-MNIST Overview:
   - Modern MNIST replacement for realistic challenges
   - 10 classes of clothing items
   - Same format as MNIST (28×28 grayscale)
   - Balanced dataset with 6,000 samples per class

✅ Key Insights:
   - More challenging than digit recognition
   - Classes have overlapping visual features
   - Good benchmark for testing model robustness
   - Requires more sophisticated features to distinguish

✅ PyTorch Skills:
   - Loading different datasets with same API
   - Using categorical labels effectively
   - Understanding dataset characteristics

✅ Comparison with MNIST:
   - Same format, different difficulty
   - Expect 7-9% lower accuracy
   - Better reflection of real-world challenges

Next Steps:
-----------
1. Continue to 03_cifar10_dataset.py for color images
2. Then train a CNN on this dataset: 05_fashion_mnist_classifier.py
3. Compare results with MNIST to understand difficulty

Challenge Yourself:
-------------------
• Which classes do you think will be hardest to distinguish?
• Can you predict the confusion matrix before training?
• Think about what features a CNN needs to learn

Fun Fact:
---------
Fashion-MNIST was created by Zalando Research to provide
a more challenging, modern dataset for the ML community!
""")

print("=" * 70)
print("Tutorial Complete! ✓")
print("=" * 70)
