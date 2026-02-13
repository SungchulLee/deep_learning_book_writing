"""
01_mnist_dataset.py
===================
MNIST Dataset Visualization

This is your first CNN tutorial! We'll explore the famous MNIST dataset
of handwritten digits. This dataset is perfect for learning because:
- Simple grayscale images (28x28 pixels)
- Clear, well-defined classes (digits 0-9)
- Small enough to train quickly
- Large enough to learn meaningful patterns

What you'll learn:
- Loading datasets with PyTorch
- Understanding data shapes and structure
- Visualizing image data
- Working with DataLoaders and batches

Difficulty: Easy
Estimated Time: 30 minutes

Author: PyTorch CNN Tutorial
Date: November 2025
"""

import matplotlib.pyplot as plt
import cnn_utils as utils

# =============================================================================
# SECTION 1: Configuration and Setup
# =============================================================================

print("=" * 70)
print("MNIST Dataset Exploration")
print("=" * 70)

# Parse command line arguments
# This loads default configuration (batch size, seed, etc.)
cfg = utils.parse_args()

print("\nConfiguration:")
print(f"  Learning rate: {cfg.lr}")
print(f"  Batch size: {cfg.batch_size}")
print(f"  Test batch size: {cfg.test_batch_size}")
print(f"  Random seed: {cfg.seed}")
print(f"  Device: {cfg.device}")

# Set random seed for reproducibility
# This ensures we get the same results each time we run
utils.set_seed(seed=cfg.seed)
print(f"\nRandom seed set to {cfg.seed} for reproducibility")

# =============================================================================
# SECTION 2: Data Loading
# =============================================================================

print("\n" + "=" * 70)
print("Loading MNIST Dataset")
print("=" * 70)

# Configure DataLoader parameters
train_kwargs = {'batch_size': cfg.batch_size, 'shuffle': True}
test_kwargs = {'batch_size': cfg.test_batch_size, 'shuffle': False}

# Load MNIST dataset
# - Training set: 60,000 images
# - Test set: 10,000 images
# - Each image: 28x28 grayscale (1 channel)
# - Classes: 10 (digits 0-9)
trainloader, testloader = utils.load_data(train_kwargs, test_kwargs)

print("\nDataset loaded successfully!")
print(f"  Training batches: {len(trainloader)}")
print(f"  Test batches: {len(testloader)}")
print(f"  Images per training batch: {cfg.batch_size}")
print(f"  Total training images: {len(trainloader) * cfg.batch_size}")

# Get a sample batch to inspect
sample_images, sample_labels = next(iter(trainloader))
print(f"\nSample batch shape:")
print(f"  Images: {sample_images.shape}")  # (batch_size, channels, height, width)
print(f"  Labels: {sample_labels.shape}")  # (batch_size,)
print(f"\nImage tensor details:")
print(f"  - Batch size: {sample_images.shape[0]}")
print(f"  - Channels: {sample_images.shape[1]} (grayscale)")
print(f"  - Height: {sample_images.shape[2]} pixels")
print(f"  - Width: {sample_images.shape[3]} pixels")

# =============================================================================
# SECTION 3: Data Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Visualizing MNIST Images")
print("=" * 70)

# Create an 8x8 grid to display 64 images
fig, axes = plt.subplots(8, 8, figsize=(10, 10))
fig.suptitle('MNIST Handwritten Digits Sample', fontsize=16)

# Get one batch of images
for images, labels in trainloader:
    # Iterate through the grid and fill with images
    for ax, image, label in zip(axes.reshape(-1), images, labels):
        # Convert from (C, H, W) to (H, W, C) for matplotlib
        # For grayscale, we just squeeze out the channel dimension
        img_display = image.squeeze().cpu().numpy()
        
        # Denormalize from [-1, 1] to [0, 1] for better visualization
        img_display = img_display / 2 + 0.5
        
        # Display the image
        ax.imshow(img_display, cmap="gray")
        ax.axis("off")
        ax.set_title(f"{label.item()}", fontsize=10)
    
    # Only process first batch
    break

plt.tight_layout()
print("\nDisplaying 64 sample images...")
print("Each image shows a handwritten digit (0-9)")
print("Close the plot window to continue.")
plt.show()

# =============================================================================
# SECTION 4: Data Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Dataset Statistics")
print("=" * 70)

# Analyze label distribution in training set
label_counts = [0] * 10
for _, labels in trainloader:
    for label in labels:
        label_counts[label.item()] += 1

print("\nClass distribution in training set:")
print(f"{'Digit':<10} | {'Count':<10} | {'Percentage':<10}")
print("-" * 35)
total = sum(label_counts)
for digit, count in enumerate(label_counts):
    percentage = 100.0 * count / total
    print(f"{digit:<10} | {count:<10} | {percentage:.2f}%")

print(f"\nTotal training samples: {total}")
print("Note: The dataset is well-balanced across all digits!")

# =============================================================================
# SECTION 5: Understanding Pixel Values
# =============================================================================

print("\n" + "=" * 70)
print("Understanding Pixel Values")
print("=" * 70)

# Get a single image to analyze
sample_image = sample_images[0].squeeze().cpu().numpy()

print("\nSample image statistics:")
print(f"  Shape: {sample_image.shape}")
print(f"  Data type: {sample_image.dtype}")
print(f"  Min value: {sample_image.min():.4f}")
print(f"  Max value: {sample_image.max():.4f}")
print(f"  Mean value: {sample_image.mean():.4f}")
print(f"  Std deviation: {sample_image.std():.4f}")

print("\nPixel value interpretation:")
print("  - Images are normalized to range [-1, 1]")
print("  - -1.0 represents black pixels")
print("  - +1.0 represents white pixels")
print("  - Values in between represent shades of gray")
print("  - This normalization helps neural networks train better!")

# =============================================================================
# SUMMARY AND KEY TAKEAWAYS
# =============================================================================

print("\n" + "=" * 70)
print("Summary - What We Learned")
print("=" * 70)

print("""
✅ Dataset Basics:
   - MNIST contains 60,000 training + 10,000 test images
   - Each image is 28x28 pixels, grayscale (1 channel)
   - 10 classes representing digits 0-9
   - Dataset is well-balanced across classes

✅ Data Structure:
   - Images shape: (batch_size, 1, 28, 28)
   - Labels shape: (batch_size,)
   - Pixels normalized to [-1, 1] range

✅ PyTorch DataLoader:
   - Automatically batches data
   - Can shuffle training data
   - Handles data loading efficiently

✅ Next Steps:
   - Move to 02_fashion_mnist_dataset.py to see a different dataset
   - Then progress to 04_mnist_classifier.py to train a CNN!
   - Try modifying batch_size to see how it affects data loading

Experiment Ideas:
-----------------
1. Change batch_size (e.g., 32, 128, 256) and see how it affects batches
2. Set shuffle=False in train_kwargs to see ordered data
3. Modify the grid size to display more/fewer images
4. Calculate and plot the pixel intensity histogram

Ready to train a model on this data? → 04_mnist_classifier.py
""")

print("=" * 70)
print("Tutorial Complete! ✓")
print("=" * 70)
