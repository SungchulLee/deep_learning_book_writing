"""
05_fashion_mnist_classifier.py
================================
Training a CNN on Fashion-MNIST Dataset

Apply the same CNN architecture to Fashion-MNIST and discover why
this dataset is more challenging than regular MNIST!

Same model, different data = different difficulty!

What you'll learn:
- How dataset complexity affects model performance
- Comparing results across datasets
- Understanding why some problems are harder
- Model transfer across similar domains

Difficulty: Intermediate
Estimated Time: 1-2 hours

Author: PyTorch CNN Tutorial
Date: November 2025
"""

import torch.nn as nn
import torch.optim as optim
import cnn_utils as utils

# =============================================================================
# SECTION 1: Configuration
# =============================================================================

print("=" * 70)
print("Training CNN on Fashion-MNIST Dataset")
print("=" * 70)

cfg = utils.parse_args()
utils.set_seed(seed=cfg.seed)

print(f"\nTraining Configuration:")
print(f"  Epochs: {cfg.epochs}")
print(f"  Learning rate: {cfg.lr}")
print(f"  Batch size: {cfg.batch_size}")
print(f"  Device: {cfg.device}")

# Fashion-MNIST class labels
FASHION_LABELS = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

# =============================================================================
# SECTION 2: Data Loading
# =============================================================================

print("\n" + "=" * 70)
print("Loading Fashion-MNIST Data")
print("=" * 70)

train_kwargs = {'batch_size': cfg.batch_size, 'shuffle': True}
test_kwargs = {'batch_size': cfg.test_batch_size, 'shuffle': False}

# Load Fashion-MNIST (note the fashion_mnist=True flag)
trainloader, testloader = utils.load_data(
    train_kwargs, test_kwargs, fashion_mnist=True
)

print(f"Data loaded!")
print(f"  Training batches: {len(trainloader)}")
print(f"  Classes: {FASHION_LABELS}")

# =============================================================================
# SECTION 3: Model Setup
# =============================================================================

print("\n" + "=" * 70)
print("Creating Model")
print("=" * 70)

# Same CNN architecture as MNIST!
model = utils.CNN().to(cfg.device)
loss_ftn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.gamma)

print("Using same CNN architecture as MNIST tutorial")
print("But expect lower accuracy - Fashion-MNIST is harder!")

# =============================================================================
# SECTION 4: Pre-Training Predictions
# =============================================================================

print("\n" + "=" * 70)
print("Before Training")
print("=" * 70)

utils.show_batch_or_ten_images_with_label_and_predict(
    testloader, model, cfg.device, classes=FASHION_LABELS, n=10
)

# =============================================================================
# SECTION 5: Training
# =============================================================================

print("\n" + "=" * 70)
print("Training Model")
print("=" * 70)

print(f"\nTraining for {cfg.epochs} epochs...")
print("Expected: 90-92% accuracy (lower than MNIST's 99%)")
print("\nWhy is it harder?")
print("  • More intra-class variation")
print("  • Some classes look very similar (T-shirt vs Shirt)")
print("  • Requires more complex feature detection")

utils.train(
    model, trainloader, loss_ftn, optimizer, scheduler,
    cfg.device, cfg.epochs, cfg.log_interval, cfg.dry_run
)

# =============================================================================
# SECTION 6: Post-Training Evaluation
# =============================================================================

print("\n" + "=" * 70)
print("After Training")
print("=" * 70)

utils.show_batch_or_ten_images_with_label_and_predict(
    testloader, model, cfg.device, classes=FASHION_LABELS, n=10
)

test_accuracy = utils.compute_accuracy(model, testloader, cfg.device)

print(f"\n{'='*70}")
print(f"Final Results")
print(f"{'='*70}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

if test_accuracy >= 92.0:
    print("🎉 Excellent! Above 92% is great for Fashion-MNIST!")
elif test_accuracy >= 90.0:
    print("✓ Very good! Above 90% is solid performance!")
elif test_accuracy >= 87.0:
    print("✓ Good! Try training longer or adjusting hyperparameters")

print("\nComparison:")
print(f"  MNIST (digits):    ~99% accuracy")
print(f"  Fashion-MNIST:     {test_accuracy:.2f}% accuracy")
print(f"  Difference:        ~{99 - test_accuracy:.0f} percentage points")

# =============================================================================
# SECTION 7: Model Saving
# =============================================================================

if cfg.save_model:
    utils.save_model(model, cfg.path)
    loaded_model = utils.load_model(utils.CNN, cfg.device, cfg.path)
    loaded_accuracy = utils.compute_accuracy(loaded_model, testloader, cfg.device)
    print(f"Loaded model accuracy: {loaded_accuracy:.2f}%")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print(f"""
✅ What We Learned:
   • Same architecture performs differently on different datasets
   • Fashion-MNIST is ~7-9% harder than digit MNIST
   • Visual similarity between classes affects accuracy
   • Model capacity needs to match problem complexity

Key Insights:
-------------
1. Dataset difficulty varies even with same format
2. Intra-class variation reduces accuracy
3. Similar classes are confusing (Shirt vs T-shirt)
4. Same CNN architecture is versatile

Most Confused Classes (typically):
---------------------------------
• T-shirt/top (0) ↔ Shirt (6)
• Pullover (2) ↔ Coat (4)
• Sneaker (7) ↔ Ankle boot (9)

Next Steps:
-----------
1. Move to CIFAR-10 for even more challenge!
2. Try data augmentation to improve accuracy
3. Experiment with deeper architectures
4. Analyze confusion matrix to see error patterns

Challenge: Can you get above 92% accuracy?
Try: More epochs, data augmentation, deeper network
""")

print("=" * 70)
print("Tutorial Complete! ✓")
print("=" * 70)
