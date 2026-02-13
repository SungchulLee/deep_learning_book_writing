"""
07_cifar10_advanced.py
======================
Advanced CNN for CIFAR-10 with Better Performance

This tutorial implements a deeper, more sophisticated CNN architecture
that achieves significantly better accuracy on CIFAR-10.

What you'll learn:
- Deeper CNN architectures
- Impact of network depth on performance
- Advanced training techniques
- Achieving competitive results on CIFAR-10

Difficulty: Challenging
Estimated Time: 2 hours

Author: PyTorch CNN Tutorial
Date: November 2025
"""

import torch.nn as nn
import torch.optim as optim
import cnn_utils as utils

# =============================================================================
# Setup
# =============================================================================

print("=" * 70)
print("Advanced CNN for CIFAR-10")
print("=" * 70)

cfg = utils.parse_args()
utils.set_seed(seed=cfg.seed)

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

# =============================================================================
# Data Loading
# =============================================================================

print("\nLoading CIFAR-10...")
train_kwargs = {'batch_size': cfg.batch_size, 'shuffle': True}
test_kwargs = {'batch_size': cfg.test_batch_size, 'shuffle': False}
trainloader, testloader = utils.load_data(train_kwargs, test_kwargs, cifar10=True)

# =============================================================================
# Model Setup
# =============================================================================

print("\nCreating Advanced CNN...")
model = utils.CNN_CIFAR10().to(cfg.device)

print("Advanced architecture features:")
print("  • Deeper network (4 conv layers vs 2)")
print("  • More filters (up to 64)")
print("  • Larger FC layer (512 neurons)")
print("  • Better suited for complex images")

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

loss_ftn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.gamma)

# =============================================================================
# Before Training
# =============================================================================

print("\nBefore training:")
utils.show_batch_or_ten_images_with_label_and_predict(
    testloader, model, cfg.device, classes=CIFAR10_CLASSES, n=10, cifar10=True
)

# =============================================================================
# Training
# =============================================================================

print(f"\nTraining for {cfg.epochs} epochs...")
print("Expected: 75-80% accuracy with proper training")
print("(Much better than the basic model's 60-70%!)\n")

utils.train(
    model, trainloader, loss_ftn, optimizer, scheduler,
    cfg.device, cfg.epochs, cfg.log_interval, cfg.dry_run
)

# =============================================================================
# After Training
# =============================================================================

print("\nAfter training:")
utils.show_batch_or_ten_images_with_label_and_predict(
    testloader, model, cfg.device, classes=CIFAR10_CLASSES, n=10, cifar10=True
)

test_accuracy = utils.compute_accuracy(model, testloader, cfg.device)

# =============================================================================
# Save Model
# =============================================================================

if cfg.save_model:
    utils.save_model(model, cfg.path)
    print(f"Model saved successfully!")

# =============================================================================
# Summary
# =============================================================================

print(f"\n{'='*70}")
print("Final Results")
print(f"{'='*70}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Total Parameters: {total_params:,}")

print("\nPerformance Comparison:")
print("  Simple CNN (06):  ~65-70%")
print(f"  Advanced CNN (07): {test_accuracy:.2f}%")
print(f"  Improvement:       ~{test_accuracy - 67:.0f} percentage points!")

if test_accuracy >= 80:
    print("\n🎉 Excellent! Above 80% is great!")
elif test_accuracy >= 75:
    print("\n✓ Very good! Above 75% is solid!")
elif test_accuracy >= 70:
    print("\n✓ Good! Try more epochs or data augmentation")

print(f"""

Key Takeaways:
--------------
✅ Deeper networks can learn more complex features
✅ Architecture matters as much as training
✅ CIFAR-10 requires sophisticated models
✅ Real-world vision is challenging!

Next Steps:
-----------
1. Binary classification: 08_binary_classification.py
2. Parallel training: 09_hogwild_training.py
3. Try data augmentation to improve further
4. Experiment with ResNet, VGG architectures

Congratulations! You've conquered CIFAR-10! 🎉
""")

print("=" * 70)
print("Tutorial Complete! ✓")
print("=" * 70)
