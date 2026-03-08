"""
04_mnist_classifier.py
======================
Training a CNN on MNIST Dataset

This is where the magic happens! We'll train a Convolutional Neural Network
to recognize handwritten digits with ~99% accuracy.

You've seen the data, now let's build a model that learns from it!

What you'll learn:
- Complete training pipeline from scratch
- CNN architecture for image classification
- Training loop mechanics
- Model evaluation and visualization
- Saving and loading trained models

Difficulty: Intermediate
Estimated Time: 1-2 hours

Author: PyTorch CNN Tutorial
Date: November 2025
"""

import torch.nn as nn
import torch.optim as optim
import cnn_utils as utils

# =============================================================================
# SECTION 1: Setup and Configuration
# =============================================================================

print("=" * 70)
print("Training CNN on MNIST Dataset")
print("=" * 70)

# Parse command line arguments
# Run with: python 04_mnist_classifier.py --epochs 10
cfg = utils.parse_args()

print("\nTraining Configuration:")
print(f"  Epochs: {cfg.epochs}")
print(f"  Learning rate: {cfg.lr}")
print(f"  Momentum: {cfg.momentum}")
print(f"  Batch size: {cfg.batch_size}")
print(f"  Device: {cfg.device}")
print(f"  Save model: {cfg.save_model}")

# Set random seed for reproducibility
utils.set_seed(seed=cfg.seed)
print(f"\nSeed set to {cfg.seed} for reproducible results")

# =============================================================================
# SECTION 2: Data Preparation
# =============================================================================

print("\n" + "=" * 70)
print("Loading MNIST Data")
print("=" * 70)

# Configure data loaders
train_kwargs = {'batch_size': cfg.batch_size, 'shuffle': True}
test_kwargs = {'batch_size': cfg.test_batch_size, 'shuffle': False}

# Load MNIST dataset
trainloader, testloader = utils.load_data(train_kwargs, test_kwargs)

print(f"Data loaded successfully!")
print(f"  Training batches: {len(trainloader)}")
print(f"  Test batches: {len(testloader)}")

# =============================================================================
# SECTION 3: Model Architecture
# =============================================================================

print("\n" + "=" * 70)
print("Creating CNN Model")
print("=" * 70)

# Create the CNN model and move to device (GPU if available)
model = utils.CNN().to(cfg.device)

print(f"Model created on {cfg.device}")
print(f"\nArchitecture:")
print(model)

# Count total trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal trainable parameters: {total_params:,}")

print("""
Architecture Details:
--------------------
Layer 1: Conv2D(1→32, kernel=3×3) + ReLU + MaxPool(2×2)
   • Input: (batch, 1, 28, 28)
   • After conv: (batch, 32, 28, 28)  - 32 feature maps
   • After pool: (batch, 32, 14, 14)  - reduced spatial size
   • Parameters: (3×3×1 + 1) × 32 = 320

Layer 2: Conv2D(32→64, kernel=3×3) + ReLU + MaxPool(2×2)
   • Input: (batch, 32, 14, 14)
   • After conv: (batch, 64, 14, 14)  - 64 feature maps
   • After pool: (batch, 64, 7, 7)    - further reduced
   • Parameters: (3×3×32 + 1) × 64 = 18,496

Layer 3: Fully Connected (3136 → 128) + ReLU + Dropout
   • Flatten: (batch, 64, 7, 7) → (batch, 3136)
   • Output: (batch, 128)
   • Parameters: 3136 × 128 + 128 = 401,536

Layer 4: Fully Connected (128 → 10)
   • Input: (batch, 128)
   • Output: (batch, 10) - one score per digit class
   • Parameters: 128 × 10 + 10 = 1,290

Why this architecture works:
• Convolutional layers extract spatial features (edges, textures, shapes)
• Pooling reduces spatial dimensions while keeping important info
• Dropout prevents overfitting
• Final FC layers combine features for classification
""")

# =============================================================================
# SECTION 4: Loss Function and Optimizer
# =============================================================================

print("\n" + "=" * 70)
print("Configuring Training Components")
print("=" * 70)

# Loss function: CrossEntropyLoss for multi-class classification
# Combines LogSoftmax and NLLLoss in one efficient operation
loss_ftn = nn.CrossEntropyLoss()

print(f"Loss function: CrossEntropyLoss")
print("""
  • Measures difference between predicted probabilities and true labels
  • Outputs high loss when prediction is wrong
  • Outputs low loss when prediction is correct
  • Ideal for multi-class classification
""")

# Optimizer: Stochastic Gradient Descent with momentum
# Momentum helps accelerate training and avoid local minima
optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)

print(f"Optimizer: SGD")
print(f"  • Learning rate: {cfg.lr}")
print(f"  • Momentum: {cfg.momentum}")
print("""
  • Updates weights in direction that reduces loss
  • Momentum: uses running average of gradients
  • Helps converge faster and more smoothly
""")

# Learning rate scheduler: reduces LR over time
# Starts with large steps, then fine-tunes with smaller steps
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.gamma)

print(f"Scheduler: StepLR")
print(f"  • Decay gamma: {cfg.gamma}")
print(f"  • New LR each epoch: LR = LR × {cfg.gamma}")
print("  • Helps fine-tune the model in later epochs")

# =============================================================================
# SECTION 5: Pre-Training Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Model Predictions BEFORE Training")
print("=" * 70)

print("\nLet's see what the untrained model predicts...")
print("(Should be random guesses since weights are random)")

# Show predictions before training
utils.show_batch_or_ten_images_with_label_and_predict(
    testloader, model, cfg.device, n=10
)

# =============================================================================
# SECTION 6: Training the Model
# =============================================================================

print("\n" + "=" * 70)
print("Starting Training")
print("=" * 70)

print(f"\nTraining for {cfg.epochs} epochs...")
print("Watch the loss decrease and accuracy increase!")
print("\nWhat to expect:")
print("  • Epoch 1-2: Fast improvement as model learns basic features")
print("  • Epoch 3-5: Steady improvement, learning digit shapes")
print("  • Epoch 6+: Fine-tuning, approaching ~99% accuracy")
print("\n" + "-" * 70)

# Train the model
# This is where the learning happens!
utils.train(
    model=model,
    train_loader=trainloader,
    loss_fn=loss_ftn,
    optimizer=optimizer,
    scheduler=scheduler,
    device=cfg.device,
    epochs=cfg.epochs,
    log_interval=cfg.log_interval,
    dry_run=cfg.dry_run
)

print("-" * 70)
print("Training complete!")

# =============================================================================
# SECTION 7: Post-Training Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Model Predictions AFTER Training")
print("=" * 70)

print("\nLet's see how well the trained model performs!")
print("Green = correct prediction, Red = incorrect prediction")

# Show predictions after training
utils.show_batch_or_ten_images_with_label_and_predict(
    testloader, model, cfg.device, n=10
)

# =============================================================================
# SECTION 8: Model Evaluation
# =============================================================================

print("\n" + "=" * 70)
print("Evaluating Model Performance")
print("=" * 70)

# Compute accuracy on test set
test_accuracy = utils.compute_accuracy(model, testloader, cfg.device)

print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")

if test_accuracy >= 99.0:
    print("🎉 Excellent! Above 99% accuracy!")
elif test_accuracy >= 98.0:
    print("✓ Very good! Above 98% accuracy!")
elif test_accuracy >= 95.0:
    print("✓ Good! Above 95% accuracy. Try training longer.")
else:
    print("Consider: more epochs, different learning rate, or data augmentation")

# =============================================================================
# SECTION 9: Model Persistence
# =============================================================================

print("\n" + "=" * 70)
print("Saving and Loading Model")
print("=" * 70)

if cfg.save_model:
    # Save the trained model
    utils.save_model(model, cfg.path)
    
    print(f"\nModel saved to: {cfg.path}")
    print("You can load it later to avoid retraining!")
    
    # Demonstrate loading
    print("\nDemonstrating model loading...")
    loaded_model = utils.load_model(utils.CNN, cfg.device, cfg.path)
    
    # Verify loaded model works
    loaded_accuracy = utils.compute_accuracy(loaded_model, testloader, cfg.device)
    print(f"Loaded model accuracy: {loaded_accuracy:.2f}%")
    print("✓ Successfully saved and loaded model!")
else:
    print("\nModel not saved (use --save-model flag to save)")
    print(f"To save: python 04_mnist_classifier.py --save-model --path ./mnist_cnn.pth")

# =============================================================================
# SUMMARY AND KEY TAKEAWAYS
# =============================================================================

print("\n" + "=" * 70)
print("Summary - What We Learned")
print("=" * 70)

print(f"""
🎯 Training Results:
   • Final test accuracy: {test_accuracy:.2f}%
   • Total parameters: {total_params:,}
   • Training epochs: {cfg.epochs}
   • Device used: {cfg.device}

✅ Complete ML Pipeline:
   1. Data loading and preprocessing
   2. Model architecture design
   3. Loss function and optimizer selection
   4. Training loop execution
   5. Model evaluation
   6. Model saving and loading

✅ CNN Components:
   • Convolutional layers: Extract spatial features
   • Pooling layers: Reduce spatial dimensions
   • Dropout: Prevent overfitting
   • Fully connected layers: Final classification

✅ Training Process:
   • Forward pass: Compute predictions
   • Loss computation: Measure error
   • Backward pass: Compute gradients
   • Weight update: Improve model
   • Repeat for multiple epochs

Key Insights:
------------
1. CNNs automatically learn hierarchical features
2. Early layers detect edges, later layers detect shapes
3. Dropout and momentum help generalization
4. Learning rate scheduling improves convergence
5. Small batch size → noisier but often better generalization

What Makes This Work:
--------------------
• Convolution: Parameter sharing across image
• Pooling: Translation invariance
• Non-linearity (ReLU): Learn complex patterns
• Gradient descent: Optimize efficiently
• Enough data: 60,000 training examples

Comparison:
-----------
• Traditional ML (SVM, etc.): ~95% accuracy
• Simple fully connected NN: ~97% accuracy
• Our CNN: ~99% accuracy ✓
• State-of-the-art: ~99.8% accuracy

Next Steps:
-----------
1. Try Fashion-MNIST: 05_fashion_mnist_classifier.py
   (Same code, different dataset, lower accuracy!)

2. Experiment:
   • Change learning rate (try 0.001, 0.1)
   • Modify architecture (add layers, change filters)
   • Adjust batch size (32, 128, 256)
   • Add data augmentation

3. Move to CIFAR-10: 06_cifar10_basic.py
   (More challenging with color images!)

Challenge Questions:
-------------------
Q: Why do we use CrossEntropyLoss instead of MSE?
A: CrossEntropyLoss is designed for classification and handles
   probabilities correctly. MSE works better for regression.

Q: What does momentum do?
A: It helps escape local minima and accelerates training by
   considering previous gradient directions.

Q: Why normalize images to [-1, 1]?
A: Normalized inputs help gradients flow better and make
   training more stable and faster.

Q: How does dropout help?
A: It randomly drops neurons during training, forcing the
   network to learn robust features that don't rely on
   specific neurons.

Congratulations! 🎉
-------------------
You've successfully trained a CNN on real data!
You now understand the complete deep learning pipeline.
""")

print("=" * 70)
print("Tutorial Complete! ✓")
print("=" * 70)


if __name__ == "__main__":
    pass
