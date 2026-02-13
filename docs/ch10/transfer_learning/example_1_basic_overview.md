# Example 1: Basic Transfer Learning (Feature Extraction)

## 🎯 Learning Objectives

By completing this example, you will learn:
- How to load a pre-trained model from torchvision
- How to freeze model parameters to use it as a feature extractor
- How to replace the final layer for your specific task
- How to train only the new classifier while keeping other layers frozen
- Basic training loop structure in PyTorch

## 📋 Overview

This example demonstrates the simplest form of transfer learning: **feature extraction**.

**The Strategy:**
1. Load a pre-trained ResNet18 model (trained on ImageNet with 1.2M images)
2. Freeze all layers except the final classification layer
3. Replace the final layer to match our number of classes (10 for CIFAR-10)
4. Train only the new final layer on CIFAR-10 dataset

**Why This Works:**
- Early layers learn general features (edges, textures, patterns)
- These features are useful across many vision tasks
- We only need to learn task-specific features in the final layer

## 🔍 What's Happening?

```
Pre-trained ResNet18 (ImageNet)
        ↓
[Conv Layers] ← FROZEN (already knows edges, textures, etc.)
        ↓
[Feature Maps] ← FROZEN (already knows object parts)
        ↓
[New Classifier] ← TRAINING (learning CIFAR-10 classes)
        ↓
[10 Classes Output]
```

## 💻 Running the Code

```bash
python basic_transfer_learning.py
```

**Expected Runtime:** 5-10 minutes on GPU, 20-30 minutes on CPU

## 📊 Expected Results

You should see:
- Training accuracy: ~85-90%
- Test accuracy: ~80-85%
- Much faster training than training from scratch
- The model converges in just a few epochs

## 🔧 Hyperparameters

Default settings:
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Epochs: 10
- Dataset: CIFAR-10

Feel free to experiment with these values!

## 🎓 Key Takeaways

1. **Transfer learning is powerful** - We achieve good results without training millions of parameters
2. **Freezing layers** - Setting `requires_grad=False` prevents updating those weights
3. **Only final layer trains** - This is fast and efficient for datasets similar to ImageNet
4. **Data normalization matters** - We use ImageNet's mean and std values

## 🚀 Next Steps

After understanding this example:
- Try different pre-trained models (ResNet50, VGG16)
- Experiment with different learning rates
- Try a different dataset
- Move on to Example 2 to learn fine-tuning!

## 🤔 Questions to Think About

1. Why do we freeze the earlier layers?
2. What would happen if we didn't use pre-trained weights?
3. When might feature extraction not be enough?

Answers to these questions become clear in Example 2!
