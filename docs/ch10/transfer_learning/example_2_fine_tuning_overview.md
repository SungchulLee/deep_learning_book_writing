# Example 2: Fine-Tuning a Pre-trained Model

## 🎯 Learning Objectives

By completing this example, you will learn:
- The difference between feature extraction and fine-tuning
- How to selectively unfreeze layers for training
- Using different learning rates for different parts of the network
- Implementing early stopping to prevent overfitting
- Best practices for fine-tuning

## 📋 Overview

This example builds on Example 1 by introducing **fine-tuning**.

**The Difference:**
- **Feature Extraction (Example 1):** Only train the final layer
- **Fine-Tuning (Example 2):** Train multiple layers with different learning rates

**The Strategy:**
1. Start with a pre-trained model (ResNet18)
2. Replace the final layer
3. Initially freeze early layers, unfreeze later layers
4. Use different learning rates: smaller for pre-trained layers, larger for new layers
5. Optionally fine-tune more layers as training progresses

## 🔍 What's Happening?

```
Pre-trained ResNet18 (ImageNet)
        ↓
[Early Conv Layers] ← FROZEN (generic features)
        ↓
[Middle Conv Layers] ← FINE-TUNING (adapt features, small LR)
        ↓
[Late Conv Layers] ← FINE-TUNING (task-specific features, small LR)
        ↓
[New Classifier] ← TRAINING (large LR)
        ↓
[10 Classes Output]
```

## 🤔 When to Use Fine-Tuning?

**Use Fine-Tuning When:**
- Your dataset is different from ImageNet
- You have sufficient training data (thousands of samples)
- You want to achieve the best possible accuracy
- You have computational resources for longer training

**Use Feature Extraction When:**
- Your dataset is similar to ImageNet
- You have limited data (hundreds of samples)
- You need quick results
- Computational resources are limited

## 💻 Running the Code

```bash
python fine_tuning.py
```

**Expected Runtime:** 10-15 minutes on GPU, 30-45 minutes on CPU

## 📊 Expected Results

Compared to Example 1, you should see:
- Training accuracy: ~90-95%
- Test accuracy: ~85-90%
- Better performance but longer training time
- The model takes more epochs to converge

## 🔧 Hyperparameters

Default settings:
- Batch size: 32
- Learning rate (pre-trained layers): 0.0001
- Learning rate (new layer): 0.001
- Optimizer: Adam
- Epochs: 15
- Dataset: CIFAR-10

## 🎓 Key Concepts

### 1. Discriminative Learning Rates
Different parts of the network learn different things:
- Early layers: Generic features (edges, colors) - use small LR
- Late layers: Task-specific features - use medium LR
- New classifier: Random initialization - use large LR

### 2. Gradual Unfreezing
Start by training only the classifier, then gradually unfreeze more layers. This prevents destroying pre-trained weights early in training.

### 3. Early Stopping
Stop training when validation performance stops improving to prevent overfitting.

## 🚀 Next Steps

After understanding this example:
- Experiment with different unfreezing strategies
- Try gradual unfreezing (unfreeze more layers over time)
- Compare results with Example 1
- Move on to Example 3 to work with custom datasets!

## 🤔 Questions to Think About

1. Why do we use a smaller learning rate for pre-trained layers?
2. What happens if we use the same learning rate for all layers?
3. When might fine-tuning perform worse than feature extraction?

Experiment with the code to find the answers!
