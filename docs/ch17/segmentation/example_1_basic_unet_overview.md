# Example 1: Basic U-Net Semantic Segmentation

## 🎯 Learning Objectives

By completing this example, you will learn:
- What semantic segmentation is and how it differs from classification
- The U-Net architecture (encoder-decoder with skip connections)
- How to implement U-Net from scratch in PyTorch
- Pixel-wise loss functions
- IoU (Intersection over Union) metric
- Data augmentation for segmentation tasks
- Binary segmentation fundamentals

## 📋 Overview

This example introduces semantic segmentation using the classic **U-Net architecture**. We'll start with a simple binary segmentation task (2 classes: background and foreground) on a synthetic dataset.

**Why U-Net?**
- Simple and intuitive architecture
- Works well with limited data
- Widely used in medical imaging
- Foundation for many modern architectures
- Skip connections preserve spatial information

## 🏗️ U-Net Architecture Explained

```
Input (256×256×3)
    ↓
┌───────────────────────────────────────────┐
│  ENCODER (Downsampling Path)              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │ Conv    │→ │ Conv    │→ │ MaxPool │  │
│  │ 64      │  │ 64      │  │ ÷2      │  │
│  └────┬────┘  └─────────┘  └────┬────┘  │
│       │ Skip Connection 1        │        │
│       │                     ┌────┴────┐  │
│       │                     │ Conv    │  │
│       │                     │ 128     │  │
│       │                     └────┬────┘  │
│       │ Skip Connection 2        │        │
│       │                     ┌────┴────┐  │
│       │                     │ Conv    │  │
│       │                     │ 256     │  │
│       │                     └────┬────┘  │
│       │                          │        │
└───────┼──────────────────────────┼────────┘
        │                          │
┌───────┼──────────────────────────┼────────┐
│       │     BOTTLENECK           │        │
│       │     ┌────────┐           │        │
│       │     │ Conv   │           │        │
│       │     │ 512    │           │        │
│       │     └────┬───┘           │        │
└───────┼──────────┼───────────────┼────────┘
        │          │               │
┌───────┼──────────┼───────────────┼────────┐
│  DECODER (Upsampling Path)       │        │
│       │     ┌────┴────┐          │        │
│       │     │ Upsample│          │        │
│       │     │ ×2      │          │        │
│       │     └────┬────┘          │        │
│       │  ┌───────┴────────┐      │        │
│       └─→│ Concatenate    │      │        │
│          └───────┬────────┘      │        │
│          ┌───────┴────────┐      │        │
│          │ Conv 256       │      │        │
│          └───────┬────────┘      │        │
│                  ⋮               │        │
│          [Repeat upsampling]     │        │
│                  │               │        │
└──────────────────┼───────────────┘        
                   ↓
        Output (256×256×2)
        [Background, Foreground]
```

**Key Components:**
1. **Encoder:** Captures context, reduces spatial size
2. **Bottleneck:** Highest level features
3. **Decoder:** Recovers spatial resolution
4. **Skip Connections:** Preserve fine-grained details

## 🎨 Binary Segmentation Task

We'll segment simple shapes from backgrounds:

```
Input Image:          Ground Truth Mask:
┌─────────────┐      ┌─────────────┐
│   ○○○○○     │      │   11111     │
│  ○     ○    │  →   │  1     1    │
│   ○○○○○     │      │   11111     │
│             │      │             │
└─────────────┘      └─────────────┘
```

## 🔍 What's Happening?

1. **Input:** RGB image (3 channels)
2. **Encoder:** Extracts features, reduces size (256→128→64→32)
3. **Bottleneck:** Processes deepest features
4. **Decoder:** Upsamples back to original size (32→64→128→256)
5. **Skip Connections:** Copy features from encoder to decoder at each level
6. **Output:** 2-channel prediction (background, foreground)

## 💻 Running the Code

```bash
python basic_unet_segmentation.py
```

**Expected Runtime:** 5-10 minutes on GPU, 15-25 minutes on CPU

## 📊 Expected Results

You should see:
- Training IoU: ~85-95%
- Validation IoU: ~80-90%
- Clear visualization of predictions
- Model learns to segment shapes accurately

## 🔧 Hyperparameters

Default settings:
- Input size: 256×256
- Batch size: 8
- Learning rate: 0.001
- Optimizer: Adam
- Epochs: 20
- Loss: Binary Cross-Entropy
- Dataset: Synthetic shapes

## 📏 Evaluation Metrics

### IoU (Intersection over Union)
The primary metric for segmentation:
```
IoU = Area of Overlap / Area of Union
    = True Positives / (True Positives + False Positives + False Negatives)
```

**Example:**
```
Prediction:    Ground Truth:   Intersection:   Union:
┌─────┐       ┌─────┐         ┌─────┐        ┌─────┐
│ ███ │       │ ███ │         │ ██  │        │ ███ │
│ ███ │   &   │ ██  │    =    │ ██  │   |    │ ███ │
└─────┘       └─────┘         └─────┘        └─────┘

IoU = 4 / 6 = 0.67
```

**IoU Interpretation:**
- 0.9-1.0: Excellent
- 0.7-0.9: Good
- 0.5-0.7: Acceptable
- <0.5: Poor

### Pixel Accuracy
```
Accuracy = Correct Pixels / Total Pixels
```
Note: Can be misleading if classes are imbalanced!

## 🎓 Key Takeaways

1. **Encoder-Decoder Structure**: Downsampling captures context, upsampling recovers resolution
2. **Skip Connections**: Critical for preserving spatial details
3. **Pixel-wise Loss**: Each pixel contributes to the loss
4. **Data Augmentation**: Must apply same transform to image AND mask
5. **IoU Metric**: Better than accuracy for segmentation

## 🚀 Next Steps

After understanding this example:
- Visualize feature maps at different layers
- Try different loss functions (Dice loss)
- Experiment with different encoder depths
- Add more augmentation
- Move on to Example 2 for pre-trained encoders!

## 🤔 Questions to Think About

1. Why do we use skip connections?
2. What happens if we remove skip connections?
3. Why is pixel accuracy not enough?
4. How does segmentation differ from classification?

Experiment with the code to find the answers!

## 💡 Extension Ideas

- Try 3+ classes (multi-class segmentation)
- Implement Dice loss instead of BCE
- Add more complex shapes
- Visualize what each layer learns
- Try different decoder architectures
