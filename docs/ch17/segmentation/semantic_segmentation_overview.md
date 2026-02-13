# PyTorch Semantic Segmentation Tutorial for Undergraduates

Welcome! This tutorial package contains four progressively challenging examples of semantic segmentation using PyTorch. Each example is fully commented and designed to help you understand semantic segmentation from the ground up.

## 📚 What is Semantic Segmentation?

Semantic segmentation is a computer vision task where we classify **every pixel** in an image into a category. Unlike classification (which predicts one label per image) or object detection (which predicts bounding boxes), semantic segmentation creates a detailed pixel-wise mask.

**Applications:**
- 🏥 Medical imaging (tumor detection, organ segmentation)
- 🚗 Autonomous driving (road, pedestrian, vehicle segmentation)
- 🛰️ Satellite imagery (land use classification)
- 📱 Image editing (background removal, filters)
- 🌾 Agriculture (crop health monitoring)

**Example:**
```
Input Image:        Segmentation Mask:
[Photo of street]   [Road=blue, Car=red, Sky=green, etc.]
```

## 🎯 Key Concepts

### Pixel-wise Classification
Each pixel gets its own label, creating a detailed understanding of the scene.

### Encoder-Decoder Architecture
- **Encoder:** Extracts features and reduces spatial dimensions
- **Decoder:** Upsamples features back to original resolution
- **Skip Connections:** Preserve spatial details

### Common Architectures
- **U-Net:** Medical imaging, symmetric encoder-decoder
- **FCN:** Fully Convolutional Network, pioneering work
- **DeepLab:** Atrous convolutions, ASPP module
- **SegNet:** Efficient architecture with pooling indices

## 📂 Project Structure

```
pytorch_semantic_segmentation_tutorial/
│
├── README.md (this file)
├── requirements.txt
│
├── example_1_basic_unet/
│   ├── README.md
│   └── basic_unet_segmentation.py
│
├── example_2_pretrained_encoders/
│   ├── README.md
│   └── pretrained_segmentation.py
│
├── example_3_medical_segmentation/
│   ├── README.md
│   └── medical_segmentation.py
│
└── example_4_advanced_techniques/
    ├── README.md
    └── advanced_segmentation.py
```

## 🎓 Examples Overview

### Example 1: Basic U-Net Architecture
**Difficulty: ⭐ Beginner**

Learn the fundamentals of semantic segmentation by implementing a simple U-Net from scratch on a synthetic dataset.

**Key Concepts:**
- U-Net architecture (encoder-decoder with skip connections)
- Pixel-wise cross-entropy loss
- Basic data augmentation for segmentation
- IoU (Intersection over Union) metric
- Binary segmentation (2 classes)

### Example 2: Pre-trained Encoders
**Difficulty: ⭐⭐ Intermediate**

Use transfer learning for segmentation! Learn to use pre-trained ResNet, VGG as encoders with decoder networks.

**Key Concepts:**
- Transfer learning for segmentation
- DeepLab architecture
- Feature Pyramid Networks (FPN)
- Multi-scale predictions
- PASCAL VOC dataset (21 classes)

### Example 3: Medical Image Segmentation
**Difficulty: ⭐⭐⭐ Intermediate-Advanced**

Apply segmentation to real medical imaging tasks. Learn domain-specific techniques and metrics.

**Key Concepts:**
- Medical imaging preprocessing
- Dice loss and Dice coefficient
- Handling class imbalance
- 3D volume handling basics
- Proper validation for medical AI
- Sensitivity, Specificity metrics

### Example 4: Advanced Techniques
**Difficulty: ⭐⭐⭐⭐ Advanced**

Master state-of-the-art segmentation techniques used in research and production.

**Key Concepts:**
- Atrous/Dilated convolutions
- Attention mechanisms (CBAM, Self-attention)
- Multi-scale training and inference
- Test-time augmentation for segmentation
- Post-processing (CRF, morphological operations)
- Mixed loss functions
- Hard example mining

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Basic understanding of CNNs and PyTorch
- Familiarity with image classification (helpful)
- GPU recommended but not required

### Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Start with Example 1 and progress through the examples in order.

### Running the Examples

Each example is self-contained and can be run independently:

```bash
# Example 1
cd example_1_basic_unet
python basic_unet_segmentation.py

# Example 2
cd example_2_pretrained_encoders
python pretrained_segmentation.py

# Example 3
cd example_3_medical_segmentation
python medical_segmentation.py

# Example 4
cd example_4_advanced_techniques
python advanced_segmentation.py
```

## 📖 Learning Path

We recommend following this learning path:

1. **Start with Example 1**: Understand U-Net and basic segmentation
2. **Move to Example 2**: Learn to use pre-trained encoders
3. **Practice with Example 3**: Apply to medical imaging
4. **Explore Example 4**: Master advanced techniques

## 💡 Key Differences from Classification

| Aspect | Classification | Segmentation |
|--------|---------------|--------------|
| **Output** | Single label | Pixel-wise labels |
| **Loss** | CrossEntropy | Pixel-wise loss |
| **Metrics** | Accuracy | IoU, Dice |
| **Architecture** | CNN + FC | Encoder-Decoder |
| **Data Aug** | Simple crops | Matching transforms |

## 🔧 Common Metrics Explained

### IoU (Intersection over Union)
```
IoU = Area of Overlap / Area of Union
```
- Range: 0 to 1
- Higher is better
- Industry standard metric

### Dice Coefficient
```
Dice = 2 × |A ∩ B| / (|A| + |B|)
```
- Range: 0 to 1
- Similar to IoU but more sensitive to small objects
- Common in medical imaging

### Pixel Accuracy
```
Accuracy = Correct Pixels / Total Pixels
```
- Can be misleading with class imbalance
- Easy to understand

## 🎯 Architecture Comparison

| Architecture | Best For | Pros | Cons |
|-------------|----------|------|------|
| **U-Net** | Medical, Small data | Simple, effective | Memory intensive |
| **DeepLab** | General purpose | State-of-art | Complex |
| **FCN** | Real-time | Fast | Lower accuracy |
| **PSPNet** | Scene parsing | Multi-scale | Slow |

## 💡 Tips for Success

- Start with small images (256×256) for faster iteration
- Use data augmentation heavily (flip, rotate, crop)
- Monitor IoU, not just loss
- Visualize predictions frequently
- Use pre-trained encoders when possible
- Consider class weights for imbalanced datasets

## 🔧 Common Issues & Solutions

### Memory Issues
- Reduce batch size
- Use smaller input sizes
- Enable gradient checkpointing
- Use mixed precision training

### Poor Boundary Prediction
- Use weighted loss at boundaries
- Add boundary refinement module
- Use multi-scale predictions

### Class Imbalance
- Use weighted cross-entropy
- Use focal loss
- Use Dice loss
- Oversample minority classes

### Overfitting
- More data augmentation
- Dropout in decoder
- Early stopping
- Reduce model size

## 📚 Additional Resources

- [U-Net Paper](https://arxiv.org/abs/1505.04597) - Original U-Net paper
- [DeepLab Paper](https://arxiv.org/abs/1606.00915) - State-of-art segmentation
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch) - Pre-built models
- [Papers with Code - Segmentation](https://paperswithcode.com/task/semantic-segmentation) - Latest research

## 🤝 Dataset Recommendations

**For Practice:**
- **PASCAL VOC 2012** - 20 object classes, ~2,900 images
- **Cityscapes** - Urban street scenes, 30 classes
- **ADE20K** - Scene parsing, 150 classes
- **CamVid** - Road scene understanding

**For Medical:**
- **ISIC** - Skin lesion segmentation
- **DRIVE** - Retinal vessel segmentation  
- **BraTS** - Brain tumor segmentation

**Synthetic (for learning):**
- Generate simple shapes
- Use this tutorial's built-in generators

## 📝 Evaluation Best Practices

1. **Split data properly:** Train/Val/Test (70/15/15)
2. **Report multiple metrics:** IoU, Dice, Pixel Accuracy
3. **Show qualitative results:** Visualize predictions
4. **Per-class metrics:** Especially for imbalanced data
5. **Test on different domains:** Check generalization

## ⚠️ Important Notes

- Segmentation requires more memory than classification
- Training takes longer due to pixel-wise loss
- Data augmentation must preserve mask alignment
- First run downloads datasets (may take time)

## 🎉 What You'll Learn

By completing all examples, you will:
- ✅ Understand encoder-decoder architectures
- ✅ Implement U-Net from scratch
- ✅ Use transfer learning for segmentation
- ✅ Apply to medical imaging
- ✅ Master advanced techniques (attention, multi-scale)
- ✅ Evaluate models properly (IoU, Dice)
- ✅ Handle class imbalance
- ✅ Deploy segmentation models

---

**Happy Segmenting! 🎨**

If you find these examples helpful, consider sharing them with fellow students or contributing improvements!

## 📧 Feedback

This is an educational resource. Feel free to modify and extend these examples for your own learning and projects!
