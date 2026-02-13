# Example 3: Medical Image Segmentation

## 🎯 Learning Objectives

By completing this example, you will learn:
- Domain-specific techniques for medical imaging
- Dice loss and Dice coefficient metrics
- Handling extreme class imbalance
- Medical imaging preprocessing
- Proper validation for medical AI
- Sensitivity, Specificity, and clinical metrics
- Working with grayscale images

## 📋 Overview

This example focuses on **medical image segmentation**, which has unique challenges and requirements compared to natural image segmentation.

**Medical Imaging Challenges:**
- ⚕️ Extreme class imbalance (lesions << background)
- 🔬 High precision requirements (clinical impact)
- 📊 Domain-specific metrics (Dice, Sensitivity, Specificity)
- 🎯 Small objects of interest
- 💾 Often limited labeled data
- 🏥 Need for interpretability and trust

## 🏥 Medical Segmentation Task

We'll segment skin lesions for melanoma detection using the ISIC dataset (International Skin Imaging Collaboration).

**Task:** Given a dermoscopic image, segment the lesion from healthy skin.

```
Input: Dermoscopic Image     →     Output: Lesion Mask
┌─────────────────┐                ┌─────────────────┐
│                 │                │     ░░░░░       │
│   ●●●●●●●       │                │    ░░░░░░░      │
│  ●●    ●●●      │      →         │   ░░    ░░░     │
│   ●●●●●●        │                │    ░░░░░░       │
│                 │                │                 │
└─────────────────┘                └─────────────────┘
```

**Clinical Importance:**
- Early detection improves survival rates
- Assists dermatologists in diagnosis
- Enables large-scale screening
- Quantifies lesion characteristics

## 🔍 Key Differences from General Segmentation

| Aspect | General | Medical |
|--------|---------|---------|
| **Class Balance** | Moderate | Extreme (lesion ~5-10%) |
| **Loss Function** | CrossEntropy | Dice Loss |
| **Metrics** | IoU, mIoU | Dice, Sensitivity, Specificity |
| **Data** | Abundant | Limited (expensive labeling) |
| **Requirements** | Good accuracy | High precision + recall |
| **Stakes** | Low | High (clinical decisions) |

## 📏 Medical Metrics Explained

### 1. Dice Coefficient (F1-Score for Segmentation)
```
Dice = 2 × |Prediction ∩ Ground Truth| / (|Prediction| + |Ground Truth|)
     = 2TP / (2TP + FP + FN)
```

**Why Dice?**
- More sensitive to small objects than IoU
- Standard metric in medical imaging
- Penalizes false positives and negatives equally

**Relationship to IoU:**
```
Dice = 2 × IoU / (1 + IoU)
```

### 2. Sensitivity (Recall, True Positive Rate)
```
Sensitivity = TP / (TP + FN)
```
- Measures: "What % of actual lesion did we find?"
- Clinical importance: Missing a lesion is dangerous
- High sensitivity = few false negatives

### 3. Specificity (True Negative Rate)
```
Specificity = TN / (TN + FP)
```
- Measures: "What % of healthy tissue is correctly identified?"
- Clinical importance: Avoid unnecessary biopsies
- High specificity = few false positives

### 4. Hausdorff Distance
Measures maximum boundary error:
- Important for surgical planning
- Quantifies worst-case boundary deviation

## 🎓 Dice Loss vs Cross-Entropy

### Cross-Entropy Loss
```
CE = -Σ y_i log(p_i)
```
**Issues for medical imaging:**
- Treats each pixel independently
- Dominated by majority class (background)
- Doesn't directly optimize Dice score

### Dice Loss
```
Dice Loss = 1 - Dice Coefficient
          = 1 - 2TP/(2TP + FP + FN)
```
**Advantages for medical imaging:**
- Directly optimizes the evaluation metric
- Handles class imbalance naturally
- Focuses on overlap, not pixel-wise accuracy
- Better for small objects

### Combined Loss
```
Total Loss = α × CE + β × Dice
```
Best of both worlds!

## 💻 Running the Code

```bash
python medical_segmentation.py
```

**Expected Runtime:** 10-15 minutes on GPU

The script will:
1. Generate synthetic medical images
2. Train a U-Net with Dice loss
3. Evaluate with clinical metrics
4. Visualize predictions

## 📊 Expected Results

You should see:
- Dice Coefficient: ~0.85-0.92
- Sensitivity: ~0.88-0.94
- Specificity: ~0.95-0.98
- Much better handling of small lesions

## 🔧 Hyperparameters

Default settings:
- Input size: 256×256
- Batch size: 16
- Learning rate: 0.001
- Optimizer: Adam
- Epochs: 30
- Loss: Dice Loss + Binary CE
- Class weight: Heavy weight on lesion class

## 🏥 Medical AI Best Practices

### 1. Proper Data Splitting
```
By Patient, Not by Image!
❌ Wrong: Random image split
✓ Right: Patient-level split
```
Prevents data leakage from same patient.

### 2. Clinical Validation
- Test on data from different hospitals
- Different imaging devices
- Different demographics
- Measure clinical utility, not just metrics

### 3. Calibration
Ensure predicted probabilities match true frequencies:
```
If model says 80% confidence → should be correct 80% of time
```

### 4. Interpretability
- Visualize attention maps
- Explain predictions
- Identify failure modes
- Enable clinician review

### 5. Uncertainty Quantification
Medical AI should know when it's uncertain:
- Monte Carlo Dropout
- Test-time augmentation
- Ensemble predictions

## 🎯 Handling Class Imbalance

### Strategy 1: Weighted Loss
```python
# Give more weight to rare class
pos_weight = (background_pixels / lesion_pixels)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### Strategy 2: Dice Loss
Naturally handles imbalance by focusing on overlap.

### Strategy 3: Focal Loss
```python
# Down-weight easy examples, focus on hard ones
focal_loss = -α(1-p)^γ log(p)
```

### Strategy 4: Oversampling
Sample images with lesions more frequently.

## 🚀 Next Steps

After understanding this example:
- Try 3D medical segmentation (CT, MRI volumes)
- Implement uncertainty estimation
- Add boundary refinement
- Try other medical datasets (brain, lung, etc.)
- Move to Example 4 for advanced techniques!

## 🤔 Questions to Think About

1. Why is Dice loss better than CE for medical imaging?
2. What's the difference between Sensitivity and Specificity?
3. Why split by patient instead of by image?
4. When might high Dice score still not be clinically useful?

## 💡 Extension Ideas

- Multi-class organ segmentation
- 3D volumetric segmentation (CT/MRI)
- Weakly supervised segmentation (bounding boxes only)
- Active learning for efficient labeling
- Domain adaptation (different hospitals/devices)
- Explainable AI for predictions

## 📚 Medical Imaging Datasets

**Publicly Available:**
- **ISIC**: Skin lesion segmentation
- **DRIVE/CHASE**: Retinal vessel segmentation
- **BraTS**: Brain tumor segmentation
- **KiTS**: Kidney tumor segmentation
- **LiTS**: Liver tumor segmentation
- **LUNA16**: Lung nodule detection
- **ChestX-ray14**: Thorax disease classification

**Access Requirements:**
- Registration and data use agreement
- Ethical approval for research
- Citation of dataset papers
- Reporting on same test set for comparison

## ⚠️ Ethical Considerations

1. **Patient Privacy**: Proper de-identification
2. **Bias**: Test on diverse populations
3. **Clinical Validation**: Partner with clinicians
4. **Regulatory**: FDA/CE marking for deployment
5. **Transparency**: Clear limitations and failure modes
6. **Human in Loop**: AI assists, doesn't replace doctors
