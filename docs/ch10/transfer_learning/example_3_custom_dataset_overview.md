# Example 3: Transfer Learning with Custom Datasets

## 🎯 Learning Objectives

By completing this example, you will learn:
- How to create custom PyTorch Dataset classes
- Organizing and loading your own image data
- Implementing proper train/validation/test splits
- Handling class imbalance
- Advanced data augmentation techniques
- Best practices for real-world applications

## 📋 Overview

This example shows you how to apply transfer learning to **your own custom datasets**.

Real-world scenarios:
- Medical image classification (X-rays, MRIs)
- Product categorization (e-commerce)
- Quality control (defect detection)
- Wildlife recognition (camera trap images)

**What's Different:**
- We'll create a synthetic custom dataset to demonstrate
- Custom Dataset and DataLoader implementation
- Handling various image formats and sizes
- Dealing with imbalanced classes

## 🗂️ Dataset Structure

We'll use the standard image folder structure:
```
dataset/
├── train/
│   ├── class_1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class_2/
│   │   ├── img1.jpg
│   │   └── ...
│   └── class_3/
│       └── ...
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── class_3/
└── test/
    ├── class_1/
    ├── class_2/
    └── class_3/
```

This structure is standard and works with PyTorch's `ImageFolder` class.

## 🔍 Key Concepts

### 1. Custom Dataset Class
Learn to create a PyTorch Dataset class for:
- Loading images from disk
- Applying transformations
- Handling labels
- Supporting various image formats

### 2. Data Splits
Proper splitting is crucial:
- **Training set (70%)**: Model learns from this
- **Validation set (15%)**: Tune hyperparameters, early stopping
- **Test set (15%)**: Final unbiased evaluation

### 3. Class Imbalance
Real-world datasets are often imbalanced:
- Class 1: 1000 images
- Class 2: 500 images
- Class 3: 200 images

We'll learn to handle this with:
- Weighted loss functions
- Class-balanced sampling
- Evaluation metrics beyond accuracy

### 4. Advanced Augmentation
More sophisticated transformations:
- Random rotation and affine transforms
- Gaussian blur and noise
- Cutout and mixup (optional)

## 💻 Running the Code

```bash
python custom_dataset_transfer.py
```

The script will:
1. Create a synthetic dataset for demonstration
2. Apply transfer learning
3. Show handling of imbalanced data

**Expected Runtime:** 10-15 minutes

## 📊 Expected Results

You should see:
- How to handle real-world data organization
- Techniques for imbalanced datasets
- Proper evaluation metrics (precision, recall, F1)
- Test accuracy: depends on dataset characteristics

## 🔧 Hyperparameters

- Batch size: 32
- Learning rate: 0.001 (new layer), 0.0001 (fine-tuned layers)
- Optimizer: Adam
- Epochs: 20
- Dataset: Custom (synthetic for demo)

## 🎓 Key Takeaways

1. **Folder structure matters** - Organize your data properly
2. **Always split your data** - Train/val/test separation is crucial
3. **Handle imbalance** - Real data is rarely perfectly balanced
4. **Use appropriate metrics** - Accuracy alone can be misleading
5. **Augmentation helps** - Especially with limited data

## 🚀 Next Steps

After completing this example:
- Apply to your own dataset (just organize it properly!)
- Experiment with different augmentation strategies
- Try different pre-trained models
- Move to Example 4 for advanced techniques!

## 💡 Applying to Your Own Data

To use your own images:

1. Organize them in the folder structure shown above
2. Update the `data_dir` path in the code
3. Adjust the number of classes
4. Run the script!

The code is designed to be easily adaptable.

## 🤔 Common Questions

**Q: What image formats are supported?**
A: JPG, PNG, and most common formats via PIL

**Q: Do all images need to be the same size?**
A: No! Transforms will resize them automatically

**Q: How much data do I need?**
A: Minimum ~100 images per class, more is better

**Q: What if my classes are very imbalanced?**
A: Use weighted loss or class-balanced sampling (shown in code)
