# PyTorch Loss Functions & Optimizers Tutorial

A comprehensive, hands-on tutorial for undergraduate students to master loss functions and optimizers in PyTorch. This package contains fully commented code examples progressing from beginner to advanced topics.

## 📚 Overview

This tutorial covers:
- **Loss Functions**: MSE, MAE, Cross-Entropy, custom losses, and more
- **Optimizers**: SGD, Adam, AdamW, RMSprop, and comparisons
- **Learning Rate Scheduling**: Step, Exponential, Cosine, Plateau-based
- **Real-World Applications**: Complete training pipelines with MNIST

## 🎯 Learning Path

### Prerequisites
- Basic Python programming
- Familiarity with PyTorch tensors
- Understanding of neural networks (helpful but not required)

### Suggested Order
1. **Beginner** (Start here if new to PyTorch training)
2. **Intermediate** (After completing beginner tutorials)
3. **Advanced** (For deeper understanding and custom implementations)
4. **Real-World Examples** (Apply everything you've learned)

## 📂 Directory Structure

```
pytorch_loss_optimizer_tutorial/
│
├── 01_beginner/                      # Start here!
│   ├── 01_intro_to_loss_functions.py
│   ├── 02_regression_losses_comparison.py
│   ├── 03_intro_to_optimizers.py
│   └── 04_classification_losses.py
│
├── 02_intermediate/
│   ├── 01_optimizer_comparison.py
│   └── 02_learning_rate_schedulers.py
│
├── 03_advanced/
│   └── 01_custom_loss_functions.py
│
├── 04_real_world_examples/
│   └── 01_complete_mnist_training.py
│
└── README.md                         # This file
```

## 📖 Tutorial Descriptions

### 01_beginner/

#### `01_intro_to_loss_functions.py` (⏱️ 10 min)
**What you'll learn:**
- What is a loss function and why we need it
- Three ways to compute loss in PyTorch
- Mean Squared Error (MSE) for regression
- How to interpret loss values

**Key concepts:** Loss calculation, MSE, reduction methods

---

#### `02_regression_losses_comparison.py` (⏱️ 15 min)
**What you'll learn:**
- MSE vs MAE vs Smooth L1 Loss
- How different losses handle outliers
- When to use each loss function
- Impact on model training

**Key concepts:** L1/L2 loss, Huber loss, outlier robustness

---

#### `03_intro_to_optimizers.py` (⏱️ 20 min)
**What you'll learn:**
- What is an optimizer and how it works
- Understanding learning rate
- Basic SGD optimizer
- Complete training loop walkthrough

**Key concepts:** Gradient descent, learning rate, parameter updates, training loop

---

#### `04_classification_losses.py` (⏱️ 20 min)
**What you'll learn:**
- Binary vs multi-class classification
- Binary Cross-Entropy (BCE)
- Cross-Entropy Loss
- Logits vs probabilities

**Key concepts:** Classification, BCE, softmax, cross-entropy

---

### 02_intermediate/

#### `01_optimizer_comparison.py` (⏱️ 25 min)
**What you'll learn:**
- Comparing SGD, Adam, RMSprop, AdamW
- When to use each optimizer
- Momentum and adaptive learning rates
- Practical performance comparison

**Key concepts:** Momentum, adaptive learning rates, optimizer choice

---

#### `02_learning_rate_schedulers.py` (⏱️ 20 min)
**What you'll learn:**
- Why learning rate scheduling matters
- StepLR, ExponentialLR, CosineAnnealingLR
- ReduceLROnPlateau
- Learning rate warmup

**Key concepts:** LR scheduling, annealing, adaptive scheduling

---

### 03_advanced/

#### `01_custom_loss_functions.py` (⏱️ 30 min)
**What you'll learn:**
- When and why to create custom losses
- Implementing custom loss functions
- Focal Loss for imbalanced data
- Dice Loss for segmentation
- Combining multiple losses

**Key concepts:** Custom losses, class imbalance, multi-task learning

---

### 04_real_world_examples/

#### `01_complete_mnist_training.py` (⏱️ 30 min)
**What you'll learn:**
- End-to-end training pipeline
- Data loading and preprocessing
- Training with validation
- Model saving and loading
- Inference

**Key concepts:** Complete workflow, best practices, production code

---

## 🚀 Quick Start

### Installation

```bash
# Install PyTorch (visit pytorch.org for your system-specific command)
pip install torch torchvision

# Optional: For visualization
pip install matplotlib
```

### Running the Tutorials

```bash
# Navigate to the tutorial directory
cd pytorch_loss_optimizer_tutorial

# Start with the first beginner tutorial
python 01_beginner/01_intro_to_loss_functions.py

# Continue in order through each directory
```

### Running in Google Colab

1. Upload the entire `pytorch_loss_optimizer_tutorial` folder to your Google Drive
2. Mount your drive in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
```
3. Navigate and run:
```python
%cd /content/drive/MyDrive/pytorch_loss_optimizer_tutorial
!python 01_beginner/01_intro_to_loss_functions.py
```

## 📊 What Each Tutorial Produces

Most tutorials produce:
- ✅ **Console Output**: Detailed explanations and results
- ✅ **Visual Output**: Some tutorials generate plots (saved to disk)
- ✅ **Model Files**: Real-world examples save trained models

## 🎓 Learning Tips

### For Beginners
1. **Don't skip tutorials**: Each builds on previous concepts
2. **Run the code**: Don't just read—execute and experiment
3. **Modify and experiment**: Change hyperparameters and see what happens
4. **Read the comments**: They explain the "why" behind the code

### For Self-Study
- **Estimated time**: 2-3 hours for all beginner tutorials
- **Full course**: 4-5 hours total including advanced topics
- **Recommended pace**: 1-2 tutorials per day

### For Instructors
- Each tutorial is self-contained and can be assigned independently
- Tutorials include concept explanations, code, and practical examples
- Can be used for:
  - Lecture demonstrations
  - Lab assignments
  - Homework exercises
  - Project starting points

## 📝 Key Concepts Summary

### Loss Functions

| Loss Function | Use Case | Pros | Cons |
|---------------|----------|------|------|
| **MSE** | Regression | Smooth gradients | Sensitive to outliers |
| **MAE** | Regression | Robust to outliers | Non-smooth at zero |
| **Smooth L1** | Regression | Best of both | More hyperparameters |
| **BCE** | Binary Classification | Standard for binary | Needs probability input |
| **Cross-Entropy** | Multi-class | Standard for classification | Class imbalance issues |
| **Focal Loss** | Imbalanced Classification | Handles imbalance | More complex |
| **Dice Loss** | Segmentation | Measures overlap | Specific use case |

### Optimizers

| Optimizer | Learning Rate | Pros | Cons | Best For |
|-----------|---------------|------|------|----------|
| **SGD** | 0.01-0.1 | Simple, reliable | Slow convergence | Production models |
| **SGD+Momentum** | 0.01-0.1 | Better than vanilla SGD | Needs tuning | CNNs |
| **Adam** | 0.001 | Fast, adaptive | Can overfit | Prototyping |
| **AdamW** | 0.0001-0.001 | Better generalization | Higher memory | Transformers |
| **RMSprop** | 0.001 | Good for RNNs | Less popular now | RNNs |

### Learning Rate Schedulers

| Scheduler | When to Use | Behavior |
|-----------|-------------|----------|
| **StepLR** | Fixed training duration | Drops LR at intervals |
| **ExponentialLR** | Smooth decay | Exponential decrease |
| **CosineAnnealingLR** | Modern deep learning | Smooth cosine curve |
| **ReduceLROnPlateau** | Variable training length | Drops when stuck |

## 🔧 Common Issues & Solutions

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or use CPU
```python
device = torch.device('cpu')  # Force CPU
```

### Issue: Loss is NaN
**Solutions**:
- Learning rate too high → Reduce it (try 0.001 or 0.0001)
- Numerical instability → Check for log(0) or division by zero
- Gradient explosion → Use gradient clipping

### Issue: Loss not decreasing
**Solutions**:
- Learning rate too low → Increase it
- Wrong loss function → Check if it matches your task
- Model too simple → Try a more complex architecture
- Data issues → Check data normalization and labels

### Issue: Import errors
**Solution**: Install missing packages
```bash
pip install torch torchvision matplotlib
```

## 💡 Additional Resources

### Official Documentation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html)

### Recommended Papers
- **Adam**: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
- **Focal Loss**: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
- **AdamW**: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)

### Books
- "Deep Learning with PyTorch" by Eli Stevens et al.
- "Deep Learning" by Ian Goodfellow et al.

### Online Courses
- PyTorch official tutorials
- Fast.ai Practical Deep Learning
- Stanford CS231n (Convolutional Neural Networks)

## 🤝 Contributing

Found an issue or want to improve a tutorial? Contributions are welcome!

### How to Contribute
1. Fork the repository
2. Make your changes
3. Test the code
4. Submit a pull request

### Guidelines
- Keep code style consistent
- Add detailed comments
- Include explanations for complex concepts
- Test on Python 3.7+

## 📄 License

This tutorial package is provided for educational purposes. Feel free to use and modify for learning and teaching.

## 🙋 FAQ

**Q: Do I need a GPU?**  
A: No! All tutorials run fine on CPU. GPU will be faster for larger examples.

**Q: What Python version do I need?**  
A: Python 3.7 or higher is recommended.

**Q: How long does it take to complete?**  
A: Full package: 4-5 hours. Beginners can start with 2-3 hours for basics.

**Q: Can I use this for my course?**  
A: Absolutely! These tutorials are designed for teaching.

**Q: I'm stuck on a tutorial. Where can I get help?**  
A: 
1. Read the comments carefully
2. Check the "Common Issues" section
3. Review the official PyTorch documentation
4. Search for the error message online

**Q: How can I practice more?**  
A:
1. Modify hyperparameters in existing tutorials
2. Try different datasets
3. Implement variations of the models
4. Create your own custom losses for specific problems

## 📧 Contact & Feedback

If you have questions, suggestions, or feedback about these tutorials, please feel free to reach out or open an issue.

---

## 🎯 Next Steps After Completing Tutorials

1. **Practice Projects**:
   - Image classification on CIFAR-10
   - Sentiment analysis on text data
   - Time series prediction
   - Object detection

2. **Advanced Topics**:
   - Mixed precision training
   - Distributed training
   - Learning rate finder
   - Advanced augmentation

3. **Competitions**:
   - Kaggle competitions
   - Local ML challenges

4. **Research**:
   - Read recent papers
   - Implement new loss functions
   - Experiment with novel optimizers

---

**Happy Learning! 🚀**

*Last updated: 2025*
