# PyTorch Logistic Regression Tutorial

A comprehensive, progressive tutorial series on logistic regression using PyTorch, designed for undergraduate students. This tutorial takes you from basic concepts to advanced applications with fully commented code.

## 📚 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Tutorial Structure](#tutorial-structure)
- [Quick Start](#quick-start)
- [Learning Path](#learning-path)
- [Additional Resources](#additional-resources)

---

## 🎯 Overview

This tutorial series covers logistic regression in PyTorch through 4 progressive levels:

1. **Basics** - Fundamental concepts and simple implementations
2. **Intermediate** - Best practices and proper training patterns
3. **Advanced** - Custom datasets, multi-class classification, and regularization
4. **Applications** - Real-world examples including text and medical data

Each script is:
- ✅ Fully commented with detailed explanations
- ✅ Self-contained and runnable independently
- ✅ Progressively more challenging
- ✅ Includes shape annotations and debugging tips

---

## 📋 Prerequisites

### Required Knowledge
- Basic Python programming (variables, functions, loops)
- Elementary linear algebra (vectors, matrices, dot products)
- Basic calculus concepts (derivatives, gradients)
- Understanding of binary classification problems

### Optional but Helpful
- Familiarity with NumPy
- Basic understanding of neural networks
- Experience with scikit-learn

---

## 🔧 Installation

### Step 1: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv pytorch_env

# Activate it
# On Windows:
pytorch_env\Scripts\activate
# On macOS/Linux:
source pytorch_env/bin/activate
```

### Step 2: Install Required Packages

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn
```

### Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## 📁 Tutorial Structure

```
pytorch_logistic_regression_tutorial/
│
├── README.md                          # This file
│
├── 01_basics/                         # Level 1: Fundamentals
│   ├── 01_introduction.py             # PyTorch basics and tensors
│   ├── 02_simple_binary_classification.py  # First logistic regression
│   ├── 03_with_sklearn_data.py        # Using real datasets
│   ├── 04_gradient_descent_with_BCE.py # Detailed BCE implementation
│   ├── 05_gradient_descent_with_BCEWithLogitsLoss.py # Numerical stability
│   ├── 06_gradient_descent_advanced.py # Advanced patterns
│   ├── 07_sklearn_comparison.py       # PyTorch vs sklearn
│   └── 08_imdb_text_classification.py # Text classification intro
│
├── 02_intermediate/                   # Level 2: Best Practices
│   ├── 01_proper_training_loop.py     # Standard training patterns
│   ├── 02_train_val_test_split.py     # Proper data splitting
│   ├── 03_with_dataloader.py          # Efficient batch processing
│   ├── 04_model_checkpointing.py      # Saving and loading models
│   └── 05_early_stopping.py           # Preventing overfitting
│
├── 03_advanced/                       # Level 3: Advanced Techniques
│   ├── 01_custom_dataset.py           # Creating custom Dataset classes
│   ├── 02_multiclass_classification.py # Softmax and cross-entropy
│   ├── 03_regularization.py           # L1, L2, and Dropout
│   ├── 04_learning_rate_scheduling.py # Adaptive learning rates
│   └── 05_class_imbalance.py          # Handling imbalanced data
│
├── 04_applications/                   # Level 4: Real-World Projects
│   ├── 01_breast_cancer_diagnosis.py  # Medical diagnosis
│   ├── 02_sentiment_analysis.py       # Text classification
│   ├── 03_customer_churn_prediction.py # Business application
│   └── 04_fraud_detection.py          # Imbalanced classification
│
└── datasets/                          # Sample datasets
    └── README.md                      # Dataset descriptions
```

---

## 🚀 Quick Start

### Option 1: Run Individual Scripts

```bash
# Start with the basics
python 01_basics/01_introduction.py

# Move to intermediate
python 02_intermediate/01_proper_training_loop.py

# Try advanced topics
python 03_advanced/01_custom_dataset.py

# Explore applications
python 04_applications/01_breast_cancer_diagnosis.py
```

### Option 2: Interactive Learning

Open each file in your favorite editor/IDE and:
1. Read the comments carefully
2. Run the code section by section
3. Modify parameters and observe changes
4. Complete the exercises (marked with `# TODO` or `# Exercise`)

---

## 🎓 Learning Path

### Week 1-2: Basics (01_basics/)
**Goal**: Understand fundamental concepts

**Core Tutorials (Complete in order):**

1. **01_introduction.py** (~30 mins)
   - Learn PyTorch tensor basics
   - Understand autograd (automatic differentiation)
   - First simple gradient descent example

2. **02_simple_binary_classification.py** (~45 mins)
   - Implement basic logistic regression
   - Understand sigmoid function
   - Train your first classifier

3. **03_with_sklearn_data.py** (~1 hour)
   - Work with real datasets
   - Learn data preprocessing
   - Evaluate model performance

**Advanced References (Optional deep dives):**

4. **04_gradient_descent_with_BCE.py** (~45 mins)
   - Detailed BCE loss implementation
   - Comprehensive shape annotations
   - Production code patterns

5. **05_gradient_descent_with_BCEWithLogitsLoss.py** (~45 mins)
   - Understand numerical stability
   - Learn BCEWithLogitsLoss
   - Compare with standard BCE

6. **06_gradient_descent_advanced.py** (~1 hour)
   - Advanced optimization patterns
   - Complex training scenarios

7. **07_sklearn_comparison.py** (~45 mins)
   - PyTorch vs scikit-learn
   - When to use each framework
   - Performance comparison

8. **08_imdb_text_classification.py** (~1.5 hours)
   - Introduction to text classification
   - Word embeddings basics
   - NLP with logistic regression

**Checkpoint**: You should be able to train a basic logistic regression model and understand the training loop.

---

### Week 3-4: Intermediate (02_intermediate/)
**Goal**: Learn PyTorch best practices

1. **01_proper_training_loop.py** (~1 hour)
   - Structure training code properly
   - Understand train vs eval mode
   - Implement proper gradient handling

2. **02_train_val_test_split.py** (~1 hour)
   - Learn proper data splitting
   - Implement validation loop
   - Understand overfitting vs underfitting

3. **03_with_dataloader.py** (~1.5 hours)
   - Use PyTorch DataLoader
   - Implement batch processing
   - Understand shuffling and sampling

4. **04_model_checkpointing.py** (~1 hour)
   - Save and load models
   - Implement checkpoint strategies
   - Resume training from checkpoints

5. **05_early_stopping.py** (~1 hour)
   - Implement early stopping
   - Prevent overfitting
   - Choose optimal stopping point

**Checkpoint**: You should be able to structure a complete training pipeline with proper validation and model saving.

---

### Week 5-6: Advanced (03_advanced/)
**Goal**: Master advanced techniques

1. **01_custom_dataset.py** (~2 hours)
   - Create custom Dataset classes
   - Implement `__len__` and `__getitem__`
   - Handle different data formats

2. **02_multiclass_classification.py** (~1.5 hours)
   - Extend to multi-class problems
   - Understand softmax activation
   - Use CrossEntropyLoss

3. **03_regularization.py** (~2 hours)
   - Implement L1 and L2 regularization
   - Add dropout layers
   - Compare regularization techniques

4. **04_learning_rate_scheduling.py** (~1.5 hours)
   - Use learning rate schedulers
   - Compare different schedules
   - Understand adaptive learning rates

5. **05_class_imbalance.py** (~2 hours)
   - Handle imbalanced datasets
   - Use weighted loss functions
   - Implement oversampling/undersampling

**Checkpoint**: You should be able to handle complex datasets and implement advanced training strategies.

---

### Week 7-8: Applications (04_applications/)
**Goal**: Apply knowledge to real-world problems

1. **01_breast_cancer_diagnosis.py** (~2 hours)
   - Medical diagnosis pipeline
   - Feature importance analysis
   - Model evaluation metrics

2. **02_sentiment_analysis.py** (~3 hours)
   - Text classification basics
   - Feature extraction from text
   - Bag-of-words approach

3. **03_customer_churn_prediction.py** (~2 hours)
   - Business analytics application
   - Handle mixed data types
   - Interpret business metrics

4. **04_fraud_detection.py** (~2.5 hours)
   - Extreme class imbalance
   - Precision-recall tradeoffs
   - Anomaly detection patterns

**Checkpoint**: You should be able to tackle real-world classification problems from data loading to deployment-ready models.

---

## 💡 Key Concepts by Level

### Level 1 (Basics)
- Tensors and operations
- Automatic differentiation (autograd)
- Sigmoid function
- Binary Cross-Entropy loss
- Basic gradient descent
- Model evaluation (accuracy)

### Level 2 (Intermediate)
- Training vs evaluation mode
- DataLoader and batching
- Train/validation/test splits
- Model checkpointing
- Early stopping
- Learning curves

### Level 3 (Advanced)
- Custom Dataset classes
- Multi-class classification
- Softmax and cross-entropy
- Regularization (L1, L2, Dropout)
- Learning rate scheduling
- Class imbalance handling

### Level 4 (Applications)
- Domain-specific preprocessing
- Feature engineering
- Model interpretation
- Performance metrics (F1, ROC-AUC, Precision-Recall)
- Handling real-world data issues
- Production considerations

---

## 🔍 Common Issues and Solutions

### Issue 1: CUDA out of memory
```python
# Solution: Reduce batch size or move to CPU
device = torch.device('cpu')  # Force CPU usage
# OR
batch_size = 16  # Reduce from 64
```

### Issue 2: Model not learning (loss not decreasing)
```python
# Check these:
1. Is learning rate too small? Try: lr = 0.01 or lr = 0.1
2. Is learning rate too large? Try: lr = 0.001 or lr = 0.0001
3. Are you calling optimizer.zero_grad()?
4. Is data properly normalized?
```

### Issue 3: Perfect training accuracy but poor test accuracy
```python
# This is overfitting. Solutions:
1. Add regularization (L2, dropout)
2. Reduce model complexity
3. Get more training data
4. Use early stopping
```

### Issue 4: Loss becomes NaN
```python
# Solutions:
1. Reduce learning rate
2. Use BCEWithLogitsLoss instead of BCE + Sigmoid
3. Check for invalid data (inf, nan)
4. Clip gradients: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 📊 Expected Learning Outcomes

After completing this tutorial, you will be able to:

✅ Implement logistic regression from scratch in PyTorch  
✅ Understand the complete training pipeline  
✅ Handle real-world datasets and preprocessing  
✅ Implement custom Dataset classes  
✅ Apply regularization and optimization techniques  
✅ Handle class imbalance problems  
✅ Evaluate models using multiple metrics  
✅ Save, load, and deploy models  
✅ Debug common training issues  
✅ Apply logistic regression to various domains  

---

## 🎯 Practice Exercises

Each level includes practice exercises:

### Basics Exercises
1. Modify the learning rate and observe convergence
2. Change the dataset and retrain
3. Implement momentum in SGD
4. Plot training loss curves

### Intermediate Exercises
1. Implement k-fold cross-validation
2. Add tensorboard logging
3. Create custom data augmentation
4. Implement gradient clipping

### Advanced Exercises
1. Implement focal loss for imbalanced data
2. Create an ensemble of models
3. Add adversarial training
4. Implement mixup augmentation

### Application Exercises
1. Apply to a new dataset from Kaggle
2. Create a REST API for model inference
3. Implement model interpretability (LIME/SHAP)
4. Deploy using ONNX or TorchScript

---

## 📚 Additional Resources

### Official Documentation
- [PyTorch Official Docs](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Forums](https://discuss.pytorch.org/)

### Recommended Reading
- "Deep Learning with PyTorch" by Eli Stevens et al.
- "Dive into Deep Learning" - Interactive book: d2l.ai
- "Neural Networks and Deep Learning" by Michael Nielsen

### Video Tutorials
- PyTorch Official YouTube Channel
- Sentdex PyTorch Tutorials
- DeepLearning.AI courses on Coursera

### Practice Datasets
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Google Dataset Search](https://datasetsearch.research.google.com/)

---

## 🤝 Contributing

Found a bug or have suggestions? This is a learning resource and feedback is welcome!

Common improvements:
- Additional examples
- Better explanations
- More exercises
- Bug fixes
- Performance optimizations

---

## 📝 Notes for Instructors

### Using in Classroom
- Each level takes approximately 2 weeks of study
- Can be used for a full semester course
- Exercises can be assigned as homework
- Applications can be final projects

### Customization
- Scripts are modular and can be rearranged
- Easy to add domain-specific examples
- Comments can be translated to other languages
- Difficulty can be adjusted by adding/removing sections

### Assessment Ideas
- Quiz on key concepts from each level
- Coding assignments based on exercises
- Final project using application templates
- Code review sessions

---

## ⚖️ License

This tutorial is provided for educational purposes. Feel free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgments

- Based on PyTorch official documentation and tutorials
- Inspired by Patrick Loeber's PyTorch tutorials
- Dataset sources: scikit-learn, UCI ML Repository, Kaggle

---

## 📧 Support

If you get stuck:
1. Read the comments in the code carefully
2. Check the Common Issues section
3. Consult PyTorch documentation
4. Search PyTorch forums
5. Review the official PyTorch tutorials

---

**Happy Learning! 🚀**

*Remember: The best way to learn is by doing. Run the code, break it, fix it, and experiment!*
