# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install torch numpy matplotlib scikit-learn
```

### 2. Run Your First Tutorial

```bash
python 01_pytorch_basics.py
```

This will introduce you to PyTorch tensors and automatic differentiation.

### 3. Follow the Learning Path

Run tutorials in order (01 → 10):

**Beginners (Start Here):**
- `01_pytorch_basics.py` - Learn PyTorch fundamentals
- `02_linear_regression_numpy.py` - Understand the math
- `03_linear_regression_manual_pytorch.py` - Bridge to PyTorch

**Intermediate:**
- `04_linear_regression_autograd.py` - Automatic gradients
- `05_linear_regression_nn_module.py` - Proper PyTorch models
- `06_multivariate_regression.py` - Real-world data

**Advanced:**
- `07_polynomial_regression.py` - Non-linear relationships
- `08_regularization.py` - Prevent overfitting
- `09_mini_batch_training.py` - Efficient training
- `10_complete_pipeline.py` - Production-ready pipeline

### 4. Experiment!

- Modify hyperparameters (learning rate, batch size, etc.)
- Try different optimizers (SGD, Adam, RMSprop)
- Add more layers to models
- Use your own datasets

## 📖 File Structure

```
pytorch_linear_regression_tutorial/
├── README.md                          # Full documentation
├── QUICK_START.md                     # This file
├── requirements.txt                   # Dependencies
├── 01_pytorch_basics.py               # Start here!
├── 02_linear_regression_numpy.py
├── 03_linear_regression_manual_pytorch.py
├── 04_linear_regression_autograd.py
├── 05_linear_regression_nn_module.py
├── 06_multivariate_regression.py
├── 07_polynomial_regression.py
├── 08_regularization.py
├── 09_mini_batch_training.py
└── 10_complete_pipeline.py           # Complete example
```

## 🎯 What You'll Learn

- ✅ PyTorch tensor operations
- ✅ Automatic differentiation (autograd)
- ✅ Building neural network models
- ✅ Training loops and optimization
- ✅ Data loading and preprocessing
- ✅ Regularization techniques
- ✅ Model evaluation and visualization
- ✅ Production-ready ML pipelines

## 💡 Tips

1. **Read the comments** - Every line is explained
2. **Run the code** - Don't just read, execute!
3. **Experiment** - Change values and see what happens
4. **Take breaks** - Each tutorial is 15-40 minutes
5. **Ask questions** - Use comments to guide your learning

## 🆘 Common Issues

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch
```

### "RuntimeError: Expected all tensors to be on the same device"
Add `.to(device)` to your tensors, where device is 'cpu' or 'cuda'

### "Loss is NaN"
Lower your learning rate or check for numerical instability

## 🎓 After Completing

You'll be ready for:
- Building custom neural networks
- Computer vision with CNNs
- NLP with RNNs/Transformers
- Kaggle competitions
- Research projects
- Production ML systems

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Book](https://www.deeplearningbook.org/)

---

**Happy Learning! 🚀**

Start with `01_pytorch_basics.py` and work your way up!
