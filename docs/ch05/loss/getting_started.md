# Getting Started Guide

Welcome to the PyTorch Loss and Optimizer Tutorial! This guide will help you get started quickly.

## 🚀 Installation (5 minutes)

### Step 1: Install Python
Make sure you have Python 3.7 or higher installed.

```bash
# Check your Python version
python --version
# or
python3 --version
```

### Step 2: Install PyTorch

**For CPU only** (recommended for learning):
```bash
pip install torch torchvision
```

**For CUDA (if you have an NVIDIA GPU)**:
Visit [pytorch.org](https://pytorch.org/) and follow the installation instructions for your system.

### Step 3: Install Optional Packages

```bash
pip install matplotlib numpy
```

Or install everything at once:
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## 📚 Your First Tutorial (10 minutes)

### 1. Navigate to the tutorial folder
```bash
cd pytorch_loss_optimizer_tutorial
```

### 2. Run your first tutorial
```bash
python 01_beginner/01_intro_to_loss_functions.py
```

### 3. What you'll see
- Explanations of loss functions
- Working code examples
- Output showing how losses work

---

## 🎯 Learning Path

### Week 1: Beginner Tutorials (2-3 hours)
Complete all tutorials in `01_beginner/`:
- [ ] 01_intro_to_loss_functions.py
- [ ] 02_regression_losses_comparison.py  
- [ ] 03_intro_to_optimizers.py
- [ ] 04_classification_losses.py

**What you'll learn**: Basics of loss functions, optimizers, and training loops

### Week 2: Intermediate Tutorials (2-3 hours)
Complete all tutorials in `02_intermediate/`:
- [ ] 01_optimizer_comparison.py
- [ ] 02_learning_rate_schedulers.py
- [ ] 03_gradient_management.py

**What you'll learn**: Advanced optimization techniques and best practices

### Week 3: Advanced Topics (2 hours)
Complete tutorials in `03_advanced/`:
- [ ] 01_custom_loss_functions.py

**What you'll learn**: Create custom losses for specific problems

### Week 4: Real-World Application (2-3 hours)
Complete `04_real_world_examples/`:
- [ ] 01_complete_mnist_training.py

**What you'll learn**: End-to-end ML pipeline

---

## 💡 Study Tips

### Active Learning
Don't just read the code—run it! Try:
1. **Running the code as-is** first
2. **Changing hyperparameters** and observing results
3. **Breaking things on purpose** to understand errors
4. **Adding print statements** to explore values

### Example Experiments
```python
# In 01_intro_to_loss_functions.py, try:

# 1. Change the predictions and see how loss changes
predicted_prices = torch.tensor([200.0, 300.0, 400.0, 500.0, 600.0])

# 2. Try different reduction methods
criterion = nn.MSELoss(reduction='sum')  # instead of 'mean'

# 3. Create intentionally wrong predictions
predicted_prices = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
```

### Taking Notes
Keep a learning journal:
- What did I learn today?
- What confused me?
- What should I practice more?

---

## 🔧 Common Issues

### "ImportError: No module named 'torch'"
**Solution**: Install PyTorch
```bash
pip install torch torchvision
```

### "CUDA out of memory"
**Solution**: You're trying to use GPU without enough memory. Use CPU instead:
```python
device = torch.device('cpu')
```

### Code runs but no output
**Solution**: Make sure you're running the file correctly:
```bash
python 01_beginner/01_intro_to_loss_functions.py
```
Not just:
```bash
python 01_intro_to_loss_functions.py
```

### Still stuck?
1. Read the error message carefully
2. Check the "Common Issues" section in README.md
3. Google the error message
4. Check PyTorch documentation

---

## 📖 Study Routine Suggestions

### 30-Minute Daily Study
- 10 min: Run one tutorial section
- 10 min: Modify and experiment
- 10 min: Review and take notes

### 2-Hour Weekend Session
- 30 min: Complete one full tutorial
- 30 min: Experiment with modifications
- 30 min: Try applying to a small project
- 30 min: Review and consolidate learning

### Group Study
- One person shares screen and runs code
- Discuss what each section does
- Everyone tries modifications
- Compare results

---

## 🎓 Next Steps After Completing Tutorials

### Immediate Practice
1. Modify the MNIST example for CIFAR-10
2. Create your own custom loss function
3. Try different optimizer combinations

### Projects to Build
1. **Image classifier** for a dataset you choose
2. **Regression model** for predicting house prices
3. **Custom loss function** for a specific problem

### Continue Learning
1. PyTorch official tutorials
2. Fast.ai course
3. Deep Learning specialization (Coursera)
4. Read research papers and implement them

---

## 📝 Quick Reference

### Running a Tutorial
```bash
cd pytorch_loss_optimizer_tutorial
python 01_beginner/01_intro_to_loss_functions.py
```

### Basic Training Loop Template
```python
# Setup
model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    for data, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

### Most Common Commands
```python
# Loss functions
criterion = nn.MSELoss()           # Regression
criterion = nn.CrossEntropyLoss()  # Classification

# Optimizers
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training step
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## ✅ Checklist: Are You Ready?

Before starting, make sure you have:
- [ ] Python 3.7+ installed
- [ ] PyTorch installed
- [ ] Downloaded this tutorial package
- [ ] 2-3 hours of dedicated study time
- [ ] A text editor (VS Code, PyCharm, or similar)
- [ ] Basic Python knowledge (variables, functions, loops)

**All checked?** Great! Start with `01_beginner/01_intro_to_loss_functions.py`

---

## 🎉 You're Ready!

Open your terminal, navigate to this folder, and run:

```bash
python 01_beginner/01_intro_to_loss_functions.py
```

**Happy learning!** 🚀

---

*Need help? Check QUICK_REFERENCE.md for code snippets and README.md for detailed information.*
