# PyTorch Gradient Computation Tutorial for Undergraduates

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

A comprehensive, hands-on tutorial package with **27 fully-commented Python scripts** teaching gradient computation in PyTorch, progressing from basics to advanced topics.

---

## 🎯 Overview

Learn automatic differentiation and gradient computation through practical examples, from simple scalar backpropagation to advanced vector-Jacobian products.

### What You'll Learn

✅ Autograd fundamentals and computational graphs  
✅ Gradient accumulation and proper management  
✅ Complete training loops from scratch  
✅ Parameter freezing and transfer learning  
✅ Vector-Jacobian products and full Jacobians  
✅ Memory-efficient training techniques  
✅ Common pitfalls and debugging strategies  

---

## 📁 Package Structure

```
pytorch_gradient_tutorial/
│
├── beginner/           # 🟢 Level 1: Fundamentals (10 files)
├── intermediate/       # 🟡 Level 2: Practical Techniques (11 files)  
├── advanced/           # 🔴 Level 3: Advanced Topics (6 files)
├── exercises/          # 💪 Practice Problems (2 files)
│
├── README.md           # This file
├── GETTING_STARTED.md  # Quick 5-minute start
├── requirements.txt    # Dependencies
└── LICENSE             # MIT License
```

---

## 🟢 Beginner Level (10 files)

**Start here!** Learn the fundamentals of PyTorch's autograd system.

| # | File | Topic | Key Concepts |
|---|------|-------|--------------|
| 01 | `01_basic_scalar_backward.py` | Basic scalar backward | Leaf tensors, .backward(), .grad |
| 02 | `02_gradient_accumulation.py` | Gradient accumulation | .zero_(), why gradients accumulate |
| 03 | `03_retain_graph.py` | Computational graph | retain_graph, graph lifecycle |
| 04 | `04_mini_training_step.py` | First training step | Forward-backward-update cycle |
| 05 | `05_complete_linear_regression.py` | **Complete training loop** ⭐ | Full workflow, visualization |
| 06 | `06_linear_regression_with_module.py` | Using nn.Module | Building blocks |
| 07 | `07_when_to_stop_tracking_grads.py` | Gradient tracking control | Efficiency |
| 08 | `08_turn_off_gradient_tracking.py` | .requires_grad manipulation | Dynamic control |
| 09 | `09_detach_vs_detach_.py` | Detaching tensors | Memory management |
| 10 | `10_with_torch_no_grad.py` | no_grad context | Inference mode |

**Learning Time:** 1-2 weeks  
**Prerequisites:** Basic Python, basic calculus

---

## 🟡 Intermediate Level (11 files)

Build on fundamentals with practical neural network techniques.

| # | File | Topic | Key Concepts |
|---|------|-------|--------------|
| 11 | `11_optimizer_like_updates.py` | Optimizer patterns | SGD implementation |
| 12 | `12_converting_to_numpy.py` | Tensor conversions | .detach(), .cpu(), .numpy() |
| 13 | `13_print_layers.py` | Model inspection | Architecture viewing |
| 14 | `14_layer_naming_with_ordereddict.py` | Layer organization | OrderedDict patterns |
| 15 | `15_layer_naming_with_custom_module.py` | Custom modules | Named submodules |
| 16 | `16_freeze_and_unfreeze_parameters.py` | **Parameter freezing** ⭐ | Transfer learning |
| 17 | `17_pitfalls_and_sanity_checks.py` | **Debugging guide** ⭐ | Common errors |
| 18 | `18_three_ways_to_zero_gradients.py` | Gradient zeroing | Performance comparison |
| 19 | `19_gradient_accumulation_for_large_batches.py` | **Large batch training** | Memory efficiency |
| 20 | `20_why_gradient_scaling_matters.py` | Numerical stability | Gradient scaling |
| 21 | `21_training_loop_with_grad_accumulation.py` | Complete example | Putting it together |

**Learning Time:** 2-3 weeks  
**Prerequisites:** Complete beginner level, understand neural networks

---

## 🔴 Advanced Level (6 files)

Master advanced gradient computation techniques.

| # | File | Topic | Key Concepts |
|---|------|-------|--------------|
| 22 | `22_vector_backward_with_v.py` | Vector-Jacobian products | VJP introduction |
| 23 | `23_vector_backward_linear_case_1.py` | Linear VJP examples | Matrix calculus |
| 24 | `24_vector_backward_linear_case_2.py` | More linear cases | Jacobian patterns |
| 25 | `25_vector_backward_nonlinear_full_jacobian.py` | **Full Jacobian** ⭐ | Complete computation |
| 26 | `26_pitfalls_of_vector_backward.py` | VJP debugging | Common mistakes |
| 27 | `27_batch_vjp.py` | Batched operations | Efficient batch VJP |

**Learning Time:** 1-2 weeks  
**Prerequisites:** Linear algebra, vector calculus, complete intermediate

---

## 💪 Exercises (2 files)

Test your understanding with hands-on challenges.

1. **`exercise_01_implement_sgd.py`**  
   Build your own SGD optimizer from scratch

2. **`exercise_02_mini_batch_training.py`**  
   Implement mini-batch training and compare batch sizes

---

## 💻 Installation

### Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} ready!')"
```

### Requirements

- Python 3.7+
- PyTorch 1.10+
- matplotlib, numpy (automatically installed)

---

## 🚀 Quick Start

### Run Your First Example (2 minutes)

```bash
cd beginner
python 01_basic_scalar_backward.py
```

### See a Complete Training Loop (5 minutes)

```bash
python 05_complete_linear_regression.py
```

This generates visualizations showing:
- Data points and fitted line
- Training loss curve over time

---

## 🗺️ Learning Path

```
Week 1-2:   Beginner (01-10)
            ↓
Week 3-4:   Intermediate (11-21)
            ↓
Week 5-6:   Advanced (22-27)
            ↓
Week 7+:    Exercises & Projects
```

**Total Time:** 4-6 weeks (3-5 hours/week)

---

## 🔑 Key Concepts

### 1. Leaf Tensors
```python
x = torch.randn(3, requires_grad=True)  # Leaf
y = x * 2  # Non-leaf
```

### 2. Computational Graph
```python
x → [*2] → y → [**2] → z → [sum] → loss
```

### 3. Gradient Accumulation
```python
loss.backward()  # Adds to existing .grad!
optimizer.zero_grad()  # Must clear first
```

### 4. Vector-Jacobian Product
```python
y.backward(v)  # Computes x.grad = v^T * (dy/dx)
```

---

## ✅ Best Practices

**1. Always zero gradients**
```python
optimizer.zero_grad()  # Before backward!
loss.backward()
optimizer.step()
```

**2. Use no_grad for inference**
```python
with torch.no_grad():
    predictions = model(x_test)
```

**3. Monitor gradient norms**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## ⚠️ Common Pitfalls

### ❌ Forgetting to Zero Gradients
```python
# WRONG - gradients accumulate!
for epoch in range(epochs):
    loss = compute_loss()
    loss.backward()
    optimizer.step()

# CORRECT
for epoch in range(epochs):
    optimizer.zero_grad()  # ← Add this!
    loss = compute_loss()
    loss.backward()
    optimizer.step()
```

### ❌ In-place Operations on Leaf Tensors
```python
# WRONG
x = torch.tensor([1., 2.], requires_grad=True)
x += 1  # Breaks autograd!

# CORRECT
x = x + 1  # Creates new tensor
```

### ❌ Calling backward() on Non-Scalar
```python
# WRONG
y = model(x)  # Shape: (batch, 10)
y.backward()  # Error!

# CORRECT
loss = y.sum()
loss.backward()
```

---

## 📚 How to Use This Tutorial

1. **Start with beginner/** - Foundation is critical
2. **Run each script** - Don't just read, execute!
3. **Read all comments** - They explain the "why"
4. **Experiment** - Change parameters, break things
5. **Do exercises** - Test your understanding
6. **Progress sequentially** - Each builds on previous

---

## 💡 Tips for Success

📝 **Take notes** while running examples  
🔬 **Experiment** with different values  
🐛 **Debug** intentionally broken code  
👥 **Discuss** with study groups  
🔄 **Review** regularly  

---

## 📊 File Naming Convention

All files use consistent numbering:

- **01-10:** Beginner
- **11-21:** Intermediate  
- **22-27:** Advanced

This makes the learning progression clear and easy to follow!

---

## 🎓 Learning Outcomes

By completing this tutorial, you will:

✅ Understand how autograd works internally  
✅ Write training loops from scratch  
✅ Debug gradient-related issues confidently  
✅ Apply transfer learning techniques  
✅ Compute Jacobians for vector-valued functions  
✅ Optimize memory usage in training  

---

## 📚 Additional Resources

- [PyTorch Autograd Docs](https://pytorch.org/docs/stable/notes/autograd.html)
- [Dive into Deep Learning](https://d2l.ai/) - Interactive book
- Andrej Karpathy's "Neural Networks: Zero to Hero"

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- The open-source ML community
- All students who provided feedback

---

<div align="center">

**Happy Learning! 🚀**

*Total: 27 Python scripts + 2 exercises + comprehensive documentation*

**Made with ❤️ for undergraduate students learning PyTorch**

</div>
