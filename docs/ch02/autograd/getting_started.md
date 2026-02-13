# 🚀 Getting Started - 5 Minute Quick Start

Welcome! This guide will get you up and running in 5 minutes.

## ✅ Prerequisites Checklist

- [ ] Python 3.7 or higher installed
- [ ] Basic understanding of Python
- [ ] Basic calculus knowledge (what is a derivative?)
- [ ] 5 minutes of your time!

## 📦 Installation (2 minutes)

### Step 1: Extract the tutorial package
You already have this if you're reading this file!

### Step 2: Create virtual environment
```bash
# Open terminal in the tutorial folder
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

You should see PyTorch and other packages installing. This may take 1-2 minutes.

## 🎯 Your First PyTorch Gradient (3 minutes)

### Run Example 1

```bash
cd beginner
python 01_basic_scalar_backward.py
```

### What You'll See

```
======================================================================
PART 1: Creating a Leaf Tensor with Gradient Tracking
======================================================================
x: tensor([ 1.5410, -0.2934, -2.1788], requires_grad=True)
x.grad (before backward): None

======================================================================
PART 2: Forward Pass - Building the Computational Graph
======================================================================
loss: tensor(7.2042, grad_fn=<SumBackward0>)

======================================================================
PART 3: Backward Pass - Computing Gradients
======================================================================
x.grad (after backward): tensor([ 3.0820, -0.5868, -4.3576])
Expected gradient (2*x): tensor([ 3.0820, -0.5868, -4.3576])
Match? True
```

### 🎉 Congratulations!

You just:
1. ✅ Created a tensor with gradient tracking
2. ✅ Performed a forward computation
3. ✅ Computed gradients automatically with `.backward()`
4. ✅ Verified the math was correct!

## 🎓 What Just Happened?

```python
# 1. Create tensor with gradient tracking
x = torch.tensor([1., 2., 3.], requires_grad=True)

# 2. Forward pass: compute loss
loss = (x ** 2).sum()  # loss = 1² + 2² + 3² = 14

# 3. Backward pass: compute gradients
loss.backward()  # Computes d(loss)/dx

# 4. Check gradients
print(x.grad)  # [2., 4., 6.] = 2*x
```

**The magic:** PyTorch automatically computed the derivative!
- Mathematical formula: d/dx(x²) = 2x
- PyTorch computed: [2×1, 2×2, 2×3] = [2, 4, 6]

## 🚶 Next Steps

### Option 1: Keep Going (Recommended)
```bash
python 02_gradient_accumulation.py
python 05_complete_linear_regression.py  # See a full training loop!
```

### Option 2: Read the README
Open `README.md` for the complete learning path

### Option 3: Jump to a Specific Topic
- **Beginner**: Basic autograd concepts
- **Intermediate**: Neural networks and training
- **Advanced**: Vector calculus and Jacobians

## 🆘 Troubleshooting

### "torch not found"
```bash
pip install torch
```

### "matplotlib not found"
```bash
pip install matplotlib
```

### Still having issues?
Check that:
1. Virtual environment is activated (you should see `(venv)` in terminal)
2. You're in the right directory
3. Python version is 3.7+ (`python --version`)

## 💡 Pro Tips

1. **Read the code comments** - They explain everything!
2. **Experiment** - Change numbers and see what happens
3. **Run multiple times** - Understanding comes from repetition
4. **Take notes** - Write down what surprises you

## 📚 Learning Path

```
Day 1-3:   Beginner files (01-10)
Day 4-7:   Intermediate files (11-21)  
Day 8-10:  Advanced files (22-27)
Day 11+:   Build your own projects!
```

## 🎯 Your Goal

By the end, you'll understand:
- How PyTorch computes gradients automatically
- How to train neural networks from scratch
- Common mistakes and how to avoid them
- Advanced techniques like vector-Jacobian products

---

<div align="center">

**Ready to learn? Let's go! 🚀**

*Next: Run `python 02_gradient_accumulation.py`*

</div>
