# Level 1: PyTorch Basics

## 🎯 Purpose

Master PyTorch's core features: automatic differentiation (autograd), the nn.Module pattern, and optimizer usage. These are the building blocks of every PyTorch model you'll ever write.

## 📚 What You'll Master

- **Autograd**: PyTorch's automatic differentiation engine
- **nn.Module**: The standard way to define models
- **Optimizers**: Adam, SGD, and how they differ
- **Training Loops**: The standard pattern for training any model

## 📖 Files in This Level

### 04_autograd_introduction.py ⭐⭐
**Difficulty**: Intermediate | **Time**: 45-60 min

Deep dive into PyTorch's automatic differentiation system.

**What You'll Learn**:
- How `.backward()` computes gradients
- The computational graph PyTorch builds
- `requires_grad` and when to use it
- `.grad` attribute and gradient accumulation
- `torch.no_grad()` context manager

**Key Concepts**:
```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()  # Compute dy/dx
print(x.grad)  # Prints: tensor([4.]) because dy/dx = 2x
```

**Why It Matters**: Autograd is what makes PyTorch powerful. Understanding it helps you debug and build custom models.

---

### 05_simple_perceptron.py ⭐
**Difficulty**: Beginner | **Time**: 30-45 min

Build the simplest possible neural network: a single neuron (perceptron).

**What You'll Learn**:
- `nn.Linear` layer (fully connected layer)
- Training loop structure
- How to use loss functions
- How to use optimizers

**Architecture**:
```
Input (1) → Linear → Output (1)
```

**Goal**: Learn `y = 2x + 3` from data

**Code Pattern** (you'll use this everywhere):
```python
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Why It Matters**: This is the template for ALL PyTorch training. Master this pattern!

---

### 06_two_layer_network.py ⭐⭐
**Difficulty**: Intermediate | **Time**: 45-60 min

Extend to a multi-layer network with non-linear activation.

**What You'll Learn**:
- Stacking multiple layers
- Why activation functions are necessary
- ReLU activation function
- Network capacity and expressiveness

**Architecture**:
```
Input → Linear(1→10) → ReLU → Linear(10→1) → Output
```

**Key Insight**: Without ReLU (or another non-linearity), multiple layers collapse to a single linear transformation!

**Experiments to Try**:
- Remove ReLU, see what happens
- Change hidden size (5, 50, 100)
- Add more hidden layers

**Why It Matters**: Understanding layer composition is crucial for designing deeper networks.

---

### 07_nn_module_and_optimizers.py ⭐⭐⭐
**Difficulty**: Intermediate | **Time**: 60-90 min

Learn the standard way to define PyTorch models and compare different optimizers.

**What You'll Learn**:
- Creating custom `nn.Module` subclasses
- The `__init__` and `forward` methods
- Different optimizers: SGD, Adam, RMSprop
- How learning rate affects training

**Custom Module Pattern**:
```python
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x
```

**Optimizer Comparison**:
| Optimizer | Speed | Stability | When to Use |
|-----------|-------|-----------|-------------|
| SGD | Slow | Stable | Small datasets, simple problems |
| Adam | Fast | Can be unstable | Most deep learning (default choice) |
| RMSprop | Medium | Stable | RNNs, online learning |

**Why It Matters**: This is how you'll write 99% of your models. The `nn.Module` pattern is fundamental.

## 🎓 Learning Path

```
04_autograd_introduction.py (Required)
    ↓
    Understand: How gradients are computed
    ↓
05_simple_perceptron.py (Required)
    ↓
    Learn: Basic training loop
    ↓
06_two_layer_network.py (Required)
    ↓
    Understand: Multi-layer networks
    ↓
07_nn_module_and_optimizers.py (Required)
    ↓
    Master: The standard PyTorch pattern
    ↓
Ready for Level 2! 🎉
```

## 💡 Study Tips

### For File 04 (Autograd):
- Print the computational graph using `torchviz`
- Try `.backward()` on different operations
- Understand gradient accumulation (run `.backward()` twice)
- Learn when to use `detach()` and `no_grad()`

### For File 05 (Perceptron):
- Monitor the loss - it should decrease
- Print model parameters before and after training
- Try different learning rates (0.1, 0.01, 0.001)

### For File 06 (Two Layers):
- Visualize the decision boundary
- Compare with vs without ReLU
- Try different hidden sizes

### For File 07 (nn.Module):
- Compare convergence speed of different optimizers
- Try different learning rates for each optimizer
- Understand why Adam usually works best

## 🔧 Common Issues & Solutions

**"Loss is not decreasing!"**
→ Learning rate too high or too low. Try 0.001 first.

**"RuntimeError: grad can be implicitly created only for scalar outputs"**
→ Your loss isn't a single number. Use `loss.mean()` or `loss.sum()`.

**"Gradients are all zero!"**
→ Did you call `optimizer.zero_grad()`? You need to clear old gradients.

**"Loss decreases then explodes!"**
→ Learning rate too high. Lower it by 10x.

**"What's the difference between .backward() and optimizer.step()?"**
→ `.backward()` computes gradients. `.step()` updates the weights using those gradients.

## 🧪 Experiments to Try

1. **Learning Rate Sweep**: Try [0.0001, 0.001, 0.01, 0.1, 1.0] and plot loss curves
2. **Optimizer Comparison**: Train same network with SGD, Adam, RMSprop
3. **Network Depth**: Try 1, 2, 3, 5 hidden layers
4. **Network Width**: Try hidden sizes of 5, 10, 50, 100, 500
5. **Activation Functions**: Compare ReLU, Sigmoid, Tanh, LeakyReLU

## ✅ Level 1 Completion Checklist

Before moving to Level 2, make sure you can:

- [ ] Explain how autograd builds and traverses the computation graph
- [ ] Write a complete training loop from scratch
- [ ] Create a custom nn.Module class
- [ ] Choose appropriate optimizer for a problem
- [ ] Debug common gradient-related errors
- [ ] Understand the purpose of `.zero_grad()`
- [ ] Explain why we need non-linear activation functions

## 🎯 Next Level Preview

**Level 2: Building Neural Networks** will teach you:
- Training on real datasets (MNIST)
- `nn.Sequential` for quick prototyping
- Different activation and loss functions
- Data loading and preprocessing
- Model evaluation and metrics

---

**Great progress!** You now understand PyTorch's core mechanics. Time to build real networks! 🚀

*Pro tip: File 07 (nn.Module pattern) is the most important. Make sure you really understand it!*
