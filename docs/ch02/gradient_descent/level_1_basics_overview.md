# Level 1: Basics - Understanding Gradient Descent 🌱

Welcome to Level 1! This is where your gradient descent journey begins.

## Learning Objectives

By completing this level, you will:
- ✅ Understand gradient descent from first principles
- ✅ Manually implement gradient descent using NumPy
- ✅ Master PyTorch's automatic differentiation (autograd)
- ✅ Build and train a simple linear regression model
- ✅ Visualize the optimization process

## Time Estimate
**2-3 hours** (including experimentation)

## Prerequisites
- Basic Python programming
- Elementary calculus (what is a derivative?)
- Basic linear algebra (vectors, matrices)

## Examples in This Level

### 01_manual_gradient_numpy.py ⭐
**What you'll learn:**
- What is gradient descent?
- How to compute gradients manually
- The core algorithm: parameter update rule
- Why learning rate matters

**Key concepts:** Gradient, learning rate, loss function, convergence

**Run it:**
```bash
python 01_manual_gradient_numpy.py
```

### 02_pytorch_autograd_basics.py ⭐
**What you'll learn:**
- PyTorch's automatic differentiation
- Understanding computational graphs
- `requires_grad`, `.backward()`, `.grad`
- Why autograd is powerful

**Key concepts:** Autograd, computational graph, backpropagation

**Run it:**
```bash
python 02_pytorch_autograd_basics.py
```

### 03_simple_linear_regression.py ⭐
**What you'll learn:**
- Using `nn.Module` to define models
- Standard training loop pattern
- Evaluation and metrics
- Model saving/loading

**Key concepts:** `nn.Module`, optimizer, training loop, evaluation

**Run it:**
```bash
python 03_simple_linear_regression.py
```

### 04_visualizing_gradient_descent.py ⭐
**What you'll learn:**
- Visualizing loss landscapes
- Watching parameters converge
- Understanding local minima
- Effects of learning rate

**Key concepts:** Loss surface, optimization trajectory, convergence

**Run it:**
```bash
python 04_visualizing_gradient_descent.py
```

## Suggested Learning Path

1. **Start with 01** - Build intuition with manual implementation
2. **Move to 02** - See how PyTorch automates this
3. **Practice with 03** - Apply to a realistic problem
4. **Visualize with 04** - See everything in action

## Quick Reference

### The Gradient Descent Algorithm
```python
# Initialize parameters
w = initial_value

# Repeat until convergence:
for epoch in range(n_epochs):
    # 1. Compute predictions
    y_pred = model(X, w)
    
    # 2. Compute loss
    loss = compute_loss(y_pred, y_true)
    
    # 3. Compute gradient
    gradient = compute_gradient(loss, w)
    
    # 4. Update parameters
    w = w - learning_rate * gradient
```

### PyTorch Training Template
```python
# Setup
model = MyModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(n_epochs):
    # Forward
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Common Issues & Solutions

### Issue: Loss not decreasing
**Possible causes:**
- Learning rate too large (try 0.01, 0.001)
- Wrong gradient direction (check signs)
- Bug in model or loss function

### Issue: Loss exploding (NaN)
**Possible causes:**
- Learning rate too large
- Gradient explosion
- Numerical instability

**Solution:** Reduce learning rate by 10x

### Issue: Slow convergence
**Possible causes:**
- Learning rate too small
- Bad parameter initialization
- Need more epochs

**Solution:** Increase learning rate, try different initialization

## Tips for Success

1. **Start simple** - Make sure basic examples work before moving on
2. **Experiment** - Change hyperparameters and observe effects
3. **Visualize** - Plot loss curves, parameter trajectories
4. **Debug** - Print intermediate values to understand flow
5. **Ask "why?"** - Understand each line of code

## Key Formulas

### Gradient Descent Update
```
w_new = w_old - learning_rate * gradient
```

### Mean Squared Error (MSE)
```
L(w) = (1/N) * Σ(y_pred - y_true)²
```

### Gradient of MSE (for y = wx)
```
dL/dw = (2/N) * Σ x * (y_pred - y_true)
```

## Assessment

Before moving to Level 2, make sure you can:
- [ ] Explain gradient descent in your own words
- [ ] Implement gradient descent manually (without looking)
- [ ] Use PyTorch autograd correctly
- [ ] Write a complete training loop
- [ ] Interpret loss curves
- [ ] Debug common training issues

## Next Steps

Once you're comfortable with these basics, proceed to:
**Level 2: Intermediate** - Learn about mini-batches, momentum, and learning rate schedules

---

**Questions? Issues?**
- Review the code comments carefully
- Try the experiment suggestions
- Compare your output with expected results
- Reach out for help if stuck!

Good luck! 🚀
