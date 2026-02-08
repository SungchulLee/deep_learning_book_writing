# Learning Rate and Step Size

## Introduction

The **learning rate** (often denoted $\eta$, $\alpha$, or `lr`) is arguably the most important hyperparameter in gradient-based optimization. It controls the magnitude of parameter updates and profoundly affects both the speed of convergence and whether convergence happens at all.

This chapter explores the learning rate's role, its effects on optimization dynamics, and practical strategies for selection and tuning.

## The Role of Learning Rate

### Update Rule Revisited

The gradient descent update rule is:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

The learning rate $\eta$ **scales** the gradient:

- **Large $\eta$**: Big steps in parameter space
- **Small $\eta$**: Small, cautious steps

### Physical Analogy

Imagine descending a mountain in fog:

- **Large learning rate**: Running downhill—fast but risky (might overshoot valley or fall off cliff)
- **Small learning rate**: Shuffling slowly—safe but tedious (might not reach base camp before nightfall)
- **Optimal learning rate**: Brisk walking—efficient progress while maintaining control

## Effects of Different Learning Rates

### Too Small ($\eta \ll 1$)

**Symptoms**:

- Very slow convergence
- Many iterations required
- May get stuck in shallow local minima
- Training takes excessively long

**Example trajectory** (1D):
```
Loss ↑
     │╲
     │ ╲
     │  ╲
     │   ╲
     │    ╲
     │     ╲
     │      ╲
     │       ╲______________  (very slow descent)
     └──────────────────────→ Iterations
```

### Too Large ($\eta \gg 1$)

**Symptoms**:

- Oscillation around minimum
- Overshooting optimal values
- May diverge completely (loss increases to infinity)
- Training becomes unstable

**Example trajectory** (1D):
```
Parameter
     │    ╱╲    ╱╲
     │   ╱  ╲  ╱  ╲
     │  ╱    ╲╱    ╲   (oscillating)
     │ ╱            ╲
─────┼─────────────────── Optimal
     │
     └──────────────────→ Iterations
```

### Just Right

**Symptoms**:

- Steady decrease in loss
- Smooth convergence
- Parameters stabilize at good values
- Efficient training time

**Example trajectory** (1D):
```
Loss ↑
     │╲
     │ ╲
     │  ╲
     │   ╲____
     │        ╲___________  (smooth descent)
     └──────────────────────→ Iterations
```

## Mathematical Analysis

### Convergence Condition

For a **quadratic loss** $L(\theta) = \frac{1}{2}a(\theta - \theta^*)^2$, the update becomes:

$$\theta_{t+1} = \theta_t - \eta \cdot a(\theta_t - \theta^*)$$

Rearranging:

$$\theta_{t+1} - \theta^* = (1 - \eta a)(\theta_t - \theta^*)$$

For convergence, we need $|1 - \eta a| < 1$, which requires:

$$0 < \eta < \frac{2}{a}$$

### Optimal Learning Rate

The fastest convergence occurs when $1 - \eta a = 0$, giving:

$$\eta^* = \frac{1}{a}$$

In this ideal case, convergence happens in **one step**!

### General Case: Lipschitz Gradients

For functions with **$L$-Lipschitz continuous gradients**:

$$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L\|\mathbf{x} - \mathbf{y}\|$$

Convergence is guaranteed for:

$$\eta \leq \frac{1}{L}$$

## Practical Learning Rate Selection

### Rule of Thumb Values

| Problem Type | Starting Learning Rate |
|--------------|----------------------|
| Linear regression | 0.01 - 0.1 |
| Logistic regression | 0.01 - 0.1 |
| Simple neural networks | 0.001 - 0.01 |
| Deep networks (SGD) | 0.01 - 0.1 |
| Deep networks (Adam) | 0.0001 - 0.001 |
| Transformers | 0.00001 - 0.0001 |
| Fine-tuning pretrained | 0.00001 - 0.00005 |

### Learning Rate Finder

A systematic approach to find good learning rates:

```python
def learning_rate_finder(model, train_loader, criterion, 
                         lr_min=1e-7, lr_max=1, num_iter=100):
    """
    Find optimal learning rate by gradually increasing it
    and monitoring loss.
    """
    # Store original state
    model_state = copy.deepcopy(model.state_dict())
    
    # Exponentially increase learning rate
    lr_schedule = np.logspace(np.log10(lr_min), np.log10(lr_max), num_iter)
    
    losses = []
    lrs = []
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_min)
    
    for i, (data, target) in enumerate(train_loader):
        if i >= num_iter:
            break
            
        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[i]
        
        # Training step
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        lrs.append(lr_schedule[i])
        
        # Stop if loss explodes
        if loss.item() > 4 * losses[0]:
            break
    
    # Restore original model
    model.load_state_dict(model_state)
    
    # Find LR where loss decreases fastest
    # (steepest negative slope on log-log plot)
    return lrs, losses
```

**Usage**:
```python
lrs, losses = learning_rate_finder(model, train_loader, criterion)
plt.semilogx(lrs, losses)
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
# Choose LR where loss is still decreasing (before it explodes)
```

### Grid Search

Simple but effective:

```python
learning_rates = [0.0001, 0.001, 0.01, 0.1]

results = {}
for lr in learning_rates:
    model = create_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    final_loss = train(model, optimizer, epochs=50)
    results[lr] = final_loss
    
best_lr = min(results, key=results.get)
```

## Visualization: Learning Rate Effects

### 1D Loss Landscape

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

def gradient_fn(w):
    return 2 * (w - 3)  # Gradient of L(w) = (w - 3)²

def run_gd(w_init, lr, n_steps):
    w = w_init
    trajectory = [w]
    for _ in range(n_steps):
        w = w - lr * gradient_fn(w)
        trajectory.append(w)
    return trajectory

# Different learning rates
learning_rates = [0.1, 0.5, 0.9, 1.1]
w_init = 7.0
n_steps = 15

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss landscape
w_vals = np.linspace(-2, 8, 100)
loss_vals = (w_vals - 3) ** 2

for idx, lr in enumerate(learning_rates):
    ax = axes[idx // 2, idx % 2]
    trajectory = run_gd(w_init, lr, n_steps)
    loss_trajectory = [(w - 3)**2 for w in trajectory]
    
    ax.plot(w_vals, loss_vals, 'gray', alpha=0.5)
    ax.plot(trajectory, loss_trajectory, 'ro-', markersize=5)
    ax.axvline(x=3, color='blue', linestyle='--', alpha=0.5)
    ax.set_xlabel('Weight w')
    ax.set_ylabel('Loss')
    ax.set_title(f'Learning Rate = {lr}')
    ax.set_ylim(-1, 20)

plt.tight_layout()
plt.show()
```

### 2D Loss Landscape

```python
# Visualize optimization path on 2D loss surface
def compute_loss_2d(w, b, X, y):
    y_pred = w * X + b
    return torch.mean((y_pred - y) ** 2).item()

# Create contour plot
w_range = np.linspace(1, 5, 50)
b_range = np.linspace(0, 4, 50)
W, B = np.meshgrid(w_range, b_range)
Z = np.array([[compute_loss_2d(w, b, X, y) for w, b in zip(row_w, row_b)] 
              for row_w, row_b in zip(W, B)])

plt.contour(W, B, Z, levels=20)
plt.colorbar(label='Loss')
# Overlay optimization trajectory
plt.plot(w_history, b_history, 'r.-', label='GD path')
plt.xlabel('Weight w')
plt.ylabel('Bias b')
```

## Learning Rate Schedules

### Why Decay Learning Rate?

- **Early training**: Large LR explores parameter space quickly
- **Late training**: Small LR enables fine-tuning near minimum

### Common Schedules

**Step Decay**:
$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)
```

**Exponential Decay**:
$$\eta_t = \eta_0 \cdot \gamma^t$$

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.95
)
```

**Cosine Annealing**:
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t\pi}{T}))$$

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100
)
```

**ReduceLROnPlateau** (adaptive):
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
# In training loop:
scheduler.step(validation_loss)
```

### Warmup

Start with small LR and increase:

```python
def warmup_lambda(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=warmup_lambda
)
```

## Learning Rate and Batch Size

### Linear Scaling Rule

When increasing batch size by factor $k$, scale learning rate by $k$:

$$\eta_{new} = k \cdot \eta_{original}$$

**Intuition**: Larger batches provide more accurate gradient estimates, allowing larger steps.

### Practical Limits

- Very large batch sizes may require warmup
- Beyond certain batch sizes, generalization may suffer
- Memory constraints often limit batch size

## Adaptive Learning Rates

### Per-Parameter Learning Rates

Algorithms like **Adam**, **RMSprop**, and **AdaGrad** maintain separate learning rates for each parameter:

$$\theta_{j,t+1} = \theta_{j,t} - \frac{\eta}{\sqrt{v_{j,t}} + \epsilon} \cdot m_{j,t}$$

**Benefits**:

- Automatically adapts to different gradient magnitudes
- Works well with sparse gradients
- Less sensitive to initial learning rate choice

**See**: [Adam Optimizer](../../ch02/optimizers/adam.md), [RMSprop](../../ch02/optimizers/rmsprop.md)

## Debugging Learning Rate Issues

### Signs of Too Large LR

- Loss increases or oscillates wildly
- `nan` or `inf` values appear
- Gradients explode

**Fix**: Reduce LR by factor of 10

### Signs of Too Small LR

- Loss decreases extremely slowly
- Validation loss plateaus early
- Training takes unreasonably long

**Fix**: Increase LR by factor of 2-10

### Diagnostic Code

```python
def diagnose_lr(train_losses, val_losses):
    # Check for divergence
    if any(np.isnan(train_losses)) or any(np.isinf(train_losses)):
        return "LR too high: NaN/Inf detected"
    
    # Check for oscillation
    if len(train_losses) > 10:
        recent = train_losses[-10:]
        if max(recent) > 2 * min(recent):
            return "LR too high: Significant oscillation"
    
    # Check for slow convergence
    if len(train_losses) > 50:
        improvement = (train_losses[0] - train_losses[-1]) / train_losses[0]
        if improvement < 0.1:
            return "LR may be too low: Slow progress"
    
    return "LR appears reasonable"
```

## Key Takeaways

1. **Learning rate scales gradient updates**: Controls step size
2. **Too large**: Oscillation, divergence, instability
3. **Too small**: Slow convergence, wasted computation
4. **Use learning rate finder**: Systematic approach to selection
5. **Learning rate schedules**: Decay over training for best results
6. **Adaptive methods**: Adam etc. reduce LR sensitivity
7. **Scale with batch size**: Larger batches can use larger LR

## Connections to Other Topics

- **Optimizers**: See [Optimizer Fundamentals](../../ch02/optimizers/fundamentals.md)
- **Schedulers**: Detailed in [Learning Rate Schedulers](../index.md)
- **Adam**: Per-parameter LR in [Adam Optimizer](../../ch02/optimizers/adam.md)
- **Batch Size**: Related to [Batch, Mini-Batch, and SGD](batch_minibatch_sgd.md)

## Exercises

1. **Convergence analysis**: For $L(w) = (w-5)^2$, starting at $w_0 = 0$:
   - Compute 10 iterations with $\eta = 0.1, 0.5, 1.0, 1.5$
   - Plot the trajectories
   - Determine which learning rates converge

2. **Learning rate finder**: Implement and apply the learning rate finder to a simple neural network on MNIST. Plot the loss vs. learning rate curve.

3. **Schedule comparison**: Train the same model with:
   - Constant LR
   - StepLR (decay every 10 epochs)
   - CosineAnnealingLR
   - ReduceLROnPlateau
   
   Compare final accuracy and convergence speed.

4. **Batch size scaling**: Train with batch sizes 32, 64, 128, 256. Apply linear scaling to LR. Does the scaling rule hold?

## References

- Smith, L. N. (2017). Cyclical learning rates for training neural networks. WACV.
- Goyal, P., et al. (2017). Accurate, large minibatch SGD: Training ImageNet in 1 hour. arXiv:1706.02677.
- You, Y., et al. (2019). Large batch optimization for deep learning: Training BERT in 76 minutes. arXiv:1904.00962.
