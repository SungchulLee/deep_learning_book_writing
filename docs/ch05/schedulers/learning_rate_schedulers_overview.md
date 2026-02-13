# Combined Learning Rate Scheduling Library

A comprehensive collection of learning rate scheduling strategies combining PyTorch's built-in schedulers with advanced custom implementations for warmup and cyclical learning rates.

## 🎯 Overview

This library provides:

1. **PyTorch Built-in Schedulers** - Complete demonstration framework with:
   - StepLR, MultiStepLR, ExponentialLR
   - CosineAnnealingLR, OneCycleLR, CyclicLR
   - ReduceLROnPlateau
   - Full training pipeline with visualization

2. **Custom Advanced Schedulers** - Specialized implementations:
   - Advanced Warmup Strategies (Linear, Exponential, Cosine, with Decay)
   - Enhanced Cyclical Methods (Custom CyclicLR, OneCycleLR, SGDR)
   - Mathematical implementations with full control

## 📦 Package Structure

```
lr_schedulers/
├── scheduler.py                   # Main entry point for PyTorch schedulers
├── enhanced_examples.py           # Examples combining both approaches
├── scheduler/
│   ├── __init__.py
│   ├── config.py                  # Configuration and CLI parsing
│   ├── data_loader.py             # Synthetic data generation
│   ├── model.py                   # Simple MLP for demos
│   ├── train.py                   # Training loop with PyTorch schedulers
│   ├── utils.py                   # Utilities and visualization
│   └── custom/                    # Custom scheduler implementations
│       ├── __init__.py
│       ├── lr_warmup.py          # Warmup strategies
│       └── cyclical_lr.py        # Cyclical strategies
├── README.md                      # This file
└── requirements.txt               # Dependencies
```

## 🚀 Installation

```bash
# Install dependencies
pip install torch numpy matplotlib

# Or use requirements file
pip install -r requirements.txt
```

## ⚡ Quick Start

### Using PyTorch Built-in Schedulers

Run the complete framework with any PyTorch scheduler:

```bash
# StepLR - Decay every N epochs
python scheduler.py --scheduler step --epochs 50 --step_size 20 --gamma 0.5

# OneCycleLR - Fast training with 1cycle policy
python scheduler.py --scheduler onecycle --epochs 5 --max_lr 0.2

# CosineAnnealing - Smooth cosine decay
python scheduler.py --scheduler cosine --epochs 50 --t_max 50

# ReduceLROnPlateau - Adaptive based on metrics
python scheduler.py --scheduler plateau --epochs 50 --patience 5
```

### Using Custom Schedulers

```python
from scheduler.custom import LinearWarmup, OneCycleLR, CosineAnnealingWarmRestarts

# Linear warmup
warmup = LinearWarmup(base_lr=1e-3, warmup_steps=1000)

# 1cycle policy
onecycle = OneCycleLR(max_lr=1e-2, total_steps=10000, pct_start=0.3)

# SGDR with warm restarts
sgdr = CosineAnnealingWarmRestarts(max_lr=1e-2, min_lr=1e-5, t_0=2000, t_mult=2)

# In training loop
for step in range(total_steps):
    lr = scheduler.get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # ... training code ...
```

### Running Enhanced Examples

See 6 comprehensive examples combining both approaches:

```bash
python enhanced_examples.py
```

## 📊 Available Schedulers

### PyTorch Built-in (via scheduler.py)

| Scheduler | Description | Best For |
|-----------|-------------|----------|
| **StepLR** | Decay by gamma every N epochs | Classic training, known decay points |
| **MultiStepLR** | Decay at specific milestones | Precise control over decay schedule |
| **ExponentialLR** | Exponential decay each epoch | Smooth gradual decay |
| **CosineAnnealingLR** | Cosine curve decay | Modern architectures, image models |
| **OneCycleLR** | 1cycle policy | Fast training, super-convergence |
| **CyclicLR** | Cycle between bounds | Avoiding local minima, LR range finding |
| **ReduceLROnPlateau** | Adaptive based on metrics | Unsure of optimal schedule |

### Custom Advanced (via scheduler/custom/)

| Scheduler | Description | Best For |
|-----------|-------------|----------|
| **LinearWarmup** | Linear increase to base_lr | Simple stable warmup |
| **ExponentialWarmup** | Exponential increase | Sensitive models |
| **CosineWarmup** | Smooth cosine warmup | Gradual stable start |
| **WarmupWithDecay** | Warmup + cosine decay combined | Transformer training |
| **Custom CyclicLR** | Advanced cycling modes | More control than PyTorch |
| **Custom OneCycleLR** | Mathematical implementation | Understanding 1cycle internals |
| **CosineAnnealingWarmRestarts** | SGDR with restarts | Long training, escape local minima |
| **ExponentialCyclicLR** | Exponential cycling | Custom decay patterns |

## 🔥 Usage Patterns

### Pattern 1: Warmup + PyTorch Scheduler

Combine custom warmup with any PyTorch scheduler:

```python
from scheduler.custom import LinearWarmup
from torch.optim.lr_scheduler import StepLR

# Warmup for first 1000 steps
warmup = LinearWarmup(base_lr=1e-3, warmup_steps=1000)

# Then use StepLR
step_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# In training loop
if step < 1000:
    lr = warmup.get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
else:
    step_scheduler.step()  # After warmup
```

### Pattern 2: Transformer-Style Training

Warmup followed by cosine decay (popular in BERT, GPT, etc.):

```python
from scheduler.custom import WarmupWithDecay

scheduler = WarmupWithDecay(
    base_lr=5e-4,
    warmup_steps=10000,
    total_steps=100000,
    min_lr=1e-6
)

# Single scheduler handles both warmup and decay
for step in range(total_steps):
    lr = scheduler.get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

### Pattern 3: Cyclical Training

Use advanced cyclical strategies:

```python
from scheduler.custom import CyclicLR

scheduler = CyclicLR(
    base_lr=1e-4,
    max_lr=1e-2,
    step_size=2000,  # Half cycle length
    mode='triangular2'  # Decreasing amplitude
)

# Updates every batch
for step, (data, target) in enumerate(dataloader):
    lr = scheduler.get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # ... training ...
```

### Pattern 4: Fast Convergence with 1cycle

Achieve super-convergence with aggressive schedule:

```python
from scheduler.custom import OneCycleLR

scheduler = OneCycleLR(
    max_lr=1e-2,
    total_steps=len(dataloader) * epochs,
    pct_start=0.3,  # 30% warmup
    anneal_strategy='cos'
)

# Single pass schedule
for step in range(total_steps):
    lr = scheduler.get_lr(step)
    # ... training ...
```

## 📖 Detailed Documentation

### Custom Warmup Strategies

**Linear Warmup**
```python
LinearWarmup(base_lr, warmup_steps)
```
- Linearly increases from 0 to base_lr
- Simplest and most common
- Good default choice

**Exponential Warmup**
```python
ExponentialWarmup(base_lr, warmup_steps, start_lr=1e-7)
```
- Exponentially increases from start_lr to base_lr
- Slower initial increase
- Better for very sensitive models

**Cosine Warmup**
```python
CosineWarmup(base_lr, warmup_steps)
```
- Follows cosine curve
- Smooth acceleration
- More gradual than linear

**Warmup With Decay**
```python
WarmupWithDecay(base_lr, warmup_steps, total_steps, min_lr=0)
```
- Complete schedule in one: warmup + cosine decay
- Popular in transformer models (BERT, GPT)
- No need to combine multiple schedulers

### Custom Cyclical Strategies

**CyclicLR**
```python
CyclicLR(base_lr, max_lr, step_size, mode='triangular')
```
- Modes: 'triangular', 'triangular2', 'exp_range'
- triangular: constant amplitude
- triangular2: halving amplitude
- exp_range: exponential decay

**OneCycleLR**
```python
OneCycleLR(max_lr, total_steps, pct_start=0.3, anneal_strategy='cos')
```
- Single cycle: increase then decrease
- pct_start: fraction of training for increase phase
- anneal_strategy: 'cos' or 'linear'
- Based on Leslie Smith's super-convergence paper

**CosineAnnealingWarmRestarts** (SGDR)
```python
CosineAnnealingWarmRestarts(max_lr, min_lr, t_0, t_mult=2)
```
- Periodic restarts of cosine annealing
- t_0: steps until first restart
- t_mult: factor to increase period after each restart
- Helps escape local minima

## 🎨 Visualization

Both frameworks automatically generate visualizations:

**PyTorch Schedulers** (via scheduler.py):
- Learning rate curves saved as PNG
- Training/validation loss curves
- Accuracy plots

**Custom Schedulers** (via custom modules):
```python
from scheduler.custom.lr_warmup import plot_warmup_schedules
from scheduler.custom.cyclical_lr import plot_cyclical_schedules

# Generate comparison plots
plot_warmup_schedules(warmup_steps=1000, total_steps=10000)
plot_cyclical_schedules(total_steps=10000)
```

## 💡 Best Practices

### Choosing Warmup Steps

- **Small models**: 100-500 steps
- **Medium models**: 500-2000 steps
- **Large models (transformers)**: 2000-10000 steps
- **Rule of thumb**: 5-10% of total training steps

### Choosing LR Bounds for Cyclical

1. Run LR range test (increase LR exponentially)
2. Find where loss starts decreasing (base_lr)
3. Find where loss starts diverging (max_lr)
4. Use these bounds for cycling

### Scheduler Selection Guide

**Use Warmup when:**
- Training large models from scratch
- Using large batch sizes (>256)
- Training is unstable initially
- Starting with high learning rates

**Use Cyclical LR when:**
- Model stuck in local minima
- Want to reduce hyperparameter tuning
- Training for moderate epochs (10-50)

**Use 1cycle when:**
- Limited training budget
- Want fastest convergence
- Know total training steps in advance
- Using SGD or AdamW

**Use SGDR when:**
- Training for many epochs (50+)
- Want periodic exploration
- Model benefits from restarts

### Combining Strategies

**Great combinations:**
- ✅ Warmup + StepLR
- ✅ Warmup + CosineAnnealing
- ✅ WarmupWithDecay (all-in-one)
- ✅ 1cycle (includes warmup)

**Avoid:**
- ❌ Multiple epoch-level schedulers simultaneously
- ❌ Warmup + 1cycle (1cycle already includes warmup)

## 🧪 Running Tests

Test custom schedulers:
```python
# In Python
from scheduler.custom import LinearWarmup

warmup = LinearWarmup(base_lr=1e-3, warmup_steps=1000)

# Test at various steps
for step in [0, 250, 500, 750, 1000]:
    print(f"Step {step}: LR = {warmup.get_lr(step)}")
```

Run full examples:
```bash
# PyTorch schedulers with full training
python scheduler.py --scheduler onecycle --epochs 5

# Custom schedulers integration
python enhanced_examples.py
```

## 📚 References

### Key Papers

1. **Cyclical Learning Rates**
   - Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks"
   - https://arxiv.org/abs/1506.01186

2. **Super-Convergence / 1cycle**
   - Smith, L. N. (2018). "A disciplined approach to neural network hyper-parameters"
   - https://arxiv.org/abs/1803.09820

3. **SGDR (Warm Restarts)**
   - Loshchilov, I., & Hutter, F. (2016). "SGDR: Stochastic Gradient Descent with Warm Restarts"
   - https://arxiv.org/abs/1608.03983

4. **Warmup**
   - Goyal, P., et al. (2017). "Accurate, Large Minibatch SGD"
   - https://arxiv.org/abs/1706.02677

5. **Transformer Training**
   - Vaswani, A., et al. (2017). "Attention Is All You Need"
   - https://arxiv.org/abs/1706.03762

## 🤝 Contributing

This is an educational package. Suggestions welcome for:
- Additional scheduling strategies
- Better visualization tools
- Integration with other frameworks
- More comprehensive examples

## 📄 License

MIT License - Free to use and modify

## 🎓 Educational Notes

### Why This Package?

1. **PyTorch Built-in Schedulers**: Great for understanding standard approaches and seeing them in action with a full training framework

2. **Custom Implementations**: Better for:
   - Understanding the mathematics
   - Implementing papers from scratch
   - Customizing behavior beyond PyTorch defaults
   - Learning how schedulers actually work

3. **Combined Approach**: Best of both worlds - use PyTorch for standard needs, custom for advanced control

### Learning Path

1. Start with `scheduler.py` to understand basic schedulers
2. Explore `enhanced_examples.py` to see custom implementations
3. Read the custom scheduler source code to understand mathematics
4. Experiment with different combinations
5. Apply to your own projects

## 🔍 Troubleshooting

**Q: Scheduler not having effect?**
- A: Make sure you're actually updating the optimizer's learning rate
- Check: `for param_group in optimizer.param_groups: param_group['lr'] = new_lr`

**Q: Loss exploding during warmup?**
- A: Start with lower initial LR or longer warmup period
- Try exponential warmup for more gradual start

**Q: Which scheduler is best?**
- A: Depends on your problem! Try:
  - OneCycleLR for fast results
  - ReduceLROnPlateau if unsure
  - WarmupWithDecay for transformers

**Q: How to visualize my custom schedule?**
- A: Generate LR values and plot them:
```python
steps = range(10000)
lrs = [scheduler.get_lr(step) for step in steps]
plt.plot(steps, lrs)
plt.show()
```

## 📞 Support

For questions about:
- **PyTorch schedulers**: Check PyTorch docs
- **Custom implementations**: Read the source code comments
- **General LR scheduling**: See references section
- **Bugs**: Check code carefully, these are educational implementations

---

**Happy Learning and Training! 🚀**
