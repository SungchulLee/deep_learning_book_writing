# Adaptive Optimizers: Adam, RMSprop, and AdaGrad

This package contains implementations and demonstrations of three important adaptive learning rate optimization algorithms used in machine learning and deep learning.

## Files Included

### 1. `adam_optimizer.py`
Implementation of the Adam (Adaptive Moment Estimation) optimizer.
- Combines momentum and adaptive learning rates
- Uses bias-corrected first and second moment estimates
- Generally the best default choice for most problems

**Run demo:**
```bash
python adam_optimizer.py
```

### 2. `rmsprop_optimizer.py`
Implementation of the RMSprop (Root Mean Square Propagation) optimizer.
- Uses moving average of squared gradients
- Fixes AdaGrad's diminishing learning rate problem
- Good for non-stationary objectives and RNNs

**Run demo:**
```bash
python rmsprop_optimizer.py
```

### 3. `adagrad_optimizer.py`
Implementation of the AdaGrad (Adaptive Gradient) optimizer.
- Accumulates all historical squared gradients
- Excellent for sparse data
- Learning rate decreases over time

**Run demo:**
```bash
python adagrad_optimizer.py
```

### 4. `optimizer_comparison.py`
Comprehensive comparison of all three optimizers on different problems.
- Simple quadratic optimization
- Ill-conditioned problems
- Noisy gradient scenarios
- Rosenbrock function benchmark

**Run comparison:**
```bash
python optimizer_comparison.py
```

## Quick Start

```python
from adam_optimizer import Adam
from rmsprop_optimizer import RMSprop
from adagrad_optimizer import AdaGrad
import numpy as np

# Initialize parameters
params = {'weight': np.array([1.0, 2.0, 3.0])}

# Initialize optimizer
optimizer = Adam(learning_rate=0.001)

# Training loop
for iteration in range(100):
    # Your gradient computation
    grads = compute_gradients(params)  # Your function
    
    # Update parameters
    params = optimizer.update(params, grads)
```

## Algorithm Overview

### AdaGrad (2011)
- **Key Idea:** Accumulate squared gradients; larger accumulated gradients → smaller learning rates
- **Pros:** Great for sparse data, automatically adapts per-parameter learning rates
- **Cons:** Learning rate can become too small and stop learning
- **Best for:** NLP, recommender systems, sparse features

### RMSprop (2012)
- **Key Idea:** Use exponential moving average of squared gradients instead of accumulation
- **Pros:** Fixes AdaGrad's diminishing learning rate, works on non-stationary problems
- **Cons:** No momentum, requires tuning decay rate
- **Best for:** RNNs, online learning

### Adam (2014)
- **Key Idea:** Combine RMSprop with momentum, add bias correction
- **Pros:** Usually works well out-of-the-box, most popular optimizer
- **Cons:** Can sometimes converge to suboptimal solutions on some problems
- **Best for:** General purpose, default choice

## Hyperparameter Recommendations

### Adam
- `learning_rate`: 0.001 (default)
- `beta1`: 0.9 (momentum decay)
- `beta2`: 0.999 (RMSprop decay)
- `epsilon`: 1e-8

### RMSprop
- `learning_rate`: 0.001
- `rho`: 0.9 (decay rate)
- `epsilon`: 1e-8

### AdaGrad
- `learning_rate`: 0.01 (can be higher than others)
- `epsilon`: 1e-8

## Mathematical Formulas

### AdaGrad Update
```
cache_t = cache_{t-1} + grad_t^2
param_t = param_{t-1} - lr * grad_t / (sqrt(cache_t) + epsilon)
```

### RMSprop Update
```
cache_t = rho * cache_{t-1} + (1-rho) * grad_t^2
param_t = param_{t-1} - lr * grad_t / (sqrt(cache_t) + epsilon)
```

### Adam Update
```
m_t = beta1 * m_{t-1} + (1-beta1) * grad_t
v_t = beta2 * v_{t-1} + (1-beta2) * grad_t^2
m_hat_t = m_t / (1 - beta1^t)
v_hat_t = v_t / (1 - beta2^t)
param_t = param_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)
```

## When to Use Which Optimizer?

**Use Adam if:**
- You're not sure which optimizer to use
- You want good performance without much tuning
- You're working on a general deep learning problem

**Use RMSprop if:**
- You're training RNNs
- You need something simpler than Adam
- Your problem is non-stationary

**Use AdaGrad if:**
- You have sparse gradients (NLP, recommender systems)
- You want automatic per-parameter learning rate adaptation
- You don't need to train for many iterations

## References

1. **AdaGrad:** Duchi, J., Hazan, E., & Singer, Y. (2011). "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"

2. **RMSprop:** Tieleman, T., & Hinton, G. (2012). Lecture 6.5 - RMSprop, COURSERA: Neural Networks for Machine Learning

3. **Adam:** Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization"

## Requirements

- Python 3.x
- NumPy

## License

Educational implementation for learning purposes.
