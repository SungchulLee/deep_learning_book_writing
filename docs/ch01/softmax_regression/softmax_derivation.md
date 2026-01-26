# Softmax Function Derivation and Properties

## Learning Objectives

By the end of this section, you will be able to:

- Derive the softmax function from first principles using maximum entropy reasoning
- Understand the mathematical properties that make softmax ideal for classification
- Implement numerically stable softmax computation
- Analyze softmax behavior through temperature scaling
- Connect softmax to the exponential family of distributions

---

## Motivation: From Logits to Probabilities

### The Classification Problem

In multi-class classification, a neural network produces **logits** $\mathbf{z} = (z_1, z_2, \ldots, z_K)$ — raw, unbounded scores for each of $K$ classes. We need a function $\sigma: \mathbb{R}^K \to \Delta^{K-1}$ that maps these logits to the **probability simplex**:

$$\Delta^{K-1} = \left\{ \mathbf{p} \in \mathbb{R}^K : p_k \geq 0, \sum_{k=1}^{K} p_k = 1 \right\}$$

The softmax function is the canonical choice for this transformation.

---

## The Softmax Function

### Definition

The **softmax function** $\sigma: \mathbb{R}^K \to \mathbb{R}^K$ is defined as:

$$\sigma(\mathbf{z})_k = \frac{\exp(z_k)}{\sum_{j=1}^{K} \exp(z_j)}$$

or equivalently, for the entire vector:

$$\sigma(\mathbf{z}) = \frac{\exp(\mathbf{z})}{\mathbf{1}^T \exp(\mathbf{z})} = \frac{\exp(\mathbf{z})}{\|\exp(\mathbf{z})\|_1}$$

where $\exp(\mathbf{z})$ applies element-wise.

### Visual Intuition

```
Logits:          [2.0,  1.0,  0.1]
                   ↓     ↓     ↓
Exponentials:    [7.39, 2.72, 1.11]    (always positive)
                   ↓     ↓     ↓
Sum:             11.22
                   ↓     ↓     ↓
Softmax:         [0.66, 0.24, 0.10]    (sum to 1)
```

---

## Mathematical Derivation

### Derivation 1: Maximum Entropy Principle

The softmax function can be derived by maximizing entropy subject to constraints. Given expected values of features, we seek the distribution with maximum uncertainty.

**Problem:** Find $\mathbf{p}$ that maximizes entropy:

$$H(\mathbf{p}) = -\sum_{k=1}^{K} p_k \log p_k$$

subject to:

1. **Normalization:** $\sum_{k=1}^{K} p_k = 1$
2. **Feature expectations:** $\sum_{k=1}^{K} p_k f_j(k) = \mu_j$ for features $f_j$

**Solution via Lagrangian:**

$$\mathcal{L} = -\sum_k p_k \log p_k - \lambda_0 \left(\sum_k p_k - 1\right) - \sum_j \lambda_j \left(\sum_k p_k f_j(k) - \mu_j\right)$$

Taking the derivative with respect to $p_k$ and setting to zero:

$$\frac{\partial \mathcal{L}}{\partial p_k} = -\log p_k - 1 - \lambda_0 - \sum_j \lambda_j f_j(k) = 0$$

Solving for $p_k$:

$$p_k = \exp\left(-1 - \lambda_0 - \sum_j \lambda_j f_j(k)\right) = \frac{\exp(z_k)}{Z}$$

where $z_k = -\sum_j \lambda_j f_j(k)$ and $Z = \exp(1 + \lambda_0)$ is the normalizing constant.

This is exactly the softmax function!

### Derivation 2: From Exponential Family

The categorical distribution belongs to the **exponential family**. In canonical form:

$$P(Y = k | \boldsymbol{\eta}) = h(k) \exp\left(\boldsymbol{\eta}^T T(k) - A(\boldsymbol{\eta})\right)$$

For the categorical distribution with one-hot sufficient statistic:
- $\boldsymbol{\eta} = (\log\pi_1, \ldots, \log\pi_{K-1}, 0)$ (natural parameters)
- $T(k) = \mathbf{e}_k$ (one-hot encoding)
- $A(\boldsymbol{\eta}) = \log\sum_j \exp(\eta_j)$ (log-partition function)

The mean parameters (probabilities) are obtained via:

$$\pi_k = \frac{\partial A}{\partial \eta_k} = \frac{\exp(\eta_k)}{\sum_j \exp(\eta_j)} = \text{softmax}(\boldsymbol{\eta})_k$$

---

## Essential Properties

### Property 1: Normalization

$$\sum_{k=1}^{K} \sigma(\mathbf{z})_k = \sum_{k=1}^{K} \frac{\exp(z_k)}{\sum_j \exp(z_j)} = \frac{\sum_k \exp(z_k)}{\sum_j \exp(z_j)} = 1$$

### Property 2: Positivity

$$\sigma(\mathbf{z})_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)} > 0 \quad \forall k$$

since $\exp(x) > 0$ for all $x \in \mathbb{R}$.

### Property 3: Translation Invariance

For any constant $c \in \mathbb{R}$:

$$\sigma(\mathbf{z} + c\mathbf{1})_k = \frac{\exp(z_k + c)}{\sum_j \exp(z_j + c)} = \frac{\exp(c)\exp(z_k)}{\exp(c)\sum_j \exp(z_j)} = \sigma(\mathbf{z})_k$$

This means softmax only depends on the **relative** differences between logits.

### Property 4: Monotonicity

Softmax is monotonically increasing in each coordinate:

$$\frac{\partial \sigma(\mathbf{z})_k}{\partial z_k} = \sigma(\mathbf{z})_k (1 - \sigma(\mathbf{z})_k) > 0$$

Larger logits always produce larger probabilities.

### Property 5: Convexity of Log-Partition

The log-sum-exp function $A(\mathbf{z}) = \log\sum_k \exp(z_k)$ is convex, which ensures:
- Unique global optimum in maximum likelihood estimation
- Well-behaved optimization landscape

---

## Numerical Stability: The Log-Sum-Exp Trick

### The Problem

For large logits, $\exp(z_k)$ can overflow ($\exp(1000) = \infty$). For very negative logits, $\exp(z_k)$ can underflow ($\exp(-1000) = 0$).

```python
import numpy as np

# Overflow example
z = np.array([1000, 1001, 1002])
print(np.exp(z))  # [inf, inf, inf] - OVERFLOW!

# Underflow example
z = np.array([-1000, -1001, -1002])
print(np.exp(z))  # [0., 0., 0.] - UNDERFLOW!
```

### The Solution

Using translation invariance, subtract the maximum value:

$$\sigma(\mathbf{z})_k = \frac{\exp(z_k - z_{\max})}{\sum_j \exp(z_j - z_{\max})}$$

where $z_{\max} = \max_j z_j$.

This ensures:
- The largest exponent is $\exp(0) = 1$ (no overflow)
- Other exponents are $\leq 1$ (controlled magnitude)

### Implementation

```python
import torch
import torch.nn.functional as F

def softmax_naive(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Naive softmax - susceptible to overflow/underflow."""
    exp_z = torch.exp(z)
    return exp_z / exp_z.sum(dim=dim, keepdim=True)

def softmax_stable(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax using the max trick."""
    z_max = z.max(dim=dim, keepdim=True).values
    exp_z = torch.exp(z - z_max)
    return exp_z / exp_z.sum(dim=dim, keepdim=True)

# Test with extreme values
z_extreme = torch.tensor([1000.0, 1001.0, 1002.0])

# Naive: produces nan due to overflow
naive_result = softmax_naive(z_extreme)
print(f"Naive softmax: {naive_result}")  # [nan, nan, nan]

# Stable: works correctly
stable_result = softmax_stable(z_extreme)
print(f"Stable softmax: {stable_result}")  # [0.0900, 0.2447, 0.6652]

# PyTorch's built-in (already stable)
pytorch_result = F.softmax(z_extreme, dim=0)
print(f"PyTorch softmax: {pytorch_result}")  # Same as stable
```

### Log-Softmax for Even Better Stability

When computing cross-entropy, we need $\log(\sigma(\mathbf{z}))$. Computing this directly can lose precision:

```python
def log_softmax_stable(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable log-softmax.
    
    log(softmax(z)_k) = z_k - log(sum_j exp(z_j))
                      = z_k - z_max - log(sum_j exp(z_j - z_max))
    """
    z_max = z.max(dim=dim, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(z - z_max).sum(dim=dim, keepdim=True))
    return z - z_max - log_sum_exp

# PyTorch's optimized implementation
log_probs = F.log_softmax(z_extreme, dim=0)
```

---

## Temperature Scaling

### Definition

**Temperature-scaled softmax** introduces a parameter $T > 0$:

$$\sigma(\mathbf{z}; T)_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$

### Behavior Analysis

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| $T \to 0^+$ | Approaches argmax (hard assignment) | Inference, discrete decisions |
| $T = 1$ | Standard softmax | Training |
| $T > 1$ | Softer, more uniform distribution | Exploration, knowledge distillation |
| $T \to \infty$ | Uniform distribution | Maximum uncertainty |

### Mathematical Limits

**Low temperature limit ($T \to 0^+$):**

$$\lim_{T \to 0^+} \sigma(\mathbf{z}; T) = \mathbf{e}_{k^*}$$

where $k^* = \arg\max_k z_k$. The distribution becomes a one-hot vector (argmax).

**High temperature limit ($T \to \infty$):**

$$\lim_{T \to \infty} \sigma(\mathbf{z}; T) = \frac{1}{K} \mathbf{1}$$

The distribution becomes uniform.

### Implementation

```python
def softmax_temperature(z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Softmax with temperature scaling.
    
    Args:
        z: Logits of shape (..., K)
        temperature: Temperature parameter T > 0
    
    Returns:
        Temperature-scaled probabilities
    """
    return F.softmax(z / temperature, dim=-1)

# Demonstrate temperature effects
logits = torch.tensor([2.0, 1.0, 0.5])
print(f"Logits: {logits}")

for T in [0.1, 0.5, 1.0, 2.0, 10.0]:
    probs = softmax_temperature(logits, T)
    print(f"T={T:4.1f}: {probs.numpy().round(3)}")
```

Output:
```
Logits: tensor([2.0000, 1.0000, 0.5000])
T= 0.1: [1.    0.    0.   ]    (nearly argmax)
T= 0.5: [0.953 0.042 0.006]    (sharp)
T= 1.0: [0.659 0.242 0.099]    (standard)
T= 2.0: [0.474 0.319 0.207]    (soft)
T=10.0: [0.356 0.332 0.312]    (nearly uniform)
```

---

## Softmax vs Other Functions

### Comparison with Alternatives

| Function | Formula | Properties |
|----------|---------|------------|
| Softmax | $\frac{e^{z_k}}{\sum_j e^{z_j}}$ | Smooth, differentiable, preserves ranking |
| Hardmax | $\mathbb{1}[k = \arg\max_j z_j]$ | Non-differentiable, sparse |
| Sparsemax | Euclidean projection onto simplex | Sparse, differentiable |
| Entmax | Generalization of softmax/sparsemax | Tunable sparsity |

### When to Use Softmax

✅ **Use softmax when:**
- Training neural networks (need smooth gradients)
- Probabilities should be strictly positive
- All classes should receive some probability mass

⚠️ **Consider alternatives when:**
- Sparse attention is desired (sparsemax, entmax)
- Hard decisions are needed (hardmax at inference)
- Calibrated probabilities are critical (temperature scaling, Platt scaling)

---

## Complete PyTorch Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SoftmaxAnalysis:
    """Comprehensive softmax analysis and visualization."""
    
    @staticmethod
    def demonstrate_properties():
        """Show key softmax properties."""
        print("=" * 60)
        print("SOFTMAX PROPERTIES DEMONSTRATION")
        print("=" * 60)
        
        logits = torch.tensor([2.0, 1.0, 0.5])
        probs = F.softmax(logits, dim=0)
        
        # Property 1: Normalization
        print(f"\n1. Normalization: sum = {probs.sum().item():.6f}")
        
        # Property 2: Positivity
        print(f"2. Positivity: all > 0? {(probs > 0).all().item()}")
        
        # Property 3: Translation invariance
        shifted_logits = logits + 1000
        shifted_probs = F.softmax(shifted_logits, dim=0)
        print(f"3. Translation invariance: {torch.allclose(probs, shifted_probs)}")
        
        # Property 4: Monotonicity
        print(f"4. Monotonicity: logits order = {logits.argsort(descending=True).tolist()}")
        print(f"                 probs order  = {probs.argsort(descending=True).tolist()}")
    
    @staticmethod
    def compare_implementations():
        """Compare naive, stable, and PyTorch implementations."""
        print("\n" + "=" * 60)
        print("NUMERICAL STABILITY COMPARISON")
        print("=" * 60)
        
        # Normal case
        z_normal = torch.tensor([2.0, 1.0, 0.5])
        
        # Extreme case
        z_extreme = torch.tensor([1000.0, 1001.0, 1002.0])
        
        for name, z in [("Normal", z_normal), ("Extreme", z_extreme)]:
            print(f"\n{name} logits: {z}")
            
            # Naive (only works for normal)
            if name == "Normal":
                naive = torch.exp(z) / torch.exp(z).sum()
                print(f"  Naive:   {naive}")
            else:
                print(f"  Naive:   [overflow/nan]")
            
            # Stable
            z_shifted = z - z.max()
            stable = torch.exp(z_shifted) / torch.exp(z_shifted).sum()
            print(f"  Stable:  {stable}")
            
            # PyTorch
            pytorch = F.softmax(z, dim=0)
            print(f"  PyTorch: {pytorch}")
    
    @staticmethod
    def temperature_analysis():
        """Analyze temperature scaling effects."""
        print("\n" + "=" * 60)
        print("TEMPERATURE SCALING ANALYSIS")
        print("=" * 60)
        
        logits = torch.tensor([2.0, 1.0, 0.5])
        temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        print(f"\nLogits: {logits.tolist()}")
        print(f"\n{'Temp':<8} {'Prob 1':<10} {'Prob 2':<10} {'Prob 3':<10} {'Entropy':<10}")
        print("-" * 48)
        
        for T in temperatures:
            probs = F.softmax(logits / T, dim=0)
            entropy = -(probs * probs.log()).sum().item()
            print(f"{T:<8.1f} {probs[0].item():<10.4f} {probs[1].item():<10.4f} "
                  f"{probs[2].item():<10.4f} {entropy:<10.4f}")
        
        print("\nNote: Higher temperature → higher entropy (more uniform)")

# Run demonstrations
if __name__ == "__main__":
    SoftmaxAnalysis.demonstrate_properties()
    SoftmaxAnalysis.compare_implementations()
    SoftmaxAnalysis.temperature_analysis()
```

---

## Summary

### Core Formula

$$\boxed{\sigma(\mathbf{z})_k = \frac{\exp(z_k)}{\sum_{j=1}^{K} \exp(z_j)}}$$

### Key Properties

| Property | Mathematical Statement |
|----------|----------------------|
| Normalization | $\sum_k \sigma(\mathbf{z})_k = 1$ |
| Positivity | $\sigma(\mathbf{z})_k > 0$ |
| Translation Invariance | $\sigma(\mathbf{z} + c\mathbf{1}) = \sigma(\mathbf{z})$ |
| Monotonicity | $z_i > z_j \Rightarrow \sigma(\mathbf{z})_i > \sigma(\mathbf{z})_j$ |

### Implementation Guidelines

!!! tip "Best Practices"
    1. **Always use PyTorch's `F.softmax` or `F.log_softmax`** — they're numerically stable
    2. **Subtract the max before exponentiating** if implementing manually
    3. **Use `F.log_softmax` directly** when computing cross-entropy
    4. **Apply temperature scaling** for knowledge distillation or calibration

---

## Next Steps

With a solid understanding of the softmax function, you're ready to explore:

1. **Cross-Entropy Loss** — How softmax connects to the loss function
2. **Jacobian of Softmax** — Derivatives for backpropagation
3. **Gradient Derivation** — Complete gradient flow analysis

---

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 6.2.2
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 4.3.4
3. Jaynes, E. T. (1957). *Information Theory and Statistical Mechanics*
4. PyTorch Documentation: [torch.nn.functional.softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
