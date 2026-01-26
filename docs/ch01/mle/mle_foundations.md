# Maximum Likelihood Estimation: Foundations

## Introduction

Maximum Likelihood Estimation (MLE) is one of the most fundamental and widely used methods for parameter estimation in statistics and machine learning. Given a statistical model and observed data, MLE finds the parameter values that maximize the probability of observing the data we actually observed.

!!! note "Why MLE Matters for Deep Learning"
    Understanding MLE is essential because virtually every loss function in deep learning can be derived from MLE principles. Cross-entropy loss, mean squared error, and many other objective functions are simply negative log-likelihoods under different probabilistic assumptions.

## The Core Idea

### Intuitive Understanding

Imagine you flip a coin 100 times and observe 70 heads. What is the most reasonable estimate for the probability of heads? Intuitively, you would say 0.70 (or 70%). This intuition is exactly what MLE formalizes mathematically.

MLE asks: **"Given my observations, what parameter values would have made these observations most probable?"**

### Formal Definition

Let $\mathbf{X} = \{x_1, x_2, \ldots, x_n\}$ be a set of $n$ independent observations from a probability distribution with unknown parameter(s) $\theta$. The **likelihood function** is defined as:

$$
L(\theta | \mathbf{X}) = P(\mathbf{X} | \theta) = \prod_{i=1}^{n} p(x_i | \theta)
$$

The **Maximum Likelihood Estimator** is:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} L(\theta | \mathbf{X})
$$

### Log-Likelihood

In practice, we almost always work with the **log-likelihood** instead of the likelihood:

$$
\ell(\theta | \mathbf{X}) = \log L(\theta | \mathbf{X}) = \sum_{i=1}^{n} \log p(x_i | \theta)
$$

!!! tip "Why Log-Likelihood?"
    1. **Numerical Stability**: Products of many small probabilities can underflow to zero
    2. **Computational Efficiency**: Sums are faster to compute than products
    3. **Mathematical Convenience**: Derivatives of sums are simpler than derivatives of products
    4. **Monotonicity**: Since $\log$ is monotonically increasing, maximizing $\ell(\theta)$ is equivalent to maximizing $L(\theta)$

## Mathematical Framework

### The Likelihood Function

For a parametric model $p(x|\theta)$, the likelihood function treats the data as fixed and the parameters as variables:

$$
L: \Theta \to \mathbb{R}^+, \quad \theta \mapsto \prod_{i=1}^{n} p(x_i | \theta)
$$

where $\Theta$ is the parameter space.

**Key Properties:**

- The likelihood is NOT a probability distribution over $\theta$
- $\int L(\theta | \mathbf{X}) d\theta$ generally does NOT equal 1
- The likelihood tells us the relative plausibility of different parameter values

### Finding the MLE

For differentiable likelihood functions, the MLE is found by:

1. **Taking the derivative** of the log-likelihood with respect to $\theta$
2. **Setting it to zero**: $\frac{\partial \ell}{\partial \theta} = 0$
3. **Solving for $\theta$**
4. **Verifying** it's a maximum (second derivative test)

The equation $\frac{\partial \ell}{\partial \theta} = 0$ is called the **score equation**.

### The Score Function

The **score function** is the gradient of the log-likelihood:

$$
s(\theta) = \nabla_\theta \ell(\theta | \mathbf{X}) = \sum_{i=1}^{n} \nabla_\theta \log p(x_i | \theta)
$$

**Important Property**: Under regularity conditions, the expected value of the score is zero:

$$
\mathbb{E}[s(\theta_0)] = 0
$$

where $\theta_0$ is the true parameter value.

## Example: Bernoulli Distribution

Let's work through MLE for the simplest case: estimating the probability $p$ of success in a Bernoulli distribution.

### Setup

- **Model**: $X \sim \text{Bernoulli}(p)$, where $P(X=1) = p$ and $P(X=0) = 1-p$
- **Data**: $\mathbf{X} = \{x_1, \ldots, x_n\}$ where each $x_i \in \{0, 1\}$
- **Parameter**: $p \in [0, 1]$

### Step 1: Write the Likelihood

For a single observation:
$$
p(x_i | p) = p^{x_i}(1-p)^{1-x_i}
$$

For all observations (assuming independence):
$$
L(p | \mathbf{X}) = \prod_{i=1}^{n} p^{x_i}(1-p)^{1-x_i} = p^{\sum x_i}(1-p)^{n - \sum x_i}
$$

Let $k = \sum_{i=1}^{n} x_i$ (number of successes). Then:
$$
L(p) = p^k (1-p)^{n-k}
$$

### Step 2: Take the Log-Likelihood

$$
\ell(p) = k \log p + (n-k) \log(1-p)
$$

### Step 3: Differentiate and Set to Zero

$$
\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0
$$

### Step 4: Solve

$$
\frac{k}{p} = \frac{n-k}{1-p}
$$
$$
k(1-p) = p(n-k)
$$
$$
k - kp = pn - pk
$$
$$
k = pn
$$
$$
\hat{p}_{\text{MLE}} = \frac{k}{n} = \frac{\sum_{i=1}^{n} x_i}{n}
$$

**Result**: The MLE is simply the sample proportion—exactly what intuition suggests!

### Step 5: Verify Maximum

$$
\frac{d^2\ell}{dp^2} = -\frac{k}{p^2} - \frac{n-k}{(1-p)^2} < 0
$$

Since the second derivative is always negative (for $0 < p < 1$), this confirms we have a maximum.

## PyTorch Implementation

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_coin_flips(n_flips: int, true_p: float, seed: int = 42) -> torch.Tensor:
    """Generate synthetic Bernoulli data (coin flips)."""
    torch.manual_seed(seed)
    return (torch.rand(n_flips) < true_p).float()

def compute_log_likelihood(data: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Compute log-likelihood for Bernoulli distribution.
    
    ℓ(p) = Σ[x_i * log(p) + (1-x_i) * log(1-p)]
    """
    epsilon = 1e-8  # Numerical stability
    p = torch.clamp(p, epsilon, 1 - epsilon)
    return torch.sum(data * torch.log(p) + (1 - data) * torch.log(1 - p))

def analytical_mle(data: torch.Tensor) -> float:
    """Compute MLE analytically: p̂ = k/n"""
    return data.mean().item()

def gradient_based_mle(data: torch.Tensor, 
                       lr: float = 0.1, 
                       n_iter: int = 500) -> tuple:
    """
    Compute MLE using gradient descent.
    
    This demonstrates the connection between MLE and optimization
    that underlies all of deep learning.
    """
    # Initialize parameter (use sigmoid parameterization for unconstrained optimization)
    logit_p = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.Adam([logit_p], lr=lr)
    
    history = []
    for i in range(n_iter):
        p = torch.sigmoid(logit_p)  # Constrain to (0, 1)
        
        # Negative log-likelihood (we minimize this)
        nll = -compute_log_likelihood(data, p)
        
        optimizer.zero_grad()
        nll.backward()
        optimizer.step()
        
        history.append(p.item())
    
    return torch.sigmoid(logit_p).item(), history

# Example usage
if __name__ == "__main__":
    # Generate data
    TRUE_P = 0.7
    N_FLIPS = 100
    
    data = generate_coin_flips(N_FLIPS, TRUE_P)
    n_heads = int(data.sum().item())
    
    print(f"Data: {n_heads} heads out of {N_FLIPS} flips")
    print(f"True p: {TRUE_P}")
    print(f"Analytical MLE: {analytical_mle(data):.4f}")
    
    p_gradient, history = gradient_based_mle(data)
    print(f"Gradient-based MLE: {p_gradient:.4f}")
```

## Visualizing the Likelihood Function

```python
def plot_likelihood_analysis(data: torch.Tensor, true_p: float):
    """Visualize the likelihood function and MLE."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    p_values = np.linspace(0.01, 0.99, 200)
    
    # Plot 1: Log-Likelihood Function
    log_liks = [compute_log_likelihood(data, torch.tensor(p)).item() 
                for p in p_values]
    
    ax = axes[0]
    ax.plot(p_values, log_liks, 'b-', linewidth=2)
    ax.axvline(true_p, color='green', linestyle='--', 
               label=f'True p = {true_p}')
    ax.axvline(analytical_mle(data), color='red', linestyle='-', 
               label=f'MLE = {analytical_mle(data):.3f}')
    ax.set_xlabel('p')
    ax.set_ylabel('Log-Likelihood')
    ax.set_title('Log-Likelihood Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Likelihood Function (not log)
    liks = np.exp(np.array(log_liks) - max(log_liks))  # Normalized for visibility
    
    ax = axes[1]
    ax.plot(p_values, liks, 'b-', linewidth=2)
    ax.axvline(true_p, color='green', linestyle='--')
    ax.axvline(analytical_mle(data), color='red', linestyle='-')
    ax.fill_between(p_values, liks, alpha=0.3)
    ax.set_xlabel('p')
    ax.set_ylabel('Normalized Likelihood')
    ax.set_title('Likelihood Function')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Gradient Descent Convergence
    _, history = gradient_based_mle(data)
    
    ax = axes[2]
    ax.plot(history, 'b-', linewidth=2)
    ax.axhline(true_p, color='green', linestyle='--', label='True p')
    ax.axhline(analytical_mle(data), color='red', linestyle='-', label='MLE')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Estimated p')
    ax.set_title('Gradient Descent Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

## Key Concepts Summary

| Concept | Definition | Importance |
|---------|------------|------------|
| **Likelihood** | $L(\theta) = P(\text{data} \| \theta)$ | Measures how probable data is under parameters |
| **Log-Likelihood** | $\ell(\theta) = \log L(\theta)$ | Numerically stable, mathematically convenient |
| **Score Function** | $s(\theta) = \nabla_\theta \ell(\theta)$ | Gradient used for optimization |
| **MLE** | $\hat{\theta} = \arg\max_\theta L(\theta)$ | Most likely parameters given data |

## Connection to Deep Learning

The relationship between MLE and deep learning loss functions is fundamental:

$$
\text{Loss}(\theta) = -\ell(\theta | \mathbf{X}) = -\log L(\theta | \mathbf{X})
$$

This means:

- **Minimizing loss** = **Maximizing likelihood**
- **Gradient descent on loss** = **Gradient ascent on log-likelihood**
- **Cross-entropy loss** = **Negative log-likelihood for classification**
- **MSE loss** = **Negative log-likelihood for Gaussian regression**

!!! important "The Deep Learning Connection"
    When you train a neural network by minimizing cross-entropy or MSE, you are performing maximum likelihood estimation. The only difference is that the model $p(y|x, \theta)$ is parameterized by a neural network.

## Exercises

1. **Analytical Practice**: Derive the MLE for the Poisson distribution $P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$

2. **Implementation**: Modify the PyTorch code to estimate the parameter of an exponential distribution

3. **Visualization**: Create a 3D plot showing how the likelihood surface changes as you collect more data

4. **Comparison**: Implement both analytical and gradient-based MLE for the geometric distribution and compare convergence

## Further Reading

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 1.2.4
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. Chapter 4.2
- Casella, G. & Berger, R. L. (2002). *Statistical Inference*. Chapter 7
