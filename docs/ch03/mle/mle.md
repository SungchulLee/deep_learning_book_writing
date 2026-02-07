# Maximum Likelihood Estimation

## Introduction

Maximum Likelihood Estimation (MLE) is one of the most fundamental methods for parameter estimation in statistics and machine learning. Given a statistical model and observed data, MLE finds the parameter values that maximize the probability of observing the data we actually observed.

!!! note "Why MLE Matters for Deep Learning"
    Understanding MLE is essential because virtually every loss function in deep learning can be derived from MLE principles. Cross-entropy loss, mean squared error, and many other objective functions are simply negative log-likelihoods under different probabilistic assumptions.

## The Core Idea

### Intuitive Understanding

Imagine you flip a coin 100 times and observe 70 heads. What is the most reasonable estimate for the probability of heads? Intuitively, you would say 0.70. This intuition is exactly what MLE formalizes mathematically.

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

where $\Theta$ is the parameter space. Key properties to keep in mind: the likelihood is NOT a probability distribution over $\theta$; $\int L(\theta | \mathbf{X}) d\theta$ generally does NOT equal 1. The likelihood tells us the relative plausibility of different parameter values.

### Finding the MLE

For differentiable likelihood functions, the MLE is found by:

1. **Taking the derivative** of the log-likelihood with respect to $\theta$
2. **Setting it to zero**: $\frac{\partial \ell}{\partial \theta} = 0$ (the **score equation**)
3. **Solving for $\theta$**
4. **Verifying** it's a maximum (second derivative test)

### The Score Function

The **score function** is the gradient of the log-likelihood:

$$
s(\theta) = \nabla_\theta \ell(\theta | \mathbf{X}) = \sum_{i=1}^{n} \nabla_\theta \log p(x_i | \theta)
$$

Under regularity conditions, the expected value of the score is zero at the true parameter:

$$
\mathbb{E}[s(\theta_0)] = 0
$$

## Worked Example: Bernoulli Distribution

Let's derive the MLE for the simplest case: estimating the probability $p$ of success in a Bernoulli distribution.

**Setup**: Model $X \sim \text{Bernoulli}(p)$, data $\mathbf{X} = \{x_1, \ldots, x_n\}$ where each $x_i \in \{0, 1\}$, parameter $p \in [0, 1]$.

**Step 1 — Likelihood.** For a single observation $p(x_i | p) = p^{x_i}(1-p)^{1-x_i}$. For all observations (assuming independence):

$$
L(p | \mathbf{X}) = \prod_{i=1}^{n} p^{x_i}(1-p)^{1-x_i} = p^{k}(1-p)^{n - k}
$$

where $k = \sum_{i=1}^{n} x_i$ is the number of successes.

**Step 2 — Log-likelihood.**

$$
\ell(p) = k \log p + (n-k) \log(1-p)
$$

**Step 3 — Score equation.**

$$
\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0
$$

**Step 4 — Solve.** Cross-multiplying yields $k(1-p) = p(n-k)$, which simplifies to:

$$
\boxed{\hat{p}_{\text{MLE}} = \frac{k}{n} = \frac{\sum_{i=1}^{n} x_i}{n}}
$$

The MLE is simply the sample proportion — exactly what intuition suggests.

**Step 5 — Verify maximum.** The second derivative $\frac{d^2\ell}{dp^2} = -\frac{k}{p^2} - \frac{n-k}{(1-p)^2} < 0$ confirms this is a maximum.

## MLE for Common Distributions

### Discrete Distributions

**Binomial** $X \sim \text{Binomial}(n, p)$ — For $m$ observations $x_1, \ldots, x_m$:

$$
\hat{p} = \frac{\sum_{i=1}^{m} x_i}{mn} = \frac{\bar{x}}{n}
$$

**Categorical** $X \sim \text{Categorical}(p_1, \ldots, p_K)$ — Using Lagrange multipliers for the constraint $\sum p_k = 1$:

$$
\hat{p}_k = \frac{n_k}{n} = \frac{n_k}{\sum_{j=1}^{K} n_j}
$$

Each probability is estimated by its relative frequency.

**Poisson** $X \sim \text{Poisson}(\lambda)$ — The log-likelihood is $\ell(\lambda) = (\sum x_i) \log \lambda - n\lambda + \text{const}$. Setting the derivative to zero:

$$
\frac{d\ell}{d\lambda} = \frac{\sum x_i}{\lambda} - n = 0 \implies \hat{\lambda} = \bar{x}
$$

**Geometric** $X \sim \text{Geometric}(p)$ (trials until first success) — The log-likelihood is $\ell(p) = (\sum x_i - n) \log(1-p) + n \log p$:

$$
\hat{p} = \frac{n}{\sum_{i=1}^{n} x_i} = \frac{1}{\bar{x}}
$$

### Continuous Distributions

**Normal** $X \sim \mathcal{N}(\mu, \sigma^2)$ — The log-likelihood is:

$$
\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2
$$

Differentiating with respect to $\mu$: $\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum(x_i - \mu) = 0 \implies \hat{\mu} = \bar{x}$.

Differentiating with respect to $\sigma^2$: $\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum(x_i - \mu)^2 = 0$, giving:

$$
\hat{\mu} = \bar{x}, \quad \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2
$$

!!! warning "Biased Variance Estimator"
    The MLE for variance is **biased**: $\mathbb{E}[\hat{\sigma}^2] = \frac{n-1}{n}\sigma^2 < \sigma^2$. The unbiased estimator uses $n-1$ in the denominator, but MLE is consistent — bias vanishes as $n \to \infty$.

**Exponential** $X \sim \text{Exponential}(\lambda)$ — With pdf $p(x|\lambda) = \lambda e^{-\lambda x}$:

$$
\ell(\lambda) = n \log \lambda - \lambda \sum x_i \implies \hat{\lambda} = \frac{1}{\bar{x}}
$$

**Uniform** $X \sim \text{Uniform}(a, b)$ — The likelihood is $L(a,b) = (b-a)^{-n}$ when $a \leq x_{(1)}$ and $x_{(n)} \leq b$, zero otherwise. To maximize, we shrink the interval as much as possible:

$$
\hat{a} = x_{(1)} = \min_i x_i, \quad \hat{b} = x_{(n)} = \max_i x_i
$$

!!! note "Non-Regular MLE"
    The uniform distribution is a "non-regular" case because the support depends on the parameters. The MLE exists but doesn't satisfy the usual regularity conditions (e.g., Fisher information is not well-defined in the standard way).

**Gamma** $X \sim \text{Gamma}(\alpha, \beta)$ — No closed-form solution for both parameters. For $\beta$ given $\alpha$: $\hat{\beta} = \alpha / \bar{x}$. For $\alpha$, we solve numerically:

$$
\log \alpha - \psi(\alpha) = \log \bar{x} - \overline{\log x}
$$

where $\psi(\alpha) = \frac{d}{d\alpha}\log\Gamma(\alpha)$ is the digamma function.

**Multivariate Normal** $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:

$$
\hat{\boldsymbol{\mu}} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{x}_i = \bar{\mathbf{x}}, \quad \hat{\boldsymbol{\Sigma}} = \frac{1}{n}\sum_{i=1}^{n} (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T
$$

### Summary Table

| Distribution | Parameters | MLE Estimator |
|-------------|------------|---------------|
| Bernoulli | $p$ | $\hat{p} = \bar{x}$ |
| Binomial | $p$ (given $n$) | $\hat{p} = \bar{x}/n$ |
| Categorical | $p_1, \ldots, p_K$ | $\hat{p}_k = n_k/n$ |
| Poisson | $\lambda$ | $\hat{\lambda} = \bar{x}$ |
| Geometric | $p$ | $\hat{p} = 1/\bar{x}$ |
| Normal | $\mu, \sigma^2$ | $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$ |
| Exponential | $\lambda$ | $\hat{\lambda} = 1/\bar{x}$ |
| Uniform | $a, b$ | $\hat{a} = \min(x_i)$, $\hat{b} = \max(x_i)$ |
| Gamma | $\alpha, \beta$ | Numerical solution required |

### Pattern Recognition

Notice the recurring patterns in MLE derivations: sample means appear frequently for location parameters; the MLE depends on data only through sufficient statistics; constraint handling for probability vectors uses Lagrange multipliers; and rate parameters often have MLEs of the form $1/\bar{x}$.

## Fisher Information

### Definition

Recall the score function $s(\theta) = \frac{\partial}{\partial \theta} \log p(X|\theta)$. The **Fisher Information** is the variance of the score:

$$
I(\theta) = \text{Var}_\theta[s(\theta)] = \mathbb{E}_\theta\left[\left(\frac{\partial \log p(X|\theta)}{\partial \theta}\right)^2\right]
$$

Under regularity conditions (interchanging differentiation and integration), an equivalent formula is:

$$
I(\theta) = -\mathbb{E}_\theta\left[\frac{\partial^2 \log p(X|\theta)}{\partial \theta^2}\right]
$$

This second form shows that **Fisher Information equals the expected curvature of the log-likelihood**.

### Intuitive Interpretation

High Fisher Information means the log-likelihood is sharply peaked around $\theta$ — small changes in $\theta$ cause large changes in likelihood, so the data is very informative about $\theta$. Low Fisher Information means the log-likelihood is flat — the data doesn't distinguish well between different parameter values.

### Fisher Information for Common Distributions

**Bernoulli** $X \sim \text{Bernoulli}(p)$: Computing $\frac{\partial^2 \log p}{\partial p^2} = -\frac{x}{p^2} - \frac{1-x}{(1-p)^2}$ and taking the expectation:

$$
\boxed{I(p) = \frac{1}{p(1-p)}}
$$

Fisher Information is highest when $p = 0.5$ (most uncertainty) and lowest near $p = 0$ or $p = 1$.

**Normal** $X \sim \mathcal{N}(\mu, \sigma^2)$ with known variance: $I(\mu) = 1/\sigma^2$. With known mean: $I(\sigma^2) = 1/(2\sigma^4)$.

**Poisson** $X \sim \text{Poisson}(\lambda)$: $I(\lambda) = 1/\lambda$.

**Exponential** $X \sim \text{Exponential}(\lambda)$: $I(\lambda) = 1/\lambda^2$.

### Fisher Information Matrix

For a parameter vector $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_k)^T$, the **Fisher Information Matrix** is:

$$
\mathbf{I}(\boldsymbol{\theta})_{ij} = -\mathbb{E}\left[\frac{\partial^2 \log p}{\partial \theta_i \partial \theta_j}\right]
$$

For $X \sim \mathcal{N}(\mu, \sigma^2)$ with $\boldsymbol{\theta} = (\mu, \sigma^2)$:

$$
\mathbf{I}(\mu, \sigma^2) = \begin{pmatrix}
\frac{1}{\sigma^2} & 0 \\
0 & \frac{1}{2\sigma^4}
\end{pmatrix}
$$

The diagonal structure indicates that $\mu$ and $\sigma^2$ are **orthogonal** — information about one doesn't inform the other.

### Properties

**Additivity.** For $n$ i.i.d. observations, $I_n(\theta) = n \cdot I(\theta)$. More data leads to more precise estimates.

**Reparameterization.** If $\eta = g(\theta)$ is a one-to-one transformation: $I_\eta(\eta) = I_\theta(\theta) \cdot (d\theta/d\eta)^2$. For multiple parameters: $\mathbf{I}_\eta = \mathbf{J}^T \mathbf{I}_\theta \mathbf{J}$ where $\mathbf{J}$ is the Jacobian.

### Cramér–Rao Lower Bound

The Fisher Information sets a fundamental limit on how precise any unbiased estimator can be.

**Theorem.** For any unbiased estimator $\hat{\theta}$ of $\theta$:

$$
\boxed{\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}}
$$

The bound is achieved by **efficient estimators**, and MLE is asymptotically efficient.

With $n$ i.i.d. samples, the bound becomes $\text{Var}(\hat{\theta}) \geq 1/(nI(\theta))$.

### Observed vs. Expected Fisher Information

The **expected** Fisher Information $I(\theta) = -\mathbb{E}[\partial^2 \ell / \partial \theta^2]$ is evaluated at the true parameter (used for theoretical analysis). The **observed** Fisher Information $J(\hat{\theta}) = -\partial^2 \ell / \partial \theta^2 \big|_{\theta = \hat{\theta}}$ is computed at the MLE (used in practice for confidence intervals).

## Asymptotic Properties of MLE

Under regularity conditions, the MLE possesses remarkable properties as sample size grows. These explain why MLE is the workhorse of statistical estimation and why neural network training works so well with large datasets.

!!! abstract "Key Asymptotic Properties"
    Under regularity conditions, as $n \to \infty$:
    
    1. **Consistency**: $\hat{\theta}_n \xrightarrow{p} \theta_0$
    2. **Asymptotic Normality**: $\sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} \mathcal{N}(0, I(\theta_0)^{-1})$
    3. **Efficiency**: MLE achieves the Cramér–Rao lower bound asymptotically
    4. **Invariance**: MLE of $g(\theta)$ is $g(\hat{\theta})$

### Regularity Conditions

The asymptotic properties hold under:

1. **Identifiability**: $\theta_1 \neq \theta_2 \implies p(x|\theta_1) \neq p(x|\theta_2)$
2. **Common support**: The support of $p(x|\theta)$ doesn't depend on $\theta$
3. **Differentiability**: $\log p(x|\theta)$ is three times differentiable in $\theta$
4. **Bounded derivatives**: Third derivatives are bounded by an integrable function
5. **Open parameter space**: True parameter $\theta_0$ is in the interior of $\Theta$

!!! warning "When Regularity Fails"
    Some important distributions violate these conditions: Uniform$[0, \theta]$ (support depends on $\theta$), mixture models (multiple local maxima), and boundary cases (parameter on boundary of $\Theta$).

### Consistency

An estimator $\hat{\theta}_n$ is **consistent** if $\hat{\theta}_n \xrightarrow{p} \theta_0$ as $n \to \infty$. The key insight is that maximizing the log-likelihood is equivalent to minimizing the KL divergence. By the Law of Large Numbers:

$$
\frac{1}{n}\sum_{i=1}^{n} \log p(x_i | \theta) \xrightarrow{p} \mathbb{E}_{\theta_0}[\log p(X | \theta)]
$$

and $\mathbb{E}_{\theta_0}[\log p(X | \theta)]$ is maximized at $\theta = \theta_0$ (information inequality).

### Asymptotic Normality

Under regularity conditions:

$$
\sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} \mathcal{N}\left(0, I(\theta_0)^{-1}\right)
$$

??? info "Derivation Sketch"
    1. **Taylor expand** the score around $\theta_0$: $s(\hat{\theta}) = s(\theta_0) + (\hat{\theta} - \theta_0) s'(\tilde{\theta})$
    2. **At the MLE**, $s(\hat{\theta}) = 0$, so $\sqrt{n}(\hat{\theta} - \theta_0) = -\frac{\sqrt{n} \cdot s(\theta_0)/n}{s'(\tilde{\theta})/n}$
    3. **By CLT**: $\sqrt{n} \cdot \bar{s}(\theta_0) \xrightarrow{d} \mathcal{N}(0, I(\theta_0))$
    4. **By LLN**: $s'(\tilde{\theta})/n \xrightarrow{p} -I(\theta_0)$
    5. **By Slutsky's theorem**: The ratio converges to $\mathcal{N}(0, I(\theta_0)^{-1})$

For parameter vectors: $\sqrt{n}(\hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta}_0) \xrightarrow{d} \mathcal{N}(\mathbf{0}, \mathbf{I}(\boldsymbol{\theta}_0)^{-1})$.

### Efficiency

Among all consistent and asymptotically normal estimators, MLE has the **smallest asymptotic variance**:

$$
\text{Avar}(\hat{\theta}_{\text{MLE}}) = \frac{1}{I(\theta_0)} \leq \text{Avar}(\hat{\theta}_{\text{other}})
$$

The **asymptotic relative efficiency** (ARE) compares two estimators: $\text{ARE}(\hat{\theta}_1, \hat{\theta}_2) = \text{Avar}(\hat{\theta}_2) / \text{Avar}(\hat{\theta}_1)$. For the normal mean, the sample median has ARE $\approx 2/\pi \approx 0.637$ relative to the MLE (sample mean).

### Invariance Property

If $\hat{\theta}$ is the MLE of $\theta$, then for any function $g$: $\widehat{g(\theta)} = g(\hat{\theta})$. For example, if the MLE of $\mu$ is $\bar{x}$, then the MLE of $e^\mu$ is $e^{\bar{x}}$.

!!! warning "Bias from Invariance"
    While invariance is convenient, $g(\hat{\theta})$ may be biased even if $\hat{\theta}$ is unbiased. For example, $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$ is biased even though it's the MLE.

### Convergence Rate

Under regularity conditions, $\|\hat{\theta}_n - \theta_0\| = O_p(n^{-1/2})$. This means to halve the estimation error you need 4× more data; to achieve 10× more precision, you need 100× more data.

### Confidence Intervals

**Wald interval** (from asymptotic normality):

$$
\hat{\theta} \pm z_{\alpha/2} \sqrt{\frac{1}{nI(\hat{\theta})}}
$$

**Profile likelihood interval** (from likelihood ratio statistic): $\{\theta : 2[\ell(\hat{\theta}) - \ell(\theta)] \leq \chi^2_{1, \alpha}\}$. The profile likelihood interval is often preferred for small samples.

## Connection to Deep Learning

The relationship between MLE and deep learning loss functions is fundamental:

$$
\text{Loss}(\theta) = -\ell(\theta | \mathbf{X}) = -\log L(\theta | \mathbf{X})
$$

This means: minimizing loss = maximizing likelihood; gradient descent on loss = gradient ascent on log-likelihood; cross-entropy loss = negative log-likelihood for classification; and MSE loss = negative log-likelihood for Gaussian regression.

!!! important "The Deep Learning Connection"
    When you train a neural network by minimizing cross-entropy or MSE, you are performing maximum likelihood estimation. The only difference is that the model $p(y|x, \theta)$ is parameterized by a neural network.

The asymptotic properties also explain deep learning phenomena: consistency justifies using large datasets; asymptotic normality enables uncertainty quantification; and the $\sqrt{n}$ convergence rate governs the data-efficiency tradeoff.

### Applications in Optimization

**Natural gradient descent** accounts for the geometry of the parameter space using the Fisher Information Matrix:

$$
\theta \leftarrow \theta - \alpha \mathbf{I}(\theta)^{-1} \nabla_\theta L
$$

The Fisher Information Matrix approximates the Hessian for networks trained with cross-entropy loss ($\mathbf{I}(\theta) \approx \mathbf{H}(\theta)$), justifying methods like K-FAC (Kronecker-Factored Approximate Curvature) for efficient second-order optimization.

## PyTorch Implementation

### Analytical and Gradient-Based MLE

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
    epsilon = 1e-8
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
    
    Demonstrates the connection between MLE and optimization
    that underlies all of deep learning.
    """
    # Initialize parameter (sigmoid parameterization for unconstrained optimization)
    logit_p = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.Adam([logit_p], lr=lr)
    
    history = []
    for i in range(n_iter):
        p = torch.sigmoid(logit_p)
        nll = -compute_log_likelihood(data, p)  # Minimize negative log-likelihood
        
        optimizer.zero_grad()
        nll.backward()
        optimizer.step()
        
        history.append(p.item())
    
    return torch.sigmoid(logit_p).item(), history

# --- Example usage ---
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

### MLE for Common Distributions

```python
def bernoulli_mle(data: torch.Tensor) -> float:
    """MLE for Bernoulli parameter p."""
    return data.mean().item()

def categorical_mle(data: torch.Tensor, num_categories: int) -> torch.Tensor:
    """MLE for categorical distribution: p_k = n_k / n."""
    counts = torch.bincount(data.long(), minlength=num_categories).float()
    return counts / counts.sum()

def poisson_mle(data: torch.Tensor) -> float:
    """MLE for Poisson rate parameter λ."""
    return data.float().mean().item()

def normal_mle(data: torch.Tensor) -> tuple:
    """MLE for Normal distribution (biased variance estimator)."""
    mu_hat = data.mean()
    sigma_hat = data.std(unbiased=False)
    return mu_hat.item(), sigma_hat.item()

def exponential_mle(data: torch.Tensor) -> float:
    """MLE for Exponential rate parameter λ."""
    return 1.0 / data.mean().item()

def uniform_mle(data: torch.Tensor) -> tuple:
    """MLE for Uniform distribution parameters."""
    return data.min().item(), data.max().item()

def multivariate_normal_mle(data: torch.Tensor) -> tuple:
    """MLE for Multivariate Normal: mean vector and covariance matrix."""
    n = data.shape[0]
    mu_hat = data.mean(dim=0)
    centered = data - mu_hat
    sigma_hat = (centered.T @ centered) / n
    return mu_hat, sigma_hat
```

### Generic Gradient-Based MLE

```python
def gradient_mle(data: torch.Tensor, 
                 log_likelihood_fn: callable,
                 init_params: dict,
                 lr: float = 0.01,
                 n_iter: int = 1000) -> dict:
    """
    Generic gradient-based MLE using PyTorch autograd.
    
    For distributions without closed-form MLEs, or for learning purposes.
    """
    params = {k: v.clone().requires_grad_(True) for k, v in init_params.items()}
    optimizer = torch.optim.Adam(params.values(), lr=lr)
    
    for i in range(n_iter):
        nll = -log_likelihood_fn(data, **params)
        optimizer.zero_grad()
        nll.backward()
        optimizer.step()
    
    return {k: v.detach() for k, v in params.items()}
```

### Fisher Information and Cramér–Rao Bound

```python
def compute_fisher_information_matrix(data: torch.Tensor,
                                      log_likelihood_fn: callable,
                                      params: torch.Tensor,
                                      eps: float = 1e-4) -> torch.Tensor:
    """
    Numerically compute Fisher Information Matrix.
    
    Uses the observed Fisher Information (negative Hessian at MLE)
    via finite differences: I_ij = -∂²ℓ/∂θ_i∂θ_j
    """
    n_params = len(params)
    hessian = torch.zeros(n_params, n_params)
    
    for i in range(n_params):
        for j in range(n_params):
            params_pp = params.clone(); params_pp[i] += eps; params_pp[j] += eps
            params_pm = params.clone(); params_pm[i] += eps; params_pm[j] -= eps
            params_mp = params.clone(); params_mp[i] -= eps; params_mp[j] += eps
            params_mm = params.clone(); params_mm[i] -= eps; params_mm[j] -= eps
            
            ll_pp = log_likelihood_fn(data, params_pp)
            ll_pm = log_likelihood_fn(data, params_pm)
            ll_mp = log_likelihood_fn(data, params_mp)
            ll_mm = log_likelihood_fn(data, params_mm)
            
            hessian[i, j] = (ll_pp - ll_pm - ll_mp + ll_mm) / (4 * eps**2)
    
    return -hessian

def verify_cramer_rao(true_p: float = 0.3, n: int = 100, n_simulations: int = 10000):
    """Verify that MLE variance matches Cramér–Rao bound for Bernoulli."""
    torch.manual_seed(42)
    
    estimates = []
    for _ in range(n_simulations):
        data = (torch.rand(n) < true_p).float()
        estimates.append(data.mean().item())
    
    empirical_var = np.var(estimates)
    cramer_rao = true_p * (1 - true_p) / n
    
    print(f"True p: {true_p}, Sample size: {n}")
    print(f"Cramér–Rao bound: {cramer_rao:.6f}")
    print(f"Empirical variance: {empirical_var:.6f}")
    print(f"Ratio (should be ≈ 1): {empirical_var / cramer_rao:.4f}")
```

### Demonstrating Asymptotic Properties

```python
def demonstrate_consistency(true_theta: float = 0.7, 
                           sample_sizes: list = None,
                           n_simulations: int = 1000):
    """Show that MLE concentrates around true value as n increases."""
    if sample_sizes is None:
        sample_sizes = [10, 50, 100, 500, 1000, 5000]
    
    torch.manual_seed(42)
    
    print("MLE Consistency Demonstration")
    print("-" * 50)
    print(f"{'n':>8} {'Mean MLE':>12} {'Std MLE':>12} {'|Bias|':>12}")
    print("-" * 50)
    
    for n in sample_sizes:
        estimates = []
        for _ in range(n_simulations):
            data = (torch.rand(n) < true_theta).float()
            estimates.append(data.mean().item())
        
        mean_est = np.mean(estimates)
        std_est = np.std(estimates)
        bias = abs(mean_est - true_theta)
        print(f"{n:>8} {mean_est:>12.6f} {std_est:>12.6f} {bias:>12.6f}")

def demonstrate_asymptotic_normality(true_theta: float = 0.7, 
                                     n: int = 100,
                                     n_simulations: int = 5000):
    """Compare empirical MLE distribution to theoretical normal approximation."""
    from scipy.stats import norm
    
    torch.manual_seed(42)
    
    fisher_info = 1 / (true_theta * (1 - true_theta))
    asymptotic_var = 1 / (n * fisher_info)
    asymptotic_std = np.sqrt(asymptotic_var)
    
    estimates = []
    for _ in range(n_simulations):
        data = (torch.rand(n) < true_theta).float()
        estimates.append(data.mean().item())
    
    estimates = np.array(estimates)
    
    print(f"Asymptotic Normality Check (n = {n}):")
    print(f"  Theoretical std: {asymptotic_std:.6f}")
    print(f"  Empirical std:   {np.std(estimates):.6f}")
    print(f"  Ratio (≈ 1):    {np.std(estimates)/asymptotic_std:.4f}")
```

### Visualizing the Likelihood Function

```python
def plot_likelihood_analysis(data: torch.Tensor, true_p: float):
    """Visualize likelihood function, normalized likelihood, and gradient descent convergence."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    p_values = np.linspace(0.01, 0.99, 200)
    
    # Log-Likelihood
    log_liks = [compute_log_likelihood(data, torch.tensor(p)).item() for p in p_values]
    
    ax = axes[0]
    ax.plot(p_values, log_liks, 'b-', linewidth=2)
    ax.axvline(true_p, color='green', linestyle='--', label=f'True p = {true_p}')
    ax.axvline(analytical_mle(data), color='red', linestyle='-', 
               label=f'MLE = {analytical_mle(data):.3f}')
    ax.set_xlabel('p'); ax.set_ylabel('Log-Likelihood')
    ax.set_title('Log-Likelihood Function')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # Normalized Likelihood
    liks = np.exp(np.array(log_liks) - max(log_liks))
    ax = axes[1]
    ax.plot(p_values, liks, 'b-', linewidth=2)
    ax.axvline(true_p, color='green', linestyle='--')
    ax.axvline(analytical_mle(data), color='red', linestyle='-')
    ax.fill_between(p_values, liks, alpha=0.3)
    ax.set_xlabel('p'); ax.set_ylabel('Normalized Likelihood')
    ax.set_title('Likelihood Function')
    ax.grid(True, alpha=0.3)
    
    # Gradient Descent Convergence
    _, history = gradient_based_mle(data)
    ax = axes[2]
    ax.plot(history, 'b-', linewidth=2)
    ax.axhline(true_p, color='green', linestyle='--', label='True p')
    ax.axhline(analytical_mle(data), color='red', linestyle='-', label='MLE')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Estimated p')
    ax.set_title('Gradient Descent Convergence')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

## Exercises

1. **Analytical Practice**: Derive the MLE for the negative binomial distribution.

2. **Biased Variance**: Prove that the MLE for normal variance is biased and calculate the bias explicitly.

3. **Laplace Median**: Show that for the Laplace distribution $p(x|\mu, b) = \frac{1}{2b}e^{-|x-\mu|/b}$, the MLE for $\mu$ is the sample median.

4. **Non-Regular MLE**: Prove that the MLE for Uniform$[0, \theta]$ is consistent but not asymptotically normal (violates regularity).

5. **Sample Size Calculation**: Calculate the sample size needed to estimate a Bernoulli $p$ within $\pm 0.02$ with 95% confidence.

6. **Fisher Information**: Derive the Fisher Information for the Beta distribution $\text{Beta}(\alpha, \beta)$.

7. **Efficiency**: Prove that the MLE for Bernoulli $p$ achieves the Cramér–Rao bound exactly (is efficient).

8. **Implementation**: Implement gradient-based MLE for the Beta distribution and compare to the method of moments estimator.

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 1.2.4
- Casella, G. & Berger, R. L. (2002). *Statistical Inference*, 2nd Edition. Chapters 7, 10
- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd Edition
- Lehmann, E. L. & Casella, G. (1998). *Theory of Point Estimation*, 2nd Edition
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. Chapter 4.2
- van der Vaart, A. W. (1998). *Asymptotic Statistics*
