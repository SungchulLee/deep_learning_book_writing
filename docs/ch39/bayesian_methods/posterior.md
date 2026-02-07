# Posterior Inference in Bayesian Neural Networks

**Posterior inference** is the central computational challenge in Bayesian neural networks. Given a prior $p(\theta)$ and likelihood $p(\mathcal{D} \mid \theta)$, we seek the posterior distribution $p(\theta \mid \mathcal{D})$. For neural networks, this posterior is intractable, necessitating approximate inference methods ranging from sampling (MCMC) to optimization (variational inference) to implicit approximations (dropout, ensembles).

---

## The Inference Challenge

### The Posterior Distribution

By Bayes' theorem, the posterior over network weights is:

$$
\boxed{p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{p(\mathcal{D})}}
$$

where:
- $p(\mathcal{D} \mid \theta) = \prod_{i=1}^N p(y_i \mid x_i, \theta)$ is the likelihood
- $p(\theta)$ is the prior
- $p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta$ is the evidence (marginal likelihood)

### Why Exact Inference is Intractable

**1. High dimensionality**: Modern networks have $d = 10^6$ to $10^9$ parameters

**2. Non-conjugacy**: Neural network likelihoods are not conjugate to standard priors:

$$
p(y \mid x, \theta) = \mathcal{N}(y \mid f_\theta(x), \sigma^2)
$$

where $f_\theta$ is a complex nonlinear function.

**3. Intractable normalization**: The evidence integral has no closed form:

$$
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta
$$

**4. Multimodality**: The posterior landscape has many modes due to:
- Weight space symmetries (permutation, scaling)
- Multiple good solutions
- Complex loss surfaces

### Desiderata for Inference Methods

| Property | Description |
|----------|-------------|
| **Scalability** | Handle millions of parameters |
| **Accuracy** | Capture posterior shape faithfully |
| **Uncertainty** | Produce well-calibrated uncertainty |
| **Efficiency** | Reasonable computation time |
| **Simplicity** | Easy to implement and tune |

No single method excels at all criteria—different methods trade off these properties.

---

## Overview of Inference Methods

### Taxonomy

```
Posterior Inference Methods
├── Sampling Methods (MCMC)
│   ├── Metropolis-Hastings
│   ├── Hamiltonian Monte Carlo (HMC)
│   ├── Stochastic Gradient MCMC
│   │   ├── SGLD (Langevin Dynamics)
│   │   ├── SGHMC (Hamiltonian)
│   │   └── SGFS (Fisher Scoring)
│   └── Ensemble Methods
├── Variational Inference
│   ├── Mean-Field VI
│   ├── Full-Covariance VI
│   └── Normalizing Flows
├── Laplace Approximation
│   ├── Full Laplace
│   ├── Diagonal Laplace
│   └── KFAC Laplace
└── Implicit Methods
    ├── MC Dropout
    ├── Deep Ensembles
    └── SWAG
```

### Method Comparison

| Method | Accuracy | Scalability | Simplicity | Cost |
|--------|----------|-------------|------------|------|
| HMC | High | Low | Low | Very High |
| SGLD | Medium | High | Medium | Low |
| Mean-Field VI | Low-Medium | High | Medium | Medium |
| Laplace | Medium | Medium | High | Low |
| MC Dropout | Low-Medium | Very High | Very High | Very Low |
| Deep Ensembles | Medium-High | High | High | Medium |

---

## Markov Chain Monte Carlo (MCMC)

### Fundamentals

MCMC constructs a Markov chain whose stationary distribution is the posterior:

$$
\theta^{(t+1)} \sim T(\theta^{(t+1)} \mid \theta^{(t)})
$$

such that $\theta^{(t)} \to p(\theta \mid \mathcal{D})$ as $t \to \infty$.

**Using samples**: Given samples $\{\theta^{(t)}\}_{t=1}^T$, approximate expectations:

$$
\mathbb{E}_{p(\theta \mid \mathcal{D})}[f(\theta)] \approx \frac{1}{T} \sum_{t=1}^T f(\theta^{(t)})
$$

### Metropolis-Hastings

**Algorithm**:
1. Propose $\theta' \sim q(\theta' \mid \theta^{(t)})$
2. Compute acceptance probability:
$$
\alpha = \min\left(1, \frac{p(\theta' \mid \mathcal{D}) \, q(\theta^{(t)} \mid \theta')}{p(\theta^{(t)} \mid \mathcal{D}) \, q(\theta' \mid \theta^{(t)})}\right)
$$
3. Accept with probability $\alpha$: $\theta^{(t+1)} = \theta'$ or $\theta^{(t+1)} = \theta^{(t)}$

**For neural networks**: Random walk proposals ($q(\theta' \mid \theta) = \mathcal{N}(\theta, \epsilon^2 I)$) are inefficient in high dimensions.

### Hamiltonian Monte Carlo (HMC)

HMC uses gradient information to make informed proposals.

**Augmented system**: Introduce momentum $\rho$ and define Hamiltonian:

$$
H(\theta, \rho) = -\log p(\theta \mid \mathcal{D}) + \frac{1}{2}\rho^\top M^{-1} \rho
$$

**Hamiltonian dynamics**:

$$
\frac{d\theta}{dt} = M^{-1} \rho, \quad \frac{d\rho}{dt} = \nabla_\theta \log p(\theta \mid \mathcal{D})
$$

**Leapfrog integrator** (for $L$ steps with step size $\epsilon$):

$$
\rho^{(t+1/2)} = \rho^{(t)} + \frac{\epsilon}{2} \nabla_\theta \log p(\theta^{(t)} \mid \mathcal{D})
$$

$$
\theta^{(t+1)} = \theta^{(t)} + \epsilon \, M^{-1} \rho^{(t+1/2)}
$$

$$
\rho^{(t+1)} = \rho^{(t+1/2)} + \frac{\epsilon}{2} \nabla_\theta \log p(\theta^{(t+1)} \mid \mathcal{D})
$$

**HMC Algorithm**:
1. Sample momentum: $\rho \sim \mathcal{N}(0, M)$
2. Run leapfrog for $L$ steps
3. Accept/reject with Metropolis correction

**Advantages**: Explores posterior efficiently, low correlation between samples.

**Challenges for NNs**: Requires full gradient computation (expensive for large datasets).

---

## Stochastic Gradient MCMC

### Motivation

Full-batch gradients are expensive. Stochastic gradient MCMC uses minibatch gradients:

$$
\nabla_\theta \log p(\theta \mid \mathcal{D}) \approx \nabla_\theta \log p(\theta) + \frac{N}{|B|} \sum_{i \in B} \nabla_\theta \log p(y_i \mid x_i, \theta)
$$

where $B$ is a minibatch of size $|B|$.

### Stochastic Gradient Langevin Dynamics (SGLD)

**Update rule**:

$$
\boxed{\theta^{(t+1)} = \theta^{(t)} + \frac{\epsilon_t}{2} \nabla_\theta \log p(\theta^{(t)} \mid \mathcal{D}) + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \epsilon_t I)}
$$

**Key insight**: With decreasing step size $\epsilon_t \to 0$, the Metropolis acceptance step can be skipped.

**Step size schedule**: Must satisfy:

$$
\sum_{t=1}^\infty \epsilon_t = \infty, \quad \sum_{t=1}^\infty \epsilon_t^2 < \infty
$$

Common choice: $\epsilon_t = a(b + t)^{-\gamma}$ with $\gamma \in (0.5, 1]$.

**Practical SGLD**:

$$
\theta^{(t+1)} = \theta^{(t)} + \frac{\epsilon_t}{2} \left[ \nabla_\theta \log p(\theta^{(t)}) + \frac{N}{|B|} \sum_{i \in B} \nabla_\theta \log p(y_i \mid x_i, \theta^{(t)}) \right] + \eta_t
$$

### Stochastic Gradient Hamiltonian Monte Carlo (SGHMC)

Adds momentum to SGLD for better exploration:

$$
\theta^{(t+1)} = \theta^{(t)} + \rho^{(t)}
$$

$$
\rho^{(t+1)} = (1 - \alpha) \rho^{(t)} + \epsilon_t \nabla_\theta \log p(\theta^{(t)} \mid \mathcal{D}) + \eta_t
$$

where $\alpha$ is a friction coefficient and $\eta_t \sim \mathcal{N}(0, 2\alpha\epsilon_t I)$.

### Preconditioned SGLD

Use preconditioning matrix $G(\theta)$ for better scaling:

$$
\theta^{(t+1)} = \theta^{(t)} + \frac{\epsilon_t}{2} \left[ G(\theta^{(t)}) \nabla_\theta \log p(\theta^{(t)} \mid \mathcal{D}) + \Gamma(\theta^{(t)}) \right] + \eta_t
$$

where $\eta_t \sim \mathcal{N}(0, \epsilon_t G(\theta^{(t)}))$ and $\Gamma$ is a correction term.

**Common choices for $G$**:
- RMSprop preconditioner
- Adam preconditioner
- Fisher information matrix

### Cyclical SGLD

Use cyclical learning rates to escape local modes:

$$
\epsilon_t = \epsilon_0 \left( \cos\left(\frac{\pi \, \text{mod}(t, T_{\text{cycle}})}{T_{\text{cycle}}}\right) + 1 \right) / 2
$$

Collect samples at the end of each cycle when step size is small.

---

## Laplace Approximation

### Concept

Approximate the posterior with a Gaussian centered at the MAP estimate:

$$
\boxed{p(\theta \mid \mathcal{D}) \approx q(\theta) = \mathcal{N}(\theta \mid \hat{\theta}_{\text{MAP}}, \Sigma)}
$$

where:
- $\hat{\theta}_{\text{MAP}} = \arg\max_\theta \log p(\theta \mid \mathcal{D})$
- $\Sigma = \left[ -\nabla^2_\theta \log p(\theta \mid \mathcal{D}) \big|_{\hat{\theta}_{\text{MAP}}} \right]^{-1}$

### Derivation

Taylor expand the log posterior around the MAP:

$$
\log p(\theta \mid \mathcal{D}) \approx \log p(\hat{\theta} \mid \mathcal{D}) - \frac{1}{2}(\theta - \hat{\theta})^\top H (\theta - \hat{\theta})
$$

where $H = -\nabla^2_\theta \log p(\theta \mid \mathcal{D})|_{\hat{\theta}}$ is the Hessian.

Exponentiating gives a Gaussian with covariance $\Sigma = H^{-1}$.

### Hessian Computation

**Full Hessian**: $O(d^2)$ storage, $O(d^3)$ inversion — intractable for large networks.

**Diagonal approximation**:

$$
\Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)
$$

where $\sigma_i^2 = 1/H_{ii}$.

**Kronecker-Factored (KFAC)**:

For layer $l$ with weights $W^{(l)}$:

$$
H^{(l)} \approx A^{(l)} \otimes G^{(l)}
$$

where:
- $A^{(l)} = \mathbb{E}[a^{(l-1)} (a^{(l-1)})^\top]$ (input activations)
- $G^{(l)} = \mathbb{E}[g^{(l)} (g^{(l)})^\top]$ (output gradients)

**Inversion**:

$$
(A \otimes G)^{-1} = A^{-1} \otimes G^{-1}
$$

Reduces $O(d^3)$ to $O(n_l^3 + n_{l-1}^3)$ per layer.

### Last-Layer Laplace

Apply Laplace only to the last layer, keeping earlier layers fixed:

$$
p(\theta_L \mid \mathcal{D}, \theta_{1:L-1}) \approx \mathcal{N}(\theta_L \mid \hat{\theta}_L, \Sigma_L)
$$

**Advantages**:
- Much smaller Hessian
- Often captures most uncertainty
- Feature extractor remains deterministic

### Predictive Distribution

For regression with Gaussian likelihood:

$$
p(y^* \mid x^*, \mathcal{D}) = \int p(y^* \mid x^*, \theta) \, q(\theta) \, d\theta
$$

**Linearization** around MAP:

$$
f_\theta(x) \approx f_{\hat{\theta}}(x) + J_{\hat{\theta}}(x)(\theta - \hat{\theta})
$$

where $J_{\hat{\theta}}(x) = \nabla_\theta f_\theta(x)|_{\hat{\theta}}$ is the Jacobian.

**Predictive variance**:

$$
\text{Var}[f(x^*)] \approx J_{\hat{\theta}}(x^*)^\top \Sigma \, J_{\hat{\theta}}(x^*)
$$

---

## Variational Inference

### The Variational Objective

Approximate $p(\theta \mid \mathcal{D})$ with a tractable distribution $q_\phi(\theta)$ by minimizing KL divergence:

$$
\phi^* = \arg\min_\phi \text{KL}(q_\phi(\theta) \| p(\theta \mid \mathcal{D}))
$$

**Evidence Lower Bound (ELBO)**:

$$
\boxed{\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}[\log p(\mathcal{D} \mid \theta)] - \text{KL}(q_\phi(\theta) \| p(\theta))}
$$

**Derivation**:

$$
\log p(\mathcal{D}) = \mathcal{L}(\phi) + \text{KL}(q_\phi \| p(\theta \mid \mathcal{D})) \geq \mathcal{L}(\phi)
$$

Maximizing ELBO is equivalent to minimizing KL to the posterior.

### Mean-Field Variational Inference

**Factorized approximation**:

$$
q_\phi(\theta) = \prod_{i=1}^d q_{\phi_i}(\theta_i)
$$

**Gaussian mean-field**:

$$
q_\phi(\theta) = \prod_{i=1}^d \mathcal{N}(\theta_i \mid \mu_i, \sigma_i^2)
$$

Parameters: $\phi = \{\mu_i, \sigma_i\}_{i=1}^d$ (or $\log \sigma_i$ for positivity).

### Reparameterization Trick

To compute gradients through stochastic sampling:

$$
\theta = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

**Gradient of ELBO**:

$$
\nabla_\phi \mathcal{L} = \nabla_\phi \mathbb{E}_{\epsilon}[\log p(\mathcal{D} \mid \mu + \sigma \odot \epsilon)] - \nabla_\phi \text{KL}(q_\phi \| p)
$$

The first term is estimated via Monte Carlo; the second often has closed form.

### KL Divergence for Gaussian Priors

For $q(\theta) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ and $p(\theta) = \mathcal{N}(0, \sigma_0^2 I)$:

$$
\text{KL}(q \| p) = \frac{1}{2} \sum_{i=1}^d \left[ \frac{\mu_i^2 + \sigma_i^2}{\sigma_0^2} - 1 - \log\frac{\sigma_i^2}{\sigma_0^2} \right]
$$

### Bayes by Backprop

**Algorithm** (Blundell et al., 2015):

1. Sample $\epsilon \sim \mathcal{N}(0, I)$
2. Compute $\theta = \mu + \log(1 + e^\rho) \odot \epsilon$ (softplus for $\sigma$)
3. Compute loss: $\mathcal{L} = \log q_\phi(\theta) - \log p(\theta) - \log p(\mathcal{D} \mid \theta)$
4. Backpropagate and update $\phi = \{\mu, \rho\}$

**Minibatch ELBO**:

$$
\mathcal{L} \approx \frac{N}{|B|} \sum_{i \in B} \log p(y_i \mid x_i, \theta) - \text{KL}(q_\phi \| p)
$$

### Beyond Mean-Field

**Full covariance**: $q(\theta) = \mathcal{N}(\mu, \Sigma)$
- $O(d^2)$ parameters — often intractable

**Low-rank plus diagonal**:

$$
\Sigma = D + VV^\top
$$

where $D$ is diagonal and $V \in \mathbb{R}^{d \times r}$ with rank $r \ll d$.

**Normalizing flows**: Transform simple distribution through invertible functions

$$
q(\theta) = q_0(f^{-1}(\theta)) \left| \det \frac{\partial f^{-1}}{\partial \theta} \right|
$$

---

## Implicit Variational Methods

### Deep Ensembles

Train $M$ networks independently with different initializations:

$$
\{\theta^{(m)}\}_{m=1}^M \quad \text{where each } \theta^{(m)} = \arg\min_\theta \mathcal{L}(\theta; \mathcal{D})
$$

**Predictive distribution**:

$$
p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{M} \sum_{m=1}^M p(y^* \mid x^*, \theta^{(m)})
$$

**Interpretation**: Implicit posterior approximation sampling different modes.

**Advantages**:
- Simple to implement
- Embarrassingly parallel
- Often well-calibrated

**Disadvantages**:
- $M\times$ training cost
- $M\times$ storage and inference cost
- Not a proper Bayesian method

### Stochastic Weight Averaging Gaussian (SWAG)

Collect statistics during SGD training:

**Running statistics**:

$$
\bar{\theta} = \frac{1}{T} \sum_{t=1}^T \theta^{(t)}
$$

$$
\bar{\theta^2} = \frac{1}{T} \sum_{t=1}^T (\theta^{(t)})^2
$$

**Diagonal variance**:

$$
\Sigma_{\text{diag}} = \text{diag}(\bar{\theta^2} - \bar{\theta}^2)
$$

**Low-rank component** (from deviations):

$$
D = [\theta^{(t_1)} - \bar{\theta}, \ldots, \theta^{(t_K)} - \bar{\theta}]
$$

**SWAG posterior**:

$$
q(\theta) = \mathcal{N}\left(\bar{\theta}, \frac{1}{2}(\Sigma_{\text{diag}} + \frac{1}{K-1}DD^\top)\right)
$$

### MC Dropout

Use dropout at test time as approximate variational inference:

$$
q(\theta) = \prod_l q(W^{(l)})
$$

where $q(W^{(l)})$ has columns randomly set to zero.

See the dedicated chapter on MC Dropout for details.

---

## Practical Considerations

### Choosing an Inference Method

**Use SGLD when**:
- Need theoretically grounded samples
- Can afford longer training
- Posterior multimodality is important

**Use Laplace approximation when**:
- Have a trained network (post-hoc uncertainty)
- Need quick uncertainty estimates
- Gaussian approximation is reasonable

**Use variational inference when**:
- Need scalable training
- Can specify a reasonable variational family
- Willing to tune hyperparameters

**Use ensembles when**:
- Simplicity is paramount
- Have computational resources for multiple models
- Need robust uncertainty

**Use MC Dropout when**:
- Need minimal code changes
- Already using dropout
- Computational efficiency is critical

### Hyperparameter Considerations

**SGLD**:
- Learning rate schedule (critical)
- Burn-in period
- Thinning interval

**Variational inference**:
- Prior variance $\sigma_0^2$
- KL weight (warm-up schedule)
- Number of MC samples

**Laplace**:
- Hessian approximation (diagonal, KFAC, etc.)
- Prior precision

### Computational Costs

| Method | Training | Inference | Storage |
|--------|----------|-----------|---------|
| MAP (baseline) | $O(1)$ | $O(1)$ | $O(d)$ |
| SGLD | $O(T)$ | $O(S)$ | $O(Sd)$ |
| Mean-Field VI | $O(1)$-$O(2)$ | $O(S)$ | $O(2d)$ |
| Laplace (diag) | $O(1) + O(d)$ | $O(1)$ | $O(2d)$ |
| Laplace (KFAC) | $O(1) + O(\sum n_l^2)$ | $O(1)$ | $O(\sum n_l^2)$ |
| Ensemble ($M$) | $O(M)$ | $O(M)$ | $O(Md)$ |
| MC Dropout | $O(1)$ | $O(S)$ | $O(d)$ |

### Evaluating Inference Quality

**Calibration**: Do predicted uncertainties match empirical errors?

**Negative log-likelihood**: $-\frac{1}{N_{\text{test}}} \sum_i \log p(y_i \mid x_i, \mathcal{D})$

**Coverage**: Fraction of true values in predicted intervals

**OOD detection**: Can uncertainty identify out-of-distribution inputs?

---

## Python Implementation

```python
"""
Posterior Inference for Bayesian Neural Networks

This module provides implementations of various posterior inference methods:
- Stochastic Gradient Langevin Dynamics (SGLD)
- Laplace Approximation
- Mean-Field Variational Inference
- Deep Ensembles
- SWAG
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List, Optional, Dict, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings


# =============================================================================
# Base Classes
# =============================================================================

class BayesianInference(ABC):
    """Abstract base class for Bayesian inference methods."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to data."""
        pass
    
    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty.
        
        Returns
        -------
        mean : ndarray
            Predictive mean
        std : ndarray
            Predictive standard deviation
        """
        pass


# =============================================================================
# Simple Neural Network
# =============================================================================

class SimpleNN:
    """Simple neural network for demonstration."""
    
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = 'tanh'
    ):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        
        if activation == 'tanh':
            self.act_fn = np.tanh
            self.act_grad = lambda x: 1 - np.tanh(x)**2
        elif activation == 'relu':
            self.act_fn = lambda x: np.maximum(x, 0)
            self.act_grad = lambda x: (x > 0).astype(float)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def init_weights(self, scale: float = 1.0) -> Dict[str, np.ndarray]:
        """Initialize weights with He scaling."""
        weights = {}
        for i in range(self.n_layers):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            std = scale * np.sqrt(2.0 / fan_in)
            weights[f'W{i}'] = np.random.randn(fan_in, fan_out) * std
            weights[f'b{i}'] = np.zeros(fan_out)
        return weights
    
    def forward(
        self,
        X: np.ndarray,
        weights: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Forward pass."""
        h = X
        for i in range(self.n_layers):
            h = h @ weights[f'W{i}'] + weights[f'b{i}']
            if i < self.n_layers - 1:
                h = self.act_fn(h)
        return h
    
    def flatten_weights(self, weights: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten weights to vector."""
        return np.concatenate([weights[k].flatten() for k in sorted(weights.keys())])
    
    def unflatten_weights(self, flat: np.ndarray) -> Dict[str, np.ndarray]:
        """Unflatten vector to weights."""
        weights = {}
        idx = 0
        for i in range(self.n_layers):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            
            size_W = fan_in * fan_out
            weights[f'W{i}'] = flat[idx:idx + size_W].reshape(fan_in, fan_out)
            idx += size_W
            
            weights[f'b{i}'] = flat[idx:idx + fan_out]
            idx += fan_out
        
        return weights
    
    def n_params(self) -> int:
        """Total number of parameters."""
        total = 0
        for i in range(self.n_layers):
            total += self.layer_sizes[i] * self.layer_sizes[i + 1]  # W
            total += self.layer_sizes[i + 1]  # b
        return total


# =============================================================================
# Stochastic Gradient Langevin Dynamics (SGLD)
# =============================================================================

class SGLD(BayesianInference):
    """
    Stochastic Gradient Langevin Dynamics for posterior sampling.
    
    Update: θ_{t+1} = θ_t + (ε_t/2) * ∇log p(θ|D) + η_t
    where η_t ~ N(0, ε_t * I)
    """
    
    def __init__(
        self,
        network: SimpleNN,
        prior_std: float = 1.0,
        noise_std: float = 1.0,
        lr_init: float = 0.01,
        lr_decay: float = 0.55,
        n_iterations: int = 10000,
        burn_in: int = 5000,
        thinning: int = 10,
        batch_size: int = 32
    ):
        """
        Parameters
        ----------
        network : SimpleNN
            Neural network architecture
        prior_std : float
            Prior standard deviation on weights
        noise_std : float
            Observation noise standard deviation
        lr_init : float
            Initial learning rate
        lr_decay : float
            Learning rate decay exponent (should be in (0.5, 1])
        n_iterations : int
            Total number of iterations
        burn_in : int
            Number of burn-in iterations
        thinning : int
            Keep every thinning-th sample
        batch_size : int
            Minibatch size
        """
        self.network = network
        self.prior_std = prior_std
        self.noise_std = noise_std
        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.thinning = thinning
        self.batch_size = batch_size
        
        self.samples = []
    
    def _learning_rate(self, t: int) -> float:
        """Compute learning rate at iteration t."""
        return self.lr_init / (1 + t) ** self.lr_decay
    
    def _log_prior_grad(self, theta: np.ndarray) -> np.ndarray:
        """Gradient of log prior (Gaussian)."""
        return -theta / (self.prior_std ** 2)
    
    def _log_likelihood_grad(
        self,
        X: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        N: int
    ) -> np.ndarray:
        """
        Gradient of log likelihood (scaled for minibatch).
        Uses numerical differentiation for simplicity.
        """
        weights = self.network.unflatten_weights(theta)
        pred = self.network.forward(X, weights)
        
        # Numerical gradient
        eps = 1e-5
        grad = np.zeros_like(theta)
        
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            weights_plus = self.network.unflatten_weights(theta_plus)
            pred_plus = self.network.forward(X, weights_plus)
            
            theta_minus = theta.copy()
            theta_minus[i] -= eps
            weights_minus = self.network.unflatten_weights(theta_minus)
            pred_minus = self.network.forward(X, weights_minus)
            
            # Gradient of log likelihood = -1/(2σ²) * d/dθ ||y - f(x,θ)||²
            ll_plus = -0.5 * np.sum((y - pred_plus)**2) / (self.noise_std**2)
            ll_minus = -0.5 * np.sum((y - pred_minus)**2) / (self.noise_std**2)
            
            grad[i] = (ll_plus - ll_minus) / (2 * eps)
        
        # Scale for minibatch
        return grad * (N / len(X))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Run SGLD sampling."""
        N = len(X)
        
        # Initialize
        weights = self.network.init_weights()
        theta = self.network.flatten_weights(weights)
        
        self.samples = []
        self.losses = []
        
        for t in range(self.n_iterations):
            # Get minibatch
            idx = np.random.choice(N, min(self.batch_size, N), replace=False)
            X_batch = X[idx]
            y_batch = y[idx]
            
            # Learning rate
            lr = self._learning_rate(t)
            
            # Compute gradients
            grad_prior = self._log_prior_grad(theta)
            grad_likelihood = self._log_likelihood_grad(X_batch, y_batch, theta, N)
            grad = grad_prior + grad_likelihood
            
            # SGLD update
            noise = np.random.randn(len(theta)) * np.sqrt(lr)
            theta = theta + (lr / 2) * grad + noise
            
            # Store sample
            if t >= self.burn_in and (t - self.burn_in) % self.thinning == 0:
                self.samples.append(theta.copy())
            
            # Track loss
            if t % 100 == 0:
                weights = self.network.unflatten_weights(theta)
                pred = self.network.forward(X, weights)
                loss = np.mean((y - pred)**2)
                self.losses.append(loss)
        
        print(f"SGLD: Collected {len(self.samples)} samples")
    
    def predict(
        self,
        X: np.ndarray,
        n_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using posterior samples."""
        if n_samples is None:
            samples = self.samples
        else:
            idx = np.random.choice(len(self.samples), min(n_samples, len(self.samples)), replace=False)
            samples = [self.samples[i] for i in idx]
        
        predictions = []
        for theta in samples:
            weights = self.network.unflatten_weights(theta)
            pred = self.network.forward(X, weights)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        # Add observation noise
        total_std = np.sqrt(std**2 + self.noise_std**2)
        
        return mean.flatten(), total_std.flatten()


# =============================================================================
# Laplace Approximation
# =============================================================================

class LaplaceApproximation(BayesianInference):
    """
    Laplace approximation for posterior inference.
    
    Approximates posterior as Gaussian centered at MAP estimate.
    Uses diagonal Hessian approximation for scalability.
    """
    
    def __init__(
        self,
        network: SimpleNN,
        prior_std: float = 1.0,
        noise_std: float = 1.0,
        n_iterations: int = 1000,
        lr: float = 0.01
    ):
        """
        Parameters
        ----------
        network : SimpleNN
            Neural network architecture
        prior_std : float
            Prior standard deviation
        noise_std : float
            Observation noise standard deviation
        n_iterations : int
            Number of optimization iterations for MAP
        lr : float
            Learning rate for MAP optimization
        """
        self.network = network
        self.prior_std = prior_std
        self.noise_std = noise_std
        self.n_iterations = n_iterations
        self.lr = lr
        
        self.theta_map = None
        self.hessian_diag = None
    
    def _neg_log_posterior(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Compute negative log posterior."""
        weights = self.network.unflatten_weights(theta)
        pred = self.network.forward(X, weights)
        
        # Log likelihood
        ll = -0.5 * np.sum((y - pred)**2) / (self.noise_std**2)
        ll -= 0.5 * len(y) * np.log(2 * np.pi * self.noise_std**2)
        
        # Log prior
        lp = -0.5 * np.sum(theta**2) / (self.prior_std**2)
        lp -= 0.5 * len(theta) * np.log(2 * np.pi * self.prior_std**2)
        
        return -(ll + lp)
    
    def _numerical_gradient(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        eps: float = 1e-5
    ) -> np.ndarray:
        """Compute gradient numerically."""
        grad = np.zeros_like(theta)
        
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            theta_minus = theta.copy()
            theta_minus[i] -= eps
            
            grad[i] = (
                self._neg_log_posterior(theta_plus, X, y) -
                self._neg_log_posterior(theta_minus, X, y)
            ) / (2 * eps)
        
        return grad
    
    def _numerical_hessian_diag(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        eps: float = 1e-4
    ) -> np.ndarray:
        """Compute diagonal of Hessian numerically."""
        hess_diag = np.zeros_like(theta)
        f0 = self._neg_log_posterior(theta, X, y)
        
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            theta_minus = theta.copy()
            theta_minus[i] -= eps
            
            f_plus = self._neg_log_posterior(theta_plus, X, y)
            f_minus = self._neg_log_posterior(theta_minus, X, y)
            
            hess_diag[i] = (f_plus - 2*f0 + f_minus) / (eps**2)
        
        return hess_diag
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Find MAP estimate and compute Hessian."""
        # Initialize
        weights = self.network.init_weights()
        theta = self.network.flatten_weights(weights)
        
        # Optimize for MAP
        for t in range(self.n_iterations):
            grad = self._numerical_gradient(theta, X, y)
            theta = theta - self.lr * grad
            
            if t % 200 == 0:
                loss = self._neg_log_posterior(theta, X, y)
                if t % 200 == 0:
                    pass  # Silent training
        
        self.theta_map = theta
        
        # Compute diagonal Hessian
        self.hessian_diag = self._numerical_hessian_diag(theta, X, y)
        
        # Ensure positive definite (add small value if needed)
        self.hessian_diag = np.maximum(self.hessian_diag, 1e-6)
        
        # Posterior variance is inverse Hessian
        self.posterior_var = 1.0 / self.hessian_diag
        
        print(f"Laplace: MAP found, posterior variance computed")
    
    def predict(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions by sampling from Laplace posterior."""
        predictions = []
        
        for _ in range(n_samples):
            # Sample from Gaussian posterior
            theta = self.theta_map + np.sqrt(self.posterior_var) * np.random.randn(len(self.theta_map))
            
            weights = self.network.unflatten_weights(theta)
            pred = self.network.forward(X, weights)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        # Add observation noise
        total_std = np.sqrt(std**2 + self.noise_std**2)
        
        return mean.flatten(), total_std.flatten()


# =============================================================================
# Mean-Field Variational Inference
# =============================================================================

class MeanFieldVI(BayesianInference):
    """
    Mean-field variational inference for BNNs.
    
    Approximates posterior with factorized Gaussian:
    q(θ) = ∏_i N(θ_i | μ_i, σ_i²)
    """
    
    def __init__(
        self,
        network: SimpleNN,
        prior_std: float = 1.0,
        noise_std: float = 1.0,
        n_iterations: int = 5000,
        lr: float = 0.01,
        n_mc_samples: int = 1,
        kl_weight: float = 1.0
    ):
        """
        Parameters
        ----------
        network : SimpleNN
            Neural network architecture
        prior_std : float
            Prior standard deviation
        noise_std : float
            Observation noise standard deviation
        n_iterations : int
            Number of optimization iterations
        lr : float
            Learning rate
        n_mc_samples : int
            Number of MC samples for gradient estimation
        kl_weight : float
            Weight on KL term (for KL annealing)
        """
        self.network = network
        self.prior_std = prior_std
        self.noise_std = noise_std
        self.n_iterations = n_iterations
        self.lr = lr
        self.n_mc_samples = n_mc_samples
        self.kl_weight = kl_weight
        
        self.mu = None
        self.log_sigma = None
    
    def _sample_weights(self) -> np.ndarray:
        """Sample weights using reparameterization trick."""
        eps = np.random.randn(len(self.mu))
        sigma = np.exp(self.log_sigma)
        return self.mu + sigma * eps
    
    def _kl_divergence(self) -> float:
        """KL divergence from q to prior."""
        sigma = np.exp(self.log_sigma)
        kl = 0.5 * np.sum(
            (self.mu**2 + sigma**2) / (self.prior_std**2) -
            1 - 2 * self.log_sigma + 2 * np.log(self.prior_std)
        )
        return kl
    
    def _elbo(
        self,
        X: np.ndarray,
        y: np.ndarray,
        N: int
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute ELBO and gradients.
        
        Returns
        -------
        elbo : float
        grad_mu : ndarray
        grad_log_sigma : ndarray
        """
        n_batch = len(X)
        
        # Monte Carlo estimate of expected log likelihood
        total_ll = 0.0
        grad_mu_ll = np.zeros_like(self.mu)
        grad_log_sigma_ll = np.zeros_like(self.log_sigma)
        
        for _ in range(self.n_mc_samples):
            # Sample weights
            eps = np.random.randn(len(self.mu))
            sigma = np.exp(self.log_sigma)
            theta = self.mu + sigma * eps
            
            # Forward pass
            weights = self.network.unflatten_weights(theta)
            pred = self.network.forward(X, weights)
            
            # Log likelihood
            ll = -0.5 * np.sum((y - pred)**2) / (self.noise_std**2)
            total_ll += ll
            
            # Numerical gradients for simplicity
            delta = 1e-5
            
            for i in range(len(self.mu)):
                # Gradient w.r.t. mu
                theta_plus = theta.copy()
                theta_plus[i] += delta
                weights_plus = self.network.unflatten_weights(theta_plus)
                pred_plus = self.network.forward(X, weights_plus)
                ll_plus = -0.5 * np.sum((y - pred_plus)**2) / (self.noise_std**2)
                
                grad_mu_ll[i] += (ll_plus - ll) / delta
                
                # Gradient w.r.t. log_sigma (through reparameterization)
                # d/d(log σ) = d/dθ * dθ/d(log σ) = d/dθ * σ * ε
                grad_log_sigma_ll[i] += (ll_plus - ll) / delta * sigma[i] * eps[i]
        
        total_ll /= self.n_mc_samples
        grad_mu_ll /= self.n_mc_samples
        grad_log_sigma_ll /= self.n_mc_samples
        
        # Scale for full dataset
        scale = N / n_batch
        total_ll *= scale
        grad_mu_ll *= scale
        grad_log_sigma_ll *= scale
        
        # KL divergence and gradients
        kl = self._kl_divergence()
        sigma = np.exp(self.log_sigma)
        grad_mu_kl = self.mu / (self.prior_std**2)
        grad_log_sigma_kl = sigma**2 / (self.prior_std**2) - 1
        
        # ELBO = E[log p(D|θ)] - KL(q||p)
        elbo = total_ll - self.kl_weight * kl
        grad_mu = grad_mu_ll - self.kl_weight * grad_mu_kl
        grad_log_sigma = grad_log_sigma_ll - self.kl_weight * grad_log_sigma_kl
        
        return elbo, grad_mu, grad_log_sigma
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Optimize variational parameters."""
        N = len(X)
        n_params = self.network.n_params()
        
        # Initialize variational parameters
        self.mu = np.random.randn(n_params) * 0.1
        self.log_sigma = np.ones(n_params) * np.log(0.1)
        
        self.elbo_history = []
        
        for t in range(self.n_iterations):
            # Compute ELBO and gradients
            elbo, grad_mu, grad_log_sigma = self._elbo(X, y, N)
            
            # Update
            self.mu += self.lr * grad_mu
            self.log_sigma += self.lr * 0.1 * grad_log_sigma  # Smaller LR for variance
            
            self.elbo_history.append(elbo)
            
            if t % 500 == 0:
                pass  # Silent training
        
        print(f"VI: Optimization complete, final ELBO = {elbo:.2f}")
    
    def predict(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions by sampling from variational posterior."""
        predictions = []
        
        for _ in range(n_samples):
            theta = self._sample_weights()
            weights = self.network.unflatten_weights(theta)
            pred = self.network.forward(X, weights)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        # Add observation noise
        total_std = np.sqrt(std**2 + self.noise_std**2)
        
        return mean.flatten(), total_std.flatten()


# =============================================================================
# Deep Ensembles
# =============================================================================

class DeepEnsemble(BayesianInference):
    """
    Deep ensemble for uncertainty estimation.
    
    Trains M networks independently with different initializations.
    """
    
    def __init__(
        self,
        network: SimpleNN,
        n_members: int = 5,
        noise_std: float = 1.0,
        n_iterations: int = 1000,
        lr: float = 0.01
    ):
        """
        Parameters
        ----------
        network : SimpleNN
            Neural network architecture
        n_members : int
            Number of ensemble members
        noise_std : float
            Observation noise standard deviation
        n_iterations : int
            Training iterations per member
        lr : float
            Learning rate
        """
        self.network = network
        self.n_members = n_members
        self.noise_std = noise_std
        self.n_iterations = n_iterations
        self.lr = lr
        
        self.members = []
    
    def _train_member(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int
    ) -> np.ndarray:
        """Train a single ensemble member."""
        np.random.seed(seed)
        
        weights = self.network.init_weights()
        theta = self.network.flatten_weights(weights)
        
        for t in range(self.n_iterations):
            # Compute gradient (numerical)
            grad = self._numerical_gradient(theta, X, y)
            theta = theta - self.lr * grad
        
        return theta
    
    def _numerical_gradient(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        eps: float = 1e-5
    ) -> np.ndarray:
        """Compute MSE gradient numerically."""
        grad = np.zeros_like(theta)
        
        weights = self.network.unflatten_weights(theta)
        pred = self.network.forward(X, weights)
        loss0 = np.mean((y - pred)**2)
        
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            
            weights_plus = self.network.unflatten_weights(theta_plus)
            pred_plus = self.network.forward(X, weights_plus)
            loss_plus = np.mean((y - pred_plus)**2)
            
            grad[i] = (loss_plus - loss0) / eps
        
        return grad
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train all ensemble members."""
        self.members = []
        
        for m in range(self.n_members):
            theta = self._train_member(X, y, seed=m * 42)
            self.members.append(theta)
        
        print(f"Ensemble: Trained {self.n_members} members")
    
    def predict(
        self,
        X: np.ndarray,
        n_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using ensemble."""
        predictions = []
        
        for theta in self.members:
            weights = self.network.unflatten_weights(theta)
            pred = self.network.forward(X, weights)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        # Add observation noise
        total_std = np.sqrt(std**2 + self.noise_std**2)
        
        return mean.flatten(), total_std.flatten()


# =============================================================================
# Visualization and Evaluation
# =============================================================================

def plot_inference_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_true: np.ndarray,
    methods: Dict[str, BayesianInference],
    figsize: Tuple[float, float] = (15, 4)
):
    """Compare different inference methods visually."""
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, (name, method) in zip(axes, methods.items()):
        mean, std = method.predict(X_test)
        
        ax.fill_between(
            X_test.flatten(),
            mean - 2*std,
            mean + 2*std,
            alpha=0.3,
            label='±2σ'
        )
        ax.plot(X_test, mean, 'b-', linewidth=2, label='Mean')
        ax.plot(X_test, y_true, 'k--', linewidth=1, label='True')
        ax.scatter(X_train, y_train, c='red', s=20, zorder=5, label='Data')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(name)
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()


def evaluate_calibration(
    method: BayesianInference,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """Evaluate calibration of uncertainty estimates."""
    mean, std = method.predict(X_test)
    
    # Compute z-scores
    z = (y_test.flatten() - mean) / std
    
    # Expected coverage at different levels
    coverages = {}
    expected_coverages = [0.5, 0.8, 0.9, 0.95, 0.99]
    
    for p in expected_coverages:
        z_crit = stats.norm.ppf((1 + p) / 2)
        actual = np.mean(np.abs(z) < z_crit)
        coverages[f'coverage_{int(p*100)}'] = actual
    
    # NLL
    nll = -np.mean(stats.norm.logpdf(y_test.flatten(), mean, std))
    
    # RMSE
    rmse = np.sqrt(np.mean((y_test.flatten() - mean)**2))
    
    return {
        'nll': nll,
        'rmse': rmse,
        **coverages
    }


# =============================================================================
# Demo Functions
# =============================================================================

def demo_inference_methods():
    """Compare different inference methods on a simple problem."""
    
    print("=" * 70)
    print("COMPARING POSTERIOR INFERENCE METHODS")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate data
    N = 20
    X_train = np.random.uniform(-3, 3, N).reshape(-1, 1)
    y_train = np.sin(X_train) + np.random.normal(0, 0.2, (N, 1))
    
    X_test = np.linspace(-5, 5, 200).reshape(-1, 1)
    y_true = np.sin(X_test)
    
    print(f"\nTraining data: {N} points")
    print(f"True function: sin(x)")
    
    # Create network
    network = SimpleNN([1, 20, 1], activation='tanh')
    print(f"Network: {network.layer_sizes}, {network.n_params()} parameters")
    
    # Train different methods
    methods = {}
    
    # Laplace
    print("\n--- Laplace Approximation ---")
    laplace = LaplaceApproximation(
        network, prior_std=1.0, noise_std=0.2,
        n_iterations=500, lr=0.05
    )
    laplace.fit(X_train, y_train)
    methods['Laplace'] = laplace
    
    # Ensemble
    print("\n--- Deep Ensemble ---")
    ensemble = DeepEnsemble(
        network, n_members=5, noise_std=0.2,
        n_iterations=500, lr=0.05
    )
    ensemble.fit(X_train, y_train)
    methods['Ensemble'] = ensemble
    
    # Evaluate
    print("\n--- Evaluation ---")
    print(f"{'Method':<15} {'NLL':>8} {'RMSE':>8} {'Cov90%':>8}")
    print("-" * 45)
    
    for name, method in methods.items():
        metrics = evaluate_calibration(method, X_test, y_true)
        print(f"{name:<15} {metrics['nll']:>8.3f} {metrics['rmse']:>8.3f} "
              f"{metrics['coverage_90']:>8.2%}")
    
    return methods


def demo_sgld():
    """Demonstrate SGLD sampling."""
    
    print("\n" + "=" * 70)
    print("STOCHASTIC GRADIENT LANGEVIN DYNAMICS")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Simple 1D problem
    N = 30
    X_train = np.random.uniform(-3, 3, N).reshape(-1, 1)
    y_train = np.sin(X_train) + np.random.normal(0, 0.2, (N, 1))
    
    network = SimpleNN([1, 10, 1], activation='tanh')
    
    print(f"\nRunning SGLD with {network.n_params()} parameters...")
    
    sgld = SGLD(
        network,
        prior_std=1.0,
        noise_std=0.2,
        lr_init=0.001,
        lr_decay=0.55,
        n_iterations=3000,
        burn_in=1500,
        thinning=5,
        batch_size=N  # Full batch for stability
    )
    
    sgld.fit(X_train, y_train)
    
    # Evaluate
    X_test = np.linspace(-5, 5, 100).reshape(-1, 1)
    mean, std = sgld.predict(X_test)
    
    print(f"\nSamples collected: {len(sgld.samples)}")
    print(f"Mean prediction std: {np.mean(std):.3f}")
    
    return sgld


def demo_variational_inference():
    """Demonstrate mean-field variational inference."""
    
    print("\n" + "=" * 70)
    print("MEAN-FIELD VARIATIONAL INFERENCE")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Simple problem
    N = 30
    X_train = np.random.uniform(-3, 3, N).reshape(-1, 1)
    y_train = np.sin(X_train) + np.random.normal(0, 0.2, (N, 1))
    
    # Small network for faster VI
    network = SimpleNN([1, 10, 1], activation='tanh')
    
    print(f"\nRunning VI with {network.n_params()} parameters...")
    print("(This may take a while due to numerical gradients)")
    
    vi = MeanFieldVI(
        network,
        prior_std=1.0,
        noise_std=0.2,
        n_iterations=1000,
        lr=0.01,
        n_mc_samples=1
    )
    
    vi.fit(X_train, y_train)
    
    # Examine learned posterior
    sigma = np.exp(vi.log_sigma)
    print(f"\nPosterior statistics:")
    print(f"  Mean |μ|: {np.mean(np.abs(vi.mu)):.4f}")
    print(f"  Mean σ:   {np.mean(sigma):.4f}")
    print(f"  Max σ:    {np.max(sigma):.4f}")
    
    return vi


if __name__ == "__main__":
    methods = demo_inference_methods()
    sgld = demo_sgld()
    vi = demo_variational_inference()
```

---

## Summary

### Inference Methods Overview

| Method | Approach | Accuracy | Scalability |
|--------|----------|----------|-------------|
| **HMC** | Exact MCMC | High | Low |
| **SGLD** | Stochastic MCMC | Medium-High | High |
| **Laplace** | Gaussian at MAP | Medium | Medium-High |
| **Mean-Field VI** | Factorized optimization | Low-Medium | High |
| **Deep Ensembles** | Multiple MAPs | Medium-High | Medium |
| **MC Dropout** | Implicit VI | Low-Medium | Very High |

### Key Formulas

**Posterior**:
$$
p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) \, p(\theta)
$$

**SGLD Update**:
$$
\theta^{(t+1)} = \theta^{(t)} + \frac{\epsilon_t}{2} \nabla \log p(\theta^{(t)} \mid \mathcal{D}) + \mathcal{N}(0, \epsilon_t I)
$$

**ELBO** (Variational Inference):
$$
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}[\log p(\mathcal{D} \mid \theta)] - \text{KL}(q_\phi \| p)
$$

**Laplace Approximation**:
$$
q(\theta) = \mathcal{N}(\theta_{\text{MAP}}, H^{-1})
$$

### Computational Complexity

| Method | Training | Inference | Storage |
|--------|----------|-----------|---------|
| MAP | $O(E \cdot N)$ | $O(1)$ | $O(d)$ |
| SGLD | $O(T \cdot B)$ | $O(S)$ | $O(Sd)$ |
| Laplace | $O(E \cdot N + d^2)$ | $O(S)$ | $O(d^2)$ |
| VI | $O(E \cdot N)$ | $O(S)$ | $O(2d)$ |
| Ensemble | $O(M \cdot E \cdot N)$ | $O(M)$ | $O(Md)$ |

### Method Selection Guide

| Scenario | Recommended Method |
|----------|-------------------|
| Post-hoc uncertainty | Laplace, SWAG |
| Scalable training | VI, MC Dropout |
| Best uncertainty | HMC (small), SGLD (large) |
| Simple implementation | Ensembles, MC Dropout |
| Limited compute | MC Dropout, Laplace |

### Connections to Other Chapters

| Topic | Chapter | Connection |
|-------|---------|------------|
| Prior specification | Ch13: Prior on Weights | Input to posterior |
| Uncertainty | Ch13: Uncertainty | Posterior enables decomposition |
| MC Dropout | Ch13: MC Dropout | Implicit variational inference |
| Variational BNN | Ch13: Variational BNN | Detailed VI treatment |
| Model comparison | Ch13: Information Criteria | Marginal likelihood |

### Key References

- Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *ICML*.
- Blundell, C., et al. (2015). Weight uncertainty in neural networks. *ICML*.
- MacKay, D. J. (1992). A practical Bayesian framework for backpropagation networks. *Neural Computation*.
- Ritter, H., et al. (2018). A scalable Laplace approximation for neural networks. *ICLR*.
- Lakshminarayanan, B., et al. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
- Maddox, W., et al. (2019). A simple baseline for Bayesian inference in deep learning. *NeurIPS*.
