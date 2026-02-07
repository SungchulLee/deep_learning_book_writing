# Variational Bayesian Neural Networks

**Variational Bayesian Neural Networks** (Variational BNNs) provide a principled framework for approximate posterior inference by casting the intractable Bayesian inference problem as an optimization problem. By maximizing the Evidence Lower Bound (ELBO), we learn a tractable approximating distribution over network weights, enabling scalable uncertainty quantification in deep learning.

---

## Motivation: Scalable Bayesian Inference

### The Posterior Inference Challenge

The true posterior over neural network weights is intractable:

$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{p(\mathcal{D})}
$$

**Challenges**:
- The evidence $p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) p(\theta) d\theta$ has no closed form
- High-dimensional parameter space ($10^6$-$10^9$ parameters)
- Complex, multimodal posterior landscape
- MCMC methods are too slow for large networks

### The Variational Approach

**Key idea**: Approximate the intractable posterior with a tractable distribution:

$$
p(\theta \mid \mathcal{D}) \approx q_\phi(\theta)
$$

where $q_\phi(\theta)$ is from a tractable family (e.g., Gaussian) with parameters $\phi$.

**Optimization objective**: Find $\phi$ that minimizes the KL divergence:

$$
\phi^* = \arg\min_\phi \text{KL}(q_\phi(\theta) \| p(\theta \mid \mathcal{D}))
$$

### Advantages of Variational Inference

| Advantage | Description |
|-----------|-------------|
| **Scalability** | Reduces inference to optimization |
| **Flexibility** | Can choose approximation family |
| **Efficiency** | Amenable to stochastic optimization |
| **Integration** | Works with standard deep learning tools |

---

## The Evidence Lower Bound (ELBO)

### Derivation

Starting from the log evidence:

$$
\log p(\mathcal{D}) = \log \int p(\mathcal{D} \mid \theta) p(\theta) d\theta
$$

Introduce the variational distribution $q_\phi(\theta)$:

$$
\log p(\mathcal{D}) = \log \int \frac{q_\phi(\theta)}{q_\phi(\theta)} p(\mathcal{D} \mid \theta) p(\theta) d\theta
$$

Apply Jensen's inequality:

$$
\log p(\mathcal{D}) \geq \int q_\phi(\theta) \log \frac{p(\mathcal{D} \mid \theta) p(\theta)}{q_\phi(\theta)} d\theta = \mathcal{L}(\phi)
$$

This lower bound $\mathcal{L}(\phi)$ is the **Evidence Lower Bound (ELBO)**.

### ELBO Decomposition

The ELBO can be written as:

$$
\boxed{\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\theta)}[\log p(\mathcal{D} \mid \theta)] - \text{KL}(q_\phi(\theta) \| p(\theta))}
$$

**Interpretation**:
- **First term**: Expected log-likelihood (data fit)
- **Second term**: KL divergence to prior (complexity penalty)

**Alternative form**:

$$
\mathcal{L}(\phi) = \log p(\mathcal{D}) - \text{KL}(q_\phi(\theta) \| p(\theta \mid \mathcal{D}))
$$

Since $\text{KL} \geq 0$, maximizing ELBO is equivalent to minimizing KL to the posterior.

### ELBO as Loss Function

For neural network training, minimize the negative ELBO:

$$
\boxed{\mathcal{L}_{\text{VI}}(\phi) = -\mathbb{E}_{q_\phi(\theta)}[\log p(\mathcal{D} \mid \theta)] + \text{KL}(q_\phi(\theta) \| p(\theta))}
$$

For regression with Gaussian likelihood:

$$
\mathcal{L}_{\text{VI}}(\phi) = \frac{1}{2\sigma^2} \mathbb{E}_{q_\phi}\left[\sum_{i=1}^N (y_i - f_\theta(x_i))^2\right] + \text{KL}(q_\phi \| p)
$$

For classification with categorical likelihood:

$$
\mathcal{L}_{\text{VI}}(\phi) = -\mathbb{E}_{q_\phi}\left[\sum_{i=1}^N \sum_c y_{ic} \log \text{softmax}(f_\theta(x_i))_c\right] + \text{KL}(q_\phi \| p)
$$

---

## Mean-Field Variational Inference

### Factorized Approximation

The **mean-field** assumption factorizes the posterior:

$$
q_\phi(\theta) = \prod_{j=1}^d q_{\phi_j}(\theta_j)
$$

Each parameter has its own independent distribution.

### Gaussian Mean-Field

The most common choice is diagonal Gaussian:

$$
\boxed{q_\phi(\theta) = \prod_{j=1}^d \mathcal{N}(\theta_j \mid \mu_j, \sigma_j^2)}
$$

**Variational parameters**: $\phi = \{(\mu_j, \sigma_j)\}_{j=1}^d$

**Total parameters**: $2d$ (double the original network)

### KL Divergence for Gaussian Distributions

For Gaussian prior $p(\theta) = \mathcal{N}(0, \sigma_p^2 I)$:

$$
\text{KL}(q_\phi \| p) = \frac{1}{2} \sum_{j=1}^d \left[ \frac{\mu_j^2 + \sigma_j^2}{\sigma_p^2} - 1 - \log \frac{\sigma_j^2}{\sigma_p^2} \right]
$$

**Per-parameter KL**:

$$
\text{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, \sigma_p^2)) = \frac{\mu^2 + \sigma^2}{2\sigma_p^2} - \frac{1}{2} - \log \frac{\sigma}{\sigma_p}
$$

### Limitations of Mean-Field

**Independence assumption**: Ignores correlations between weights
- Cannot capture weight interactions
- May underestimate uncertainty
- Posterior covariance is diagonal

**Unimodality**: Gaussian approximation captures single mode
- Cannot represent multimodal posteriors
- May miss important posterior structure

---

## The Reparameterization Trick

### The Gradient Problem

To optimize the ELBO, we need:

$$
\nabla_\phi \mathbb{E}_{q_\phi(\theta)}[\log p(\mathcal{D} \mid \theta)]
$$

**Challenge**: The expectation is over $q_\phi$, which depends on $\phi$.

### Reparameterization Solution

**Key insight**: Express $\theta$ as a deterministic function of $\phi$ and noise:

$$
\theta = g(\phi, \epsilon) = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Now the expectation is over $\epsilon$, independent of $\phi$:

$$
\mathbb{E}_{q_\phi(\theta)}[f(\theta)] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}[f(\mu + \sigma \odot \epsilon)]
$$

### Gradient Computation

The gradient becomes:

$$
\nabla_\phi \mathbb{E}_{q_\phi}[f(\theta)] = \mathbb{E}_{\epsilon}\left[\nabla_\phi f(\mu + \sigma \odot \epsilon)\right]
$$

**Monte Carlo estimate** with single sample:

$$
\nabla_\phi \mathbb{E}_{q_\phi}[f(\theta)] \approx \nabla_\phi f(\mu + \sigma \odot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)
$$

### Practical Implementation

**Parameterization for positivity**: Use $\rho$ such that $\sigma = \log(1 + e^\rho)$ (softplus)

**Gradient flow**:

$$
\frac{\partial \mathcal{L}}{\partial \mu} = \frac{\partial \mathcal{L}}{\partial \theta}, \quad \frac{\partial \mathcal{L}}{\partial \rho} = \frac{\partial \mathcal{L}}{\partial \theta} \cdot \epsilon \cdot \frac{e^\rho}{1 + e^\rho}
$$

---

## Bayes by Backprop

### Algorithm Overview

**Bayes by Backprop** (Blundell et al., 2015) is the foundational algorithm for training variational BNNs:

**Algorithm: Bayes by Backprop**

```
Input: Dataset D, prior p(θ), network architecture
Output: Variational parameters φ = {μ, ρ}

Initialize μ, ρ randomly
For each epoch:
    For each minibatch B:
        1. Sample ε ~ N(0, I)
        2. Compute θ = μ + softplus(ρ) ⊙ ε
        3. Compute loss:
           L = -log p(B|θ) + (1/M) * KL(q_φ || p)
           where M = number of minibatches
        4. Compute gradients ∇_μ L, ∇_ρ L
        5. Update μ ← μ - α ∇_μ L
        6. Update ρ ← ρ - α ∇_ρ L
```

### Minibatch ELBO

For minibatch training, scale the ELBO appropriately:

$$
\mathcal{L}(\phi) \approx \frac{N}{|B|} \sum_{i \in B} \log p(y_i \mid x_i, \theta) - \text{KL}(q_\phi \| p)
$$

**KL weighting**: The KL term is computed once per minibatch, scaled by $1/M$ where $M$ is the number of minibatches.

### Weight Uncertainty

The trained variational distribution gives:

$$
W_{ij} \sim \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)
$$

**Interpretation**:
- $\mu_{ij}$: Most likely weight value
- $\sigma_{ij}$: Uncertainty about weight value
- Large $\sigma$ → high uncertainty → less confident predictions

---

## Local Reparameterization Trick

### Motivation

Standard reparameterization samples weights:

$$
W = \mu_W + \sigma_W \odot \epsilon_W
$$

Then computes activations:

$$
a = Wx
$$

**Problem**: High variance in gradient estimates when network is wide.

### Local Reparameterization

**Key insight**: For linear layers, directly sample activations:

$$
a = Wx = (\mu_W + \sigma_W \odot \epsilon_W)x
$$

$$
a \sim \mathcal{N}(\mu_W x, (\sigma_W^2 \odot x^2) \mathbf{1})
$$

**Reparameterize at activation level**:

$$
a = \mu_W x + \sqrt{\sigma_W^2 \odot x^2} \odot \epsilon_a
$$

where $\epsilon_a \sim \mathcal{N}(0, I)$ has dimension equal to output size.

### Benefits

| Aspect | Standard | Local |
|--------|----------|-------|
| Noise dimension | $d$ (weights) | $n$ (activations) |
| Gradient variance | Higher | Lower |
| Computational cost | Same | Same |
| Correlation | Weights correlated | Activations independent |

### Implementation

For a layer with weight matrix $W \in \mathbb{R}^{m \times n}$:

```python
# Standard reparameterization
W = mu_W + sigma_W * eps_W  # eps_W: (m, n)
a = W @ x                    # a: (n,)

# Local reparameterization
a_mu = mu_W @ x                           # (n,)
a_var = (sigma_W**2) @ (x**2)            # (n,)
a = a_mu + sqrt(a_var) * eps_a           # eps_a: (n,)
```

---

## KL Divergence Strategies

### Exact KL (Closed Form)

For Gaussian-to-Gaussian KL:

$$
\text{KL}(q \| p) = \frac{1}{2}\left[\text{tr}(\Sigma_p^{-1}\Sigma_q) + (\mu_p - \mu_q)^\top \Sigma_p^{-1}(\mu_p - \mu_q) - d + \log\frac{|\Sigma_p|}{|\Sigma_q|}\right]
$$

For diagonal Gaussians with zero-mean prior:

$$
\text{KL} = \frac{1}{2}\sum_j \left[\frac{\mu_j^2 + \sigma_j^2}{\sigma_p^2} - 1 - \log\frac{\sigma_j^2}{\sigma_p^2}\right]
$$

### Monte Carlo KL Estimation

For non-conjugate priors, estimate KL via sampling:

$$
\text{KL}(q \| p) = \mathbb{E}_{q}[\log q(\theta) - \log p(\theta)] \approx \frac{1}{S}\sum_{s=1}^S [\log q(\theta^{(s)}) - \log p(\theta^{(s)})]
$$

### KL Annealing

**Problem**: KL term can dominate early in training, collapsing $q$ to prior.

**Solution**: Gradually increase KL weight:

$$
\mathcal{L}_t(\phi) = \mathbb{E}_{q_\phi}[\log p(\mathcal{D} \mid \theta)] - \beta_t \cdot \text{KL}(q_\phi \| p)
$$

**Annealing schedules**:

**Linear**:
$$
\beta_t = \min(1, t / T_{\text{warmup}})
$$

**Sigmoid**:
$$
\beta_t = \frac{1}{1 + \exp(-(t - T_{\text{mid}})/\tau)}
$$

**Cyclical**:
$$
\beta_t = \min(1, \text{mod}(t, T_{\text{cycle}}) / T_{\text{rise}})
$$

---

## Beyond Mean-Field

### Full Covariance Gaussian

$$
q(\theta) = \mathcal{N}(\mu, \Sigma)
$$

**Parameters**: $d + d(d+1)/2$ (mean + lower triangular Cholesky)

**Problem**: $O(d^2)$ parameters and $O(d^3)$ computation — intractable for large networks.

### Low-Rank Approximations

**Low-rank plus diagonal**:

$$
\Sigma = D + VV^\top
$$

where $D$ is diagonal and $V \in \mathbb{R}^{d \times r}$ with $r \ll d$.

**Parameters**: $d + dr$

**Sampling**: $\theta = \mu + D^{1/2}\epsilon_1 + V\epsilon_2$ where $\epsilon_1 \in \mathbb{R}^d$, $\epsilon_2 \in \mathbb{R}^r$.

### Matrix-Variate Gaussian

For weight matrix $W \in \mathbb{R}^{m \times n}$:

$$
q(W) = \mathcal{MN}(M, U, V)
$$

where $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$.

**Parameters**: $mn + m^2 + n^2$ (much less than $mn(mn+1)/2$)

### Normalizing Flows

Transform a simple distribution through invertible functions:

$$
\theta = f_K \circ f_{K-1} \circ \cdots \circ f_1(z), \quad z \sim \mathcal{N}(0, I)
$$

**Density**:

$$
q(\theta) = q_0(f^{-1}(\theta)) \left|\det \frac{\partial f^{-1}}{\partial \theta}\right|
$$

**Popular flows**:
- **Planar flows**: $f(z) = z + u \cdot \tanh(w^\top z + b)$
- **Radial flows**: $f(z) = z + \beta h(\alpha, r)(z - z_0)$
- **RealNVP**: Coupling layers with tractable Jacobian
- **IAF**: Inverse autoregressive flow

---

## Practical Considerations

### Prior Selection

The prior $p(\theta)$ affects both regularization and uncertainty:

**Standard Gaussian**:
$$
p(\theta) = \mathcal{N}(0, \sigma_p^2 I)
$$

**Scale mixture** (for robustness):
$$
p(\theta) = \pi \mathcal{N}(0, \sigma_1^2) + (1-\pi) \mathcal{N}(0, \sigma_2^2)
$$

**Empirical guidelines**:
- Start with $\sigma_p = 1$
- Tune based on validation performance
- Consider hierarchical priors for automatic tuning

### Initialization

**Mean initialization**: 
- Standard initialization (Xavier, He)
- Or pre-trained weights

**Variance initialization**:
- Small initial variance: $\sigma_{\text{init}} \approx 0.01$-$0.1$
- Ensures early training resembles deterministic network

### Number of Monte Carlo Samples

**Training**: Often $S = 1$ sample suffices (unbiased gradient)

**Evaluation**: Use more samples ($S = 10$-$100$) for stable predictions

**Trade-off**: More samples → better estimates, higher cost

### Computational Overhead

| Component | Cost vs. Standard NN |
|-----------|---------------------|
| Parameters | 2× (mean + variance) |
| Forward pass | ~1× (with local reparam) |
| Backward pass | ~1.5× |
| Memory | 2× |
| Inference | S× (for S samples) |

---

## Variational Inference Variants

### Multiplicative Normalizing Flows (MNF)

Auxiliary variables for more expressive posteriors:

$$
q(\theta) = \int q(\theta \mid z) q(z) dz
$$

where $q(\theta \mid z)$ is Gaussian and $q(z)$ is a normalizing flow.

### Noisy Natural Gradient (NNG)

Use natural gradient with noise for better optimization:

$$
\theta_{t+1} = \theta_t - \alpha F^{-1} (\nabla \mathcal{L} + \epsilon)
$$

where $F$ is the Fisher information matrix.

### Variational Online Gauss-Newton (VOGN)

Approximate natural gradient variational inference:

$$
\mu_{t+1} = \mu_t - \alpha \Sigma_t \nabla_\mu \mathcal{L}
$$

$$
\Sigma_{t+1}^{-1} = (1-\alpha)\Sigma_t^{-1} + \alpha \hat{F}
$$

### Functional Variational Inference

Place variational distribution in function space:

$$
q(f) \approx p(f \mid \mathcal{D})
$$

**Advantages**:
- More interpretable priors
- Better uncertainty in function space
- Avoids weight-space pathologies

---

## Python Implementation

```python
"""
Variational Bayesian Neural Networks

This module provides complete implementations of variational inference
for neural networks, including Bayes by Backprop, local reparameterization,
KL annealing, and various posterior approximations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import softmax
from typing import Tuple, List, Optional, Dict, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


# =============================================================================
# Variational Layers
# =============================================================================

class VariationalLayer(ABC):
    """Abstract base class for variational layers."""
    
    @abstractmethod
    def forward(self, x: np.ndarray, sample: bool = True) -> np.ndarray:
        """Forward pass with optional sampling."""
        pass
    
    @abstractmethod
    def kl_divergence(self) -> float:
        """Compute KL divergence to prior."""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, np.ndarray]:
        """Get variational parameters."""
        pass
    
    @abstractmethod
    def set_params(self, params: Dict[str, np.ndarray]):
        """Set variational parameters."""
        pass


class VariationalLinear(VariationalLayer):
    """
    Variational linear layer with Gaussian weights.
    
    W_ij ~ N(mu_W_ij, sigma_W_ij^2)
    b_j ~ N(mu_b_j, sigma_b_j^2)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0,
        init_sigma: float = 0.1,
        use_local_reparam: bool = True
    ):
        """
        Parameters
        ----------
        in_features : int
            Input dimension
        out_features : int
            Output dimension
        prior_sigma : float
            Prior standard deviation
        init_sigma : float
            Initial posterior standard deviation
        use_local_reparam : bool
            Use local reparameterization trick
        """
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        self.use_local_reparam = use_local_reparam
        
        # Initialize variational parameters
        # Weight mean: Xavier initialization
        self.mu_W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        # Weight log-variance (use rho parameterization: sigma = softplus(rho))
        self.rho_W = np.full((in_features, out_features), np.log(np.exp(init_sigma) - 1))
        
        # Bias mean and log-variance
        self.mu_b = np.zeros(out_features)
        self.rho_b = np.full(out_features, np.log(np.exp(init_sigma) - 1))
        
        # Store last sampled weights for gradient computation
        self.last_eps_W = None
        self.last_eps_b = None
    
    @property
    def sigma_W(self) -> np.ndarray:
        """Compute sigma from rho using softplus."""
        return np.log(1 + np.exp(self.rho_W))
    
    @property
    def sigma_b(self) -> np.ndarray:
        """Compute sigma from rho using softplus."""
        return np.log(1 + np.exp(self.rho_b))
    
    def forward(self, x: np.ndarray, sample: bool = True) -> np.ndarray:
        """
        Forward pass.
        
        Parameters
        ----------
        x : ndarray of shape (batch_size, in_features)
            Input
        sample : bool
            If True, sample weights; if False, use mean
        
        Returns
        -------
        ndarray of shape (batch_size, out_features)
            Output
        """
        if not sample:
            # Deterministic forward pass (use mean)
            return x @ self.mu_W + self.mu_b
        
        if self.use_local_reparam:
            # Local reparameterization: sample activations directly
            # a ~ N(x @ mu_W + mu_b, x^2 @ sigma_W^2 + sigma_b^2)
            
            mu_a = x @ self.mu_W + self.mu_b
            var_a = (x ** 2) @ (self.sigma_W ** 2) + self.sigma_b ** 2
            
            eps_a = np.random.randn(*mu_a.shape)
            return mu_a + np.sqrt(var_a + 1e-8) * eps_a
        
        else:
            # Standard reparameterization: sample weights
            self.last_eps_W = np.random.randn(*self.mu_W.shape)
            self.last_eps_b = np.random.randn(*self.mu_b.shape)
            
            W = self.mu_W + self.sigma_W * self.last_eps_W
            b = self.mu_b + self.sigma_b * self.last_eps_b
            
            return x @ W + b
    
    def kl_divergence(self) -> float:
        """
        Compute KL divergence from q(W) to prior p(W).
        
        KL(N(mu, sigma^2) || N(0, sigma_p^2)) = 
            0.5 * (mu^2/sigma_p^2 + sigma^2/sigma_p^2 - 1 - log(sigma^2/sigma_p^2))
        """
        prior_var = self.prior_sigma ** 2
        
        # KL for weights
        kl_W = 0.5 * np.sum(
            self.mu_W ** 2 / prior_var +
            self.sigma_W ** 2 / prior_var -
            1 -
            np.log(self.sigma_W ** 2 / prior_var + 1e-10)
        )
        
        # KL for biases
        kl_b = 0.5 * np.sum(
            self.mu_b ** 2 / prior_var +
            self.sigma_b ** 2 / prior_var -
            1 -
            np.log(self.sigma_b ** 2 / prior_var + 1e-10)
        )
        
        return kl_W + kl_b
    
    def get_params(self) -> Dict[str, np.ndarray]:
        """Get variational parameters."""
        return {
            'mu_W': self.mu_W.copy(),
            'rho_W': self.rho_W.copy(),
            'mu_b': self.mu_b.copy(),
            'rho_b': self.rho_b.copy()
        }
    
    def set_params(self, params: Dict[str, np.ndarray]):
        """Set variational parameters."""
        self.mu_W = params['mu_W'].copy()
        self.rho_W = params['rho_W'].copy()
        self.mu_b = params['mu_b'].copy()
        self.rho_b = params['rho_b'].copy()
    
    def n_params(self) -> int:
        """Number of variational parameters."""
        return 2 * (self.in_features * self.out_features + self.out_features)


# =============================================================================
# Variational Neural Network
# =============================================================================

class VariationalMLP:
    """
    Variational Multi-Layer Perceptron.
    
    Implements Bayes by Backprop with mean-field Gaussian posterior.
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        prior_sigma: float = 1.0,
        init_sigma: float = 0.1,
        activation: str = 'relu',
        use_local_reparam: bool = True
    ):
        """
        Parameters
        ----------
        layer_sizes : list
            [input_dim, hidden1, ..., output_dim]
        prior_sigma : float
            Prior standard deviation
        init_sigma : float
            Initial posterior standard deviation
        activation : str
            'relu' or 'tanh'
        use_local_reparam : bool
            Use local reparameterization
        """
        self.layer_sizes = layer_sizes
        self.prior_sigma = prior_sigma
        self.n_layers = len(layer_sizes) - 1
        
        # Activation function
        if activation == 'relu':
            self.act_fn = lambda x: np.maximum(x, 0)
        elif activation == 'tanh':
            self.act_fn = np.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Create variational layers
        self.layers = []
        for i in range(self.n_layers):
            layer = VariationalLinear(
                layer_sizes[i],
                layer_sizes[i + 1],
                prior_sigma=prior_sigma,
                init_sigma=init_sigma,
                use_local_reparam=use_local_reparam
            )
            self.layers.append(layer)
    
    def forward(self, x: np.ndarray, sample: bool = True) -> np.ndarray:
        """Forward pass through all layers."""
        h = x
        for i, layer in enumerate(self.layers):
            h = layer.forward(h, sample=sample)
            # Activation except last layer
            if i < self.n_layers - 1:
                h = self.act_fn(h)
        return h
    
    def kl_divergence(self) -> float:
        """Total KL divergence across all layers."""
        return sum(layer.kl_divergence() for layer in self.layers)
    
    def predict(
        self,
        x: np.ndarray,
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
        predictions = []
        for _ in range(n_samples):
            pred = self.forward(x, sample=True)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        return mean, std
    
    def n_variational_params(self) -> int:
        """Total number of variational parameters."""
        return sum(layer.n_params() for layer in self.layers)


# =============================================================================
# Training
# =============================================================================

class BayesByBackprop:
    """
    Bayes by Backprop training algorithm.
    """
    
    def __init__(
        self,
        model: VariationalMLP,
        likelihood_sigma: float = 1.0,
        kl_weight: float = 1.0,
        lr: float = 0.001,
        lr_decay: float = 0.0
    ):
        """
        Parameters
        ----------
        model : VariationalMLP
            Variational neural network
        likelihood_sigma : float
            Observation noise standard deviation
        kl_weight : float
            Weight on KL term (for annealing)
        lr : float
            Learning rate
        lr_decay : float
            Learning rate decay per epoch
        """
        self.model = model
        self.likelihood_sigma = likelihood_sigma
        self.kl_weight = kl_weight
        self.lr = lr
        self.lr_decay = lr_decay
    
    def compute_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_total: int,
        n_samples: int = 1
    ) -> Tuple[float, float, float]:
        """
        Compute ELBO loss.
        
        Returns
        -------
        loss : float
            Total loss (-ELBO)
        nll : float
            Negative log-likelihood term
        kl : float
            KL divergence term
        """
        batch_size = len(X)
        
        # Monte Carlo estimate of expected NLL
        nll = 0.0
        for _ in range(n_samples):
            pred = self.model.forward(X, sample=True)
            # Gaussian NLL
            nll += 0.5 * np.sum((y - pred) ** 2) / (self.likelihood_sigma ** 2)
        nll /= n_samples
        
        # Scale for full dataset
        nll *= (n_total / batch_size)
        
        # KL divergence
        kl = self.model.kl_divergence()
        
        # Total loss
        loss = nll + self.kl_weight * kl
        
        return loss, nll, kl
    
    def compute_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_total: int,
        eps: float = 1e-5
    ) -> List[Dict[str, np.ndarray]]:
        """
        Compute gradients numerically (for simplicity).
        
        In practice, use automatic differentiation.
        """
        gradients = []
        
        for layer in self.model.layers:
            params = layer.get_params()
            grads = {}
            
            for param_name, param_value in params.items():
                grad = np.zeros_like(param_value)
                
                for idx in np.ndindex(param_value.shape):
                    # Compute finite difference
                    original = param_value[idx]
                    
                    param_value[idx] = original + eps
                    layer.set_params({**params, param_name: param_value})
                    loss_plus, _, _ = self.compute_loss(X, y, n_total)
                    
                    param_value[idx] = original - eps
                    layer.set_params({**params, param_name: param_value})
                    loss_minus, _, _ = self.compute_loss(X, y, n_total)
                    
                    param_value[idx] = original
                    layer.set_params(params)
                    
                    grad[idx] = (loss_plus - loss_minus) / (2 * eps)
                
                grads[param_name] = grad
            
            gradients.append(grads)
        
        return gradients
    
    def train_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_total: int
    ) -> Tuple[float, float, float]:
        """
        Single training step.
        
        Returns
        -------
        loss, nll, kl : float
            Loss components
        """
        # Compute gradients
        gradients = self.compute_gradients(X, y, n_total)
        
        # Update parameters
        for layer, grads in zip(self.model.layers, gradients):
            params = layer.get_params()
            for param_name in params:
                params[param_name] -= self.lr * grads[param_name]
            layer.set_params(params)
        
        # Compute final loss
        return self.compute_loss(X, y, n_total)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 100,
        batch_size: Optional[int] = None,
        kl_annealing: bool = False,
        annealing_epochs: int = 50,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the variational BNN.
        
        Returns
        -------
        history : dict
            Training history with 'loss', 'nll', 'kl'
        """
        N = len(X)
        if batch_size is None:
            batch_size = min(N, 32)
        
        history = {'loss': [], 'nll': [], 'kl': []}
        
        for epoch in range(n_epochs):
            # KL annealing
            if kl_annealing:
                self.kl_weight = min(1.0, epoch / annealing_epochs)
            
            # Learning rate decay
            current_lr = self.lr * (1 - self.lr_decay) ** epoch
            
            # Shuffle data
            perm = np.random.permutation(N)
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            
            epoch_loss = 0.0
            epoch_nll = 0.0
            epoch_kl = 0.0
            n_batches = 0
            
            for i in range(0, N, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                loss, nll, kl = self.train_step(X_batch, y_batch, N)
                
                epoch_loss += loss
                epoch_nll += nll
                epoch_kl += kl
                n_batches += 1
            
            epoch_loss /= n_batches
            epoch_nll /= n_batches
            epoch_kl /= n_batches
            
            history['loss'].append(epoch_loss)
            history['nll'].append(epoch_nll)
            history['kl'].append(epoch_kl)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, "
                      f"NLL={epoch_nll:.4f}, KL={epoch_kl:.4f}")
        
        return history


# =============================================================================
# KL Annealing Schedules
# =============================================================================

def linear_annealing(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    """Linear KL annealing schedule."""
    return min(1.0, epoch / warmup_epochs)


def sigmoid_annealing(epoch: int, total_epochs: int, midpoint: int, steepness: float = 0.1) -> float:
    """Sigmoid KL annealing schedule."""
    return 1.0 / (1.0 + np.exp(-steepness * (epoch - midpoint)))


def cyclical_annealing(epoch: int, cycle_length: int, ratio: float = 0.5) -> float:
    """Cyclical KL annealing schedule."""
    cycle_position = epoch % cycle_length
    rise_length = int(cycle_length * ratio)
    return min(1.0, cycle_position / rise_length)


# =============================================================================
# Scale Mixture Prior
# =============================================================================

class ScaleMixturePrior:
    """
    Scale mixture of Gaussians prior.
    
    p(w) = pi * N(0, sigma1^2) + (1-pi) * N(0, sigma2^2)
    """
    
    def __init__(
        self,
        pi: float = 0.5,
        sigma1: float = 1.0,
        sigma2: float = 0.1
    ):
        """
        Parameters
        ----------
        pi : float
            Mixture weight
        sigma1 : float
            Standard deviation of first component
        sigma2 : float
            Standard deviation of second component
        """
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
    
    def log_prob(self, w: np.ndarray) -> float:
        """Compute log probability."""
        log_p1 = stats.norm.logpdf(w, 0, self.sigma1)
        log_p2 = stats.norm.logpdf(w, 0, self.sigma2)
        
        # Log-sum-exp for numerical stability
        log_mix = np.logaddexp(
            np.log(self.pi) + log_p1,
            np.log(1 - self.pi) + log_p2
        )
        
        return np.sum(log_mix)


class VariationalLinearMixturePrior(VariationalLayer):
    """
    Variational linear layer with scale mixture prior.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior: ScaleMixturePrior,
        init_sigma: float = 0.1
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.prior = prior
        
        # Initialize variational parameters
        self.mu_W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.rho_W = np.full((in_features, out_features), np.log(np.exp(init_sigma) - 1))
        self.mu_b = np.zeros(out_features)
        self.rho_b = np.full(out_features, np.log(np.exp(init_sigma) - 1))
    
    @property
    def sigma_W(self) -> np.ndarray:
        return np.log(1 + np.exp(self.rho_W))
    
    @property
    def sigma_b(self) -> np.ndarray:
        return np.log(1 + np.exp(self.rho_b))
    
    def forward(self, x: np.ndarray, sample: bool = True) -> np.ndarray:
        if not sample:
            return x @ self.mu_W + self.mu_b
        
        eps_W = np.random.randn(*self.mu_W.shape)
        eps_b = np.random.randn(*self.mu_b.shape)
        
        W = self.mu_W + self.sigma_W * eps_W
        b = self.mu_b + self.sigma_b * eps_b
        
        return x @ W + b
    
    def kl_divergence(self, n_samples: int = 1) -> float:
        """
        Monte Carlo estimate of KL divergence.
        
        KL(q||p) = E_q[log q - log p]
        """
        kl = 0.0
        
        for _ in range(n_samples):
            # Sample weights
            eps_W = np.random.randn(*self.mu_W.shape)
            eps_b = np.random.randn(*self.mu_b.shape)
            
            W = self.mu_W + self.sigma_W * eps_W
            b = self.mu_b + self.sigma_b * eps_b
            
            # Log q (variational posterior)
            log_q_W = np.sum(stats.norm.logpdf(W, self.mu_W, self.sigma_W))
            log_q_b = np.sum(stats.norm.logpdf(b, self.mu_b, self.sigma_b))
            
            # Log p (prior)
            log_p_W = self.prior.log_prob(W)
            log_p_b = self.prior.log_prob(b)
            
            kl += (log_q_W + log_q_b) - (log_p_W + log_p_b)
        
        return kl / n_samples
    
    def get_params(self) -> Dict[str, np.ndarray]:
        return {
            'mu_W': self.mu_W.copy(),
            'rho_W': self.rho_W.copy(),
            'mu_b': self.mu_b.copy(),
            'rho_b': self.rho_b.copy()
        }
    
    def set_params(self, params: Dict[str, np.ndarray]):
        self.mu_W = params['mu_W'].copy()
        self.rho_W = params['rho_W'].copy()
        self.mu_b = params['mu_b'].copy()
        self.rho_b = params['rho_b'].copy()


# =============================================================================
# Visualization
# =============================================================================

def plot_weight_distributions(
    model: VariationalMLP,
    layer_idx: int = 0,
    n_weights: int = 5
):
    """Visualize learned weight distributions."""
    
    layer = model.layers[layer_idx]
    
    fig, axes = plt.subplots(1, n_weights, figsize=(3*n_weights, 3))
    
    # Flatten weights for visualization
    mu_flat = layer.mu_W.flatten()
    sigma_flat = layer.sigma_W.flatten()
    
    # Select random weights
    indices = np.random.choice(len(mu_flat), n_weights, replace=False)
    
    x = np.linspace(-3, 3, 200)
    
    for i, (ax, idx) in enumerate(zip(axes, indices)):
        mu = mu_flat[idx]
        sigma = sigma_flat[idx]
        
        # Plot posterior
        posterior = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, posterior, 'b-', linewidth=2, label='Posterior')
        
        # Plot prior
        prior = stats.norm.pdf(x, 0, model.prior_sigma)
        ax.plot(x, prior, 'k--', linewidth=1, label='Prior')
        
        ax.axvline(mu, color='red', linestyle=':', label=f'μ={mu:.2f}')
        ax.set_title(f'Weight {idx}\nσ={sigma:.3f}')
        ax.set_xlabel('Weight value')
        
        if i == 0:
            ax.legend()
    
    plt.suptitle(f'Layer {layer_idx} Weight Distributions')
    plt.tight_layout()
    plt.show()


def plot_training_history(history: Dict[str, List[float]]):
    """Plot training curves."""
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Total loss
    axes[0].plot(history['loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss (-ELBO)')
    
    # NLL
    axes[1].plot(history['nll'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('NLL')
    axes[1].set_title('Negative Log-Likelihood')
    
    # KL
    axes[2].plot(history['kl'], 'r-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL')
    axes[2].set_title('KL Divergence')
    
    plt.tight_layout()
    plt.show()


def plot_predictions_with_uncertainty(
    model: VariationalMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    n_samples: int = 100
):
    """Plot predictions with uncertainty bands."""
    
    mean, std = model.predict(X_test, n_samples=n_samples)
    
    plt.figure(figsize=(10, 6))
    
    # Uncertainty band
    X_flat = X_test.flatten()
    mean_flat = mean.flatten()
    std_flat = std.flatten()
    
    plt.fill_between(
        X_flat,
        mean_flat - 2*std_flat,
        mean_flat + 2*std_flat,
        alpha=0.3, color='blue', label='±2σ'
    )
    
    # Mean prediction
    plt.plot(X_flat, mean_flat, 'b-', linewidth=2, label='Mean')
    
    # True function
    if y_true is not None:
        plt.plot(X_flat, y_true.flatten(), 'k--', linewidth=1, label='True')
    
    # Training data
    plt.scatter(X_train.flatten(), y_train.flatten(), 
                c='red', s=30, zorder=5, label='Data')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Variational BNN Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# =============================================================================
# Demo Functions
# =============================================================================

def demo_variational_bnn():
    """Demonstrate variational BNN training."""
    
    print("=" * 70)
    print("VARIATIONAL BAYESIAN NEURAL NETWORK")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate data
    N = 50
    X_train = np.random.uniform(-4, 4, N).reshape(-1, 1)
    y_train = np.sin(X_train) + np.random.normal(0, 0.2, (N, 1))
    
    X_test = np.linspace(-6, 6, 200).reshape(-1, 1)
    y_true = np.sin(X_test)
    
    print(f"\nTraining data: {N} points")
    print(f"Test data: {len(X_test)} points")
    
    # Create model
    model = VariationalMLP(
        layer_sizes=[1, 20, 20, 1],
        prior_sigma=1.0,
        init_sigma=0.1,
        activation='tanh',
        use_local_reparam=True
    )
    
    print(f"Model: {model.layer_sizes}")
    print(f"Variational parameters: {model.n_variational_params()}")
    
    # Train
    trainer = BayesByBackprop(
        model,
        likelihood_sigma=0.2,
        kl_weight=1.0,
        lr=0.01
    )
    
    print("\nTraining (this may take a while)...")
    history = trainer.train(
        X_train, y_train,
        n_epochs=50,
        batch_size=N,
        kl_annealing=True,
        annealing_epochs=25,
        verbose=True
    )
    
    # Evaluate
    mean, std = model.predict(X_test, n_samples=100)
    
    print(f"\nPrediction statistics:")
    print(f"  Mean std (epistemic): {np.mean(std):.4f}")
    print(f"  Max std: {np.max(std):.4f}")
    
    return model, history


def demo_kl_annealing():
    """Demonstrate different KL annealing schedules."""
    
    print("\n" + "=" * 70)
    print("KL ANNEALING SCHEDULES")
    print("=" * 70)
    
    n_epochs = 100
    epochs = np.arange(n_epochs)
    
    schedules = {
        'Linear (warmup=30)': [linear_annealing(e, n_epochs, 30) for e in epochs],
        'Sigmoid (mid=30)': [sigmoid_annealing(e, n_epochs, 30, 0.2) for e in epochs],
        'Cyclical (cycle=40)': [cyclical_annealing(e, 40, 0.5) for e in epochs],
    }
    
    plt.figure(figsize=(10, 5))
    
    for name, values in schedules.items():
        plt.plot(epochs, values, linewidth=2, label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('KL Weight (β)')
    plt.title('KL Annealing Schedules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nKL annealing prevents posterior collapse in early training.")


def demo_local_reparameterization():
    """Compare standard vs local reparameterization."""
    
    print("\n" + "=" * 70)
    print("LOCAL REPARAMETERIZATION TRICK")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Create two layers
    layer_standard = VariationalLinear(10, 5, use_local_reparam=False)
    layer_local = VariationalLinear(10, 5, use_local_reparam=True)
    
    # Copy parameters
    layer_local.mu_W = layer_standard.mu_W.copy()
    layer_local.rho_W = layer_standard.rho_W.copy()
    layer_local.mu_b = layer_standard.mu_b.copy()
    layer_local.rho_b = layer_standard.rho_b.copy()
    
    # Test input
    x = np.random.randn(1, 10)
    
    # Multiple forward passes
    n_samples = 1000
    
    outputs_standard = np.array([layer_standard.forward(x).flatten() for _ in range(n_samples)])
    outputs_local = np.array([layer_local.forward(x).flatten() for _ in range(n_samples)])
    
    print("\nOutput statistics (should be similar):")
    print(f"  Standard - Mean: {np.mean(outputs_standard, axis=0)[:3]}")
    print(f"  Local    - Mean: {np.mean(outputs_local, axis=0)[:3]}")
    print(f"  Standard - Std:  {np.std(outputs_standard, axis=0)[:3]}")
    print(f"  Local    - Std:  {np.std(outputs_local, axis=0)[:3]}")
    
    print("\n*** Local reparameterization gives same distribution")
    print("*** but with lower gradient variance")


def demo_uncertainty_quality():
    """Demonstrate uncertainty behavior of variational BNN."""
    
    print("\n" + "=" * 70)
    print("UNCERTAINTY QUALITY")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate data with gap
    X_train = np.concatenate([
        np.random.uniform(-4, -1, 25),
        np.random.uniform(1, 4, 25)
    ]).reshape(-1, 1)
    y_train = np.sin(X_train) + np.random.normal(0, 0.15, (50, 1))
    
    X_test = np.linspace(-6, 6, 200).reshape(-1, 1)
    
    print("Data has gap in [-1, 1] region")
    
    # Create and train model
    model = VariationalMLP(
        layer_sizes=[1, 30, 1],
        prior_sigma=1.0,
        init_sigma=0.05,
        activation='tanh'
    )
    
    trainer = BayesByBackprop(model, likelihood_sigma=0.15, lr=0.02)
    trainer.train(X_train, y_train, n_epochs=100, verbose=False)
    
    # Evaluate
    mean, std = model.predict(X_test, n_samples=100)
    
    # Analyze uncertainty in different regions
    in_gap = (X_test.flatten() > -1) & (X_test.flatten() < 1)
    near_data = ~in_gap & (np.abs(X_test.flatten()) < 4)
    extrapolation = np.abs(X_test.flatten()) > 4
    
    print(f"\nMean uncertainty (std):")
    print(f"  In gap region:      {np.mean(std[in_gap]):.4f}")
    print(f"  Near training data: {np.mean(std[near_data]):.4f}")
    print(f"  Extrapolation:      {np.mean(std[extrapolation]):.4f}")
    
    print("\n*** Uncertainty should be higher in gap and extrapolation regions")


if __name__ == "__main__":
    model, history = demo_variational_bnn()
    demo_kl_annealing()
    demo_local_reparameterization()
    demo_uncertainty_quality()
```

---

## Summary

### Core Concepts

**Variational Inference** approximates the intractable posterior:

$$
p(\theta \mid \mathcal{D}) \approx q_\phi(\theta)
$$

**ELBO** (Evidence Lower Bound):

$$
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}[\log p(\mathcal{D} \mid \theta)] - \text{KL}(q_\phi \| p)
$$

**Reparameterization Trick**:

$$
\theta = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

### Mean-Field Gaussian

| Component | Formula |
|-----------|---------|
| **Posterior** | $q(\theta) = \prod_j \mathcal{N}(\theta_j \mid \mu_j, \sigma_j^2)$ |
| **Parameters** | $\phi = \{\mu, \rho\}$ where $\sigma = \text{softplus}(\rho)$ |
| **KL to prior** | $\frac{1}{2}\sum_j \left[\frac{\mu_j^2 + \sigma_j^2}{\sigma_p^2} - 1 - \log\frac{\sigma_j^2}{\sigma_p^2}\right]$ |

### Bayes by Backprop Algorithm

1. Sample $\epsilon \sim \mathcal{N}(0, I)$
2. Compute $\theta = \mu + \text{softplus}(\rho) \odot \epsilon$
3. Compute loss: $\mathcal{L} = \text{NLL}(\theta) + \beta \cdot \text{KL}(q \| p)$
4. Backpropagate and update $\mu, \rho$

### Practical Considerations

| Aspect | Recommendation |
|--------|----------------|
| **Prior** | $\sigma_p = 1.0$ (tune on validation) |
| **Init variance** | $\sigma_{\text{init}} = 0.01$-$0.1$ |
| **MC samples (train)** | 1 (unbiased gradient) |
| **MC samples (test)** | 30-100 |
| **KL annealing** | Linear warmup over 20-50 epochs |
| **Learning rate** | 0.001-0.01 |

### Beyond Mean-Field

| Method | Expressiveness | Cost |
|--------|---------------|------|
| Mean-Field | Low | $O(d)$ |
| Low-Rank | Medium | $O(dr)$ |
| Matrix-Variate | Medium | $O(m^2 + n^2)$ |
| Normalizing Flows | High | $O(dK)$ |

### Advantages and Limitations

| Advantages | Limitations |
|------------|-------------|
| Scalable to large networks | Approximate posterior |
| Principled uncertainty | May underestimate uncertainty |
| Works with backprop | KL requires careful handling |
| Flexible prior specification | Mean-field ignores correlations |

### Connections to Other Chapters

| Topic | Chapter | Connection |
|-------|---------|------------|
| Prior specification | Ch13: Prior on Weights | Prior in KL term |
| Uncertainty | Ch13: Uncertainty | Posterior enables decomposition |
| MC Dropout | Ch13: MC Dropout | Implicit VI alternative |
| Posterior inference | Ch13: Posterior Inference | VI as inference method |
| Model comparison | Ch13: Model Evidence | ELBO bounds evidence |

### Key References

- Blundell, C., et al. (2015). Weight uncertainty in neural networks. *ICML*.
- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *ICLR*.
- Kingma, D. P., et al. (2015). Variational dropout and the local reparameterization trick. *NeurIPS*.
- Louizos, C., & Welling, M. (2017). Multiplicative normalizing flows for variational Bayesian neural networks. *ICML*.
- Zhang, G., et al. (2018). Noisy natural gradient as variational inference. *ICML*.
- Osawa, K., et al. (2019). Practical deep learning with Bayesian principles. *NeurIPS*.
