# Prior Distributions on Neural Network Weights

The **prior distribution** over neural network weights encodes our beliefs about plausible parameter values before observing data. In Bayesian neural networks, the choice of prior profoundly affects both the induced function space and the resulting uncertainty estimates. This chapter explores principled approaches to specifying priors that lead to well-behaved posterior inference.

---

## Motivation: Why Priors Matter

### The Role of Priors in BNNs

In standard neural networks, weights are point estimates found by optimization:

$$
\hat{\theta} = \arg\max_\theta \log p(\mathcal{D} \mid \theta)
$$

In Bayesian neural networks, we maintain a distribution:

$$
p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) \, p(\theta)
$$

The prior $p(\theta)$ determines:
1. **Regularization strength**: Constrains weight magnitudes
2. **Function space properties**: Smoothness, periodicity, length scales
3. **Epistemic uncertainty**: How uncertainty behaves away from data
4. **Posterior geometry**: Affects inference tractability

### Challenges Specific to Neural Networks

**High dimensionality**: Modern networks have millions of parameters
- Simple independent priors may not capture complex dependencies
- Correlations between weights in same layer often important

**Non-identifiability**: Multiple weight configurations yield identical functions
- Permutation symmetry: Swapping hidden units
- Scaling symmetry: Rescaling weights between layers

**Overparameterization**: More parameters than data points
- Prior becomes dominant in posterior
- Need priors that induce sensible function behavior

### Weight Space vs Function Space

**Key insight**: We care about functions, not weights.

$$
\text{Prior on weights } p(\theta) \implies \text{Prior on functions } p(f)
$$

A simple prior on weights can induce complex behavior in function space:

$$
f(x) = W_L \sigma(W_{L-1} \sigma(\cdots \sigma(W_1 x)))
$$

The challenge is to specify $p(\theta)$ such that $p(f)$ has desirable properties.

---

## Standard Gaussian Priors

### Independent Gaussian Prior

The most common choice is independent Gaussians:

$$
\boxed{p(\theta) = \prod_{l=1}^L \prod_{i,j} \mathcal{N}(w_{ij}^{(l)} \mid 0, \sigma_l^2)}
$$

**Equivalently** in matrix form:

$$
p(W^{(l)}) = \mathcal{N}(\text{vec}(W^{(l)}) \mid 0, \sigma_l^2 I)
$$

### Connection to L2 Regularization

The MAP estimate with Gaussian prior equals L2-regularized MLE:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[ \log p(\mathcal{D} \mid \theta) - \frac{1}{2\sigma^2} \|\theta\|^2 \right]
$$

**Regularization strength**: $\lambda = 1/(2\sigma^2)$

| Prior variance $\sigma^2$ | L2 penalty $\lambda$ | Effect |
|---------------------------|---------------------|--------|
| Large | Small | Weak regularization |
| Small | Large | Strong regularization |

### Layer-wise Variance Scaling

Different layers may need different prior variances:

$$
p(W^{(l)}) = \mathcal{N}(0, \sigma_l^2 I)
$$

**Common choices**:

**1. Constant variance**:
$$
\sigma_l^2 = \sigma^2 \quad \forall l
$$

**2. Fan-in scaling** (Xavier/Glorot-like):
$$
\sigma_l^2 = \frac{1}{n_{l-1}}
$$

**3. Fan-out scaling**:
$$
\sigma_l^2 = \frac{1}{n_l}
$$

**4. Fan-average scaling**:
$$
\sigma_l^2 = \frac{2}{n_{l-1} + n_l}
$$

where $n_l$ is the width of layer $l$.

### Variance Scaling for Activation Functions

The prior variance should account for the activation function:

**ReLU activations**:
$$
\sigma_l^2 = \frac{2}{n_{l-1}}
$$

**Tanh/Sigmoid activations**:
$$
\sigma_l^2 = \frac{1}{n_{l-1}}
$$

**Rationale**: Maintain stable variance of activations across layers at initialization.

---

## Priors Inducing Function Space Properties

### The Neal (1996) Prior

Neal showed that infinitely wide neural networks with specific priors converge to Gaussian processes.

**Setup**: Single hidden layer network with $H$ hidden units:

$$
f(x) = \sum_{h=1}^H v_h \, \sigma(w_h^\top x + b_h)
$$

**Prior**:
$$
v_h \sim \mathcal{N}(0, \sigma_v^2/H), \quad w_h \sim \mathcal{N}(0, \sigma_w^2 I), \quad b_h \sim \mathcal{N}(0, \sigma_b^2)
$$

**Result**: As $H \to \infty$, $f(x) \to \mathcal{GP}(0, k(x, x'))$ where:

$$
k(x, x') = \sigma_v^2 \, \mathbb{E}_{w, b}[\sigma(w^\top x + b) \, \sigma(w^\top x' + b)]
$$

### Neural Network Gaussian Process (NNGP) Kernel

For ReLU activation, the kernel has closed form:

$$
k(x, x') = \frac{\sigma_v^2}{\pi} \|x\| \|x'\| \left( \sin\phi + (\pi - \phi)\cos\phi \right)
$$

where $\phi = \cos^{-1}\left(\frac{x^\top x'}{\|x\| \|x'\|}\right)$.

**Implications**:
- Prior on weights induces specific function smoothness
- Deep networks induce compositional kernels
- Provides principled initialization guidance

### Depth and Prior Variance

For deep networks, variance must be scaled carefully:

**Without scaling**: Variance explodes or vanishes

$$
\text{Var}[f(x)] \propto \prod_{l=1}^L \text{Var}[W^{(l)}]
$$

**With proper scaling**: Variance remains $O(1)$

$$
\sigma_l^2 = \frac{c}{n_{l-1}}
$$

where $c$ depends on the activation function.

---

## Sparsity-Inducing Priors

### Motivation for Sparsity

**Benefits of sparse networks**:
1. Reduced overfitting
2. Improved interpretability
3. Computational efficiency
4. Better generalization

### Laplace Prior (L1 Regularization)

$$
p(w) = \frac{\lambda}{2} \exp(-\lambda |w|)
$$

**Properties**:
- Mode at zero (promotes exact sparsity in MAP)
- Heavier tails than Gaussian
- MAP equivalent to L1 regularization (Lasso)

**Limitation**: Not conjugate, complicates inference.

### Spike-and-Slab Prior

A mixture of a point mass at zero and a continuous distribution:

$$
\boxed{p(w) = \pi \, \delta_0(w) + (1-\pi) \, \mathcal{N}(w \mid 0, \sigma^2)}
$$

**Parameters**:
- $\pi$: Prior probability of weight being exactly zero
- $\sigma^2$: Variance of non-zero weights

**Inference**: Requires sampling the binary inclusion indicators.

### Continuous Relaxations

**Horseshoe prior**:

$$
w \mid \lambda \sim \mathcal{N}(0, \lambda^2), \quad \lambda \sim \text{Half-Cauchy}(0, \tau)
$$

**Properties**:
- Continuous (no point mass)
- Heavy tails allow large weights
- Strong shrinkage toward zero
- Global-local structure: $\tau$ is global, $\lambda$ is local

**Regularized horseshoe**:

$$
w \mid \lambda, c \sim \mathcal{N}(0, \tilde{\lambda}^2), \quad \tilde{\lambda}^2 = \frac{c^2 \lambda^2}{c^2 + \lambda^2}
$$

Bounds the maximum variance at $c^2$.

### Automatic Relevance Determination (ARD)

Per-input or per-feature precision:

$$
p(w_j \mid \alpha_j) = \mathcal{N}(w_j \mid 0, \alpha_j^{-1})
$$

$$
p(\alpha_j) = \text{Gamma}(\alpha_j \mid a_0, b_0)
$$

**Effect**: Features with large $\alpha_j$ are effectively pruned.

---

## Hierarchical Priors

### Motivation

**Problem**: Choosing prior hyperparameters (e.g., $\sigma^2$) is difficult.

**Solution**: Place hyperpriors on the hyperparameters and let data inform them.

### Two-Level Hierarchy

$$
p(\theta \mid \sigma^2) = \mathcal{N}(\theta \mid 0, \sigma^2 I)
$$

$$
p(\sigma^2) = \text{Inv-Gamma}(\sigma^2 \mid \alpha_0, \beta_0)
$$

**Marginal prior** (after integrating out $\sigma^2$):

$$
p(\theta) = \int p(\theta \mid \sigma^2) \, p(\sigma^2) \, d\sigma^2 = \text{Student-}t(\theta \mid 0, \frac{\beta_0}{\alpha_0}, 2\alpha_0)
$$

**Properties**:
- Heavier tails than Gaussian
- More robust to outliers
- Data-adaptive regularization

### Layer-wise Hierarchical Prior

Different variance for each layer:

$$
p(W^{(l)} \mid \sigma_l^2) = \mathcal{N}(0, \sigma_l^2 I)
$$

$$
p(\sigma_l^2) = \text{Inv-Gamma}(\alpha_0, \beta_0)
$$

**Advantages**:
- Each layer learns appropriate regularization
- Adapts to varying layer complexities
- Reduces sensitivity to initialization

### Group-wise Priors

**Per-neuron variance**:

$$
p(w_{:j}^{(l)} \mid \sigma_{lj}^2) = \mathcal{N}(0, \sigma_{lj}^2 I)
$$

**Per-filter variance** (for CNNs):

$$
p(W_k^{(l)} \mid \sigma_{lk}^2) = \mathcal{N}(0, \sigma_{lk}^2 I)
$$

This enables automatic pruning of entire neurons or filters.

---

## Scale Mixtures and Heavy-Tailed Priors

### Gaussian Scale Mixtures

Many useful priors can be written as:

$$
p(w) = \int \mathcal{N}(w \mid 0, \sigma^2) \, p(\sigma^2) \, d\sigma^2
$$

| Mixing distribution $p(\sigma^2)$ | Resulting $p(w)$ |
|-----------------------------------|------------------|
| Exponential | Laplace |
| Inverse-Gamma | Student-$t$ |
| Half-Cauchy on $\sigma$ | Horseshoe |
| Bernoulli-scaled | Spike-and-slab |

### Student-$t$ Prior

$$
p(w) = \frac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2})\sqrt{\nu\pi\sigma^2}} \left(1 + \frac{w^2}{\nu\sigma^2}\right)^{-\frac{\nu+1}{2}}
$$

**Degrees of freedom** $\nu$ controls tail heaviness:
- $\nu = 1$: Cauchy (very heavy tails)
- $\nu \to \infty$: Approaches Gaussian
- $\nu = 3$-$7$: Good compromise

**As scale mixture**:

$$
w \mid \tau \sim \mathcal{N}(0, \tau), \quad \tau \sim \text{Inv-Gamma}(\nu/2, \nu\sigma^2/2)
$$

### Benefits of Heavy Tails

1. **Robustness**: Less sensitive to prior misspecification
2. **Large weights**: Allows occasional large values when supported by data
3. **Automatic relevance**: Heavy tails combined with peaked center enable soft sparsity

---

## Correlated and Structured Priors

### Beyond Independence

Independent priors ignore structure:
- Weights connecting to same unit may be related
- Spatial structure in convolutional filters
- Temporal structure in recurrent networks

### Matrix-Variate Gaussian

For weight matrix $W \in \mathbb{R}^{m \times n}$:

$$
p(W) = \mathcal{MN}(W \mid M, U, V)
$$

where:
- $M$ is the mean matrix
- $U \in \mathbb{R}^{m \times m}$ captures row correlations
- $V \in \mathbb{R}^{n \times n}$ captures column correlations

**Equivalently**:

$$
\text{vec}(W) \sim \mathcal{N}(\text{vec}(M), V \otimes U)
$$

### Low-Rank Priors

Encourage weight matrices to be approximately low-rank:

$$
W = UV^\top + E
$$

where $U \in \mathbb{R}^{m \times r}$, $V \in \mathbb{R}^{n \times r}$, $r \ll \min(m,n)$.

**Prior**:

$$
p(U) = \mathcal{N}(0, I), \quad p(V) = \mathcal{N}(0, I), \quad p(E) = \mathcal{N}(0, \sigma_E^2 I)
$$

**Benefits**:
- Reduces effective parameter count
- Encourages compression
- May improve generalization

### Convolutional Structure

For convolutional layers, priors can respect spatial structure:

**Local smoothness**: Nearby filter weights should be similar

$$
p(W) \propto \exp\left(-\frac{1}{2\sigma^2} \sum_{i,j} (w_{ij} - w_{i+1,j})^2 + (w_{ij} - w_{i,j+1})^2 \right)
$$

**Translation equivariance**: Already built into CNN architecture.

---

## Empirical and Data-Dependent Priors

### Empirical Bayes

Use data to estimate prior hyperparameters:

$$
\hat{\eta} = \arg\max_\eta \log p(\mathcal{D} \mid \eta) = \arg\max_\eta \log \int p(\mathcal{D} \mid \theta) \, p(\theta \mid \eta) \, d\theta
$$

**Type II Maximum Likelihood**: Maximizes marginal likelihood.

**Pros**:
- Data-adaptive
- Can improve posterior accuracy

**Cons**:
- Uses data twice (for prior and posterior)
- May underestimate uncertainty
- Computationally expensive

### Transfer Learning Priors

Use posteriors from related tasks as priors:

$$
p(\theta) \approx p(\theta \mid \mathcal{D}_{\text{source}})
$$

**Approaches**:

**1. Mean-field approximation**:
$$
p(\theta) = \mathcal{N}(\theta \mid \mu_{\text{source}}, \sigma_{\text{source}}^2 I)
$$

**2. Mixture of pre-trained models**:
$$
p(\theta) = \sum_{k} \pi_k \, p(\theta \mid \mathcal{D}_k)
$$

**3. Centered prior** (L2-SP):
$$
p(\theta) = \mathcal{N}(\theta \mid \theta_{\text{pretrained}}, \sigma^2 I)
$$

### Functional Priors

Specify priors directly in function space:

$$
p(f) = \mathcal{GP}(m(x), k(x, x'))
$$

Then find $p(\theta)$ that induces approximately this $p(f)$.

**Challenges**:
- No closed-form relationship $p(\theta) \to p(f)$ for finite networks
- Requires sampling-based approaches
- Active research area

---

## Practical Considerations

### Choosing Prior Variances

**Rule of thumb**: Initialize such that:
- Pre-activations have variance $\approx 1$
- Gradients have variance $\approx 1$

**For fully connected layers**:

$$
\sigma_W^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}, \quad \sigma_b^2 = 0.01
$$

**For convolutional layers** with kernel size $k \times k$ and $c$ channels:

$$
\sigma_W^2 = \frac{2}{k^2 \cdot c_{\text{in}}}
$$

### Bias Priors

**Common approaches**:

**1. Zero-centered Gaussian**:
$$
p(b) = \mathcal{N}(0, \sigma_b^2)
$$
Use small $\sigma_b^2$ (e.g., $0.01$-$0.1$).

**2. Fixed at zero**:
$$
p(b) = \delta_0(b)
$$
Reduces parameters; sometimes works well.

**3. Data-dependent initialization**:
Center biases based on data statistics.

### Handling Different Layer Types

| Layer Type | Prior Consideration |
|------------|---------------------|
| **Dense** | Standard Gaussian, fan-in/out scaling |
| **Conv** | Per-filter or per-channel variance |
| **BatchNorm** | Often fixed; scale/shift learnable |
| **Embedding** | Per-embedding or tied variance |
| **Attention** | Key-query-value may need different scales |

### Prior Predictive Checks

**Validate priors** by sampling and examining induced functions:

```python
# Sample weights from prior
theta_prior = sample_from_prior()

# Generate predictions on test inputs
f_prior = network(x_test, theta_prior)

# Check: Are these reasonable functions?
```

**What to check**:
- Output scale: Are predictions in reasonable range?
- Smoothness: Are functions too wiggly or too flat?
- Extrapolation: What happens far from training region?

---

## Python Implementation

```python
"""
Prior Distributions on Neural Network Weights

This module provides implementations of various prior distributions
for Bayesian neural networks, including standard Gaussians, sparsity-inducing
priors, hierarchical priors, and utilities for prior predictive checking.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln, gamma
from typing import Tuple, List, Optional, Dict, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod


# =============================================================================
# Base Prior Classes
# =============================================================================

class Prior(ABC):
    """Abstract base class for weight priors."""
    
    @abstractmethod
    def log_prob(self, w: np.ndarray) -> float:
        """Compute log probability of weights."""
        pass
    
    @abstractmethod
    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Sample weights from the prior."""
        pass
    
    def prob(self, w: np.ndarray) -> float:
        """Compute probability (may underflow)."""
        return np.exp(self.log_prob(w))


class GaussianPrior(Prior):
    """
    Isotropic Gaussian prior: w ~ N(0, sigma^2 I)
    """
    
    def __init__(self, sigma: float = 1.0, mean: float = 0.0):
        """
        Parameters
        ----------
        sigma : float
            Prior standard deviation
        mean : float
            Prior mean (default 0)
        """
        self.sigma = sigma
        self.mean = mean
        self.var = sigma ** 2
    
    def log_prob(self, w: np.ndarray) -> float:
        """Compute log p(w)."""
        w = np.asarray(w)
        n = w.size
        
        return (
            -0.5 * n * np.log(2 * np.pi * self.var)
            - 0.5 * np.sum((w - self.mean) ** 2) / self.var
        )
    
    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Sample from prior."""
        return np.random.normal(self.mean, self.sigma, shape)
    
    def __repr__(self):
        return f"GaussianPrior(mean={self.mean}, sigma={self.sigma})"


class LaplacePrior(Prior):
    """
    Laplace prior: p(w) = (lambda/2) * exp(-lambda * |w|)
    
    Equivalent to L1 regularization in MAP estimation.
    """
    
    def __init__(self, scale: float = 1.0):
        """
        Parameters
        ----------
        scale : float
            Scale parameter (1/lambda)
        """
        self.scale = scale
        self.rate = 1.0 / scale
    
    def log_prob(self, w: np.ndarray) -> float:
        """Compute log p(w)."""
        w = np.asarray(w)
        n = w.size
        
        return n * np.log(self.rate / 2) - self.rate * np.sum(np.abs(w))
    
    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Sample from prior."""
        return np.random.laplace(0, self.scale, shape)
    
    def __repr__(self):
        return f"LaplacePrior(scale={self.scale})"


class StudentTPrior(Prior):
    """
    Student-t prior with specified degrees of freedom.
    
    Heavier tails than Gaussian; robust to outliers.
    """
    
    def __init__(self, df: float = 3.0, scale: float = 1.0):
        """
        Parameters
        ----------
        df : float
            Degrees of freedom (nu)
        scale : float
            Scale parameter
        """
        self.df = df
        self.scale = scale
    
    def log_prob(self, w: np.ndarray) -> float:
        """Compute log p(w)."""
        w = np.asarray(w)
        return np.sum(stats.t.logpdf(w, df=self.df, scale=self.scale))
    
    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Sample from prior."""
        return stats.t.rvs(df=self.df, scale=self.scale, size=shape)
    
    def __repr__(self):
        return f"StudentTPrior(df={self.df}, scale={self.scale})"


# =============================================================================
# Sparsity-Inducing Priors
# =============================================================================

class SpikeAndSlabPrior(Prior):
    """
    Spike-and-slab prior: p(w) = pi * delta_0 + (1-pi) * N(0, sigma^2)
    
    For continuous relaxation suitable for gradient-based inference.
    """
    
    def __init__(
        self,
        pi: float = 0.5,
        sigma_slab: float = 1.0,
        sigma_spike: float = 0.01
    ):
        """
        Parameters
        ----------
        pi : float
            Prior probability of being in the spike (near zero)
        sigma_slab : float
            Standard deviation of the slab component
        sigma_spike : float
            Standard deviation of the spike component (small)
        """
        self.pi = pi
        self.sigma_slab = sigma_slab
        self.sigma_spike = sigma_spike
    
    def log_prob(self, w: np.ndarray) -> float:
        """Compute log p(w) using log-sum-exp."""
        w = np.asarray(w)
        
        # Log probability under each component
        log_spike = (
            np.log(self.pi)
            + stats.norm.logpdf(w, 0, self.sigma_spike)
        )
        log_slab = (
            np.log(1 - self.pi)
            + stats.norm.logpdf(w, 0, self.sigma_slab)
        )
        
        # Log-sum-exp for numerical stability
        log_probs = np.logaddexp(log_spike, log_slab)
        
        return np.sum(log_probs)
    
    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Sample from prior."""
        # Sample component indicators
        is_spike = np.random.random(shape) < self.pi
        
        # Sample from appropriate component
        samples = np.where(
            is_spike,
            np.random.normal(0, self.sigma_spike, shape),
            np.random.normal(0, self.sigma_slab, shape)
        )
        
        return samples
    
    def __repr__(self):
        return f"SpikeAndSlabPrior(pi={self.pi}, sigma_spike={self.sigma_spike}, sigma_slab={self.sigma_slab})"


class HorseshoePrior(Prior):
    """
    Horseshoe prior: w | lambda ~ N(0, lambda^2), lambda ~ Half-Cauchy(0, tau)
    
    Strong shrinkage toward zero with heavy tails.
    """
    
    def __init__(self, tau: float = 1.0):
        """
        Parameters
        ----------
        tau : float
            Global scale parameter
        """
        self.tau = tau
    
    def log_prob(self, w: np.ndarray) -> float:
        """
        Approximate log probability (marginalizing over lambda is intractable).
        Uses the approximation: p(w) ≈ log(1 + 2*tau^2/w^2) for |w| >> 0
        """
        w = np.asarray(w)
        # Avoid log(0) by adding small epsilon
        eps = 1e-10
        
        # Approximation to marginal horseshoe density
        log_prob = np.sum(np.log(np.log(1 + 2 * self.tau**2 / (w**2 + eps))))
        
        return log_prob
    
    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Sample from prior."""
        # Sample local scales from half-Cauchy
        lambdas = np.abs(stats.cauchy.rvs(size=shape)) * self.tau
        
        # Sample weights
        return np.random.normal(0, lambdas)
    
    def __repr__(self):
        return f"HorseshoePrior(tau={self.tau})"


# =============================================================================
# Hierarchical Priors
# =============================================================================

class HierarchicalGaussianPrior(Prior):
    """
    Hierarchical Gaussian prior with Inverse-Gamma on variance.
    
    w | sigma^2 ~ N(0, sigma^2)
    sigma^2 ~ Inv-Gamma(alpha, beta)
    
    Marginalizes to Student-t.
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 1.0):
        """
        Parameters
        ----------
        alpha : float
            Shape parameter of Inverse-Gamma
        beta : float
            Scale parameter of Inverse-Gamma
        """
        self.alpha = alpha
        self.beta = beta
        
        # Marginal is Student-t with df = 2*alpha
        self.marginal_df = 2 * alpha
        self.marginal_scale = np.sqrt(beta / alpha)
    
    def log_prob(self, w: np.ndarray) -> float:
        """Compute log probability of marginal (Student-t)."""
        w = np.asarray(w)
        return np.sum(stats.t.logpdf(
            w, df=self.marginal_df, scale=self.marginal_scale
        ))
    
    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Sample from marginal (Student-t)."""
        return stats.t.rvs(
            df=self.marginal_df,
            scale=self.marginal_scale,
            size=shape
        )
    
    def sample_conditional(
        self,
        shape: Tuple[int, ...],
        return_variance: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Sample hierarchically: first variance, then weights.
        """
        # Sample variance from Inverse-Gamma
        sigma_sq = stats.invgamma.rvs(self.alpha, scale=self.beta)
        
        # Sample weights given variance
        w = np.random.normal(0, np.sqrt(sigma_sq), shape)
        
        if return_variance:
            return w, sigma_sq
        return w
    
    def __repr__(self):
        return f"HierarchicalGaussianPrior(alpha={self.alpha}, beta={self.beta})"


class LayerWisePrior:
    """
    Different priors for different layers.
    """
    
    def __init__(self, layer_priors: Dict[str, Prior]):
        """
        Parameters
        ----------
        layer_priors : dict
            Mapping from layer name to Prior object
        """
        self.layer_priors = layer_priors
    
    def log_prob(self, weights: Dict[str, np.ndarray]) -> float:
        """Compute total log probability across all layers."""
        total = 0.0
        for name, w in weights.items():
            if name in self.layer_priors:
                total += self.layer_priors[name].log_prob(w)
        return total
    
    def sample(self, shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, np.ndarray]:
        """Sample weights for all layers."""
        return {
            name: self.layer_priors[name].sample(shape)
            for name, shape in shapes.items()
            if name in self.layer_priors
        }


# =============================================================================
# Variance Scaling Utilities
# =============================================================================

def compute_glorot_variance(fan_in: int, fan_out: int) -> float:
    """
    Glorot/Xavier variance scaling.
    
    Maintains variance of activations across layers for tanh/sigmoid.
    """
    return 2.0 / (fan_in + fan_out)


def compute_he_variance(fan_in: int) -> float:
    """
    He variance scaling.
    
    Maintains variance for ReLU activations.
    """
    return 2.0 / fan_in


def compute_lecun_variance(fan_in: int) -> float:
    """
    LeCun variance scaling.
    
    Maintains variance for SELU activations.
    """
    return 1.0 / fan_in


def create_scaled_gaussian_prior(
    layer_shapes: List[Tuple[int, int]],
    scaling: str = 'glorot'
) -> LayerWisePrior:
    """
    Create layer-wise Gaussian priors with proper variance scaling.
    
    Parameters
    ----------
    layer_shapes : list of tuples
        (fan_in, fan_out) for each layer
    scaling : str
        'glorot', 'he', or 'lecun'
    
    Returns
    -------
    LayerWisePrior
        Prior with layer-appropriate variances
    """
    priors = {}
    
    for i, (fan_in, fan_out) in enumerate(layer_shapes):
        if scaling == 'glorot':
            var = compute_glorot_variance(fan_in, fan_out)
        elif scaling == 'he':
            var = compute_he_variance(fan_in)
        elif scaling == 'lecun':
            var = compute_lecun_variance(fan_in)
        else:
            raise ValueError(f"Unknown scaling: {scaling}")
        
        priors[f'W{i}'] = GaussianPrior(sigma=np.sqrt(var))
        priors[f'b{i}'] = GaussianPrior(sigma=0.1)  # Small bias prior
    
    return LayerWisePrior(priors)


# =============================================================================
# Prior Predictive Sampling
# =============================================================================

class SimpleMLP:
    """Simple MLP for prior predictive checking."""
    
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = 'relu'
    ):
        """
        Parameters
        ----------
        layer_sizes : list
            [input_dim, hidden1, hidden2, ..., output_dim]
        activation : str
            'relu', 'tanh', or 'sigmoid'
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        
        # Define activation function
        if activation == 'relu':
            self.act_fn = lambda x: np.maximum(x, 0)
        elif activation == 'tanh':
            self.act_fn = np.tanh
        elif activation == 'sigmoid':
            self.act_fn = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(
        self,
        x: np.ndarray,
        weights: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Forward pass with given weights."""
        h = x
        n_layers = len(self.layer_sizes) - 1
        
        for i in range(n_layers):
            W = weights[f'W{i}']
            b = weights.get(f'b{i}', np.zeros(W.shape[1]))
            
            h = h @ W + b
            
            # Apply activation (except last layer)
            if i < n_layers - 1:
                h = self.act_fn(h)
        
        return h
    
    def get_weight_shapes(self) -> Dict[str, Tuple[int, int]]:
        """Get shapes of all weight matrices."""
        shapes = {}
        for i in range(len(self.layer_sizes) - 1):
            shapes[f'W{i}'] = (self.layer_sizes[i], self.layer_sizes[i + 1])
            shapes[f'b{i}'] = (self.layer_sizes[i + 1],)
        return shapes


def prior_predictive_check(
    model: SimpleMLP,
    prior: Union[Prior, LayerWisePrior],
    x_test: np.ndarray,
    n_samples: int = 100
) -> np.ndarray:
    """
    Sample functions from the prior predictive distribution.
    
    Parameters
    ----------
    model : SimpleMLP
        Neural network architecture
    prior : Prior or LayerWisePrior
        Prior on weights
    x_test : ndarray
        Test inputs
    n_samples : int
        Number of function samples
    
    Returns
    -------
    ndarray of shape (n_samples, n_test_points, output_dim)
        Function samples
    """
    shapes = model.get_weight_shapes()
    
    predictions = []
    
    for _ in range(n_samples):
        # Sample weights from prior
        if isinstance(prior, LayerWisePrior):
            weights = prior.sample(shapes)
        else:
            weights = {name: prior.sample(shape) for name, shape in shapes.items()}
        
        # Forward pass
        y = model.forward(x_test, weights)
        predictions.append(y)
    
    return np.array(predictions)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_prior_comparison(
    priors: Dict[str, Prior],
    x_range: Tuple[float, float] = (-5, 5),
    n_points: int = 1000,
    figsize: Tuple[float, float] = (12, 5)
):
    """
    Compare different prior distributions.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    # PDF comparison
    ax = axes[0]
    for name, prior in priors.items():
        if isinstance(prior, GaussianPrior):
            pdf = stats.norm.pdf(x, prior.mean, prior.sigma)
        elif isinstance(prior, LaplacePrior):
            pdf = stats.laplace.pdf(x, 0, prior.scale)
        elif isinstance(prior, StudentTPrior):
            pdf = stats.t.pdf(x, prior.df, scale=prior.scale)
        else:
            # Numerical approximation
            pdf = np.array([prior.prob(np.array([xi])) for xi in x])
        
        ax.plot(x, pdf, label=name, linewidth=2)
    
    ax.set_xlabel('Weight value')
    ax.set_ylabel('Density')
    ax.set_title('Prior Densities')
    ax.legend()
    ax.set_ylim(0, None)
    
    # Log-PDF comparison (to see tails)
    ax = axes[1]
    for name, prior in priors.items():
        if isinstance(prior, GaussianPrior):
            log_pdf = stats.norm.logpdf(x, prior.mean, prior.sigma)
        elif isinstance(prior, LaplacePrior):
            log_pdf = stats.laplace.logpdf(x, 0, prior.scale)
        elif isinstance(prior, StudentTPrior):
            log_pdf = stats.t.logpdf(x, prior.df, scale=prior.scale)
        else:
            log_pdf = np.array([prior.log_prob(np.array([xi])) for xi in x])
        
        ax.plot(x, log_pdf, label=name, linewidth=2)
    
    ax.set_xlabel('Weight value')
    ax.set_ylabel('Log Density')
    ax.set_title('Log Prior Densities (shows tail behavior)')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_prior_predictive(
    predictions: np.ndarray,
    x_test: np.ndarray,
    title: str = "Prior Predictive Distribution",
    n_show: int = 20
):
    """
    Visualize prior predictive samples.
    
    Parameters
    ----------
    predictions : ndarray of shape (n_samples, n_points)
        Function samples
    x_test : ndarray
        Test inputs (1D)
    title : str
        Plot title
    n_show : int
        Number of samples to plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sample functions
    ax = axes[0]
    for i in range(min(n_show, len(predictions))):
        ax.plot(x_test, predictions[i], alpha=0.3, color='blue')
    
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'{title}\n(showing {n_show} samples)')
    
    # Mean and uncertainty
    ax = axes[1]
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    
    ax.fill_between(x_test.flatten(), mean - 2*std, mean + 2*std,
                    alpha=0.3, label='±2σ')
    ax.plot(x_test, mean, 'b-', linewidth=2, label='Mean')
    
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Prior Predictive Mean and Uncertainty')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_sparsity_pattern(
    prior: Prior,
    n_samples: int = 10000,
    title: str = "Sparsity Pattern"
):
    """
    Visualize the sparsity-inducing property of a prior.
    """
    samples = prior.sample((n_samples,))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(samples, bins=100, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='Zero')
    ax.set_xlabel('Weight value')
    ax.set_ylabel('Density')
    ax.set_title(f'{title}\nHistogram of samples')
    ax.legend()
    
    # Fraction near zero
    ax = axes[1]
    thresholds = np.logspace(-3, 0, 50)
    fractions = [np.mean(np.abs(samples) < t) for t in thresholds]
    
    ax.semilogx(thresholds, fractions, 'b-', linewidth=2)
    ax.set_xlabel('Threshold |w| < τ')
    ax.set_ylabel('Fraction of weights')
    ax.set_title('Cumulative near-zero fraction')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nSparsity statistics for {prior}:")
    print(f"  Fraction |w| < 0.01: {np.mean(np.abs(samples) < 0.01):.3f}")
    print(f"  Fraction |w| < 0.1:  {np.mean(np.abs(samples) < 0.1):.3f}")
    print(f"  Fraction |w| > 1.0:  {np.mean(np.abs(samples) > 1.0):.3f}")


# =============================================================================
# Demo Functions
# =============================================================================

def demo_standard_priors():
    """Compare standard prior distributions."""
    
    print("=" * 70)
    print("STANDARD PRIOR DISTRIBUTIONS")
    print("=" * 70)
    
    priors = {
        'Gaussian (σ=1)': GaussianPrior(sigma=1.0),
        'Gaussian (σ=0.5)': GaussianPrior(sigma=0.5),
        'Laplace (scale=1)': LaplacePrior(scale=1.0),
        'Student-t (ν=3)': StudentTPrior(df=3.0, scale=1.0),
        'Student-t (ν=10)': StudentTPrior(df=10.0, scale=1.0),
    }
    
    print("\nPrior summary:")
    for name, prior in priors.items():
        samples = prior.sample((10000,))
        print(f"  {name:25s}: mean={np.mean(samples):+.3f}, "
              f"std={np.std(samples):.3f}, "
              f"|w|>2: {np.mean(np.abs(samples) > 2):.3f}")
    
    return priors


def demo_sparsity_priors():
    """Demonstrate sparsity-inducing priors."""
    
    print("\n" + "=" * 70)
    print("SPARSITY-INDUCING PRIORS")
    print("=" * 70)
    
    priors = {
        'Gaussian': GaussianPrior(sigma=1.0),
        'Laplace': LaplacePrior(scale=1.0),
        'Spike-and-Slab': SpikeAndSlabPrior(pi=0.8, sigma_spike=0.01, sigma_slab=1.0),
        'Horseshoe': HorseshoePrior(tau=1.0),
    }
    
    print("\nSparsity comparison (fraction near zero):")
    print(f"{'Prior':<20} {'|w|<0.01':>10} {'|w|<0.1':>10} {'|w|>2':>10}")
    print("-" * 55)
    
    for name, prior in priors.items():
        samples = prior.sample((10000,))
        print(f"{name:<20} {np.mean(np.abs(samples) < 0.01):>10.3f} "
              f"{np.mean(np.abs(samples) < 0.1):>10.3f} "
              f"{np.mean(np.abs(samples) > 2):>10.3f}")
    
    return priors


def demo_hierarchical_prior():
    """Demonstrate hierarchical priors."""
    
    print("\n" + "=" * 70)
    print("HIERARCHICAL PRIORS")
    print("=" * 70)
    
    # Compare different alpha values
    alphas = [1.0, 2.0, 5.0, 10.0]
    
    print("\nHierarchical Gaussian with Inv-Gamma(α, β=1) on variance:")
    print(f"{'α':>5} {'Marginal df':>12} {'Sample std':>12} {'Kurtosis':>12}")
    print("-" * 45)
    
    for alpha in alphas:
        prior = HierarchicalGaussianPrior(alpha=alpha, beta=1.0)
        samples = prior.sample((10000,))
        
        # Kurtosis (excess, Gaussian = 0)
        kurtosis = stats.kurtosis(samples)
        
        print(f"{alpha:>5.1f} {prior.marginal_df:>12.1f} "
              f"{np.std(samples):>12.3f} {kurtosis:>12.3f}")
    
    print("\n*** Lower α → heavier tails (higher kurtosis)")
    print("*** As α → ∞, marginal approaches Gaussian")


def demo_variance_scaling():
    """Demonstrate variance scaling for different architectures."""
    
    print("\n" + "=" * 70)
    print("VARIANCE SCALING")
    print("=" * 70)
    
    # Example architecture
    layer_sizes = [784, 256, 128, 10]
    
    print(f"\nArchitecture: {layer_sizes}")
    print("\nRecommended prior standard deviations:")
    print(f"{'Layer':<10} {'Shape':<15} {'Glorot σ':>12} {'He σ':>12} {'LeCun σ':>12}")
    print("-" * 65)
    
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        shape = f"({fan_in}, {fan_out})"
        
        glorot_var = compute_glorot_variance(fan_in, fan_out)
        he_var = compute_he_variance(fan_in)
        lecun_var = compute_lecun_variance(fan_in)
        
        print(f"W{i:<8} {shape:<15} {np.sqrt(glorot_var):>12.4f} "
              f"{np.sqrt(he_var):>12.4f} {np.sqrt(lecun_var):>12.4f}")
    
    print("\n*** Glorot: for tanh/sigmoid")
    print("*** He: for ReLU")
    print("*** LeCun: for SELU")


def demo_prior_predictive():
    """Demonstrate prior predictive checking."""
    
    print("\n" + "=" * 70)
    print("PRIOR PREDICTIVE CHECKING")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Create simple MLP
    model = SimpleMLP([1, 50, 50, 1], activation='tanh')
    
    # Test inputs
    x_test = np.linspace(-3, 3, 200).reshape(-1, 1)
    
    # Different priors
    prior_configs = {
        'Wide Gaussian (σ=1)': GaussianPrior(sigma=1.0),
        'Narrow Gaussian (σ=0.1)': GaussianPrior(sigma=0.1),
        'He-scaled': None,  # Will use proper scaling
    }
    
    print("\nPrior predictive statistics (over test points):")
    print(f"{'Prior':<25} {'Output mean':>12} {'Output std':>12} {'Max |y|':>12}")
    print("-" * 65)
    
    for name, prior in prior_configs.items():
        if prior is None:
            # He-scaled prior
            shapes = [(1, 50), (50, 50), (50, 1)]
            prior = create_scaled_gaussian_prior(shapes, scaling='he')
        
        predictions = prior_predictive_check(model, prior, x_test, n_samples=100)
        predictions = predictions.squeeze()
        
        print(f"{name:<25} {np.mean(predictions):>12.3f} "
              f"{np.std(predictions):>12.3f} {np.max(np.abs(predictions)):>12.3f}")
    
    print("\n*** Proper scaling keeps outputs in reasonable range")
    print("*** Wide priors can lead to extreme function values")


def demo_l2_equivalence():
    """Demonstrate Gaussian prior ↔ L2 regularization equivalence."""
    
    print("\n" + "=" * 70)
    print("GAUSSIAN PRIOR ↔ L2 REGULARIZATION")
    print("=" * 70)
    
    print("\nMAP estimation with Gaussian prior N(0, σ²) is equivalent to")
    print("L2-regularized MLE with penalty λ = 1/(2σ²)")
    
    print(f"\n{'σ²':>10} {'σ':>10} {'λ = 1/(2σ²)':>15} {'Interpretation':>25}")
    print("-" * 65)
    
    variances = [0.01, 0.1, 0.5, 1.0, 10.0, 100.0]
    
    for var in variances:
        sigma = np.sqrt(var)
        lam = 1 / (2 * var)
        
        if lam > 10:
            interp = "Very strong regularization"
        elif lam > 1:
            interp = "Strong regularization"
        elif lam > 0.1:
            interp = "Moderate regularization"
        elif lam > 0.01:
            interp = "Weak regularization"
        else:
            interp = "Very weak regularization"
        
        print(f"{var:>10.2f} {sigma:>10.3f} {lam:>15.4f} {interp:>25}")


if __name__ == "__main__":
    demo_standard_priors()
    demo_sparsity_priors()
    demo_hierarchical_prior()
    demo_variance_scaling()
    demo_prior_predictive()
    demo_l2_equivalence()
```

---

## Summary

### Standard Priors

| Prior | Formula | MAP Equivalent | Properties |
|-------|---------|----------------|------------|
| **Gaussian** | $\mathcal{N}(0, \sigma^2)$ | L2 regularization | Smooth, well-behaved |
| **Laplace** | $\frac{\lambda}{2}e^{-\lambda\|w\|}$ | L1 regularization | Promotes sparsity |
| **Student-$t$** | Heavy-tailed | Robust penalty | Allows outliers |

### Variance Scaling

| Method | Formula | Best For |
|--------|---------|----------|
| **Glorot/Xavier** | $\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$ | Tanh, Sigmoid |
| **He** | $\sigma^2 = \frac{2}{n_{\text{in}}}$ | ReLU |
| **LeCun** | $\sigma^2 = \frac{1}{n_{\text{in}}}$ | SELU |

### Sparsity-Inducing Priors

| Prior | Key Feature | Use Case |
|-------|-------------|----------|
| **Spike-and-Slab** | Exact zeros | Feature selection |
| **Horseshoe** | Heavy tails + shrinkage | Sparse signals |
| **ARD** | Per-feature variance | Automatic pruning |

### Hierarchical Priors

$$
p(w \mid \sigma^2) = \mathcal{N}(0, \sigma^2), \quad p(\sigma^2) = \text{Inv-Gamma}(\alpha, \beta)
$$

**Benefits**:
- Data-adaptive regularization
- Heavier tails than Gaussian
- Reduced sensitivity to hyperparameters

### Key Design Principles

1. **Scale appropriately**: Match variance to layer width
2. **Consider function space**: Prior on weights induces prior on functions
3. **Use hierarchical priors**: Let data inform regularization strength
4. **Validate with prior predictive checks**: Sample and visualize before training

### Connections to Other Chapters

| Topic | Chapter | Connection |
|-------|---------|------------|
| Posterior inference | Ch13: Posterior Inference | Prior affects posterior shape |
| Uncertainty | Ch13: Uncertainty | Prior affects epistemic uncertainty |
| MC Dropout | Ch13: MC Dropout | Implicitly defines prior |
| Variational BNN | Ch13: Variational BNN | Prior in KL divergence |
| Regularization | Ch6: Regularization | MAP ↔ penalized MLE |

### Key References

- Neal, R. M. (1996). *Bayesian Learning for Neural Networks*. Springer.
- Blundell, C., et al. (2015). Weight uncertainty in neural networks. *ICML*.
- Louizos, C., et al. (2017). Bayesian compression for deep learning. *NeurIPS*.
- Fortuin, V. (2022). Priors in Bayesian deep learning: A review. *International Statistical Review*.
- Wenzel, F., et al. (2020). How good is the Bayes posterior in deep neural networks really? *ICML*.
