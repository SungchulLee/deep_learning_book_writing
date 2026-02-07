# SWAG: Stochastic Weight Averaging Gaussian

**SWAG (Stochastic Weight Averaging Gaussian)** provides a simple, scalable approach to Bayesian inference in neural networks by fitting a Gaussian distribution to the trajectory of SGD iterates. This post-hoc method captures both the mean and covariance of the weight posterior using statistics collected during standard training.

---

## Motivation

### The Gap Between Training and Bayesian Inference

Standard neural network training produces a point estimate $\hat{\theta}$:

$$
\hat{\theta} = \arg\min_\theta \mathcal{L}(\theta; \mathcal{D})
$$

To get uncertainty estimates, we need the posterior $p(\theta \mid \mathcal{D})$. Full Bayesian methods (MCMC, VI) are expensive and require architectural changes.

**SWAG's insight**: The SGD trajectory near convergence implicitly explores the posterior landscape. By collecting statistics along this trajectory, we can approximate the posterior without changing the training procedure.

### Stochastic Weight Averaging (SWA) Background

**SWA** (Izmailov et al., 2018) improves generalization by averaging weights:

$$
\bar{\theta}_{\text{SWA}} = \frac{1}{T} \sum_{t=1}^T \theta_t
$$

where $\theta_t$ are weights from the last $T$ epochs with cyclical or constant learning rate.

**Key observation**: SWA weights lie in flatter, wider regions of the loss landscape, leading to better generalization.

**SWAG extends SWA** by also capturing the spread of the weight trajectory, not just the mean.

---

## SWAG Algorithm

### Core Idea

Fit a Gaussian to the SGD trajectory:

$$
\boxed{q(\theta) = \mathcal{N}(\theta \mid \bar{\theta}, \Sigma_{\text{SWAG}})}
$$

The covariance $\Sigma_{\text{SWAG}}$ is approximated using a low-rank plus diagonal structure.

### Moment Collection

During training (after initial burn-in), collect:

**First moment** (running mean):

$$
\bar{\theta} = \frac{1}{T} \sum_{t=1}^T \theta_t
$$

**Second moment** (running squared mean):

$$
\overline{\theta^2} = \frac{1}{T} \sum_{t=1}^T \theta_t^2
$$

**Deviation matrix** (for low-rank component):

$$
D = [\theta_1 - \bar{\theta}, \theta_2 - \bar{\theta}, \ldots, \theta_K - \bar{\theta}]
$$

where we keep only the last $K$ deviations (typically $K = 20$).

### Covariance Approximation

**Full covariance** would be $O(d^2)$ — intractable for large networks.

**SWAG approximation**:

$$
\boxed{\Sigma_{\text{SWAG}} = \Sigma_{\text{diag}} + \Sigma_{\text{low-rank}}}
$$

**Diagonal component**:

$$
\Sigma_{\text{diag}} = \text{diag}\left(\overline{\theta^2} - \bar{\theta}^2\right)
$$

**Low-rank component**:

$$
\Sigma_{\text{low-rank}} = \frac{1}{K-1} D D^\top
$$

### Sampling from SWAG

To sample $\theta \sim q(\theta)$:

$$
\theta = \bar{\theta} + \frac{1}{\sqrt{2}} \sqrt{\Sigma_{\text{diag}}} \odot z_1 + \frac{1}{\sqrt{2(K-1)}} D z_2
$$

where $z_1 \sim \mathcal{N}(0, I_d)$ and $z_2 \sim \mathcal{N}(0, I_K)$.

The $\frac{1}{\sqrt{2}}$ factors ensure proper scaling when combining diagonal and low-rank components.

---

## Algorithm Details

### Training Protocol

```
Algorithm: SWAG Training
────────────────────────
Input: Pre-trained network θ₀, learning rate schedule, 
       collection frequency c, rank K
Output: SWAG parameters (θ̄, Σ_diag, D)

1. Initialize: θ̄ = 0, θ̄² = 0, D = [], n = 0
2. Set learning rate to SWA schedule (constant or cyclical)
3. For each epoch after burn-in:
   a. Train for c iterations with SGD
   b. Update running statistics:
      n ← n + 1
      θ̄ ← (n-1)/n · θ̄ + 1/n · θ
      θ̄² ← (n-1)/n · θ̄² + 1/n · θ²
   c. Store deviation:
      D.append(θ - θ̄)
      If len(D) > K: D.pop(0)
4. Compute: Σ_diag = diag(θ̄² - θ̄²)
5. Return (θ̄, Σ_diag, D)
```

### Learning Rate Schedule

**Cyclical schedule** (recommended):

$$
\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})\left(1 + \cos\left(\frac{\pi \cdot \text{mod}(t, c)}{c}\right)\right)
$$

Collect samples at the end of each cycle when learning rate is low.

**Constant schedule**:

$$
\alpha_t = \alpha_{\text{SWA}}
$$

Simpler but may provide less diverse samples.

### Hyperparameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| SWA start | 75% of training | When to begin collecting |
| Collection freq $c$ | 1 epoch | How often to update statistics |
| Low-rank $K$ | 20 | Number of deviation vectors |
| Learning rate | 0.01-0.05 | SWA learning rate |

---

## Prediction with SWAG

### Monte Carlo Integration

Given test input $x^*$, sample $S$ weight configurations:

$$
\theta^{(s)} \sim q(\theta) = \mathcal{N}(\bar{\theta}, \Sigma_{\text{SWAG}})
$$

**Predictive mean**:

$$
\hat{\mu}(x^*) = \frac{1}{S} \sum_{s=1}^S f_{\theta^{(s)}}(x^*)
$$

**Predictive variance** (epistemic):

$$
\hat{\sigma}^2(x^*) = \frac{1}{S} \sum_{s=1}^S \left(f_{\theta^{(s)}}(x^*) - \hat{\mu}(x^*)\right)^2
$$

### For Classification

Average softmax probabilities:

$$
p(y = c \mid x^*, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^S \text{softmax}(f_{\theta^{(s)}}(x^*))_c
$$

**Uncertainty measures**:
- **Entropy**: $\mathbb{H}[\bar{p}] = -\sum_c \bar{p}_c \log \bar{p}_c$
- **Mutual information**: $\mathbb{I}[y; \theta] = \mathbb{H}[\bar{p}] - \frac{1}{S}\sum_s \mathbb{H}[p_s]$

---

## Theoretical Justification

### Connection to Laplace Approximation

SWAG can be viewed as an approximate Laplace approximation:

**Laplace**: $q(\theta) = \mathcal{N}(\theta_{\text{MAP}}, H^{-1})$

**SWAG**: $q(\theta) = \mathcal{N}(\bar{\theta}_{\text{SWA}}, \Sigma_{\text{SWAG}})$

The key differences:
- SWAG uses SWA mean (potentially flatter region) vs MAP
- SWAG approximates Hessian inverse via trajectory statistics

### Connection to SGLD

Under certain conditions, SGD with noise explores the posterior:

$$
\theta_{t+1} = \theta_t - \alpha \nabla \mathcal{L}(\theta_t) + \epsilon_t
$$

SWAG captures the marginal statistics of this exploration.

### Loss Landscape Perspective

SWAG samples from the "basin" around the SWA solution:
- Diagonal captures per-parameter variance
- Low-rank captures principal directions of variation
- Together they approximate the local posterior geometry

---

## SWAG Variants

### SWAG-Diagonal

Use only the diagonal covariance (no low-rank):

$$
\Sigma = \text{diag}\left(\overline{\theta^2} - \bar{\theta}^2\right)
$$

**Advantages**: Simpler, less storage
**Disadvantages**: Ignores weight correlations

### Multi-SWAG

Run SWAG from multiple random initializations:

$$
p(\theta \mid \mathcal{D}) \approx \frac{1}{M} \sum_{m=1}^M q_m(\theta)
$$

Captures multiple modes of the posterior.

### SWAG with BatchNorm

**Challenge**: BatchNorm statistics depend on the batch, not just weights.

**Solution**: After sampling weights, run a forward pass on training data to update BatchNorm running statistics before evaluation.

---

## Python Implementation

```python
"""
SWAG: Stochastic Weight Averaging Gaussian

A simple, scalable approach to approximate Bayesian inference
by fitting a Gaussian to the SGD trajectory.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque


class SWAG:
    """
    Stochastic Weight Averaging Gaussian.
    
    Approximates the posterior as a Gaussian with low-rank plus diagonal
    covariance, estimated from the SGD trajectory.
    """
    
    def __init__(
        self,
        n_params: int,
        max_rank: int = 20,
        var_clamp: float = 1e-6
    ):
        """
        Parameters
        ----------
        n_params : int
            Number of model parameters
        max_rank : int
            Maximum rank of low-rank covariance component
        var_clamp : float
            Minimum variance (for numerical stability)
        """
        self.n_params = n_params
        self.max_rank = max_rank
        self.var_clamp = var_clamp
        
        # Statistics
        self.mean = np.zeros(n_params)
        self.sq_mean = np.zeros(n_params)
        self.deviations = deque(maxlen=max_rank)
        self.n_models = 0
    
    def update(self, params: np.ndarray):
        """
        Update SWAG statistics with new parameters.
        
        Parameters
        ----------
        params : ndarray of shape (n_params,)
            Current model parameters
        """
        params = np.asarray(params).flatten()
        assert len(params) == self.n_params
        
        self.n_models += 1
        n = self.n_models
        
        # Update running mean
        old_mean = self.mean.copy()
        self.mean = (n - 1) / n * self.mean + 1 / n * params
        
        # Update running squared mean
        self.sq_mean = (n - 1) / n * self.sq_mean + 1 / n * (params ** 2)
        
        # Store deviation from current mean
        deviation = params - self.mean
        self.deviations.append(deviation)
    
    @property
    def variance(self) -> np.ndarray:
        """Diagonal variance."""
        var = self.sq_mean - self.mean ** 2
        return np.maximum(var, self.var_clamp)
    
    @property
    def deviation_matrix(self) -> np.ndarray:
        """Matrix of deviations for low-rank component."""
        if len(self.deviations) == 0:
            return np.zeros((self.n_params, 1))
        return np.column_stack(self.deviations)
    
    def sample(self, scale: float = 1.0) -> np.ndarray:
        """
        Sample from SWAG distribution.
        
        Parameters
        ----------
        scale : float
            Scale factor for uncertainty (1.0 = full uncertainty)
        
        Returns
        -------
        ndarray of shape (n_params,)
            Sampled parameters
        """
        # Sample from diagonal component
        z1 = np.random.randn(self.n_params)
        
        # Sample from low-rank component
        D = self.deviation_matrix
        K = D.shape[1]
        z2 = np.random.randn(K)
        
        # Combine (with proper scaling)
        std_diag = np.sqrt(self.variance)
        
        if K > 1:
            # Full SWAG: diagonal + low-rank
            sample = (
                self.mean 
                + scale * (1.0 / np.sqrt(2.0)) * std_diag * z1
                + scale * (1.0 / np.sqrt(2.0 * (K - 1))) * D @ z2
            )
        else:
            # SWAG-Diagonal only
            sample = self.mean + scale * std_diag * z1
        
        return sample
    
    def sample_many(self, n_samples: int, scale: float = 1.0) -> np.ndarray:
        """
        Sample multiple parameter vectors.
        
        Parameters
        ----------
        n_samples : int
            Number of samples
        scale : float
            Scale factor for uncertainty
        
        Returns
        -------
        ndarray of shape (n_samples, n_params)
            Sampled parameters
        """
        return np.array([self.sample(scale) for _ in range(n_samples)])
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """Get state for saving."""
        return {
            'mean': self.mean,
            'sq_mean': self.sq_mean,
            'deviations': np.array(list(self.deviations)),
            'n_models': self.n_models
        }
    
    def load_state(self, state: Dict[str, np.ndarray]):
        """Load state."""
        self.mean = state['mean']
        self.sq_mean = state['sq_mean']
        self.deviations = deque(state['deviations'], maxlen=self.max_rank)
        self.n_models = state['n_models']


class SWAGTrainer:
    """
    Trainer for SWAG with cyclical learning rate.
    """
    
    def __init__(
        self,
        model,
        swag: SWAG,
        lr_init: float = 0.01,
        lr_min: float = 0.001,
        cycle_length: int = 5,
        swa_start: int = 50
    ):
        """
        Parameters
        ----------
        model : object
            Neural network with get_params() and set_params() methods
        swag : SWAG
            SWAG object for collecting statistics
        lr_init : float
            Initial (max) learning rate
        lr_min : float
            Minimum learning rate
        cycle_length : int
            Number of epochs per cycle
        swa_start : int
            Epoch to start SWAG collection
        """
        self.model = model
        self.swag = swag
        self.lr_init = lr_init
        self.lr_min = lr_min
        self.cycle_length = cycle_length
        self.swa_start = swa_start
    
    def get_lr(self, epoch: int) -> float:
        """Cyclical learning rate schedule."""
        if epoch < self.swa_start:
            # Linear warmup or constant
            return self.lr_init
        
        # Cyclical after swa_start
        cycle_epoch = (epoch - self.swa_start) % self.cycle_length
        t = cycle_epoch / self.cycle_length
        
        return self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1 + np.cos(np.pi * t))
    
    def train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epoch: int,
        batch_size: int = 32
    ) -> float:
        """
        Train for one epoch.
        
        Returns
        -------
        float
            Average loss
        """
        lr = self.get_lr(epoch)
        losses = []
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        
        for start in range(0, len(X), batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            # Compute gradients and loss (model-specific)
            loss, grads = self.model.compute_gradients(X_batch, y_batch)
            losses.append(loss)
            
            # SGD update
            params = self.model.get_params()
            new_params = params - lr * grads
            self.model.set_params(new_params)
        
        # Update SWAG at end of cycle (when lr is low)
        if epoch >= self.swa_start:
            cycle_epoch = (epoch - self.swa_start) % self.cycle_length
            if cycle_epoch == self.cycle_length - 1:
                self.swag.update(self.model.get_params())
        
        return np.mean(losses)


def swag_predict(
    model,
    swag: SWAG,
    X: np.ndarray,
    n_samples: int = 30,
    scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with SWAG uncertainty.
    
    Parameters
    ----------
    model : object
        Neural network with set_params() and predict() methods
    swag : SWAG
        Fitted SWAG object
    X : ndarray
        Input data
    n_samples : int
        Number of posterior samples
    scale : float
        Uncertainty scale factor
    
    Returns
    -------
    mean : ndarray
        Mean prediction
    std : ndarray
        Standard deviation (epistemic uncertainty)
    """
    predictions = []
    
    for _ in range(n_samples):
        # Sample weights
        params = swag.sample(scale=scale)
        model.set_params(params)
        
        # Make prediction
        pred = model.predict(X)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    
    # Restore mean weights
    model.set_params(swag.mean)
    
    return mean, std


# =============================================================================
# Demo
# =============================================================================

def demo_swag():
    """Demonstrate SWAG on a simple regression problem."""
    
    print("=" * 60)
    print("SWAG DEMONSTRATION")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create toy data
    N = 100
    X = np.random.uniform(-4, 4, N).reshape(-1, 1)
    y = np.sin(X) + 0.2 * np.random.randn(N, 1)
    
    print(f"\nTraining data: {N} points")
    
    # Simple MLP for demonstration
    class SimpleMLP:
        def __init__(self, layers):
            self.layers = layers
            self.params = self._init_params()
        
        def _init_params(self):
            params = []
            for i in range(len(self.layers) - 1):
                W = np.random.randn(self.layers[i], self.layers[i+1]) * 0.5
                b = np.zeros(self.layers[i+1])
                params.extend([W.flatten(), b])
            return np.concatenate(params)
        
        def get_params(self):
            return self.params.copy()
        
        def set_params(self, params):
            self.params = params.copy()
        
        def predict(self, X):
            # Unpack params and forward pass
            idx = 0
            h = X
            for i in range(len(self.layers) - 1):
                n_in, n_out = self.layers[i], self.layers[i+1]
                W = self.params[idx:idx + n_in * n_out].reshape(n_in, n_out)
                idx += n_in * n_out
                b = self.params[idx:idx + n_out]
                idx += n_out
                h = h @ W + b
                if i < len(self.layers) - 2:
                    h = np.tanh(h)
            return h
        
        def compute_gradients(self, X, y):
            # Numerical gradients for simplicity
            pred = self.predict(X)
            loss = np.mean((pred - y) ** 2)
            
            eps = 1e-5
            grads = np.zeros_like(self.params)
            for i in range(len(self.params)):
                self.params[i] += eps
                loss_plus = np.mean((self.predict(X) - y) ** 2)
                self.params[i] -= 2 * eps
                loss_minus = np.mean((self.predict(X) - y) ** 2)
                self.params[i] += eps
                grads[i] = (loss_plus - loss_minus) / (2 * eps)
            
            return loss, grads
    
    # Create model
    model = SimpleMLP([1, 30, 30, 1])
    n_params = len(model.get_params())
    print(f"Model parameters: {n_params}")
    
    # Create SWAG
    swag = SWAG(n_params, max_rank=20)
    
    # Pre-train
    print("\nPre-training...")
    for epoch in range(50):
        lr = 0.1 * (0.95 ** epoch)
        _, grads = model.compute_gradients(X, y)
        model.set_params(model.get_params() - lr * grads)
    
    # SWAG collection phase
    print("Collecting SWAG statistics...")
    for epoch in range(30):
        # Cyclical LR
        t = (epoch % 5) / 5
        lr = 0.001 + 0.009 * (1 + np.cos(np.pi * t)) / 2
        
        _, grads = model.compute_gradients(X, y)
        model.set_params(model.get_params() - lr * grads)
        
        # Collect at end of cycle
        if (epoch + 1) % 5 == 0:
            swag.update(model.get_params())
    
    print(f"SWAG models collected: {swag.n_models}")
    
    # Make predictions
    X_test = np.linspace(-6, 6, 100).reshape(-1, 1)
    
    mean, std = swag_predict(model, swag, X_test, n_samples=50)
    
    print(f"\nMean epistemic uncertainty: {np.mean(std):.4f}")
    print(f"Uncertainty in training region [-4,4]: {np.mean(std[np.abs(X_test) < 4]):.4f}")
    print(f"Uncertainty outside training region: {np.mean(std[np.abs(X_test) > 4]):.4f}")
    
    print("\n*** SWAG uncertainty should be higher outside training region")
    
    return swag, model


if __name__ == "__main__":
    demo_swag()
```

---

## Comparison with Other Methods

### vs. Deep Ensembles

| Aspect | SWAG | Deep Ensembles |
|--------|------|----------------|
| Training | Single network + statistics | M independent networks |
| Storage | Mean + diagonal + K vectors | M complete networks |
| Diversity | Gaussian approximation | True ensemble diversity |
| Uncertainty quality | Good | Often better |
| Implementation | Post-hoc | From scratch |

### vs. MC Dropout

| Aspect | SWAG | MC Dropout |
|--------|------|------------|
| Training | Modified schedule | Standard + dropout |
| Posterior | Explicit Gaussian | Implicit (Bernoulli masks) |
| Covariance | Low-rank + diagonal | Implicit |
| Flexibility | Any architecture | Requires dropout layers |

### vs. Variational Inference

| Aspect | SWAG | VI (Bayes by Backprop) |
|--------|------|------------------------|
| Training | Post-hoc | From scratch |
| Parameters | Same as base | 2× (mean + variance) |
| Approximation | Empirical Gaussian | Mean-field Gaussian |
| Scalability | Very high | High |

---

## Practical Guidelines

### When to Use SWAG

**Good candidates**:
- Already have trained networks
- Need quick uncertainty estimates
- Large models where VI is expensive
- Standard architectures (ResNets, etc.)

**Less suitable**:
- Need very accurate posteriors
- Multimodal posteriors expected
- Real-time inference requirements

### Implementation Tips

1. **Pre-train normally**: Get a good solution first
2. **Use cyclical LR**: Better exploration than constant
3. **Collect enough samples**: At least 10-20 for stable covariance
4. **Handle BatchNorm**: Update running stats after sampling
5. **Scale factor tuning**: Start with 1.0, adjust based on calibration

### Calibration

If predictions are under/overconfident:
- **Underconfident**: Reduce scale factor
- **Overconfident**: Increase scale factor, collect more samples

---

## Summary

### Key Formulas

**SWAG distribution**:
$$
q(\theta) = \mathcal{N}(\bar{\theta}, \Sigma_{\text{diag}} + \Sigma_{\text{low-rank}})
$$

**Sampling**:
$$
\theta = \bar{\theta} + \frac{1}{\sqrt{2}} \sqrt{\Sigma_{\text{diag}}} \odot z_1 + \frac{1}{\sqrt{2(K-1)}} D z_2
$$

**Running statistics**:
$$
\bar{\theta} = \frac{1}{T}\sum_t \theta_t, \quad \Sigma_{\text{diag}} = \text{diag}(\overline{\theta^2} - \bar{\theta}^2)
$$

### Advantages and Limitations

| Advantages | Limitations |
|------------|-------------|
| Simple to implement | Gaussian approximation only |
| Post-hoc (no retraining) | May underestimate uncertainty |
| Scalable to large networks | Requires SWA-style training phase |
| Good empirical performance | Single-mode approximation |

### Connections to Other Topics

| Topic | Connection |
|-------|------------|
| SWA | SWAG extends SWA with covariance |
| Laplace | Both fit Gaussian at convergence |
| Deep Ensembles | Can combine (Multi-SWAG) |
| Uncertainty | Provides epistemic uncertainty estimates |

### Key References

- Maddox, W., et al. (2019). A simple baseline for Bayesian inference in deep learning. *NeurIPS*.
- Izmailov, P., et al. (2018). Averaging weights leads to wider optima and better generalization. *UAI*.
- Wilson, A. G., & Izmailov, P. (2020). Bayesian deep learning and a probabilistic perspective of generalization. *NeurIPS*.
