# Laplace Approximation for Neural Networks

The **Laplace approximation** provides a simple, post-hoc approach to Bayesian inference by fitting a Gaussian distribution centered at the MAP (maximum a posteriori) estimate. This classical technique, when applied to neural networks, offers a computationally tractable way to obtain uncertainty estimates without retraining the network.

---

## Motivation: Post-Hoc Uncertainty

### The Appeal of Post-Hoc Methods

You've trained a neural network using standard methods (SGD, Adam). Now you want uncertainty estimates. Options:

1. **Retrain with Bayesian method** (VI, MCMC) — expensive, requires new training
2. **Use dropout at test time** (MC Dropout) — cheap, but requires dropout in architecture
3. **Train ensemble** — requires training $M$ networks
4. **Laplace approximation** — use your trained network, fit Gaussian around it

The Laplace approximation is **post-hoc**: it takes an already-trained network and adds uncertainty quantification.

### The Core Idea

Approximate the posterior with a Gaussian centered at the mode (MAP estimate):

$$
\boxed{p(\theta \mid \mathcal{D}) \approx q(\theta) = \mathcal{N}(\theta \mid \hat{\theta}_{\text{MAP}}, \Sigma)}
$$

where:
- $\hat{\theta}_{\text{MAP}}$ is your trained network's weights
- $\Sigma = H^{-1}$ is the inverse Hessian of the loss at the MAP

---

## Mathematical Foundation

### Derivation

Start with the unnormalized log posterior:

$$
\log p(\theta \mid \mathcal{D}) = \log p(\mathcal{D} \mid \theta) + \log p(\theta) - \log p(\mathcal{D})
$$

Taylor expand around the MAP estimate $\hat{\theta}$:

$$
\log p(\theta \mid \mathcal{D}) \approx \log p(\hat{\theta} \mid \mathcal{D}) + \underbrace{\nabla \log p(\hat{\theta} \mid \mathcal{D})}_{= 0 \text{ at MAP}} (\theta - \hat{\theta}) - \frac{1}{2}(\theta - \hat{\theta})^\top H (\theta - \hat{\theta})
$$

where:

$$
H = -\nabla^2_\theta \log p(\theta \mid \mathcal{D})\big|_{\hat{\theta}}
$$

is the Hessian of the negative log posterior (positive definite at a local minimum).

Exponentiating gives:

$$
p(\theta \mid \mathcal{D}) \propto \exp\left(-\frac{1}{2}(\theta - \hat{\theta})^\top H (\theta - \hat{\theta})\right)
$$

This is a Gaussian with covariance $\Sigma = H^{-1}$.

### The Hessian

For a neural network with negative log-likelihood loss $\mathcal{L}(\theta) = -\log p(\mathcal{D} \mid \theta)$ and Gaussian prior:

$$
H = \nabla^2 \mathcal{L}(\theta)\big|_{\hat{\theta}} + \frac{1}{\sigma_0^2} I
$$

**Components**:
- **Likelihood Hessian**: Curvature of the loss surface
- **Prior contribution**: Regularization (corresponds to weight decay $\lambda = 1/\sigma_0^2$)

### Generalized Gauss-Newton (GGN)

The exact Hessian is expensive and may not be positive definite. The **GGN approximation** is commonly used:

$$
H_{\text{GGN}} = J^\top \nabla^2 \mathcal{L}_{\text{out}} J + \frac{1}{\sigma_0^2} I
$$

where:
- $J = \nabla_\theta f_\theta(X)$ is the Jacobian of network outputs w.r.t. parameters
- $\nabla^2 \mathcal{L}_{\text{out}}$ is the Hessian of the loss w.r.t. network outputs

**For MSE loss**: GGN equals the Fisher information matrix.

**Advantages**: Always positive semi-definite, often a good approximation.

---

## Hessian Approximations

### The Scalability Challenge

For a network with $d$ parameters:
- **Full Hessian**: $O(d^2)$ storage, $O(d^3)$ inversion
- Modern networks: $d = 10^6$ to $10^9$

**Solution**: Use structured approximations.

### Diagonal Approximation

Assume independence between parameters:

$$
\Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2) \quad \text{where} \quad \sigma_i^2 = 1/H_{ii}
$$

**Computation**: Only need diagonal of Hessian — can be estimated efficiently.

**Limitation**: Ignores correlations between parameters.

### Kronecker-Factored Approximate Curvature (KFAC)

For layer $l$ with weight matrix $W^{(l)} \in \mathbb{R}^{n_{l-1} \times n_l}$:

$$
\boxed{H^{(l)} \approx A^{(l)} \otimes G^{(l)}}
$$

where:
- $A^{(l)} = \frac{1}{N}\sum_{n=1}^N a_n^{(l-1)} (a_n^{(l-1)})^\top$ — input activation covariance
- $G^{(l)} = \frac{1}{N}\sum_{n=1}^N g_n^{(l)} (g_n^{(l)})^\top$ — output gradient covariance

**Key property**: Kronecker product inversion is efficient:

$$
(A \otimes G)^{-1} = A^{-1} \otimes G^{-1}
$$

**Complexity**: $O(n_l^3 + n_{l-1}^3)$ per layer instead of $O((n_l \cdot n_{l-1})^3)$.

### Low-Rank Approximation

Keep only top eigencomponents of the Hessian:

$$
H \approx V \Lambda V^\top
$$

where $V \in \mathbb{R}^{d \times r}$ contains top $r$ eigenvectors.

**Inversion**:

$$
\Sigma = V \Lambda^{-1} V^\top + \frac{1}{\lambda_{\min}} (I - VV^\top)
$$

### Comparison of Approximations

| Approximation | Storage | Inversion | Captures Correlations |
|---------------|---------|-----------|----------------------|
| Full | $O(d^2)$ | $O(d^3)$ | Yes (all) |
| Diagonal | $O(d)$ | $O(d)$ | No |
| KFAC | $O(\sum_l n_l^2)$ | $O(\sum_l n_l^3)$ | Within layers |
| Low-rank | $O(dr)$ | $O(dr^2)$ | Top $r$ directions |

---

## Last-Layer Laplace

### Motivation

A practical simplification: apply Laplace only to the last layer:

$$
p(\theta_L \mid \mathcal{D}, \theta_{1:L-1}) \approx \mathcal{N}(\theta_L \mid \hat{\theta}_L, \Sigma_L)
$$

**Rationale**:
- Earlier layers learn features (representation)
- Last layer does final classification/regression
- Most prediction uncertainty comes from last layer

### Implementation

1. Train network normally
2. Freeze all layers except last
3. Compute Hessian only for last-layer weights
4. Much smaller: $d_L = n_{L-1} \times n_L + n_L$ parameters

### Predictive Distribution

For the last layer $f_\theta(x) = W^\top \phi(x) + b$ where $\phi(x)$ is the feature extractor:

**Linearization**:

$$
f_\theta(x) \approx f_{\hat{\theta}}(x) + \phi(x)^\top (\theta - \hat{\theta})
$$

**Predictive mean**:

$$
\mathbb{E}[f(x^*)] = f_{\hat{\theta}}(x^*)
$$

**Predictive variance**:

$$
\text{Var}[f(x^*)] = \phi(x^*)^\top \Sigma_L \phi(x^*)
$$

---

## Predictive Inference

### For Regression

With Gaussian likelihood $p(y \mid f, \sigma^2)$:

**Predictive distribution**:

$$
p(y^* \mid x^*, \mathcal{D}) = \mathcal{N}(y^* \mid \mu(x^*), \sigma^2(x^*))
$$

**Mean**:

$$
\mu(x^*) = f_{\hat{\theta}}(x^*)
$$

**Variance** (with linearization):

$$
\sigma^2(x^*) = \underbrace{J(x^*)^\top \Sigma \, J(x^*)}_{\text{epistemic}} + \underbrace{\sigma^2_{\text{noise}}}_{\text{aleatoric}}
$$

where $J(x^*) = \nabla_\theta f_\theta(x^*)|_{\hat{\theta}}$ is the Jacobian.

### For Classification

**Approach 1: Probit approximation**

For binary classification with sigmoid output:

$$
p(y=1 \mid x^*) \approx \sigma\left(\frac{\mu(x^*)}{\sqrt{1 + \pi \sigma^2(x^*)/8}}\right)
$$

**Approach 2: Monte Carlo sampling**

Sample $\theta^{(s)} \sim \mathcal{N}(\hat{\theta}, \Sigma)$ and average:

$$
p(y = c \mid x^*) \approx \frac{1}{S} \sum_{s=1}^S \text{softmax}(f_{\theta^{(s)}}(x^*))_c
$$

---

## Python Implementation

```python
"""
Laplace Approximation for Neural Networks

Post-hoc uncertainty quantification by fitting a Gaussian 
at the MAP estimate.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Callable
from scipy.linalg import cho_factor, cho_solve


class LaplaceMLP:
    """
    MLP with Laplace approximation for uncertainty.
    """
    
    def __init__(
        self,
        layer_sizes: list,
        activation: str = 'tanh',
        prior_precision: float = 1.0
    ):
        """
        Parameters
        ----------
        layer_sizes : list
            Sizes of each layer [input, hidden..., output]
        activation : str
            Activation function ('tanh' or 'relu')
        prior_precision : float
            Prior precision (1/sigma_0^2), equivalent to weight decay
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.prior_precision = prior_precision
        
        # Initialize weights
        self.params = self._init_params()
        
        # Laplace quantities (computed after fitting)
        self.hessian = None
        self.covariance = None
    
    def _init_params(self) -> np.ndarray:
        """Initialize weights with Xavier initialization."""
        params = []
        for i in range(len(self.layer_sizes) - 1):
            n_in, n_out = self.layer_sizes[i], self.layer_sizes[i+1]
            W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out))
            b = np.zeros(n_out)
            params.extend([W.flatten(), b])
        return np.concatenate(params)
    
    def _unpack_params(self, params: np.ndarray):
        """Unpack flat parameter vector into weight matrices."""
        weights = []
        idx = 0
        for i in range(len(self.layer_sizes) - 1):
            n_in, n_out = self.layer_sizes[i], self.layer_sizes[i+1]
            W = params[idx:idx + n_in * n_out].reshape(n_in, n_out)
            idx += n_in * n_out
            b = params[idx:idx + n_out]
            idx += n_out
            weights.append((W, b))
        return weights
    
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _activate_grad(self, x: np.ndarray) -> np.ndarray:
        """Derivative of activation function."""
        if self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def forward(self, X: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass.
        
        Parameters
        ----------
        X : ndarray of shape (N, input_dim)
            Input data
        params : ndarray, optional
            Parameters to use (default: self.params)
        
        Returns
        -------
        ndarray of shape (N, output_dim)
            Network output
        """
        if params is None:
            params = self.params
        
        weights = self._unpack_params(params)
        h = X
        
        for i, (W, b) in enumerate(weights):
            h = h @ W + b
            if i < len(weights) - 1:
                h = self._activate(h)
        
        return h
    
    def loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute negative log posterior (up to constant).
        
        Loss = MSE + (prior_precision/2) * ||params||^2
        """
        if params is None:
            params = self.params
        
        pred = self.forward(X, params)
        mse = np.mean((pred - y) ** 2)
        reg = 0.5 * self.prior_precision * np.sum(params ** 2)
        
        return mse + reg / len(X)
    
    def _compute_jacobian(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of network output w.r.t. parameters.
        
        Returns
        -------
        ndarray of shape (N * output_dim, n_params)
        """
        N = len(X)
        n_params = len(self.params)
        output_dim = self.layer_sizes[-1]
        
        # Numerical Jacobian (for simplicity)
        eps = 1e-5
        J = np.zeros((N * output_dim, n_params))
        
        f0 = self.forward(X).flatten()
        
        for i in range(n_params):
            params_plus = self.params.copy()
            params_plus[i] += eps
            f_plus = self.forward(X, params_plus).flatten()
            J[:, i] = (f_plus - f0) / eps
        
        return J
    
    def compute_hessian_diagonal(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute diagonal of the Hessian.
        
        Uses numerical differentiation for simplicity.
        """
        n_params = len(self.params)
        diag_H = np.zeros(n_params)
        
        eps = 1e-4
        
        for i in range(n_params):
            params_plus = self.params.copy()
            params_plus[i] += eps
            
            params_minus = self.params.copy()
            params_minus[i] -= eps
            
            loss_plus = self.loss(X, y, params_plus)
            loss_minus = self.loss(X, y, params_minus)
            loss_center = self.loss(X, y)
            
            diag_H[i] = (loss_plus - 2 * loss_center + loss_minus) / (eps ** 2)
        
        # Add prior contribution
        diag_H += self.prior_precision / len(X)
        
        return np.maximum(diag_H, 1e-6)  # Ensure positive
    
    def compute_ggn(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute Generalized Gauss-Newton approximation to Hessian.
        
        H_GGN = J^T J / (N * sigma^2) + prior_precision * I
        """
        N = len(X)
        J = self._compute_jacobian(X)
        
        # Estimate noise variance from residuals
        residuals = self.forward(X) - y
        sigma2 = np.var(residuals) + 1e-6
        
        # GGN = J^T J / (N * sigma^2)
        H = J.T @ J / (N * sigma2)
        
        # Add prior
        H += self.prior_precision * np.eye(len(self.params))
        
        return H
    
    def fit_laplace(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hessian_type: str = 'diagonal'
    ):
        """
        Fit Laplace approximation after training.
        
        Parameters
        ----------
        X : ndarray
            Training inputs
        y : ndarray
            Training targets
        hessian_type : str
            'diagonal' or 'full' (GGN)
        """
        if hessian_type == 'diagonal':
            diag_H = self.compute_hessian_diagonal(X, y)
            self.hessian = np.diag(diag_H)
            self.covariance = np.diag(1.0 / diag_H)
        
        elif hessian_type == 'full':
            self.hessian = self.compute_ggn(X, y)
            # Regularize and invert
            self.hessian += 1e-4 * np.eye(len(self.params))
            self.covariance = np.linalg.inv(self.hessian)
        
        else:
            raise ValueError(f"Unknown hessian_type: {hessian_type}")
    
    def predict(
        self,
        X: np.ndarray,
        n_samples: int = 0,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with uncertainty.
        
        Parameters
        ----------
        X : ndarray
            Test inputs
        n_samples : int
            If > 0, use MC sampling; else use linearization
        return_std : bool
            Whether to return standard deviation
        
        Returns
        -------
        mean : ndarray
            Predictive mean
        std : ndarray (optional)
            Predictive standard deviation
        """
        mean = self.forward(X)
        
        if not return_std:
            return mean, None
        
        if self.covariance is None:
            raise ValueError("Must call fit_laplace() first")
        
        if n_samples > 0:
            # Monte Carlo sampling
            predictions = []
            for _ in range(n_samples):
                # Sample parameters
                params = np.random.multivariate_normal(
                    self.params, self.covariance
                )
                pred = self.forward(X, params)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            mean = np.mean(predictions, axis=0)
            std = np.std(predictions, axis=0)
        
        else:
            # Linearization
            J = self._compute_jacobian(X)
            
            # Variance = J @ Sigma @ J^T
            # For each output, compute J_i @ Sigma @ J_i^T
            var = np.sum(J @ self.covariance * J, axis=1)
            var = var.reshape(X.shape[0], -1)
            std = np.sqrt(np.maximum(var, 1e-10))
        
        return mean, std


class LastLayerLaplace:
    """
    Laplace approximation applied only to the last layer.
    """
    
    def __init__(
        self,
        feature_extractor: Callable,
        n_features: int,
        n_outputs: int,
        prior_precision: float = 1.0
    ):
        """
        Parameters
        ----------
        feature_extractor : Callable
            Function that maps X -> features
        n_features : int
            Dimension of feature space
        n_outputs : int
            Number of outputs
        prior_precision : float
            Prior precision
        """
        self.feature_extractor = feature_extractor
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.prior_precision = prior_precision
        
        # Last layer parameters: W (n_features x n_outputs) + b (n_outputs)
        self.W = np.zeros((n_features, n_outputs))
        self.b = np.zeros(n_outputs)
        
        # Laplace covariance
        self.precision = None
        self.covariance = None
    
    @property
    def n_params(self) -> int:
        return self.n_features * self.n_outputs + self.n_outputs
    
    def _pack_params(self) -> np.ndarray:
        return np.concatenate([self.W.flatten(), self.b])
    
    def _unpack_params(self, params: np.ndarray):
        W = params[:self.n_features * self.n_outputs].reshape(
            self.n_features, self.n_outputs
        )
        b = params[self.n_features * self.n_outputs:]
        return W, b
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute features and apply last layer."""
        phi = self.feature_extractor(X)
        return phi @ self.W + self.b
    
    def fit(self, X: np.ndarray, y: np.ndarray, n_iter: int = 100, lr: float = 0.01):
        """Fit last layer weights."""
        phi = self.feature_extractor(X)
        
        for _ in range(n_iter):
            # Gradient of MSE
            pred = phi @ self.W + self.b
            error = pred - y
            
            grad_W = phi.T @ error / len(X) + self.prior_precision * self.W / len(X)
            grad_b = np.mean(error, axis=0)
            
            self.W -= lr * grad_W
            self.b -= lr * grad_b
    
    def fit_laplace(self, X: np.ndarray, y: np.ndarray):
        """Compute Laplace approximation for last layer."""
        N = len(X)
        phi = self.feature_extractor(X)  # (N, n_features)
        
        # Estimate noise variance
        pred = phi @ self.W + self.b
        sigma2 = np.var(y - pred) + 1e-6
        
        # For last layer, Hessian has nice form
        # H_W = Phi^T Phi / (N * sigma^2) + prior_precision * I
        # For simplicity, compute full Hessian
        
        # Augment features with ones for bias
        phi_aug = np.column_stack([phi, np.ones(N)])  # (N, n_features + 1)
        
        # Hessian per output (assuming independent outputs)
        H_single = phi_aug.T @ phi_aug / (N * sigma2)
        H_single += self.prior_precision * np.eye(self.n_features + 1)
        
        # Store (same for all outputs in this simple case)
        self.precision = H_single
        self.covariance = np.linalg.inv(H_single)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty."""
        N = len(X)
        phi = self.feature_extractor(X)
        
        # Mean prediction
        mean = phi @ self.W + self.b
        
        # Variance via linearization
        phi_aug = np.column_stack([phi, np.ones(N)])
        
        # var[i] = phi_aug[i] @ Sigma @ phi_aug[i]
        var = np.sum(phi_aug @ self.covariance * phi_aug, axis=1)
        var = np.maximum(var, 1e-10)
        
        # Tile for each output
        std = np.sqrt(var)[:, np.newaxis] * np.ones((1, self.n_outputs))
        
        return mean, std


# =============================================================================
# Demo
# =============================================================================

def demo_laplace():
    """Demonstrate Laplace approximation."""
    
    print("=" * 60)
    print("LAPLACE APPROXIMATION DEMO")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create toy data
    N = 50
    X_train = np.random.uniform(-4, 4, N).reshape(-1, 1)
    y_train = np.sin(X_train) + 0.2 * np.random.randn(N, 1)
    
    X_test = np.linspace(-6, 6, 100).reshape(-1, 1)
    
    print(f"\nTraining data: {N} points")
    
    # Create and train MLP
    model = LaplaceMLP([1, 30, 30, 1], activation='tanh', prior_precision=0.1)
    
    # Train with gradient descent
    print("Training network...")
    lr = 0.1
    for epoch in range(200):
        # Compute gradient numerically
        grads = np.zeros_like(model.params)
        eps = 1e-5
        for i in range(len(model.params)):
            model.params[i] += eps
            loss_plus = model.loss(X_train, y_train)
            model.params[i] -= 2 * eps
            loss_minus = model.loss(X_train, y_train)
            model.params[i] += eps
            grads[i] = (loss_plus - loss_minus) / (2 * eps)
        
        model.params -= lr * grads
        lr *= 0.995  # Decay
    
    final_loss = model.loss(X_train, y_train)
    print(f"Final training loss: {final_loss:.4f}")
    
    # Fit Laplace approximation
    print("\nFitting Laplace approximation (diagonal)...")
    model.fit_laplace(X_train, y_train, hessian_type='diagonal')
    
    # Make predictions
    mean, std = model.predict(X_test, n_samples=0)
    
    print(f"\nUncertainty statistics:")
    print(f"  Mean std in training region [-4,4]: {np.mean(std[np.abs(X_test) < 4]):.4f}")
    print(f"  Mean std outside training region: {np.mean(std[np.abs(X_test) > 4]):.4f}")
    
    # Compare with MC sampling
    print("\nComparing linearization vs MC sampling...")
    mean_mc, std_mc = model.predict(X_test, n_samples=100)
    
    print(f"  Correlation of means: {np.corrcoef(mean.flatten(), mean_mc.flatten())[0,1]:.4f}")
    print(f"  Correlation of stds: {np.corrcoef(std.flatten(), std_mc.flatten())[0,1]:.4f}")
    
    print("\n*** Uncertainty should be higher outside training region")
    
    return model


def demo_last_layer_laplace():
    """Demonstrate last-layer Laplace."""
    
    print("\n" + "=" * 60)
    print("LAST-LAYER LAPLACE DEMO")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create data
    N = 100
    X_train = np.random.uniform(-4, 4, N).reshape(-1, 1)
    y_train = np.sin(X_train) + 0.2 * np.random.randn(N, 1)
    
    # Simple feature extractor (fixed random features)
    n_features = 50
    W_feat = np.random.randn(1, n_features) * 0.5
    b_feat = np.random.randn(n_features)
    
    def feature_extractor(X):
        return np.tanh(X @ W_feat + b_feat)
    
    # Create and fit
    model = LastLayerLaplace(
        feature_extractor=feature_extractor,
        n_features=n_features,
        n_outputs=1,
        prior_precision=1.0
    )
    
    model.fit(X_train, y_train, n_iter=100, lr=0.1)
    model.fit_laplace(X_train, y_train)
    
    # Predict
    X_test = np.linspace(-6, 6, 100).reshape(-1, 1)
    mean, std = model.predict(X_test)
    
    print(f"\nLast-layer Laplace (only {model.n_params} parameters)")
    print(f"  Mean std in training region: {np.mean(std[np.abs(X_test) < 4]):.4f}")
    print(f"  Mean std outside: {np.mean(std[np.abs(X_test) > 4]):.4f}")
    
    return model


if __name__ == "__main__":
    demo_laplace()
    demo_last_layer_laplace()
```

---

## Practical Considerations

### Choosing the Hessian Approximation

| Method | When to Use |
|--------|-------------|
| **Diagonal** | Large networks, quick estimate, rough uncertainty |
| **KFAC** | Medium networks, better uncertainty, structured |
| **Last-layer** | Pre-trained networks, interpretable features |
| **Full GGN** | Small networks, best approximation |

### Hyperparameter Tuning

**Prior precision** ($\lambda = 1/\sigma_0^2$):
- Corresponds to weight decay used in training
- Higher → more regularization, tighter posterior
- Cross-validate on held-out data

**Temperature scaling**:
If uncertainty is miscalibrated, scale the covariance:

$$
\Sigma_{\text{scaled}} = T \cdot \Sigma
$$

where $T > 1$ increases uncertainty, $T < 1$ decreases it.

### Computational Tips

1. **Precompute Jacobian**: For fixed test points, compute $J(x^*)$ once
2. **Use sparse/structured Hessian**: KFAC or block-diagonal
3. **GPU acceleration**: Matrix operations parallelize well
4. **Incremental updates**: For online learning, update Hessian incrementally

---

## Summary

### Key Formulas

**Laplace posterior**:
$$
q(\theta) = \mathcal{N}(\hat{\theta}_{\text{MAP}}, H^{-1})
$$

**Predictive variance** (linearized):
$$
\text{Var}[f(x^*)] = J(x^*)^\top \Sigma J(x^*)
$$

**GGN approximation**:
$$
H_{\text{GGN}} = J^\top \nabla^2 \mathcal{L}_{\text{out}} J + \lambda I
$$

### Advantages and Limitations

| Advantages | Limitations |
|------------|-------------|
| Post-hoc (no retraining) | Gaussian assumption |
| Principled foundation | Local approximation (single mode) |
| Works with any architecture | Hessian computation can be expensive |
| Connects to classic statistics | May underestimate uncertainty |

### Connections to Other Methods

| Method | Relationship |
|--------|--------------|
| **SWAG** | Also fits Gaussian, but from trajectory |
| **Variational** | Laplace is a special case of VI |
| **Ensembles** | Can combine with Laplace |
| **Fisher information** | GGN ≈ Fisher for certain losses |

### Key References

- MacKay, D. J. (1992). A practical Bayesian framework for backpropagation networks. *Neural Computation*.
- Ritter, H., et al. (2018). A scalable Laplace approximation for neural networks. *ICLR*.
- Daxberger, E., et al. (2021). Laplace redux — Effortless Bayesian deep learning. *NeurIPS*.
- Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored approximate curvature. *ICML*.
