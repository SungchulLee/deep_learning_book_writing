# MC Dropout: Dropout as Approximate Bayesian Inference
**Monte Carlo Dropout (MC Dropout)** recasts dropout—a regularization technique—as approximate Bayesian inference. By keeping dropout active at test time and running multiple forward passes, we obtain uncertainty estimates from networks that were trained with standard dropout, without any additional cost or architectural changes.

---

## The Key Insight

### Dropout During Training

Standard dropout randomly zeros activations during training:

$$
\tilde{h}_j = z_j \cdot h_j, \quad z_j \sim \text{Bernoulli}(1-p)
$$

where $p$ is the dropout probability.

### Dropout as Approximate Posterior

Gal & Ghahramani (2016) showed that training with dropout approximately minimizes the KL divergence between:

- An approximate posterior $q(\theta)$ defined by the dropout distribution
- The true posterior $p(\theta \mid \mathcal{D})$

**The variational distribution**:

$$
q(W) = M \cdot \text{diag}(z), \quad z_i \sim \text{Bernoulli}(1-p)
$$

where $M$ is the learned weight matrix and $z$ is a vector of Bernoulli random variables.

### From Regularization to Uncertainty

**Standard inference**: Disable dropout, use scaled weights → Single deterministic prediction

**MC Dropout inference**: Keep dropout active, run $T$ forward passes → Variance across passes gives uncertainty

$$
\boxed{p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{T}\sum_{t=1}^T p(y^* \mid x^*, \hat{W} \cdot \text{diag}(z_t))}
$$

---

## Theoretical Foundation

### Variational Inference Perspective

MC Dropout minimizes the KL divergence to the true posterior through the ELBO:

$$
\mathcal{L} = -\mathbb{E}_{q(\theta)}[\log p(\mathcal{D} \mid \theta)] + \text{KL}(q(\theta) \| p(\theta))
$$

**For dropout**:

- The expected log-likelihood is approximated by standard dropout training loss
- The KL term corresponds to L2 regularization (weight decay)

### The Equivalence

**Dropout training** with cross-entropy loss and L2 regularization is equivalent to **variational inference** with:

- Bernoulli approximate posterior
- Gaussian prior $p(\theta) = \mathcal{N}(0, \sigma^2 I)$

**Prior precision** relates to weight decay:

$$
\lambda = \frac{p \cdot l^2}{2N\tau}
$$

---

## Implementation

```python
"""
MC Dropout: Dropout as Bayesian Approximation
"""

import numpy as np
from typing import Tuple, List


class MCDropoutMLP:
    """MLP with MC Dropout for uncertainty estimation."""
    
    def __init__(self, layer_sizes: List[int], dropout_prob: float = 0.5):
        self.layer_sizes = layer_sizes
        self.dropout_prob = dropout_prob
        
        # Initialize weights
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i+1]
            W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros(n_out)
            self.weights.append(W)
            self.biases.append(b)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with optional dropout."""
        h = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W + b
            if i < len(self.weights) - 1:  # Not last layer
                h = np.maximum(0, h)  # ReLU
                if training:
                    mask = (np.random.rand(*h.shape) > self.dropout_prob)
                    h = h * mask / (1 - self.dropout_prob)
        return h
    
    def predict_with_uncertainty(
        self, x: np.ndarray, n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """MC Dropout prediction with uncertainty."""
        predictions = [self.forward(x, training=True) for _ in range(n_samples)]
        predictions = np.array(predictions)
        
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        return mean, std
```

---

## Uncertainty Decomposition

### For Regression

$$
\text{Var}[y^*] = \underbrace{\frac{1}{T}\sum_{t=1}^T (f_t(x^*) - \bar{f}(x^*))^2}_{\text{Epistemic}} + \underbrace{\sigma^2_{\text{noise}}}_{\text{Aleatoric}}
$$

### For Classification

**Total uncertainty**: $\mathbb{H}[\bar{p}] = -\sum_c \bar{p}_c \log \bar{p}_c$

**Aleatoric**: $\mathbb{E}[\mathbb{H}[p]] = \frac{1}{T}\sum_t \mathbb{H}[p_t]$

**Epistemic (mutual information)**: $\mathbb{I}[y; \theta] = \mathbb{H}[\bar{p}] - \mathbb{E}[\mathbb{H}[p]]$

---

## Practical Considerations

### Number of Samples

| Samples $T$ | Use Case |
|-------------|----------|
| 10-30 | Quick estimates |
| 50-100 | Standard inference |
| 100-1000 | Critical decisions |

### Dropout Probability

- $p = 0.1$-$0.3$: Light dropout
- $p = 0.5$: Standard
- Higher $p$: More uncertainty, may hurt accuracy

---

## Comparison with Other Methods

| Aspect | MC Dropout | Deep Ensembles | Variational BNN |
|--------|------------|----------------|-----------------|
| Training cost | 1 network | M networks | 1 network (modified) |
| Inference cost | T passes | M passes | T passes |
| Implementation | Trivial | Easy | Moderate |
| Uncertainty quality | Good | Often better | Good |

---

## Summary

### Key Formulas

**Posterior approximation**: $q(W) = M \cdot \text{diag}(z)$, $z \sim \text{Bernoulli}(1-p)$

**Predictive distribution**: $p(y^*) \approx \frac{1}{T}\sum_{t=1}^T p(y^* \mid W_t)$

### Advantages and Limitations

| Advantages | Limitations |
|------------|-------------|
| No architecture change | Requires dropout |
| Use existing models | May underestimate uncertainty |
| Simple implementation | Coarse approximation |

### Key References

- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML*.
- Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning? *NeurIPS*.

## Exercises

**Exercise 1.**
For a two-layer neural network with ReLU activations and Gaussian weight priors, derive the form of the approximate posterior under the method described in this section.

??? success "Solution to Exercise 1"
    With weights $W_1, W_2$ and Gaussian prior $p(W) = \mathcal{N}(0, \sigma_p^2 I)$, the posterior $p(W | D) \propto p(D | W) p(W)$ is intractable. The approximation method from this section produces a tractable form: for variational inference, each weight has an independent Gaussian posterior $q(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$; for Laplace approximation, the posterior is a single Gaussian centered at the MAP estimate with covariance equal to the inverse Hessian; for MC Dropout, the posterior is implicitly defined by the dropout mask distribution. Each approximation captures different aspects of the true posterior's shape. $\square$

---

**Exercise 2.**
Design an experiment to compare the calibration of uncertainty estimates from this method against MC Dropout and deep ensembles. Specify the metrics and visualization.

??? success "Solution to Exercise 2"
    Metrics: (1) Expected Calibration Error (ECE) with 15 bins; (2) Brier score; (3) negative log-likelihood (NLL); (4) AUROC for OOD detection. Visualization: reliability diagrams plotting observed frequency vs. predicted confidence for each method. Protocol: train all methods on CIFAR-10 (in-distribution), evaluate calibration on CIFAR-10 test set, and OOD detection on SVHN. Use temperature scaling as a post-hoc baseline. Report means and standard errors over 5 random seeds. A well-calibrated method has points close to the diagonal in the reliability diagram and low ECE. $\square$

---

**Exercise 3.**
Prove that the predictive variance from a Bayesian neural network decomposes into epistemic and aleatoric components. Show how each component behaves as the training set size $N \to \infty$.

??? success "Solution to Exercise 3"
    The predictive variance decomposes via the law of total variance: $\text{Var}[y | x, D] = \underbrace{\mathbb{E}_{p(\theta|D)}[\text{Var}[y | x, \theta]]}_{\text{aleatoric}} + \underbrace{\text{Var}_{p(\theta|D)}[\mathbb{E}[y | x, \theta]]}_{\text{epistemic}}$. The aleatoric component captures irreducible noise in the data-generating process and remains constant as $N \to \infty$. The epistemic component reflects parameter uncertainty, which decreases as $O(1/N)$ because the posterior concentrates around the true parameters. In the limit, only aleatoric uncertainty remains. This decomposition is crucial for deciding when to collect more data (high epistemic) vs. accepting inherent noise (high aleatoric). $\square$

---

**Exercise 4.**
Discuss how the uncertainty quantification method from this section could be used for position sizing in a trading system. Propose a concrete decision rule.

??? success "Solution to Exercise 4"
    Decision rule: the position size is inversely proportional to the epistemic uncertainty. Let $\hat{y}$ be the predicted return and $\sigma_e^2$ be the epistemic variance. The position is $w = \frac{\hat{y}}{\lambda \sigma_e^2}$ where $\lambda$ is a risk aversion parameter. When epistemic uncertainty is high (novel market conditions), positions are reduced; when low (familiar regimes), the system trades with higher conviction. Additionally, set a maximum epistemic uncertainty threshold above which no trade is placed (abstention). This framework naturally implements a Kelly-criterion-like sizing scaled by model confidence. Backtest with walk-forward validation to calibrate $\lambda$. $\square$
