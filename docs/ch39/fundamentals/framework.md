# Mathematical Framework

## The Bayesian Framework for Uncertainty

The principled approach to uncertainty quantification in neural networks is Bayesian inference. Instead of learning a point estimate $\hat{\mathbf{w}}$, we maintain a posterior distribution over model parameters:

$$p(\mathbf{w}|\mathcal{D}) = \frac{p(\mathcal{D}|\mathbf{w}) p(\mathbf{w})}{p(\mathcal{D})}$$

where $p(\mathcal{D}|\mathbf{w}) = \prod_{i=1}^N p(y_i|\mathbf{x}_i, \mathbf{w})$ is the likelihood, $p(\mathbf{w})$ is the prior, and $p(\mathcal{D}) = \int p(\mathcal{D}|\mathbf{w}) p(\mathbf{w}) d\mathbf{w}$ is the evidence (marginal likelihood).

### The Predictive Distribution

The posterior predictive distribution integrates predictions over all plausible weight configurations:

$$p(y^*|\mathbf{x}^*, \mathcal{D}) = \int p(y^*|\mathbf{x}^*, \mathbf{w}) p(\mathbf{w}|\mathcal{D}) d\mathbf{w}$$

This integral is intractable for neural networks due to the high dimensionality and nonlinearity of $p(y|\mathbf{x}, \mathbf{w})$. The methods in this chapter—MC Dropout, ensembles, variational inference, SWAG, Laplace approximation—all provide different approximations to this integral.

### Monte Carlo Approximation

Given samples $\{\mathbf{w}^{(s)}\}_{s=1}^S$ from the posterior (or an approximation), we estimate:

**Regression mean**: $\hat{\mu}(\mathbf{x}^*) = \frac{1}{S} \sum_{s=1}^S f_{\mathbf{w}^{(s)}}(\mathbf{x}^*)$

**Regression variance**: $\hat{\sigma}^2(\mathbf{x}^*) = \frac{1}{S} \sum_{s=1}^S [f_{\mathbf{w}^{(s)}}(\mathbf{x}^*) - \hat{\mu}(\mathbf{x}^*)]^2$

**Classification**: $p(y^* = c|\mathbf{x}^*, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^S \text{softmax}(f_{\mathbf{w}^{(s)}}(\mathbf{x}^*))_c$

## The Softmax Function and Overconfidence

### Standard Softmax

The softmax function converts logits $\mathbf{z} = (z_1, \ldots, z_K)$ to probabilities:

$$p_i = \frac{\exp(z_i)}{\sum_{j=1}^K \exp(z_j)}$$

### Why Neural Networks Are Overconfident

Modern neural networks produce overconfident predictions due to several mechanisms:

| Source | Explanation |
|--------|-------------|
| **Cross-entropy loss** | Encourages pushing probabilities toward 0/1 |
| **High capacity** | Deep networks can memorize, producing sharp boundaries |
| **Batch normalization** | Can amplify logit magnitudes |
| **No confidence penalty** | Standard training doesn't penalize overconfidence |
| **ReLU activations** | Produce unbounded logit magnitudes |

Cross-entropy loss for correct class $y$:

$$\mathcal{L} = -\log(p_y) = -\log\left(\frac{\exp(z_y)}{\sum_j \exp(z_j)}\right)$$

To minimize loss, the network increases $z_y$ relative to other logits. This drives the winning logit arbitrarily high, pushing confidence toward 1.0 even when the model is genuinely uncertain.

### Temperature Scaling

Temperature scaling modifies the softmax with parameter $T$:

$$p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^K \exp(z_j / T)}$$

**Effect of temperature**:

- $T = 1$: Standard softmax
- $T > 1$: Softer probabilities → less confident
- $T < 1$: Sharper probabilities → more confident
- $T \to 0$: Approaches argmax (hard assignment)
- $T \to \infty$: Approaches uniform distribution

Temperature scaling is the simplest post-hoc calibration method, covered in detail in Section 38.5.

```python
import torch
import torch.nn.functional as F


def softmax_with_temperature(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Apply softmax with temperature scaling."""
    return F.softmax(logits / temperature, dim=-1)


def demonstrate_temperature_effect():
    """Show how temperature affects probability distributions."""
    logits = torch.tensor([[2.0, 1.0, 0.1]])
    
    temperatures = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    print(f"{'Temperature':<12} {'Class 0':<10} {'Class 1':<10} {'Class 2':<10}")
    print("-" * 45)
    
    for T in temperatures:
        probs = softmax_with_temperature(logits, T)
        p = probs[0]
        print(f"{T:<12.1f} {p[0]:<10.4f} {p[1]:<10.4f} {p[2]:<10.4f}")
```

## Total Uncertainty Decomposition

### Regression: Law of Total Variance

$$\text{Var}[y|\mathbf{x}, \mathcal{D}] = \underbrace{\mathbb{E}_{\mathbf{w}}[\text{Var}[y|\mathbf{x}, \mathbf{w}]]}_{\text{Aleatoric}} + \underbrace{\text{Var}_{\mathbf{w}}[\mathbb{E}[y|\mathbf{x}, \mathbf{w}]]}_{\text{Epistemic}}$$

### Classification: Entropy Decomposition

$$\underbrace{\mathbb{H}[\bar{p}]}_{\text{Total}} = \underbrace{\mathbb{I}[y; \mathbf{w} | \mathbf{x}, \mathcal{D}]}_{\text{Epistemic (MI)}} + \underbrace{\mathbb{E}_{\mathbf{w}}[\mathbb{H}[p_\mathbf{w}]]}_{\text{Aleatoric}}$$

where $\bar{p} = \mathbb{E}_{\mathbf{w}}[p_\mathbf{w}]$ is the mean predictive distribution.

## Entropy as Uncertainty Measure

Predictive entropy provides a scalar summary of total uncertainty:

$$\mathbb{H}[p] = -\sum_{c=1}^C p_c \log p_c$$

Properties:

- $\mathbb{H} = 0$ when prediction is deterministic (one class has probability 1)
- $\mathbb{H} = \log C$ when prediction is uniform (maximum uncertainty)
- Entropy is concave and non-negative

```python
def compute_predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of predicted distribution.
    
    H(p) = -Σ p_i log(p_i)
    """
    epsilon = 1e-10
    return -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
```

## The Calibration Problem

A model is **calibrated** if its predicted probabilities match empirical frequencies:

$$\mathbb{P}(Y = y \mid \hat{p}(Y = y) = p) = p, \quad \forall p \in [0, 1]$$

Modern neural networks are typically **overconfident**: among predictions with 90% confidence, far fewer than 90% are actually correct. The gap between stated confidence and actual accuracy is measured by calibration metrics (Section 38.5-38.6).

### Calibration vs Accuracy

Calibration and accuracy are independent properties:

| | Well-Calibrated | Poorly Calibrated |
|--|----------------|-------------------|
| **High Accuracy** | Ideal: confident and correct | Dangerous: accurate but overconfident |
| **Low Accuracy** | Honest: wrong but admits uncertainty | Worst: wrong and confident |

A well-calibrated low-accuracy model may be more useful than a poorly calibrated high-accuracy model, because the former provides honest uncertainty that enables appropriate decision-making.

## Key Takeaways

!!! success "Summary"
    1. The **Bayesian predictive distribution** is the gold standard for uncertainty
    2. Neural networks are **systematically overconfident** due to cross-entropy training and high capacity
    3. **Temperature scaling** provides simple post-hoc control over confidence levels
    4. Total uncertainty **decomposes exactly** into epistemic and aleatoric components
    5. **Calibration** measures whether stated probabilities match observed frequencies
    6. Good uncertainty requires both **calibration** and **sharpness** (informativeness)

## References

- Guo, C., et al. (2017). "On Calibration of Modern Neural Networks"
- Hinton, G., et al. (2015). "Distilling the Knowledge in a Neural Network"
- Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting Good Probabilities With Supervised Learning"
