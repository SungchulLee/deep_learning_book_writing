# Uncertainty Quantification in Neural Networks

**Uncertainty quantification** addresses a critical limitation of standard neural networks: they produce point predictions without any measure of confidence. Bayesian neural networks provide a principled framework for quantifying both what the model doesn't know about the data (aleatoric uncertainty) and what the model doesn't know about itself (epistemic uncertainty).

---

## Motivation: Why Uncertainty Matters

### The Overconfidence Problem

Standard neural networks trained with maximum likelihood produce point estimates:

$$
\hat{y} = f_{\hat{\theta}}(x)
$$

**Critical issues**:

1. **No confidence measure**: The network outputs a single prediction with no indication of reliability

2. **Overconfident extrapolation**: Networks often make confident predictions far from training data

3. **Silent failures**: When inputs are out-of-distribution, the model fails without warning

4. **Miscalibrated probabilities**: Softmax outputs don't reflect true predictive uncertainty

### Real-World Consequences

**Medical diagnosis**: A model predicts "95% probability of benign tumor" — but is this because:
- The model has seen many similar cases (low uncertainty)?
- The model is extrapolating wildly (high uncertainty)?

**Autonomous driving**: A perception system must know when it's uncertain to trigger human intervention.

**Scientific discovery**: Uncertainty guides where to collect more data (active learning).

### What Bayesian Methods Provide

The Bayesian approach maintains a **posterior distribution** over model parameters:

$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{p(\mathcal{D})}
$$

This enables:
1. **Predictive distributions** instead of point predictions
2. **Decomposition** into aleatoric and epistemic components
3. **Principled propagation** of uncertainty through the model
4. **Automatic calibration** under model assumptions

---

## Types of Uncertainty

### Taxonomy Overview

$$
\boxed{\text{Total Uncertainty} = \text{Aleatoric Uncertainty} + \text{Epistemic Uncertainty}}
$$

| Type | Also Called | Source | Reducible? |
|------|-------------|--------|------------|
| **Aleatoric** | Data uncertainty | Inherent noise in observations | No |
| **Epistemic** | Model uncertainty | Limited data, model ignorance | Yes (with more data) |

### Aleatoric Uncertainty

**Definition**: Uncertainty inherent in the data-generating process that cannot be reduced by collecting more data.

**Sources**:
- Measurement noise
- Stochastic processes
- Incomplete observations (hidden variables)
- Class overlap in classification

**Mathematical formulation** (regression):

$$
y = f(x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2(x))
$$

The noise variance $\sigma^2(x)$ can be:
- **Homoscedastic**: Constant across input space
- **Heteroscedastic**: Varies with input $x$

**Example**: Predicting stock prices
- Even with perfect knowledge of all relevant factors, prices have inherent randomness
- More data won't eliminate this fundamental unpredictability

### Epistemic Uncertainty

**Definition**: Uncertainty arising from limited knowledge about the model, which can be reduced by collecting more data.

**Sources**:
- Limited training data
- Model misspecification
- Parameter uncertainty
- Out-of-distribution inputs

**Mathematical formulation**:

Epistemic uncertainty is captured by the posterior distribution $p(\theta \mid \mathcal{D})$:
- Narrow posterior → Low epistemic uncertainty (confident about parameters)
- Wide posterior → High epistemic uncertainty (uncertain about parameters)

**Example**: Predicting house prices in a new neighborhood
- With few observations, we're uncertain about the price-feature relationship
- More data from this neighborhood would reduce our uncertainty

### Homoscedastic vs Heteroscedastic Noise

**Homoscedastic** (constant noise):

$$
p(y \mid x, \theta) = \mathcal{N}(y \mid f_\theta(x), \sigma^2)
$$

- Single noise parameter $\sigma^2$ for all inputs
- Simpler to model
- Often unrealistic

**Heteroscedastic** (input-dependent noise):

$$
p(y \mid x, \theta) = \mathcal{N}(y \mid f_\theta(x), \sigma^2_\theta(x))
$$

- Network predicts both mean and variance
- More flexible and realistic
- Requires careful training

---

## The Predictive Distribution

### Definition

The **posterior predictive distribution** integrates over parameter uncertainty:

$$
\boxed{p(y^* \mid x^*, \mathcal{D}) = \int p(y^* \mid x^*, \theta) \, p(\theta \mid \mathcal{D}) \, d\theta}
$$

where:
- $x^*$ is the test input
- $y^*$ is the predicted output
- $p(y^* \mid x^*, \theta)$ is the likelihood (model prediction given parameters)
- $p(\theta \mid \mathcal{D})$ is the posterior over parameters

### Monte Carlo Approximation

Since the integral is intractable, approximate using samples $\{\theta^{(s)}\}_{s=1}^S$ from the posterior:

$$
p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^S p(y^* \mid x^*, \theta^{(s)})
$$

**For regression** (Gaussian likelihood):

$$
\hat{\mu}(x^*) = \frac{1}{S} \sum_{s=1}^S f_{\theta^{(s)}}(x^*)
$$

$$
\hat{\sigma}^2(x^*) = \frac{1}{S} \sum_{s=1}^S \left[ f_{\theta^{(s)}}(x^*) - \hat{\mu}(x^*) \right]^2 + \frac{1}{S} \sum_{s=1}^S \sigma^2_{\theta^{(s)}}(x^*)
$$

**For classification**:

$$
p(y^* = c \mid x^*, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^S \text{softmax}(f_{\theta^{(s)}}(x^*))_c
$$

### Predictive Mean and Variance

For regression with heteroscedastic noise, the predictive distribution has:

**Predictive mean**:

$$
\mathbb{E}[y^* \mid x^*, \mathcal{D}] = \mathbb{E}_{\theta \mid \mathcal{D}}[\mu_\theta(x^*)]
$$

**Predictive variance** (law of total variance):

$$
\text{Var}[y^* \mid x^*, \mathcal{D}] = \underbrace{\mathbb{E}_{\theta \mid \mathcal{D}}[\sigma^2_\theta(x^*)]}_{\text{Aleatoric}} + \underbrace{\text{Var}_{\theta \mid \mathcal{D}}[\mu_\theta(x^*)]}_{\text{Epistemic}}
$$

This decomposition is fundamental for understanding the sources of uncertainty.

---

## Uncertainty Decomposition

### Law of Total Variance

The decomposition follows from the **law of total variance**:

$$
\text{Var}[Y] = \mathbb{E}[\text{Var}[Y \mid X]] + \text{Var}[\mathbb{E}[Y \mid X]]
$$

Applied to our setting with $Y = y^*$ and $X = \theta$:

$$
\text{Var}[y^* \mid x^*, \mathcal{D}] = \mathbb{E}_\theta[\text{Var}[y^* \mid x^*, \theta]] + \text{Var}_\theta[\mathbb{E}[y^* \mid x^*, \theta]]
$$

### Aleatoric Uncertainty

$$
\boxed{\text{Aleatoric}(x^*) = \mathbb{E}_{\theta \mid \mathcal{D}}[\sigma^2_\theta(x^*)]}
$$

**Interpretation**: Expected observation noise, averaged over parameter uncertainty

**Estimation**:

$$
\widehat{\text{Aleatoric}}(x^*) = \frac{1}{S} \sum_{s=1}^S \sigma^2_{\theta^{(s)}}(x^*)
$$

**Properties**:
- Irreducible: doesn't decrease with more data
- Data-dependent: varies across input space
- Captures inherent noise in the problem

### Epistemic Uncertainty

$$
\boxed{\text{Epistemic}(x^*) = \text{Var}_{\theta \mid \mathcal{D}}[\mu_\theta(x^*)]}
$$

**Interpretation**: Variance in predictions due to parameter uncertainty

**Estimation**:

$$
\widehat{\text{Epistemic}}(x^*) = \frac{1}{S} \sum_{s=1}^S \left[\mu_{\theta^{(s)}}(x^*) - \bar{\mu}(x^*)\right]^2
$$

where $\bar{\mu}(x^*) = \frac{1}{S} \sum_s \mu_{\theta^{(s)}}(x^*)$.

**Properties**:
- Reducible: decreases with more data
- High in regions far from training data
- Captures model ignorance

### Total Predictive Uncertainty

$$
\boxed{\text{Total}(x^*) = \text{Aleatoric}(x^*) + \text{Epistemic}(x^*)}
$$

**Estimation**:

$$
\widehat{\text{Total}}(x^*) = \frac{1}{S} \sum_{s=1}^S \sigma^2_{\theta^{(s)}}(x^*) + \frac{1}{S} \sum_{s=1}^S \left[\mu_{\theta^{(s)}}(x^*) - \bar{\mu}(x^*)\right]^2
$$

---

## Uncertainty in Classification

### Predictive Entropy

For classification, the predictive distribution is categorical:

$$
p(y^* = c \mid x^*, \mathcal{D}) = \bar{p}_c = \frac{1}{S} \sum_{s=1}^S p(y^* = c \mid x^*, \theta^{(s)})
$$

**Total uncertainty** via entropy:

$$
\boxed{\mathbb{H}[y^* \mid x^*, \mathcal{D}] = -\sum_{c=1}^C \bar{p}_c \log \bar{p}_c}
$$

### Mutual Information Decomposition

The total entropy decomposes into:

$$
\underbrace{\mathbb{H}[y^* \mid x^*, \mathcal{D}]}_{\text{Total}} = \underbrace{\mathbb{I}[y^*; \theta \mid x^*, \mathcal{D}]}_{\text{Epistemic (MI)}} + \underbrace{\mathbb{E}_{\theta \mid \mathcal{D}}[\mathbb{H}[y^* \mid x^*, \theta]]}_{\text{Aleatoric}}
$$

**Aleatoric uncertainty** (expected entropy):

$$
\text{Aleatoric}(x^*) = \mathbb{E}_{\theta \mid \mathcal{D}}[\mathbb{H}[y^* \mid x^*, \theta]] = -\frac{1}{S} \sum_{s=1}^S \sum_{c=1}^C p_{c,s} \log p_{c,s}
$$

where $p_{c,s} = p(y^* = c \mid x^*, \theta^{(s)})$.

**Epistemic uncertainty** (mutual information):

$$
\text{Epistemic}(x^*) = \mathbb{H}[y^* \mid x^*, \mathcal{D}] - \mathbb{E}_{\theta \mid \mathcal{D}}[\mathbb{H}[y^* \mid x^*, \theta]]
$$

### BALD: Bayesian Active Learning by Disagreement

The mutual information (epistemic uncertainty) is used in **BALD** for active learning:

$$
\text{BALD}(x^*) = \mathbb{I}[y^*; \theta \mid x^*, \mathcal{D}] = \mathbb{H}[\bar{p}] - \frac{1}{S} \sum_{s=1}^S \mathbb{H}[p_s]
$$

**Interpretation**: 
- High when different parameter samples disagree
- Points with high BALD are informative for learning

**Selection criterion**:

$$
x^*_{\text{next}} = \arg\max_{x \in \mathcal{X}_{\text{pool}}} \text{BALD}(x)
$$

---

## Calibration and Reliability

### What is Calibration?

A model is **well-calibrated** if predicted probabilities match empirical frequencies:

$$
P(y = 1 \mid p(y = 1 \mid x) = q) = q \quad \forall q \in [0, 1]
$$

**Example**: Among all predictions with 80% confidence, 80% should be correct.

### Expected Calibration Error (ECE)

Partition predictions into $M$ bins by confidence:

$$
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

where:
- $B_m$ is the set of samples in bin $m$
- $\text{acc}(B_m)$ is the accuracy in bin $m$
- $\text{conf}(B_m)$ is the average confidence in bin $m$

### Reliability Diagrams

Plot accuracy vs. confidence for each bin:
- **Perfect calibration**: Points on the diagonal
- **Overconfident**: Points below the diagonal
- **Underconfident**: Points above the diagonal

### Neural Network Miscalibration

Modern neural networks are typically **overconfident**:

**Causes**:
1. **Cross-entropy training** encourages confident predictions
2. **ReLU activations** produce unbounded logits
3. **Batch normalization** changes calibration
4. **Model capacity** exceeds data complexity

**Bayesian solution**: Marginalizing over parameters naturally improves calibration.

---

## Out-of-Distribution Detection

### The OOD Problem

Standard neural networks cannot reliably detect inputs from outside the training distribution:

$$
x_{\text{OOD}} \notin \text{support}(p_{\text{train}}(x))
$$

**Desired behavior**: High uncertainty on OOD inputs

### Using Uncertainty for OOD Detection

**Epistemic uncertainty** should be high for OOD inputs because the model has never seen similar data.

**Detection score**:

$$
s(x) = \text{Epistemic}(x) \quad \text{or} \quad s(x) = \text{Total}(x)
$$

**Decision rule**:

$$
\text{OOD if } s(x) > \tau
$$

### Evaluation Metrics

**AUROC**: Area under ROC curve for OOD detection
- In-distribution samples: negative class
- OOD samples: positive class

**AUPRC**: Area under precision-recall curve

**FPR@95**: False positive rate when true positive rate is 95%

### Challenges

1. **Overconfident extrapolation**: ReLU networks can be confident far from data
2. **Feature collapse**: Deep networks may map OOD inputs to in-distribution features
3. **Distributional shift**: Gradual shifts harder to detect than clear OOD

---

## Heteroscedastic Neural Networks

### Architecture

A heteroscedastic network predicts both mean and variance:

$$
f_\theta(x) = [\mu_\theta(x), \log \sigma^2_\theta(x)]
$$

**Why log variance?**
- Ensures $\sigma^2 > 0$
- Numerically stable
- Easier optimization

### Training Objective

**Negative log-likelihood** for Gaussian observations:

$$
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \left[ \frac{(y_i - \mu_\theta(x_i))^2}{2\sigma^2_\theta(x_i)} + \frac{1}{2}\log \sigma^2_\theta(x_i) \right]
$$

**Interpretation**:
- First term: Prediction error weighted by inverse variance
- Second term: Regularizes variance (prevents infinite variance)

### Practical Considerations

**Training stability**:
- Initialize variance predictions conservatively
- Use separate learning rates for mean and variance heads
- Clip variance to prevent numerical issues

**Architecture choices**:
- Shared backbone with two output heads
- Separate networks (more flexible but more parameters)
- Variance can depend on features or just input

---

## Python Implementation

```python
"""
Uncertainty Quantification in Neural Networks

This module provides implementations for:
- Aleatoric and epistemic uncertainty estimation
- Heteroscedastic neural networks
- Calibration metrics and reliability diagrams
- Out-of-distribution detection
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import softmax, logsumexp
from typing import Tuple, List, Optional, Dict, Callable
from dataclasses import dataclass
import warnings


# =============================================================================
# Uncertainty Estimation
# =============================================================================

def predictive_uncertainty_regression(
    predictions: np.ndarray,
    variances: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute uncertainty decomposition for regression.
    
    Parameters
    ----------
    predictions : ndarray of shape (n_samples, n_points)
        Mean predictions from each posterior sample
    variances : ndarray of shape (n_samples, n_points), optional
        Predicted variances (aleatoric) from each sample.
        If None, assumes homoscedastic with variance=0.
    
    Returns
    -------
    mean : ndarray of shape (n_points,)
        Predictive mean
    total_var : ndarray of shape (n_points,)
        Total predictive variance
    aleatoric : ndarray of shape (n_points,)
        Aleatoric (data) uncertainty
    epistemic : ndarray of shape (n_points,)
        Epistemic (model) uncertainty
    """
    n_samples, n_points = predictions.shape
    
    # Predictive mean
    mean = np.mean(predictions, axis=0)
    
    # Epistemic: variance of the means
    epistemic = np.var(predictions, axis=0)
    
    # Aleatoric: mean of the variances
    if variances is not None:
        aleatoric = np.mean(variances, axis=0)
    else:
        aleatoric = np.zeros(n_points)
    
    # Total variance
    total_var = epistemic + aleatoric
    
    return mean, total_var, aleatoric, epistemic


def predictive_uncertainty_classification(
    logits: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute uncertainty decomposition for classification.
    
    Parameters
    ----------
    logits : ndarray of shape (n_samples, n_points, n_classes)
        Logits from each posterior sample
    
    Returns
    -------
    mean_probs : ndarray of shape (n_points, n_classes)
        Mean predicted probabilities
    total_entropy : ndarray of shape (n_points,)
        Total predictive entropy
    aleatoric : ndarray of shape (n_points,)
        Aleatoric uncertainty (expected entropy)
    epistemic : ndarray of shape (n_points,)
        Epistemic uncertainty (mutual information)
    """
    n_samples, n_points, n_classes = logits.shape
    
    # Convert to probabilities
    probs = softmax(logits, axis=2)  # (n_samples, n_points, n_classes)
    
    # Mean probabilities
    mean_probs = np.mean(probs, axis=0)  # (n_points, n_classes)
    
    # Total entropy: H[E[p]]
    total_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
    
    # Expected entropy: E[H[p]] (aleatoric)
    sample_entropies = -np.sum(probs * np.log(probs + 1e-10), axis=2)
    aleatoric = np.mean(sample_entropies, axis=0)
    
    # Mutual information (epistemic)
    epistemic = total_entropy - aleatoric
    
    return mean_probs, total_entropy, aleatoric, epistemic


def entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute entropy of probability distribution."""
    return -np.sum(probs * np.log(probs + 1e-10), axis=axis)


def mutual_information(logits: np.ndarray) -> np.ndarray:
    """
    Compute mutual information I[y; theta | x, D] for BALD.
    
    Parameters
    ----------
    logits : ndarray of shape (n_samples, n_points, n_classes)
    
    Returns
    -------
    mi : ndarray of shape (n_points,)
        Mutual information for each point
    """
    _, _, _, mi = predictive_uncertainty_classification(logits)
    return mi


# =============================================================================
# Calibration Metrics
# =============================================================================

def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute data for reliability diagram.
    
    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True labels (0 or 1 for binary, class indices for multiclass)
    y_prob : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities
    n_bins : int
        Number of bins
    
    Returns
    -------
    bin_centers : ndarray
        Center of each bin
    bin_accs : ndarray
        Accuracy in each bin
    bin_confs : ndarray
        Mean confidence in each bin
    """
    if y_prob.ndim == 1:
        # Binary classification
        confidences = y_prob
        predictions = (y_prob > 0.5).astype(int)
        accuracies = (predictions == y_true).astype(float)
    else:
        # Multiclass: use maximum probability
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_accs[i] = np.mean(accuracies[mask])
            bin_confs[i] = np.mean(confidences[mask])
        else:
            bin_accs[i] = np.nan
            bin_confs[i] = np.nan
    
    return bin_centers, bin_accs, bin_confs


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Parameters
    ----------
    y_true : ndarray
        True labels
    y_prob : ndarray
        Predicted probabilities
    n_bins : int
        Number of bins
    
    Returns
    -------
    float
        ECE value
    """
    if y_prob.ndim == 1:
        confidences = y_prob
        predictions = (y_prob > 0.5).astype(int)
    else:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
    
    accuracies = (predictions == y_true).astype(float)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        n_bin = np.sum(mask)
        
        if n_bin > 0:
            acc = np.mean(accuracies[mask])
            conf = np.mean(confidences[mask])
            ece += (n_bin / len(y_true)) * np.abs(acc - conf)
    
    return ece


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    """
    _, bin_accs, bin_confs = reliability_diagram_data(y_true, y_prob, n_bins)
    
    valid = ~np.isnan(bin_accs)
    if not np.any(valid):
        return 0.0
    
    return np.max(np.abs(bin_accs[valid] - bin_confs[valid]))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier score for probabilistic predictions.
    
    Lower is better. Range: [0, 1] for binary classification.
    """
    if y_prob.ndim == 1:
        # Binary
        return np.mean((y_prob - y_true) ** 2)
    else:
        # Multiclass: one-hot encode true labels
        n_classes = y_prob.shape[1]
        y_true_onehot = np.eye(n_classes)[y_true]
        return np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))


# =============================================================================
# Out-of-Distribution Detection
# =============================================================================

def ood_detection_metrics(
    in_scores: np.ndarray,
    out_scores: np.ndarray
) -> Dict[str, float]:
    """
    Compute OOD detection metrics.
    
    Higher uncertainty scores should indicate OOD samples.
    
    Parameters
    ----------
    in_scores : ndarray
        Uncertainty scores for in-distribution samples
    out_scores : ndarray
        Uncertainty scores for OOD samples
    
    Returns
    -------
    dict
        Dictionary with AUROC, AUPRC, FPR@95
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    
    # Labels: 0 for in-distribution, 1 for OOD
    y_true = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
    y_score = np.concatenate([in_scores, out_scores])
    
    # AUROC
    auroc = roc_auc_score(y_true, y_score)
    
    # AUPRC
    auprc = average_precision_score(y_true, y_score)
    
    # FPR@95: False positive rate when true positive rate is 95%
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95 = fpr[idx]
    
    return {
        'auroc': auroc,
        'auprc': auprc,
        'fpr_at_95': fpr_at_95
    }


def max_softmax_probability(logits: np.ndarray) -> np.ndarray:
    """
    Baseline OOD detector: negative max softmax probability.
    
    Higher value = more uncertain = more likely OOD.
    """
    probs = softmax(logits, axis=-1)
    return -np.max(probs, axis=-1)


def predictive_entropy_score(logits: np.ndarray) -> np.ndarray:
    """
    OOD detector: predictive entropy.
    
    For ensemble/Bayesian: logits shape (n_samples, n_points, n_classes)
    """
    if logits.ndim == 3:
        # Bayesian: average probabilities then compute entropy
        probs = softmax(logits, axis=2)
        mean_probs = np.mean(probs, axis=0)
        return entropy(mean_probs, axis=1)
    else:
        # Single model
        probs = softmax(logits, axis=1)
        return entropy(probs, axis=1)


# =============================================================================
# Heteroscedastic Networks
# =============================================================================

class HeteroscedasticLoss:
    """
    Heteroscedastic Gaussian negative log-likelihood loss.
    
    Loss = (y - mu)^2 / (2 * sigma^2) + 0.5 * log(sigma^2)
    """
    
    def __call__(
        self,
        y_true: np.ndarray,
        mu: np.ndarray,
        log_var: np.ndarray
    ) -> float:
        """
        Compute loss.
        
        Parameters
        ----------
        y_true : ndarray
            True values
        mu : ndarray
            Predicted means
        log_var : ndarray
            Predicted log variances
        
        Returns
        -------
        float
            Mean loss
        """
        var = np.exp(log_var)
        loss = 0.5 * ((y_true - mu) ** 2 / var + log_var)
        return np.mean(loss)
    
    def gradient(
        self,
        y_true: np.ndarray,
        mu: np.ndarray,
        log_var: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients.
        
        Returns
        -------
        grad_mu : ndarray
            Gradient w.r.t. predicted mean
        grad_log_var : ndarray
            Gradient w.r.t. predicted log variance
        """
        var = np.exp(log_var)
        residual = y_true - mu
        
        grad_mu = -residual / var
        grad_log_var = 0.5 * (1 - residual ** 2 / var)
        
        return grad_mu, grad_log_var


class SimpleHeteroscedasticNetwork:
    """
    Simple heteroscedastic neural network for demonstration.
    
    Uses a shared backbone with two output heads:
    - Mean head: predicts E[y|x]
    - Log-variance head: predicts log Var[y|x]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [50, 50],
        init_log_var: float = 0.0
    ):
        """
        Initialize network.
        
        Parameters
        ----------
        input_dim : int
            Input dimension
        hidden_dims : list
            Hidden layer sizes
        init_log_var : float
            Initial log variance (conservative start)
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.init_log_var = init_log_var
        
        # Initialize weights (simplified)
        self.weights = self._initialize_weights()
        self.loss_fn = HeteroscedasticLoss()
    
    def _initialize_weights(self) -> Dict:
        """Xavier initialization."""
        weights = {}
        
        dims = [self.input_dim] + self.hidden_dims
        
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
            weights[f'W{i}'] = np.random.randn(dims[i], dims[i + 1]) * scale
            weights[f'b{i}'] = np.zeros(dims[i + 1])
        
        # Output heads
        last_dim = dims[-1]
        
        # Mean head
        weights['W_mu'] = np.random.randn(last_dim, 1) * np.sqrt(2.0 / last_dim)
        weights['b_mu'] = np.zeros(1)
        
        # Log variance head (initialized conservatively)
        weights['W_logvar'] = np.random.randn(last_dim, 1) * 0.01
        weights['b_logvar'] = np.full(1, self.init_log_var)
        
        return weights
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.
        
        Returns
        -------
        mu : ndarray
            Predicted means
        log_var : ndarray
            Predicted log variances
        """
        h = X
        
        # Hidden layers
        for i in range(len(self.hidden_dims)):
            h = h @ self.weights[f'W{i}'] + self.weights[f'b{i}']
            h = np.maximum(h, 0)  # ReLU
        
        # Output heads
        mu = h @ self.weights['W_mu'] + self.weights['b_mu']
        log_var = h @ self.weights['W_logvar'] + self.weights['b_logvar']
        
        return mu.flatten(), log_var.flatten()
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with uncertainty.
        
        Returns
        -------
        mu : ndarray
            Predicted means
        std : ndarray (if return_std=True)
            Predicted standard deviations (aleatoric only)
        """
        mu, log_var = self.forward(X)
        
        if return_std:
            std = np.sqrt(np.exp(log_var))
            return mu, std
        return mu, None


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_uncertainty_decomposition(
    x: np.ndarray,
    mean: np.ndarray,
    aleatoric: np.ndarray,
    epistemic: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    x_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    title: str = "Uncertainty Decomposition"
):
    """
    Plot predictive uncertainty with decomposition.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    total = np.sqrt(aleatoric + epistemic)
    aleatoric_std = np.sqrt(aleatoric)
    epistemic_std = np.sqrt(epistemic)
    
    # Total uncertainty
    ax = axes[0]
    ax.fill_between(x, mean - 2*total, mean + 2*total, 
                    alpha=0.3, label='Total ±2σ')
    ax.plot(x, mean, 'b-', label='Mean prediction')
    if y_true is not None:
        ax.plot(x, y_true, 'k--', label='True function')
    if x_train is not None:
        ax.scatter(x_train, y_train, c='red', s=20, zorder=5, label='Training data')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Total Uncertainty')
    ax.legend()
    
    # Aleatoric
    ax = axes[1]
    ax.fill_between(x, mean - 2*aleatoric_std, mean + 2*aleatoric_std,
                    alpha=0.3, color='orange', label='Aleatoric ±2σ')
    ax.plot(x, mean, 'b-')
    if y_true is not None:
        ax.plot(x, y_true, 'k--')
    if x_train is not None:
        ax.scatter(x_train, y_train, c='red', s=20, zorder=5)
    ax.set_xlabel('x')
    ax.set_title('Aleatoric Uncertainty')
    
    # Epistemic
    ax = axes[2]
    ax.fill_between(x, mean - 2*epistemic_std, mean + 2*epistemic_std,
                    alpha=0.3, color='green', label='Epistemic ±2σ')
    ax.plot(x, mean, 'b-')
    if y_true is not None:
        ax.plot(x, y_true, 'k--')
    if x_train is not None:
        ax.scatter(x_train, y_train, c='red', s=20, zorder=5)
    ax.set_xlabel('x')
    ax.set_title('Epistemic Uncertainty')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram"
):
    """
    Plot reliability diagram with calibration metrics.
    """
    bin_centers, bin_accs, bin_confs = reliability_diagram_data(y_true, y_prob, n_bins)
    ece = expected_calibration_error(y_true, y_prob, n_bins)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    ax = axes[0]
    valid = ~np.isnan(bin_accs)
    
    ax.bar(bin_centers[valid], bin_accs[valid], width=0.08, alpha=0.7, 
           edgecolor='black', label='Accuracy')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'{title}\nECE = {ece:.4f}')
    ax.legend()
    
    # Confidence histogram
    ax = axes[1]
    if y_prob.ndim == 1:
        confidences = y_prob
    else:
        confidences = np.max(y_prob, axis=1)
    
    ax.hist(confidences, bins=n_bins, range=(0, 1), alpha=0.7, edgecolor='black')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution')
    
    plt.tight_layout()
    plt.show()


def plot_ood_detection(
    in_scores: np.ndarray,
    out_scores: np.ndarray,
    title: str = "OOD Detection"
):
    """
    Visualize OOD detection performance.
    """
    try:
        metrics = ood_detection_metrics(in_scores, out_scores)
    except ImportError:
        print("sklearn required for ROC metrics")
        metrics = {}
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Score distributions
    ax = axes[0]
    ax.hist(in_scores, bins=50, alpha=0.7, label='In-distribution', density=True)
    ax.hist(out_scores, bins=50, alpha=0.7, label='OOD', density=True)
    ax.set_xlabel('Uncertainty Score')
    ax.set_ylabel('Density')
    ax.set_title('Score Distributions')
    ax.legend()
    
    # ROC curve
    ax = axes[1]
    if metrics:
        from sklearn.metrics import roc_curve
        y_true = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
        y_score = np.concatenate([in_scores, out_scores])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, 
                label=f'AUROC = {metrics["auroc"]:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'sklearn required', ha='center', va='center')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    if metrics:
        print(f"\nOOD Detection Metrics:")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  AUPRC: {metrics['auprc']:.4f}")
        print(f"  FPR@95: {metrics['fpr_at_95']:.4f}")


# =============================================================================
# Demo Functions
# =============================================================================

def demo_uncertainty_decomposition():
    """Demonstrate uncertainty decomposition in regression."""
    
    print("=" * 70)
    print("UNCERTAINTY DECOMPOSITION: REGRESSION")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate heteroscedastic data
    n_train = 20
    x_train = np.sort(np.random.uniform(-3, 3, n_train))
    noise_std = 0.1 + 0.2 * np.abs(x_train)  # Heteroscedastic noise
    y_train = np.sin(x_train) + np.random.normal(0, noise_std)
    
    # Test points
    x_test = np.linspace(-5, 5, 200)
    y_true = np.sin(x_test)
    
    # Simulate ensemble predictions (simplified BNN approximation)
    n_samples = 100
    predictions = np.zeros((n_samples, len(x_test)))
    variances = np.zeros((n_samples, len(x_test)))
    
    for s in range(n_samples):
        # Perturbed parameters (simulating posterior samples)
        a = 1.0 + np.random.normal(0, 0.1)
        b = np.random.normal(0, 0.2)
        
        predictions[s] = a * np.sin(x_test + b)
        
        # Predicted aleatoric variance
        variances[s] = (0.1 + 0.2 * np.abs(x_test)) ** 2
    
    # Compute decomposition
    mean, total, aleatoric, epistemic = predictive_uncertainty_regression(
        predictions, variances
    )
    
    print(f"\nTraining points: {n_train}")
    print(f"Ensemble samples: {n_samples}")
    
    print("\n--- Uncertainty Summary ---")
    print(f"Mean aleatoric (data region):  {np.mean(aleatoric[50:150]):.4f}")
    print(f"Mean epistemic (data region):  {np.mean(epistemic[50:150]):.4f}")
    print(f"Mean epistemic (extrapolation): {np.mean(epistemic[:30]):.4f}")
    
    print("\n*** Epistemic uncertainty is high outside training data")
    print("*** Aleatoric uncertainty reflects inherent noise pattern")
    
    return x_test, mean, aleatoric, epistemic


def demo_classification_uncertainty():
    """Demonstrate uncertainty in classification."""
    
    print("\n" + "=" * 70)
    print("UNCERTAINTY DECOMPOSITION: CLASSIFICATION")
    print("=" * 70)
    
    np.random.seed(42)
    
    n_samples = 50  # Ensemble members
    n_points = 4
    n_classes = 3
    
    # Simulate different scenarios
    scenarios = {
        'confident_correct': np.tile([5.0, 0.0, 0.0], (n_samples, 1)),
        'confident_wrong': np.tile([0.0, 5.0, 0.0], (n_samples, 1)),
        'high_aleatoric': np.tile([0.5, 0.5, 0.0], (n_samples, 1)),
        'high_epistemic': np.random.randn(n_samples, n_classes) * 2
    }
    
    print("\n--- Scenario Analysis ---")
    print(f"{'Scenario':<20} {'Total H':>10} {'Aleatoric':>10} {'Epistemic':>10}")
    print("-" * 55)
    
    for name, logits in scenarios.items():
        # Reshape for function
        logits_shaped = logits.reshape(n_samples, 1, n_classes)
        
        mean_probs, total, aleatoric, epistemic = predictive_uncertainty_classification(
            logits_shaped
        )
        
        print(f"{name:<20} {total[0]:>10.4f} {aleatoric[0]:>10.4f} {epistemic[0]:>10.4f}")
    
    print("\n*** Confident predictions have low total uncertainty")
    print("*** High aleatoric = inherent class overlap")
    print("*** High epistemic = model disagreement (ensemble variance)")


def demo_calibration():
    """Demonstrate calibration metrics and reliability diagrams."""
    
    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS")
    print("=" * 70)
    
    np.random.seed(42)
    n = 1000
    
    # Generate well-calibrated predictions
    true_probs = np.random.uniform(0, 1, n)
    y_true_calibrated = (np.random.uniform(0, 1, n) < true_probs).astype(int)
    y_prob_calibrated = true_probs + np.random.normal(0, 0.05, n)
    y_prob_calibrated = np.clip(y_prob_calibrated, 0.01, 0.99)
    
    # Generate overconfident predictions
    y_prob_overconfident = np.where(
        y_prob_calibrated > 0.5,
        0.5 + (y_prob_calibrated - 0.5) * 1.5,
        0.5 - (0.5 - y_prob_calibrated) * 1.5
    )
    y_prob_overconfident = np.clip(y_prob_overconfident, 0.01, 0.99)
    
    # Compute metrics
    ece_calib = expected_calibration_error(y_true_calibrated, y_prob_calibrated)
    ece_over = expected_calibration_error(y_true_calibrated, y_prob_overconfident)
    
    brier_calib = brier_score(y_true_calibrated, y_prob_calibrated)
    brier_over = brier_score(y_true_calibrated, y_prob_overconfident)
    
    print("\n--- Calibration Metrics ---")
    print(f"{'Model':<20} {'ECE':>10} {'Brier':>10}")
    print("-" * 45)
    print(f"{'Well-calibrated':<20} {ece_calib:>10.4f} {brier_calib:>10.4f}")
    print(f"{'Overconfident':<20} {ece_over:>10.4f} {brier_over:>10.4f}")
    
    print("\n*** Lower ECE = better calibration")
    print("*** Lower Brier score = better probabilistic predictions")


def demo_ood_detection():
    """Demonstrate out-of-distribution detection."""
    
    print("\n" + "=" * 70)
    print("OUT-OF-DISTRIBUTION DETECTION")
    print("=" * 70)
    
    np.random.seed(42)
    
    n_samples = 50  # Ensemble members
    n_in = 500
    n_out = 500
    n_classes = 10
    
    # In-distribution: confident predictions
    logits_in = np.random.randn(n_samples, n_in, n_classes)
    # Make confident by scaling and adding class-specific bias
    confident_class = np.random.randint(0, n_classes, n_in)
    for i in range(n_in):
        logits_in[:, i, confident_class[i]] += 3.0
    
    # OOD: uncertain predictions (ensemble disagrees)
    logits_out = np.random.randn(n_samples, n_out, n_classes) * 0.5
    
    # Compute epistemic uncertainty (mutual information)
    _, _, _, epistemic_in = predictive_uncertainty_classification(logits_in)
    _, _, _, epistemic_out = predictive_uncertainty_classification(logits_out)
    
    print("\n--- Uncertainty Statistics ---")
    print(f"In-distribution epistemic:  mean={np.mean(epistemic_in):.4f}, "
          f"std={np.std(epistemic_in):.4f}")
    print(f"OOD epistemic:              mean={np.mean(epistemic_out):.4f}, "
          f"std={np.std(epistemic_out):.4f}")
    
    # Compute detection metrics
    try:
        metrics = ood_detection_metrics(epistemic_in, epistemic_out)
        print("\n--- Detection Performance (using epistemic uncertainty) ---")
        print(f"AUROC: {metrics['auroc']:.4f}")
        print(f"AUPRC: {metrics['auprc']:.4f}")
        print(f"FPR@95: {metrics['fpr_at_95']:.4f}")
    except ImportError:
        print("\n(sklearn required for detailed metrics)")
    
    print("\n*** Higher epistemic uncertainty indicates OOD samples")
    print("*** Bayesian methods naturally provide OOD detection capability")


def demo_heteroscedastic_loss():
    """Demonstrate heteroscedastic loss computation."""
    
    print("\n" + "=" * 70)
    print("HETEROSCEDASTIC GAUSSIAN LOSS")
    print("=" * 70)
    
    np.random.seed(42)
    
    loss_fn = HeteroscedasticLoss()
    
    # Scenario 1: Good predictions with correct uncertainty
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mu_good = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    log_var_good = np.log(np.array([0.1, 0.1, 0.1, 0.1, 0.1]))
    
    loss_good = loss_fn(y_true, mu_good, log_var_good)
    
    # Scenario 2: Good predictions with overestimated uncertainty
    log_var_overest = np.log(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    loss_overest = loss_fn(y_true, mu_good, log_var_overest)
    
    # Scenario 3: Bad predictions with underestimated uncertainty
    mu_bad = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    log_var_underest = np.log(np.array([0.01, 0.01, 0.01, 0.01, 0.01]))
    loss_underest = loss_fn(y_true, mu_bad, log_var_underest)
    
    # Scenario 4: Bad predictions with high uncertainty (honest about errors)
    log_var_honest = np.log(np.array([2.0, 2.0, 2.0, 2.0, 2.0]))
    loss_honest = loss_fn(y_true, mu_bad, log_var_honest)
    
    print("\n--- Loss Comparison ---")
    print(f"{'Scenario':<45} {'Loss':>10}")
    print("-" * 60)
    print(f"{'Good predictions, correct uncertainty':<45} {loss_good:>10.4f}")
    print(f"{'Good predictions, overestimated uncertainty':<45} {loss_overest:>10.4f}")
    print(f"{'Bad predictions, underestimated uncertainty':<45} {loss_underest:>10.4f}")
    print(f"{'Bad predictions, honest high uncertainty':<45} {loss_honest:>10.4f}")
    
    print("\n*** Loss penalizes both prediction errors AND miscalibrated uncertainty")
    print("*** Underestimating uncertainty on wrong predictions is heavily penalized")


if __name__ == "__main__":
    demo_uncertainty_decomposition()
    demo_classification_uncertainty()
    demo_calibration()
    demo_ood_detection()
    demo_heteroscedastic_loss()
```

---

## Summary

### Types of Uncertainty

| Type | Definition | Source | Reducible? |
|------|------------|--------|------------|
| **Aleatoric** | Inherent data noise | Measurement, stochasticity | No |
| **Epistemic** | Model ignorance | Limited data, parameters | Yes |
| **Total** | Aleatoric + Epistemic | Both sources combined | Partially |

### Mathematical Decomposition

**Regression** (law of total variance):

$$
\text{Var}[y^* \mid x^*, \mathcal{D}] = \underbrace{\mathbb{E}_\theta[\sigma^2_\theta(x^*)]}_{\text{Aleatoric}} + \underbrace{\text{Var}_\theta[\mu_\theta(x^*)]}_{\text{Epistemic}}
$$

**Classification** (mutual information):

$$
\underbrace{\mathbb{H}[\bar{p}]}_{\text{Total}} = \underbrace{\mathbb{I}[y; \theta]}_{\text{Epistemic}} + \underbrace{\mathbb{E}_\theta[\mathbb{H}[p_\theta]]}_{\text{Aleatoric}}
$$

### Estimation from Posterior Samples

| Quantity | Formula |
|----------|---------|
| Predictive mean | $\hat{\mu} = \frac{1}{S}\sum_s \mu_{\theta^{(s)}}$ |
| Epistemic | $\frac{1}{S}\sum_s (\mu_{\theta^{(s)}} - \hat{\mu})^2$ |
| Aleatoric | $\frac{1}{S}\sum_s \sigma^2_{\theta^{(s)}}$ |
| Total | Epistemic + Aleatoric |

### Calibration Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **ECE** | $\sum_m \frac{|B_m|}{n}\|acc_m - conf_m\|$ | Expected calibration error |
| **MCE** | $\max_m \|acc_m - conf_m\|$ | Worst-case calibration |
| **Brier** | $\frac{1}{n}\sum_i (p_i - y_i)^2$ | Probabilistic accuracy |

### Applications

| Application | Key Uncertainty | Usage |
|-------------|-----------------|-------|
| **Active Learning** | Epistemic (MI/BALD) | Select informative points |
| **OOD Detection** | Epistemic | Flag unfamiliar inputs |
| **Risk Assessment** | Total | Decision confidence |
| **Model Debugging** | Both | Identify failure modes |

### Connections to Other Chapters

| Topic | Chapter | Connection |
|-------|---------|------------|
| Prior specification | Ch13: Prior on Weights | Affects epistemic uncertainty |
| Posterior inference | Ch13: Posterior Inference | Source of parameter samples |
| MC Dropout | Ch13: MC Dropout | Approximate epistemic uncertainty |
| Variational BNN | Ch13: Variational BNN | Scalable uncertainty estimation |
| Model comparison | Ch13: Information Criteria | Uncertainty in model selection |

### Key References

- Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? *NeurIPS*.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *ICML*.
- Guo, C., et al. (2017). On calibration of modern neural networks. *ICML*.
- Lakshminarayanan, B., et al. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
- Houlsby, N., et al. (2011). Bayesian active learning for classification and preference learning. *arXiv*.
