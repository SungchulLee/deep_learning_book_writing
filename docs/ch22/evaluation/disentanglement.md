# Disentanglement Metrics

Quantifying how well latent dimensions correspond to independent factors of variation.

---

## Learning Objectives

By the end of this section, you will be able to:

- Define disentanglement formally and intuitively
- Implement and compute common disentanglement metrics
- Understand the total correlation decomposition
- Evaluate β-VAE representations quantitatively

---

## What Is Disentanglement?

A representation is **disentangled** when each latent dimension controls a single, independent factor of variation in the data. For handwritten digits, ideal disentangled factors might include stroke width, slant angle, digit identity, and size — each controlled by a separate latent dimension.

---

## Qualitative Assessment: Latent Traversals

The simplest evaluation is visual inspection of latent traversals:

```python
def latent_traversal(model, device, dim_idx, num_steps=11, range_val=3.0):
    """Vary one latent dimension while fixing others at zero."""
    model.eval()
    values = torch.linspace(-range_val, range_val, num_steps)
    images = []
    
    with torch.no_grad():
        for val in values:
            z = torch.zeros(1, model.latent_dim, device=device)
            z[0, dim_idx] = val
            images.append(model.decode(z).cpu())
    
    return torch.stack(images)
```

**Disentangled:** Each row changes only one visual attribute. **Entangled:** Multiple attributes change simultaneously.

---

## Quantitative Metrics

### Mutual Information Gap (MIG)

MIG (Chen et al., 2018) measures the gap between the two latent dimensions with highest mutual information with each factor:

$$\text{MIG} = \frac{1}{K}\sum_{k=1}^{K} \frac{1}{H(v_k)}\left(I(z_{j^{(1)}(k)}; v_k) - I(z_{j^{(2)}(k)}; v_k)\right)$$

where $v_k$ is the $k$-th ground truth factor, $j^{(1)}(k)$ is the latent dimension with highest MI with $v_k$, and $j^{(2)}(k)$ is the second highest. Higher MIG means better disentanglement — each factor is captured primarily by a single latent dimension.

### DCI Disentanglement

DCI (Eastwood & Williams, 2018) trains a predictor from latent dimensions to each factor and measures how concentrated the importance weights are:

```python
def compute_dci(z_train, factors_train, z_test, factors_test):
    """
    Compute DCI disentanglement score.
    
    Uses feature importance from gradient boosted trees.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    import numpy as np
    
    n_factors = factors_train.shape[1]
    n_latents = z_train.shape[1]
    importance_matrix = np.zeros((n_latents, n_factors))
    
    for k in range(n_factors):
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3)
        clf.fit(z_train, factors_train[:, k])
        importance_matrix[:, k] = clf.feature_importances_
    
    # Disentanglement: for each factor, how concentrated is importance?
    disentanglement = 0
    for k in range(n_factors):
        imp = importance_matrix[:, k]
        imp = imp / (imp.sum() + 1e-8)
        # Entropy of importance distribution (lower = more disentangled)
        entropy = -(imp * np.log(imp + 1e-8)).sum()
        max_entropy = np.log(n_latents)
        disentanglement += 1 - entropy / max_entropy
    
    return disentanglement / n_factors
```

### Factor VAE Metric

The Factor VAE metric (Kim & Mnih, 2018) measures whether a classifier can identify which factor was held fixed given a pair of latent representations:

1. Fix one factor of variation $v_k$, sample two data points with different other factors
2. Encode both, compute the normalized variance of the difference across latent dimensions
3. The dimension with lowest variance should correspond to the fixed factor
4. Accuracy of this assignment = Factor VAE score

---

## Total Correlation Decomposition

The KL term decomposes into three components (β-TCVAE):

$$\mathbb{E}[D_{KL}(q(z|x) \| p(z))] = \underbrace{I(X; Z)}_{\text{Index-Code MI}} + \underbrace{TC(Z)}_{\text{Total Correlation}} + \underbrace{\sum_j D_{KL}(q(z_j) \| p(z_j))}_{\text{Dimension-wise KL}}$$

The **Total Correlation** $TC(Z) = D_{KL}(q(z) \| \prod_j q(z_j))$ measures statistical dependence between latent dimensions. Penalizing TC specifically (rather than the full KL) encourages disentanglement while allowing each dimension to use its capacity freely.

```python
def estimate_total_correlation(z_samples):
    """
    Estimate TC using the minibatch-weighted sampling approach.
    
    TC(Z) = E_q(z)[log q(z) - sum_j log q(z_j)]
    """
    batch_size, latent_dim = z_samples.shape
    
    # Estimate log q(z) using minibatch density estimation
    # log q(z_i) ≈ log(1/N * sum_j N(z_i; mu_j, sigma_j^2))
    # Simplified: use kernel density with batch samples
    
    # For each sample, compute log density under the batch
    log_qz = []
    for j in range(latent_dim):
        zj = z_samples[:, j:j+1]  # [B, 1]
        # KDE estimate for marginal q(z_j)
        diff = zj - zj.t()  # [B, B]
        log_qzj = torch.logsumexp(-0.5 * diff.pow(2), dim=1) - torch.log(
            torch.tensor(float(batch_size)))
        log_qz.append(log_qzj)
    
    log_prod_qz = torch.stack(log_qz, dim=1).sum(dim=1)  # sum of log marginals
    
    # Joint log q(z) via KDE
    diff = z_samples.unsqueeze(1) - z_samples.unsqueeze(0)  # [B, B, D]
    log_joint = torch.logsumexp(-0.5 * diff.pow(2).sum(dim=2), dim=1) - torch.log(
        torch.tensor(float(batch_size)))
    
    tc = (log_joint - log_prod_qz).mean()
    return tc
```

---

## Summary

| Metric | Measures | Range | Higher = |
|--------|----------|-------|----------|
| **MIG** | MI gap between top-2 latent dims per factor | [0, 1] | More disentangled |
| **DCI** | Concentration of feature importance | [0, 1] | More disentangled |
| **Factor VAE** | Classifier accuracy for fixed-factor identification | [0, 1] | More disentangled |
| **TC** | Statistical dependence between latent dims | ≥ 0 | Lower = more independent |

---

## Exercises

### Exercise 1

Train β-VAE with β ∈ {1, 4, 10, 20} on a dataset with known factors (e.g., dSprites). Compute MIG for each.

### Exercise 2

Implement the total correlation estimator and track TC during training for different β values.

---

## What's Next

The next section covers [Synthetic Data Generation](../finance/synthetic_data.md) for quantitative finance applications.
