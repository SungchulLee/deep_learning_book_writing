# Domain Shift

Domain shift occurs when the data distribution changes between training (source) and deployment (target) environments. Understanding the types and severity of domain shift is essential for choosing appropriate transfer learning strategies.

## Formal Definition

Given:

- **Source domain** $\mathcal{D}_S$ with labeled data $\{(\mathbf{x}_i^s, y_i^s)\}_{i=1}^{n_s}$
- **Target domain** $\mathcal{D}_T$ with unlabeled data $\{\mathbf{x}_j^t\}_{j=1}^{n_t}$

The domains differ: $P_S(\mathbf{X}) \neq P_T(\mathbf{X})$

Goal: Learn a model that performs well on $\mathcal{D}_T$ using knowledge from $\mathcal{D}_S$.

## Types of Domain Shift

| Type | Definition | Example |
|------|------------|---------|
| Covariate Shift | $P_S(\mathbf{X}) \neq P_T(\mathbf{X})$, $P(Y|\mathbf{X})$ same | Day→Night photos |
| Label Shift | $P_S(Y) \neq P_T(Y)$, $P(\mathbf{X}|Y)$ same | Different class proportions |
| Concept Drift | $P_S(Y|\mathbf{X}) \neq P_T(Y|\mathbf{X})$ | Word meanings change over time |

### Covariate Shift

The most common form of domain shift. The input distribution changes but the labeling function remains the same. A classifier trained on daytime images may fail on nighttime images, even though the mapping from visual features to labels (e.g., "car") hasn't changed.

**Correction via importance weighting:**

$$\mathcal{L}_{\text{reweighted}} = \frac{1}{n_s} \sum_{i=1}^{n_s} \frac{P_T(\mathbf{x}_i)}{P_S(\mathbf{x}_i)} \ell(f(\mathbf{x}_i), y_i)$$

### Label Shift

The class distribution changes between domains. For example, a medical diagnosis model trained on hospital data (high disease prevalence) deployed in general screening (low prevalence).

### Concept Drift

The relationship between inputs and outputs changes. This is the hardest form to handle because the labeling function itself has shifted. Common in temporal settings where the meaning of features evolves.

## The Domain Adaptation Bound

Ben-David et al. (2010) established a foundational theoretical result:

$$\epsilon_T(h) \leq \epsilon_S(h) + \frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda^*$$

Target error is bounded by:

1. **Source error** $\epsilon_S(h)$: How well the model performs on source data
2. **Domain divergence** $d_{\mathcal{H}\Delta\mathcal{H}}$: How different the two domains are
3. **Irreducible error** $\lambda^*$: The error of the ideal joint hypothesis

This bound implies that reducing domain divergence (through domain adaptation) directly improves the target error guarantee.

## Measuring Domain Gap

### Feature-Level Assessment

```python
import torch
import torch.nn as nn
import numpy as np


def assess_domain_gap(model, source_loader, target_loader, device='cuda'):
    """Compute statistics to assess domain gap between source and target."""
    model.eval()
    model = model.to(device)

    def get_features(loader):
        features = []
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                feat = feature_extractor(inputs).squeeze()
                features.append(feat.cpu())
        return torch.cat(features, dim=0)

    source_feat = get_features(source_loader)
    target_feat = get_features(target_loader)

    # Compare statistics
    source_mean = source_feat.mean(dim=0)
    target_mean = target_feat.mean(dim=0)
    mean_diff = (source_mean - target_mean).norm().item()

    # MMD estimate
    mmd = compute_mmd(source_feat[:1000], target_feat[:1000]).item()

    print(f"Domain Gap Assessment:")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  MMD: {mmd:.4f}")

    return mean_diff, mmd


def compute_mmd(source_features, target_features, gamma=1.0):
    """Compute Maximum Mean Discrepancy between source and target features."""
    n_s, n_t = source_features.size(0), target_features.size(0)

    def rbf_kernel(x, y):
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        yy = torch.sum(y ** 2, dim=1, keepdim=True)
        distances = xx + yy.t() - 2 * torch.mm(x, y.t())
        return torch.exp(-gamma * distances)

    ss = rbf_kernel(source_features, source_features)
    tt = rbf_kernel(target_features, target_features)
    st = rbf_kernel(source_features, target_features)

    mmd = (ss.sum() / (n_s * n_s) +
           tt.sum() / (n_t * n_t) -
           2 * st.sum() / (n_s * n_t))

    return torch.sqrt(torch.clamp(mmd, min=0))
```

### Strategy Selection Based on Gap

```
Domain Gap Assessment
        │
        ├─► Gap is small (MMD < 0.1)
        │       └─► BatchNorm adaptation or standard fine-tuning
        │
        ├─► Gap is moderate (0.1 < MMD < 0.5)
        │       ├─► Labeled target data → Fine-tuning with augmentation
        │       └─► No target labels → MMD-based adaptation
        │
        └─► Gap is large (MMD > 0.5)
                ├─► Labeled target data → Full fine-tuning on target
                └─► No target labels → DANN or advanced methods
```

For detailed domain adaptation methods, see [Section 9.2](../domain_adaptation/domain_adaptation.md).

## Practical Implications for Transfer Learning

Domain shift severity directly determines the optimal transfer strategy:

| Gap Severity | Feature Extraction | Fine-Tuning | Domain Adaptation |
|-------------|-------------------|-------------|-------------------|
| Minimal | ✅ Excellent | ✅ Good | Not needed |
| Moderate | ⚠️ Degraded | ✅ Recommended | Helpful |
| Severe | ❌ Fails | ⚠️ Risk of overfitting | ✅ Required |

## References

1. Ben-David, S., et al. (2010). "A Theory of Learning from Different Domains." *Machine Learning*.
2. Shimodaira, H. (2000). "Improving Predictive Inference Under Covariate Shift." *Journal of Statistical Planning and Inference*.
3. Quiñonero-Candela, J., et al. (2009). *Dataset Shift in Machine Learning*. MIT Press.
