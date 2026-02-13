# Maximum Mean Discrepancy for Domain Adaptation

## Introduction

Maximum Mean Discrepancy (MMD) provides a principled, theoretically-grounded method for measuring and minimizing distribution differences between source and target domains. By learning representations where source and target distributions exhibit minimal discrepancy, MMD-based domain adaptation enables robust transfer learning without requiring labeled target data.

MMD's foundation in kernel theory and probability metrics makes it particularly attractive for practitioners requiring theoretical guarantees about distribution matching. In quantitative finance, MMD enables adaptation across market regimes and asset classes while maintaining interpretable, distance-based measures of adaptation success.

## Key Concepts

- **Distribution Discrepancy**: Measure of difference between probability distributions
- **Maximum Mean Discrepancy**: Kernel-based divergence measure with theoretical properties
- **Kernel Embedding**: Implicit mapping of distributions to Reproducing Kernel Hilbert Space
- **Domain-Invariant Features**: Representations where source and target distributions align
- **Unsupervised Domain Adaptation**: Learning without target domain labels

## Mathematical Framework

### MMD Definition

Maximum Mean Discrepancy between distributions $P$ and $Q$ is defined as:

$$\text{MMD}^2(P, Q) = \left\| \mathbb{E}_{\mathbf{x} \sim P}[\phi(\mathbf{x})] - \mathbb{E}_{\mathbf{y} \sim Q}[\phi(\mathbf{y})] \right\|_H^2$$

where:
- $\phi(\cdot)$ is a feature map to Reproducing Kernel Hilbert Space (RKHS)
- $\|\cdot\|_H$ is the RKHS norm
- $\mathbb{E}$ denotes expectation

### Kernel Representation

Using kernel function $k(\mathbf{x}, \mathbf{y}) = \langle \phi(\mathbf{x}), \phi(\mathbf{y}) \rangle_H$, MMD expands to:

$$\text{MMD}^2(P, Q) = \mathbb{E}_{x,x' \sim P}[k(\mathbf{x}, \mathbf{x}')] - 2\mathbb{E}_{x \sim P, y \sim Q}[k(\mathbf{x}, \mathbf{y})] + \mathbb{E}_{y,y' \sim Q}[k(\mathbf{y}, \mathbf{y}')]$$

### Empirical MMD Computation

Given finite samples $\{\mathbf{x}_i\}_{i=1}^m$ from $P$ and $\{\mathbf{y}_j\}_{j=1}^n$ from $Q$:

$$\widehat{\text{MMD}}^2 = \frac{1}{m(m-1)} \sum_{i \neq i'} k(\mathbf{x}_i, \mathbf{x}_{i'}) - \frac{2}{mn} \sum_{i,j} k(\mathbf{x}_i, \mathbf{y}_j) + \frac{1}{n(n-1)} \sum_{j \neq j'} k(\mathbf{y}_j, \mathbf{y}_{j'})$$

## Theoretical Properties

### Universality

!!! tip "Universal Kernels"
    MMD is a proper metric (satisfies symmetry, non-negativity, triangle inequality) when using universal kernels (RBF, Laplace).

For universal kernels, $\text{MMD}(P, Q) = 0 \iff P = Q$.

### Convergence Rate

By the law of large numbers, with $N = m + n$ samples:

$$\mathbb{E}[\widehat{\text{MMD}}^2 - \text{MMD}^2(P,Q)] = O\left(\frac{1}{\min(m,n)}\right)$$

Convergence is independent of feature dimension, providing robustness to high-dimensional representations.

## Kernel Selection

### Common Kernels for Domain Adaptation

| Kernel | Formula | Advantages | Trade-offs |
|--------|---------|-----------|-----------|
| **RBF** | $\exp(-\gamma\|\mathbf{x}-\mathbf{y}\|^2)$ | Universal, interpretable | Bandwidth selection critical |
| **Laplacian** | $\exp(-\gamma\|\mathbf{x}-\mathbf{y}\|)$ | Robust to outliers | Less smooth |
| **Polynomial** | $(\mathbf{x}^T\mathbf{y} + c)^d$ | Captures interactions | Curse of dimensionality |
| **Multi-Kernel** | $\sum_k \alpha_k k_k$ | Flexible | Complex tuning |

### Bandwidth Selection for RBF

For RBF kernel $k(\mathbf{x}, \mathbf{y}) = \exp(-\gamma \|\mathbf{x} - \mathbf{y}\|^2)$:

**Median Heuristic**:
$$\gamma = \frac{1}{2 \text{median}\{\|\mathbf{x}_i - \mathbf{x}_j\|^2\}_{i,j}}$$

**Cross-Validation**: Tune $\gamma$ to minimize domain adaptation loss.

## Domain Adaptation Framework

### Training Objective

Domain adaptation minimizes combined loss:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \cdot \text{MMD}^2(\mathbf{f}_s, \mathbf{f}_t)$$

where:
- $\mathcal{L}_{\text{task}}$ is supervised loss on source labels
- $\mathbf{f}_s, \mathbf{f}_t$ are learned representations
- $\lambda$ balances task performance and distribution matching

### Architecture Design

MMD-based adaptation typically uses:

1. **Shared Feature Extractor**: $\mathbf{f}(\mathbf{x}) = \Phi(\mathbf{x}; \theta)$ applied to both domains
2. **Task-Specific Classifier**: $y = W^T \mathbf{f}(\mathbf{x}) + b$
3. **MMD Loss**: Computed in feature space

```
Input Domain S ──┐
                 ├──> Feature Extractor ──> Task Classifier ──> Output
Input Domain T ──┘     (Shared)
                     │
                     └──> MMD Loss (Domain Alignment)
```

## Gradient Flow for MMD

During backpropagation, MMD gradients guide feature alignment:

$$\frac{\partial \text{MMD}^2}{\partial \Phi} = \frac{2}{m} \sum_i \phi'(\mathbf{x}_i) k'(\mathbf{x}_i, \cdot) - \frac{2}{n} \sum_j \phi'(\mathbf{y}_j) k'(\mathbf{y}_j, \cdot)$$

This naturally drives source features toward target distribution.

## Practical Considerations

### Computational Complexity

Computing pairwise kernel evaluations requires:

$$O(N^2 d)$$

where $N = m + n$ is total sample size and $d$ is feature dimension.

!!! note "Mini-Batch Optimization"
    Use mini-batches to reduce complexity: $O(k^2 d)$ where $k \ll N$.

### Scalability Strategies

**Random Fourier Features**: Approximate RBF kernel with $O(N d')$ complexity where $d' \ll d$.

$$k(\mathbf{x}, \mathbf{y}) \approx \frac{1}{m} \sum_{i=1}^m \cos(\mathbf{w}_i^T \mathbf{x} + b_i) \cos(\mathbf{w}_i^T \mathbf{y} + b_i)$$

**Nyström Approximation**: Use subset of samples for kernel matrix approximation.

## Extensions and Variants

### Multi-Kernel MMD

Combine multiple kernels with learned weights:

$$\text{MMD}^2_{\text{multi}} = \sum_{k} \beta_k \text{MMD}^2_k(P, Q)$$

where $\beta_k \geq 0$ and $\sum_k \beta_k = 1$.

### Joint Distribution Adaptation (JDA)

Adapt both marginal and conditional distributions:

$$\mathcal{L} = \text{MMD}(P_s, P_t) + \text{MMD}(P_s(y), P_t(y))$$

### Conditional MMD

Reduce discrepancy conditional on class labels (requires some target labels):

$$\text{CMMD} = \sum_c P(c) \text{MMD}(P_s(\mathbf{x}|c), P_t(\mathbf{x}|c))$$

## Applications in Quantitative Finance

!!! warning "Market Regime Adaptation"
    Use MMD to adapt models across market regimes:
    
    - **Bull → Sideways**: Minimize discrepancy between bull market and sideways features
    - **US → Emerging Markets**: Cross-market transfer through MMD alignment
    - **Equities → Bonds**: Asset class adaptation maintaining feature alignment

## Theoretical Guarantees

MMD provides bounds on target domain error:

$$\mathcal{E}_t(\mathbf{f}) \leq \mathcal{E}_s(\mathbf{f}) + \text{MMD}^2(P_s, P_t) + O(\sqrt{\frac{d_A}{n}})$$

where $d_A$ is domain adaptation dimension and $n$ is target sample size.

## Related Topics

- Domain Adaptation Overview (Chapter 10.2)
- Self-Training Methods (Chapter 10.2.3)
- Multi-Source Domain Adaptation (Chapter 10.2.2)
- Adversarial Domain Adaptation
