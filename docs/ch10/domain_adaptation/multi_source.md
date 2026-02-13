# Multi-Source Domain Adaptation

## Introduction

Multi-source domain adaptation addresses the realistic scenario where labeled data exists across multiple source domains with different characteristics, requiring models to leverage information from all sources while adapting to an unlabeled target domain. Unlike single-source adaptation, multi-source learning enables richer feature representations by combining complementary information from diverse domains.

In quantitative finance, multi-source adaptation is particularly valuable: models can leverage historical market data from multiple time periods, different asset classes, and various market conditions simultaneously. This multiplicity provides robustness that single-source transfer learning cannot achieve, while properly weighting sources to prioritize those most similar to the target.

## Key Concepts

- **Source Weighting**: Automatic assessment of each source domain's relevance
- **Domain Consensus**: Learning representations consistent across all sources
- **Selective Transfer**: Emphasize relevant sources, downweight problematic ones
- **Multi-Task Learning**: Simultaneously adapt across multiple source-target pairs
- **Feature Alignment**: Minimize discrepancy across all source-target pairs

## Multi-Source Adaptation Framework

### Problem Formulation

Given:
- $K$ source domains with labeled data: $\{(\mathbf{x}^k_i, y^k_i)\}_{i=1}^{n_k}$ for $k = 1, \ldots, K$
- One target domain with unlabeled data: $\{\mathbf{x}^t_j\}_{j=1}^{n_t}$

Minimize:

$$\mathcal{L} = \sum_{k=1}^{K} w_k \mathcal{L}_{\text{task}}^k + \lambda \sum_{k=1}^{K} w_k \text{MMD}(\mathbf{f}^k_s, \mathbf{f}_t)$$

where $w_k$ are learned source weights and $\mathbf{f}^k_s, \mathbf{f}_t$ are domain-specific and target features.

### Architecture Design

Multi-source adaptation requires careful architectural choices:

```
Source 1 ──┐
Source 2 ──┼──> Task Classifiers ──┐
Source K ──┘                       │
                                   ├──> Ensemble Decision
Target ─────────> Shared Feature ──┘
                   Extractor
```

### Source Weighting Mechanisms

#### Instance-Level Weighting

Weight samples based on predicted source relevance:

$$w_i^k = p(\text{source}_k | \mathbf{x}_i^t)$$

Computed through a small domain classifier network trained to distinguish source predictions on target data.

#### Domain-Level Weighting

Assign fixed weights to entire source domains:

$$w_k = \frac{\exp(\text{similarity}_k)}{\sum_{k'} \exp(\text{similarity}_{k'})}$$

where similarity is measured by validation performance or feature distribution matching.

#### Entropy-Based Weighting

Use model uncertainty to weight sources:

$$w_k = \frac{1}{H_k}$$

where $H_k$ is the entropy of source $k$'s predictions on target domain. High-confidence sources receive higher weight.

## Multi-Domain Discrepancy

### Joint MMD

Extend MMD to multiple source domains:

$$\text{MMD}^2_{\text{multi}} = \sum_{k=1}^{K} \text{MMD}^2(\mathbf{f}^k_s, \mathbf{f}_t) + \alpha \sum_{k<k'} \text{MMD}^2(\mathbf{f}^k_s, \mathbf{f}^{k'}_s)$$

The second term encourages agreement among source representations (domain consensus).

### H-Divergence Based Weighting

!!! tip "Theory-Grounded Weighting"
    H-divergence, related to error on a held-out source validation set, provides theoretical bounds on target error.

Weight sources based on their ability to discriminate from target:

$$w_k = 1 - \frac{1}{2} h_k(\mathbf{f}^k_s, \mathbf{f}_t)$$

where $h_k$ is the h-divergence between source $k$ and target feature distributions.

## Training Procedures

### Naive Multi-Source Approach

Simply train on all sources simultaneously:

**Pros**: Simple implementation, no source weighting needed

**Cons**: Problematic sources can dominate; no adaptation prioritization

### Curriculum-Based Multi-Source

Train sources in order of target relevance:

1. **Rank sources** by similarity to target domain
2. **Phase 1**: Train on most similar source
3. **Phase 2**: Add next most similar source
4. **Phase N**: Integrate all sources

$$\mathcal{L}(t) = \sum_{k \leq f(t)} \mathcal{L}^k$$

where $f(t)$ increases with training iteration.

### Adversarial Multi-Source

Learn which sources to trust through adversarial objectives:

$$\min_{\Phi} \max_{D} \sum_k w_k \mathcal{L}_{\text{task}}^k - \lambda \sum_k w_k \text{KL}(D(P^k_s) \| D(P_t))$$

Domain discriminator learns to distinguish sources, guide source weighting.

## Source Selection and Rejection

### Negative Transfer Detection

Identify sources hurting target performance:

$$\text{Harm}_k = A_{\text{target}}(\text{with}_k) - A_{\text{target}}(\text{without}_k)$$

Sources with $\text{Harm}_k > 0$ contribute negatively and should be downweighted or removed.

### Theoretical Source Importance

Based on domain adaptation theory:

$$\text{Importance}_k \propto \frac{1}{1 + \text{MMD}^2(\mathbf{f}^k_s, \mathbf{f}_t)}$$

Sources similar to target (low MMD) are more important.

## Applications in Quantitative Finance

!!! warning "Multi-Asset Transfer Learning"
    Leverage multiple asset classes and time periods:
    
    - **Cross-Asset Transfer**: Stock returns → bond returns → credit spreads
    - **Time Period Adaptation**: Pre-crisis → crisis → post-crisis regimes
    - **Geographic Transfer**: US markets → emerging markets → frontier markets
    - **Strategy Combination**: Momentum → value → quality factors

### Asset-Specific Example

Predict emerging market equity returns using:

**Source 1** (high relevance): Developed market equities
**Source 2** (medium relevance): Emerging market bonds  
**Source 3** (low relevance): Commodities

Automatic weighting emphasizes Source 1 while incorporating information from others.

## Empirical Performance

### Comparison with Single-Source

| Setting | Single-Source | Multi-Source | Improvement |
|---------|--------------|--------------|-------------|
| **Few Sources** | Baseline | +2-5% | Good |
| **Many Sources** | Decreases | +8-15% | Significant |
| **Diverse Sources** | Limited | +5-20% | Very Good |
| **Highly Similar Sources** | Adequate | +1-3% | Minimal |

## Advanced Topics

### Progressive Domain Alignment

Align domains hierarchically:

1. **Pairwise**: Align each source to target
2. **Consensus**: Find common feature space balancing all domains
3. **Refinement**: Fine-tune with weighted multi-source objective

### Domain-Specific Batch Normalization

Use separate batch norm statistics per source during training:

$$\text{BN}_k(\mathbf{x}) = \frac{\mathbf{x} - \mu_k}{\sqrt{\sigma_k^2 + \epsilon}} \gamma_k + \beta_k$$

Prevents feature statistics from averaging across domains prematurely.

## Research Directions

- Theoretical analysis of multi-source optimal weighting
- Automatic source ranking without validation data
- Dynamic source weighting during training
- Partial target labels for semi-supervised multi-source adaptation

## Related Topics

- Domain Adaptation Overview (Chapter 10.2)
- Maximum Mean Discrepancy (Chapter 10.2.1)
- Self-Training Methods (Chapter 10.2.3)
- Transfer Learning Fundamentals (Chapter 10)
