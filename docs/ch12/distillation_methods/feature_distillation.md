# Feature Distillation for Continual Learning

## Introduction

Feature distillation in continual learning focuses on preserving learned representations across task sequences by matching intermediate layer features between previous and current models. Rather than only matching final outputs as in standard knowledge distillation, feature-level constraints preserve the rich learned feature hierarchies that encode task-relevant patterns across multiple abstraction levels.

In quantitative finance, feature-level preservation proves crucial: financial models learn hierarchical representations of market microstructure (order flow, tick patterns), technical factors (trends, momentum), and macroeconomic relationships. Feature distillation ensures these learned hierarchies transfer to new prediction tasks while adapting to novel market conditions.

## Key Concepts

- **Intermediate Representations**: Layer activations capturing learned features
- **Multi-Level Matching**: Distill from multiple layers simultaneously
- **Feature Alignment**: Handle dimension differences between models
- **Hierarchy Preservation**: Maintain feature abstraction levels
- **Task-Relevant Features**: Focus distillation on important representations
- **Computational Efficiency**: Trade-off between matching granularity and cost

## Mathematical Framework

### Single-Layer Feature Distillation

For layer $\ell$, match feature activations:

$$\mathcal{L}_{\text{feat}}^{(\ell)} = \sum_{\mathbf{x} \in D_{\text{prev}}} \|f_t^{(\ell)}(\mathbf{x}) - f_{t-1}^{(\ell)}(\mathbf{x})\|_2^2$$

where $f^{(\ell)}(\mathbf{x}) \in \mathbb{R}^{d_{\ell}}$ are layer $\ell$ activations.

### Multi-Layer Feature Distillation

Combine losses from multiple layers:

$$\mathcal{L}_{\text{feat}} = \sum_{\ell \in L} \alpha_{\ell} \mathcal{L}_{\text{feat}}^{(\ell)}$$

where $L$ is set of matched layers and $\alpha_{\ell}$ are layer-specific weights.

**Layer Selection**: Match 3-5 intermediate layers rather than all (balances information and cost).

### Total Continual Learning Loss

Combine feature distillation with task and output losses:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{out}} \mathcal{L}_{\text{output}} + \lambda_{\text{feat}} \mathcal{L}_{\text{feat}}$$

where:
- $\mathcal{L}_{\text{task}}$ is current task loss
- $\mathcal{L}_{\text{output}}$ is output distillation loss
- $\mathcal{L}_{\text{feat}}$ is feature distillation loss

## Feature Alignment Strategies

### Direct L2 Matching (Same Dimensions)

When feature dimensions match naturally:

$$\mathcal{L} = \|f_t^{(\ell)} - f_{t-1}^{(\ell)}\|_F^2$$

Works when both models have identical architecture.

### Projection-Based Alignment (Different Dimensions)

!!! tip "Handling Architecture Changes"
    When $d_t^{(\ell)} \neq d_{t-1}^{(\ell)}$, insert learnable projections:

$$\mathcal{L} = \|P_t(f_t^{(\ell)}) - P_{t-1}(f_{t-1}^{(\ell)})\|_F^2$$

where:
- $P_t: \mathbb{R}^{d_t^{(\ell)}} \to \mathbb{R}^{d_{\text{shared}}}$ projects current features
- $P_{t-1}: \mathbb{R}^{d_{t-1}^{(\ell)}} \to \mathbb{R}^{d_{\text{shared}}}$ projects previous features

Both projects to shared dimension $d_{\text{shared}}$.

### Similarity-Based Alignment

Instead of matching raw features, match feature similarity structures:

$$\mathcal{L} = \|\text{Sim}(f_t^{(\ell)}) - \text{Sim}(f_{t-1}^{(\ell)})\|_F^2$$

where similarity matrix:

$$\text{Sim}(\mathbf{F})_{ij} = \frac{\mathbf{F}_i \cdot \mathbf{F}_j}{\|\mathbf{F}_i\| \|\mathbf{F}_j\|}$$

This matches feature relationships rather than absolute values.

## Layer Selection Strategy

### Which Layers to Match?

Different layer depths serve different purposes:

| Layer Depth | Features | Matching Value | Cost |
|-------------|----------|----------------|------|
| **Early** | Low-level patterns | Low | Low |
| **Middle** | Task-relevant abstractions | High | Medium |
| **Late** | High-level concepts | Very High | High |

**Recommendation**: Match 2-3 middle-to-late layers capturing semantic features.

### Adaptive Layer Selection

Weight layer matching by importance:

$$\alpha_{\ell} = \frac{\text{gradient\_norm}(\mathcal{L}_{\text{task}}, \theta^{(\ell)})}{\sum_{\ell'} \text{gradient\_norm}(\mathcal{L}_{\text{task}}, \theta^{(\ell')})}$$

Emphasize layers critical to task performance.

## Handling Feature Scale Differences

### Normalization Strategies

Features across models may have different scales:

**Instance Normalization**:
$$\tilde{\mathbf{f}} = \frac{\mathbf{f} - \mathbb{E}[\mathbf{f}]}{\sqrt{\text{Var}(\mathbf{f}) + \epsilon}}$$

Normalize each feature sample independently.

**Layer Normalization**:
$$\tilde{\mathbf{f}} = \frac{\mathbf{f} - \mathbb{E}[\mathbf{f}]}{\sqrt{\text{Var}(\mathbf{f}) + \epsilon}}$$

Normalize across features for each sample.

**No Normalization**: Use projection layers to learn appropriate scaling.

## Computational Considerations

### Memory and Computation Cost

Storing intermediate features from both models:

$$\text{Memory} = \sum_{\ell} 2 \times d_{\ell} \times \text{batch\_size}$$

Selecting only key layers reduces overhead.

### Efficiency Optimizations

**Selective Matching**: Only compute distillation loss on subset of training data

$$\mathcal{L}_{\text{feat}} = \frac{1}{|S|} \sum_{\mathbf{x} \in S} \|\cdots\|^2$$

where $S$ is sampled subset of training data.

**Feature Caching**: Precompute previous model features, store offline

!!! note "Practical Trade-off"
    Cache previous model features during training to avoid redundant computation.

## Continual Learning Dynamics

### Task Sequence Learning

```
Task 1: Train f_1 on D_1
        Store f_1

Task 2: Initialize f_2 with f_1 or random
        Train f_2 on D_2 with distillation:
        L = L_task(f_2, D_2) + λ·L_feat(f_2, f_1, D_1_subset)
        f_2 learns D_2 while preserving f_1's features

Task 3: Initialize f_3 with f_2 or random
        Train f_3 on D_3 with distillation from f_2
        ...
```

### Feature Degradation Over Long Sequences

!!! warning "Sequential Feature Drift"
    In long task sequences, features may drift cumulatively:
    
    Task 1 → Task 2 → Task 3 → ... → Task T
    
    Features gradually diverge from original Task 1 representation.

**Mitigation**: 
- Periodically consolidate learned features
- Use stronger distillation on early tasks
- Store "anchor" features from early tasks, reference them in late tasks

## Application Examples

### Parameter-Efficient Fine-Tuning

Use feature distillation with adapters for efficiency:

```
Base Model (Frozen)
    ↓
Shared Features
    ↓
Task Adapter ← Feature Distillation Loss
    ↓
Task Output
```

Adapters learn task-specific transformations while features match previous knowledge.

### Cross-Task Feature Sharing

Match features that transfer across tasks:

**Task 1**: Predict asset returns using technical features

**Task 2**: Predict asset volatility using features from Task 1

**Distillation**: Share common features (price momentum, volatility regimes) across tasks

## Comparison with Output Distillation

| Aspect | Output | Feature | Combined |
|--------|--------|---------|----------|
| **Information** | Final decisions | Intermediate knowledge | Comprehensive |
| **Flexibility** | High | Requires matching | Very High |
| **Computational** | Low | Medium | Higher |
| **Preserves** | Decision boundaries | Learned hierarchy | Both |

## Financial Implementation

!!! warning "Feature Distillation in Trading"
    
    **Task Sequence**:
    1. **Task 1**: Predict equity returns using price/volume features
    2. **Task 2**: Predict bond spreads, distill equity price momentum features
    3. **Task 3**: Predict FX, distill volatility features from Task 1-2
    
    Progressive feature accumulation across asset classes.

### Multi-Horizon Forecasting

Match features across prediction horizons:

- **1-Day Forecast**: High-frequency patterns, microstructure features
- **5-Day Forecast**: Distill 1-day feature hierarchy
- **20-Day Forecast**: Distill 5-day features
- **252-Day Forecast**: Distill longer-term trend features

## Research Directions

- Optimal layer selection for feature matching
- Theoretically-grounded feature distillation objective design
- Hierarchical feature distillation across multiple scales
- Adaptive weighting of feature distillation in continual learning

## Related Topics

- Feature Distillation Overview (Chapter 12.2)
- Knowledge Distillation Basics (Chapter 9.2.1)
- Architecture-Based Continual Learning (Chapter 12.1)
- Representation Learning
