# Overview of Distillation Methods for Continual Learning

## Introduction

Distillation-based continual learning leverages knowledge from previous tasks to regularize learning on new tasks, preventing catastrophic forgetting through soft target constraints. By distilling knowledge from models trained on previous tasks into models learning new ones, practitioners maintain performance across task sequences while avoiding explicit experience replay or architectural constraints.

In quantitative finance, distillation-based continual learning enables graceful degradation when market regimes change: models trained on previous regime data continue providing valuable guidance through distilled knowledge while adapting to new patterns. This approach balances between maintaining historical market relationships and learning contemporary dynamics.

## Key Concepts

- **Knowledge Distillation**: Transfer learned representations between models
- **Teacher-Student**: Previous task model guides current task learning
- **Soft Targets**: Probability distributions rather than hard labels
- **Feature Distillation**: Matching intermediate representations
- **Regularization**: Distillation loss acts as regularizer preventing forgetting
- **Task Sequence**: Sequential task learning with continual adaptation

## Distillation-Based Continual Learning Framework

### Basic Setup

For task sequence $\{1, 2, \ldots, T\}$:

**Task $t$**: 
- Previous model $f_{t-1}$ trained on tasks $\{1, \ldots, t-1\}$
- Current model $f_t$ learning task $t$
- Distillation transfers knowledge from $f_{t-1}$ to regularize $f_t$

### Training Objective

Combine task-specific loss with distillation loss:

$$\mathcal{L}_t = \mathcal{L}_{\text{task}}(f_t(\mathbf{x}), y) + \lambda \mathcal{L}_{\text{distill}}(f_t(\mathbf{x}), f_{t-1}(\mathbf{x}))$$

where:
- $\mathcal{L}_{\text{task}}$ optimizes current task performance
- $\mathcal{L}_{\text{distill}}$ constrains previous task knowledge
- $\lambda$ balances contributions

## Distillation Loss Variants

### Response-Based Distillation

Match final output distributions:

$$\mathcal{L}_{\text{response}} = \text{KL}(p_{t-1}(\mathbf{x}) \| p_t(\mathbf{x}))$$

where softened probabilities use temperature $T$:

$$p(\mathbf{x}) = \text{softmax}(\text{logits}/T)$$

**Pros**: Works with any architecture

**Cons**: Only constrains final decisions, not intermediate features

### Feature-Based Distillation

Match intermediate layer representations:

$$\mathcal{L}_{\text{feature}} = \sum_{\ell} \|f_t^{(\ell)}(\mathbf{x}) - f_{t-1}^{(\ell)}(\mathbf{x})\|_2^2$$

where $f^{(\ell)}$ are layer $\ell$ features.

**Pros**: Richer supervision, preserves learned representations

**Cons**: Requires architectural compatibility

!!! tip "Feature Dimension Matching"
    When feature dimensions differ, insert projection layers:
    
    $$\mathcal{L}_{\text{feature}} = \|P(f_t(\mathbf{x})) - f_{t-1}(\mathbf{x})\|^2$$

### Attention Transfer

Transfer attention maps (spatial/channel importance):

$$\mathcal{L}_{\text{attention}} = \sum_{\ell} \|A_t^{(\ell)}(\mathbf{x}) - A_{t-1}^{(\ell)}(\mathbf{x})\|_F^2$$

where $A^{(\ell)} = \text{normalize}(|f^{(\ell)}|)$ are attention maps.

## Architecture Considerations

### Shared vs. Task-Specific Networks

**Shared Network**: Single network learns all tasks, distillation prevents forgetting

$$f_{\text{all}}(\mathbf{x}) = \text{Shared Features} + \text{Task Head}_t$$

**Task-Specific**: Separate networks per task, distilled features guide new task

$$f_t(\mathbf{x}) = \text{New Network} \text{ regularized by } f_{t-1}$$

**Hybrid**: Shared backbone with task-specific heads

```
Shared Features
      ↓
Distillation ←──────────
      ↓                 │
Task 1 Head    Task 2 Head
      ↓                 ↓
Output 1       Output 2
```

## Learning Dynamics in Continual Settings

### Task Interference

When learning task $t$, model risks losing performance on $\{1, \ldots, t-1\}$.

Distillation loss constrains:

$$\mathbb{E}_{\mathbf{x} \sim D_{t-1}}[\text{KL}(p_{t-1}(\mathbf{x}) \| p_t(\mathbf{x}))] \leq \epsilon$$

preventing excessive drift from previous task performance.

### Forward-Backward Tradeoff

!!! warning "Adaptation-Stability Trade-off"
    Increasing $\lambda$ (distillation weight):
    - Improves stability on previous tasks
    - Reduces plasticity on new tasks
    - Can prevent learning new task effectively

Requires careful tuning based on task similarity.

### Annealing Distillation Weight

Adapt regularization strength during training:

$$\lambda(t) = \lambda_0 \cdot \exp(-t / \tau)$$

Start with strong distillation (stabilize previous knowledge), gradually reduce to allow new learning.

## Comparison with Related Approaches

| Method | Mechanism | Forgetting | Compute | Data Needed |
|--------|-----------|-----------|--------|------------|
| **Distillation** | Soft targets | Moderate | Low | New + Old output |
| **EWC** | Parameter penalty | Low | Low | New only |
| **Replay** | Data storage | Very Low | High | New + Stored old |
| **Adapters** | Architecture | Minimal | Medium | New only |

## Financial Applications

!!! warning "Market Regime Distillation"
    
    When market regime changes:
    1. Previous regime model $f_{t-1}$ provides distillation targets
    2. New regime model $f_t$ learns current patterns
    3. Distillation maintains historical regime knowledge
    4. Model gracefully adapts to new conditions
    5. Can switch back to $f_{t-1}$ if regime repeats

### Credit Risk Model Evolution

As credit markets change:

**Previous Model**: Trained on pre-2020 credit data

**New Model**: Learns COVID-era credit dynamics

**Distillation**: New model maintains previous credit relationships while learning new patterns

## Hyperparameter Selection

| Parameter | Role | Typical Range |
|-----------|------|---------------|
| **$\lambda$** | Distillation weight | [0.1, 2.0] |
| **$T$** | Temperature | [1, 10] |
| **Layers** | Feature matching layers | [1, 4] |
| **$\alpha$ (LR)** | Learning rate | [0.001, 0.01] |

!!! note "Task-Similarity Based Tuning"
    Increase $\lambda$ for similar sequential tasks; decrease for diverse tasks.

## Advanced Topics

### Multi-Teacher Distillation

Distill from multiple previous task models:

$$\mathcal{L}_{\text{multi}} = \sum_{t'=1}^{t-1} w_{t'} \mathcal{L}_{\text{distill}}(f_t, f_{t'})$$

Weight teachers by relevance to current task.

### Bidirectional Distillation

Previous model also benefits from new task learning:

$$f_{t-1} \leftarrow \text{Distill}(f_t \to f_{t-1})$$

Improves old model with new patterns while maintaining stability.

## Research Directions

- Optimal distillation loss design for continual learning
- Automatic weighting of historical vs. current task objectives
- Combining distillation with other continual learning methods
- Theoretical analysis of distillation's regularization properties

## Related Topics

- Feature Distillation (Chapter 12.2.2)
- Knowledge Distillation Basics (Chapter 9.2.1)
- Regularization-Based Methods (Chapter 12.3)
- Self-Distillation (Chapter 9.2)
