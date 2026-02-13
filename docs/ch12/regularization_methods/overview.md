# Overview of Regularization-Based Continual Learning

## Introduction

Regularization-based continual learning prevents catastrophic forgetting by constraining updates to important parameters during new task learning. Rather than architectural modifications or explicit replay, these methods identify which parameters were critical to previous tasks, then penalize large changes to those parameters during new task training.

In quantitative finance, regularization-based methods offer elegant solutions: market microstructure knowledge (order flow patterns, volatility regimes) often remains stable across time, and selectively protecting these parameters while freely updating others enables rapid adaptation to new instruments or markets. This approach maintains knowledge about fundamental market relationships while learning contemporary patterns.

## Key Concepts

- **Parameter Importance**: Measure which parameters contributed most to previous task performance
- **Fisher Information**: Information-theoretic measure of parameter importance
- **Elastic Weight Consolidation**: Quadratic penalty on parameter changes weighted by importance
- **Synaptic Importance**: Alternative importance measures based on weight changes
- **Sparse Gradients**: Identify critical parameters through gradient analysis
- **Sequential Learning**: Learn tasks in sequence with regularization constraints

## Fundamental Framework

### Regularization Principle

Constrain task $t$ learning to preserve previous task performance:

$$\mathcal{L}_t = \mathcal{L}_{\text{task}_t}(\theta) + \lambda \sum_i \Omega_i (\theta_i - \theta_i^*)$$

where:
- $\mathcal{L}_{\text{task}_t}$ is current task loss
- $\Omega_i$ measures importance of parameter $i$
- $\theta_i^*$ is previously learned parameter value
- $\lambda$ controls regularization strength

### Parameter Importance Estimation

Key challenge: Which parameters matter for previous tasks?

**Fisher Information** (information-theoretic):
$$F_i = \mathbb{E}[(\partial \log p(y|\mathbf{x}, \theta))^2 / \partial \theta_i]$$

**Gradient Magnitude** (empirical):
$$G_i = |\partial \mathcal{L} / \partial \theta_i|$$

**Weight Change** (synaptic):
$$S_i = |\Delta \theta_i|$$

## Regularization Method Taxonomy

### Elastic Weight Consolidation (EWC)

$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{task}_t} + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$$

- Uses Fisher information as importance
- Quadratic penalty on parameter changes
- Computationally efficient (diagonal Fisher approximation)

### Synaptic Intelligence

$$\mathcal{L}_{\text{SI}} = \mathcal{L}_{\text{task}_t} + \lambda \sum_i \frac{S_i}{(\omega_i + \epsilon)^2} (\theta_i - \theta_i^*)^2$$

- Importance based on synaptic relevance
- Accumulates importance across task sequence
- Small parameter changes affect learned importance

### Memory Aware Synapses (MAS)

$$\mathcal{L}_{\text{MAS}} = \mathcal{L}_{\text{task}_t} + \lambda \sum_i M_i (\theta_i - \theta_i^*)^2$$

where:
$$M_i = \left| \frac{\partial f(\mathbf{x})}{\partial \theta_i} \right|$$

- Directly uses output gradient magnitude
- More computationally tractable than Fisher
- Task-agnostic importance estimation

## Comparative Analysis

| Method | Importance Metric | Computation | Accuracy | Scalability |
|--------|------------------|-----------|----------|------------|
| **EWC** | Fisher Info | Medium | High | Good |
| **Synaptic** | Weight Change | Low | Medium | Excellent |
| **MAS** | Output Gradient | Low | High | Good |
| **Simple Penalty** | Uniform | None | Low | Excellent |

!!! note "Trade-offs"
    Sophisticated importance estimation improves performance but increases computational cost and hyperparameter tuning complexity.

## Advantages and Limitations

### Advantages

**Theoretical Grounding**: Information-theoretic foundations for parameter importance

**Interpretability**: Identify which parameters support which tasks

**Flexibility**: Compatible with any optimizer and architecture

**Simplicity**: Straightforward implementation

### Limitations

!!! warning "Regularization-Based Challenges"
    
    - **Parameter Importance Estimation**: Inaccurate importance leads to poor performance
    - **Task Sequence**: Performance degrades with many tasks (compounding errors)
    - **Task Diversity**: Similar tasks may have overlapping importance, causing conflicts
    - **Interference**: Highly connected parameters may be important to multiple tasks

## Information-Theoretic Foundation

### Fisher Information Matrix

The Fisher Information Matrix (FIM) provides optimal importance measure:

$$F = \mathbb{E}_{y \sim p(y|\mathbf{x}, \theta)} \left[ \nabla_\theta \log p(y|\mathbf{x}, \theta) \nabla_\theta \log p(y|\mathbf{x}, \theta)^T \right]$$

For neural networks:
$$F_i \approx \mathbb{E}[\frac{\partial \mathcal{L}}{\partial \theta_i}]^2$$

High Fisher information indicates parameter significantly affects predictions.

### Diagonal Approximation

Full FIM is prohibitively expensive (dimension²). Diagonal approximation:

$$F_i \approx \mathbb{E}\left[\left(\frac{\partial \mathcal{L}}{\partial \theta_i}\right)^2\right]$$

assumes parameter changes are independent (ignores parameter correlations).

## Training Dynamics

### Task Learning with Regularization

```
for each task t:
    Compute importance Ω_{t-1} from previous task
    Initialize θ from θ_{t-1}
    for each epoch:
        for each batch (x, y) from task t:
            L_task = task_loss(θ, x, y)
            L_reg = λ * Σ_i Ω_i (θ_i - θ_{i-1}*)²
            L_total = L_task + L_reg
            θ ← θ - α ∇L_total
    θ_t* ← θ (store learned parameters)
    Ω_t ← compute_importance(θ)
```

### Stability-Plasticity Dilemma

!!! warning "Stability vs. Learning"
    Increasing $\lambda$ (regularization strength):
    - Improves stability (preserves previous tasks)
    - Reduces plasticity (harder to learn new tasks)
    - Optimal $\lambda$ task and data-dependent

## Financial Applications

**Market Microstructure**: Protect parameters learning order book dynamics

**Volatility Regimes**: Regularize parameters capturing regime-specific volatility

**Cross-Asset Relationships**: Preserve correlations while learning asset-specific patterns

### Multi-Asset Continual Learning

1. Train on Asset 1 (equities), compute importance
2. Learn Asset 2 (bonds) with regularization protecting Asset 1 knowledge
3. Learn Asset 3 (commodities) protecting both previous assets
4. Maintain cross-asset relationships through regularization

## Hyperparameter Selection

| Parameter | Role | Tuning |
|-----------|------|--------|
| **$\lambda$** | Regularization strength | Increase if forgetting, decrease if slow learning |
| **Epochs** | Per-task training | Longer training for complex tasks |
| **Importance Scale** | Normalization factor | Prevent overflow in early layers |

### Cross-Validation for $\lambda$

```
for λ in [0.1, 0.5, 1.0, 2.0, 5.0]:
    for t in 1 to T_tasks:
        Train task t with regularization λ
        Evaluate on all tasks 1:t
    Score[λ] = weighted average accuracy
λ* = argmax Score[λ]
```

## Advanced Extensions

### Continual Learning with Shared and Task-Specific Parameters

Separate parameters into:
- **Shared**: Used by all tasks (protect strongly)
- **Task-Specific**: Used by single task (flexible)

Different regularization for each:
$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_s \text{Reg}_{\text{shared}} + \lambda_t \text{Reg}_{\text{task-specific}}$$

### Online EWC

Update importance estimates during task training, not just after:

$$F_i^{(t)} \leftarrow (1-\rho) F_i^{(t-1)} + \rho \left(\frac{\partial \mathcal{L}}{\partial \theta_i}\right)^2$$

Smoother importance evolution, better adaptation.

## Research Directions

- Improved importance estimation for parameters
- Combining multiple importance metrics
- Scalable Fisher information computation for modern networks
- Theoretical analysis of regularization-based continual learning

## Related Topics

- Elastic Weight Consolidation (Chapter 12.3.2)
- Online EWC (Chapter 12.3.3)
- Replay-Based Methods (Chapter 12.4)
- Architecture-Based Methods (Chapter 12.1)
