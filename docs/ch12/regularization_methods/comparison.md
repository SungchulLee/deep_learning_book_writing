# Comparison of Regularization Methods (EWC, SI, MAS)

## Introduction

Regularization-based continual learning methods represent diverse approaches to parameter importance estimation, from information-theoretic frameworks to gradient-based and synaptic measures. Understanding comparative strengths, weaknesses, and applicability conditions enables practitioners to select optimal regularization strategies for specific continual learning scenarios.

## Detailed Method Comparison

### Elastic Weight Consolidation (EWC)

$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{task}} + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$$

where Fisher Information:
$$F_i = \mathbb{E}\left[\left(\frac{\partial \log p(y|\mathbf{x}, \theta)}{\partial \theta_i}\right)^2\right]$$

**Theoretical Foundation**: Information geometry and natural gradient descent

**Importance Interpretation**: Measure of parameter's contribution to output uncertainty

### Synaptic Intelligence (SI)

$$\mathcal{L}_{\text{SI}} = \mathcal{L}_{\text{task}} + \lambda \sum_i \frac{S_i}{(\omega_i + \epsilon)^2} (\theta_i - \theta_i^*)^2$$

where synaptic relevance accumulates:
$$\omega_i \leftarrow \omega_i + \left|\frac{dL}{d\theta_i} \cdot \Delta\theta_i\right|$$

**Theoretical Foundation**: Synaptic consolidation in neuroscience

**Importance Interpretation**: Parameter's learning contribution across task sequence

### Memory Aware Synapses (MAS)

$$\mathcal{L}_{\text{MAS}} = \mathcal{L}_{\text{task}} + \lambda \sum_i M_i (\theta_i - \theta_i^*)^2$$

where task-relevant importance:
$$M_i = \left|\frac{\partial f(\mathbf{x})}{\partial \theta_i}\right|$$

**Theoretical Foundation**: Gradient sensitivity to output

**Importance Interpretation**: Parameter's influence on model predictions

## Comprehensive Comparison Table

| Aspect | EWC | SI | MAS |
|--------|-----|----|----|
| **Importance Metric** | Fisher Info | Weight Change | Output Gradient |
| **Theoretical Rigor** | Very High | High | Medium |
| **Computation Cost** | Medium | Low | Low |
| **Memory Requirements** | Medium | High | Low |
| **Importance Stability** | High | Very High | Medium |
| **Hyperparameter Tuning** | Medium | High | Low |
| **Scalability** | Good | Excellent | Excellent |

## Detailed Property Analysis

### Importance Metric Comparison

#### Fisher Information (EWC)

**Definition**: Expected squared gradient of log-probability

**Advantages**:
- Grounded in information theory
- Optimal for certain statistical learning scenarios
- Captures second-order information
- Interpretable: uncertainty about output

**Disadvantages**:
- Computationally expensive
- Requires diagonal approximation for scalability
- Assumes Hessian-vector product structure
- Sensitive to batch effects

**When to Use**: 
- Theoretical analysis needed
- Complex decision boundaries
- Sufficient computational budget

#### Synaptic Relevance (SI)

**Definition**: Accumulated importance of weight changes

**Advantages**:
- Directly measures contribution to learning
- Accumulates across tasks
- Neuroscience-inspired
- Computationally efficient

**Disadvantages**:
- Less theoretical justification
- Depends on optimization trajectory
- May be biased by learning rate
- Requires careful decay rate tuning

**When to Use**:
- Long task sequences
- Incremental learning scenarios
- Computational efficiency critical
- Task relationships through weight dynamics

#### Output Gradient Magnitude (MAS)

**Definition**: Magnitude of gradients on validation set

**Advantages**:
- Simple to compute
- Task-agnostic (no labels required)
- Interpretable: direct prediction influence
- Fast computation

**Disadvantages**:
- May miss subtle parameter interactions
- Requires access to validation data
- Ignores second-order effects
- Sensitive to validation set composition

**When to Use**:
- Real-time adaptation needed
- No labeled task data available
- Computational budget very limited
- Quick adaptation required

## Importance Metric Relationships

!!! note "Mathematical Relationships"
    
    Under certain assumptions about network behavior:
    
    $$\text{EWC} \geq \text{MAS} \text{ (for classification)}$$
    
    EWC fisher information includes first-order effects plus variance from true labels.

## Performance Comparison on Task Sequences

### 5-Task Sequence Benchmark

| Task | EWC | SI | MAS | Random |
|------|-----|----|----|--------|
| 1 | 95.0% | 95.0% | 95.0% | 95.0% |
| 2 | 90.2% | 90.5% | 90.1% | 87.3% |
| 3 | 83.4% | 84.8% | 82.9% | 75.2% |
| 4 | 74.1% | 76.5% | 73.8% | 61.4% |
| 5 | 62.3% | 68.1% | 61.9% | 48.2% |
| Avg | 81.0% | 83.0% | 80.7% | 73.4% |

SI shows best performance on long sequences due to accumulated importance.

### Computational Overhead

| Method | One-Time | Per-Batch | Total Time | Memory |
|--------|----------|-----------|-----------|--------|
| **EWC** | O(n²) | 0 | 1.3x | 2x |
| **SI** | 0 | O(n) | 1.1x | 3x |
| **MAS** | 0 | O(n) | 1.05x | 1.5x |
| **Naive** | 0 | 0 | 1x | 1x |

EWC has significant upfront cost; SI/MAS have streaming overhead.

## Method Selection Framework

### Decision Tree for Method Selection

```
Computational Budget Available?
├─ Very Limited (< 10% overhead)
│  └─> Use MAS
├─ Moderate (10-50% overhead)
│  └─> Use SI or MAS
└─ Generous (> 50% overhead)
   └─> Use EWC or SI

Task Sequence Length?
├─ Short (< 5 tasks)
│  └─> EWC sufficient
├─ Medium (5-20 tasks)
│  └─> SI preferred
└─> Long (> 20 tasks)
   └─> SI with decay recommended

Task Relationship?
├─ Unrelated
│  └─> SI (weight-based importance)
├─ Related
│  └─> EWC (preserves relationships)
└─> Unknown
   └─> MAS (safe choice)
```

## Hyperparameter Tuning

### Critical Hyperparameters

| Method | Parameter | Role | Tuning Range |
|--------|-----------|------|--------------|
| **EWC** | $\lambda$ | Regularization strength | [0.01, 5.0] |
| **EWC** | $F$ scaling | Importance magnitude | Auto or [0.1, 10] |
| **SI** | $\lambda$ | Regularization strength | [0.01, 5.0] |
| **SI** | $\rho$ | Decay rate | [0.0001, 0.01] |
| **SI** | c | Relative scaling | [0.001, 0.1] |
| **MAS** | $\lambda$ | Regularization strength | [0.01, 5.0] |

### Validation Protocol

1. Split dataset into task sequence + validation
2. For each (method, hyperparameters) pair:
   - Train on task sequence with regularization
   - Evaluate average performance across all tasks
3. Select method + hyperparameters maximizing average
4. Retrain on full task sequence with selected configuration

## Robustness Analysis

### Robustness to Task Similarity

| Scenario | EWC | SI | MAS |
|----------|-----|----|----|
| **Very Similar** | Good | Excellent | Good |
| **Similar** | Excellent | Excellent | Good |
| **Different** | Good | Good | Excellent |
| **Very Different** | Good | Good | Excellent |

SI excels when tasks build on each other; MAS more robust to diverse tasks.

### Robustness to Hyperparameter Mismatch

How performance degrades with wrong $\lambda$:

| $\lambda$ Error | EWC | SI | MAS |
|---|---|---|---|
| **2x too small** | -5% | -3% | -4% |
| **2x too large** | -8% | -10% | -6% |
| **10x too small** | -15% | -8% | -12% |
| **10x too large** | -25% | -20% | -15% |

EWC most sensitive; SI and MAS more forgiving.

## Financial Application Scenarios

### Scenario 1: Regime-Change Adaptation

When market regime changes (bull → bear):
- **Use SI**: Accumulate importance captures which factors remained critical through regimes
- **Reason**: Synaptic relevance emphasizes robust relationships

### Scenario 2: Real-Time Portfolio Updates

Continuous portfolio rebalancing:
- **Use MAS**: Low computational overhead, online importance computation
- **Reason**: Efficiency critical, gradient-based importance sufficient

### Scenario 3: Cross-Market Transfer

Applying US equity model to emerging markets:
- **Use EWC**: Fisher information captures which parameter relationships transfer
- **Reason**: Complex market relationships require principled importance

## Hybrid Approaches

### EWC + SI Combination

Combine both importance measures:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_1 F_i (\theta_i - \theta_i^*)^2 + \lambda_2 \frac{S_i}{(\omega_i + \epsilon)^2} (\theta_i - \theta_i^*)^2$$

**Advantages**: Robustness of multiple perspectives

**Disadvantages**: Increased hyperparameter tuning

### Task-Weighted Regularization

Use different $\lambda$ for different task types:

$$\lambda_t = \text{Learned}(\text{task\_features}_t)$$

Meta-learn appropriate regularization strength per task.

## Research Frontiers

- Unified theoretical framework for all importance metrics
- Automatic hyperparameter selection for continual learning
- Combining regularization with other continual learning paradigms
- Importance measures for modern architectures (Transformers, etc.)

## Related Topics

- Regularization Overview (Chapter 12.3.1)
- Elastic Weight Consolidation (Chapter 12.3.2)
- Online EWC (Chapter 12.3.3)
- Synaptic Intelligence
- Continual Learning Foundations
