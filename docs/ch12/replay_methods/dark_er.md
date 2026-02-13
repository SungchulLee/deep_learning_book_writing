# Dark Experience Replay

## Introduction

Dark Experience Replay (DER) extends experience replay by learning task-specific weighting schemes for replayed data, allowing models to emphasize exemplars most relevant to current learning objectives. Rather than uniform or random replay, DER dynamically adjusts replay priorities based on which historical examples provide the most valuable learning signal for new tasks.

In quantitative finance, dark experience replay enables intelligent historical data selection: not all past market conditions are equally relevant to current learning. By learning to weight historical scenarios, models can emphasize crisis patterns when volatility spikes, focus on normal regimes during calm markets, and weight seasonal patterns appropriately.

## Key Concepts

- **Learned Weighting**: Trainable importance weights for replayed exemplars
- **Task-Specific Priorities**: Different tasks prioritize different exemplars
- **Gradient Alignment**: Emphasize exemplars with positive gradient alignment
- **Soft Prioritization**: Continuous weighting rather than hard selection
- **Online Adaptation**: Update weights as learning progresses
- **Meta-Learning Connection**: Learn to select useful examples

## Mathematical Framework

### Standard Replay (Baseline)

Uniform weighting of all exemplars:

$$\mathcal{L} = \mathbb{E}_{(x,y) \sim \mathcal{D}_t}[\ell(\theta, x, y)] + \frac{1}{|\mathcal{M}|} \sum_{(x',y') \in \mathcal{M}} \ell(\theta, x', y')$$

All exemplars contribute equally regardless of relevance.

### Dark Experience Replay

Learn exemplar-specific weights:

$$\mathcal{L}_{\text{DER}} = \mathbb{E}_{(x,y) \sim \mathcal{D}_t}[\ell(\theta, x, y)] + \frac{1}{|\mathcal{M}|} \sum_{(x',y') \in \mathcal{M}} w(x', y'; \theta) \cdot \ell(\theta, x', y')$$

where $w(x', y'; \theta) \in [0, 1]$ are learned weights.

### Weight Function

Implement weights through neural network:

$$w(x', y'; \theta) = \sigma(\text{MLP}([\phi(\mathbf{x}'), \phi(\mathbf{x})_{\text{current}}]))$$

where:
- $\phi(\cdot)$ are learned feature representations
- $\sigma$ is sigmoid activation
- MLP learns to compare exemplar with current data

## Learning Weight Parameters

### Weight Loss Component

Jointly optimize task loss and weight quality:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{weight}}$$

### Approaches to Learning Weights

#### Gradient Alignment Weighting

!!! tip "Alignment-Based Weights"
    Weight exemplar by alignment with current task gradient:

$$w_i = \max(0, \frac{g_i \cdot g_t}{\|g_i\| \|g_t\|})$$

where $g_i$ is gradient on exemplar $i$ and $g_t$ is current task gradient.

High-weight exemplars have gradients aligned with current learning.

#### Metadata-Based Weighting

Learn weights from exemplar properties:

$$w_i = \sigma([\text{features}(x_i), \text{features}(x_t)]))$$

Network learns which exemplar properties (age, task type, difficulty) matter for weighting.

#### Loss-Based Weighting

Use exemplar loss to determine weight:

$$w_i = \sigma(\alpha \cdot \ell(x_i) - \beta)$$

Exemplars with higher loss receive higher weight (focus on difficult examples).

## Algorithm: Dark Experience Replay

### Training Procedure

```
procedure DarkER(tasks, memory_size M):
    θ ← random initialization
    w_network ← random initialization
    buffer ← empty
    
    for each task t:
        for epoch in 1 to num_epochs:
            for minibatch (x, y) from task_t:
                L_task = L(θ, x, y)  // new task loss
                
                if buffer not empty:
                    for (x', y') in buffer:
                        w_i = w_network(x', x, θ)
                        L_replay += w_i * L(θ, x', y')
                    
                    L_total = L_task + L_replay
                    
                    // Update both θ and w_network
                    θ ← θ - α ∇_θ L_total
                    w_network ← update_weights(w_network, ∇ L_total)
                else:
                    θ ← θ - α ∇_θ L_task
        
        buffer ← update_exemplars(buffer, task_t, M)
    
    return θ, buffer
```

### Weight Network Architecture

Simple weight prediction network:

```
[Exemplar Features]
        ↓
   [MLP Layer 1] (shared)
        ↓
   [MLP Layer 2] (shared)
        ↓
   [Weight Output] (sigmoid) → [0,1]
```

Minimal architecture: 1-2 hidden layers, < 5% of main model size.

## Preventing Weight Degeneracy

### Weight Distribution Regularization

Prevent all weights from collapsing to 0 or 1:

$$\mathcal{L}_{\text{entropy}} = -\sum_i [w_i \log w_i + (1-w_i) \log(1-w_i)]$$

Encourages diversity in learned weights.

### Min-Max Weight Constraints

Enforce minimum and maximum weight values:

$$w_i \in [\epsilon, 1-\epsilon]$$

where $\epsilon \approx 0.01$ prevents degenerate solutions.

## Comparison with A-GEM and Standard Replay

| Method | Flexibility | Computation | Memory | Forgetting |
|--------|------------|-------------|--------|-----------|
| **Standard Replay** | Low | Low | High | Very Low |
| **A-GEM** | Low | Low | Low | Very Low |
| **Dark ER** | High | Medium | Medium | Very Low |

Dark ER trades computational cost for flexibility in exemplar selection.

## Financial Applications

### Dynamic Scenario Weighting

For portfolio optimization with historical scenarios:

**Scenario 1**: 2008 Financial Crisis
**Scenario 2**: 2020 COVID Crash  
**Scenario 3**: Normal market
**Scenario 4**: High volatility regime

Weight network learns:
- Heavy weight on crash scenarios in volatile periods
- Lower weight on crash scenarios during calm markets
- Dynamic adjustment as market conditions change

### Time-Series Example

```
Task 1: Predict returns with 10-year history
  Store: Historical data from past 20 years
  Learn: Which historical periods similar to current?

Task 2: Predict in new market regime
  Weight Network: Recent history → high weight
                  Similar regimes → high weight
                  Different regimes → low weight
  Result: Intelligent focus on relevant historical data
```

## Practical Implementation Details

### Weight Network Initialization

**Option 1**: Uniform weights initially (all exemplars equally important)

$$w_i(0) = 0.5$$

**Option 2**: Implicit weighting from distance

$$w_i(0) = \exp(-d(\mathbf{x}_i, \mathbf{x}_{\text{current}}))$$

### Computational Efficiency

Dark ER adds overhead for weight computation:

**Per Exemplar**:
- Feature extraction: $O(d)$
- Weight network forward: $O(d^2)$ (for small network)
- Total: $\sim 10-20\%$ overhead per exemplar

With buffer size 1000, total overhead $\sim 50-100\%$.

### Batch-Level Weight Updates

Instead of updating weights every exemplar, batch them:

```
accumulate_exemplar_gradients(batch_size=64)
update_weights_once_per_batch
```

Reduces communication overhead.

## Advanced Techniques

### Meta-Weighted Experience Replay

Use meta-learning to optimize weight function:

$$w^* = \arg\min_w \sum_t \mathcal{L}(\text{task}_t | \text{weights}_w)$$

Learn weight function across task distribution.

### Hierarchical Weight Functions

Different weight networks per task type:

```
Weight Network 1: Equity tasks
Weight Network 2: Bond tasks
Weight Network 3: Cross-asset tasks
```

Each specializes in its domain.

### Adaptive Replay Ratio

Dynamically adjust ratio of replay vs. new data:

$$\text{replay\_ratio}(t) = f(\text{task\_difficulty}(t))$$

More difficult tasks use higher replay ratio.

## Theoretical Properties

### Information-Theoretic View

Dark ER learns importance weights through information bottleneck:

$$w_i = p(x_i \text{ useful} | \text{current task})$$

Maximizes mutual information between exemplar features and current task performance.

### Convergence Analysis

With learned weights, convergence somewhat slower than fixed strategy:

$$\mathbb{E}[\mathcal{L}(T)] \leq \mathcal{L}(0) - cT + O(\text{weight error} + \sigma^2/\sqrt{T})$$

Weight learning error adds bias to convergence.

## Research Frontiers

- Optimal architecture for weight networks
- Theoretical analysis of weight learning
- Combining learned weights with gradient constraints (DER + A-GEM)
- Application to non-vision domains

## Related Topics

- Replay Methods Overview (Chapter 12.4.1)
- Averaged Gradient Episodic Memory (Chapter 12.4.2)
- Importance Weighting in Machine Learning
- Meta-Learning for Sample Selection
