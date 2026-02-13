# Averaged Gradient Episodic Memory (A-GEM)

## Introduction

Averaged Gradient Episodic Memory (A-GEM) addresses continual learning by storing a small memory buffer of past task exemplars and constraining gradient updates to maintain consistency with previous tasks. By ensuring that gradients on new tasks have non-negative projection onto average gradients on stored exemplars, A-GEM prevents catastrophic forgetting while maintaining computational efficiency and minimal memory requirements.

In quantitative finance, A-GEM provides an elegant framework for portfolio management: maintain exemplars of important market conditions (crises, bull runs, volatility regimes), and ensure that model updates improving current performance don't negatively impact predictions for these critical historical scenarios.

## Key Concepts

- **Exemplar Buffer**: Store representative samples from previous tasks
- **Gradient Projection**: Constrain new task gradients
- **Feasible Gradient Cone**: Update direction must respect past task performance
- **Computational Efficiency**: O(m) memory for m exemplars
- **Task-Agnostic**: Works without task boundaries
- **Strict Gradient Constraint**: Prevents catastrophic forgetting

## Mathematical Framework

### Core Principle

For new task gradient $\mathbf{g}_t$ and average gradient on past exemplars $\bar{\mathbf{g}}$:

$$\mathbf{g}_t \cdot \bar{\mathbf{g}} \geq 0$$

This orthogonal constraint ensures forward progress on new task while maintaining past task performance.

### Average Gradient Computation

Compute reference gradient from memory buffer $\mathcal{M}$:

$$\bar{\mathbf{g}} = \frac{1}{|\mathcal{M}|} \sum_{(\mathbf{x}, y) \in \mathcal{M}} \nabla_\theta \mathcal{L}(\theta, \mathbf{x}, y)$$

This represents "average direction" that improves past task performance.

### Gradient Constraint

If new gradient violates constraint ($\mathbf{g}_t \cdot \bar{\mathbf{g}} < 0$), project:

$$\mathbf{g}_t^{\text{projected}} = \mathbf{g}_t - \frac{\mathbf{g}_t \cdot \bar{\mathbf{g}}}{\|\bar{\mathbf{g}}\|^2} \bar{\mathbf{g}}$$

This removes the component opposing past task performance.

## A-GEM Algorithm

### Training Procedure

```
procedure A-GEM(tasks, memory_size M):
    θ ← random initialization
    buffer ← empty episodic memory
    
    for each task t:
        for epoch in 1 to num_epochs:
            for minibatch (x, y) from task_t:
                g_t ← ∇L(θ, x, y)
                
                if buffer not empty:
                    g_bar ← average_gradient(buffer, θ)
                    
                    if g_t · g_bar < 0:  // constraint violation
                        g_t ← project_gradient(g_t, g_bar)
                
                θ ← update(θ, g_t)  // optimizer step
        
        // Update memory buffer
        buffer ← update_exemplars(buffer, task_t, M)
    
    return θ, buffer
```

### Exemplar Selection

Select which samples to store from task $t$:

**Random Sampling**:
$$\mathcal{M}_t = \text{RandomSample}(\mathcal{D}_t, m)$$

Simple and unbiased.

**Uncertainty-Based**:
$$\text{Priority}_i = H(p(\hat{y}|\mathbf{x}_i))$$

Store uncertain samples.

**Herding**:
$$\mathcal{M}_t = \arg\min_S \left\| \frac{1}{m} \sum_{\mathbf{x} \in S} \phi(\mathbf{x}) - \bar{\phi}(D_t) \right\|$$

Match feature statistics to full dataset.

## Gradient Projection Mechanics

### Projection Visualization

In gradient space, new gradient $\mathbf{g}_t$ and past gradient $\bar{\mathbf{g}}$ define feasible cone:

```
        g_t (new task)
         /
        /  feasible region (forward progress on both)
       /___
      |    \
      |     \ (g_t · g_bar > 0)
   g_bar    \
(past task)  \
              \__g_t_projected
```

If $\mathbf{g}_t$ points against $\bar{\mathbf{g}}$ (violates constraint), project into feasible region.

### Projection Formula Derivation

Minimize $\|\mathbf{g}' - \mathbf{g}\|^2$ subject to $\mathbf{g}' \cdot \bar{\mathbf{g}} \geq 0$:

$$\mathbf{g}' = \mathbf{g} - \text{max}(0, \frac{\mathbf{g} \cdot \bar{\mathbf{g}}}{\|\bar{\mathbf{g}}\|^2}) \bar{\mathbf{g}}$$

Removes only the problematic component, preserves orthogonal directions.

## Memory Management

### Fixed Memory Budget

With memory constraint, decide when to add exemplars:

**Strategy 1: Reservoir Sampling**
- Maintain random sample of all past data
- Equal probability for all exemplars

**Strategy 2: Importance-Based**
- Prioritize important exemplars
- Expensive computationally

**Strategy 3: Task-Based Allocation**
$$m_t = \frac{M}{\text{num\_tasks}}$$

Equal allocation per task.

!!! tip "Memory Refresh"
    Periodically refresh buffer with current predictions:
    
    $$\text{Keep exemplar if } \text{model\_uncertainty}(\mathbf{x}) > \text{threshold}$$
    
    Replace outdated exemplars with currently-difficult samples.

## Convergence Analysis

### Constraint Satisfaction

With A-GEM, constraint is always satisfied:

$$\mathbf{g}_{\text{updated}} \cdot \bar{\mathbf{g}} \geq 0$$

This is enforced by design through projection.

### Convergence Rate

Under convexity assumptions:

$$\mathbb{E}[\mathcal{L}(T)] \leq \mathcal{L}(0) - c \cdot T + O(\sigma^2/\sqrt{T})$$

where:
- $c$ is step size dependent convergence rate
- $\sigma$ is gradient variance
- $T$ is training steps

Convergence slightly slower than standard SGD due to projection overhead.

## Comparison with Related Methods

| Method | Memory | Computation | Forgetting | Theory |
|--------|--------|------------|-----------|--------|
| **A-GEM** | Low | Low | Very Low | Strong |
| **EWC** | None | Medium | Moderate | Strong |
| **Replay** | High | Low | Very Low | Medium |
| **SI** | None | Low | Moderate | Medium |

A-GEM balances memory efficiency with strong theoretical guarantees.

## Practical Considerations

### Memory Buffer Size Selection

How many exemplars needed? Empirical analysis:

| Buffer Size | Forgetting | Computation |
|------------|-----------|------------|
| 100 | 8% | 1.2x |
| 500 | 3% | 1.6x |
| 1000 | 1.5% | 2.1x |
| 5000 | 0.5% | 4.5x |

Diminishing returns beyond 1000 exemplars for most problems.

### Computational Overhead

Gradient projection cost:

**Per-batch overhead**:
- Compute new gradient: $O(n_{\theta})$
- Compute average gradient: $O(m \cdot n_{\theta})$ where $m$ is buffer size
- Projection: $O(n_{\theta})$
- **Total**: $O((1+m) n_{\theta})$

For $m=100, n_{\theta}=10^6$: ~10% overhead.

## Financial Applications

!!! warning "Crisis-Aware Portfolio Learning"
    
    Maintain episodic memory of crisis scenarios:
    - 2008 Financial Crisis returns
    - 2020 COVID Crash patterns
    - Prior bull market peaks
    
    When optimizing new portfolio:
    1. Compute gradient for current market
    2. Compute average gradient for crisis scenarios
    3. Ensure portfolio changes don't hurt crisis performance
    4. Result: Robust to regime changes

### Asset-Specific A-GEM

Store exemplars per asset class:

```
Equities Buffer: Historical equity returns + crashes
Bonds Buffer: Interest rate moves, credit events
Commodities Buffer: Supply shocks, seasonal patterns
```

Learn new trading strategy while maintaining knowledge of each asset's behavior.

## Advanced Topics

### Importance-Weighted A-GEM

Weight exemplars by importance rather than uniform:

$$\bar{\mathbf{g}} = \frac{\sum_i w_i \nabla \mathcal{L}_i}{\sum_i w_i}$$

where $w_i$ measures exemplar importance.

### Task-Aware Memory Management

Dynamically adjust memory allocation based on task:

$$m(t) = f(\text{task\_properties}_t)$$

Allocate more memory to difficult/important tasks.

### Distributed A-GEM

For federated learning or distributed training:

```
Local A-GEM on each client
Share exemplars (not raw data) across clients
Global gradient projection
```

Maintains privacy while enabling continual learning.

## Research Directions

- Adaptive buffer size selection
- Optimal exemplar selection strategies
- Theoretical convergence with projection
- Combining A-GEM with other methods

## Related Topics

- Replay Methods Overview (Chapter 12.4.1)
- Dark Experience Replay (Chapter 12.4.3)
- Gradient-Based Continual Learning
- Episodic Memory Systems
