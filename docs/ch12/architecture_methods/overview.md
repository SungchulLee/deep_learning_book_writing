# Overview of Architecture-Based Continual Learning Methods

## Introduction

Architecture-based continual learning approaches address catastrophic forgetting through network design that enables learning new tasks without degrading performance on previous tasks. Rather than relying on regularization or replay mechanisms, these methods allocate distinct network capacity to different tasks, preventing parameter interference and enabling independent learning trajectories.

In quantitative finance, architecture-based continual learning proves particularly valuable: as new market conditions emerge or trading strategies evolve, models must learn new patterns without forgetting previously-learned market relationships. Dynamic network expansion provides a principled way to accumulate financial knowledge while maintaining clean task separation.

## Key Concepts

- **Task-Specific Modules**: Separate network components for each task
- **Dynamic Expansion**: Grow network capacity as new tasks arrive
- **Parameter Isolation**: Prevent parameter sharing between tasks causing interference
- **Sparse Connectivity**: Use selective connections reducing parameter overlap
- **Adapter Modules**: Small, trainable components per task
- **Expert Selection**: Route data to task-relevant experts

## Architecture Paradigms

### Multi-Head Architectures

Maintain task-specific output heads while sharing backbone:

$$\mathbf{y}_t = \text{Head}_t(f_{\text{shared}}(\mathbf{x}))$$

**Pros**: Simple, parameter-efficient for backbone sharing

**Cons**: Shared backbone may suffer catastrophic forgetting

### Progressive Neural Networks

Add new network columns for new tasks, connecting to previous columns:

```
Task 1 ──┬─────────────────┐
         │                 │
    [Shared backbone 1]    │
         │                 │
    Task 1 Head            │
                           │
Task 2 ──────┬──────────────┼──────┐
             │              │      │
        [New backbone 2] ←──┴─────┘
             │
        Task 2 Head
```

New task's backbone receives lateral connections from previous backbones.

### Adapter-Based Methods

!!! tip "Parameter Efficiency"
    Adapters add small trainable modules while freezing base model parameters. Each task gets custom adapter.

$$\mathbf{y}_t = f_{\text{base}}(\mathbf{x}) + \text{Adapter}_t(f_{\text{base}}(\mathbf{x}))$$

Minimal parameter overhead per task ($\sim 5\%$ of base model).

## Detailed Architecture Approaches

### PackNet: Adding Structure to Continual Learning

Use binary masks to allocate network capacity:

$$\mathbf{w}^{(t)} = m^{(t)} \odot \mathbf{w}^{(t-1)}$$

where $m^{(t)} \in \{0,1\}^{\text{dim}(\mathbf{w})}$ is learned binary mask for task $t$.

**Pruning**: Masks identify critical parameters for past tasks
**Expansion**: New tasks learn on unused parameters

### Supermask in Superposition (Chapter 12.1.3)

Train binary masks to select subnetworks for tasks.

### Expert Gating (Chapter 12.1.2)

Route tasks through different expert modules with learned gating:

$$\mathbf{y}_t = \sum_e g_t(e) \cdot \mathbf{E}_e(\mathbf{x})$$

where $g_t(e)$ are learned gating weights selecting experts.

## Comparison of Architecture Methods

| Method | Capacity Growth | Parameter Sharing | Forgetting | Task ID |
|--------|-----------------|-------------------|-----------|---------|
| **Multi-Head** | Fixed | High | Possible | Required |
| **Progressive** | Linear | Controlled | Minimal | Required |
| **Adapters** | Sublinear | High | Minimal | Required |
| **Expert Gates** | Linear | Partial | Minimal | Not Required |
| **Masks** | Linear | High | Minimal | Required |

## Advantages and Limitations

### Advantages

**Zero Forgetting**: Task-specific parameters prevent interference

**Theoretical Properties**: Clean separation enables analysis of learning dynamics

**Flexibility**: Support arbitrary new task types without retraining

### Limitations

!!! warning "Capacity Requirements"
    Network grows with number of tasks. For T tasks and N parameters:
    - Progressive NNs: O(TN) parameters
    - Adapters: O(N + TδN) where δ ≤ 0.05
    - Expert Gates: O(E·N) where E is number of experts

**Task Identification**: Most methods require knowing task ID at test time

**Limited Sharing**: May underutilize common structure between tasks

## Training Procedures

### Standard Training Loop

```
for each task t:
    Initialize task-specific components
    for epoch in 1 to T_epochs:
        for batch in dataset_t:
            Freeze previous task parameters
            Update current task parameters only
            No access to previous task data
    Store task representation for future reference
```

### Selective Memory Access

Some methods maintain memory buffer of past tasks:

$$\mathcal{L}(t) = \mathcal{L}_{\text{task}_t} + \lambda \sum_{t' < t} \mathcal{L}_{\text{task}_{t'}}^{\text{sampled}}$$

### Consolidation Phase

After learning task $t$, consolidate parameters before learning task $t+1$:

1. **Identify Critical Parameters**: Which parameters are important for task $t$?
2. **Protect Critical Parameters**: Make them harder to change
3. **Mark Available Capacity**: Which parameters can be used for task $t+1$?

## Financial Applications

!!! warning "Market Regime Continual Learning"
    
    As market regimes evolve:
    1. Train experts on historical regime data
    2. Route current market data through active expert
    3. When new regime emerges, train new expert
    4. Maintain prediction capability across all regimes
    5. Gradually consolidate as regime matures

### Multi-Asset Trading System

Maintain separate experts for:
- Individual stocks
- Sectors
- Indices
- Bonds
- Derivatives

Route each asset through appropriate expert(s).

## Research Directions

- Optimal allocation of network capacity to tasks
- Automatic detection of task boundaries
- Theoretical analysis of multi-task learning bounds
- Efficient parameter sharing across related tasks
- Learning task relationship structure

## Related Topics

- Expert Gate Methods (Chapter 12.1.2)
- Supermask in Superposition (Chapter 12.1.3)
- Adapter-Based Methods
- Progressive Neural Networks
- Distillation-Based Continual Learning (Chapter 12.2)
