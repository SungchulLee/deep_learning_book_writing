# Continual Learning

## Overview

**Continual learning** (also known as lifelong learning, incremental learning, or sequential learning) addresses one of the fundamental challenges in machine learning: how to train neural networks on a sequence of tasks without forgetting previously learned knowledge. This chapter provides a comprehensive treatment of the theoretical foundations, mathematical frameworks, and practical implementations of continual learning strategies.

## The Central Problem: Catastrophic Forgetting

When neural networks are trained sequentially on multiple tasks, they exhibit **catastrophic forgetting**—a phenomenon where learning new information catastrophically interferes with previously acquired knowledge. This occurs because:

1. **Weight Plasticity**: Neural networks update all weights to minimize the current task's loss
2. **Representational Overlap**: New task optimization overwrites weights crucial for previous tasks
3. **Stability-Plasticity Dilemma**: The fundamental tension between adapting to new information (plasticity) and retaining old knowledge (stability)

!!! warning "Why This Matters"
    Standard deep learning training procedures are fundamentally incompatible with continual learning scenarios. Without specialized techniques, a model trained sequentially on tasks A, B, and C will typically retain good performance only on task C, with catastrophic degradation on tasks A and B.

## Mathematical Formulation

### Problem Setup

Consider a sequence of $T$ tasks $\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_T$, where each task $\tau$ consists of:

- **Training data**: $\mathcal{D}_\tau = \{(x_i, y_i)\}_{i=1}^{N_\tau}$
- **Loss function**: $\mathcal{L}_\tau(\theta)$
- **Model parameters**: $\theta \in \mathbb{R}^d$

The goal of continual learning is to find parameters $\theta^*$ that perform well across all tasks:

$$
\theta^* = \arg\min_\theta \sum_{\tau=1}^{T} \mathcal{L}_\tau(\theta)
$$

subject to the constraint that task $\tau$ data $\mathcal{D}_\tau$ is only available during training on task $\tau$.

### Quantifying Forgetting

**Per-task forgetting** after learning task $T$ is measured as:

$$
F_i = \text{Acc}_{i,i} - \text{Acc}_{i,T}
$$

where $\text{Acc}_{i,j}$ denotes the accuracy on task $i$ after training on task $j$.

**Average Forgetting** across all previous tasks:

$$
\bar{F} = \frac{1}{T-1} \sum_{i=1}^{T-1} \max_{j \in \{i, \ldots, T-1\}} \left( \text{Acc}_{i,j} - \text{Acc}_{i,T} \right)
$$

## Learning Scenarios

### Task-Incremental Learning
- Model knows task identity at test time
- Different output heads for different tasks
- Focus: preventing interference between task-specific knowledge

### Domain-Incremental Learning
- Input distribution changes but task structure remains constant
- Example: Object recognition across different lighting conditions
- Focus: adapting to distribution shifts while maintaining task performance

### Class-Incremental Learning
- Most challenging scenario
- New classes added over time without explicit task boundaries
- Model must distinguish between all classes seen so far
- No task identity provided at inference time

## Main Approaches

| Approach | Key Idea | Storage | Privacy | Scalability |
|----------|----------|---------|---------|-------------|
| **Regularization** | Protect important weights | None | ✓ High | ✓ Good |
| **Replay** | Store/replay previous examples | Examples | ✗ Low | Medium |
| **Architecture** | Dedicate capacity per task | Parameters | ✓ High | ✗ Poor |
| **Meta-Learning** | Learn to learn continually | Varies | Varies | Varies |

## Chapter Structure

This chapter is organized into the following sections:

### Foundations
- [**Catastrophic Forgetting**](catastrophic_forgetting.md): Understanding and visualizing the problem
- [**Evaluation Metrics**](evaluation_metrics.md): Standard metrics for continual learning

### Learning Scenarios
- [**Task-Incremental Learning**](task_incremental.md): Learning with task boundaries
- [**Class-Incremental Learning**](class_incremental.md): Learning without task identities
- [**Online Continual Learning**](online_continual.md): Single-pass learning

### Regularization Methods
- [**Elastic Weight Consolidation**](ewc.md): Fisher information-based protection
- [**Synaptic Intelligence**](synaptic_intelligence.md): Online importance estimation
- [**Memory Aware Synapses**](mas.md): Unsupervised importance weights

### Replay Methods
- [**Experience Replay**](experience_replay.md): Memory buffer strategies
- [**Generative Replay**](generative_replay.md): Pseudo-rehearsal with generative models
- [**Gradient Episodic Memory**](gem.md): Constrained optimization approach

### Knowledge Distillation
- [**Learning Without Forgetting**](lwf.md): Distillation for continual learning
- [**Dark Experience Replay**](dark_experience_replay.md): Combining replay with distillation

### Architecture Methods
- [**Progressive Neural Networks**](progressive_networks.md): Growing architectures
- [**PackNet**](packnet.md): Network pruning and freezing
- [**Dynamically Expandable Networks**](den.md): Selective expansion

### Advanced Topics
- [**Task-Free Continual Learning**](task_free.md): Learning without boundaries
- [**Meta-Continual Learning**](meta_continual.md): Learning to learn continually
- [**Continual Learning Theory**](theory.md): Theoretical foundations

### Applications
- [**Finance Applications**](finance_applications.md): Continual learning in quantitative finance
- [**Benchmark Comparison**](benchmarks.md): Comprehensive method evaluation

## Prerequisites

Before studying this chapter, you should be familiar with:

- Deep learning fundamentals (feedforward networks, backpropagation)
- Optimization theory (gradient descent, Adam optimizer)
- Regularization techniques (L2 regularization, dropout)
- Information theory basics (KL divergence, entropy)

## Key Takeaways

By completing this chapter, you will:

1. **Understand** the catastrophic forgetting problem and its mathematical formulation
2. **Implement** major continual learning strategies from scratch in PyTorch
3. **Evaluate** methods using standard metrics (average accuracy, backward transfer)
4. **Compare** regularization, replay, and architecture-based approaches
5. **Apply** continual learning techniques to real-world sequential learning problems
6. **Analyze** computational and memory trade-offs between methods

## References

!!! quote "Foundational Papers"
    - McCloskey & Cohen (1989). "Catastrophic Interference in Connectionist Networks"
    - Kirkpatrick et al. (2017). "Overcoming Catastrophic Forgetting in Neural Networks" (EWC)
    - Zenke et al. (2017). "Continual Learning Through Synaptic Intelligence"
    - Lopez-Paz & Ranzato (2017). "Gradient Episodic Memory for Continual Learning"
    - Li & Hoiem (2017). "Learning Without Forgetting"
    - Parisi et al. (2019). "Continual Lifelong Learning with Neural Networks: A Review"
