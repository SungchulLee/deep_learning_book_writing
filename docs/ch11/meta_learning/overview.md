# Overview of Meta-Learning Approaches

## Introduction

Meta-learning, often termed "learning to learn," represents a paradigm where models acquire knowledge about how to learn effectively, rather than learning specific tasks directly. By training on diverse task distributions, meta-learning systems develop inductive biases and algorithms that enable rapid adaptation to new tasks with minimal data.

In quantitative finance, meta-learning proves transformative: market conditions change continuously, new assets emerge, and trading strategies must adapt rapidly. Meta-learning enables building systems that learn from experience across multiple markets, timeframes, and regimes, then apply that meta-knowledge to new financial instruments with few examples.

## Key Concepts

- **Task Distribution**: Collection of related tasks for meta-training
- **Inner Loop**: Task-specific learning (typically few examples)
- **Outer Loop**: Meta-parameter updates across task distribution
- **Few-Shot Learning**: Rapid adaptation with minimal examples
- **Model-Agnostic**: Meta-learning objectives applicable across architectures
- **Meta-Generalization**: Performance on unseen test tasks

## Meta-Learning Paradigms

### Optimization-Based Meta-Learning

Learn optimizer (or learning algorithm) to quickly adapt to new tasks.

**MAML (Model-Agnostic Meta-Learning)**:

$$\theta_{\text{meta}}^* = \arg\min_{\theta} \sum_{\mathcal{T} \in \mathcal{D}} \mathcal{L}_{\text{test}}(\theta - \alpha \nabla \mathcal{L}_{\text{train}}(\theta); \mathcal{T})$$

The meta-parameters are initialized such that one gradient step on task training data improves validation performance.

### Metric Learning-Based Meta-Learning

Learn embedding space where similar tasks/samples are close.

**Prototypical Networks**:

$$d(\mathbf{x}, c_k) = \|\Phi(\mathbf{x}) - \frac{1}{|S_k|} \sum_{s \in S_k} \Phi(s)\|_2$$

where $c_k$ are class prototypes (means in embedding space).

### Memory-Augmented Meta-Learning

Use external memory to store task-relevant information.

**Memory Networks**: Store key-value pairs from support set, retrieve when predicting queries.

### Probabilistic Meta-Learning

Learn posterior distribution over task-specific parameters.

$$p(\theta_t | \mathcal{T}) = \frac{p(\mathcal{D}_{\text{train}}^t | \theta_t) p(\theta_t | \mu_\phi)}{p(\mathcal{D}_{\text{train}}^t | \phi)}$$

where $\phi$ are meta-parameters of prior distribution.

## Few-Shot Learning Problem Setting

### Task Structure

Meta-training and meta-testing follow the same structure:

**Support Set** $\mathcal{S}_t = \{(\mathbf{x}_i, y_i)\}_{i=1}^{k}$: Few labeled examples per task

**Query Set** $\mathcal{Q}_t = \{(\mathbf{x}_j, y_j)\}_{j=1}^{q}$: Validation examples for task

**Objective**: Minimize query loss after training on support set

### Standard Benchmarks

| Benchmark | Classes | Shots | Queries | Evaluation |
|-----------|---------|-------|---------|-----------|
| **Omniglot** | 5 | 1,5 | 15 | Accuracy |
| **mini-ImageNet** | 5 | 1,5 | 15 | Accuracy |
| **CUB** | 5 | 1,5 | 15 | Accuracy |
| **tiered-ImageNet** | 160 | 1,5 | 15 | Accuracy |

## Comparative Meta-Learning Approaches

| Approach | Mechanism | Strength | Weakness |
|----------|-----------|----------|----------|
| **MAML** | Gradient-based | General-purpose | Slow meta-training |
| **Prototypical** | Metric-learning | Simple, interpretable | Limited expressiveness |
| **Matching Networks** | Attention-based | Flexible | Complex training |
| **Relation Networks** | Learned comparison | Task-adaptive | Higher complexity |

## Meta-Learning for Quantitative Finance

!!! warning "Financial Few-Shot Learning"
    
    - **New Assets**: Adapt trading models to newly-listed securities with minimal data
    - **Market Regimes**: Learn to switch between regime-specific strategies
    - **Cross-Market Transfer**: Apply models trained on liquid markets to illiquid instruments
    - **Factor Combinations**: Quickly identify useful factor combinations for different markets

### Task Distribution in Finance

Define meta-training tasks as:

**Tasks**: Different time periods, assets, or market conditions

**Support Set**: Historical data from task with known labels

**Query Set**: Subsequent period for same task with true labels

Meta-learning system develops knowledge about:
- Which features transfer across assets
- How quickly to adapt to new market conditions  
- When existing models fail and need retraining

## Architecture Considerations

### Model-Agnostic Design

Meta-learned algorithms should work with various base models:

- **Neural Networks**: Standard approach
- **Kernel Methods**: Learned kernel parameters
- **Decision Trees**: Meta-learned split criteria
- **Ensemble Methods**: Meta-learned component weighting

### Hyperparameter Meta-Learning

Learn hyperparameters that work across task distribution:

$$\text{LR}^* = \arg\min \sum_{\mathcal{T}} \text{error}(\text{Adapt}(\mathcal{T}, \text{LR}))$$

Eliminates task-specific tuning, improving robustness.

## Training Dynamics

### Two-Loop Optimization

```
Outer Loop (Meta-Update):
  for each batch of tasks T:
    Inner Loop (Task Adaptation):
      for each task t in T:
        θ_t ← θ - α ∇L_train(θ; t)
        Accumulate meta-gradient from L_test(θ_t; t)
    θ ← θ - β ∇_meta L
```

Outer loop typically larger mini-batch, inner loop more iterations.

### Gradient Flow Challenges

!!! warning "Computational Complexity"
    Meta-learning requires backpropagation through adaptation process, creating deep computational graphs and significant memory requirements.

## Advantages Over Traditional Transfer Learning

| Aspect | Transfer Learning | Meta-Learning |
|--------|------------------|---------------|
| **Adaptation Data** | Full dataset | Few examples |
| **Adaptation Speed** | Days/hours | Minutes |
| **Cross-Domain** | Limited | Excellent |
| **Interpretability** | High | Lower |
| **Computational Cost** | Low | High |

## Related Concepts

- **Transfer Learning** (Chapter 10): Broader framework
- **Multi-Task Learning**: Related but different objective
- **Continual Learning** (Chapter 12): Sequential task learning
- **Active Learning**: Selecting informative examples

## Implementation Challenges

**Computational Cost**: Meta-training is expensive, requiring many task batches

**Stability**: Nested optimization loops can be unstable without careful tuning

**Hyperparameter Sensitivity**: Meta-learning introduces additional hyperparameters

## Research Directions

- Reducing computational overhead through improved algorithms
- Theoretical understanding of meta-generalization
- Combining meta-learning with other learning paradigms
- Application to non-Euclidean data and complex tasks

## Related Topics

- Learned Optimizers (Chapter 11.2)
- Task Distribution Design (Chapter 11.3)
- Few-Shot Learning Theory
- Neural Architecture Search (learned optimization approach)
