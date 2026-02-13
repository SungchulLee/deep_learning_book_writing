# Task Distribution Design for Meta-Learning

## Introduction

The quality and characteristics of the task distribution fundamentally determine meta-learning success. While meta-learning theory assumes access to an ideal task distribution, practitioners must design task distributions that reflect real-world scenarios, enable efficient meta-training, and ensure generalization to test tasks.

In quantitative finance, task distribution design directly impacts trading system robustness. By constructing meta-training tasks from diverse market periods, asset classes, and regimes, practitioners develop meta-learners that gracefully adapt to novel market conditions. Careful task distribution design prevents overfitting to historical patterns while ensuring realistic adaptation requirements.

## Key Concepts

- **Task Complexity**: Difficulty of individual tasks and meta-learning objectives
- **Task Diversity**: Variation in task characteristics and problem types
- **Domain Coverage**: Representation of real-world conditions in task distribution
- **Curriculum Design**: Ordering of tasks to facilitate meta-learning
- **Meta-Generalization**: Test task performance given training task distribution

## Task Definition in Meta-Learning

### Formal Task Specification

A task consists of:

$$\mathcal{T} = \{\mathcal{D}_{\text{train}}, \mathcal{D}_{\text{test}}, \ell(\cdot)\}$$

where:
- $\mathcal{D}_{\text{train}} = \{(\mathbf{x}_i, y_i)\}_{i=1}^k$ is support set
- $\mathcal{D}_{\text{test}} = \{(\mathbf{x}_j, y_j)\}_{j=1}^q$ is query set
- $\ell(\cdot)$ is task-specific loss function

### Task Similarity

Measure similarity between tasks:

$$\text{Sim}(\mathcal{T}_i, \mathcal{T}_j) = \frac{1}{1 + \text{distance}(\mathcal{T}_i, \mathcal{T}_j)}$$

where distance measures:
- Distributional divergence: $\text{KL}(P_i \| P_j)$
- Task performance correlation
- Feature space overlap

## Task Distribution Characteristics

### Diversity Dimensions

High-quality task distributions vary along multiple axes:

| Dimension | Purpose | Example |
|-----------|---------|---------|
| **Input Distribution** | Feature variation | Different market regimes |
| **Task Difficulty** | Convergence rates | Easy vs. hard predictions |
| **Output Structure** | Objective variety | Classification, regression |
| **Noise Levels** | Robustness | Clean vs. noisy signals |
| **Temporal Structure** | Temporal reasoning | Static vs. sequential |

### Complexity-Difficulty Balance

!!! tip "Curriculum Design"
    Balance task distribution complexity for efficient meta-learning:
    - Too easy: Insufficient learning signal
    - Too hard: Unstable meta-training
    - Mixed: Optimal meta-generalization

## Financial Task Distribution Design

### Asset-Based Partitioning

Create tasks from different securities and instrument classes:

**Financial Task Example**:
- **Support Set**: Historical returns for Stock A from January-March
- **Query Set**: April returns for Stock A (unlabeled during training)
- **Loss**: Prediction error on April returns

```
Tasks = {
  Task 1: Predict AAPL returns using Jan-Mar AAPL data
  Task 2: Predict MSFT returns using Jan-Mar MSFT data
  Task 3: Predict BTC returns using Jan-Mar BTC data
  ...
  Task N: Predict sector index using historical data
}
```

Meta-learning discovers how to predict across different assets.

### Temporal Partitioning

Create tasks from different time periods:

**Regime-Based Tasks**:
- Bull market periods
- Bear market periods
- Sideways/consolidation periods
- Transition periods
- High volatility periods
- Low volatility periods

Meta-learning develops regime-specific adaptations.

### Market Condition Partitioning

Group tasks by market characteristics:

| Market Condition | Task Characteristics |
|-----------------|----------------------|
| **High Liquidity** | Low spreads, tight bid-ask |
| **Low Liquidity** | Wide spreads, execution challenges |
| **High Volatility** | Large returns, uncertain forecasts |
| **Low Volatility** | Stable patterns, predictable |
| **Trending** | Directional bias, momentum |
| **Mean-Reverting** | Range-bound, reversion signal |

## Constructing Task Distributions

### Sampling Strategy

**Uniform Sampling**: Equal probability for each task type (simplest)

**Weighted Sampling**: Emphasize important or difficult task types:

$$P(\mathcal{T}) \propto \text{Importance}(\mathcal{T})$$

**Curriculum**: Gradually increase difficulty:

$$P(\mathcal{T}) = P_{\text{curriculum}}(\text{Difficulty}(t))$$

where difficulty increases with meta-training iteration $t$.

### Support-Query Split Strategy

**Few-Shot Setting**: 5 examples per class, 15 queries per class (standard)

**Different Data Regimes**:
- Ultra-low data: 1-5 support examples
- Low data: 5-10 support examples
- Moderate data: 10-50 support examples
- High data: 50+ support examples

**Financial Recommendation**: Use 10-20 trading days as support set (2-4 weeks), 5-10 days as query set (1-2 weeks) for daily trading.

## Avoiding Task Distribution Pitfalls

### Avoiding Distribution Mismatch

!!! warning "Meta-Test Mismatch"
    If test tasks come from different distribution than meta-training tasks, meta-generalization fails.

**Solution**: Ensure task distribution reflects real-world test scenarios.

**Financial Example**: If testing on 2024 data, meta-train on 2020-2023 market conditions.

### Handling Task Contamination

Prevent information leakage between support and query sets:

$$\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset$$

Strict separation prevents memorization.

### Avoiding Trivial Tasks

Some task distributions are too easy, not providing learning signal:

**Example (Bad)**: All tasks are MNIST with same distribution
**Example (Good)**: MNIST tasks with varying rotation, scale, style

For financial data:
- **Bad**: All tasks from same asset in same regime
- **Good**: Mix assets, regimes, time periods

## Measuring Task Distribution Quality

### Task Diversity Metrics

**Average Pairwise Distance**:
$$\bar{d} = \frac{1}{\binom{|T|}{2}} \sum_{i<j} d(\mathcal{T}_i, \mathcal{T}_j)$$

Higher distance indicates more diverse task distribution.

**Spectrum Analysis**: Principal component analysis on task descriptors reveals dominant variation directions.

**Distribution Entropy**: How uniform is task distribution?

$$H = -\sum_i P(\mathcal{T}_i) \log P(\mathcal{T}_i)$$

### Meta-Generalization Evaluation

**Oracle Performance**: Test meta-learner on test tasks from same distribution (upper bound).

**Distribution Shift Performance**: Test on tasks from different distribution (realistic evaluation).

**Convergence Behavior**: Does meta-learner converge smoothly, or experience sudden performance drops?

## Advanced Task Distribution Design

### Curriculum Learning

Progressive difficulty increase during meta-training:

```
Phase 1 (Iterations 1-1000): Easy tasks only
Phase 2 (Iterations 1001-3000): Mix easy and medium
Phase 3 (Iterations 3001-5000): All difficulty levels
Phase 4 (Iterations 5001+): Hard tasks emphasized
```

Benefits: Faster convergence, better final performance, more stable training.

### Adversarial Task Generation

Automatically generate difficult tasks that stress-test meta-learner:

$$\mathcal{T}_{\text{adv}} = \arg\max_{\mathcal{T}} \text{Error}_{\text{meta-learner}}(\mathcal{T})$$

Iteratively add challenging tasks to distribution.

### Adaptive Task Weighting

Weight tasks by meta-learner's uncertainty:

$$w_i(t) = 1 - \text{Confidence}(\mathcal{T}_i, \text{meta-learner at iteration } t)$$

Focus on tasks where meta-learner struggles most.

## Financial Implementation Examples

### Daily Trading System Task Distribution

```
Tasks = {
  Stock 1 - Bull Period: Support [2017-2018], Query [2019]
  Stock 1 - Bear Period: Support [2020], Query [2020-2021]
  Stock 2 - Bull Period: Support [2017-2018], Query [2019]
  ...
  Index - All Periods: Support [2015-2019], Query [2020-2021]
}
```

Meta-learner learns to adapt across different assets and market regimes.

### Multi-Asset Portfolio Task Distribution

```
Tasks = {
  Equities: Support [train data], Query [test data]
  Bonds: Support [train data], Query [test data]
  Commodities: Support [train data], Query [test data]
  FX: Support [train data], Query [test data]
}
```

Meta-learner learns cross-asset patterns.

## Related Topics

- Meta-Learning Overview (Chapter 11.1)
- Learned Optimizers (Chapter 11.2)
- Few-Shot Learning Theory
- Curriculum Learning
- Task Clustering and Selection
