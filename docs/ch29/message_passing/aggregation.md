# 29.2.2 Aggregation Functions

## Introduction

The **aggregation function** is a critical component of message passing that combines messages from a node's neighborhood into a single representation. The aggregation must be **permutation-invariant** since graphs have no canonical node ordering. The choice of aggregation directly impacts the expressiveness and learning capacity of a GNN.

## Requirements

An aggregation function $\bigoplus$ must satisfy:

1. **Permutation invariance**: The output must be the same regardless of the order in which neighbors are processed
2. **Variable-size input**: Must handle different neighborhood sizes
3. **Differentiability**: Must support gradient-based optimization

## Standard Aggregation Functions

### Sum Aggregation

$$\mathbf{m}_v = \sum_{u \in \mathcal{N}(v)} \mathbf{m}_{u \rightarrow v}$$

- **Preserves**: Multiplicity information (how many neighbors send similar messages)
- **Sensitivity**: To neighborhood size
- **Expressiveness**: Most expressive among basic aggregators (used in GIN)
- **Use when**: Counting matters (e.g., molecular graphs: number of bonds)

### Mean Aggregation

$$\mathbf{m}_v = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{m}_{u \rightarrow v}$$

- **Preserves**: Distribution of messages (normalized)
- **Sensitivity**: Invariant to neighborhood size
- **Expressiveness**: Cannot distinguish graphs with different-sized neighborhoods having the same distribution
- **Use when**: Normalization is important (e.g., social networks with varying degree)

### Max Aggregation

$$\mathbf{m}_v = \max_{u \in \mathcal{N}(v)} \mathbf{m}_{u \rightarrow v}$$

- **Preserves**: Extreme values (element-wise maximum)
- **Sensitivity**: Only to the largest values
- **Expressiveness**: Cannot capture distribution shape
- **Use when**: Detecting presence of a feature (regardless of how many neighbors have it)

## Advanced Aggregation Functions

### Attention-Weighted Aggregation

$$\mathbf{m}_v = \sum_{u \in \mathcal{N}(v)} \alpha_{uv} \mathbf{m}_{u \rightarrow v}$$

where $\alpha_{uv}$ is a learned attention coefficient (used in GAT). This allows the model to learn which neighbors are most important.

### Multi-Aggregation

Concatenating multiple aggregations captures different aspects:

$$\mathbf{m}_v = [\text{SUM}(\cdot) \| \text{MEAN}(\cdot) \| \text{MAX}(\cdot) \| \text{STD}(\cdot)]$$

### Softmax Aggregation (DirGNN, PowerMean)

$$\mathbf{m}_v = \sum_{u \in \mathcal{N}(v)} \frac{\exp(\beta \cdot \mathbf{m}_{u \rightarrow v})}{\sum_{w} \exp(\beta \cdot \mathbf{m}_{w \rightarrow v})} \cdot \mathbf{m}_{u \rightarrow v}$$

Interpolates between mean ($\beta \to 0$) and max ($\beta \to \infty$) aggregation.

### Principal Neighborhood Aggregation (PNA)

PNA combines multiple aggregators with degree-scalers:

$$\mathbf{m}_v = \underset{\text{aggregator } \alpha}{\|} \underset{\text{scaler } s}{\|} s\left(\alpha\left(\{\mathbf{m}_{u \rightarrow v}\}_{u \in \mathcal{N}(v)}\right), d_v\right)$$

Aggregators: $\{\text{mean}, \text{max}, \text{min}, \text{std}\}$
Scalers: $\{1, \log(d_v + 1), 1 / (d_v + 1)\}$

### LSTM Aggregation

$$\mathbf{m}_v = \text{LSTM}(\pi(\{\mathbf{m}_{u \rightarrow v}\}_{u \in \mathcal{N}(v)}))$$

where $\pi$ is a random permutation. Not truly permutation-invariant but empirically effective (used in GraphSAGE).

## Expressiveness Hierarchy

From a theoretical perspective:

$$\text{Max} < \text{Mean} < \text{Sum}$$

Sum aggregation is the most expressive because it can distinguish multisets that mean and max cannot. This is formalized by the GIN paper (Xu et al., 2019).

**Example**: Consider two nodes with neighborhoods:
- Node A: neighbors have features $\{1, 1, 1\}$
- Node B: neighbors have features $\{1\}$

Sum distinguishes them ($3$ vs $1$), but mean does not ($1$ vs $1$).

## Quantitative Finance Considerations

- **Sum**: Total exposure, aggregate transaction volume
- **Mean**: Average portfolio return, normalized risk
- **Max**: Worst-case loss, maximum exposure to a single counterparty
- **Attention**: Weight by importance (e.g., larger positions, higher-risk counterparties)

## Summary

The aggregation function determines what information a node can extract from its neighborhood. While sum aggregation is theoretically most expressive, the optimal choice depends on the task, data characteristics, and desired invariances. Modern approaches often combine multiple aggregations for robustness.
