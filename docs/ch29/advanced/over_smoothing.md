# 29.4.2 Over-Smoothing

## Introduction

**Over-smoothing** is the phenomenon where node representations become increasingly similar as more GNN layers are applied, eventually converging to indistinguishable states. It is the primary bottleneck limiting GNN depth.

## Mathematical Analysis

After $K$ layers of graph convolution with normalized adjacency $\hat{A}$:

$$H^{(K)} = \hat{A}^K H^{(0)} W_{eff}$$

As $K \to \infty$, $\hat{A}^K$ converges to a rank-1 matrix for connected graphs (related to the stationary distribution of the random walk). All rows become proportional, destroying node-level information.

## Measuring Over-Smoothing

**Mean Average Distance (MAD)**: $\text{MAD} = \frac{1}{n^2} \sum_{i,j} \|\mathbf{h}_i - \mathbf{h}_j\|$

**Dirichlet Energy**: $E(H) = \text{tr}(H^T L H) = \sum_{(i,j) \in E} \|\mathbf{h}_i - \mathbf{h}_j\|^2$

Over-smoothing occurs when both metrics approach zero.

## Mitigation Strategies

1. **Residual connections**: Preserve early-layer information
2. **Jumping Knowledge**: Aggregate representations from all layers
3. **DropEdge**: Reduce message passing rate
4. **PairNorm**: $\tilde{h}_i = s \cdot \frac{h_i - \mu}{\sqrt{\frac{1}{n}\sum_j \|h_j - \mu\|^2}}$
5. **DiffPool / graph coarsening**: Reduce graph size between layers
6. **Graph transformers**: Global attention avoids iterative smoothing

## Summary

Over-smoothing fundamentally limits GNN depth. Understanding and measuring it guides architecture design, with residual connections and jumping knowledge being the most effective mitigations.
