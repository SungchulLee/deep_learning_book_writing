# 35.2.2 Multi-Asset Allocation

## Learning Objectives

- Implement deep RL agents for multi-asset portfolio allocation
- Design neural network architectures that respect portfolio constraints
- Compare different policy parameterizations for allocation problems
- Handle diverse asset classes with heterogeneous characteristics

## Introduction

Multi-asset allocation extends single-asset trading to the simultaneous management of positions across many assets. The challenge scales significantly: with $N$ assets, the action space is $N$-dimensional, correlations create complex dependencies, and the curse of dimensionality threatens sample efficiency.

Deep RL addresses this through function approximation—learning a policy $\pi_\theta(a|s)$ that maps high-dimensional market states to portfolio weight vectors. The key architectural challenge is ensuring the output satisfies portfolio constraints (e.g., weights sum to 1) while remaining differentiable for gradient-based optimization.

## Architecture Designs

### 1. Softmax Policy (Long-Only)

The simplest approach uses a softmax output layer to produce valid portfolio weights:

$$w_i = \frac{e^{z_i}}{\sum_{j=1}^{N} e^{z_j}}, \quad z = f_\theta(s)$$

**Advantages**: Automatically satisfies $w_i \geq 0$ and $\sum w_i = 1$.

**Disadvantage**: Cannot represent zero-weight positions exactly; biased toward equal weights.

### 2. EIIE Architecture (Ensemble of Identical Independent Evaluators)

Proposed by Jiang et al. (2017), EIIE processes each asset through an identical sub-network, enabling scalability:

```
Asset 1 features → Shared CNN/LSTM → Score_1 ─┐
Asset 2 features → Shared CNN/LSTM → Score_2 ──┤→ Softmax → Weights
Asset 3 features → Shared CNN/LSTM → Score_3 ──┤
   ...                                          │
Cash reserve ─────────────────────→ Score_cash ─┘
```

The weight-shared architecture ensures the number of parameters is independent of portfolio size, making it scalable to large universes.

### 3. Attention-Based Allocation

Transformer-style attention captures cross-asset dependencies:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Each asset attends to all others, learning which correlations matter for allocation decisions. This naturally handles time-varying correlation structures.

### 4. Hierarchical Allocation

For large universes, hierarchical allocation first assigns to sectors/clusters, then within each:

$$\mathbf{w} = \mathbf{w}^{\text{sector}} \otimes \mathbf{w}^{\text{within-sector}}$$

This decomposes the $N$-dimensional problem into smaller sub-problems.

## Constraint Handling

### Hard Constraints via Architecture

| Constraint | Implementation |
|-----------|---------------|
| Long-only, fully invested | Softmax output |
| Position limits $w_i \leq w_{\max}$ | Clipped softmax with re-normalization |
| Sector exposure limits | Grouped softmax |
| Cash allocation | Add cash as $(N+1)$-th asset |
| Leverage limit $\sum \|w_i\| \leq L$ | Tanh + scaling |

### Soft Constraints via Reward

Constraints that are difficult to enforce architecturally can be added as penalties:

$$r_t^{\text{constrained}} = r_t - \lambda_1 \max(0, \text{conc} - c_{\max}) - \lambda_2 \max(0, \text{turnover} - \tau_{\max})$$

## Multi-Asset Feature Processing

Different asset classes require different feature engineering:

| Asset Class | Key Features |
|-------------|-------------|
| Equities | Returns, volume, market cap, P/E, momentum |
| Fixed Income | Yield, duration, credit spread, curve shape |
| Commodities | Spot price, roll yield, inventory, seasonality |
| FX | Interest rate differential, purchasing power parity, momentum |

A unified state representation normalizes features across asset classes:

$$x_i^{\text{norm}} = \frac{x_i - \mu_i^{\text{rolling}}}{\sigma_i^{\text{rolling}} + \epsilon}$$

## Training Strategies

### 1. Curriculum Learning

Start with a small number of assets and gradually increase:

- Phase 1: 2-3 assets (learn basic allocation)
- Phase 2: 5-10 assets (learn diversification)
- Phase 3: Full universe (learn selection + allocation)

### 2. Experience Replay with Priority

Prioritize transitions with large absolute returns or regime changes:

$$p_i \propto |r_i| + \epsilon + \alpha \cdot \mathbb{1}[\text{regime change}]$$

### 3. Multi-Period Training

Train on overlapping windows of different lengths to learn both short-term and long-term patterns:

$$\mathcal{L} = \sum_{k \in \{1, 5, 20\}} \lambda_k \cdot \mathcal{L}_k$$

## Equal Risk Contribution (ERC) Comparison

A strong non-RL baseline is the Equal Risk Contribution portfolio:

$$w_i \cdot (\Sigma \mathbf{w})_i = \frac{1}{N} \mathbf{w}^T \Sigma \mathbf{w} \quad \forall i$$

This ensures each asset contributes equally to total portfolio risk.

## Summary

Multi-asset allocation via deep RL requires careful attention to architecture design for constraint satisfaction, feature processing for heterogeneous assets, and training strategies that address the high-dimensional action space. The EIIE and attention-based architectures scale well to large portfolios, while hierarchical approaches decompose the problem for very large universes. Curriculum learning and multi-period training improve sample efficiency and robustness.

## References

- Jiang, Z., Xu, D., & Liang, J. (2017). A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem. arXiv:1706.10059.
- Ye, Y., et al. (2020). Reinforcement-Learning Based Portfolio Management with Augmented Asset Movement Prediction States. AAAI.
- Maillard, S., Roncalli, T., & Teïletche, J. (2010). The Properties of Equally Weighted Risk Contribution Portfolios. The Journal of Portfolio Management.
