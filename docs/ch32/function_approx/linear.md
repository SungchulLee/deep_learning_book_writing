# 32.8.1 Linear Function Approximation

## Motivation

Tabular methods become infeasible for large or continuous state spaces. **Function approximation** replaces the value table with a parameterized function that generalizes across similar states.

## Linear Value Function

$$\hat{V}(s; \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s) = \sum_{j=1}^{d} w_j x_j(s)$$

where $\mathbf{x}(s) \in \mathbb{R}^d$ is a feature vector and $\mathbf{w} \in \mathbb{R}^d$ are learnable weights.

For Q-functions: $\hat{Q}(s, a; \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s, a)$

## Gradient Descent Updates

### Semi-Gradient TD(0)

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \left[R_{t+1} + \gamma \hat{V}(S_{t+1}; \mathbf{w}) - \hat{V}(S_t; \mathbf{w})\right] \nabla_\mathbf{w} \hat{V}(S_t; \mathbf{w})$$

For linear approximation: $\nabla_\mathbf{w} \hat{V}(s; \mathbf{w}) = \mathbf{x}(s)$

So: $\mathbf{w} \leftarrow \mathbf{w} + \alpha \delta_t \mathbf{x}(S_t)$

### Why "Semi-Gradient"?

The TD target $R + \gamma \hat{V}(S'; \mathbf{w})$ also depends on $\mathbf{w}$, but we don't differentiate through it. This makes the update a **semi-gradient** — not a true gradient of any objective.

## Linear TD Convergence

For on-policy linear TD(0) with fixed policy $\pi$, the weights converge to the **TD fixed point**:

$$\mathbf{w}_{TD} = \mathbf{A}^{-1} \mathbf{b}$$

where:
- $\mathbf{A} = \mathbb{E}[\mathbf{x}(S_t)(\mathbf{x}(S_t) - \gamma \mathbf{x}(S_{t+1}))^\top]$
- $\mathbf{b} = \mathbb{E}[R_{t+1} \mathbf{x}(S_t)]$

The TD fixed point satisfies an error bound:

$$\|V_{TD} - V_\pi\|_d \leq \frac{1}{\sqrt{1-\gamma^2}} \min_\mathbf{w} \|V_\mathbf{w} - V_\pi\|_d$$

where $\|\cdot\|_d$ is the norm weighted by the stationary distribution.

## Tabular as Special Case

Tabular methods are a special case of linear function approximation where $\mathbf{x}(s) = \mathbf{e}_s$ (one-hot encoding). Each weight $w_s$ corresponds to $V(s)$.

## Advantages and Limitations

| Advantage | Limitation |
|-----------|-----------|
| Generalization across states | Restricted to linear value surfaces |
| Convergence guarantees (on-policy) | May diverge off-policy (deadly triad) |
| Computational efficiency O(d) | Feature engineering required |
| Interpretable weights | Cannot represent complex value functions |

## Financial Application

Linear function approximation is natural for factor models in finance:

$$\hat{V}(\text{market state}; \mathbf{w}) = w_1 \cdot \text{momentum} + w_2 \cdot \text{volatility} + w_3 \cdot \text{value} + \cdots$$

The weights $w_j$ represent the sensitivity of strategy value to each market factor — analogous to factor loadings in asset pricing models.
