# 33.2.4 Noisy Networks

## Motivation

Epsilon-greedy exploration is simple but inefficient: it explores uniformly over all actions regardless of state. **Noisy Networks** (Fortunato et al., 2018) replace ε-greedy with learned, state-dependent exploration by adding parametric noise to network weights.

## Approach

Replace standard linear layers $y = Wx + b$ with noisy linear layers:

$$y = (\mu^W + \sigma^W \odot \epsilon^W)x + (\mu^b + \sigma^b \odot \epsilon^b)$$

where:
- $\mu^W, \mu^b$: Learnable mean parameters (like standard weights/biases)
- $\sigma^W, \sigma^b$: Learnable noise scale parameters
- $\epsilon^W, \epsilon^b$: Random noise sampled at each forward pass
- $\odot$: Element-wise multiplication

## Noise Variants

### Independent Gaussian Noise
- Each weight has its own noise: $\epsilon_{ij} \sim \mathcal{N}(0, 1)$
- Number of noise parameters: $p \times q + q$ for a $(p, q)$ layer
- More expressive but slower

### Factorized Gaussian Noise (Recommended)
- Factor noise as: $\epsilon_{ij} = f(\epsilon_i) \cdot f(\epsilon_j)$
- where $f(x) = \text{sign}(x)\sqrt{|x|}$
- Number of noise vectors: $p + q$ instead of $p \times q$
- Much more efficient with minimal performance loss

## Key Properties

1. **State-dependent exploration**: The noise is applied to weights, so different states produce different exploration patterns
2. **Learned exploration**: $\sigma$ parameters shrink in states where the agent is confident, increasing exploitation
3. **No hyperparameter**: Eliminates the need for ε schedule tuning
4. **Self-annealing**: As the agent learns, $\sigma$ values naturally decrease

## Implementation Notes

- Only replace the final layer(s) of the Q-network (typically the last 2 linear layers)
- Resample noise at the beginning of each forward pass (or each episode)
- For evaluation, use zero noise (mean parameters only)
- Initialize $\sigma$ to a small constant (e.g., 0.5 for factorized noise)
- $\mu$ is initialized like standard layers (uniform in $[-1/\sqrt{p}, 1/\sqrt{p}]$)

## Comparison with ε-Greedy

| Aspect | ε-Greedy | Noisy Networks |
|--------|----------|---------------|
| Exploration type | Action-space uniform | State-dependent, parameter space |
| Hyperparameters | ε start, end, decay | Initial σ only |
| Annealing | Manual schedule | Automatic (learned) |
| Computational cost | Negligible | Small overhead per forward pass |
| Compatibility | Any agent | Replaces ε-greedy entirely |
