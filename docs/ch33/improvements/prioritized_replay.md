# 33.2.3 Prioritized Experience Replay

## Motivation

Uniform replay treats all transitions equally, but some transitions are more informative than others. Transitions with high **TD error** indicate surprising outcomes—the agent's prediction was far from the target—and are thus more useful for learning.

## Prioritized Replay

**Prioritized Experience Replay (PER)** (Schaul et al., 2016) samples transitions with probability proportional to their TD error magnitude:

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

where $p_i = |\delta_i| + \epsilon$ is the priority of transition $i$, $\delta_i$ is its TD error, $\epsilon$ is a small constant preventing zero probability, and $\alpha$ controls how much prioritization is used ($\alpha = 0$ is uniform, $\alpha = 1$ is full prioritization).

## Importance Sampling Correction

Prioritized sampling changes the distribution over transitions, introducing bias. To correct this, PER uses **importance sampling (IS) weights**:

$$w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$$

where $\beta$ controls the correction strength. $\beta$ is annealed from a small value (e.g., 0.4) to 1.0 over training. The loss becomes:

$$\mathcal{L} = \frac{1}{B}\sum_{i \in \mathcal{B}} w_i \cdot (y_i - Q_\theta(s_i, a_i))^2$$

Weights are normalized by $\max_i w_i$ to prevent very large weight values.

## Implementation: Sum Tree

Efficient prioritized sampling requires $O(\log N)$ operations. A **sum tree** data structure stores priorities in leaves and partial sums in internal nodes, enabling:
- **Sampling**: Draw $u \sim \text{Uniform}(0, \text{total\_priority})$, then traverse tree to find the corresponding leaf
- **Update**: After computing new TD errors, update leaf priorities and propagate sums up the tree

Both operations are $O(\log N)$ vs $O(N)$ for naive approaches.

## Hyperparameters

| Parameter | Typical value | Effect |
|-----------|--------------|--------|
| $\alpha$ | 0.6 | Priority exponent (0=uniform, 1=full) |
| $\beta_0$ | 0.4 | Initial IS correction |
| $\beta$ annealing | 0.4 → 1.0 | Linear over training |
| $\epsilon$ | $10^{-6}$ | Minimum priority |

## Benefits and Limitations

**Benefits:**
- Faster learning on difficult transitions
- Better sample efficiency (3–4× on some Atari games)
- Focuses learning where the model is most wrong

**Limitations:**
- Increased computational overhead ($O(\log N)$ per sample instead of $O(1)$)
- Can overfit to outlier transitions
- IS weights can be unstable early in training
- Diversity loss: "easy" transitions are rarely revisited

## Finance Considerations

In financial RL, prioritized replay naturally focuses on:
- Market regime changes (high TD error at transitions between bull/bear)
- Extreme events (crashes, spikes) which are rare but impactful
- Incorrect position sizing (large losses from wrong actions)

However, care must be taken that the agent doesn't overfit to tail events at the expense of steady-state behavior.
