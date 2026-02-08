# 33.1.2 Experience Replay

## Motivation

In online RL, an agent learns from transitions as they arrive sequentially. This creates two problems:

1. **Temporal correlation**: Consecutive transitions are highly correlated (states change incrementally), violating the i.i.d. assumption required by stochastic gradient descent
2. **Data inefficiency**: Each transition is used once and discarded

**Experience replay** solves both problems by storing transitions in a buffer and sampling random mini-batches for training.

## The Replay Buffer

A replay buffer $\mathcal{D}$ stores the most recent $N$ transitions as tuples:

$$\mathcal{D} = \{(s_i, a_i, r_i, s'_i, d_i)\}_{i=1}^{N}$$

where $d_i$ is a terminal flag. At each training step, a mini-batch $\mathcal{B} \sim \text{Uniform}(\mathcal{D})$ of size $B$ is sampled uniformly at random.

### Buffer Properties

| Property | Typical Value | Effect |
|----------|--------------|--------|
| Capacity $N$ | $10^5$ to $10^6$ | Larger → more diverse data, more memory |
| Batch size $B$ | 32 to 256 | Larger → lower variance, more compute |
| Minimum size before training | $10^3$ to $5 \times 10^4$ | Ensures diverse initial samples |

## Why Uniform Sampling Works

Uniform sampling provides several benefits:

- **Decorrelation**: Random sampling breaks temporal dependencies between consecutive transitions
- **Data reuse**: Important transitions may be sampled multiple times, improving sample efficiency
- **Stability**: The effective training distribution changes slowly as new data enters the buffer

## Implementation Considerations

### Circular Buffer vs. Deque
- **Deque**: Simple, automatically discards oldest entries; Python's `deque(maxlen=N)` is efficient
- **NumPy arrays**: Pre-allocated arrays with index wrapping; more memory-efficient for large buffers, avoids Python object overhead

### Memory Management
For Atari-style environments with 84×84×4 frames:
- Each state: $84 \times 84 \times 4 \times 1$ byte = 28 KB
- Buffer of $10^6$: ~28 GB (states + next_states)
- **Optimization**: Store frames once, reconstruct stacked states on-the-fly using frame indices

### Batch Construction
Efficient batching requires converting Python objects to tensors:
```
Sample indices → Gather transitions → Stack into tensors → Transfer to GPU
```

Avoiding redundant copies and using pinned memory can significantly speed up training.

## Variants of Experience Replay

### Standard (Uniform) Replay
- Samples uniformly from buffer
- Simple and effective baseline
- Used in the original DQN paper

### Prioritized Experience Replay (Section 33.2.3)
- Samples proportional to TD error magnitude
- Focuses learning on "surprising" transitions
- Requires importance sampling correction

### Hindsight Experience Replay (HER)
- For goal-conditioned RL
- Relabels failed trajectories with achieved goals
- Dramatically improves sample efficiency in sparse-reward settings

### Combined Experience Replay (CER)
- Always includes the most recent transition in each batch
- Ensures new data is immediately used for learning
- Marginal improvement over uniform replay

## Off-Policy Learning Connection

Experience replay inherently makes DQN an **off-policy** algorithm: the agent learns from transitions collected by previous versions of its policy (stored in the buffer). This is valid because Q-learning's update rule doesn't depend on the behavior policy:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

The max operator selects the greedy action regardless of how $(s, a, r, s')$ was generated.

## Trade-offs

### Buffer Too Small
- Rapid forgetting of useful transitions
- Higher correlation between sampled transitions
- Effectively approaches online learning

### Buffer Too Large
- Stale data from very old policies
- Slower adaptation to distribution shift
- More memory consumption

### Practical Guideline
A buffer holding the last 100K–1M transitions typically balances diversity and relevance. For financial applications, the buffer size should be calibrated to the regime length: if market conditions change every ~6 months, a buffer spanning 1–2 years provides sufficient diversity while not including overly stale data.
