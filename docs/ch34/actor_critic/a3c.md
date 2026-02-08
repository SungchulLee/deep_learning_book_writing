# 34.2.3 Asynchronous Advantage Actor-Critic (A3C)

## Introduction

A3C (Mnih et al., 2016) was a landmark algorithm that demonstrated effective deep RL training on a single multi-core CPU by running multiple actor-learner workers asynchronously. Each worker interacts with its own environment instance, computes gradients locally, and applies them to a shared global network. While largely superseded by A2C and PPO for GPU-based training, A3C introduced key ideas that remain influential.

## Algorithm

### A3C Architecture

A3C uses a **global shared network** with parameters $\theta$ and $\phi$ (actor and critic), plus $N$ worker threads, each with local parameter copies.

### A3C Pseudocode

```
Initialize global shared network parameters θ, φ
Initialize global shared optimizer

Parallel for each worker w = 1, ..., N:
    Initialize local network θ_w, φ_w
    Loop:
        Sync local parameters: θ_w ← θ, φ_w ← φ
        Collect T-step rollout using π_{θ_w}
        
        Compute advantages:
            If terminal: R = 0
            Else: R = V_{φ_w}(s_T)
            For t = T-1, ..., 0:
                R = r_t + γR
                A_t = R - V_{φ_w}(s_t)
        
        Compute gradients:
            dθ = ∇_{θ_w} Σ_t [-log π_{θ_w}(a_t|s_t) · A_t + c_e · H(π_{θ_w}(·|s_t))]
            dφ = ∇_{φ_w} Σ_t (R_t - V_{φ_w}(s_t))²
        
        Apply gradients to global network:
            θ ← θ - α · dθ
            φ ← φ - α · dφ
```

## Key Innovations

### Asynchronous Parallelism

Multiple workers explore different parts of the state space simultaneously. This provides:

1. **Decorrelated data**: Different workers encounter different transitions, reducing correlation in gradient updates
2. **Implicit exploration**: Workers with different random seeds explore diverse trajectories
3. **CPU efficiency**: No GPU required; workers use CPU threads

### Gradient Asynchrony

Workers compute gradients using potentially stale parameters. This staleness introduces noise but empirically does not prevent convergence:

$$\theta_\text{global} \leftarrow \theta_\text{global} - \alpha \nabla_{\theta_w} L(\theta_w)$$

The gradient $\nabla_{\theta_w} L$ is computed with parameters $\theta_w$ that may be several updates behind $\theta_\text{global}$.

### No Replay Buffer

Unlike DQN which requires a replay buffer for decorrelation, A3C achieves decorrelation through parallel workers. This makes A3C:
- Purely on-policy (no off-policy corrections needed)
- Memory efficient (no replay buffer storage)
- Simple to implement (no prioritization or importance sampling)

## Implementation Considerations

### Thread Safety

The global network parameters must be accessed safely:
- **Shared memory**: Use `torch.multiprocessing` with shared tensors
- **Lock-free updates**: Hogwild-style updates work because gradient updates are commutative to first order
- **Shared optimizer**: The global optimizer's state (e.g., Adam momentum) must also be shared

### Worker Synchronization

Each worker cycle:
1. Copy global → local parameters
2. Collect rollout with local parameters
3. Compute local gradients
4. Apply gradients to global parameters

No explicit synchronization between workers is needed.

### Multiprocessing in PyTorch

```python
import torch.multiprocessing as mp

# Global model with shared memory
global_model = ActorCritic(obs_dim, act_dim)
global_model.share_memory()

# Shared optimizer
optimizer = SharedAdam(global_model.parameters())

# Spawn workers
processes = []
for rank in range(n_workers):
    p = mp.Process(target=worker, args=(rank, global_model, optimizer))
    p.start()
    processes.append(p)
```

## A3C vs. A2C Trade-offs

| Aspect | A3C | A2C |
|--------|-----|-----|
| Hardware | CPU (multi-core) | GPU (batched) |
| Gradient staleness | Present | None |
| Implementation | Complex (multiprocessing) | Simple (vectorized) |
| Reproducibility | Difficult (non-deterministic) | Easy (deterministic) |
| Throughput | Lower per step | Higher per step |
| Modern usage | Rarely used | Common baseline |

## Historical Significance

A3C was significant for several reasons:
- Demonstrated that on-policy methods could be competitive with DQN
- Showed that parallel exploration could replace replay buffers
- Trained Atari games on a single CPU machine
- Introduced the actor-critic with shared features paradigm
- Laid the groundwork for PPO and other modern methods

## Limitations

- **Gradient staleness**: Workers may compute gradients with outdated parameters
- **Reproducibility**: Non-deterministic execution order prevents exact reproduction
- **GPU inefficiency**: Sequential per-worker updates cannot leverage GPU parallelism
- **Complexity**: Multiprocessing introduces debugging challenges

## Summary

A3C pioneered asynchronous deep RL training with parallel workers. While its synchronous variant A2C has become the preferred implementation, A3C's key insight—using parallel environments for decorrelation—remains a cornerstone of modern policy gradient methods. The shared actor-critic architecture and n-step advantage computation introduced by A3C continue to be used in state-of-the-art algorithms.
