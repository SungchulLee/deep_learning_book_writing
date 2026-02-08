# 33.1.4 DQN Implementation

## Complete DQN Algorithm

The full DQN algorithm combines Q-learning with experience replay and target networks:

### Algorithm: Deep Q-Network (DQN)

```
Initialize online network Q_θ with random weights θ
Initialize target network Q_{θ⁻} with weights θ⁻ = θ
Initialize replay buffer D with capacity N
Initialize exploration schedule ε

for episode = 1 to M:
    s ← env.reset()
    for t = 1 to T:
        # Action selection
        With probability ε: a ← random action
        Otherwise: a ← argmax_a Q_θ(s, a)
        
        # Environment interaction
        s', r, done ← env.step(a)
        
        # Store transition
        D.push(s, a, r, s', done)
        
        # Learn from replay
        if |D| ≥ min_buffer_size:
            Sample mini-batch {(s_i, a_i, r_i, s'_i, d_i)} from D
            
            # Compute targets
            y_i = r_i + (1 - d_i) · γ · max_{a'} Q_{θ⁻}(s'_i, a')
            
            # Update online network
            L = (1/B) Σ (y_i - Q_θ(s_i, a_i))²
            θ ← θ - α ∇_θ L
            
            # Update target network (every C steps)
            if step_count % C == 0:
                θ⁻ ← θ
        
        s ← s'
        ε ← decay(ε)
```

## Implementation Details

### Gradient Clipping

Gradient clipping prevents exploding gradients that can destabilize training:

```python
nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
```

Alternatively, Huber loss (smooth L1) provides implicit gradient clipping:

$$\mathcal{L}_\text{Huber}(\delta) = \begin{cases} \frac{1}{2}\delta^2 & |\delta| \leq 1 \\ |\delta| - \frac{1}{2} & |\delta| > 1 \end{cases}$$

### Network Initialization

Xavier/Glorot initialization is standard for DQN:
```python
for layer in q_network.modules():
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
```

### Frame Preprocessing (Atari)

For image-based environments:
1. Convert to grayscale
2. Resize to 84×84
3. Stack last 4 frames (provides velocity information)
4. Clip rewards to [-1, 1] (reward clipping)

### Training Stability Tips

1. **Warm-up period**: Fill buffer with random transitions before training (e.g., 10K steps)
2. **Learning rate**: Start with 1e-4 for Atari, 1e-3 for simpler environments
3. **Batch size**: 32 for basic DQN, 64–256 for larger problems
4. **Target update frequency**: Start with $C = 1000$, tune based on convergence speed
5. **Gradient clipping**: Clip at 10.0 or use Huber loss
6. **Evaluation frequency**: Evaluate every N episodes with ε=0 (greedy policy)

## Logging and Monitoring

Effective DQN training requires monitoring:
- **Episode rewards**: Primary performance metric (use rolling average)
- **Loss**: Should decrease but may be noisy
- **Q-value estimates**: Mean Q-values should increase but not diverge
- **Epsilon**: Verify annealing schedule
- **Buffer utilization**: Ensure buffer is filling properly
- **Gradient norms**: Detect exploding/vanishing gradients

## Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Reward doesn't improve | Epsilon too high / not decaying | Check schedule |
| Sudden reward collapse | Target net update too frequent | Increase C |
| Q-values diverge to ±∞ | Learning rate too high | Reduce LR, add gradient clipping |
| Training very slow | Buffer too small / batch too small | Increase both |
| Reward oscillates | Deadly triad instability | Double DQN, reduce LR |
