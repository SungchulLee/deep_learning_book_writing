# GRU Gates: Update and Reset

## Introduction

The Gated Recurrent Unit simplifies LSTM's three gates into two: the **update gate** and **reset gate**. This reduction isn't just about fewer parameters—it represents a different philosophy of information control that proves equally effective for many tasks.

## Gate Overview

| Gate | Symbol | Purpose | LSTM Equivalent |
|------|--------|---------|-----------------|
| Update | $z_t$ | Balance old vs new information | Forget + Input combined |
| Reset | $r_t$ | Control previous state influence on candidate | No direct equivalent |

## The Update Gate

### Purpose

The update gate decides how much of the previous hidden state to keep versus how much to replace with new information. It performs the roles of both LSTM's forget and input gates.

### Computation

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

### Application

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

This is a **convex combination**: the coefficients $(1-z_t)$ and $z_t$ sum to 1.

### Interpretation

| $z_t$ Value | Behavior |
|-------------|----------|
| $z_t \approx 0$ | Keep previous state (copy mode) |
| $z_t \approx 1$ | Use new candidate (update mode) |
| $z_t \approx 0.5$ | Blend equally |

### Key Insight

Unlike LSTM where forget and input gates are independent (can both be high or both be low), GRU enforces a trade-off: more remembering means less updating, and vice versa.

```python
import torch
import torch.nn as nn

def demonstrate_update_gate():
    """Show how update gate controls information flow."""
    hidden_size = 4
    
    # Previous state and candidate
    h_prev = torch.tensor([1.0, 2.0, 3.0, 4.0])
    h_tilde = torch.tensor([5.0, 6.0, 7.0, 8.0])
    
    scenarios = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print("Update Gate Effect:")
    print(f"h_prev:  {h_prev.tolist()}")
    print(f"h_tilde: {h_tilde.tolist()}")
    print()
    
    for z in scenarios:
        z_t = torch.full((hidden_size,), z)
        h_new = (1 - z_t) * h_prev + z_t * h_tilde
        print(f"z={z}: h_new = {h_new.tolist()}")

demonstrate_update_gate()
```

## The Reset Gate

### Purpose

The reset gate controls how much of the previous hidden state influences the computation of the new candidate. When reset is low, the candidate is computed almost entirely from the current input—effectively "resetting" the memory.

### Computation

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

### Application (in Candidate Computation)

$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

The reset gate multiplies the previous hidden state *before* it's used to compute the candidate.

### Interpretation

| $r_t$ Value | Behavior |
|-------------|----------|
| $r_t \approx 0$ | Ignore previous state when computing candidate |
| $r_t \approx 1$ | Fully consider previous state |

### Key Insight

The reset gate allows GRU to detect "discontinuities" in sequences—moments where past context becomes irrelevant (e.g., sentence boundaries, topic changes).

```python
def demonstrate_reset_gate():
    """Show how reset gate affects candidate computation."""
    
    h_prev = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x_t = torch.tensor([0.5, 0.5, 0.5, 0.5])
    
    # Simplified: candidate ≈ tanh(W_h @ [r * h_prev, x_t])
    # For demonstration, assume linear combination
    
    print("Reset Gate Effect on Candidate:")
    print(f"h_prev: {h_prev.tolist()}")
    print(f"x_t:    {x_t.tolist()}")
    print()
    
    for r in [0.0, 0.5, 1.0]:
        r_t = torch.full((4,), r)
        reset_h = r_t * h_prev
        print(f"r={r}: r*h_prev = {reset_h.tolist()}")
        # Candidate would be computed from [reset_h, x_t]

demonstrate_reset_gate()
```

## Gate Interaction: The Full Picture

```python
def gru_step_detailed(x_t, h_prev, W_z, W_r, W_h, U_z, U_r, U_h, b_z, b_r, b_h):
    """
    Detailed GRU step showing gate interactions.
    
    Args:
        x_t: Current input
        h_prev: Previous hidden state
        W_*: Input weight matrices
        U_*: Recurrent weight matrices
        b_*: Biases
    """
    # Step 1: Update gate - how much to update?
    z_t = torch.sigmoid(W_z @ x_t + U_z @ h_prev + b_z)
    
    # Step 2: Reset gate - how much history for candidate?
    r_t = torch.sigmoid(W_r @ x_t + U_r @ h_prev + b_r)
    
    # Step 3: Candidate - what could the new state be?
    # Reset gate applied HERE, before candidate computation
    h_tilde = torch.tanh(W_h @ x_t + U_h @ (r_t * h_prev) + b_h)
    
    # Step 4: Final state - interpolate based on update gate
    h_t = (1 - z_t) * h_prev + z_t * h_tilde
    
    return h_t, {'z': z_t, 'r': r_t, 'h_tilde': h_tilde}
```

## Visualizing GRU Gates

```python
def extract_gru_gates(gru, x):
    """
    Extract gate activations from a GRU for visualization.
    
    Args:
        gru: nn.GRU module
        x: Input tensor (batch, seq_len, input_size)
    
    Returns:
        Dictionary with update and reset gate activations
    """
    batch_size, seq_len, input_size = x.shape
    hidden_size = gru.hidden_size
    
    # Get weights (PyTorch GRU weight order: reset, update, new)
    W_ir, W_iz, W_in = gru.weight_ih_l0.chunk(3, 0)
    W_hr, W_hz, W_hn = gru.weight_hh_l0.chunk(3, 0)
    b_ir, b_iz, b_in = gru.bias_ih_l0.chunk(3, 0)
    b_hr, b_hz, b_hn = gru.bias_hh_l0.chunk(3, 0)
    
    update_gates = []
    reset_gates = []
    
    h = torch.zeros(batch_size, hidden_size)
    
    for t in range(seq_len):
        x_t = x[:, t, :]
        
        # Reset gate
        r = torch.sigmoid(x_t @ W_ir.T + h @ W_hr.T + b_ir + b_hr)
        
        # Update gate
        z = torch.sigmoid(x_t @ W_iz.T + h @ W_hz.T + b_iz + b_hz)
        
        # Candidate
        n = torch.tanh(x_t @ W_in.T + (r * h) @ W_hn.T + b_in + b_hn)
        
        # New hidden state
        h = (1 - z) * h + z * n
        
        reset_gates.append(r)
        update_gates.append(z)
    
    return {
        'update': torch.stack(update_gates, dim=1),
        'reset': torch.stack(reset_gates, dim=1)
    }


def plot_gru_gates(gates, tokens=None):
    """Plot GRU gate activations."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, (name, data) in zip(axes, [('Update Gate', gates['update']), 
                                        ('Reset Gate', gates['reset'])]):
        im = ax.imshow(data[0].detach().numpy().T, aspect='auto', 
                       cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Hidden Dimension')
        
        if tokens:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
        
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()
```

## Comparing GRU and LSTM Gates

### Functional Equivalence

GRU's update gate combines LSTM's forget and input gates:

| GRU | LSTM | Relationship |
|-----|------|--------------|
| $z_t$ | $f_t$, $i_t$ | $z_t \approx i_t$ and $(1-z_t) \approx f_t$ |
| $r_t$ | — | No direct equivalent |
| — | $o_t$ | GRU has no output gate |

### Key Differences

1. **Coupling**: GRU couples forgetting and updating; LSTM allows independent control
2. **Output filtering**: LSTM can hide cell state via output gate; GRU exposes full state
3. **Parameter count**: GRU has ~25% fewer parameters

```python
def compare_gate_behavior():
    """Compare LSTM and GRU gate behavior on same input."""
    
    input_size = 32
    hidden_size = 64
    seq_length = 20
    
    lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    gru = nn.GRU(input_size, hidden_size, batch_first=True)
    
    x = torch.randn(1, seq_length, input_size)
    
    # Extract gates (simplified - would need full extraction code)
    lstm_out, _ = lstm(x)
    gru_out, _ = gru(x)
    
    print(f"LSTM output shape: {lstm_out.shape}")
    print(f"GRU output shape: {gru_out.shape}")
    print(f"LSTM parameters: {sum(p.numel() for p in lstm.parameters())}")
    print(f"GRU parameters: {sum(p.numel() for p in gru.parameters())}")
```

## Gate Patterns and Interpretations

### Common Patterns

**Sequence Boundaries** (e.g., end of sentence):
- Reset gate $\to$ low (ignore past)
- Update gate $\to$ high (accept new)

**Continuing Context**:
- Reset gate $\to$ high (use past)
- Update gate $\to$ variable (blend as needed)

**Important Information**:
- Reset gate $\to$ high
- Update gate $\to$ high (store in state)

**Irrelevant Input**:
- Update gate $\to$ low (preserve state)

### Diagnostic Analysis

```python
def analyze_gate_patterns(update_gates, reset_gates, threshold=0.1):
    """
    Analyze gate activation patterns.
    
    Args:
        update_gates: (seq_len, hidden_size)
        reset_gates: (seq_len, hidden_size)
    """
    seq_len = update_gates.shape[0]
    
    patterns = {
        'copy': 0,      # z low (preserve state)
        'reset': 0,     # z high, r low (fresh start)
        'update': 0,    # z high, r high (use context)
        'blend': 0      # intermediate values
    }
    
    for t in range(seq_len):
        z_avg = update_gates[t].mean().item()
        r_avg = reset_gates[t].mean().item()
        
        if z_avg < threshold:
            patterns['copy'] += 1
        elif z_avg > (1 - threshold) and r_avg < threshold:
            patterns['reset'] += 1
        elif z_avg > (1 - threshold) and r_avg > (1 - threshold):
            patterns['update'] += 1
        else:
            patterns['blend'] += 1
    
    print("Gate Pattern Analysis:")
    for pattern, count in patterns.items():
        print(f"  {pattern}: {count}/{seq_len} ({100*count/seq_len:.1f}%)")
    
    return patterns
```

## Summary

GRU's two gates provide efficient sequence modeling:

1. **Update gate** ($z_t$): Interpolates between old and new states
   - Combines LSTM's forget and input gate functions
   - Enforces complementary forgetting/updating

2. **Reset gate** ($r_t$): Controls history influence on candidate
   - Enables detection of sequence discontinuities
   - Applied before candidate computation

Key insights:
- Fewer parameters than LSTM (faster training)
- Coupled gates simplify learning dynamics
- Often achieves similar performance to LSTM
- Better for smaller datasets (less prone to overfitting)
