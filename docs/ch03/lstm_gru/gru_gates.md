# GRU Gates: Update and Reset

## Introduction

GRU achieves LSTM-like performance with only two gates instead of three. This architectural simplification is not merely a reduction—it represents a fundamentally different approach to memory management. While LSTM uses separate mechanisms for forgetting, adding, and outputting information, GRU couples these operations through its reset and update gates.

## Gate Overview

| Gate | Symbol | Purpose | Output Range | LSTM Equivalent |
|------|--------|---------|--------------|-----------------|
| Update | $z_t$ | Balance old vs new information | $(0, 1)$ | Forget + Input combined |
| Reset | $r_t$ | Control previous state influence on candidate | $(0, 1)$ | No direct equivalent |

## The Two-Gate Philosophy

GRU's design philosophy centers on complementary gates:

1. **Update Gate ($z_t$)**: Controls the balance between old and new information
2. **Reset Gate ($r_t$)**: Controls how the previous state influences new candidate information

These gates are **complementary**: the update gate decides "how much to change," while the reset gate decides "what to consider when computing the change."

---

## The Update Gate

### Purpose and Mechanism

The update gate performs **gated interpolation** between the previous hidden state and a new candidate:

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

This single equation replaces LSTM's separate forget and input gates. The coupling is explicit:
- Amount retained: $(1 - z_t)$
- Amount added: $z_t$

By construction: $(1 - z_t) + z_t = 1$, so the total "weight" on states is conserved. This is a **convex combination**.

### Interpretation as Forget + Input

| GRU Component | LSTM Equivalent | Function |
|---------------|-----------------|----------|
| $(1 - z_t)$ | Forget gate $f_t$ | Retain old information |
| $z_t$ | Input gate $i_t$ | Add new information |

The key difference: LSTM allows $f_t + i_t \neq 1$, giving more flexibility but also more parameters.

### Update Gate Behavior

| $z_t$ Value | Behavior | Use Case |
|-------------|----------|----------|
| $z_t \approx 0$ | Keep previous state (copy mode) | Noise, irrelevant info, long-term memory |
| $z_t \approx 0.5$ | Blend equally | Gradual transitions |
| $z_t \approx 1$ | Use new candidate (update mode) | Topic changes, sentence boundaries |

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def visualize_update_gate():
    """Visualize how update gate creates interpolation."""
    
    # Previous and candidate states
    h_prev = torch.tensor([1.0, 0.5, -0.3, 0.8, -0.5])
    h_tilde = torch.tensor([-0.2, 0.8, 0.5, -0.3, 0.9])
    
    print("Update Gate Interpolation:")
    print(f"h_prev:  {h_prev.tolist()}")
    print(f"h_tilde: {h_tilde.tolist()}")
    print()
    
    for z in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z_t = torch.full_like(h_prev, z)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        print(f"z={z:.2f}: h_t = {[f'{v:.2f}' for v in h_t.tolist()]}")

visualize_update_gate()
```

### When Update Gate is Low ($z \approx 0$)

When $z_t \approx 0$:
- $h_t \approx h_{t-1}$ (state persists)
- Network "ignores" current input
- Useful for: noise, irrelevant information, long-term memory preservation

**Example scenario**: Processing "The cat, which was very fluffy, sat..."
- During "which was very fluffy" (relative clause), $z_t$ can stay low
- The main subject information persists unchanged

### When Update Gate is High ($z \approx 1$)

When $z_t \approx 1$:
- $h_t \approx \tilde{h}_t$ (state replaced)
- Network "resets" to focus on new information
- Useful for: topic changes, sentence boundaries, important events

---

## The Reset Gate

### Purpose and Mechanism

The reset gate controls how the previous state influences the candidate computation:

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

The reset gate acts **before** the candidate is computed, unlike the update gate which acts **after**. This allows GRU to detect "discontinuities" in sequences—moments where past context becomes irrelevant.

### Reset Gate Interpretation

| Reset Value | Effect | Use Case |
|-------------|--------|----------|
| $r_t \approx 0$ | Ignore previous state when computing candidate | Fresh start, new topic |
| $r_t \approx 1$ | Fully consider previous state | Continuing trend, smooth transitions |
| $r_t$ selective | Partial relevance | Some context useful, some not |

```python
def demonstrate_reset_gate():
    """Show how reset gate affects candidate computation."""
    
    h_prev = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x_t = torch.tensor([0.5, 0.5, 0.5, 0.5])
    
    print("Reset Gate Effect on Candidate:")
    print(f"h_prev: {h_prev.tolist()}")
    print(f"x_t:    {x_t.tolist()}")
    print()
    
    for r in [0.0, 0.5, 1.0]:
        r_t = torch.full((4,), r)
        reset_h = r_t * h_prev
        print(f"r={r}: r*h_prev = {[f'{v:.1f}' for v in reset_h.tolist()]}")
        # Candidate would be computed from [reset_h, x_t]

demonstrate_reset_gate()
```

---

## Reset vs Update: Division of Labor

The two gates operate at different stages of the computation:

```
Previous State h_{t-1}
        |
        ├──────────────────────────────────────┐
        |                                       |
        v                                       v
   [Reset Gate]                          [Update Gate]
        |                                       |
        v                                       |
   r_t ⊙ h_{t-1}                               |
        |                                       |
        v                                       |
   Candidate h̃_t = tanh(W_h·[r⊙h, x])          |
        |                                       |
        └──────────────────────> [Interpolation] <─┘
                                        |
                                        v
                                New State h_t
```

**Reset gate**: Decides what previous information to *consider* when computing new candidate
**Update gate**: Decides how to *blend* old state with new candidate

### Complete GRU Step

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

### Gate Interaction Scenarios

```python
def demonstrate_reset_update_interaction():
    """Show how reset and update gates work together."""
    
    hidden_size = 4
    input_size = 3
    
    torch.manual_seed(42)
    
    # Initialize weights
    W_h = torch.randn(hidden_size, hidden_size + input_size) * 0.1
    
    # Previous state and input
    h_prev = torch.tensor([0.8, -0.5, 0.3, 0.9])
    x = torch.tensor([0.5, -0.2, 0.7])
    
    scenarios = [
        ("Low reset, high update (fresh start)", 
         torch.tensor([0.1, 0.1, 0.1, 0.1]), 
         torch.tensor([0.9, 0.9, 0.9, 0.9])),
        ("High reset, low update (preserve with context)", 
         torch.tensor([0.9, 0.9, 0.9, 0.9]), 
         torch.tensor([0.1, 0.1, 0.1, 0.1])),
        ("Mixed gates (selective processing)", 
         torch.tensor([0.2, 0.8, 0.3, 0.7]), 
         torch.tensor([0.6, 0.4, 0.7, 0.3])),
    ]
    
    def compute_gru_step(h_prev, x, r, z, W_h):
        h_reset = r * h_prev
        combined_reset = torch.cat([h_reset, x])
        h_tilde = torch.tanh(W_h @ combined_reset)
        h_new = (1 - z) * h_prev + z * h_tilde
        return h_new, h_tilde
    
    print("Reset-Update Gate Interaction:")
    print(f"h_prev: {[f'{v:.2f}' for v in h_prev.tolist()]}")
    print(f"x:      {[f'{v:.2f}' for v in x.tolist()]}")
    print()
    
    for name, r, z in scenarios:
        h_new, h_tilde = compute_gru_step(h_prev, x, r, z, W_h)
        print(f"{name}:")
        print(f"  Reset r:     {[f'{v:.1f}' for v in r.tolist()]}")
        print(f"  Update z:    {[f'{v:.1f}' for v in z.tolist()]}")
        print(f"  Candidate:   {[f'{v:.2f}' for v in h_tilde.tolist()]}")
        print(f"  New state:   {[f'{v:.2f}' for v in h_new.tolist()]}")
        print()

demonstrate_reset_update_interaction()
```

---

## Gate Dynamics Across Sequences

Different sequence patterns trigger different gate behaviors:

| Scenario | Update Gate $z$ | Reset Gate $r$ | Effect |
|----------|-----------------|----------------|--------|
| Steady topic | 0.2–0.4 (mostly retain) | 0.7–0.9 (consider history) | Smooth continuation |
| Topic change | 0.8–0.9 (replace state) | 0.1–0.3 (ignore old) | Fresh start |
| Gradual transition | 0.4–0.6 (balanced) | 0.4–0.6 (partial) | Blended update |
| Important info | High | High | Store with context |
| Irrelevant input | Low | Any | Preserve state |

### Common Patterns

**Sequence Boundaries** (e.g., end of sentence):
- Reset gate → low (ignore past)
- Update gate → high (accept new)

**Continuing Context**:
- Reset gate → high (use past)
- Update gate → variable (blend as needed)

**Important Information**:
- Reset gate → high
- Update gate → high (store in state)

---

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
    """Plot GRU gate activations as heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, (name, data) in zip(axes, [('Update Gate (z)', gates['update']), 
                                        ('Reset Gate (r)', gates['reset'])]):
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

---

## Gradient Flow Through GRU Gates

A key benefit of gated architectures: they enable gradient flow even when gates are partially "closed."

### Update Gate Gradient Path

The update gate creates a direct gradient path:

$$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1 - z_t) + z_t \odot \frac{\partial \tilde{h}_t}{\partial h_{t-1}}$$

When $z_t \approx 0$, the first term dominates, giving near-identity Jacobian. This allows gradients to flow almost unattenuated through time.

### Reset Gate Gradient Path

The reset gate affects gradients through the candidate:

$$\frac{\partial \tilde{h}_t}{\partial h_{t-1}} = \text{diag}(\tilde{h}'_t) \cdot W_h^{(h)} \cdot \text{diag}(r_t)$$

where $\tilde{h}'_t = 1 - \tilde{h}_t^2$ (tanh derivative).

### Comparison: GRU vs Vanilla RNN

```python
def analyze_gru_gradients():
    """Analyze gradient flow through GRU gates vs vanilla RNN."""
    
    sequence_length = 100
    
    # Simulate typical gate values
    np.random.seed(42)
    
    # Update gates (typically biased low to preserve state)
    z_values = 0.2 + 0.1 * np.random.randn(sequence_length)
    z_values = np.clip(z_values, 0.05, 0.95)
    
    # Gradient through direct path (1-z)
    gru_gradient = 1.0
    gru_gradients = [gru_gradient]
    
    for z in z_values:
        gru_gradient *= (1 - z)
        gru_gradients.append(gru_gradient)
    
    # Compare with vanilla RNN
    rnn_gradient = 1.0
    rnn_gradients = [rnn_gradient]
    rnn_factor = 0.7  # Typical tanh derivative * weight magnitude
    
    for _ in range(sequence_length):
        rnn_gradient *= rnn_factor
        rnn_gradients.append(rnn_gradient)
    
    print(f"Gradient Analysis over {sequence_length} steps:")
    print(f"  Mean update gate: {np.mean(z_values):.3f}")
    print()
    print(f"  GRU direct path gradient: {gru_gradients[-1]:.2e}")
    print(f"  RNN gradient:             {rnn_gradients[-1]:.2e}")
    print(f"  GRU/RNN ratio:            {gru_gradients[-1]/rnn_gradients[-1]:.1e}×")
    
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.semilogy(rnn_gradients, label='Vanilla RNN', linewidth=2)
    plt.semilogy(gru_gradients, label='GRU (z≈0.2)', linewidth=2)
    plt.xlabel('Time Steps Back')
    plt.ylabel('Gradient Magnitude (log scale)')
    plt.title('Gradient Flow: GRU vs Vanilla RNN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

analyze_gru_gradients()
```

---

## Comparing GRU and LSTM Gates

### Structural Mapping

| GRU | LSTM | Notes |
|-----|------|-------|
| Update gate $z$ | Forget $f$ + Input $i$ | GRU couples them: $i = z$, $f = 1-z$ |
| Reset gate $r$ | (partial) Output gate | Both filter state before use |
| — | Output gate $o$ | GRU has no explicit output gating |
| — | Cell state $C$ | GRU uses only hidden state |

### Expressiveness Trade-off

**LSTM's uncoupled gates allow:**
- Forget without adding ($f$ high, $i$ low)
- Add without forgetting ($f$ high, $i$ high)  
- Reset with new info ($f$ low, $i$ high)

**GRU's coupled gates enforce:**
- Forgetting implies adding: $(1-z) + z = 1$
- Cannot independently control retention and addition

```python
def compare_gate_flexibility():
    """Demonstrate the flexibility difference between GRU and LSTM gates."""
    
    print("Gate Flexibility Comparison:")
    print()
    print("LSTM can do:")
    print("  - Retain much, add little:  f=0.9, i=0.1 → retains 90%, adds 10%")
    print("  - Retain much, add much:    f=0.9, i=0.9 → retains 90%, adds 90%")
    print("  - Retain little, add much:  f=0.1, i=0.9 → retains 10%, adds 90%")
    print("  Total weight on states: f + i (can be <1, =1, or >1)")
    print()
    print("GRU must do:")
    print("  - z=0.1 → retains 90%, adds 10%")
    print("  - z=0.5 → retains 50%, adds 50%")
    print("  - z=0.9 → retains 10%, adds 90%")
    print("  Total weight on states: (1-z) + z = 1 always")
    print()
    
    # Numerical example
    old_state = 1.0
    new_candidate = 2.0
    
    print("Numerical example: old=1.0, new=2.0")
    print()
    
    print("LSTM scenarios:")
    for f, i in [(0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]:
        result = f * old_state + i * new_candidate
        print(f"  f={f}, i={i}: result = {result:.2f}")
    
    print()
    print("GRU scenarios (constrained to convex combination):")
    for z in [0.1, 0.5, 0.9]:
        result = (1-z) * old_state + z * new_candidate
        print(f"  z={z}: result = {result:.2f}")

compare_gate_flexibility()
```

### Parameter Comparison

```python
def compare_parameters():
    """Compare parameter counts between LSTM and GRU."""
    
    input_size = 32
    hidden_size = 64
    
    lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    gru = nn.GRU(input_size, hidden_size, batch_first=True)
    
    lstm_params = sum(p.numel() for p in lstm.parameters())
    gru_params = sum(p.numel() for p in gru.parameters())
    
    print("Parameter Comparison:")
    print(f"  Input size: {input_size}, Hidden size: {hidden_size}")
    print(f"  LSTM parameters: {lstm_params:,}")
    print(f"  GRU parameters:  {gru_params:,}")
    print(f"  GRU reduction:   {100*(1 - gru_params/lstm_params):.1f}%")
    print()
    
    # Breakdown
    print("LSTM: 4 weight matrices × (input→hidden + hidden→hidden)")
    print(f"  = 4 × ({input_size}×{hidden_size} + {hidden_size}×{hidden_size}) + biases")
    print()
    print("GRU: 3 weight matrices × (input→hidden + hidden→hidden)")
    print(f"  = 3 × ({input_size}×{hidden_size} + {hidden_size}×{hidden_size}) + biases")

compare_parameters()
```

---

## Practical Gate Analysis

### Inspecting Learned Gates

```python
class GRUWithGateInspection(nn.Module):
    """GRU that exposes gate activations for analysis."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(input_size, hidden_size)
        
        # Storage for gate activations
        self.last_gates = {}
    
    def forward_with_gates(self, x, h=None):
        """Forward pass that also returns gate activations."""
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size)
        
        # Access internal weights
        weight_ih = self.gru.weight_ih
        weight_hh = self.gru.weight_hh
        bias_ih = self.gru.bias_ih
        bias_hh = self.gru.bias_hh
        
        n = self.hidden_size
        
        # Compute gates (PyTorch order: reset, update, new)
        gi = x @ weight_ih.t() + bias_ih
        gh = h @ weight_hh.t() + bias_hh
        
        r = torch.sigmoid(gi[:, :n] + gh[:, :n])
        z = torch.sigmoid(gi[:, n:2*n] + gh[:, n:2*n])
        
        # Store for inspection
        self.last_gates = {'r': r.detach(), 'z': z.detach()}
        
        # Complete the computation
        h_new = self.gru(x, h)
        
        return h_new, self.last_gates
    
    def forward_sequence(self, x_seq):
        """Process sequence and return all gate activations."""
        batch_size, seq_len, _ = x_seq.shape
        h = torch.zeros(batch_size, self.hidden_size)
        
        all_gates = {'r': [], 'z': []}
        outputs = []
        
        for t in range(seq_len):
            h, gates = self.forward_with_gates(x_seq[:, t, :], h)
            outputs.append(h)
            all_gates['r'].append(gates['r'])
            all_gates['z'].append(gates['z'])
        
        return (torch.stack(outputs, dim=1),
                {k: torch.stack(v, dim=1) for k, v in all_gates.items()})
```

### Diagnostic Analysis

```python
def diagnose_gru_training(gates, epoch=None):
    """
    Diagnose potential training issues from gate statistics.
    
    Args:
        gates: Dictionary with 'update' and 'reset' tensors
        epoch: Optional epoch number for tracking
    """
    prefix = f"Epoch {epoch}: " if epoch is not None else ""
    
    z = gates['update']
    r = gates['reset']
    
    # Check for saturation
    z_saturated_low = (z < 0.05).float().mean()
    z_saturated_high = (z > 0.95).float().mean()
    r_saturated_low = (r < 0.05).float().mean()
    r_saturated_high = (r > 0.95).float().mean()
    
    print(f"{prefix}Gate Diagnostics:")
    print(f"  Update gate z: mean={z.mean():.3f}, std={z.std():.3f}")
    print(f"    Saturated low (<0.05): {z_saturated_low:.1%}")
    print(f"    Saturated high (>0.95): {z_saturated_high:.1%}")
    print(f"  Reset gate r: mean={r.mean():.3f}, std={r.std():.3f}")
    print(f"    Saturated low (<0.05): {r_saturated_low:.1%}")
    print(f"    Saturated high (>0.95): {r_saturated_high:.1%}")
    
    # Warnings
    if z_saturated_low > 0.5:
        print("  ⚠️  Warning: Update gate mostly closed - model may not be learning")
    if z_saturated_high > 0.5:
        print("  ⚠️  Warning: Update gate mostly open - not preserving long-term memory")
    if r_saturated_low > 0.5:
        print("  ⚠️  Warning: Reset gate mostly closed - ignoring history")
```

---

## When to Choose GRU vs LSTM

| Criterion | GRU | LSTM |
|-----------|-----|------|
| Dataset size | Smaller (less overfitting) | Larger (more capacity) |
| Training speed | Faster (fewer parameters) | Slower |
| Sequence length | Comparable | Comparable |
| Task complexity | Simpler patterns | Complex dependencies |
| Memory requirements | Lower | Higher |
| Interpretability | Simpler gate dynamics | More nuanced control |

**Choose GRU when:**
- Dataset is small to medium
- Training time is a constraint
- Task doesn't require fine-grained memory control

**Choose LSTM when:**
- Dataset is large
- Task requires independent forgetting/updating
- Cell state separation from output is beneficial

---

## Summary

GRU's two-gate architecture offers elegant simplicity:

| Gate | Symbol | Formula | Role |
|------|--------|---------|------|
| Reset | $r_t$ | $\sigma(W_r[h_{t-1}, x_t] + b_r)$ | What history to consider |
| Update | $z_t$ | $\sigma(W_z[h_{t-1}, x_t] + b_z)$ | How to blend old/new |

### Key Insights

1. **Update gate couples forgetting and adding**: Conservation law $(1-z) + z = 1$
2. **Reset gate enables fresh starts**: When $r \approx 0$, candidate ignores history
3. **Gates work hierarchically**: Reset affects candidate computation, update affects final blend
4. **Fewer parameters, similar expressiveness**: ~25% fewer parameters than LSTM

### Practical Takeaways

- GRU often achieves similar performance to LSTM with faster training
- The coupled gate constraint can be a feature (simpler optimization) or limitation (less flexibility)
- Gate visualization helps debug and interpret model behavior
- Better choice for smaller datasets due to reduced overfitting risk

---

## References

1. Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP*.

2. Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. *NIPS Workshop on Deep Learning*.

3. Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). An Empirical Exploration of Recurrent Network Architectures. *ICML*.

4. Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A Search Space Odyssey. *IEEE Transactions on Neural Networks and Learning Systems*, 28(10), 2222-2232.
