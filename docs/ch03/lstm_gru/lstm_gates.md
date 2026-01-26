# LSTM Gates: Forget, Input, and Output

## Introduction

The power of LSTM lies in its gating mechanisms—learned controllers that regulate information flow through the network. Each gate serves a specific purpose: deciding what to forget, what to remember, and what to output. Understanding these gates deeply is essential for debugging, interpreting, and improving LSTM models.

## Gate Overview

| Gate | Symbol | Purpose | Output Range |
|------|--------|---------|--------------|
| Forget | $f_t$ | What to discard from cell state | [0, 1] |
| Input | $i_t$ | What new information to store | [0, 1] |
| Output | $o_t$ | What to expose as hidden state | [0, 1] |

All gates use sigmoid activation $\sigma(x) = \frac{1}{1+e^{-x}}$, producing values between 0 (fully closed) and 1 (fully open).

## The Forget Gate

### Purpose
Decides what information to discard from the cell state. This allows the LSTM to "forget" irrelevant history.

### Computation

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

Or equivalently:

$$f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)$$

### Application

$$c_t = f_t \odot c_{t-1} + \ldots$$

When $f_t \approx 0$: Cell state is erased (forget everything)
When $f_t \approx 1$: Cell state is preserved (remember everything)

### Intuition

Consider language modeling: when encountering a period (end of sentence), the forget gate learns to reset context that's no longer relevant for the next sentence.

```python
import torch
import torch.nn as nn

class ForgetGateAnalysis:
    """Analyze forget gate behavior."""
    
    @staticmethod
    def compute_forget_gate(W_xf, W_hf, b_f, x_t, h_prev):
        """Manually compute forget gate activation."""
        pre_activation = W_xf @ x_t + W_hf @ h_prev + b_f
        f_t = torch.sigmoid(pre_activation)
        return f_t
    
    @staticmethod
    def visualize_forget_patterns(forget_gates, tokens):
        """
        Visualize when LSTM forgets.
        
        Args:
            forget_gates: (seq_len, hidden_size) tensor
            tokens: List of input tokens
        """
        import matplotlib.pyplot as plt
        
        # Average across hidden dimensions
        avg_forget = forget_gates.mean(dim=-1).numpy()
        
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(tokens)), avg_forget)
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.ylabel('Average Forget Gate Value')
        plt.title('Forget Gate Activations (lower = more forgetting)')
        plt.tight_layout()
        plt.show()
```

## The Input Gate

### Purpose
Controls what new information to add to the cell state. Works in conjunction with the candidate cell state.

### Computation

**Input gate:**
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidate cell state:**
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

### Application

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

- $i_t$ controls *how much* of the new information to add
- $\tilde{c}_t$ represents *what* new information could be added
- Their element-wise product determines actual update

### Intuition

When reading "The capital of France is Paris":
- At "Paris", the input gate opens to store this important fact
- At "is", the input gate may stay relatively closed (less informative)

```python
class InputGateAnalysis:
    """Analyze input gate and candidate interactions."""
    
    @staticmethod
    def decompose_cell_update(f_t, c_prev, i_t, c_tilde):
        """
        Decompose cell state update into forget and input contributions.
        """
        forget_contribution = f_t * c_prev
        input_contribution = i_t * c_tilde
        c_new = forget_contribution + input_contribution
        
        # Relative contributions
        total_magnitude = forget_contribution.abs() + input_contribution.abs() + 1e-8
        forget_ratio = forget_contribution.abs() / total_magnitude
        input_ratio = input_contribution.abs() / total_magnitude
        
        return {
            'cell_new': c_new,
            'forget_contribution': forget_contribution,
            'input_contribution': input_contribution,
            'forget_ratio': forget_ratio.mean().item(),
            'input_ratio': input_ratio.mean().item()
        }
```

## The Output Gate

### Purpose
Controls what parts of the cell state to expose as the hidden state output. The cell state may contain information not immediately relevant.

### Computation

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

### Application

$$h_t = o_t \odot \tanh(c_t)$$

The $\tanh$ squashes cell state to [-1, 1], and $o_t$ selectively exposes dimensions.

### Intuition

The cell might store both subject ("cat") and verb tense information. When predicting the next word after "The cat", the output gate might emphasize subject information while suppressing tense until it's needed.

```python
class OutputGateAnalysis:
    """Analyze output gate behavior."""
    
    @staticmethod
    def compute_output_filtering(o_t, c_t):
        """Show how output gate filters cell state."""
        cell_activated = torch.tanh(c_t)
        h_t = o_t * cell_activated
        
        # How much information is passed vs blocked
        pass_ratio = (o_t > 0.5).float().mean()
        
        return {
            'hidden_state': h_t,
            'cell_activated': cell_activated,
            'pass_ratio': pass_ratio.item()
        }
```

## Gate Interactions

The three gates work together in a coordinated dance:

```python
def lstm_step_detailed(x_t, h_prev, c_prev, W_xf, W_hf, b_f,
                       W_xi, W_hi, b_i, W_xc, W_hc, b_c,
                       W_xo, W_ho, b_o):
    """
    Detailed LSTM step showing all gate computations.
    """
    # Forget gate: What to discard?
    f_t = torch.sigmoid(W_xf @ x_t + W_hf @ h_prev + b_f)
    
    # Input gate: How much new info to add?
    i_t = torch.sigmoid(W_xi @ x_t + W_hi @ h_prev + b_i)
    
    # Candidate: What new info is available?
    c_tilde = torch.tanh(W_xc @ x_t + W_hc @ h_prev + b_c)
    
    # Cell update: Forget old + add new
    c_t = f_t * c_prev + i_t * c_tilde
    
    # Output gate: What to expose?
    o_t = torch.sigmoid(W_xo @ x_t + W_ho @ h_prev + b_o)
    
    # Hidden state: Filtered cell state
    h_t = o_t * torch.tanh(c_t)
    
    return h_t, c_t, {'f': f_t, 'i': i_t, 'o': o_t, 'c_tilde': c_tilde}
```

## Visualizing Gate Activations

```python
def extract_gate_activations(lstm, x, return_all=True):
    """
    Extract gate activations from an LSTM for analysis.
    
    Args:
        lstm: nn.LSTM module
        x: Input tensor (batch, seq_len, input_size)
    
    Returns:
        Dictionary of gate activations
    """
    batch_size, seq_len, _ = x.shape
    hidden_size = lstm.hidden_size
    
    # Storage for gates
    forget_gates = []
    input_gates = []
    output_gates = []
    cell_states = []
    
    # Get weights
    W_ii, W_if, W_ig, W_io = lstm.weight_ih_l0.chunk(4, 0)
    W_hi, W_hf, W_hg, W_ho = lstm.weight_hh_l0.chunk(4, 0)
    b_ii, b_if, b_ig, b_io = lstm.bias_ih_l0.chunk(4, 0)
    b_hi, b_hf, b_hg, b_ho = lstm.bias_hh_l0.chunk(4, 0)
    
    # Initialize
    h = torch.zeros(batch_size, hidden_size)
    c = torch.zeros(batch_size, hidden_size)
    
    for t in range(seq_len):
        x_t = x[:, t, :]
        
        # Gates
        i = torch.sigmoid(x_t @ W_ii.T + h @ W_hi.T + b_ii + b_hi)
        f = torch.sigmoid(x_t @ W_if.T + h @ W_hf.T + b_if + b_hf)
        g = torch.tanh(x_t @ W_ig.T + h @ W_hg.T + b_ig + b_hg)
        o = torch.sigmoid(x_t @ W_io.T + h @ W_ho.T + b_io + b_ho)
        
        # Update
        c = f * c + i * g
        h = o * torch.tanh(c)
        
        forget_gates.append(f)
        input_gates.append(i)
        output_gates.append(o)
        cell_states.append(c)
    
    return {
        'forget': torch.stack(forget_gates, dim=1),
        'input': torch.stack(input_gates, dim=1),
        'output': torch.stack(output_gates, dim=1),
        'cell': torch.stack(cell_states, dim=1)
    }


def plot_gate_heatmaps(gates, tokens=None):
    """Plot heatmaps of gate activations."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    gate_names = ['Forget Gate', 'Input Gate', 'Output Gate']
    gate_keys = ['forget', 'input', 'output']
    
    for ax, name, key in zip(axes, gate_names, gate_keys):
        data = gates[key][0].detach().numpy()  # First sample
        im = ax.imshow(data.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Hidden Dimension')
        
        if tokens is not None:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
        
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()
```

## Gate Initialization: Forget Gate Bias

A crucial initialization trick: set forget gate bias to 1 or higher:

```python
def init_lstm_forget_bias(lstm, value=1.0):
    """
    Initialize forget gate bias to encourage remembering.
    
    Without this, randomly initialized forget gates may start near 0.5,
    causing information loss before the network learns to remember.
    """
    for name, param in lstm.named_parameters():
        if 'bias' in name:
            n = param.size(0)
            # Bias order: input, forget, cell, output
            start = n // 4
            end = n // 2
            param.data[start:end].fill_(value)
            print(f"Set forget gate bias to {value}")
```

## Interpreting Gate Patterns

| Pattern | Forget Gate | Input Gate | Interpretation |
|---------|-------------|------------|----------------|
| Copying | ~1.0 | ~0.0 | Preserve information |
| Resetting | ~0.0 | ~1.0 | Replace with new info |
| Updating | ~0.5 | ~0.5 | Blend old and new |
| Selective | Variable | Variable | Content-dependent |

## Summary

LSTM gates provide fine-grained control over information flow:

1. **Forget gate** ($f_t$): Erases irrelevant history
2. **Input gate** ($i_t$): Adds important new information  
3. **Output gate** ($o_t$): Exposes relevant cell content

Key insights:
- Gates learn task-specific information routing
- Forget gate bias initialization is crucial
- Visualizing gates aids debugging and interpretation
- Gate patterns reveal what the network has learned
