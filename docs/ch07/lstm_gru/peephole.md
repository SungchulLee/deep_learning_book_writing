# Peephole Connections

## Introduction

Standard LSTM gates compute their activations based on the previous hidden state $h_{t-1}$ and the current input $x_t$. However, this means gates cannot directly observe the cell state $c_{t-1}$—the very memory they are supposed to regulate. **Peephole connections**, introduced by Gers and Schmidhuber (2000), address this limitation by giving gates direct access to the cell state.

---

## Motivation

In the standard LSTM, gates make decisions about the cell state without actually seeing it:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

The hidden state $h_t = o_t \odot \tanh(c_t)$ is a filtered, transformed view of $c_t$. Information may be stored in the cell state with the output gate closed ($o_t \approx 0$), rendering it invisible to the gates at the next timestep via $h_{t-1}$.

**The problem:** If the output gate at step $t-1$ suppresses certain cell dimensions, the forget and input gates at step $t$ cannot see those stored values when deciding what to keep or add.

**Peephole connections solve this** by adding direct, element-wise connections from the cell state to each gate.

---

## Mathematical Formulation

### Standard LSTM (Without Peepholes)

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(c_t)$$

### Peephole LSTM

The forget and input gates receive peephole connections from the **previous** cell state $c_{t-1}$, while the output gate receives a peephole from the **current** cell state $c_t$ (since it's computed after the cell update):

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + \mathbf{w_f \odot c_{t-1}} + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + \mathbf{w_i \odot c_{t-1}} + b_i)$$
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + \mathbf{w_o \odot c_t} + b_o)$$
$$h_t = o_t \odot \tanh(c_t)$$

Where $w_f, w_i, w_o \in \mathbb{R}^n$ are **diagonal** peephole weight vectors (not full matrices).

!!! note "Why Diagonal Weights?"
    Peephole connections use element-wise multiplication ($w \odot c$) rather than full matrix multiplication ($W \cdot c$). This means each gate dimension only sees the corresponding cell dimension. This diagonal constraint keeps the additional parameter count small ($3n$ new parameters) while allowing gates to monitor their "own" cell state values.

---

## Parameter Count Impact

| Component | Standard LSTM | Peephole LSTM |
|-----------|---------------|---------------|
| Weight matrices | $4n(n+d)$ | $4n(n+d)$ |
| Biases | $4n$ | $4n$ |
| Peephole vectors | — | $3n$ |
| **Total** | $4n(n+d+1)$ | $4n(n+d+1) + 3n$ |

For typical hidden sizes ($n = 256$), peepholes add only $3 \times 256 = 768$ parameters—a negligible increase compared to the $4 \times 256 \times (256 + d + 1)$ total.

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn

class PeepholeLSTMCell(nn.Module):
    """LSTM cell with peephole connections."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Standard weights
        combined_size = input_size + hidden_size
        self.W_f = nn.Linear(combined_size, hidden_size)
        self.W_i = nn.Linear(combined_size, hidden_size)
        self.W_c = nn.Linear(combined_size, hidden_size)
        self.W_o = nn.Linear(combined_size, hidden_size)
        
        # Peephole weights (diagonal, so just vectors)
        self.p_f = nn.Parameter(torch.randn(hidden_size) * 0.1)
        self.p_i = nn.Parameter(torch.randn(hidden_size) * 0.1)
        self.p_o = nn.Parameter(torch.randn(hidden_size) * 0.1)
        
        # Initialize forget bias to 1
        nn.init.constant_(self.W_f.bias, 1.0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        states: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input (batch_size, input_size)
            states: Tuple of (h_prev, c_prev)
        
        Returns:
            h: New hidden state
            (h, c): New states tuple
        """
        batch_size = x.size(0)
        
        if states is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_prev, c_prev = states
        
        combined = torch.cat([x, h_prev], dim=-1)
        
        # Gates with peephole connections to cell state
        # Forget and input gates peek at PREVIOUS cell state
        f = torch.sigmoid(self.W_f(combined) + self.p_f * c_prev)
        i = torch.sigmoid(self.W_i(combined) + self.p_i * c_prev)
        c_tilde = torch.tanh(self.W_c(combined))
        
        # Cell state update (same as standard LSTM)
        c = f * c_prev + i * c_tilde
        
        # Output gate peephole uses NEW cell state
        o = torch.sigmoid(self.W_o(combined) + self.p_o * c)
        h = o * torch.tanh(c)
        
        return h, (h, c)


class PeepholeLSTM(nn.Module):
    """Full peephole LSTM layer processing sequences."""
    
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            PeepholeLSTMCell(
                input_size if layer == 0 else hidden_size, 
                hidden_size
            )
            for layer in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x, states=None):
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_size)
            states: Initial (h_0, c_0) for all layers
        
        Returns:
            output: Top-layer hidden states (batch_size, seq_len, hidden_size)
            (h_n, c_n): Final states for all layers
        """
        batch_size, seq_len, _ = x.size()
        
        if states is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                 for _ in range(self.num_layers)]
        else:
            h = [states[0][i] for i in range(self.num_layers)]
            c = [states[1][i] for i in range(self.num_layers)]
        
        outputs = []
        for t in range(seq_len):
            layer_input = x[:, t, :]
            
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], (_, c[layer_idx]) = cell(
                    layer_input, (h[layer_idx], c[layer_idx])
                )
                layer_input = h[layer_idx]
                
                if self.dropout and layer_idx < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
            
            outputs.append(h[-1])
        
        output = torch.stack(outputs, dim=1)
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)
        
        return output, (h_n, c_n)
```

---

## When to Use Peephole Connections

### Tasks That Benefit

Peephole connections are most useful for tasks requiring **precise timing** or **magnitude-aware gating**:

| Task Type | Why Peepholes Help |
|-----------|-------------------|
| **Rhythm/tempo learning** | Gates can track accumulated timing in cell state |
| **Precise interval prediction** | Cell state magnitude encodes elapsed time |
| **Threshold-based decisions** | Gates can trigger when cell values cross thresholds |
| **Counting tasks** | Cell state acts as counter; gates can read the count |

### When Standard LSTM Suffices

For most practical tasks, standard LSTM without peepholes performs comparably:

- **Language modeling**: Token-level patterns well-captured by $h_{t-1}$
- **Sentiment analysis**: Global sentiment doesn't require precise timing
- **General time series**: Most patterns don't require cell-state awareness

### Empirical Evidence

Greff et al. (2017) found in their extensive LSTM variant study that:

- Peephole connections provide marginal improvement on most benchmarks
- The forget gate and output activation are far more critical components
- Peepholes can help on specific tasks requiring precise timing

---

## Gradient Flow Impact

Peephole connections add additional gradient paths through the cell state:

$$\frac{\partial f_t}{\partial c_{t-1}} = \sigma'(\cdot) \cdot \text{diag}(w_f)$$

This creates a direct feedback loop: the cell state influences the forget gate, which influences the next cell state. While this can improve learning for timing-sensitive tasks, it can also make optimization more complex.

---

## Summary

Peephole connections extend standard LSTM by giving gates direct visibility into the cell state:

| Gate | Standard LSTM Input | Peephole Addition |
|------|--------------------|--------------------|
| Forget $f_t$ | $[h_{t-1}, x_t]$ | $+ w_f \odot c_{t-1}$ |
| Input $i_t$ | $[h_{t-1}, x_t]$ | $+ w_i \odot c_{t-1}$ |
| Output $o_t$ | $[h_{t-1}, x_t]$ | $+ w_o \odot c_t$ |

**Key takeaways:**

- Peepholes add negligible parameters ($3n$) compared to total LSTM parameters
- Most useful for tasks requiring precise timing or magnitude-aware gating
- For general-purpose sequence modeling, standard LSTM is typically sufficient
- Not widely used in modern practice, but worth trying for timing-critical applications

---

## References

1. Gers, F. A., & Schmidhuber, J. (2000). Recurrent Nets that Time and Count. *IJCNN*.

2. Gers, F. A., Schraudolph, N. N., & Schmidhuber, J. (2003). Learning Precise Timing with LSTM Recurrent Networks. *Journal of Machine Learning Research*, 3, 115-143.

3. Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A Search Space Odyssey. *IEEE Transactions on Neural Networks and Learning Systems*, 28(10), 2222-2232.
