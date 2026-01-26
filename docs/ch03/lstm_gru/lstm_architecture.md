# Long Short-Term Memory (LSTM)

## Introduction

Long Short-Term Memory networks, introduced by Hochreiter and Schmidhuber (1997), fundamentally solve the vanishing gradient problem through a gated architecture. The key innovation is a separate **cell state** that allows information to flow unchanged through time, combined with learnable gates that regulate information flow.

## Architecture Overview

LSTM maintains two state vectors at each timestep:
- **Hidden state** $h_t$: Short-term memory, output of the cell
- **Cell state** $c_t$: Long-term memory, information highway

Three gates control information flow:
- **Forget gate** $f_t$: What to discard from cell state
- **Input gate** $i_t$: What new information to store
- **Output gate** $o_t$: What to output from cell state

## Mathematical Formulation

At each timestep $t$, given input $x_t$ and previous states $h_{t-1}$, $c_{t-1}$:

### Forget Gate
Decides what information to discard from the cell state:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

When $f_t \approx 0$, the gate forgets; when $f_t \approx 1$, it remembers.

### Input Gate
Decides what new information to store:

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

### Candidate Cell State
Proposes new values to add:

$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

### Cell State Update
Combines forgetting old information and adding new:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

The $\odot$ denotes element-wise (Hadamard) product.

### Output Gate
Determines what to output:

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

### Hidden State
Filtered version of cell state:

$$h_t = o_t \odot \tanh(c_t)$$

## Why LSTM Solves Vanishing Gradients

The cell state update equation reveals the key insight:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

The gradient flowing back through cell states:

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

When $f_t \approx 1$ (the forget gate is open), gradients flow through **unchanged**:

$$\frac{\partial \mathcal{L}}{\partial c_{t-k}} = \frac{\partial \mathcal{L}}{\partial c_t} \cdot \prod_{i=t-k}^{t-1} f_{i+1}$$

Unlike vanilla RNNs where gradients pass through weight matrices and saturating activations, LSTM gradients pass through learned gates that can remain near 1.

## PyTorch Implementation from Scratch

```python
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    """Single LSTM cell."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Combined weight matrices for efficiency
        # [W_i, W_f, W_c, W_o] stacked
        self.W_ih = nn.Linear(input_size, 4 * hidden_size)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
    
    def forward(self, x, states):
        """
        Args:
            x: Input (batch, input_size)
            states: Tuple of (h, c), each (batch, hidden_size)
        
        Returns:
            h_new, c_new: New hidden and cell states
        """
        h, c = states
        
        # Compute all gates at once
        gates = self.W_ih(x) + self.W_hh(h)
        
        # Split into individual gates
        i, f, g, o = gates.chunk(4, dim=-1)
        
        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Candidate cell state
        o = torch.sigmoid(o)  # Output gate
        
        # Update cell state
        c_new = f * c + i * g
        
        # Compute new hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class LSTM(nn.Module):
    """Full LSTM processing sequences."""
    
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x, initial_states=None):
        """
        Args:
            x: Input sequence (batch, seq_len, input_size)
            initial_states: Tuple of (h_0, c_0), each (num_layers, batch, hidden)
        
        Returns:
            outputs: All hidden states (batch, seq_len, hidden_size)
            (h_n, c_n): Final states for each layer
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize states
        if initial_states is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
        else:
            h = [initial_states[0][i] for i in range(self.num_layers)]
            c = [initial_states[1][i] for i in range(self.num_layers)]
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            layer_input = x[:, t, :]
            
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell(layer_input, (h[layer_idx], c[layer_idx]))
                layer_input = h[layer_idx]
                
                # Apply dropout between layers (not after last)
                if self.dropout and layer_idx < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
            
            outputs.append(h[-1])  # Output from top layer
        
        outputs = torch.stack(outputs, dim=1)
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)
        
        return outputs, (h_n, c_n)
```

## Using PyTorch's Built-in LSTM

```python
import torch.nn as nn

# Create LSTM layer
lstm = nn.LSTM(
    input_size=100,      # Feature dimension of input
    hidden_size=256,     # Hidden/cell state dimension
    num_layers=2,        # Stacked LSTM layers
    batch_first=True,    # Input shape: (batch, seq, features)
    dropout=0.3,         # Dropout between layers
    bidirectional=False  # Process forward only
)

# Input: (batch=32, seq_len=50, input_size=100)
x = torch.randn(32, 50, 100)

# Forward pass
outputs, (h_n, c_n) = lstm(x)

print(f"Outputs: {outputs.shape}")  # (32, 50, 256)
print(f"h_n: {h_n.shape}")          # (2, 32, 256) - 2 layers
print(f"c_n: {c_n.shape}")          # (2, 32, 256)
```

## LSTM for Classification

```python
class LSTMClassifier(nn.Module):
    """LSTM for sequence classification."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, 
                 num_layers=1, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len) - token indices
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Use final hidden state for classification
        # h_n[-1] is the last layer's hidden state
        final_hidden = self.dropout(h_n[-1])  # (batch, hidden_size)
        
        logits = self.fc(final_hidden)  # (batch, num_classes)
        return logits

# Usage
model = LSTMClassifier(
    vocab_size=10000,
    embed_dim=100,
    hidden_size=256,
    num_classes=2,
    num_layers=2,
    dropout=0.5
)

# Input: batch of 32 sequences, each length 100
x = torch.randint(0, 10000, (32, 100))
logits = model(x)  # (32, 2)
```

## LSTM for Sequence Generation

```python
class LSTMGenerator(nn.Module):
    """LSTM for text generation."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        logits = self.fc(lstm_out)
        return logits, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)
    
    def generate(self, start_tokens, max_length, temperature=1.0, device='cpu'):
        """Autoregressive generation."""
        self.eval()
        tokens = start_tokens.tolist()
        hidden = self.init_hidden(1, device)
        
        # Process start tokens
        x = torch.tensor([tokens], device=device)
        _, hidden = self.forward(x, hidden)
        
        # Generate new tokens
        current_token = torch.tensor([[tokens[-1]]], device=device)
        
        with torch.no_grad():
            for _ in range(max_length - len(tokens)):
                logits, hidden = self.forward(current_token, hidden)
                
                # Apply temperature
                probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
                
                # Sample
                next_token = torch.multinomial(probs, 1).item()
                tokens.append(next_token)
                current_token = torch.tensor([[next_token]], device=device)
        
        return tokens
```

## LSTM for Time Series

```python
class LSTMForecaster(nn.Module):
    """LSTM for time series forecasting."""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 horizon=1, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size * horizon)
        self.horizon = horizon
        self.output_size = output_size
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        final_hidden = h_n[-1]  # (batch, hidden_size)
        
        # Predict horizon steps
        predictions = self.fc(final_hidden)  # (batch, output_size * horizon)
        predictions = predictions.view(-1, self.horizon, self.output_size)
        
        return predictions

# Usage: Predict 5 steps of 3 features from 20 historical steps
model = LSTMForecaster(
    input_size=3,
    hidden_size=128,
    num_layers=2,
    output_size=3,
    horizon=5
)

x = torch.randn(32, 20, 3)  # 32 samples, 20 timesteps, 3 features
predictions = model(x)  # (32, 5, 3)
```

## Initialization Best Practices

```python
def init_lstm_weights(lstm):
    """Initialize LSTM weights for better training."""
    for name, param in lstm.named_parameters():
        if 'weight_ih' in name:
            # Input-hidden weights: Xavier
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            # Hidden-hidden weights: Orthogonal
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            # Biases: Set forget gate bias to 1
            n = param.size(0)
            start, end = n // 4, n // 2
            param.data.fill_(0.)
            param.data[start:end].fill_(1.)  # Forget gate bias = 1
```

Setting the forget gate bias to 1 helps the LSTM remember information by default.

## Summary

LSTM networks solve the vanishing gradient problem through:

1. **Cell state**: Information highway that allows gradients to flow unchanged
2. **Forget gate**: Learns what information to discard
3. **Input gate**: Learns what new information to store
4. **Output gate**: Learns what information to output

Key properties:
- Gradients can flow through cell state without vanishing
- Gates learn task-specific information flow
- Captures long-range dependencies effectively

The architecture adds computational cost over simple RNNs but enables learning over much longer sequences—essential for most practical applications.
