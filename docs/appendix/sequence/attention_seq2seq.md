# Attention SEQ2SEQ

Attention SEQ2SEQ was introduced in the 2014 paper "Neural Machine Translation by Jointly Learning to Align and Translate." Attention mechanism to focus on relevant parts of input.

This implementation provides a concise, educational reference for Attention SEQ2SEQ. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
'''
Attention Mechanism in Seq2Seq
Paper: "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
Key: Attention mechanism to focus on relevant parts of input
'''
import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch, hidden_size]
        # encoder_outputs: [batch, seq_len, hidden_size]
        
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state seq_len times
        hidden = hidden.permute(1, 0, 2)  # [batch, 1, hidden_size]
        hidden = hidden.repeat(1, seq_len, 1)  # [batch, seq_len, hidden_size]
        
        # Calculate attention energies
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # [batch, seq_len]
        
        return torch.nn.functional.softmax(attention, dim=1)

class AttentionDecoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super().__init__()
        self.attention = Attention(hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        
        # Calculate attention weights
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)  # [batch, 1, seq_len]
        
        # Calculate context vector
        context = torch.bmm(a, encoder_outputs)  # [batch, 1, hidden_size]
        
        # Concatenate embedded input and context
        rnn_input = torch.cat((embedded, context), dim=2)
        
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc(output.squeeze(1))
        
        return prediction, hidden, a.squeeze(1)

class AttentionSeq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = AttentionDecoder(output_size, hidden_size)
    
    def forward(self, src, trg):
        encoder_outputs, (hidden, _) = self.encoder(src)
        
        outputs = []
        for t in range(trg.shape[1]):
            output, hidden, _ = self.decoder(trg[:, t:t+1], hidden, encoder_outputs)
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)

if __name__ == "__main__":
    model = AttentionSeq2Seq(input_size=100, output_size=100)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")```

## Discussion

The implementation defines 3 classes (`Attention`, `AttentionDecoder`, `AttentionSeq2Seq`) that work together to form the complete sequence model architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `Attention` with the default initialization. Break down the count by layer, including both weights and biases.

??? success "Solution to Exercise 1"
    For each `nn.Linear(in_features, out_features)`, there are `in_features * out_features` weight parameters plus `out_features` bias parameters (unless `bias=False`). For `nn.Conv2d(in_c, out_c, k)`, there are `in_c * out_c * k * k` weight parameters plus `out_c` bias parameters. For `nn.Embedding(num, dim)`, there are `num * dim` parameters. Sum across all layers. You can verify with `sum(p.numel() for p in model.parameters())`.

---

**Exercise 2.**
Add a dropout layer after the attention weights (before multiplying with values). Use a dropout rate of 0.1 during training. Explain why attention dropout helps with regularization.

??? success "Solution to Exercise 2"
    Add `self.attn_dropout = nn.Dropout(0.1)` in `__init__` and apply it after the softmax: `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. Attention dropout randomly zeroes some attention weights during training, preventing the model from relying too heavily on specific token-to-token relationships. This encourages the model to distribute attention more broadly and learn more robust representations, similar to how standard dropout prevents co-adaptation of neurons.

---

**Exercise 3.**
Explain the computational complexity of self-attention as a function of sequence length $n$ and model dimension $d$. Why does this motivate architectures like Longformer or Linformer for long sequences?

??? success "Solution to Exercise 3"
    Standard self-attention computes an $n \times n$ attention matrix, giving $O(n^2 d)$ time complexity and $O(n^2)$ memory for the attention weights. For long sequences (e.g., $n = 4096$), this becomes prohibitive. Longformer uses a combination of local sliding-window attention ($O(n \cdot w \cdot d)$ where $w$ is window size) and sparse global attention for selected tokens. Linformer projects keys and values to a lower dimension $k \ll n$, reducing complexity to $O(n \cdot k \cdot d)$. Both trade some expressiveness for practical efficiency on long inputs.

---

**Exercise 4.**
Extend `Attention` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = Attention(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
