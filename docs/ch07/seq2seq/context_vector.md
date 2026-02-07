# Context Vector

The context vector is the bridge between encoder and decoder in a sequence-to-sequence model—a fixed-dimensional representation that must capture all relevant information from the source sequence. Understanding its properties, limitations, and design choices is essential for building effective seq2seq systems and appreciating why attention mechanisms became necessary.

## The Role of the Context Vector

In the basic encoder-decoder architecture, the context vector $\mathbf{c}$ is the sole channel through which source information flows to the decoder. The encoder compresses the entire input sequence $\mathbf{x} = (x_1, \ldots, x_T)$ into this single vector, and the decoder must reconstruct the appropriate output using only this compressed representation:

$$\mathbf{c} = \mathbf{h}_T^{enc} \quad \text{(final hidden state)}$$

$$P(\mathbf{y} | \mathbf{x}) = \prod_{t=1}^{T'} P(y_t | y_{<t}, \mathbf{c})$$

The context vector serves simultaneously as the decoder's initial state and its persistent source of input information. Every output token, regardless of which part of the input it corresponds to, must draw its source information from this same fixed representation.

## Context Vector Construction

### Last Hidden State

The simplest approach uses the encoder's final hidden state directly:

$$\mathbf{c} = \mathbf{h}_T^{enc}$$

For LSTM encoders, both the hidden state $\mathbf{h}_T$ and cell state $\mathbf{c}_T$ are passed to the decoder, providing two complementary views of the encoded information. The hidden state captures the "working memory" while the cell state retains longer-term information through the gating mechanism.

### Bidirectional Concatenation

With bidirectional encoders, the context combines forward and backward final states:

$$\mathbf{c} = [\overrightarrow{\mathbf{h}}_T ; \overleftarrow{\mathbf{h}}_1] \in \mathbb{R}^{2d_h}$$

This doubles the context dimensionality and ensures that information from both the beginning and end of the source sequence is directly represented. However, the bidirectional context requires projection to match the decoder's expected input dimension:

```python
class ContextProjection(nn.Module):
    """
    Project bidirectional encoder states to decoder-compatible dimensions.
    
    The tanh activation bounds the projected values, preventing
    extreme initializations from destabilizing the decoder.
    """
    
    def __init__(self, encoder_hidden: int, decoder_hidden: int):
        super().__init__()
        self.fc_h = nn.Linear(encoder_hidden * 2, decoder_hidden)
        self.fc_c = nn.Linear(encoder_hidden * 2, decoder_hidden)
    
    def forward(self, h_forward, h_backward, c_forward=None, c_backward=None):
        h_combined = torch.cat([h_forward, h_backward], dim=-1)
        h_projected = torch.tanh(self.fc_h(h_combined))
        
        if c_forward is not None and c_backward is not None:
            c_combined = torch.cat([c_forward, c_backward], dim=-1)
            c_projected = torch.tanh(self.fc_c(c_combined))
            return h_projected, c_projected
        
        return h_projected
```

### Pooled Representations

Rather than relying solely on the final state, alternative construction methods aggregate information across all encoder positions:

```python
def construct_context(
    encoder_outputs: torch.Tensor,
    hidden: torch.Tensor,
    method: str = 'last'
) -> torch.Tensor:
    """
    Construct context vector from encoder outputs.
    
    Args:
        encoder_outputs: All encoder hidden states (batch, seq_len, hidden)
        hidden: Final encoder hidden state (layers, batch, hidden)
        method: Construction strategy
        
    Returns:
        context: Context vector (batch, hidden)
    """
    if method == 'last':
        # Use final hidden state (standard approach)
        return hidden[-1]  # (batch, hidden)
    
    elif method == 'mean':
        # Average all encoder hidden states
        # Captures "overall" meaning but loses positional information
        return encoder_outputs.mean(dim=1)
    
    elif method == 'max':
        # Max-pool across time dimension
        # Each dimension captures the strongest activation
        return encoder_outputs.max(dim=1).values
    
    elif method == 'concat_last':
        # Concatenate first and last hidden states
        # Preserves information about sequence boundaries
        return torch.cat([
            encoder_outputs[:, 0, :],
            encoder_outputs[:, -1, :]
        ], dim=-1)
    
    else:
        raise ValueError(f"Unknown context method: {method}")
```

**Mean pooling** gives equal weight to all positions, capturing the overall "average meaning" but losing the emphasis on any particular part of the input. **Max pooling** selects the strongest activation for each hidden dimension, which can capture salient features regardless of their position. Neither approach fully resolves the information bottleneck, as dimensionality remains fixed regardless of input length.

## The Information Bottleneck Problem

### Fundamental Limitation

The basic encoder-decoder architecture forces the entire input sequence through a single fixed-dimensional vector. This creates a fundamental capacity constraint: as input length grows, the context vector must represent more information in the same number of dimensions.

**Capacity limitation**: For long sequences, the fixed-size context vector is insufficient to capture all relevant information. Information from early parts of the sequence tends to be overwritten by later processing, a consequence of the recurrent computation's sequential nature.

**Gradient flow**: During backpropagation, gradients must flow through the entire sequence, making it difficult to learn long-range dependencies effectively. Even with LSTM gating, gradient magnitude diminishes over very long distances.

**Uniform representation**: All input tokens contribute to a single representation, even though different output tokens may need to focus on different parts of the input. A decoder generating the first word of a translation needs different source information than when generating the last word.

### Information-Theoretic Analysis

The severity of the bottleneck can be analyzed using information theory. The mutual information between the input $\mathbf{X}$ and the context vector $\mathbf{c}$ is bounded:

$$I(\mathbf{X}; \mathbf{c}) \leq H(\mathbf{c}) \leq d_c \cdot \log(|\mathcal{V}|)$$

where $d_c$ is the context dimension and $|\mathcal{V}|$ represents the range of each dimension. This upper bound on information transfer is **independent of input length**, creating a fundamental bottleneck for long sequences.

Consider the practical implications: a context vector of dimension 512 with 32-bit floating point values has a theoretical capacity of $512 \times 32 = 16{,}384$ bits. While this is substantial, a 100-word sentence with a 50,000-word vocabulary carries approximately $100 \times \log_2(50{,}000) \approx 1{,}561$ bits of raw token information, plus syntactic and semantic structure. For very long documents, the information demand vastly exceeds the context capacity.

### Empirical Evidence

The bottleneck manifests clearly in translation quality as a function of sentence length. Cho et al. (2014) showed that BLEU scores for basic encoder-decoder models drop sharply for sentences exceeding 20-30 tokens:

```
Sentence Length vs. BLEU Score (schematic)

BLEU
 │
40├─ ●───●───●
 │              ╲
30├               ●───●
 │                      ╲
20├                        ●───●
 │                                ╲
10├                                  ●───●
 │
 └──┬───┬───┬───┬───┬───┬───┬───┬───┬──
    5   10  15  20  25  30  35  40  45
                Sentence Length
```

This degradation is a direct consequence of the information bottleneck: longer sentences contain more information that must be compressed into the same fixed-dimensional representation.

## Multi-Layer Context Representations

### Deep Encoder with Residual Connections

Stacking multiple RNN layers increases model capacity and allows learning hierarchical representations. Residual connections enable gradient flow through deep architectures:

```python
class DeepEncoder(nn.Module):
    """
    Multi-layer encoder with residual connections.
    
    Deeper architectures can capture more abstract representations,
    with residual connections enabling gradient flow.
    """
    
    def __init__(
        self,
        input_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.input_projection = nn.Linear(embedding_dim, hidden_size)
        
        # Stack of LSTM layers with residual connections
        self.layers = nn.ModuleList([
            nn.LSTM(
                hidden_size, hidden_size,
                batch_first=True, bidirectional=True
            )
            for _ in range(num_layers)
        ])
        
        # Project bidirectional output back to hidden_size
        self.layer_projections = nn.ModuleList([
            nn.Linear(hidden_size * 2, hidden_size)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> tuple:
        # Embed and project
        embedded = self.input_projection(self.embedding(x))
        
        outputs = embedded
        hidden_states = []
        cell_states = []
        
        for lstm, projection in zip(self.layers, self.layer_projections):
            # Store residual
            residual = outputs
            
            # Process through bidirectional LSTM
            lstm_out, (h, c) = lstm(outputs)
            
            # Project and apply residual connection
            outputs = projection(lstm_out)
            outputs = self.dropout(outputs) + residual
            outputs = self.layer_norm(outputs)
            
            hidden_states.append(h)
            cell_states.append(c)
        
        # Combine layer states
        final_hidden = torch.cat(hidden_states, dim=0)
        final_cell = torch.cat(cell_states, dim=0)
        
        return outputs, final_hidden, final_cell
```

Each layer in the deep encoder captures progressively more abstract features. Lower layers tend to learn local patterns (morphology, word-level features), while upper layers capture broader semantic relationships. The residual connections ensure that low-level features remain accessible to the decoder, preventing information loss through deep stacking.

### Hierarchical Representations

In multi-layer architectures, each layer's hidden state provides a different level of abstraction. The context vector can be constructed from multiple layers rather than just the top:

$$\mathbf{c} = f(\mathbf{h}_T^{(1)}, \mathbf{h}_T^{(2)}, \ldots, \mathbf{h}_T^{(L)})$$

where $L$ is the number of layers. This provides the decoder with a richer, multi-scale representation of the source. The combining function $f$ can be concatenation, learned weighted sum, or gating:

```python
class MultiLayerContext(nn.Module):
    """
    Construct context from multiple encoder layers.
    
    Different layers capture different abstraction levels,
    and the decoder may benefit from accessing all of them.
    """
    
    def __init__(self, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        # Learned attention over layers
        self.layer_attention = nn.Linear(hidden_size, 1)
        self.projection = nn.Linear(hidden_size, output_size)
    
    def forward(self, layer_hiddens: list) -> torch.Tensor:
        """
        Args:
            layer_hiddens: List of (batch, hidden) tensors, one per layer
            
        Returns:
            context: (batch, output_size)
        """
        # Stack: (batch, num_layers, hidden)
        stacked = torch.stack(layer_hiddens, dim=1)
        
        # Compute layer-level attention weights
        scores = self.layer_attention(stacked).squeeze(-1)  # (batch, num_layers)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, num_layers, 1)
        
        # Weighted combination
        combined = (stacked * weights).sum(dim=1)  # (batch, hidden)
        
        return self.projection(combined)
```

## Mitigating the Bottleneck

### Input Reversal

Sutskever et al. (2014) found that reversing the input sequence significantly improved translation quality. By placing the first source tokens closest to the decoder's initial state, short-term dependencies between the beginning of the source and beginning of the target are shortened:

```
Original:    A B C D → W X Y Z
Reversed:    D C B A → W X Y Z

With reversal, A (first source token) is adjacent to W (first target token)
in terms of recurrent processing distance.
```

This simple trick partially mitigates the bottleneck by reducing the effective distance between corresponding source and target positions, though it only helps for left-aligned dependencies.

### Increased Hidden Dimensions

A larger context vector can store more information, but this approach has diminishing returns. Doubling the hidden size doubles the number of parameters in recurrent connections (quadratic growth), while the information capacity grows only linearly. Beyond a certain point, the additional parameters make training harder without proportional gains in representation quality.

### The Attention Solution

The information bottleneck ultimately motivated the development of attention mechanisms (Bahdanau et al., 2015). Rather than compressing the entire input into a single vector, attention allows the decoder to **dynamically access different encoder positions at each decoding step**:

$$\alpha_{t,j} = \frac{\exp(e_{t,j})}{\sum_{k=1}^{T} \exp(e_{t,k})}$$

$$\mathbf{c}_t = \sum_{j=1}^{T} \alpha_{t,j} \mathbf{h}_j^{enc}$$

With attention, the "context vector" becomes a **time-varying weighted combination** of all encoder states, computed separately for each decoder timestep. The decoder can focus on relevant source positions for each output token, fundamentally resolving the fixed-capacity limitation.

The information available to the decoder now scales with the source length—each encoder position contributes its hidden state to the attention pool—rather than being constrained by a single vector's dimensionality. This insight transformed seq2seq performance and laid the groundwork for the transformer architecture.

## Summary

The context vector is the critical bridge in the encoder-decoder framework, carrying all source information to the decoder. Its fixed dimensionality creates a fundamental information bottleneck that degrades performance on long sequences. Key insights include: the theoretical bound on mutual information between source and context is independent of input length; multi-layer architectures with residual connections can capture richer hierarchical representations; and pooling strategies (mean, max, multi-layer) can partially improve information retention. However, the definitive solution to the bottleneck is the attention mechanism, which replaces the single static context with dynamic, time-varying access to all encoder positions.
