# Attention Mechanisms for RNNs

## Introduction

Attention mechanisms, introduced by Bahdanau et al. (2015), revolutionized sequence-to-sequence models by allowing the decoder to dynamically focus on relevant parts of the source sequence. Rather than compressing the entire input into a single context vector, attention computes a weighted combination of encoder states for each decoder step.

## The Attention Intuition

Consider translating "The cat sat on the mat" to French. When generating "chat" (cat), the decoder should focus on "cat"; when generating "tapis" (mat), it should focus on "mat". Attention provides this selective focus mechanism.

```
Source:  The  cat  sat  on  the  mat
          ↓    ↓    ↓    ↓    ↓    ↓
Weights: 0.1  0.7  0.1  0.0  0.0  0.1  → "chat"
         0.0  0.1  0.0  0.0  0.1  0.8  → "tapis"
```

## Mathematical Formulation

At each decoder timestep $t$, attention computes:

### 1. Alignment Scores
Measure how well source position $j$ matches decoder state $s_{t-1}$:

$$e_{tj} = \text{score}(s_{t-1}, h_j)$$

### 2. Attention Weights
Normalize scores across all source positions:

$$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{T_s} \exp(e_{tk})}$$

### 3. Context Vector
Weighted sum of encoder hidden states:

$$c_t = \sum_{j=1}^{T_s} \alpha_{tj} h_j$$

### 4. Decoder Update
Incorporate context into decoder:

$$s_t = \text{DecoderRNN}(y_{t-1}, s_{t-1}, c_t)$$

## Scoring Functions

### Additive (Bahdanau) Attention

$$e_{tj} = v^T \tanh(W_s s_{t-1} + W_h h_j + b)$$

Parameters: $W_s \in \mathbb{R}^{d_a \times d_s}$, $W_h \in \mathbb{R}^{d_a \times d_h}$, $v \in \mathbb{R}^{d_a}$

```python
class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention."""
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.W_s = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.W_h = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden: (batch, decoder_dim)
            encoder_outputs: (batch, src_len, encoder_dim)
            mask: (batch, src_len) - 1 for valid, 0 for padding
        
        Returns:
            context: (batch, encoder_dim)
            weights: (batch, src_len)
        """
        # Project decoder hidden state
        # (batch, 1, attention_dim)
        decoder_proj = self.W_s(decoder_hidden).unsqueeze(1)
        
        # Project encoder outputs
        # (batch, src_len, attention_dim)
        encoder_proj = self.W_h(encoder_outputs)
        
        # Compute scores
        # (batch, src_len, 1)
        scores = self.v(torch.tanh(decoder_proj + encoder_proj))
        scores = scores.squeeze(-1)  # (batch, src_len)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute weights
        weights = torch.softmax(scores, dim=-1)  # (batch, src_len)
        
        # Compute context
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # (batch, encoder_dim)
        
        return context, weights
```

### Multiplicative (Luong) Attention

$$e_{tj} = s_{t-1}^T W_a h_j$$

```python
class MultiplicativeAttention(nn.Module):
    """Luong-style multiplicative attention."""
    
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.W_a = nn.Linear(encoder_dim, decoder_dim, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden: (batch, decoder_dim)
            encoder_outputs: (batch, src_len, encoder_dim)
        """
        # Project encoder outputs
        # (batch, src_len, decoder_dim)
        encoder_proj = self.W_a(encoder_outputs)
        
        # Compute scores via dot product
        # (batch, src_len)
        scores = torch.bmm(encoder_proj, decoder_hidden.unsqueeze(-1)).squeeze(-1)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, weights
```

### Scaled Dot-Product Attention

$$e_{tj} = \frac{s_{t-1}^T h_j}{\sqrt{d}}$$

The scaling prevents softmax saturation with high-dimensional vectors.

```python
class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention (as in Transformers)."""
    
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
    
    def forward(self, query, keys, values, mask=None):
        """
        Args:
            query: (batch, query_len, dim)
            keys: (batch, key_len, dim)
            values: (batch, key_len, dim)
        
        Returns:
            output: (batch, query_len, dim)
            weights: (batch, query_len, key_len)
        """
        # (batch, query_len, key_len)
        scores = torch.bmm(query, keys.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = torch.softmax(scores, dim=-1)
        output = torch.bmm(weights, values)
        
        return output, weights
```

## Seq2Seq with Attention

```python
class AttentionDecoder(nn.Module):
    """LSTM decoder with attention."""
    
    def __init__(self, vocab_size, embed_dim, encoder_dim, decoder_dim, 
                 attention_dim, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = AdditiveAttention(encoder_dim, decoder_dim, attention_dim)
        
        # LSTM takes embedding + context as input
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        
        # Output projection
        self.fc = nn.Linear(decoder_dim + encoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward_step(self, input_token, hidden, cell, encoder_outputs, mask=None):
        """
        Single decoding step with attention.
        
        Args:
            input_token: (batch, 1)
            hidden: Decoder hidden state (batch, decoder_dim)
            cell: Decoder cell state (batch, decoder_dim)
            encoder_outputs: (batch, src_len, encoder_dim)
        """
        # Embed input
        embedded = self.dropout(self.embedding(input_token.squeeze(1)))
        
        # Compute attention
        context, weights = self.attention(hidden, encoder_outputs, mask)
        
        # LSTM step
        lstm_input = torch.cat([embedded, context], dim=-1)
        hidden, cell = self.lstm(lstm_input, (hidden, cell))
        
        # Output projection
        output = self.fc(torch.cat([hidden, context], dim=-1))
        
        return output, hidden, cell, weights


class Seq2SeqAttention(nn.Module):
    """Complete Seq2Seq model with attention."""
    
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, tgt, src_mask=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.fc.out_features
        
        # Store outputs and attention weights
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=self.device)
        attentions = torch.zeros(batch_size, tgt_len, src.size(1), device=self.device)
        
        # Encode source
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # Use last layer's hidden state
        hidden = hidden[-1]
        cell = cell[-1]
        
        # First input is <SOS>
        input_token = tgt[:, 0:1]
        
        for t in range(1, tgt_len):
            output, hidden, cell, attn_weights = self.decoder.forward_step(
                input_token, hidden, cell, encoder_outputs, src_mask
            )
            
            outputs[:, t] = output
            attentions[:, t] = attn_weights
            
            # Teacher forcing decision
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tgt[:, t:t+1]
            else:
                input_token = output.argmax(dim=-1, keepdim=True)
        
        return outputs, attentions
```

## Visualizing Attention

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention_weights, src_tokens, tgt_tokens, save_path=None):
    """
    Visualize attention weights as heatmap.
    
    Args:
        attention_weights: (tgt_len, src_len)
        src_tokens: List of source tokens
        tgt_tokens: List of target tokens
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        attention_weights,
        xticklabels=src_tokens,
        yticklabels=tgt_tokens,
        cmap='Blues',
        ax=ax
    )
    
    ax.set_xlabel('Source')
    ax.set_ylabel('Target')
    ax.set_title('Attention Weights')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

# Usage
# attention_weights: (batch, tgt_len, src_len) -> [0] for first sample
# plot_attention(attentions[0].cpu().numpy(), src_tokens, tgt_tokens)
```

## Local vs Global Attention

### Global Attention
Attends to all source positions (as described above).

### Local Attention (Luong et al., 2015)
Attends to a window around predicted alignment position:

```python
class LocalAttention(nn.Module):
    """Local attention attending to a window around predicted position."""
    
    def __init__(self, encoder_dim, decoder_dim, window_size=5):
        super().__init__()
        self.window_size = window_size
        
        # Predict alignment position
        self.position_predictor = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim // 2),
            nn.Tanh(),
            nn.Linear(decoder_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.attention = MultiplicativeAttention(encoder_dim, decoder_dim)
    
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        batch_size, src_len, _ = encoder_outputs.shape
        
        # Predict alignment position (0 to src_len-1)
        position = self.position_predictor(decoder_hidden) * (src_len - 1)
        position = position.squeeze(-1)  # (batch,)
        
        # Create window mask
        positions = torch.arange(src_len, device=encoder_outputs.device).float()
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Gaussian window centered at predicted position
        window_mask = torch.exp(-((positions - position.unsqueeze(-1)) ** 2) / 
                                 (2 * (self.window_size / 2) ** 2))
        
        # Combine with padding mask
        if mask is not None:
            window_mask = window_mask * mask.float()
        
        # Apply attention within window
        context, weights = self.attention(decoder_hidden, encoder_outputs, window_mask)
        
        return context, weights
```

## Multi-Head Attention

Split attention into multiple heads for diverse focus:

```python
class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, model_dim, num_heads):
        super().__init__()
        assert model_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        self.W_q = nn.Linear(model_dim, model_dim)
        self.W_k = nn.Linear(model_dim, model_dim)
        self.W_v = nn.Linear(model_dim, model_dim)
        self.W_o = nn.Linear(model_dim, model_dim)
        
        self.scale = self.head_dim ** 0.5
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into heads
        # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, 
                                                            self.num_heads * self.head_dim)
        
        # Final projection
        output = self.W_o(context)
        
        return output, weights
```

## Inference with Attention

```python
def translate_with_attention(model, src, src_vocab, tgt_vocab, max_length=50):
    """
    Translate source sequence and return attention weights for visualization.
    """
    model.eval()
    
    sos_idx = tgt_vocab.token2idx['<SOS>']
    eos_idx = tgt_vocab.token2idx['<EOS>']
    
    with torch.no_grad():
        # Encode
        encoder_outputs, (hidden, cell) = model.encoder(src)
        hidden, cell = hidden[-1], cell[-1]
        
        # Decode
        input_token = torch.tensor([[sos_idx]], device=src.device)
        tokens = []
        all_attention = []
        
        for _ in range(max_length):
            output, hidden, cell, attn = model.decoder.forward_step(
                input_token, hidden, cell, encoder_outputs
            )
            
            predicted = output.argmax(dim=-1)
            token_idx = predicted.item()
            
            tokens.append(token_idx)
            all_attention.append(attn.squeeze(0).cpu())
            
            if token_idx == eos_idx:
                break
            
            input_token = predicted.unsqueeze(0)
    
    # Convert to words
    words = [tgt_vocab.idx2token.get(idx, '<UNK>') for idx in tokens]
    attention_matrix = torch.stack(all_attention).numpy()
    
    return words, attention_matrix
```

## Summary

Attention mechanisms transform sequence-to-sequence models by:

1. **Dynamic focus**: Decoder selects relevant source positions at each step
2. **Context vector**: Weighted combination of encoder states
3. **Alignment learning**: Network learns which source words correspond to each target word

Types of attention:
- **Additive (Bahdanau)**: Learned alignment function
- **Multiplicative (Luong)**: Dot product with learned projection
- **Scaled dot-product**: Efficient, used in Transformers
- **Multi-head**: Multiple attention patterns in parallel

Attention visualization provides interpretability, showing which input elements influence each output—a key advantage for debugging and understanding model behavior.

This mechanism laid the foundation for Transformer architectures, which extend self-attention to replace recurrence entirely.
