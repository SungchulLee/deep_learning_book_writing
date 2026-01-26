# Attention Mechanisms in Sequence-to-Sequence Models

Attention mechanisms represent one of the most consequential innovations in deep learning, fundamentally transforming how neural networks process sequential information. Introduced by Bahdanau et al. (2015), attention enables decoders to dynamically focus on relevant parts of the input sequence during generation, addressing the critical information bottleneck inherent in fixed-context encoder-decoder architectures.

## Motivation and Intuition

### The Information Bottleneck Problem

In basic seq2seq models, the entire input sequence is compressed into a single fixed-dimensional context vector. This creates a fundamental limitation: the decoder must extract all necessary information from this compressed representation, regardless of input length. As sequences grow longer, the fixed-capacity context vector becomes increasingly inadequate—information from early positions gets overwritten or diluted by subsequent positions.

**Deep Insight**: The bottleneck problem is mathematically inevitable. If we have a sequence of $T$ tokens, each with $d$-dimensional hidden representations, we're compressing $O(Td)$ bits of information into $O(d)$ bits. The compression ratio grows linearly with sequence length, guaranteeing information loss for sufficiently long sequences. Attention circumvents this by maintaining $O(Td)$ accessible information throughout decoding.

### Cognitive Inspiration

Attention mechanisms draw inspiration from human cognitive attention, where we selectively focus on relevant information when performing tasks. When translating a sentence, a human translator focuses on different source words while generating each target word, rather than processing the entire source at once.

Consider translating "The cat sat on the mat" to French:
- When generating "chat" (cat), the model should focus on "cat"
- When generating "tapis" (mat), attention should shift to "mat"

```
Source:  The  cat  sat  on  the  mat
          ↓    ↓    ↓    ↓    ↓    ↓
Weights: 0.1  0.7  0.1  0.0  0.0  0.1  → "chat"
         0.0  0.1  0.0  0.0  0.1  0.8  → "tapis"
```

**Deep Insight**: This selective focus is not merely a heuristic—it reflects the compositional structure of language. Translation exhibits approximate monotonicity (words roughly align left-to-right), but with local reorderings due to syntax differences. Attention learns this alignment structure implicitly, without explicit supervision.

## Mathematical Framework

### Attention Mechanism Overview

Given encoder hidden states $\mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_T] \in \mathbb{R}^{T \times d_h}$ and decoder hidden state $\mathbf{s}_t \in \mathbb{R}^{d_s}$, attention computes a context vector $\mathbf{c}_t$ as a weighted sum of encoder states:

$$\mathbf{c}_t = \sum_{i=1}^{T} \alpha_{ti} \mathbf{h}_i$$

where the attention weights $\alpha_{ti}$ satisfy the probability simplex constraints:

$$\sum_{i=1}^{T} \alpha_{ti} = 1, \quad \alpha_{ti} \geq 0$$

The weights are computed through a softmax over attention scores:

$$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{j=1}^{T} \exp(e_{tj})}$$

The score function $e_{ti} = \text{score}(\mathbf{s}_t, \mathbf{h}_i)$ measures compatibility between the decoder state and each encoder state.

**Deep Insight**: Attention can be interpreted through multiple theoretical lenses:
1. **Soft Addressing**: Attention is content-based memory addressing, where the decoder queries a memory bank (encoder states) and retrieves a weighted combination based on content similarity.
2. **Variational Perspective**: Hard attention (sampling one position) can be seen as sampling from a categorical latent variable; soft attention approximates the expectation under this distribution.
3. **Kernel Smoothing**: The softmax attention over dot products implements a form of kernel density estimation in the embedding space.

### The Complete Attention Pipeline

At each decoder timestep $t$, attention computes:

**Step 1 - Alignment Scores**: Measure how well source position $j$ matches decoder state:
$$e_{tj} = \text{score}(\mathbf{s}_{t-1}, \mathbf{h}_j)$$

**Step 2 - Attention Weights**: Normalize scores across all source positions:
$$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{T_s} \exp(e_{tk})}$$

**Step 3 - Context Vector**: Weighted sum of encoder hidden states:
$$\mathbf{c}_t = \sum_{j=1}^{T_s} \alpha_{tj} \mathbf{h}_j$$

**Step 4 - Decoder Update**: Incorporate context into decoder:
$$\mathbf{s}_t = \text{DecoderRNN}(y_{t-1}, \mathbf{s}_{t-1}, \mathbf{c}_t)$$

## Scoring Functions: A Taxonomy

### Bahdanau (Additive) Attention

Bahdanau et al. (2015) proposed additive attention, which computes scores through a feedforward network:

$$e_{ti} = \mathbf{v}^\top \tanh(\mathbf{W}_s \mathbf{s}_{t-1} + \mathbf{W}_h \mathbf{h}_i + \mathbf{b})$$

where $\mathbf{W}_s \in \mathbb{R}^{d_a \times d_s}$, $\mathbf{W}_h \in \mathbb{R}^{d_a \times d_h}$, and $\mathbf{v} \in \mathbb{R}^{d_a}$ are learned parameters.

**Key Characteristics**:
- Uses the **previous** decoder state $\mathbf{s}_{t-1}$ before the current prediction
- The attention context is used as **input** to the decoder RNN
- More expressive due to the learned transformation matrices
- Allows different dimensions for encoder and decoder ($d_h \neq d_s$)

**Deep Insight**: The additive score function can be interpreted as a two-layer neural network with a shared bias. The tanh nonlinearity creates a bounded, smooth similarity surface that prevents extreme score values. The vector $\mathbf{v}$ acts as a learned importance weighting over the $d_a$ "alignment features."

### Luong (Multiplicative) Attention

Luong et al. (2015) proposed simpler multiplicative variants:

**Dot Product**:
$$e_{ti} = \mathbf{s}_t^\top \mathbf{h}_i$$

Requires $d_s = d_h$ for dimension compatibility.

**General**:
$$e_{ti} = \mathbf{s}_t^\top \mathbf{W}_a \mathbf{h}_i$$

where $\mathbf{W}_a \in \mathbb{R}^{d_s \times d_h}$ allows different dimensions.

**Concat**:
$$e_{ti} = \mathbf{v}^\top \tanh(\mathbf{W}_a [\mathbf{s}_t; \mathbf{h}_i])$$

**Key Characteristics**:
- Uses the **current** decoder state $\mathbf{s}_t$
- Attention context is used **after** the decoder RNN step
- Computationally more efficient, especially dot product

**Deep Insight**: The dot product can be interpreted geometrically as measuring the cosine of the angle between vectors (scaled by magnitudes). This makes it sensitive to both direction (semantic similarity) and magnitude (confidence/norm). The general form $\mathbf{s}^\top \mathbf{W} \mathbf{h}$ learns a bilinear form, which can model asymmetric relationships between query and key spaces.

### Scaled Dot-Product Attention

$$e_{ti} = \frac{\mathbf{s}_{t}^\top \mathbf{h}_i}{\sqrt{d}}$$

The scaling factor $\sqrt{d}$ prevents softmax saturation with high-dimensional vectors.

**Deep Insight**: Without scaling, dot products grow as $O(\sqrt{d})$ for random unit vectors, pushing softmax into saturation regions where gradients vanish. The $\sqrt{d}$ scaling normalizes variance to 1, maintaining healthy gradient flow. This is critical insight that enabled the Transformer architecture to scale to very high dimensions.

### Comparative Analysis

| Aspect | Bahdanau (Additive) | Luong (General) | Luong (Dot) | Scaled Dot |
|--------|---------------------|-----------------|-------------|------------|
| **Computation** | $O(d_a(d_s + d_h))$ | $O(d_s \cdot d_h)$ | $O(d)$ | $O(d)$ |
| **Parameters** | $d_a(d_s + d_h) + d_a$ | $d_s \cdot d_h$ | 0 | 0 |
| **Expressiveness** | High | Medium | Low | Low |
| **Speed** | Slower | Medium | Fastest | Fastest |
| **Gradient Flow** | Good | Good | May saturate | Stable |

## PyTorch Implementation

### Bahdanau Attention Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.
    
    Computes attention scores using a feedforward network that takes
    the sum of transformed decoder state and encoder states.
    
    Architecture Insight: The additive formulation allows the network to learn
    complex, non-linear alignment patterns. The separate projections for
    decoder and encoder states enable dimension mismatch handling and
    provide independent capacity for each modality.
    
    Args:
        decoder_hidden_size: Dimension of decoder hidden state
        encoder_hidden_size: Dimension of encoder hidden states
        attention_dim: Dimension of attention hidden layer (default: None, uses decoder_hidden_size)
    """
    
    def __init__(
        self,
        decoder_hidden_size: int,
        encoder_hidden_size: int,
        attention_dim: int = None
    ):
        super().__init__()
        
        attention_dim = attention_dim or decoder_hidden_size
        
        # Learnable transformations
        # W_decoder projects the query (decoder state) into alignment space
        self.W_decoder = nn.Linear(decoder_hidden_size, attention_dim, bias=False)
        # W_encoder projects keys (encoder states) - can be precomputed
        self.W_encoder = nn.Linear(encoder_hidden_size, attention_dim, bias=False)
        # v computes final scalar score from alignment features
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        Compute attention weights and context vector.
        
        Args:
            decoder_state: Current decoder hidden state (batch_size, decoder_hidden_size)
                          or (batch_size, 1, decoder_hidden_size)
            encoder_outputs: All encoder hidden states (batch_size, src_len, encoder_hidden_size)
            mask: Optional mask for padding (batch_size, src_len), True for valid positions
            
        Returns:
            context: Weighted sum of encoder outputs (batch_size, 1, encoder_hidden_size)
            attention_weights: Attention distribution (batch_size, src_len)
        """
        # Ensure decoder_state has sequence dimension for broadcasting
        if decoder_state.dim() == 2:
            decoder_state = decoder_state.unsqueeze(1)  # (batch, 1, hidden)
        
        # Transform decoder state: (batch, 1, attention_dim)
        decoder_transformed = self.W_decoder(decoder_state)
        
        # Transform encoder outputs: (batch, src_len, attention_dim)
        # Note: This can be precomputed once per sequence for efficiency
        encoder_transformed = self.W_encoder(encoder_outputs)
        
        # Compute attention scores via broadcasting addition
        # (batch, 1, attn_dim) + (batch, src_len, attn_dim) = (batch, src_len, attn_dim)
        combined = torch.tanh(decoder_transformed + encoder_transformed)
        
        # Project to scalar scores: (batch, src_len)
        scores = self.v(combined).squeeze(-1)
        
        # Apply mask before softmax (critical for variable-length sequences)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Normalize to get attention weights (probability distribution)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute context vector via weighted sum: (batch, 1, encoder_hidden)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        
        return context, attention_weights
```

### Luong Attention Module

```python
class LuongAttention(nn.Module):
    """
    Luong (Multiplicative) Attention Mechanism.
    
    Supports three scoring functions: dot, general, and concat.
    
    Design Insight: Luong attention computes attention AFTER the RNN step,
    using the current hidden state. This creates a different information
    flow compared to Bahdanau, where attention influences the RNN input.
    
    Args:
        decoder_hidden_size: Dimension of decoder hidden state
        encoder_hidden_size: Dimension of encoder hidden states
        method: Scoring method - 'dot', 'general', or 'concat' (default: 'general')
    """
    
    def __init__(
        self,
        decoder_hidden_size: int,
        encoder_hidden_size: int,
        method: str = 'general'
    ):
        super().__init__()
        
        self.method = method
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        
        if method == 'general':
            # Bilinear form: s^T W h
            self.W = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=False)
        elif method == 'concat':
            # Additive form similar to Bahdanau but with current state
            self.W = nn.Linear(decoder_hidden_size + encoder_hidden_size, decoder_hidden_size)
            self.v = nn.Linear(decoder_hidden_size, 1, bias=False)
        elif method == 'dot':
            assert decoder_hidden_size == encoder_hidden_size, \
                "Dot attention requires equal dimensions"
        else:
            raise ValueError(f"Unknown attention method: {method}")
            
    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        Compute attention weights and context vector.
        
        Args:
            decoder_state: (batch_size, decoder_hidden_size) or (batch_size, 1, decoder_hidden_size)
            encoder_outputs: (batch_size, src_len, encoder_hidden_size)
            mask: Optional padding mask (batch_size, src_len)
            
        Returns:
            context: (batch_size, 1, encoder_hidden_size)
            attention_weights: (batch_size, src_len)
        """
        if decoder_state.dim() == 2:
            decoder_state = decoder_state.unsqueeze(1)
        
        if self.method == 'dot':
            # Direct dot product: (batch, 1, hidden) @ (batch, hidden, src_len)
            scores = torch.bmm(decoder_state, encoder_outputs.transpose(1, 2))
            scores = scores.squeeze(1)  # (batch, src_len)
            
        elif self.method == 'general':
            # Transform encoder, then dot with decoder
            encoder_transformed = self.W(encoder_outputs)  # (batch, src_len, dec_hidden)
            scores = torch.bmm(decoder_state, encoder_transformed.transpose(1, 2))
            scores = scores.squeeze(1)
            
        elif self.method == 'concat':
            src_len = encoder_outputs.size(1)
            # Expand decoder state to match encoder length
            decoder_expanded = decoder_state.expand(-1, src_len, -1)
            # Concatenate: (batch, src_len, dec_hidden + enc_hidden)
            combined = torch.cat([decoder_expanded, encoder_outputs], dim=-1)
            scores = self.v(torch.tanh(self.W(combined))).squeeze(-1)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        
        return context, attention_weights
```

### Scaled Dot-Product Attention

```python
class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention (as in Transformers).
    
    Deep Insight: This formulation generalizes to batch matrix operations,
    enabling efficient parallel computation across multiple queries.
    The scaling factor sqrt(d) is crucial for training stability.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
    
    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        Args:
            query: (batch, query_len, dim)
            keys: (batch, key_len, dim)
            values: (batch, key_len, dim)
        
        Returns:
            output: (batch, query_len, dim)
            weights: (batch, query_len, key_len)
        """
        # Compute scaled dot-product: (batch, query_len, key_len)
        scores = torch.bmm(query, keys.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        output = torch.bmm(weights, values)
        
        return output, weights
```

## Attention Decoder Integration

### Bahdanau-Style Integration

```python
class AttentionDecoder(nn.Module):
    """
    Decoder with attention mechanism for seq2seq models.
    
    Integrates attention context into the decoding process, allowing
    the decoder to focus on different parts of the input at each step.
    
    Design Insight: The choice between Bahdanau and Luong integration
    affects information flow:
    - Bahdanau: Context influences what the RNN processes (input-feeding)
    - Luong: Context influences final prediction after RNN processing
    
    Args:
        output_size: Size of output vocabulary
        embedding_dim: Dimension of token embeddings
        hidden_size: Decoder hidden state dimension
        encoder_hidden_size: Encoder hidden state dimension
        num_layers: Number of RNN layers (default: 1)
        dropout: Dropout probability (default: 0.1)
        attention_type: 'bahdanau' or 'luong' (default: 'bahdanau')
        rnn_type: 'LSTM' or 'GRU' (default: 'LSTM')
    """
    
    def __init__(
        self,
        output_size: int,
        embedding_dim: int,
        hidden_size: int,
        encoder_hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        attention_type: str = 'bahdanau',
        rnn_type: str = 'LSTM'
    ):
        super().__init__()
        
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.attention_type = attention_type
        
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Create attention module
        if attention_type == 'bahdanau':
            self.attention = BahdanauAttention(hidden_size, encoder_hidden_size)
            # Bahdanau: context concatenated with embedding as RNN input
            rnn_input_size = embedding_dim + encoder_hidden_size
        else:
            self.attention = LuongAttention(hidden_size, encoder_hidden_size)
            # Luong: attention applied after RNN
            rnn_input_size = embedding_dim
        
        # RNN layer
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.rnn = rnn_cls(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection combines RNN output, context, and embedding
        # This "maxout" style combination provides richer features for prediction
        self.output_projection = nn.Linear(
            hidden_size + encoder_hidden_size + embedding_dim,
            output_size
        )
        
    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        cell: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        Single decoding step with attention.
        
        Args:
            input_token: Previous token indices (batch_size, 1)
            hidden: Previous hidden state (num_layers, batch_size, hidden_size)
            encoder_outputs: Encoder hidden states (batch_size, src_len, encoder_hidden_size)
            cell: Previous cell state for LSTM (optional)
            mask: Attention mask (batch_size, src_len)
            
        Returns:
            output: Token logits (batch_size, output_size)
            hidden: Updated hidden state
            cell: Updated cell state (LSTM)
            attention_weights: Attention distribution (batch_size, src_len)
        """
        # Embed input token
        embedded = self.dropout(self.embedding(input_token))  # (batch, 1, embed_dim)
        
        if self.attention_type == 'bahdanau':
            # Bahdanau: compute attention BEFORE RNN step
            # Use top layer hidden state as attention query
            query = hidden[-1]  # (batch, hidden_size)
            context, attention_weights = self.attention(query, encoder_outputs, mask)
            
            # Concatenate embedding and context as RNN input
            # This is "input feeding" - context influences RNN state
            rnn_input = torch.cat([embedded, context], dim=-1)
        else:
            # Luong: embedding only as RNN input
            rnn_input = embedded
        
        # RNN step
        if self.rnn_type == 'LSTM':
            rnn_output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        else:
            rnn_output, hidden = self.rnn(rnn_input, hidden)
            cell = None
        
        if self.attention_type == 'luong':
            # Luong: compute attention AFTER RNN step
            query = rnn_output.squeeze(1)
            context, attention_weights = self.attention(query, encoder_outputs, mask)
        
        # Combine RNN output, context, and embedding for prediction
        combined = torch.cat([
            rnn_output.squeeze(1),
            context.squeeze(1),
            embedded.squeeze(1)
        ], dim=-1)
        
        output = self.output_projection(combined)
        
        return output, hidden, cell, attention_weights
```

### Complete Seq2Seq Model

```python
class Seq2SeqAttention(nn.Module):
    """
    Sequence-to-sequence model with attention mechanism.
    
    Combines encoder, attention-based decoder, and provides methods
    for both training and inference.
    
    Deep Insight: The model learns to jointly optimize:
    1. Encoder representations that are informative for attention
    2. Attention patterns that capture source-target alignment
    3. Decoder that effectively uses attended context
    
    Args:
        encoder: Encoder module
        decoder: Attention decoder module
        device: Computation device
        pad_idx: Padding token index for masking
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
        pad_idx: int = 0
    ):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx
        
    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create attention mask for padding positions."""
        return src != self.pad_idx
        
    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
        src_lengths: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass with teacher forcing.
        
        Args:
            src: Source sequences (batch_size, src_len)
            trg: Target sequences (batch_size, trg_len)
            teacher_forcing_ratio: Probability of using ground truth
            src_lengths: Actual source lengths
            
        Returns:
            outputs: Predicted logits (batch_size, trg_len, vocab_size)
            attentions: Attention weights (batch_size, trg_len, src_len)
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        src_len = src.size(1)
        vocab_size = self.decoder.output_size
        
        # Allocate output tensors
        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)
        attentions = torch.zeros(batch_size, trg_len, src_len, device=self.device)
        
        # Create attention mask for padding
        mask = self.create_mask(src)
        
        # Encode source sequence
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # Initialize decoder input with <sos> token
        decoder_input = trg[:, 0].unsqueeze(1)
        
        # Decode step by step
        for t in range(1, trg_len):
            output, hidden, cell, attention = self.decoder(
                decoder_input, hidden, encoder_outputs, cell, mask
            )
            
            outputs[:, t] = output
            attentions[:, t] = attention
            
            # Teacher forcing: use ground truth or model prediction
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            
            if use_teacher:
                decoder_input = trg[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(dim=-1).unsqueeze(1)
        
        return outputs, attentions
    
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 50,
        sos_idx: int = 1,
        eos_idx: int = 2,
        src_lengths: torch.Tensor = None
    ) -> tuple:
        """
        Generate output sequence with attention visualization.
        
        Args:
            src: Source sequence (batch_size, src_len)
            max_len: Maximum generation length
            sos_idx: Start token index
            eos_idx: End token index
            src_lengths: Actual source lengths
            
        Returns:
            generated: Generated token indices
            attention_weights: Attention weights for each step
        """
        self.eval()
        batch_size = src.size(0)
        
        mask = self.create_mask(src)
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        decoder_input = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=self.device
        )
        
        generated = [decoder_input]
        attention_list = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for _ in range(max_len):
            output, hidden, cell, attention = self.decoder(
                decoder_input, hidden, encoder_outputs, cell, mask
            )
            
            attention_list.append(attention.unsqueeze(1))
            predicted = output.argmax(dim=-1).unsqueeze(1)
            generated.append(predicted)
            
            # Track finished sequences
            finished |= (predicted.squeeze(1) == eos_idx)
            if finished.all():
                break
                
            decoder_input = predicted
        
        generated = torch.cat(generated, dim=1)
        attentions = torch.cat(attention_list, dim=1)
        
        return generated, attentions
```

## Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions:

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Deep Insight: Multiple attention heads provide several benefits:
    1. Different heads can capture different types of relationships
       (syntactic, semantic, positional)
    2. Ensemble effect reduces variance
    3. Different heads can attend to different scales of context
    
    The concatenation of heads followed by linear projection allows
    information mixing across attention patterns.
    """
    
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        # Separate projections for queries, keys, values
        self.W_q = nn.Linear(model_dim, model_dim)
        self.W_k = nn.Linear(model_dim, model_dim)
        self.W_v = nn.Linear(model_dim, model_dim)
        
        # Output projection mixes information across heads
        self.W_o = nn.Linear(model_dim, model_dim)
        
        self.scale = self.head_dim ** 0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        Args:
            query: (batch, query_len, model_dim)
            key: (batch, key_len, model_dim)
            value: (batch, key_len, model_dim)
            mask: Optional attention mask
            
        Returns:
            output: (batch, query_len, model_dim)
            weights: (batch, num_heads, query_len, key_len)
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into heads: (batch, seq_len, num_heads, head_dim)
        # Then transpose: (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention per head
        # (batch, num_heads, query_len, key_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            # Expand mask for heads dimension
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        # (batch, num_heads, query_len, head_dim)
        context = torch.matmul(weights, V)
        
        # Concatenate heads: (batch, query_len, model_dim)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.head_dim
        )
        
        # Final projection
        output = self.W_o(context)
        
        return output, weights
```

## Local vs Global Attention

### Global Attention
Attends to all source positions (as described above). Complexity is $O(T_s)$ per decoder step.

### Local Attention

Luong et al. also introduced local attention, which focuses on a window around a predicted alignment position, reducing complexity to $O(D)$ where $D$ is the window size:

```python
class LocalAttention(nn.Module):
    """
    Local attention mechanism that focuses on a window of positions.
    
    Instead of attending to all source positions, local attention
    predicts a center position and attends to a Gaussian window around it.
    
    Deep Insight: Local attention is particularly effective for:
    1. Monotonic alignment tasks (e.g., speech recognition)
    2. Very long sequences where global attention is expensive
    3. Tasks where relevant context is spatially localized
    
    The Gaussian window provides soft boundaries, allowing gradient flow
    to update the position predictor.
    
    Args:
        decoder_hidden_size: Decoder hidden dimension
        encoder_hidden_size: Encoder hidden dimension
        window_size: Half-window size D (window = 2*D + 1)
    """
    
    def __init__(
        self,
        decoder_hidden_size: int,
        encoder_hidden_size: int,
        window_size: int = 5
    ):
        super().__init__()
        
        self.window_size = window_size
        
        # Position predictor: learns to predict alignment center
        self.position_predictor = nn.Sequential(
            nn.Linear(decoder_hidden_size, decoder_hidden_size // 2),
            nn.Tanh(),
            nn.Linear(decoder_hidden_size // 2, 1),
            nn.Sigmoid()  # Bounds output to [0, 1]
        )
        
        # Base attention within window
        self.attention = LuongAttention(decoder_hidden_size, encoder_hidden_size)
        
    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        Compute local attention.
        
        Args:
            decoder_state: (batch_size, decoder_hidden_size)
            encoder_outputs: (batch_size, src_len, encoder_hidden_size)
            mask: Optional padding mask
            
        Returns:
            context: Context vector
            attention_weights: Attention distribution (sparse, window only)
        """
        batch_size, src_len, _ = encoder_outputs.shape
        
        # Predict alignment position p_t in [0, src_len-1]
        position = self.position_predictor(decoder_state) * (src_len - 1)
        position = position.squeeze(-1)  # (batch_size,)
        
        # Create position indices
        positions = torch.arange(src_len, device=encoder_outputs.device).float()
        positions = positions.unsqueeze(0).expand(batch_size, -1)  # (batch, src_len)
        
        # Gaussian window centered at predicted position
        # Variance controls window sharpness
        sigma = self.window_size / 2
        gaussian = torch.exp(
            -((positions - position.unsqueeze(-1)) ** 2) / (2 * sigma ** 2)
        )
        
        # Compute base attention scores
        context, attention_weights = self.attention(
            decoder_state, encoder_outputs, mask
        )
        
        # Apply Gaussian window as multiplicative mask
        attention_weights = attention_weights * gaussian
        
        # Renormalize
        attention_weights = attention_weights / (
            attention_weights.sum(dim=-1, keepdim=True) + 1e-10
        )
        
        # Recompute context with windowed attention
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        
        return context, attention_weights
```

## Coverage Mechanism

Coverage tracking prevents the decoder from attending to the same positions repeatedly (over-generation) or ignoring important positions (under-generation):

```python
class CoverageAttention(nn.Module):
    """
    Attention with coverage mechanism to track attended positions.
    
    Maintains a coverage vector that accumulates attention over time,
    penalizing repeated attention to the same positions.
    
    Deep Insight: Coverage addresses a fundamental failure mode in
    attention-based models. Without coverage:
    - The same content word might be translated multiple times
    - Some source words might never be attended to
    
    Coverage creates a "memory" of past attention, encouraging
    the model to attend to new positions. The coverage loss term
    provides explicit supervision for this behavior.
    
    Args:
        decoder_hidden_size: Decoder hidden dimension
        encoder_hidden_size: Encoder hidden dimension
        attention_dim: Attention layer dimension
    """
    
    def __init__(
        self,
        decoder_hidden_size: int,
        encoder_hidden_size: int,
        attention_dim: int
    ):
        super().__init__()
        
        self.W_decoder = nn.Linear(decoder_hidden_size, attention_dim, bias=False)
        self.W_encoder = nn.Linear(encoder_hidden_size, attention_dim, bias=False)
        # Coverage feature: how much attention each position has received
        self.W_coverage = nn.Linear(1, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        coverage: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        Compute attention with coverage.
        
        Args:
            decoder_state: (batch_size, decoder_hidden_size)
            encoder_outputs: (batch_size, src_len, encoder_hidden_size)
            coverage: Cumulative attention (batch_size, src_len)
            mask: Padding mask
            
        Returns:
            context: Context vector
            attention_weights: Current attention distribution
            coverage: Updated cumulative coverage
        """
        if decoder_state.dim() == 2:
            decoder_state = decoder_state.unsqueeze(1)
        
        # Transform inputs
        decoder_transformed = self.W_decoder(decoder_state)
        encoder_transformed = self.W_encoder(encoder_outputs)
        
        # Coverage as additional feature
        coverage_transformed = self.W_coverage(coverage.unsqueeze(-1))
        
        # Combine: decoder + encoder + coverage
        combined = torch.tanh(
            decoder_transformed + encoder_transformed + coverage_transformed
        )
        
        scores = self.v(combined).squeeze(-1)
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Update coverage with current attention
        coverage = coverage + attention_weights
        
        # Compute context
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        
        return context, attention_weights, coverage


def coverage_loss(
    attention_weights: torch.Tensor,
    coverage: torch.Tensor
) -> torch.Tensor:
    """
    Compute coverage loss to penalize repeated attention.
    
    The loss is the sum of minimum between attention and coverage,
    encouraging the model to attend to positions not yet covered.
    
    Deep Insight: The min operation is crucial here. It only penalizes
    when BOTH the current attention AND previous coverage are high
    at the same position. This allows:
    - Attending to new positions freely
    - Re-attending only at penalty
    
    Args:
        attention_weights: Current attention (batch_size, src_len)
        coverage: Cumulative coverage before this step (batch_size, src_len)
        
    Returns:
        loss: Coverage loss scalar
    """
    # Penalize only where both attention and coverage are high
    loss = torch.sum(torch.min(attention_weights, coverage))
    return loss
```

## Attention Visualization and Analysis

### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def visualize_attention(
    source_tokens: list,
    target_tokens: list,
    attention_weights: np.ndarray,
    figsize: tuple = (10, 8),
    cmap: str = 'Blues',
    save_path: str = None
) -> plt.Figure:
    """
    Visualize attention weights as a heatmap.
    
    Deep Insight: Attention visualization reveals:
    1. Whether the model learns sensible alignments
    2. Attention patterns (diagonal for monotonic, scattered for reordering)
    3. Failure modes (attention collapse, ignoring positions)
    
    Args:
        source_tokens: List of source tokens
        target_tokens: List of target tokens
        attention_weights: Attention matrix (trg_len, src_len)
        figsize: Figure size
        cmap: Colormap
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Trim attention to actual token lengths
    trg_len = len(target_tokens)
    src_len = len(source_tokens)
    attention = attention_weights[:trg_len, :src_len]
    
    # Create heatmap
    sns.heatmap(
        attention,
        xticklabels=source_tokens,
        yticklabels=target_tokens,
        cmap=cmap,
        ax=ax,
        cbar=True,
        linewidths=0.5,
        vmin=0,
        vmax=1
    )
    
    ax.set_xlabel('Source', fontsize=12)
    ax.set_ylabel('Target', fontsize=12)
    ax.set_title('Attention Weights', fontsize=14)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
```

### Attention Diagnostics

```python
def attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of attention distributions.
    
    Lower entropy indicates more focused (confident) attention,
    higher entropy indicates diffuse (uncertain) attention.
    
    Deep Insight: Entropy tracking reveals:
    - Training dynamics (entropy often decreases as model learns)
    - Token difficulty (function words often have higher entropy)
    - Model uncertainty per position
    
    Args:
        attention_weights: (batch_size, trg_len, src_len)
        
    Returns:
        entropy: (batch_size, trg_len) entropy values
    """
    eps = 1e-10  # Prevent log(0)
    entropy = -torch.sum(
        attention_weights * torch.log(attention_weights + eps),
        dim=-1
    )
    return entropy


def attention_coverage_analysis(attention_weights: torch.Tensor) -> dict:
    """
    Analyze attention patterns for diagnostic purposes.
    
    Returns:
        Dictionary with coverage statistics
    """
    # Cumulative coverage: how much each source position was attended
    total_coverage = attention_weights.sum(dim=1)  # (batch, src_len)
    
    # Coverage statistics
    coverage_mean = total_coverage.mean(dim=-1)
    coverage_std = total_coverage.std(dim=-1)
    coverage_max = total_coverage.max(dim=-1).values
    coverage_min = total_coverage.min(dim=-1).values
    
    # Under/over-translation indicators
    under_translated = (total_coverage < 0.5).sum(dim=-1).float()
    over_translated = (total_coverage > 1.5).sum(dim=-1).float()
    
    return {
        'coverage_mean': coverage_mean,
        'coverage_std': coverage_std,
        'coverage_max': coverage_max,
        'coverage_min': coverage_min,
        'under_translated_count': under_translated,
        'over_translated_count': over_translated
    }
```

## Practical Considerations

### Attention Regularization

Attention can sometimes produce degenerate patterns. Regularization techniques help:

```python
def attention_regularization_loss(
    attention_weights: torch.Tensor,
    lambda_entropy: float = 0.1,
    lambda_sparse: float = 0.01
) -> torch.Tensor:
    """
    Regularization losses for attention.
    
    Deep Insight: Different regularizers serve different purposes:
    - Entropy: Encourages peaky attention (lower entropy)
    - Sparsity: Encourages attending to few positions
    - These can conflict: tune carefully for your task
    
    Args:
        attention_weights: (batch_size, trg_len, src_len)
        lambda_entropy: Weight for entropy regularization
        lambda_sparse: Weight for sparsity regularization
        
    Returns:
        Combined regularization loss
    """
    # Entropy regularization: encourage focused attention
    entropy = attention_entropy(attention_weights).mean()
    
    # L1 sparsity: encourage sparse attention patterns
    sparsity = torch.abs(attention_weights).mean()
    
    return lambda_entropy * entropy + lambda_sparse * sparsity
```

### Computational Efficiency

For long sequences, attention computation can be expensive. Pre-computing encoder transformations helps:

```python
class EfficientAttention(nn.Module):
    """
    Efficient attention that pre-computes encoder transformations.
    
    Deep Insight: The key insight is that encoder projections are
    the same across all decoder timesteps. Pre-computing them once
    reduces complexity from O(T_t * T_s * d) to O(T_t * T_s) + O(T_s * d).
    """
    
    def __init__(self, decoder_hidden_size: int, encoder_hidden_size: int):
        super().__init__()
        self.W_decoder = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        self.W_encoder = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)
        
        self._encoder_transformed = None
        
    def precompute_encoder(self, encoder_outputs: torch.Tensor) -> None:
        """Pre-compute encoder transformation once per sequence."""
        self._encoder_transformed = self.W_encoder(encoder_outputs)
        
    def clear_cache(self) -> None:
        """Clear cached encoder transformation."""
        self._encoder_transformed = None
        
    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        if decoder_state.dim() == 2:
            decoder_state = decoder_state.unsqueeze(1)
            
        decoder_transformed = self.W_decoder(decoder_state)
        
        # Use pre-computed transformation if available
        if self._encoder_transformed is not None:
            encoder_transformed = self._encoder_transformed
        else:
            encoder_transformed = self.W_encoder(encoder_outputs)
        
        scores = self.v(torch.tanh(decoder_transformed + encoder_transformed))
        scores = scores.squeeze(-1)
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        
        return context, attention_weights
```

## Inference with Attention

```python
def translate_with_attention(
    model: Seq2SeqAttention,
    src: torch.Tensor,
    src_vocab,
    tgt_vocab,
    max_length: int = 50
) -> tuple:
    """
    Translate source sequence and return attention weights for visualization.
    
    Args:
        model: Seq2Seq model with attention
        src: Source tensor (1, src_len)
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        max_length: Maximum generation length
        
    Returns:
        words: List of generated tokens
        attention_matrix: Attention weights (tgt_len, src_len)
    """
    model.eval()
    
    sos_idx = tgt_vocab.token2idx['<SOS>']
    eos_idx = tgt_vocab.token2idx['<EOS>']
    
    with torch.no_grad():
        # Encode source
        encoder_outputs, (hidden, cell) = model.encoder(src)
        hidden, cell = hidden[-1], cell[-1]
        
        # Decode autoregressively
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

## Historical Significance and Evolution

### Timeline of Key Developments

| Year | Contribution | Authors | Impact |
|------|-------------|---------|--------|
| 2014 | Seq2Seq with LSTM | Sutskever et al. | Established encoder-decoder paradigm |
| 2015 | Additive Attention | Bahdanau et al. | Solved bottleneck problem |
| 2015 | Multiplicative Attention | Luong et al. | Efficient alternatives |
| 2017 | Self-Attention / Transformer | Vaswani et al. | Replaced recurrence entirely |
| 2018+ | BERT, GPT, etc. | Various | Pretrained attention-based models |

### Path to Transformers

**Deep Insight**: Attention mechanisms in seq2seq models revealed several key insights that enabled the Transformer architecture:

1. **Attention Can Replace Recurrence**: In seq2seq with attention, most of the "heavy lifting" is done by attention, not the RNN. This suggested recurrence might be unnecessary.

2. **Self-Attention**: If attention can connect encoder positions to decoder positions, why not connect positions within the same sequence? This insight led to self-attention.

3. **Parallelization**: RNNs process sequentially; attention is inherently parallel. Removing recurrence enables massive parallelization.

4. **Positional Information**: Without recurrence, position must be encoded explicitly—leading to positional encodings.

## Summary

Attention mechanisms fundamentally transformed sequence-to-sequence learning by allowing dynamic, content-based access to encoder representations. The key contributions and insights include:

1. **Attention as Soft Alignment**: Attention weights represent soft alignment between decoder and encoder positions, enabling the model to focus on relevant input positions for each output. This provides both better performance and interpretability.

2. **Bahdanau vs Luong**: Bahdanau attention computes alignment before the RNN step using an additive score function; Luong attention computes alignment after the RNN step with multiplicative variants. The choice affects information flow and computational cost.

3. **Scaling and Stability**: The scaling factor in dot-product attention ($\sqrt{d}$) is crucial for training stability with high-dimensional representations, preventing softmax saturation.

4. **Coverage Tracking**: Coverage mechanisms prevent attention from repeatedly focusing on or ignoring certain positions, addressing over/under-translation problems.

5. **Multi-Head Attention**: Multiple attention heads capture different types of relationships in parallel, providing an ensemble effect and richer representations.

6. **Local Attention**: Restricting attention to a predicted window can improve efficiency and sometimes quality for tasks with monotonic alignments.

7. **Foundation for Transformers**: The attention mechanism laid the groundwork for transformer architectures, where self-attention replaces recurrence entirely, enabling massive parallelization and leading to the current generation of large language models.

The evolution from seq2seq with attention to Transformers represents one of the most significant paradigm shifts in deep learning history, with attention mechanisms at its core.
