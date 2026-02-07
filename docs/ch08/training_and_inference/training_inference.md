# Transformer Training and Inference

## Overview

Training and inference follow fundamentally different paradigms in the Transformer architecture. During training, the model receives both source and target sequences and uses **teacher forcing** for efficient parallel computation. During inference, the model generates tokens autoregressively, producing one token at a time and feeding it back as input for the next step.

## Training Pipeline

### Input Pairs

A Transformer is trained on paired sequences $(x, y)$:

- $x$: Source sequence (e.g., English sentence)
- $y$: Target sequence (e.g., French translation)

The encoder receives $x$ directly, while the decoder receives a **right-shifted** version of $y$.

### Teacher Forcing

Teacher forcing feeds the ground-truth target sequence into the decoder during training, rather than the model's own predictions. This enables parallel computation across all target positions simultaneously.

For a target sequence $y = [y_1, y_2, \ldots, y_T]$:

- **Decoder input** (right-shifted): $[\langle\text{start}\rangle, y_1, y_2, \ldots, y_{T-1}]$
- **Ground truth labels**: $[y_1, y_2, \ldots, y_T, \langle\text{end}\rangle]$

The right shift ensures that predicting token $y_t$ depends only on tokens $y_1, \ldots, y_{t-1}$, not on $y_t$ itself.

### Example: Machine Translation

For translating "The cat sat on the mat" → "Le chat était assis sur le tapis":

$$
\begin{aligned}
\text{Encoder input } (x) &: [\text{The}, \text{cat}, \text{sat}, \text{on}, \text{the}, \text{mat}] \\
\text{Decoder input } (y_{\text{shifted}}) &: [\langle\text{start}\rangle, \text{Le}, \text{chat}, \text{était}, \text{assis}, \text{sur}, \text{le}] \\
\text{Target labels} &: [\text{Le}, \text{chat}, \text{était}, \text{assis}, \text{sur}, \text{le}, \text{tapis}, \langle\text{end}\rangle]
\end{aligned}
$$

### Loss Computation

The training loss is the cross-entropy between predicted probability distributions and ground-truth tokens, summed across all positions:

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(y_t \mid y_{<t}, x)
$$

where $P_\theta(y_t \mid y_{<t}, x)$ is the model's predicted probability for the correct token $y_t$ at position $t$, conditioned on previous target tokens $y_{<t}$ and the full source sequence $x$.

At each position, the decoder outputs a probability distribution over the entire vocabulary. The cross-entropy loss penalizes the model when it assigns low probability to the correct next token.

### Embedding Training

Word embeddings are typically trained jointly with the rest of the Transformer. The embedding layer is initialized randomly and updated via backpropagation. Three common approaches exist:

1. **Training from scratch**: Random initialization; embeddings learn task-specific representations during training.
2. **Pretrained initialization**: Initialize with Word2Vec, GloVe, or contextual embeddings, then either fine-tune or freeze them.
3. **Subword embeddings**: Models using Byte-Pair Encoding (BPE) or WordPiece learn embeddings for subword units, providing better vocabulary coverage and handling of rare words.

## PyTorch Training Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim


def create_masks(src, tgt, src_pad_idx=0, tgt_pad_idx=0):
    """
    Create padding and causal masks for Transformer training.
    
    Args:
        src: Source tokens [batch_size, src_len]
        tgt: Target tokens [batch_size, tgt_len]
        src_pad_idx: Padding token index for source
        tgt_pad_idx: Padding token index for target
    
    Returns:
        src_key_padding_mask: [batch_size, src_len] - True where padded
        tgt_key_padding_mask: [batch_size, tgt_len] - True where padded
        tgt_mask: [tgt_len, tgt_len] - Causal mask
        memory_key_padding_mask: Same as src_key_padding_mask
    """
    # Padding masks: True at pad positions
    src_key_padding_mask = (src == src_pad_idx)       # [batch_size, src_len]
    tgt_key_padding_mask = (tgt == tgt_pad_idx)       # [batch_size, tgt_len]
    
    # Causal mask for decoder: prevents attending to future tokens
    tgt_len = tgt.size(1)
    tgt_mask = torch.triu(                            # [tgt_len, tgt_len]
        torch.ones(tgt_len, tgt_len, device=tgt.device),
        diagonal=1
    ).bool()
    
    return src_key_padding_mask, tgt_key_padding_mask, tgt_mask


def train_step(model, optimizer, criterion, src, tgt, src_pad_idx=0, tgt_pad_idx=0):
    """
    Single training step for the Transformer.
    
    Args:
        model: Transformer model
        optimizer: Optimizer instance
        criterion: Loss function (CrossEntropyLoss)
        src: Source tokens [batch_size, src_len]
        tgt: Full target sequence [batch_size, tgt_len]
        src_pad_idx: Source padding index
        tgt_pad_idx: Target padding index
    
    Returns:
        loss: Scalar loss value
    """
    model.train()
    
    # Prepare decoder input (right-shifted target) and labels
    # Decoder sees: [<start>, y1, y2, ..., y_{T-1}]
    # Labels are:   [y1, y2, ..., y_T]
    tgt_input = tgt[:, :-1]     # [batch_size, tgt_len - 1]
    tgt_labels = tgt[:, 1:]     # [batch_size, tgt_len - 1]
    
    # Create masks
    src_pad_mask, tgt_pad_mask, tgt_causal_mask = create_masks(
        src, tgt_input, src_pad_idx, tgt_pad_idx
    )
    
    # Forward pass
    optimizer.zero_grad()
    output = model(                                   # [batch_size, tgt_len-1, vocab_size]
        src, tgt_input,
        src_mask=None,
        tgt_mask=tgt_causal_mask,
        src_key_padding_mask=src_pad_mask,
        tgt_key_padding_mask=tgt_pad_mask,
        memory_key_padding_mask=src_pad_mask
    )
    
    # Compute loss (flatten for cross-entropy)
    vocab_size = output.size(-1)
    loss = criterion(
        output.reshape(-1, vocab_size),               # [batch_size * (tgt_len-1), vocab_size]
        tgt_labels.reshape(-1)                        # [batch_size * (tgt_len-1)]
    )
    
    # Backward pass and parameter update
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch.
    
    Args:
        model: Transformer model
        dataloader: DataLoader yielding (src, tgt) batches
        optimizer: Optimizer instance
        criterion: Loss function
        device: torch.device
    
    Returns:
        average_loss: Mean loss over all batches
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for src, tgt in dataloader:
        src = src.to(device)                          # [batch_size, src_len]
        tgt = tgt.to(device)                          # [batch_size, tgt_len]
        
        loss = train_step(model, optimizer, criterion, src, tgt)
        total_loss += loss
        num_batches += 1
    
    return total_loss / num_batches
```

## Inference Process

### Autoregressive Generation

During inference, only the source sequence $x$ is provided. The decoder generates tokens one at a time, feeding each predicted token back as input for the next step.

The generation loop proceeds as:

$$
\hat{y}_t = \arg\max_{v \in \mathcal{V}} P_\theta(v \mid \hat{y}_1, \ldots, \hat{y}_{t-1}, x)
$$

1. Encode the source sequence $x$ through the encoder (done once)
2. Initialize decoder input with $\langle\text{start}\rangle$ token
3. At each step $t$:
   - Pass all generated tokens $[\langle\text{start}\rangle, \hat{y}_1, \ldots, \hat{y}_{t-1}]$ through the decoder
   - Extract the prediction at the last position
   - Select the next token (via greedy decoding, beam search, or sampling)
   - Append the selected token to the decoder input
4. Stop when $\langle\text{end}\rangle$ is generated or maximum length is reached

### PyTorch Inference Implementation

```python
@torch.no_grad()
def greedy_decode(
    model,
    src: torch.Tensor,
    max_len: int = 100,
    start_token: int = 1,
    end_token: int = 2,
    pad_idx: int = 0
) -> torch.Tensor:
    """
    Autoregressive greedy decoding for the Transformer.
    
    Args:
        model: Trained Transformer model
        src: Source sequence [1, src_len]
        max_len: Maximum generation length
        start_token: Start-of-sequence token ID
        end_token: End-of-sequence token ID
        pad_idx: Padding token ID
    
    Returns:
        generated: Generated token sequence [1, gen_len]
    """
    model.eval()
    device = src.device
    
    # Encode source (computed once)
    src_pad_mask = (src == pad_idx)                    # [1, src_len]
    memory = model.encode(src, src_key_padding_mask=src_pad_mask)
    
    # Initialize decoder input with <start> token
    generated = torch.tensor([[start_token]], device=device)  # [1, 1]
    
    for _ in range(max_len):
        tgt_len = generated.size(1)
        
        # Create causal mask for current sequence length
        tgt_mask = torch.triu(                        # [tgt_len, tgt_len]
            torch.ones(tgt_len, tgt_len, device=device),
            diagonal=1
        ).bool()
        
        # Decode
        output = model.decode(                        # [1, tgt_len, vocab_size]
            generated, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_pad_mask
        )
        
        # Get prediction at last position
        next_token_logits = output[:, -1, :]          # [1, vocab_size]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [1, 1]
        
        # Append to generated sequence
        generated = torch.cat([generated, next_token], dim=1)  # [1, tgt_len + 1]
        
        # Stop if <end> token is generated
        if next_token.item() == end_token:
            break
    
    return generated


@torch.no_grad()
def translate(
    model,
    src_tokens: torch.Tensor,
    idx_to_token: dict,
    max_len: int = 100,
    start_token: int = 1,
    end_token: int = 2
) -> str:
    """
    Translate a source sequence to text.
    
    Args:
        model: Trained Transformer model
        src_tokens: Source token IDs [1, src_len]
        idx_to_token: Mapping from token index to string
        max_len: Maximum generation length
        start_token: Start token ID
        end_token: End token ID
    
    Returns:
        translated_text: Generated translation as string
    """
    generated = greedy_decode(
        model, src_tokens, max_len, start_token, end_token
    )
    
    # Convert token IDs to strings, excluding <start> and <end>
    tokens = generated.squeeze().tolist()
    words = [
        idx_to_token.get(idx, "<unk>")
        for idx in tokens
        if idx not in (start_token, end_token, 0)
    ]
    
    return " ".join(words)
```

## Training vs Inference Comparison

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Encoder input** | Source sequence $x$ | Source sequence $x$ |
| **Decoder input** | Right-shifted ground truth $y_{\text{shifted}}$ | Previously generated tokens $\hat{y}_{<t}$ |
| **Target required** | Yes (for loss computation) | No |
| **Parallelism** | Full (all positions computed simultaneously) | Sequential (one token per step) |
| **Strategy** | Teacher forcing | Autoregressive (greedy, beam, sampling) |
| **Causal mask** | Applied during training to prevent future-peeking | Applied at each generation step |
| **Computation** | Single forward pass for entire sequence | $T$ forward passes for $T$-length output |

## Complete Training Example

```python
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))     # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    """Wrapper around nn.Transformer with embedding and output layers."""
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings and positional encoding
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Core Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
    
    def encode(self, src, src_key_padding_mask=None):
        """Encode source sequence."""
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        return self.transformer.encoder(x, src_key_padding_mask=src_key_padding_mask)
    
    def decode(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        """Decode target sequence given encoder memory."""
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer.decoder(
            x, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.output_proj(x)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """Full forward pass."""
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.output_proj(output)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 512,
    nhead: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1
) -> TransformerSeq2Seq:
    """
    Build a Transformer model with Xavier initialization.
    
    Returns a TransformerSeq2Seq with embedding layers and output projection.
    """
    model = TransformerSeq2Seq(
        src_vocab_size, tgt_vocab_size,
        d_model, nhead,
        num_encoder_layers, num_decoder_layers,
        dim_feedforward, dropout
    )
    
    # Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model


# Training loop example
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 256
    nhead = 8
    num_layers = 3
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    
    # Build model
    model = build_transformer(
        src_vocab_size, tgt_vocab_size,
        d_model, nhead, num_layers, num_layers
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Dummy data (replace with real dataset)
    src_data = torch.randint(1, src_vocab_size, (640, 20))  # 640 samples, length 20
    tgt_data = torch.randint(1, tgt_vocab_size, (640, 22))  # Include <start> and <end>
    
    dataset = TensorDataset(src_data, tgt_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

## Label Smoothing

The original Transformer paper uses label smoothing with $\epsilon = 0.1$, which prevents the model from becoming overconfident by spreading a small amount of probability mass across all vocabulary tokens:

$$
y_{\text{smooth}}(k) = (1 - \epsilon) \cdot \mathbf{1}_{k=y} + \frac{\epsilon}{|\mathcal{V}|}
$$

where $|\mathcal{V}|$ is the vocabulary size and $y$ is the ground-truth token.

```python
class LabelSmoothingLoss(nn.Module):
    """Cross-entropy loss with label smoothing."""
    
    def __init__(self, vocab_size: int, smoothing: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size * seq_len, vocab_size]
            target: [batch_size * seq_len]
        """
        log_probs = torch.log_softmax(logits, dim=-1)       # [N, vocab_size]
        
        # Create smoothed target distribution
        smooth_target = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # Zero out padding positions
        non_pad_mask = target != self.pad_idx                # [N]
        smooth_target[~non_pad_mask] = 0.0
        
        # KL divergence loss
        loss = -(smooth_target * log_probs).sum(dim=-1)      # [N]
        loss = loss[non_pad_mask].mean()
        
        return loss
```

## Summary

The Transformer training and inference processes differ fundamentally in how they handle the decoder:

1. **Training** uses teacher forcing with right-shifted ground truth, enabling full parallelism across positions and computing loss against the true next tokens.
2. **Inference** generates autoregressively, producing one token per step and feeding predictions back as decoder input.
3. **Label smoothing** prevents overconfidence and acts as a regularizer.
4. **Mask construction** combines padding masks (for variable-length sequences) with causal masks (for autoregressive generation).

Understanding this distinction is essential for implementing, debugging, and optimizing Transformer-based systems.

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Szegedy, C., et al. (2016). "Rethinking the Inception Architecture for Computer Vision." CVPR. (Label smoothing)
3. Williams, R. J., & Zipser, D. (1989). "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks." (Teacher forcing)
