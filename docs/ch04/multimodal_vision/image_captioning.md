# Image Captioning

## Learning Objectives

By the end of this section, you will be able to:

1. Understand the encoder-decoder architecture for image captioning
2. Implement attention mechanisms for grounding captions in visual content
3. Apply different decoding strategies (greedy, beam search, sampling)
4. Evaluate caption quality using automated metrics
5. Build a complete image captioning pipeline in PyTorch

## Introduction

Image captioning is the task of generating natural language descriptions for images. This seemingly simple task requires understanding visual content at multiple levels—recognizing objects, understanding their relationships, inferring actions and context—and expressing this understanding in fluent, accurate language.

Image captioning serves as a fundamental benchmark for vision-language understanding and has practical applications in accessibility (describing images for visually impaired users), content indexing, and human-robot interaction.

## Problem Formulation

Given an image $I$, the goal is to generate a caption $C = (c_1, c_2, ..., c_T)$ that accurately describes the image content:

$$C^* = \arg\max_C P(C | I) = \arg\max_C \prod_{t=1}^{T} P(c_t | c_1, ..., c_{t-1}, I)$$

This factorization enables autoregressive generation: predict one word at a time, conditioned on the image and previously generated words.

## Architecture Evolution

### Classic Encoder-Decoder

The foundational approach uses a CNN encoder and RNN decoder:

```
Image → [CNN Encoder] → Visual Features → [LSTM Decoder] → Caption
                              ↓
                        Single vector
```

**Limitation:** The entire image is compressed into a single vector, losing spatial information.

### Attention-Based Captioning

The attention mechanism allows the decoder to focus on relevant image regions:

```
Image → [CNN Encoder] → Spatial Features (H×W×D)
                              ↓
              ┌───────────────┴───────────────┐
              ↓                               ↓
        [Attention] ←──── Query ←──── [LSTM Decoder]
              ↓                               ↑
        Context Vector ───────────────────────┘
```

At each decoding step, attention computes a weighted combination of spatial features based on the decoder's current state.

### Transformer-Based Captioning

Modern approaches use transformers for both encoding and decoding:

```
Image → [ViT/CNN] → Visual Tokens
                        ↓
              [Cross-Attention Decoder] → Caption
                        ↑
              Previous Tokens (masked)
```

## Mathematical Foundations

### Soft Attention Mechanism

Given image features $V = \{v_1, ..., v_L\}$ where $v_i \in \mathbb{R}^D$ and $L$ is the number of spatial locations:

**Attention weights** at time $t$:

$$e_{ti} = f_{att}(v_i, h_{t-1})$$

$$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{j=1}^{L} \exp(e_{tj})}$$

where $f_{att}$ is typically an MLP:

$$f_{att}(v, h) = w^T \tanh(W_v v + W_h h)$$

**Context vector**:

$$\hat{v}_t = \sum_{i=1}^{L} \alpha_{ti} v_i$$

### Decoder Dynamics

The LSTM decoder updates as:

$$h_t = \text{LSTM}([\text{embed}(c_{t-1}); \hat{v}_t], h_{t-1})$$

$$p(c_t | c_{<t}, I) = \text{softmax}(W_o h_t)$$

where $[\cdot; \cdot]$ denotes concatenation.

### Training Objective

**Cross-entropy loss** over the ground-truth caption:

$$\mathcal{L}_{XE} = -\sum_{t=1}^{T} \log P(c_t^* | c_1^*, ..., c_{t-1}^*, I)$$

**Self-critical sequence training (SCST)** for non-differentiable metrics:

$$\mathcal{L}_{RL} = -\mathbb{E}_{c \sim p_\theta}[(r(c) - b) \log p_\theta(c | I)]$$

where $r(c)$ is the reward (e.g., CIDEr score) and $b$ is a baseline.

## PyTorch Implementation

### Visual Encoder

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class VisualEncoder(nn.Module):
    """
    Extract spatial features from images using a CNN backbone.
    
    For simplicity, we assume pre-extracted features. In practice,
    you'd use ResNet, EfficientNet, or ViT.
    """
    
    def __init__(self, 
                 feature_dim: int = 2048,
                 embed_dim: int = 512,
                 num_regions: int = 49):  # 7x7 grid
        super().__init__()
        
        # Project CNN features to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Learnable positional encoding for spatial features
        self.pos_encoding = nn.Parameter(
            torch.randn(1, num_regions, embed_dim) * 0.02
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: CNN features (batch, num_regions, feature_dim)
        
        Returns:
            Projected features with positional encoding (batch, num_regions, embed_dim)
        """
        # Project to embedding space
        x = self.projection(features)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        return x
```

### Attention Module

```python
class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention.
    
    Computes attention weights based on decoder hidden state
    and encoder outputs.
    """
    
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        
        # Project encoder features
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        
        # Project decoder state
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)
        
        # Compute attention energy
        self.attention = nn.Linear(attention_dim, 1)
        
    def forward(self, 
                encoder_out: torch.Tensor,
                decoder_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_out: Visual features (batch, num_regions, encoder_dim)
            decoder_hidden: Decoder hidden state (batch, decoder_dim)
        
        Returns:
            context: Attended visual features (batch, encoder_dim)
            alpha: Attention weights (batch, num_regions)
        """
        # Project encoder outputs: (batch, num_regions, attention_dim)
        encoder_proj = self.encoder_proj(encoder_out)
        
        # Project decoder hidden: (batch, attention_dim)
        decoder_proj = self.decoder_proj(decoder_hidden)
        
        # Add and apply tanh: (batch, num_regions, attention_dim)
        combined = torch.tanh(encoder_proj + decoder_proj.unsqueeze(1))
        
        # Compute attention energies: (batch, num_regions, 1)
        energy = self.attention(combined)
        
        # Softmax over regions: (batch, num_regions)
        alpha = F.softmax(energy.squeeze(-1), dim=1)
        
        # Weighted sum of encoder outputs: (batch, encoder_dim)
        context = (encoder_out * alpha.unsqueeze(-1)).sum(dim=1)
        
        return context, alpha
```

### Decoder with Attention

```python
class AttentionDecoder(nn.Module):
    """
    LSTM decoder with attention mechanism for image captioning.
    
    At each step:
    1. Attend to visual features based on previous hidden state
    2. Concatenate attention context with word embedding
    3. Update LSTM state
    4. Predict next word
    """
    
    def __init__(self,
                 embed_dim: int,
                 decoder_dim: int,
                 attention_dim: int,
                 vocab_size: int,
                 encoder_dim: int = 512,
                 dropout: float = 0.5):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Attention module
        self.attention = AdditiveAttention(encoder_dim, decoder_dim, attention_dim)
        
        # LSTM cell
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        
        # Initialize hidden state from mean visual features
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        # Output projection
        self.fc = nn.Linear(decoder_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Gating mechanism for attention (optional)
        self.gate = nn.Linear(decoder_dim, encoder_dim)
        
    def init_hidden_state(self, encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden state from visual features.
        
        Args:
            encoder_out: (batch, num_regions, encoder_dim)
        
        Returns:
            Initial (h, c) states
        """
        mean_encoder = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder)
        c = self.init_c(mean_encoder)
        return h, c
    
    def forward_step(self,
                    word_embedding: torch.Tensor,
                    encoder_out: torch.Tensor,
                    h: torch.Tensor,
                    c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single decoding step.
        
        Args:
            word_embedding: Embedded input word (batch, embed_dim)
            encoder_out: Visual features (batch, num_regions, encoder_dim)
            h, c: Previous LSTM states
        
        Returns:
            logits: Word logits (batch, vocab_size)
            alpha: Attention weights (batch, num_regions)
            h, c: Updated LSTM states
        """
        # Compute attention
        context, alpha = self.attention(encoder_out, h)
        
        # Optional: gate the attention context
        gate = torch.sigmoid(self.gate(h))
        context = gate * context
        
        # LSTM input: concatenate word embedding and context
        lstm_input = torch.cat([word_embedding, context], dim=1)
        
        # LSTM update
        h, c = self.lstm(lstm_input, (h, c))
        
        # Predict next word
        logits = self.fc(self.dropout(h))
        
        return logits, alpha, h, c
    
    def forward(self,
               encoder_out: torch.Tensor,
               captions: torch.Tensor,
               caption_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass during training with teacher forcing.
        
        Args:
            encoder_out: Visual features (batch, num_regions, encoder_dim)
            captions: Ground truth captions (batch, max_length)
            caption_lengths: Actual lengths of each caption
        
        Returns:
            predictions: Word logits (batch, max_length, vocab_size)
            alphas: Attention weights (batch, max_length, num_regions)
        """
        batch_size = encoder_out.size(0)
        max_length = captions.size(1)
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)
        
        # Embed all words at once
        embeddings = self.embedding(captions)
        
        # Collect outputs
        predictions = []
        alphas = []
        
        # Decode step by step
        for t in range(max_length):
            # Get current word embedding (teacher forcing)
            word_emb = embeddings[:, t, :]
            
            # Decode one step
            logits, alpha, h, c = self.forward_step(word_emb, encoder_out, h, c)
            
            predictions.append(logits)
            alphas.append(alpha)
        
        predictions = torch.stack(predictions, dim=1)
        alphas = torch.stack(alphas, dim=1)
        
        return predictions, alphas
```

### Complete Captioning Model

```python
class ImageCaptioningModel(nn.Module):
    """
    Complete image captioning model with encoder and decoder.
    """
    
    def __init__(self,
                 feature_dim: int = 2048,
                 embed_dim: int = 512,
                 decoder_dim: int = 512,
                 attention_dim: int = 512,
                 vocab_size: int = 10000,
                 num_regions: int = 49):
        super().__init__()
        
        self.encoder = VisualEncoder(feature_dim, embed_dim, num_regions)
        self.decoder = AttentionDecoder(
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim,
            vocab_size=vocab_size,
            encoder_dim=embed_dim
        )
        
        self.vocab_size = vocab_size
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor,
               caption_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        """
        encoder_out = self.encoder(images)
        predictions, alphas = self.decoder(encoder_out, captions, caption_lengths)
        return predictions, alphas
    
    @torch.no_grad()
    def generate(self, 
                images: torch.Tensor,
                max_length: int = 50,
                method: str = 'greedy',
                beam_size: int = 5,
                temperature: float = 1.0,
                start_token: int = 1,
                end_token: int = 2) -> torch.Tensor:
        """
        Generate captions for images.
        
        Args:
            images: Visual features (batch, num_regions, feature_dim)
            max_length: Maximum caption length
            method: 'greedy', 'beam', or 'sample'
            beam_size: Beam width for beam search
            temperature: Sampling temperature
            start_token: Token ID for <START>
            end_token: Token ID for <END>
        
        Returns:
            Generated captions (batch, max_length) or (batch, beam_size, max_length)
        """
        encoder_out = self.encoder(images)
        
        if method == 'greedy':
            return self._greedy_decode(encoder_out, max_length, 
                                       start_token, end_token)
        elif method == 'beam':
            return self._beam_search(encoder_out, max_length, beam_size,
                                    start_token, end_token)
        elif method == 'sample':
            return self._sample_decode(encoder_out, max_length, temperature,
                                      start_token, end_token)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _greedy_decode(self, encoder_out, max_length, start_token, end_token):
        """Greedy decoding: always pick the most likely next word."""
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        
        # Initialize
        h, c = self.decoder.init_hidden_state(encoder_out)
        current_token = torch.full((batch_size,), start_token, 
                                   dtype=torch.long, device=device)
        
        captions = [current_token]
        
        for _ in range(max_length - 1):
            word_emb = self.decoder.embedding(current_token)
            logits, _, h, c = self.decoder.forward_step(word_emb, encoder_out, h, c)
            
            # Greedy selection
            current_token = logits.argmax(dim=-1)
            captions.append(current_token)
        
        return torch.stack(captions, dim=1)
    
    def _sample_decode(self, encoder_out, max_length, temperature, 
                      start_token, end_token):
        """Sampling with temperature for diverse captions."""
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        
        h, c = self.decoder.init_hidden_state(encoder_out)
        current_token = torch.full((batch_size,), start_token,
                                   dtype=torch.long, device=device)
        
        captions = [current_token]
        
        for _ in range(max_length - 1):
            word_emb = self.decoder.embedding(current_token)
            logits, _, h, c = self.decoder.forward_step(word_emb, encoder_out, h, c)
            
            # Temperature-scaled sampling
            probs = F.softmax(logits / temperature, dim=-1)
            current_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            captions.append(current_token)
        
        return torch.stack(captions, dim=1)
```

### Beam Search Implementation

```python
def beam_search(model: ImageCaptioningModel,
               encoder_out: torch.Tensor,
               beam_size: int = 5,
               max_length: int = 50,
               start_token: int = 1,
               end_token: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Beam search decoding for better caption quality.
    
    Maintains top-k hypotheses at each step, exploring multiple
    possible captions simultaneously.
    """
    device = encoder_out.device
    
    # Only handle batch_size=1 for simplicity
    assert encoder_out.size(0) == 1
    
    # Initialize beams
    k = beam_size
    encoder_out = encoder_out.expand(k, -1, -1)  # (k, num_regions, dim)
    
    # Start tokens
    k_prev_words = torch.full((k,), start_token, dtype=torch.long, device=device)
    
    # Store sequences and scores
    seqs = k_prev_words.unsqueeze(1)  # (k, 1)
    scores = torch.zeros(k, device=device)  # (k,)
    
    # Initialize LSTM states
    h, c = model.decoder.init_hidden_state(encoder_out)
    
    complete_seqs = []
    complete_seqs_scores = []
    
    for step in range(max_length):
        word_emb = model.decoder.embedding(k_prev_words)
        logits, _, h, c = model.decoder.forward_step(word_emb, encoder_out, h, c)
        
        # Log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (k, vocab_size)
        
        # Add to cumulative scores
        next_scores = scores.unsqueeze(1) + log_probs  # (k, vocab_size)
        
        # Flatten for top-k selection
        next_scores = next_scores.view(-1)  # (k * vocab_size)
        
        # Select top k
        top_scores, top_indices = next_scores.topk(k, dim=0)
        
        # Convert flat indices to beam and word indices
        prev_beam_indices = top_indices // model.vocab_size
        next_word_indices = top_indices % model.vocab_size
        
        # Update sequences
        seqs = torch.cat([seqs[prev_beam_indices], 
                         next_word_indices.unsqueeze(1)], dim=1)
        
        # Check for completed sequences
        incomplete = []
        for idx, (beam_idx, word_idx) in enumerate(zip(prev_beam_indices, next_word_indices)):
            if word_idx.item() == end_token:
                complete_seqs.append(seqs[idx].clone())
                complete_seqs_scores.append(top_scores[idx].item())
            else:
                incomplete.append(idx)
        
        if len(incomplete) == 0:
            break
        
        # Continue with incomplete sequences
        incomplete = torch.tensor(incomplete, device=device)
        k = len(incomplete)
        
        seqs = seqs[incomplete]
        scores = top_scores[incomplete]
        k_prev_words = next_word_indices[incomplete]
        h = h[prev_beam_indices[incomplete]]
        c = c[prev_beam_indices[incomplete]]
        encoder_out = encoder_out[:k]
    
    # Handle case where no sequence completed
    if len(complete_seqs) == 0:
        complete_seqs = [seqs[0]]
        complete_seqs_scores = [scores[0].item()]
    
    # Return best sequence
    best_idx = complete_seqs_scores.index(max(complete_seqs_scores))
    return complete_seqs[best_idx], complete_seqs_scores[best_idx]
```

## Training Pipeline

```python
def train_captioning_model(model: ImageCaptioningModel,
                          train_loader,
                          val_loader,
                          num_epochs: int = 30,
                          learning_rate: float = 4e-4,
                          device: str = 'cuda') -> dict:
    """
    Train image captioning model with cross-entropy loss.
    """
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for images, captions, lengths in train_loader:
            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths.to(device)
            
            # Forward pass (teacher forcing)
            predictions, alphas = model(images, captions[:, :-1], lengths - 1)
            
            # Compute loss
            # predictions: (batch, max_len-1, vocab_size)
            # targets: (batch, max_len-1)
            targets = captions[:, 1:].contiguous()
            loss = criterion(predictions.view(-1, model.vocab_size), targets.view(-1))
            
            # Optional: doubly stochastic attention regularization
            # Encourages attention to sum to 1 over time for each region
            alpha_reg = ((1 - alphas.sum(dim=1)) ** 2).mean()
            loss = loss + 0.01 * alpha_reg
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, captions, lengths in val_loader:
                images = images.to(device)
                captions = captions.to(device)
                
                predictions, _ = model(images, captions[:, :-1], lengths - 1)
                targets = captions[:, 1:].contiguous()
                loss = criterion(predictions.view(-1, model.vocab_size), targets.view(-1))
                val_loss += loss.item()
        
        scheduler.step()
        
        # Logging
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return history
```

## Evaluation Metrics

### BLEU Score

Measures n-gram precision between generated and reference captions:

$$\text{BLEU-N} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

where $p_n$ is the modified n-gram precision and $BP$ is the brevity penalty.

### CIDEr Score

Consensus-based metric using TF-IDF weighting:

$$\text{CIDEr}_n(c, S) = \frac{1}{M} \sum_{j=1}^{M} \frac{g^n(c) \cdot g^n(s_j)}{\|g^n(c)\| \|g^n(s_j)\|}$$

### METEOR

Combines precision, recall, and semantic similarity.

### Evaluation Code

```python
from collections import Counter
import math

def compute_bleu(candidate: list, references: list, max_n: int = 4) -> dict:
    """
    Compute BLEU scores for a single candidate against references.
    """
    # Count n-grams in candidate
    candidate_counts = {}
    for n in range(1, max_n + 1):
        ngrams = [tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1)]
        candidate_counts[n] = Counter(ngrams)
    
    # Count n-grams in references (max count for each n-gram)
    reference_counts = {n: Counter() for n in range(1, max_n + 1)}
    ref_lengths = []
    
    for ref in references:
        ref_lengths.append(len(ref))
        for n in range(1, max_n + 1):
            ngrams = [tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)]
            ref_count = Counter(ngrams)
            for ngram, count in ref_count.items():
                reference_counts[n][ngram] = max(reference_counts[n][ngram], count)
    
    # Compute modified precision for each n
    precisions = []
    for n in range(1, max_n + 1):
        clipped_count = 0
        total_count = 0
        
        for ngram, count in candidate_counts[n].items():
            clipped_count += min(count, reference_counts[n].get(ngram, 0))
            total_count += count
        
        precision = clipped_count / max(total_count, 1)
        precisions.append(precision)
    
    # Brevity penalty
    c = len(candidate)
    r = min(ref_lengths, key=lambda x: abs(x - c))
    bp = 1 if c > r else math.exp(1 - r / max(c, 1))
    
    # Compute BLEU scores
    bleu_scores = {}
    for n in range(1, max_n + 1):
        if all(p > 0 for p in precisions[:n]):
            bleu = bp * math.exp(sum(math.log(p) for p in precisions[:n]) / n)
        else:
            bleu = 0
        bleu_scores[f'bleu_{n}'] = bleu
    
    return bleu_scores
```

## Advanced Topics

### Transformer-Based Captioning

Modern systems use transformer decoders:

```python
class TransformerCaptioner(nn.Module):
    """
    Transformer-based image captioning model.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 max_length: int = 50):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_length, embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.max_length = max_length
    
    def generate_square_mask(self, size: int, device) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.bool()
    
    def forward(self, visual_features: torch.Tensor, 
               captions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (batch, num_regions, embed_dim)
            captions: (batch, seq_len)
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = captions.shape
        device = captions.device
        
        # Embed captions
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        caption_emb = self.embedding(captions) + self.pos_embedding(positions)
        
        # Create causal mask
        causal_mask = self.generate_square_mask(seq_len, device)
        
        # Decode with cross-attention to visual features
        decoder_out = self.decoder(
            caption_emb, 
            visual_features,
            tgt_mask=causal_mask
        )
        
        logits = self.fc(decoder_out)
        return logits
```

## Key Takeaways

1. **Attention is crucial**: Allows focusing on relevant image regions for each word
2. **Teacher forcing**: Efficient training but creates exposure bias
3. **Beam search**: Better quality than greedy but more expensive
4. **CIDEr optimization**: Self-critical training improves metric scores
5. **Transformers**: Modern architectures replace LSTM with cross-attention

## References

1. Xu, K., et al. "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention." ICML 2015.
2. Anderson, P., et al. "Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering." CVPR 2018.
3. Rennie, S., et al. "Self-Critical Sequence Training for Image Captioning." CVPR 2017.
4. Li, J., et al. "BLIP: Bootstrapping Language-Image Pre-training." ICML 2022.
