# Text Generation with Neural Networks

## Introduction

Text generation—producing coherent sequences of text—demonstrates the power of neural networks as language models. From machine translation to creative writing, character-level to word-level generation, this comprehensive guide covers the principles, architectures, and techniques for building sophisticated text generators. We explore both autoregressive language models and sequence-to-sequence architectures, examining the mathematical foundations and practical implementation details that determine generation quality.

---

## Mathematical Foundation

### Autoregressive Language Modeling

A language model estimates the probability distribution over sequences. The autoregressive factorization decomposes joint probability into a chain of conditionals:

$$P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^{T} P(w_t | w_1, \ldots, w_{t-1})$$

**Key Insight**: This factorization is exact—no independence assumptions are made. The challenge lies entirely in how well we can model each conditional $P(w_t | w_{<t})$.

For RNN-based models, this conditional probability is parameterized as:

$$P(w_t | w_{1:t-1}) = \text{softmax}(W_o h_t + b_o)$$

where $h_t$ is the hidden state encoding all previous context through the recurrence:

$$h_t = f(h_{t-1}, w_{t-1}; \theta)$$

### Sequence-to-Sequence Formulation

For conditional generation tasks (translation, summarization), we model:

$$P(y_1, y_2, \ldots, y_T | \mathbf{x}) = \prod_{t=1}^{T} P(y_t | y_{<t}, \mathbf{x})$$

The encoder compresses source information into representations that the decoder conditions on throughout generation. The choice of decoding strategy significantly impacts output quality, diversity, and coherence.

### Information-Theoretic Perspective

**Cross-Entropy Loss**: Training minimizes the cross-entropy between model distribution and data distribution:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log P_\theta(w_i | w_{<i})$$

**Perplexity**: The exponential of average cross-entropy provides an interpretable measure:

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i | w_{<i})\right)$$

**Interpretation**: Perplexity represents the effective vocabulary size the model is "uncertain" among at each step. A perplexity of 50 means the model is, on average, equally uncertain between 50 words.

---

## Architecture Design

### Character-Level RNN

Character-level models operate on the finest granularity, learning sub-word patterns, morphology, and can generate novel words.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class CharRNN(nn.Module):
    """
    Character-level RNN for text generation.
    
    Advantages:
    - No OOV (out-of-vocabulary) problem
    - Learns morphological patterns
    - Smaller vocabulary (typically < 256)
    - Can generate novel words
    
    Disadvantages:
    - Longer sequences for same text
    - Harder to capture long-range dependencies
    - Higher computational cost per character of output
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional hidden state continuation.
        
        Args:
            x: Input tokens (batch_size, seq_len)
            hidden: Optional (h, c) tuple from previous call
            
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            hidden: Updated (h, c) tuple
        """
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden
    
    def init_hidden(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state to zeros."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)


class CharDataset(torch.utils.data.Dataset):
    """
    Dataset for character-level language modeling.
    
    Uses sliding window approach for efficient training on long texts.
    """
    
    def __init__(self, text: str, seq_length: int = 100):
        self.seq_length = seq_length
        
        # Build vocabulary from unique characters
        self.chars = sorted(list(set(text)))
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # Encode entire text
        self.encoded = [self.char2idx[c] for c in text]
    
    def __len__(self) -> int:
        return len(self.encoded) - self.seq_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (input, target) pair with target shifted by 1."""
        x = torch.tensor(self.encoded[idx:idx + self.seq_length])
        y = torch.tensor(self.encoded[idx + 1:idx + self.seq_length + 1])
        return x, y
```

### Sequence-to-Sequence Architecture

For conditional generation, encoder-decoder architectures separate understanding from generation.

```python
class Seq2SeqGenerator:
    """
    Base interface for sequence-to-sequence generation.
    
    Encapsulates the encoder-decoder pattern with attention support.
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_length: int = 100,
        sos_idx: int = 1,
        eos_idx: int = 2,
        pad_idx: int = 0,
        device: torch.device = None
    ):
        self.model = model
        self.max_length = max_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.device = device or torch.device('cpu')
    
    def encode(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Encode source sequence."""
        encoder_outputs, hidden, cell = self.model.encoder(src, src_lengths)
        
        mask = None
        if hasattr(self.model, 'create_mask'):
            mask = self.model.create_mask(src)
        
        return encoder_outputs, hidden, cell, mask
    
    def decode_step(
        self,
        current_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: Optional[torch.Tensor],
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Single decoder step."""
        if hasattr(self.model.decoder, 'attention'):
            output, hidden, cell, attn_weights = self.model.decoder(
                current_token, hidden, encoder_outputs, cell, mask
            )
        else:
            output, hidden, cell = self.model.decoder(
                current_token, hidden, cell
            )
            attn_weights = None
        
        return output, hidden, cell, attn_weights
```

---

## Decoding Strategies

### Deterministic Decoding

#### Greedy Decoding

The simplest approach selects the most probable token at each step:

$$\hat{y}_t = \arg\max_{y} P(y | y_{<t}, \mathbf{x})$$

**Analysis**: Greedy decoding is locally optimal but globally suboptimal. It can miss better sequences where a slightly worse early choice leads to much better subsequent choices.

```python
class GreedyDecoder:
    """
    Greedy decoding for text generation.
    
    Characteristics:
    - Deterministic: same input always produces same output
    - Fast: O(T × V) per sequence where T=length, V=vocab
    - May produce repetitive or generic outputs
    - Suitable for tasks requiring consistency (translation)
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_length: int = 100,
        sos_idx: int = 1,
        eos_idx: int = 2,
        pad_idx: int = 0,
        device: torch.device = None
    ):
        self.model = model
        self.max_length = max_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.device = device or torch.device('cpu')
        
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Generate sequence using greedy decoding.
        
        Args:
            src: Source sequence (batch_size, src_len)
            src_lengths: Actual source lengths for packing
            
        Returns:
            generated: Generated tokens (batch_size, gen_len)
            log_probs: Log probabilities per step
        """
        self.model.eval()
        batch_size = src.size(0)
        
        # Encode source
        encoder_outputs, hidden, cell = self.model.encoder(src, src_lengths)
        
        # Create attention mask if supported
        mask = None
        if hasattr(self.model, 'create_mask'):
            mask = self.model.create_mask(src)
        
        # Initialize with SOS token
        current_token = torch.full(
            (batch_size, 1), self.sos_idx, 
            dtype=torch.long, device=self.device
        )
        
        generated = [current_token]
        log_probs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for _ in range(self.max_length):
            # Decoder step
            if hasattr(self.model.decoder, 'attention'):
                output, hidden, cell, _ = self.model.decoder(
                    current_token, hidden, encoder_outputs, cell, mask
                )
            else:
                output, hidden, cell = self.model.decoder(
                    current_token, hidden, cell
                )
            
            # Compute log probabilities
            log_prob = F.log_softmax(output, dim=-1)
            
            # Select highest probability token
            best_log_prob, best_token = log_prob.max(dim=-1)
            current_token = best_token.unsqueeze(-1)
            
            generated.append(current_token)
            log_probs.append(best_log_prob)
            
            # Track finished sequences
            finished |= (current_token.squeeze(-1) == self.eos_idx)
            
            if finished.all():
                break
        
        generated = torch.cat(generated, dim=1)
        return generated, log_probs
```

#### Beam Search

Maintains multiple hypotheses for better global optimization. See dedicated beam search documentation for details.

**Key Insight**: Beam search is an approximation to the intractable exact search. With beam width $B$, complexity is $O(T × B × V × \log B)$.

---

### Stochastic Decoding

Stochastic methods introduce randomness to increase output diversity, essential for creative applications.

#### Temperature Sampling

Temperature scaling modifies the sharpness of the probability distribution:

$$P'(w_i) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

**Mathematical Insight**: Temperature is equivalent to raising probabilities to a power:
- $\tau \to 0$: Approaches argmax (greedy)
- $\tau = 1$: Original distribution
- $\tau \to \infty$: Approaches uniform distribution

The entropy of the distribution scales approximately linearly with temperature for moderate $\tau$.

```python
def temperature_sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample with temperature scaling.
    
    Args:
        logits: Unnormalized log probabilities (..., vocab_size)
        temperature: Sampling temperature
            - < 1.0: More confident/deterministic
            - = 1.0: Original distribution  
            - > 1.0: More random/diverse
            
    Returns:
        Sampled token indices
    """
    if temperature == 0:
        return logits.argmax(dim=-1)
    
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class TemperatureSampler:
    """
    Temperature-scaled sampling for diverse generation.
    
    Suitable for creative writing where diversity is valued.
    """
    
    def __init__(
        self,
        model: nn.Module,
        temperature: float = 1.0,
        max_length: int = 100,
        sos_idx: int = 1,
        eos_idx: int = 2,
        device: torch.device = None
    ):
        self.model = model
        self.temperature = temperature
        self.max_length = max_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device or torch.device('cpu')
        
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate with temperature sampling."""
        self.model.eval()
        batch_size = src.size(0)
        
        encoder_outputs, hidden, cell = self.model.encoder(src, src_lengths)
        
        mask = None
        if hasattr(self.model, 'create_mask'):
            mask = self.model.create_mask(src)
        
        current_token = torch.full(
            (batch_size, 1), self.sos_idx,
            dtype=torch.long, device=self.device
        )
        
        generated = [current_token]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for _ in range(self.max_length):
            if hasattr(self.model.decoder, 'attention'):
                output, hidden, cell, _ = self.model.decoder(
                    current_token, hidden, encoder_outputs, cell, mask
                )
            else:
                output, hidden, cell = self.model.decoder(
                    current_token, hidden, cell
                )
            
            # Apply temperature scaling
            scaled_logits = output / self.temperature
            probs = F.softmax(scaled_logits, dim=-1)
            
            # Sample from distribution
            current_token = torch.multinomial(probs.squeeze(1), 1)
            
            generated.append(current_token)
            finished |= (current_token.squeeze(-1) == self.eos_idx)
            
            if finished.all():
                break
        
        return torch.cat(generated, dim=1)
```

#### Top-K Sampling

Restricts sampling to the $k$ most probable tokens, preventing sampling from the long tail of unlikely tokens.

**Analysis**: Top-K is distribution-agnostic—it always keeps exactly $k$ tokens regardless of whether probability mass is concentrated or spread out.

```python
def top_k_sample(
    logits: torch.Tensor,
    k: int = 50,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Sample from top-k most probable tokens.
    
    Args:
        logits: Unnormalized log probabilities
        k: Number of top tokens to consider
        temperature: Sampling temperature applied after filtering
        
    Returns:
        Sampled token indices
    """
    scaled_logits = logits / temperature
    top_k_logits, top_k_indices = scaled_logits.topk(k, dim=-1)
    probs = torch.softmax(top_k_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    return top_k_indices.gather(-1, sampled_idx).squeeze(-1)


class TopKSampler:
    """
    Top-k sampling restricts vocabulary to k most likely tokens.
    
    Advantages:
    - Prevents sampling unlikely/incoherent tokens
    - Maintains diversity within reasonable candidates
    - Simple hyperparameter interpretation
    
    Disadvantages:
    - Fixed k ignores distribution shape
    - May include unlikely tokens when distribution is peaked
    - May exclude reasonable tokens when distribution is flat
    """
    
    def __init__(
        self,
        model: nn.Module,
        k: int = 50,
        temperature: float = 1.0,
        max_length: int = 100,
        sos_idx: int = 1,
        eos_idx: int = 2,
        device: torch.device = None
    ):
        self.model = model
        self.k = k
        self.temperature = temperature
        self.max_length = max_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device or torch.device('cpu')
    
    def top_k_filtering(
        self, 
        logits: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """
        Filter logits to keep only top k values.
        
        All other positions are set to -inf so they have zero
        probability after softmax.
        """
        top_k_values, _ = logits.topk(k, dim=-1)
        threshold = top_k_values[:, -1].unsqueeze(-1)
        
        filtered = logits.clone()
        filtered[logits < threshold] = float('-inf')
        
        return filtered
    
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate with top-k sampling."""
        self.model.eval()
        batch_size = src.size(0)
        
        encoder_outputs, hidden, cell = self.model.encoder(src, src_lengths)
        
        mask = None
        if hasattr(self.model, 'create_mask'):
            mask = self.model.create_mask(src)
        
        current_token = torch.full(
            (batch_size, 1), self.sos_idx,
            dtype=torch.long, device=self.device
        )
        
        generated = [current_token]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for _ in range(self.max_length):
            if hasattr(self.model.decoder, 'attention'):
                output, hidden, cell, _ = self.model.decoder(
                    current_token, hidden, encoder_outputs, cell, mask
                )
            else:
                output, hidden, cell = self.model.decoder(
                    current_token, hidden, cell
                )
            
            # Apply temperature and top-k filtering
            scaled_logits = output.squeeze(1) / self.temperature
            filtered_logits = self.top_k_filtering(scaled_logits, self.k)
            
            probs = F.softmax(filtered_logits, dim=-1)
            current_token = torch.multinomial(probs, 1)
            
            generated.append(current_token)
            finished |= (current_token.squeeze(-1) == self.eos_idx)
            
            if finished.all():
                break
        
        return torch.cat(generated, dim=1)
```

#### Nucleus (Top-P) Sampling

Samples from the smallest set whose cumulative probability exceeds $p$—an adaptive approach that adjusts to distribution shape.

**Key Insight**: Nucleus sampling is distribution-aware. When the model is confident (peaked distribution), few tokens are selected. When uncertain (flat distribution), more tokens are included. This naturally adapts vocabulary size to context.

```python
def top_p_sample(
    logits: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Nucleus sampling - sample from top-p cumulative probability mass.
    
    Dynamically selects the smallest set of tokens whose cumulative
    probability exceeds threshold p.
    
    Args:
        logits: Unnormalized log probabilities
        p: Cumulative probability threshold (typically 0.9-0.95)
        temperature: Sampling temperature
        
    Returns:
        Sampled token indices
    """
    scaled_logits = logits / temperature
    sorted_logits, sorted_indices = scaled_logits.sort(descending=True, dim=-1)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    
    # Find cutoff: first position where cumulative prob > p
    # Shift mask to include the token that crossed threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    probs = torch.softmax(sorted_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    
    return sorted_indices.gather(-1, sampled_idx).squeeze(-1)


class NucleusSampler:
    """
    Nucleus (Top-p) sampling for adaptive vocabulary restriction.
    
    The "Goldilocks" method: not too many tokens (like pure sampling),
    not too few (like top-k with small k), but just right based on
    how confident the model is.
    """
    
    def __init__(
        self,
        model: nn.Module,
        p: float = 0.9,
        temperature: float = 1.0,
        max_length: int = 100,
        sos_idx: int = 1,
        eos_idx: int = 2,
        device: torch.device = None
    ):
        self.model = model
        self.p = p
        self.temperature = temperature
        self.max_length = max_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device or torch.device('cpu')
    
    def nucleus_filtering(
        self,
        logits: torch.Tensor,
        p: float
    ) -> torch.Tensor:
        """Filter logits to keep smallest set with cumulative prob >= p."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Mask tokens beyond cumulative threshold
        sorted_mask = cumulative_probs > p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False
        
        sorted_logits[sorted_mask] = float('-inf')
        
        # Restore original order
        filtered_logits = torch.zeros_like(logits)
        filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        
        return filtered_logits
    
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate with nucleus sampling."""
        self.model.eval()
        batch_size = src.size(0)
        
        encoder_outputs, hidden, cell = self.model.encoder(src, src_lengths)
        
        mask = None
        if hasattr(self.model, 'create_mask'):
            mask = self.model.create_mask(src)
        
        current_token = torch.full(
            (batch_size, 1), self.sos_idx,
            dtype=torch.long, device=self.device
        )
        
        generated = [current_token]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for _ in range(self.max_length):
            if hasattr(self.model.decoder, 'attention'):
                output, hidden, cell, _ = self.model.decoder(
                    current_token, hidden, encoder_outputs, cell, mask
                )
            else:
                output, hidden, cell = self.model.decoder(
                    current_token, hidden, cell
                )
            
            scaled_logits = output.squeeze(1) / self.temperature
            filtered_logits = self.nucleus_filtering(scaled_logits, self.p)
            
            probs = F.softmax(filtered_logits, dim=-1)
            current_token = torch.multinomial(probs, 1)
            
            generated.append(current_token)
            finished |= (current_token.squeeze(-1) == self.eos_idx)
            
            if finished.all():
                break
        
        return torch.cat(generated, dim=1)
```

#### Combined Strategies

```python
class CombinedSampler:
    """
    Combined top-k and nucleus sampling with temperature.
    
    Applies both restrictions for maximum control:
    1. Temperature adjusts distribution sharpness
    2. Top-k provides hard upper bound on candidates
    3. Nucleus adapts within that bound
    
    This combination is often the default in production systems.
    """
    
    def __init__(
        self,
        model: nn.Module,
        k: int = 50,
        p: float = 0.9,
        temperature: float = 1.0,
        max_length: int = 100,
        sos_idx: int = 1,
        eos_idx: int = 2,
        device: torch.device = None
    ):
        self.model = model
        self.k = k
        self.p = p
        self.temperature = temperature
        self.max_length = max_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device or torch.device('cpu')
        
    def combined_filtering(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply both top-k and nucleus filtering."""
        # First apply top-k (hard limit)
        if self.k > 0:
            top_k_values, _ = logits.topk(min(self.k, logits.size(-1)), dim=-1)
            threshold_k = top_k_values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < threshold_k,
                torch.full_like(logits, float('-inf')),
                logits
            )
        
        # Then apply nucleus (adaptive within top-k)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_mask = cumulative_probs > self.p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False
        
        sorted_logits[sorted_mask] = float('-inf')
        
        filtered_logits = torch.zeros_like(logits)
        filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        
        return filtered_logits
```

---

## Repetition Control

### The Repetition Problem

Neural text generators often fall into repetitive loops, especially with greedy or low-temperature sampling. This manifests as:
- **Token repetition**: "the the the the..."
- **Phrase repetition**: "I think that I think that I think that..."
- **Thematic repetition**: Cycling through the same ideas

### N-gram Blocking

Prevent exact repetition of n-grams by blocking tokens that would complete a repeated sequence.

```python
def block_repeated_ngrams(
    generated_ids: List[int],
    logits: torch.Tensor,
    n: int = 3
) -> torch.Tensor:
    """
    Block tokens that would create repeated n-grams.
    
    This is a hard constraint—blocked tokens have zero probability.
    
    Args:
        generated_ids: Previously generated token IDs
        logits: Current step logits (vocab_size,)
        n: N-gram size to check (3 = no repeated trigrams)
        
    Returns:
        Modified logits with blocked tokens set to -inf
    """
    if len(generated_ids) < n - 1:
        return logits
    
    # Get the current (n-1)-gram prefix
    prefix = tuple(generated_ids[-(n-1):])
    
    # Find all tokens that would complete a previously seen n-gram
    blocked_tokens = set()
    for i in range(len(generated_ids) - n + 1):
        if tuple(generated_ids[i:i+n-1]) == prefix:
            blocked_tokens.add(generated_ids[i + n - 1])
    
    logits = logits.clone()
    for token in blocked_tokens:
        logits[token] = float('-inf')
    
    return logits
```

### Repetition Penalty

Apply a soft penalty to previously generated tokens—allows repetition but discourages it.

```python
def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: List[int],
    penalty: float = 1.2
) -> torch.Tensor:
    """
    Apply multiplicative repetition penalty.
    
    For each token that has appeared before:
    - If logit > 0: divide by penalty (reduce probability)
    - If logit < 0: multiply by penalty (make more negative)
    
    Args:
        logits: Current step logits (vocab_size,)
        generated_ids: Previously generated token IDs
        penalty: Multiplicative penalty (> 1 reduces probability)
        
    Returns:
        Modified logits
    """
    logits = logits.clone()
    
    for token_id in set(generated_ids):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    
    return logits


def apply_presence_penalty(
    logits: torch.Tensor,
    generated_ids: List[int],
    penalty: float = 0.5
) -> torch.Tensor:
    """
    Apply additive presence penalty (same for all occurrences).
    
    Unlike repetition penalty, this applies the same penalty regardless
    of how many times a token appeared—only whether it appeared at all.
    """
    logits = logits.clone()
    unique_tokens = set(generated_ids)
    
    for token_id in unique_tokens:
        logits[token_id] -= penalty
    
    return logits


def apply_frequency_penalty(
    logits: torch.Tensor,
    generated_ids: List[int],
    penalty: float = 0.5
) -> torch.Tensor:
    """
    Apply additive penalty proportional to token count.
    
    More appearances = larger penalty. This strongly discourages
    overused tokens while allowing occasional repetition.
    """
    from collections import Counter
    
    logits = logits.clone()
    token_counts = Counter(generated_ids)
    
    for token_id, count in token_counts.items():
        logits[token_id] -= penalty * count
    
    return logits
```

**Comparison of Penalties**:

| Penalty Type | Formula | Use Case |
|-------------|---------|----------|
| Repetition (multiplicative) | $z_i \gets z_i / \alpha$ | General anti-repetition |
| Presence (additive constant) | $z_i \gets z_i - \beta$ | Encourage diversity |
| Frequency (additive scaled) | $z_i \gets z_i - \gamma \cdot c_i$ | Long-form generation |

---

## Length Control

### Minimum Length Enforcement

Prevent premature termination by blocking EOS token.

```python
def enforce_minimum_length(
    logits: torch.Tensor,
    current_length: int,
    min_length: int,
    eos_idx: int
) -> torch.Tensor:
    """
    Prevent EOS token before minimum length reached.
    
    This is crucial for summarization where very short outputs
    are useless, even if they're technically valid.
    """
    if current_length < min_length:
        logits = logits.clone()
        logits[eos_idx] = float('-inf')
    return logits
```

### Length Bias

Soft encouragement toward target length.

```python
def apply_length_bias(
    logits: torch.Tensor,
    current_length: int,
    target_length: int,
    eos_idx: int,
    alpha: float = 0.1
) -> torch.Tensor:
    """
    Bias toward or away from EOS based on target length.
    
    Creates smooth pressure toward desired length without
    hard cutoffs.
    
    Args:
        logits: Current step logits
        current_length: Current sequence length
        target_length: Desired sequence length
        eos_idx: EOS token index
        alpha: Bias strength (higher = stronger length control)
    """
    logits = logits.clone()
    
    # Positive bias encourages EOS, negative discourages
    length_ratio = current_length / target_length
    bias = alpha * (length_ratio - 1.0)
    
    logits[eos_idx] += bias
    
    return logits
```

---

## Complete Generation Pipeline

### Character-Level Generation

```python
def generate_text(
    model: CharRNN,
    start_string: str,
    char2idx: dict,
    idx2char: dict,
    length: int = 500,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: str = 'cpu'
) -> str:
    """
    Generate text from a trained character-level model.
    
    Args:
        model: Trained CharRNN model
        start_string: Prompt text to condition generation
        char2idx: Character to index mapping
        idx2char: Index to character mapping
        length: Number of characters to generate
        temperature: Sampling temperature
        top_k: If set, use top-k sampling
        top_p: If set, use nucleus sampling (takes precedence)
        device: Device for computation
        
    Returns:
        Generated text string including prompt
    """
    model.eval()
    
    # Encode prompt
    chars = [char2idx.get(c, 0) for c in start_string]
    input_seq = torch.tensor([chars], device=device)
    hidden = model.init_hidden(1, device)
    
    # Process prompt through model to get context
    with torch.no_grad():
        for i in range(len(chars) - 1):
            _, hidden = model(input_seq[:, i:i+1], hidden)
    
    generated = list(start_string)
    current_char = input_seq[:, -1:]
    
    # Generate new characters
    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(current_char, hidden)
            logits = logits[:, -1, :]  # Last timestep
            
            # Apply sampling strategy
            if top_p is not None:
                next_char = top_p_sample(logits, p=top_p, temperature=temperature)
            elif top_k is not None:
                next_char = top_k_sample(logits, k=top_k, temperature=temperature)
            else:
                next_char = temperature_sample(logits, temperature=temperature)
            
            generated.append(idx2char[next_char.item()])
            current_char = next_char.unsqueeze(0)
    
    return ''.join(generated)
```

### Unified Text Generator

```python
class TextGenerator:
    """
    Complete text generation pipeline with configurable strategies.
    
    Unifies all decoding strategies, sampling parameters, and control
    mechanisms into a production-ready interface.
    """
    
    def __init__(
        self,
        model: nn.Module,
        vocab,  # Vocabulary object with idx2token mapping
        device: torch.device = None,
        # Decoding strategy
        strategy: str = 'nucleus',  # 'greedy', 'beam', 'top_k', 'nucleus'
        beam_width: int = 5,
        # Sampling parameters
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        # Length control
        max_length: int = 100,
        min_length: int = 1,
        length_penalty: float = 1.0,
        # Repetition control
        repetition_penalty: float = 1.0,
        no_repeat_ngram: int = 0,
        # Special tokens
        sos_idx: int = 1,
        eos_idx: int = 2,
        pad_idx: int = 0
    ):
        self.model = model
        self.vocab = vocab
        self.device = device or torch.device('cpu')
        
        self.strategy = strategy
        self.beam_width = beam_width
        
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        self.max_length = max_length
        self.min_length = min_length
        self.length_penalty = length_penalty
        
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram = no_repeat_ngram
        
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        
    def process_logits(
        self,
        logits: torch.Tensor,
        generated_ids: List[int],
        current_length: int
    ) -> torch.Tensor:
        """Apply all logit modifications in sequence."""
        # 1. Temperature scaling
        logits = logits / self.temperature
        
        # 2. Repetition penalty
        if self.repetition_penalty != 1.0:
            logits = apply_repetition_penalty(
                logits, generated_ids, self.repetition_penalty
            )
        
        # 3. N-gram blocking
        if self.no_repeat_ngram > 0:
            logits = block_repeated_ngrams(
                generated_ids, logits, self.no_repeat_ngram
            )
        
        # 4. Minimum length enforcement
        if current_length < self.min_length:
            logits[self.eos_idx] = float('-inf')
        
        # 5. Sampling restrictions
        if self.strategy == 'top_k':
            top_k_values, _ = logits.topk(self.top_k)
            threshold = top_k_values[-1]
            logits[logits < threshold] = float('-inf')
            
        elif self.strategy == 'nucleus':
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            mask = cumulative_probs > self.top_p
            mask[1:] = mask[:-1].clone()
            mask[0] = False
            
            sorted_logits[mask] = float('-inf')
            logits = torch.zeros_like(logits).scatter_(
                dim=-1, index=sorted_indices, src=sorted_logits
            )
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text sequences.
        
        Args:
            src: Source sequence (batch_size, src_len)
            src_lengths: Actual source lengths
            num_return_sequences: Number of sequences to return
            
        Returns:
            List of generated text strings
        """
        self.model.eval()
        
        # Encode source
        encoder_outputs, hidden, cell = self.model.encoder(src, src_lengths)
        
        mask = None
        if hasattr(self.model, 'create_mask'):
            mask = self.model.create_mask(src)
        
        results = []
        
        for _ in range(num_return_sequences):
            generated_ids = [self.sos_idx]
            current_hidden = hidden.clone()
            current_cell = cell.clone() if cell is not None else None
            
            for step in range(self.max_length):
                current_token = torch.tensor(
                    [[generated_ids[-1]]], device=self.device
                )
                
                if hasattr(self.model.decoder, 'attention'):
                    output, current_hidden, current_cell, _ = self.model.decoder(
                        current_token, current_hidden, encoder_outputs,
                        current_cell, mask
                    )
                else:
                    output, current_hidden, current_cell = self.model.decoder(
                        current_token, current_hidden, current_cell
                    )
                
                logits = output.squeeze(0).squeeze(0)
                logits = self.process_logits(logits, generated_ids, step + 1)
                
                # Sample or select
                if self.strategy == 'greedy':
                    next_token = logits.argmax().item()
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                
                generated_ids.append(next_token)
                
                if next_token == self.eos_idx:
                    break
            
            # Decode to text
            text = self.decode_tokens(generated_ids)
            results.append(text)
        
        return results
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Convert token IDs to text string."""
        tokens = []
        for idx in token_ids:
            if idx == self.sos_idx:
                continue
            if idx == self.eos_idx:
                break
            if idx == self.pad_idx:
                continue
            tokens.append(self.vocab.idx2token.get(idx, '<unk>'))
        return ' '.join(tokens)
```

---

## Evaluation Metrics

### Perplexity

```python
def evaluate_perplexity(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Compute perplexity on validation/test set.
    
    Perplexity is the exponential of average cross-entropy loss.
    Lower is better—represents effective vocabulary uncertainty.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    hidden = None
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, hidden = model(x, hidden)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            # Detach hidden state to prevent gradient accumulation
            hidden = tuple(h.detach() for h in hidden)
    
    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()
```

### Diversity Metrics

```python
def compute_distinct_ngrams(
    texts: List[str],
    n: int = 2
) -> float:
    """
    Compute distinct n-gram ratio (diversity metric).
    
    Distinct-n = (unique n-grams) / (total n-grams)
    
    Higher values indicate more diverse generation.
    Typical values: Distinct-1 ≈ 0.1-0.5, Distinct-2 ≈ 0.3-0.8
    """
    all_ngrams = []
    total_ngrams = 0
    
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            all_ngrams.append(ngram)
            total_ngrams += 1
    
    if total_ngrams == 0:
        return 0.0
    
    unique_ngrams = len(set(all_ngrams))
    return unique_ngrams / total_ngrams


def compute_self_bleu(
    texts: List[str],
    n: int = 4
) -> float:
    """
    Compute Self-BLEU (diversity metric).
    
    Measures similarity between generated samples.
    Lower values indicate more diverse generation.
    
    Uses each text as hypothesis and others as references.
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    
    smoother = SmoothingFunction().method1
    scores = []
    
    for i, hypothesis in enumerate(texts):
        references = [texts[j].split() for j in range(len(texts)) if j != i]
        hypothesis_tokens = hypothesis.split()
        
        if len(references) == 0 or len(hypothesis_tokens) == 0:
            continue
            
        score = sentence_bleu(
            references, hypothesis_tokens,
            weights=[1.0/n] * n,
            smoothing_function=smoother
        )
        scores.append(score)
    
    return sum(scores) / len(scores) if scores else 0.0
```

---

## Practical Recommendations

### Hyperparameter Guidelines

| Application | Temperature | Top-K | Top-P | Repetition Penalty |
|-------------|-------------|-------|-------|-------------------|
| Translation | 0.0 (greedy) | - | - | 1.0 |
| Summarization | 0.3-0.7 | 40 | 0.9 | 1.1 |
| Creative Writing | 0.8-1.2 | 50 | 0.95 | 1.2 |
| Dialogue | 0.7-1.0 | 40 | 0.9 | 1.1-1.3 |
| Code Generation | 0.2-0.5 | 40 | 0.95 | 1.0 |

### Common Issues and Solutions

| Problem | Symptoms | Solutions |
|---------|----------|-----------|
| Repetition | Same phrase loops | Increase repetition penalty, enable n-gram blocking |
| Generic output | Bland, safe responses | Increase temperature, try nucleus sampling |
| Incoherence | Random word sequences | Lower temperature, increase top-k/top-p |
| Premature ending | Very short outputs | Set minimum length, reduce EOS bias |
| Never ending | Extremely long outputs | Increase EOS bias, reduce max length |

### Quality vs Diversity Trade-off

**Key Insight**: There is a fundamental tension between quality and diversity in text generation. The "degeneration" phenomenon—where high-probability samples are often repetitive or boring—motivates the use of sampling strategies over pure greedy/beam search.

The optimal operating point depends on the application:
- **High-stakes factual** (translation, QA): Favor quality (beam search, low temperature)
- **Creative applications** (stories, poetry): Favor diversity (nucleus sampling, higher temperature)
- **Interactive systems** (chatbots): Balance both (nucleus with moderate temperature)

---

## Summary

Text generation with neural networks requires balancing multiple competing objectives:

1. **Fluency**: Grammatical, coherent text
2. **Relevance**: Appropriate to context/prompt
3. **Diversity**: Novel, interesting content
4. **Consistency**: No contradictions or repetition

Key techniques:
- **Temperature** controls distribution sharpness
- **Top-K** provides hard vocabulary limit
- **Nucleus (Top-P)** adapts to distribution shape
- **Repetition penalties** prevent degenerate loops
- **Length control** shapes output length distribution

The optimal configuration depends entirely on the application—translation favors beam search for accuracy, while creative writing benefits from nucleus sampling for diversity. Experimentation with different parameter combinations is essential for achieving desired generation characteristics.
