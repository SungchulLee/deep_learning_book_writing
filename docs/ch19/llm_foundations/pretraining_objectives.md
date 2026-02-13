# Pretraining Objectives for Large Language Models

## Learning Objectives

- Compare causal language modeling (CLM) vs masked language modeling (MLM)
- Understand the mathematical foundations of next-token prediction
- Analyze denoising objectives and span corruption
- Evaluate trade-offs between different pretraining approaches
- Implement modern pretraining strategies including UL2 and FIM

## Introduction

Pretraining objectives define the self-supervised task that LLMs learn from unlabeled text. The choice of objective fundamentally shapes model capabilities, determining whether the model excels at generation, understanding, or both.

## Causal Language Modeling (CLM)

Used by GPT, LLaMA, Mistral, and all decoder-only models.

### Autoregressive Formulation

CLM models the probability of text as a product of conditional probabilities:

$$
P(x_1, x_2, \ldots, x_n) = \prod_{t=1}^{n} P(x_t | x_1, \ldots, x_{t-1})
$$

The training objective minimizes negative log-likelihood:

$$
\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{n} \log P_\theta(x_t | x_{<t})
$$

### Causal Attention Mask

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create causal attention mask.
    
    Position i can only attend to positions <= i.
    
    Returns:
        Lower triangular mask (seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # 1 = attend, 0 = mask


# Example for sequence length 4:
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```

### Implementation

```python
class CausalLMHead(nn.Module):
    """Causal language modeling head."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        labels: torch.Tensor = None
    ):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            labels: (batch, seq_len) - shifted by 1 for next-token prediction
        """
        logits = self.lm_head(hidden_states)  # (batch, seq_len, vocab_size)
        
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {'loss': loss, 'logits': logits}


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Standalone causal language modeling loss.
    
    Args:
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]
    """
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )
```

### CLM Characteristics

| Aspect | Characteristic |
|--------|----------------|
| Attention | Unidirectional (left-to-right) |
| Generation | Natural (autoregressive sampling) |
| Context | Only past tokens |
| Training | Simple, every token provides signal |
| Models | GPT series, LLaMA, Mistral, Claude |

## Masked Language Modeling (MLM)

Used by BERT, RoBERTa, DeBERTa, and encoder-only models.

### Formulation

MLM randomly masks tokens and predicts them from bidirectional context:

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P_\theta(x_i | x_{\backslash \mathcal{M}})
$$

Where $\mathcal{M}$ is the set of masked positions and $x_{\backslash \mathcal{M}}$ denotes all non-masked tokens.

### BERT-Style Masking Strategy

15% of tokens are selected for prediction:
- 80% replaced with [MASK]
- 10% replaced with random token
- 10% kept unchanged

This strategy prevents the model from learning that [MASK] tokens always need prediction.

```python
import torch
import random
from typing import Tuple


def bert_masking(
    tokens: list,
    mask_token: int,
    vocab_size: int,
    mask_prob: float = 0.15
) -> Tuple[list, list]:
    """
    BERT-style masking: 15% of tokens modified.
    
    Of masked tokens:
    - 80% replaced with [MASK]
    - 10% replaced with random token
    - 10% kept unchanged
    """
    masked_tokens = tokens.copy()
    labels = [-100] * len(tokens)  # -100 = ignore in loss
    
    for i in range(len(tokens)):
        if random.random() < mask_prob:
            labels[i] = tokens[i]  # Original token is the label
            
            r = random.random()
            if r < 0.8:
                masked_tokens[i] = mask_token
            elif r < 0.9:
                masked_tokens[i] = random.randint(0, vocab_size - 1)
            # else: keep original (10%)
    
    return masked_tokens, labels


def create_mlm_batch(
    input_ids: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
    mask_prob: float = 0.15,
    special_token_ids: set = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create MLM training batch with proper tensor operations.
    
    Args:
        input_ids: [batch, seq_len] input token ids
        vocab_size: Size of vocabulary
        mask_token_id: ID of [MASK] token
        mask_prob: Probability of masking each token
        special_token_ids: Set of token ids to never mask (e.g., [CLS], [SEP], [PAD])
    """
    labels = input_ids.clone()
    
    # Create probability matrix
    probability_matrix = torch.full(input_ids.shape, mask_prob)
    
    # Don't mask special tokens
    if special_token_ids:
        for token_id in special_token_ids:
            probability_matrix.masked_fill_(input_ids == token_id, 0.0)
    
    # Sample masked indices
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # Only compute loss on masked tokens
    labels[~masked_indices] = -100
    
    # 80% -> [MASK]
    indices_replaced = torch.bernoulli(
        torch.full(input_ids.shape, 0.8)
    ).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id
    
    # 10% -> random token
    indices_random = torch.bernoulli(
        torch.full(input_ids.shape, 0.5)
    ).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, input_ids.shape, dtype=input_ids.dtype)
    input_ids[indices_random] = random_words[indices_random]
    
    # 10% -> unchanged (already handled by not modifying)
    
    return input_ids, labels
```

### MLM Characteristics

| Aspect | Characteristic |
|--------|----------------|
| Attention | Bidirectional (full context) |
| Generation | Requires iterative refinement or separate decoder |
| Context | All non-masked tokens |
| Training | Only 15% of tokens provide gradient signal |
| Models | BERT, RoBERTa, DeBERTa, ALBERT |

## Span Corruption (T5)

Used by T5, BART, and encoder-decoder models.

### Denoising Objective

Replace consecutive spans with sentinel tokens, then predict the original spans:

```
Input:  "The quick brown [X] the lazy dog"
Target: "[X] fox jumps over"
```

$$
\mathcal{L}_{\text{denoise}} = -\log P_\theta(\text{corrupted spans} | \text{context})
$$

### Implementation

```python
import numpy as np
from typing import List, Tuple


def span_corruption(
    tokens: List[int],
    sentinel_start_id: int,
    mean_span_length: float = 3.0,
    corruption_rate: float = 0.15
) -> Tuple[List[int], List[int]]:
    """
    T5-style span corruption.
    
    Args:
        tokens: Input token ids
        sentinel_start_id: Starting id for sentinel tokens ([X], [Y], ...)
        mean_span_length: Average length of corrupted spans
        corruption_rate: Fraction of tokens to corrupt
        
    Returns:
        (corrupted_input, target) tuple
    """
    n = len(tokens)
    num_to_corrupt = int(n * corruption_rate)
    
    if num_to_corrupt == 0:
        return tokens, []
    
    # Determine number and lengths of spans
    num_spans = max(1, int(num_to_corrupt / mean_span_length))
    
    # Sample span lengths from geometric distribution
    span_lengths = np.random.geometric(1.0 / mean_span_length, num_spans)
    span_lengths = np.clip(span_lengths, 1, n // num_spans)
    
    # Adjust to match target corruption
    total_length = span_lengths.sum()
    if total_length > num_to_corrupt:
        span_lengths = (span_lengths * num_to_corrupt / total_length).astype(int)
        span_lengths = np.maximum(span_lengths, 1)
    
    num_spans = len(span_lengths)
    
    # Sample non-overlapping span positions
    # Divide sequence into num_spans segments, sample one start per segment
    segment_length = n // num_spans
    span_starts = []
    for i in range(num_spans):
        start = i * segment_length
        end = min((i + 1) * segment_length - span_lengths[i], n - span_lengths[i])
        if start < end:
            span_starts.append(np.random.randint(start, end))
        else:
            span_starts.append(start)
    
    # Sort spans by position
    spans = sorted(zip(span_starts, span_lengths))
    
    # Build corrupted input and target
    input_tokens = []
    target_tokens = []
    sentinel_id = sentinel_start_id
    pos = 0
    
    for start, length in spans:
        # Add tokens before this span
        input_tokens.extend(tokens[pos:start])
        
        # Add sentinel to input
        input_tokens.append(sentinel_id)
        
        # Add sentinel + original span to target
        target_tokens.append(sentinel_id)
        target_tokens.extend(tokens[start:start + length])
        
        sentinel_id += 1
        pos = start + length
    
    # Add remaining tokens to input
    input_tokens.extend(tokens[pos:])
    
    # Add final sentinel to target
    target_tokens.append(sentinel_id)
    
    return input_tokens, target_tokens


# Example usage
if __name__ == "__main__":
    tokens = list(range(20))  # [0, 1, 2, ..., 19]
    corrupted, target = span_corruption(tokens, sentinel_start_id=100)
    print(f"Original: {tokens}")
    print(f"Corrupted: {corrupted}")
    print(f"Target: {target}")
```

## Prefix Language Modeling

### Hybrid Approach

Prefix LM uses bidirectional attention on a prefix, then causal attention for generation:

```
Prefix (bidirectional): "Translate English to French:"
Generation (causal):    " Le chat est sur le tapis"
```

$$
\mathcal{L} = -\sum_{t > L_{\text{prefix}}} \log P(x_t | x_1, \ldots, x_{t-1})
$$

### Attention Pattern

```python
def prefix_lm_mask(seq_len: int, prefix_len: int) -> torch.Tensor:
    """
    Prefix LM attention mask.
    
    - Prefix tokens: bidirectional attention (can see all prefix tokens)
    - Generation tokens: causal attention (can see prefix + prior generation)
    
    Args:
        seq_len: Total sequence length
        prefix_len: Length of the prefix portion
        
    Returns:
        Attention mask (seq_len, seq_len)
    """
    mask = torch.zeros(seq_len, seq_len)
    
    # Prefix: full attention within prefix
    mask[:prefix_len, :prefix_len] = 1
    
    # Generation: can see all of prefix + causal within generation
    for i in range(prefix_len, seq_len):
        mask[i, :prefix_len] = 1  # See all prefix
        mask[i, prefix_len:i+1] = 1  # Causal within generation
    
    return mask


# Example: seq_len=6, prefix_len=3
# [[1, 1, 1, 0, 0, 0],   <- prefix token 0
#  [1, 1, 1, 0, 0, 0],   <- prefix token 1  
#  [1, 1, 1, 0, 0, 0],   <- prefix token 2
#  [1, 1, 1, 1, 0, 0],   <- gen token 0 (sees prefix + self)
#  [1, 1, 1, 1, 1, 0],   <- gen token 1
#  [1, 1, 1, 1, 1, 1]]   <- gen token 2
```

## Replaced Token Detection (ELECTRA)

ELECTRA trains a discriminator to detect tokens replaced by a small generator:

$$
\mathcal{L} = -\sum_{t=1}^{T} \left[ y_t \log D(x_t) + (1-y_t) \log(1 - D(x_t)) \right]
$$

Where $y_t = 1$ if token $t$ was replaced by the generator.

```python
class ELECTRA(nn.Module):
    """
    ELECTRA: Pre-training Text Encoders as Discriminators.
    
    Uses a small generator to corrupt text, main model learns to detect corruptions.
    More sample-efficient than MLM since every token provides signal.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        gen_weight: float = 1.0,
        disc_weight: float = 50.0
    ):
        super().__init__()
        self.generator = generator  # Small MLM model
        self.discriminator = discriminator  # Main model
        self.gen_weight = gen_weight
        self.disc_weight = disc_weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        masked_indices: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Args:
            input_ids: Input with [MASK] tokens
            masked_indices: Boolean mask of corrupted positions
            labels: Original token ids at masked positions
        """
        # Generator predicts masked tokens (MLM)
        gen_logits = self.generator(input_ids).logits
        gen_loss = F.cross_entropy(
            gen_logits[masked_indices],
            labels[masked_indices]
        )
        
        # Sample replacements from generator
        with torch.no_grad():
            gen_probs = F.softmax(gen_logits, dim=-1)
            sampled = torch.multinomial(
                gen_probs.view(-1, gen_probs.size(-1)), 1
            ).view(input_ids.shape)
        
        # Create corrupted sequence
        corrupted = input_ids.clone()
        corrupted[masked_indices] = sampled[masked_indices]
        
        # Discriminator predicts which tokens are replaced
        disc_logits = self.discriminator(corrupted).logits
        disc_labels = (corrupted != input_ids).float()
        
        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits.squeeze(-1),
            disc_labels
        )
        
        total_loss = self.gen_weight * gen_loss + self.disc_weight * disc_loss
        
        return {
            'loss': total_loss,
            'gen_loss': gen_loss,
            'disc_loss': disc_loss
        }
```

## Additional Denoising Objectives

### Document Rotation
Rotate document at random point and predict rotation amount.

### Sentence Permutation
Shuffle sentences and reconstruct original order (used in BART).

### Token Deletion
Randomly delete tokens and predict original sequence.

### Token Infilling
Replace spans with single mask token (unlike T5 which uses one mask per span).

## Mixture of Denoisers (UL2)

Google's UL2 combines multiple objectives during pretraining:

```python
from dataclasses import dataclass
from typing import Optional
import random


@dataclass
class UL2Config:
    """Configuration for a UL2 denoising objective."""
    name: str
    mean_span_length: Optional[float]
    corruption_rate: Optional[float]
    prefix: str  # Mode token added to input


UL2_OBJECTIVES = [
    UL2Config('R', mean_span_length=3.0, corruption_rate=0.15, prefix='[R]'),   # Regular
    UL2Config('S', mean_span_length=None, corruption_rate=None, prefix='[S]'),  # Sequential (Prefix LM)
    UL2Config('X', mean_span_length=32.0, corruption_rate=0.50, prefix='[X]'),  # Extreme
]

UL2_WEIGHTS = [0.5, 0.25, 0.25]  # Sampling weights


def sample_ul2_objective() -> UL2Config:
    """Sample a UL2 training objective."""
    return random.choices(UL2_OBJECTIVES, weights=UL2_WEIGHTS)[0]


def ul2_transform(
    tokens: List[int],
    sentinel_start_id: int,
    mode_token_ids: dict
) -> Tuple[List[int], List[int]]:
    """
    Apply UL2 transformation to a sequence.
    
    Args:
        tokens: Input token ids
        sentinel_start_id: Starting id for sentinel tokens
        mode_token_ids: Dict mapping mode names ('R', 'S', 'X') to token ids
    """
    config = sample_ul2_objective()
    
    if config.name == 'S':
        # Sequential (Prefix LM): split into prefix and target
        split_point = random.randint(len(tokens) // 4, 3 * len(tokens) // 4)
        input_tokens = [mode_token_ids['S']] + tokens[:split_point]
        target_tokens = tokens[split_point:]
    else:
        # R or X: span corruption with different parameters
        corrupted, target = span_corruption(
            tokens,
            sentinel_start_id,
            mean_span_length=config.mean_span_length,
            corruption_rate=config.corruption_rate
        )
        input_tokens = [mode_token_ids[config.name]] + corrupted
        target_tokens = target
    
    return input_tokens, target_tokens
```

## Fill-in-the-Middle (FIM)

For code models, FIM enables infilling capabilities while maintaining autoregressive training:

```python
def fill_in_middle_transform(
    code: str,
    fim_rate: float = 0.5,
    fim_spm_rate: float = 0.5
) -> str:
    """
    Transform code for fill-in-the-middle training.
    
    Two formats:
    - PSM (prefix-suffix-middle): <PRE>prefix<SUF>suffix<MID>middle
    - SPM (suffix-prefix-middle): <SUF>suffix<PRE>prefix<MID>middle
    
    Args:
        code: Original code string
        fim_rate: Probability of applying FIM (vs standard CLM)
        fim_spm_rate: When FIM applied, probability of SPM format
    """
    if random.random() > fim_rate:
        return code  # Standard CLM
    
    # Random split point
    split = random.randint(0, len(code))
    prefix = code[:split]
    
    # Optional: random end point for middle
    if random.random() < 0.5:
        end = random.randint(split, len(code))
    else:
        end = len(code)
    
    middle = code[split:end]
    suffix = code[end:]
    
    # Choose format
    if random.random() < fim_spm_rate:
        # SPM format
        return f"<SUF>{suffix}<PRE>{prefix}<MID>{middle}"
    else:
        # PSM format
        return f"<PRE>{prefix}<SUF>{suffix}<MID>{middle}"


# Example:
# Original: "def foo():\n    return 42"
# FIM PSM:  "<PRE>def foo():\n<SUF>\n<MID>    return 42"
```

## Training Efficiency Comparison

### Effective Training Signal

```python
def effective_tokens_per_example(
    seq_len: int,
    objective: str,
    mask_rate: float = 0.15
) -> float:
    """
    Calculate effective training signal per sequence.
    
    Not all objectives provide gradient signal from every token.
    """
    if objective == 'CLM':
        # Every token (except first) provides signal
        return seq_len - 1
    
    elif objective == 'MLM':
        # Only masked tokens provide signal
        return seq_len * mask_rate
    
    elif objective == 'span_corruption':
        # Similar to MLM but with better context
        return seq_len * mask_rate
    
    elif objective == 'prefix_lm':
        # Only generation portion provides signal
        # Assuming ~50% prefix
        return seq_len * 0.5
    
    elif objective == 'ELECTRA':
        # Every token provides discriminator signal
        return seq_len


def training_equivalence(clm_tokens: int, mlm_mask_rate: float = 0.15) -> dict:
    """
    Compute how many MLM tokens needed to match CLM training signal.
    """
    return {
        'clm_effective': clm_tokens,
        'mlm_effective_per_token': mlm_mask_rate,
        'mlm_tokens_to_match': clm_tokens / mlm_mask_rate,
        'ratio': 1 / mlm_mask_rate  # ~6.7x more MLM tokens needed
    }
```

## Comprehensive Comparison

| Objective | Architecture | Bidirectional | Generation | Signal/Token | Best For |
|-----------|--------------|---------------|------------|--------------|----------|
| CLM | Decoder | ✗ | Natural | 100% | Generation, chat |
| MLM | Encoder | ✓ | Limited | 15% | Understanding, embeddings |
| Span Corruption | Enc-Dec | Partial | ✓ | 15% | Seq2seq, translation |
| Prefix LM | Decoder | Partial | ✓ | ~50% | Conditional generation |
| ELECTRA | Encoder | ✓ | Limited | 100% | Efficient pretraining |
| UL2 | Enc-Dec | Mixed | ✓ | Mixed | General purpose |

## Key Equations Summary

**Causal Language Modeling**:
$$
\boxed{\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{n} \log P(x_t | x_{<t})}
$$

**Masked Language Modeling**:
$$
\boxed{\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i | x_{\backslash \mathcal{M}})}
$$

**ELECTRA Discriminator**:
$$
\boxed{\mathcal{L}_{\text{disc}} = -\sum_{t=1}^{T} \left[ y_t \log D(x_t) + (1-y_t) \log(1 - D(x_t)) \right]}
$$

## Summary

| Objective | Key Models | Primary Use Case |
|-----------|------------|------------------|
| **CLM** | GPT, LLaMA, Mistral, Claude | Text generation, chat, reasoning |
| **MLM** | BERT, RoBERTa, DeBERTa | Classification, NLU, embeddings |
| **Span Corruption** | T5, BART, mT5 | Translation, summarization, seq2seq |
| **Prefix LM** | PaLM (partial), UniLM | Conditional generation |
| **ELECTRA** | ELECTRA, DeBERTa v3 | Efficient encoder pretraining |
| **UL2** | Flan-UL2, PaLM 2 | General purpose, multi-task |
| **FIM** | CodeLLaMA, StarCoder | Code completion, infilling |

## References

1. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." (GPT-2)
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers."
3. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with T5."
4. Clark, K., et al. (2020). "ELECTRA: Pre-training Text Encoders as Discriminators."
5. Tay, Y., et al. (2022). "UL2: Unifying Language Learning Paradigms."
6. Bavarian, M., et al. (2022). "Efficient Training of Language Models to Fill in the Middle."
