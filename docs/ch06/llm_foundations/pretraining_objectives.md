# Pretraining Objectives for Large Language Models

## Learning Objectives

- Compare causal language modeling (CLM) vs masked language modeling (MLM)
- Understand the mathematical foundations of next-token prediction
- Analyze denoising objectives and span corruption
- Evaluate trade-offs between different pretraining approaches

## Introduction

Pretraining objectives define the self-supervised task that LLMs learn from unlabeled text. The choice of objective fundamentally shapes model capabilities, determining whether the model excels at generation, understanding, or both.

## Causal Language Modeling (CLM)

### Autoregressive Formulation

CLM models the probability of text as a product of conditional probabilities:

$$P(x_1, x_2, \ldots, x_n) = \prod_{t=1}^{n} P(x_t | x_1, \ldots, x_{t-1})$$

The training objective minimizes negative log-likelihood:

$$\mathcal{L}_{CLM} = -\sum_{t=1}^{n} \log P_\theta(x_t | x_{<t})$$

### Causal Attention Mask

```python
import torch
import torch.nn as nn

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
```

### CLM Characteristics

| Aspect | Characteristic |
|--------|----------------|
| Attention | Unidirectional (left-to-right) |
| Generation | Natural (autoregressive sampling) |
| Context | Only past tokens |
| Models | GPT series, LLaMA, Mistral |

## Masked Language Modeling (MLM)

### BERT-Style Masking

MLM randomly masks tokens and predicts them from bidirectional context:

$$\mathcal{L}_{MLM} = -\sum_{i \in \mathcal{M}} \log P_\theta(x_i | x_{\backslash \mathcal{M}})$$

Where $\mathcal{M}$ is the set of masked positions.

### Masking Strategy (BERT)

```python
import random

def bert_masking(
    tokens: list,
    mask_token: int,
    vocab_size: int,
    mask_prob: float = 0.15
) -> tuple:
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
```

### MLM Characteristics

| Aspect | Characteristic |
|--------|----------------|
| Attention | Bidirectional (full context) |
| Generation | Requires iterative refinement |
| Context | All non-masked tokens |
| Models | BERT, RoBERTa, DeBERTa |

## Span Corruption (T5)

### Denoising Objective

T5 uses span corruption: replace random spans with sentinel tokens, predict original spans.

```
Input:  "The quick brown [X] the lazy dog"
Target: "[X] fox jumps over"
```

$$\mathcal{L}_{denoise} = -\log P_\theta(\text{corrupted spans} | \text{context})$$

### Implementation

```python
def span_corruption(
    tokens: list,
    sentinel_start_id: int,
    mean_span_length: float = 3.0,
    corruption_rate: float = 0.15
) -> tuple:
    """
    T5-style span corruption.
    
    Args:
        tokens: Input token ids
        sentinel_start_id: Starting id for sentinel tokens ([X], [Y], ...)
        mean_span_length: Average length of corrupted spans
        corruption_rate: Fraction of tokens to corrupt
    """
    n = len(tokens)
    num_corrupted = int(n * corruption_rate)
    
    # Sample span lengths (Poisson distribution)
    span_lengths = []
    total = 0
    while total < num_corrupted:
        length = max(1, int(random.expovariate(1/mean_span_length)))
        span_lengths.append(length)
        total += length
    
    # Sample span start positions (non-overlapping)
    num_spans = len(span_lengths)
    available_positions = list(range(n - max(span_lengths)))
    random.shuffle(available_positions)
    
    # Create corrupted input and target
    input_tokens = []
    target_tokens = []
    sentinel_id = sentinel_start_id
    
    pos = 0
    span_idx = 0
    spans = sorted(zip(available_positions[:num_spans], span_lengths))
    
    for start, length in spans:
        # Add tokens before span
        input_tokens.extend(tokens[pos:start])
        
        # Add sentinel to input
        input_tokens.append(sentinel_id)
        
        # Add sentinel + span to target
        target_tokens.append(sentinel_id)
        target_tokens.extend(tokens[start:start+length])
        
        sentinel_id += 1
        pos = start + length
    
    # Add remaining tokens
    input_tokens.extend(tokens[pos:])
    
    return input_tokens, target_tokens
```

## Prefix Language Modeling

### Hybrid Approach

Prefix LM uses bidirectional attention on a prefix, then causal attention for generation:

```
Prefix (bidirectional): "Translate English to French:"
Generation (causal):    " Le chat est sur le tapis"
```

### Attention Pattern

```python
def prefix_lm_mask(seq_len: int, prefix_len: int) -> torch.Tensor:
    """
    Prefix LM attention mask.
    
    - Prefix tokens: bidirectional attention
    - Generation tokens: causal attention (can see prefix + prior generation)
    """
    mask = torch.zeros(seq_len, seq_len)
    
    # Prefix: full attention to prefix
    mask[:, :prefix_len] = 1
    
    # Generation: causal within generation, full to prefix
    for i in range(prefix_len, seq_len):
        mask[i, :i+1] = 1
    
    return mask
```

## Comparison of Objectives

### Objective Properties

| Objective | Bidirectional | Generative | Compute | Use Case |
|-----------|---------------|------------|---------|----------|
| CLM | ✗ | ✓ | 1x | Generation |
| MLM | ✓ | Limited | 1.15x | Understanding |
| Span Corruption | ✓ | ✓ | 1.15x | Seq2Seq |
| Prefix LM | Partial | ✓ | 1x | Conditional generation |

### Effective Training Tokens

Different objectives see different amounts of "effective" training signal:

```python
def effective_tokens_per_example(
    seq_len: int,
    objective: str,
    mask_rate: float = 0.15
) -> float:
    """Calculate effective training signal per sequence."""
    
    if objective == 'CLM':
        # Every token provides signal (except first)
        return seq_len - 1
    
    elif objective == 'MLM':
        # Only masked tokens provide signal
        return seq_len * mask_rate
    
    elif objective == 'span_corruption':
        # Similar to MLM but with span context
        return seq_len * mask_rate * 1.5  # Approximate
    
    elif objective == 'prefix_lm':
        # Generation portion provides signal
        # Assuming 50% prefix, 50% generation
        return seq_len * 0.5
```

### Training Efficiency

```python
def compute_training_equivalence(
    clm_tokens: int,
    mlm_tokens: int,
    mask_rate: float = 0.15
) -> dict:
    """
    Compare training compute between CLM and MLM.
    
    MLM sees ~15% of tokens as training signal per pass.
    CLM sees ~100% of tokens.
    """
    clm_effective = clm_tokens
    mlm_effective = mlm_tokens * mask_rate
    
    # To match CLM effective tokens, MLM needs more passes
    mlm_needed = clm_tokens / mask_rate
    
    return {
        'clm_effective_tokens': clm_effective,
        'mlm_effective_tokens': mlm_effective,
        'mlm_tokens_to_match_clm': mlm_needed,
        'ratio': mlm_needed / clm_tokens
    }
```

## Modern Pretraining Strategies

### Fill-in-the-Middle (FIM)

For code models, train on multiple objectives:

```python
def fill_in_middle_transform(code: str, fim_rate: float = 0.5) -> str:
    """
    Transform code for fill-in-the-middle training.
    
    Original: "def foo():\n    return 42"
    FIM:      "<PRE>def foo():\n<SUF>\n<MID>    return 42"
    """
    if random.random() > fim_rate:
        return code  # Standard CLM
    
    # Random split point
    split = random.randint(0, len(code))
    prefix = code[:split]
    suffix = code[split:]
    
    # PSM format (prefix-suffix-middle)
    return f"<PRE>{prefix}<SUF>{suffix}<MID>"
```

### UL2 (Mixture of Denoisers)

Google's UL2 mixes multiple objectives:

```python
def ul2_objective_sample() -> dict:
    """
    Sample UL2 training objective.
    
    Mixture:
    - R-denoiser: Regular span corruption (short spans)
    - S-denoiser: Sequential denoising (prefix LM)
    - X-denoiser: Extreme span corruption (long spans)
    """
    objectives = [
        {'name': 'R', 'mean_span': 3, 'corruption': 0.15, 'weight': 0.5},
        {'name': 'S', 'mean_span': None, 'corruption': None, 'weight': 0.25},  # Prefix LM
        {'name': 'X', 'mean_span': 32, 'corruption': 0.5, 'weight': 0.25},
    ]
    
    r = random.random()
    cumsum = 0
    for obj in objectives:
        cumsum += obj['weight']
        if r < cumsum:
            return obj
    
    return objectives[-1]
```

## Summary

| Objective | Best For | Key Models |
|-----------|----------|------------|
| **CLM** | Text generation, chat | GPT, LLaMA, Mistral |
| **MLM** | Understanding, embeddings | BERT, RoBERTa |
| **Span Corruption** | Seq2seq, translation | T5, BART |
| **Prefix LM** | Conditional generation | PaLM (partial) |
| **UL2** | General purpose | Flan-UL2 |

## Key Equations

**Causal LM**:
$$\boxed{\mathcal{L}_{CLM} = -\sum_{t=1}^{n} \log P(x_t | x_{<t})}$$

**Masked LM**:
$$\boxed{\mathcal{L}_{MLM} = -\sum_{i \in \mathcal{M}} \log P(x_i | x_{\backslash \mathcal{M}})}$$

## References

1. Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners.
2. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
3. Raffel, C., et al. (2020). Exploring the Limits of Transfer Learning with T5.
4. Tay, Y., et al. (2022). UL2: Unifying Language Learning Paradigms.
