# Pre-training Objectives

## Overview

Pre-training objectives define how language models learn representations from unlabeled text. Different objectives lead to different model capabilities.

## Causal Language Modeling (CLM)

Used by GPT, LLaMA, and decoder-only models.

$$
\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log P(x_t | x_1, \ldots, x_{t-1})
$$

**Characteristics:**
- Unidirectional (left-to-right)
- Natural for generation
- Simple training setup

```python
import torch
import torch.nn.functional as F

def causal_lm_loss(logits, labels):
    """
    Causal language modeling loss.
    
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

## Masked Language Modeling (MLM)

Used by BERT, RoBERTa, and encoder-only models.

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i | \mathbf{x}_{\backslash\mathcal{M}})
$$

**Masking strategy (BERT):**
- 15% of tokens selected
- 80% replaced with [MASK]
- 10% replaced with random token
- 10% kept unchanged

```python
import torch
import torch.nn as nn

def create_mlm_data(input_ids, vocab_size, mask_token_id, mask_prob=0.15):
    """Create MLM training data."""
    labels = input_ids.clone()
    
    # Create mask
    probability_matrix = torch.full(input_ids.shape, mask_prob)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # Don't mask special tokens
    labels[~masked_indices] = -100
    
    # 80% [MASK]
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id
    
    # 10% random
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, input_ids.shape)
    input_ids[indices_random] = random_words[indices_random]
    
    # 10% unchanged
    return input_ids, labels
```

## Span Corruption (T5)

Used by T5 and encoder-decoder models.

Replace consecutive spans with sentinel tokens:

```
Input:  "Thank you <X> to your party <Y> week."
Target: "<X> for inviting me <Y> last"
```

```python
def create_span_corruption_data(input_ids, sentinel_start_id, mean_span_length=3, noise_density=0.15):
    """Create span corruption training data (simplified)."""
    seq_len = input_ids.size(1)
    num_masked = int(seq_len * noise_density)
    
    # Select span starts
    num_spans = max(1, num_masked // mean_span_length)
    
    # Create corrupted input and target
    # (Full implementation requires careful span selection)
    pass
```

## Prefix Language Modeling

Combines bidirectional prefix with causal continuation:

$$
\mathcal{L} = -\sum_{t > L_{\text{prefix}}} \log P(x_t | x_1, \ldots, x_{t-1})
$$

Used in UL2 and some encoder-decoder models.

## Replaced Token Detection (ELECTRA)

Train a discriminator to detect replaced tokens:

$$
\mathcal{L} = -\sum_{t=1}^{T} \left[ y_t \log D(x_t) + (1-y_t) \log(1 - D(x_t)) \right]
$$

Where $y_t = 1$ if token was replaced by generator.

```python
class ELECTRA(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator  # Small MLM model
        self.discriminator = discriminator  # Main model
    
    def forward(self, input_ids, masked_indices):
        # Generator predicts masked tokens
        gen_logits = self.generator(input_ids)
        
        # Sample replacements
        with torch.no_grad():
            gen_probs = F.softmax(gen_logits, dim=-1)
            sampled = torch.multinomial(gen_probs.view(-1, gen_probs.size(-1)), 1)
            sampled = sampled.view(input_ids.shape)
        
        # Replace masked positions
        corrupted = input_ids.clone()
        corrupted[masked_indices] = sampled[masked_indices]
        
        # Discriminator predicts which are replaced
        disc_logits = self.discriminator(corrupted)
        labels = (corrupted != input_ids).float()
        
        return disc_logits, labels
```

## Denoising Objectives

### Document Rotation
Rotate document and predict rotation amount.

### Sentence Permutation
Shuffle sentences and reconstruct order.

### Token Deletion
Delete tokens and predict original sequence.

## Comparison

| Objective | Architecture | Bidirectional | Best For |
|-----------|--------------|---------------|----------|
| CLM | Decoder | No | Generation |
| MLM | Encoder | Yes | Understanding |
| Span Corruption | Enc-Dec | Prefix: Yes | Seq2Seq |
| ELECTRA | Encoder | Yes | Efficient pretraining |

## Mixture of Denoisers (UL2)

Combines multiple objectives:

```python
def ul2_objective(input_ids, mode='R'):
    """
    UL2 mixture of denoisers.
    
    Modes:
    - R: Regular denoising (like T5)
    - S: Sequential denoising (prefix LM)
    - X: Extreme denoising (high corruption)
    """
    if mode == 'R':
        # 15% corruption, mean span 3
        return span_corruption(input_ids, 0.15, 3)
    elif mode == 'S':
        # Prefix LM style
        return prefix_lm(input_ids)
    elif mode == 'X':
        # 50% corruption, mean span 32
        return span_corruption(input_ids, 0.50, 32)
```

## Summary

Pre-training objectives shape model capabilities:

1. **CLM**: Best for generation, simple training
2. **MLM**: Best for understanding, bidirectional
3. **Span Corruption**: Efficient, good for seq2seq
4. **ELECTRA**: Sample efficient, strong discriminative
5. **UL2**: Versatile, combines multiple objectives

## References

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers."
2. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with T5."
3. Clark, K., et al. (2020). "ELECTRA: Pre-training Text Encoders as Discriminators."
4. Tay, Y., et al. (2022). "UL2: Unifying Language Learning Paradigms."
