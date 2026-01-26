# Beam Search

## Introduction

Beam search is a decoding algorithm that maintains multiple hypothesis sequences during generation, exploring a broader search space than greedy decoding while remaining computationally tractable. It's essential for achieving high-quality outputs in sequence-to-sequence models.

## The Decoding Problem

Given an encoder representation, the decoder must find the most likely output sequence:

$$\hat{y} = \arg\max_{y} P(y|x) = \arg\max_{y} \prod_{t=1}^{T} P(y_t|y_{<t}, x)$$

### Greedy Decoding

Select the most probable token at each step:

$$\hat{y}_t = \arg\max_{y} P(y|y_{<t}, x)$$

**Problem**: Locally optimal choices may lead to globally suboptimal sequences.

## Beam Search Algorithm

Maintain $k$ best partial hypotheses (the "beam") at each step:

```
Beam width k=3

Step 0: [<SOS>]

Step 1: Expand <SOS> → top 3:
  ["The", "A", "In"]

Step 2: Expand each → 3V candidates → keep top 3:
  ["The cat", "A dog", "The dog"]

Step 3: Continue until <EOS> or max length
```

## Implementation

```python
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Hypothesis:
    """A single beam search hypothesis."""
    tokens: List[int]
    score: float
    hidden: Tuple[torch.Tensor, ...]

def beam_search(model, encoder_outputs, encoder_hidden, 
                sos_idx, eos_idx, beam_width=5, max_length=50,
                length_penalty=0.6):
    """
    Beam search decoding for seq2seq models.
    """
    device = encoder_outputs.device
    
    # Initialize beam with <SOS>
    beams = [Hypothesis(
        tokens=[sos_idx],
        score=0.0,
        hidden=encoder_hidden
    )]
    
    completed = []
    
    for step in range(max_length):
        if not beams:
            break
            
        all_candidates = []
        
        for beam in beams:
            if beam.tokens[-1] == eos_idx:
                completed.append(beam)
                continue
            
            input_token = torch.tensor([[beam.tokens[-1]]], device=device)
            
            with torch.no_grad():
                output, new_hidden = model.decoder(
                    input_token, beam.hidden, encoder_outputs
                )
            
            log_probs = F.log_softmax(output.squeeze(1), dim=-1)
            topk_log_probs, topk_indices = log_probs.topk(beam_width)
            
            for log_prob, idx in zip(topk_log_probs[0], topk_indices[0]):
                all_candidates.append(Hypothesis(
                    tokens=beam.tokens + [idx.item()],
                    score=beam.score + log_prob.item(),
                    hidden=new_hidden
                ))
        
        all_candidates.sort(key=lambda h: h.score, reverse=True)
        beams = all_candidates[:beam_width]
    
    completed.extend(beams)
    
    # Length normalization
    def normalized_score(hyp):
        return hyp.score / (len(hyp.tokens) ** length_penalty)
    
    completed.sort(key=normalized_score, reverse=True)
    return completed[0].tokens if completed else [sos_idx]
```

## Length Normalization

Raw beam search favors shorter sequences. Length normalization corrects this:

$$\text{score}(y) = \frac{\log P(y|x)}{|y|^\alpha}$$

Where $\alpha \in [0, 1]$:
- $\alpha = 0$: No normalization (favors short)
- $\alpha = 1$: Full normalization
- $\alpha = 0.6-0.7$: Common choice

## Coverage Penalty

Prevent repetitive attention patterns:

```python
def coverage_penalty(attention_weights, beta=0.2):
    """Penalize under-attended source positions."""
    coverage = attention_weights.sum(dim=0)
    penalty = beta * torch.min(coverage, torch.ones_like(coverage)).sum()
    return penalty
```

## Diverse Beam Search

Standard beam search often produces similar hypotheses. Diverse beam search encourages variety by penalizing tokens already selected by other groups:

```python
def diverse_beam_search(model, encoder_outputs, encoder_hidden,
                        sos_idx, eos_idx, num_groups=3, 
                        beam_width_per_group=3, diversity_penalty=0.5):
    """Diverse beam search with grouped beams."""
    all_hypotheses = []
    
    for group in range(num_groups):
        # Track tokens used by previous groups
        previous_tokens = set()
        for hyp in all_hypotheses:
            previous_tokens.update(hyp.tokens)
        
        # Run beam search, penalizing previous tokens
        group_beams = beam_search_with_penalty(
            model, encoder_outputs, encoder_hidden,
            sos_idx, eos_idx, 
            beam_width=beam_width_per_group,
            previous_tokens=previous_tokens,
            diversity_penalty=diversity_penalty
        )
        all_hypotheses.extend(group_beams)
    
    return all_hypotheses
```

## N-Best List Generation

Return multiple hypotheses for downstream reranking:

```python
def generate_nbest(model, encoder_outputs, encoder_hidden,
                   sos_idx, eos_idx, n=10, beam_width=20):
    """Generate n-best list of hypotheses."""
    completed = beam_search_all_hypotheses(
        model, encoder_outputs, encoder_hidden,
        sos_idx, eos_idx, beam_width=beam_width
    )
    
    # Sort by normalized score, deduplicate
    completed.sort(key=lambda h: h.score / len(h.tokens), reverse=True)
    
    seen = set()
    nbest = []
    for hyp in completed:
        tokens_tuple = tuple(hyp.tokens)
        if tokens_tuple not in seen:
            seen.add(tokens_tuple)
            nbest.append(hyp)
            if len(nbest) >= n:
                break
    
    return nbest
```

## Batched Beam Search

Process multiple beams in parallel for efficiency:

```python
def batched_beam_search(model, encoder_outputs, encoder_hidden,
                        sos_idx, eos_idx, beam_width=5, max_length=50):
    """Efficient batched beam search."""
    batch_size = encoder_outputs.size(0)
    vocab_size = model.decoder.vocab_size
    device = encoder_outputs.device
    
    # Expand encoder outputs for beam width
    encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, beam_width, 1, 1)
    encoder_outputs = encoder_outputs.view(batch_size * beam_width, -1, -1)
    
    beam_scores = torch.zeros(batch_size, beam_width, device=device)
    beam_tokens = torch.full((batch_size, beam_width, max_length), 
                             sos_idx, dtype=torch.long, device=device)
    
    for step in range(max_length):
        current = beam_tokens[:, :, step].view(-1, 1)
        logits, hidden = model.decoder(current, hidden, encoder_outputs)
        log_probs = F.log_softmax(logits.squeeze(1), dim=-1)
        
        log_probs = log_probs.view(batch_size, beam_width, vocab_size)
        next_scores = beam_scores.unsqueeze(-1) + log_probs
        next_scores = next_scores.view(batch_size, -1)
        
        top_scores, top_indices = next_scores.topk(beam_width, dim=-1)
        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size
        
        beam_scores = top_scores
        # Update beam_tokens and hidden states...
    
    return beam_tokens[:, 0, :]
```

## Comparison of Decoding Strategies

| Method | Quality | Speed | Diversity |
|--------|---------|-------|-----------|
| Greedy | Low | Fast | None |
| Beam Search | High | Medium | Low |
| Sampling | Variable | Fast | High |
| Diverse Beam | Good | Slow | High |

## Hyperparameter Guidelines

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Beam width | 4-10 | Larger = better but slower |
| Length penalty α | 0.6-1.0 | Task-dependent |
| Max length | 1.5-2× source | Prevent runaway |
| Coverage penalty β | 0.0-0.4 | For attention models |

## Summary

Beam search improves generation quality by:

1. **Exploring multiple hypotheses**: Avoids greedy local optima
2. **Length normalization**: Balances short vs long sequences
3. **Coverage penalty**: Ensures full source coverage

Key considerations:
- Beam width trades quality for speed
- Length penalty prevents degenerate outputs
- Batched implementation is crucial for efficiency
- N-best lists enable downstream reranking
