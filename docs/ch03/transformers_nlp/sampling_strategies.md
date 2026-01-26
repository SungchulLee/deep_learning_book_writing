# Sampling Strategies for Text Generation

## Overview

Sampling strategies control how tokens are selected from the model's probability distribution. The choice of strategy significantly impacts output quality, diversity, and coherence.

## Temperature Sampling

Scale logits before softmax to control randomness:

$$
P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

- **T < 1**: Sharper distribution (more deterministic)
- **T = 1**: Original distribution
- **T > 1**: Flatter distribution (more random)

```python
import torch
import torch.nn.functional as F

def temperature_sample(logits, temperature=1.0):
    """Sample with temperature scaling."""
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

## Top-k Sampling

Only sample from the k most probable tokens:

```python
def top_k_sample(logits, k=50, temperature=1.0):
    """Sample from top-k tokens."""
    logits = logits / temperature
    
    # Get top-k values and indices
    top_k_values, top_k_indices = torch.topk(logits, k)
    
    # Create filtered distribution
    probs = F.softmax(top_k_values, dim=-1)
    
    # Sample from top-k
    sample_idx = torch.multinomial(probs, num_samples=1)
    return top_k_indices.gather(-1, sample_idx)
```

**Limitation**: Fixed k doesn't adapt to varying probability mass.

## Nucleus (Top-p) Sampling

Sample from smallest set whose cumulative probability ≥ p:

```python
def nucleus_sample(logits, p=0.9, temperature=1.0):
    """Nucleus (top-p) sampling."""
    logits = logits / temperature
    
    # Sort by probability
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Find cutoff
    sorted_indices_to_remove = cumulative_probs > p
    # Keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Apply mask
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    probs = F.softmax(sorted_logits, dim=-1)
    
    # Sample
    sample_idx = torch.multinomial(probs, num_samples=1)
    return sorted_indices.gather(-1, sample_idx)
```

**Advantage**: Adapts to distribution shape.

## Combined Top-k and Top-p

Use both for more control:

```python
def combined_sample(logits, k=50, p=0.9, temperature=1.0):
    """Combined top-k and top-p sampling."""
    logits = logits / temperature
    
    # First apply top-k
    if k > 0:
        indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    # Then apply top-p
    if p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

## Repetition Penalty

Discourage repeating tokens:

```python
def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    """Apply repetition penalty to previously generated tokens."""
    for token_id in set(generated_ids.tolist()):
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty
    return logits
```

## Frequency and Presence Penalties

OpenAI-style penalties:

```python
def apply_frequency_presence_penalty(
    logits, 
    generated_ids, 
    frequency_penalty=0.5,  # Penalize by count
    presence_penalty=0.5    # Penalize if present at all
):
    """Apply frequency and presence penalties."""
    token_counts = {}
    for token_id in generated_ids.tolist():
        token_counts[token_id] = token_counts.get(token_id, 0) + 1
    
    for token_id, count in token_counts.items():
        logits[0, token_id] -= frequency_penalty * count
        logits[0, token_id] -= presence_penalty  # Once per unique token
    
    return logits
```

## Beam Search

Find approximately optimal sequence:

```python
def beam_search(model, input_ids, num_beams=5, max_length=50, length_penalty=1.0):
    """Beam search decoding."""
    device = input_ids.device
    batch_size = input_ids.size(0)
    
    # Initialize beams: (score, sequence)
    beams = [(0.0, input_ids[0].tolist())]
    
    for _ in range(max_length):
        all_candidates = []
        
        for score, seq in beams:
            seq_tensor = torch.tensor([seq], device=device)
            
            with torch.no_grad():
                outputs = model(seq_tensor)
                logits = outputs.logits[0, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
            
            # Expand each beam with top-k tokens
            top_log_probs, top_indices = torch.topk(log_probs, num_beams)
            
            for log_prob, token_id in zip(top_log_probs.tolist(), top_indices.tolist()):
                new_seq = seq + [token_id]
                new_score = score + log_prob
                
                # Apply length penalty
                length_factor = ((5 + len(new_seq)) / 6) ** length_penalty
                adjusted_score = new_score / length_factor
                
                all_candidates.append((adjusted_score, new_score, new_seq))
        
        # Keep top beams
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = [(c[1], c[2]) for c in all_candidates[:num_beams]]
    
    return beams[0][1]  # Return best sequence
```

## Diverse Beam Search

Generate diverse outputs:

```python
def diverse_beam_search(model, input_ids, num_beams=5, num_groups=5, diversity_penalty=0.5):
    """Diverse beam search with groups."""
    # Each group runs beam search with penalty for tokens from other groups
    all_groups = []
    
    for g in range(num_groups):
        # Standard beam search with diversity penalty
        beams = beam_search_with_diversity(
            model, input_ids, num_beams,
            previous_tokens=[seq for group in all_groups for seq in group],
            penalty=diversity_penalty
        )
        all_groups.append(beams)
    
    return all_groups
```

## Contrastive Search

Balance quality and diversity:

$$
\text{score}(x) = (1 - \alpha) \cdot P(x) - \alpha \cdot \max_i \cos(h_x, h_i)
$$

```python
def contrastive_sample(model, input_ids, k=4, alpha=0.6):
    """Contrastive search: penalize similarity to context."""
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    
    logits = outputs.logits[0, -1, :]
    context_hidden = outputs.hidden_states[-1][0, :-1, :]  # All but last
    
    # Get top-k candidates
    top_probs, top_indices = torch.topk(F.softmax(logits, dim=-1), k)
    
    best_score = float('-inf')
    best_token = None
    
    for prob, token_id in zip(top_probs.tolist(), top_indices.tolist()):
        # Get hidden state for candidate
        candidate_input = torch.cat([input_ids, torch.tensor([[token_id]])], dim=-1)
        with torch.no_grad():
            candidate_output = model(candidate_input, output_hidden_states=True)
        candidate_hidden = candidate_output.hidden_states[-1][0, -1, :]
        
        # Max cosine similarity to context
        similarities = F.cosine_similarity(
            candidate_hidden.unsqueeze(0),
            context_hidden,
            dim=-1
        )
        max_sim = similarities.max().item()
        
        # Contrastive score
        score = (1 - alpha) * prob - alpha * max_sim
        
        if score > best_score:
            best_score = score
            best_token = token_id
    
    return best_token
```

## Speculative Decoding

Use small model to draft, large model to verify:

```python
def speculative_decode(draft_model, target_model, input_ids, gamma=4):
    """Speculative decoding for faster inference."""
    # Draft gamma tokens with small model
    draft_tokens = []
    draft_probs = []
    
    current_ids = input_ids
    for _ in range(gamma):
        with torch.no_grad():
            outputs = draft_model(current_ids)
        probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
        token = torch.multinomial(probs, 1)
        draft_tokens.append(token.item())
        draft_probs.append(probs[token].item())
        current_ids = torch.cat([current_ids, token.unsqueeze(0)], dim=-1)
    
    # Verify with target model
    with torch.no_grad():
        target_outputs = target_model(current_ids)
    
    # Accept/reject each draft token
    accepted = []
    for i, (token, draft_p) in enumerate(zip(draft_tokens, draft_probs)):
        target_p = F.softmax(target_outputs.logits[0, len(input_ids[0]) + i - 1, :], dim=-1)[token].item()
        
        if torch.rand(1).item() < min(1, target_p / draft_p):
            accepted.append(token)
        else:
            # Resample from adjusted distribution
            break
    
    return accepted
```

## Comparison Table

| Strategy | Diversity | Coherence | Speed | Best For |
|----------|-----------|-----------|-------|----------|
| Greedy | Low | High | Fast | Factual |
| Temperature | Tunable | Varies | Fast | Creative |
| Top-k | Medium | Good | Fast | General |
| Top-p | Medium | Good | Fast | Most uses |
| Beam Search | Low | High | Slow | Translation |
| Contrastive | High | High | Slow | Quality |

## Recommended Settings

| Task | Temperature | Top-p | Top-k |
|------|-------------|-------|-------|
| Code generation | 0.2-0.4 | 0.95 | - |
| Creative writing | 0.7-1.0 | 0.9 | 50 |
| Chat/dialogue | 0.7 | 0.9 | 40 |
| Summarization | 0.3-0.5 | 0.9 | - |
| Translation | 0.0 (greedy) | - | - |

## Summary

1. **Top-p (0.9)** is a good default
2. **Temperature** controls randomness
3. **Repetition penalty** prevents loops
4. **Combine strategies** for best results
5. **Task-specific tuning** is important

## References

1. Holtzman, A., et al. (2020). "The Curious Case of Neural Text Degeneration."
2. Su, Y., et al. (2022). "A Contrastive Framework for Neural Text Generation."
3. Leviathan, Y., et al. (2023). "Fast Inference from Transformers via Speculative Decoding."
