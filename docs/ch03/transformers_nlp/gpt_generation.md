# Text Generation with GPT

## Overview

GPT models generate text autoregressively, predicting one token at a time. This document covers generation strategies, implementation, and best practices.

## Generation Process

```
Prompt: "The quick brown"
     ↓
Step 1: P(fox|The quick brown) → "fox"
Step 2: P(jumps|The quick brown fox) → "jumps"
Step 3: P(over|...) → "over"
...
```

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Callable


class GPTGenerator:
    """Text generation with GPT models."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[int]] = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Sample from top k tokens
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
            stop_tokens: Token IDs that stop generation
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated = input_ids
        past_kv = None
        
        for _ in range(max_new_tokens):
            # Get model output
            if past_kv is not None:
                outputs = self.model(generated[:, -1:], past_key_values=past_kv, use_cache=True)
            else:
                outputs = self.model(generated, use_cache=True)
            
            logits = outputs.logits[:, -1, :]
            past_kv = outputs.past_key_values
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for stop tokens
            if stop_tokens and next_token.item() in stop_tokens:
                break
            
            generated = torch.cat([generated, next_token], dim=-1)
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
    
    @torch.no_grad()
    def generate_beam_search(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        num_beams: int = 5,
        length_penalty: float = 1.0,
        early_stopping: bool = True
    ) -> str:
        """Generate using beam search."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Initialize beams: (log_prob, sequence)
        beams = [(0.0, input_ids)]
        
        for _ in range(max_new_tokens):
            all_candidates = []
            
            for log_prob, seq in beams:
                outputs = self.model(seq)
                logits = outputs.logits[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get top k tokens
                topk_log_probs, topk_indices = torch.topk(log_probs, num_beams)
                
                for i in range(num_beams):
                    new_log_prob = log_prob + topk_log_probs[0, i].item()
                    new_seq = torch.cat([seq, topk_indices[:, i:i+1]], dim=-1)
                    
                    # Length penalty
                    score = new_log_prob / (len(new_seq[0]) ** length_penalty)
                    all_candidates.append((score, new_log_prob, new_seq))
            
            # Select top beams
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = [(c[1], c[2]) for c in all_candidates[:num_beams]]
        
        # Return best sequence
        best_seq = beams[0][1]
        return self.tokenizer.decode(best_seq[0], skip_special_tokens=True)
    
    @torch.no_grad()
    def generate_contrastive(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        k: int = 4,
        alpha: float = 0.6
    ) -> str:
        """
        Contrastive search: balance quality and diversity.
        
        score = (1-α) * prob - α * max_sim_to_context
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated = input_ids
        
        for _ in range(max_new_tokens):
            outputs = self.model(generated, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]
            hidden = outputs.hidden_states[-1][:, -1, :]  # Last layer, last token
            
            # Get top-k candidates
            top_probs, top_indices = torch.topk(F.softmax(logits, dim=-1), k)
            
            best_score = float('-inf')
            best_token = None
            
            for i in range(k):
                token_id = top_indices[0, i]
                prob = top_probs[0, i].item()
                
                # Get hidden state for this token
                candidate_seq = torch.cat([generated, token_id.unsqueeze(0).unsqueeze(0)], dim=-1)
                candidate_out = self.model(candidate_seq, output_hidden_states=True)
                candidate_hidden = candidate_out.hidden_states[-1][:, -1, :]
                
                # Compute max similarity to previous context
                context_hiddens = outputs.hidden_states[-1][0, :-1, :]  # All except last
                similarities = F.cosine_similarity(
                    candidate_hidden.expand(context_hiddens.size(0), -1),
                    context_hiddens,
                    dim=-1
                )
                max_sim = similarities.max().item()
                
                # Contrastive score
                score = (1 - alpha) * prob - alpha * max_sim
                
                if score > best_score:
                    best_score = score
                    best_token = token_id
            
            generated = torch.cat([generated, best_token.unsqueeze(0).unsqueeze(0)], dim=-1)
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)


# Streaming generation
class StreamingGenerator:
    """Generator that yields tokens as they're generated."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
    
    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50
    ):
        """Yield tokens one at a time."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        past_kv = None
        
        for _ in range(max_new_tokens):
            if past_kv is not None:
                outputs = self.model(input_ids[:, -1:], past_key_values=past_kv, use_cache=True)
            else:
                outputs = self.model(input_ids, use_cache=True)
            
            logits = outputs.logits[:, -1, :] / temperature
            past_kv = outputs.past_key_values
            
            # Top-k sampling
            if top_k:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Decode and yield the new token
            token_str = self.tokenizer.decode(next_token[0])
            yield token_str


# Constrained generation
class ConstrainedGenerator:
    """Generate with constraints (e.g., must include certain words)."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
    
    @torch.no_grad()
    def generate_with_keywords(
        self,
        prompt: str,
        keywords: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0
    ) -> str:
        """Generate text that must include given keywords."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Encode keywords
        keyword_ids = [self.tokenizer.encode(kw, add_special_tokens=False) for kw in keywords]
        remaining_keywords = set(range(len(keywords)))
        
        for step in range(max_new_tokens):
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :] / temperature
            
            # Boost probability of keyword tokens if not yet used
            if remaining_keywords and step < max_new_tokens - 10:  # Leave room for completion
                for kw_idx in remaining_keywords:
                    for token_id in keyword_ids[kw_idx]:
                        logits[0, token_id] += 5.0  # Boost
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check if keyword was generated
            for kw_idx in list(remaining_keywords):
                if next_token.item() in keyword_ids[kw_idx]:
                    remaining_keywords.discard(kw_idx)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)


# Example usage
if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    generator = GPTGenerator(model, tokenizer, device='cpu')
    
    prompt = "The future of artificial intelligence"
    
    # Different sampling strategies
    print("=== Sampling Strategies ===\n")
    
    print("Greedy (temperature=0.1):")
    print(generator.generate(prompt, max_new_tokens=50, temperature=0.1))
    
    print("\nTop-k (k=50):")
    print(generator.generate(prompt, max_new_tokens=50, top_k=50))
    
    print("\nNucleus (top_p=0.9):")
    print(generator.generate(prompt, max_new_tokens=50, top_p=0.9))
    
    print("\nWith repetition penalty:")
    print(generator.generate(prompt, max_new_tokens=50, top_p=0.9, repetition_penalty=1.2))
```

## Sampling Strategies Comparison

| Strategy | Pros | Cons | Use Case |
|----------|------|------|----------|
| Greedy | Deterministic | Repetitive | Factual output |
| Temperature | Simple control | Can be incoherent | Creative writing |
| Top-k | Limits bad tokens | Fixed cutoff | General use |
| Nucleus (top-p) | Dynamic cutoff | May cut good tokens | Most versatile |
| Beam Search | Optimal under model | Generic output | Translation |
| Contrastive | Diverse, coherent | Slower | High quality |

## Best Practices

1. **Temperature 0.7-0.9** for creative tasks
2. **Top-p 0.9-0.95** as default
3. **Repetition penalty 1.1-1.3** to reduce loops
4. **Combine top-k and top-p** for best results

## Summary

GPT generation involves:
1. Process prompt through model
2. Sample next token from distribution
3. Apply sampling strategy (temperature, top-k, top-p)
4. Append token and repeat

## References

1. Holtzman, A., et al. (2020). "The Curious Case of Neural Text Degeneration."
2. Su, Y., et al. (2022). "A Contrastive Framework for Neural Text Generation."
