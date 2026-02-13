# Speculative Decoding

## Introduction

Speculative decoding is a technique that accelerates autoregressive generation by using a smaller **draft model** to propose multiple tokens, which are then verified in parallel by the larger **target model**. This exploits the fact that verification is cheaper than generation due to parallelization.

## The Autoregressive Bottleneck

### Standard Generation

Autoregressive models generate one token at a time:

$$
p(x_{1:T}) = \prod_{t=1}^{T} p(x_t | x_{<t})
$$

Each token requires a full forward pass through the model, making generation:
- **Memory-bound**: Low arithmetic intensity
- **Sequential**: Cannot parallelize across tokens
- **Slow**: Large models have high latency per token

### Key Insight

**Verification is faster than generation**: Given $K$ draft tokens, the target model can verify all of them in a single forward pass (parallel), whereas generating $K$ tokens requires $K$ sequential passes.

## Algorithm

### Overview

```
1. Draft model generates K candidate tokens quickly
2. Target model scores all K tokens in one forward pass
3. Accept tokens until first rejection
4. Sample correction token at rejection point
5. Repeat
```

### Mathematical Framework

Let $p(x)$ be the target distribution and $q(x)$ be the draft distribution.

**Acceptance criterion** for token $x_t$:

$$
\text{Accept with probability } \min\left(1, \frac{p(x_t | x_{<t})}{q(x_t | x_{<t})}\right)
$$

**Rejection sampling correction**: If rejected, sample from the residual:

$$
p'(x) = \text{norm}\left(\max\left(0, p(x) - q(x)\right)\right)
$$

This ensures the output distribution exactly matches the target model.

### Detailed Algorithm

```
Algorithm: Speculative Decoding

Input: Target model p, Draft model q, Prompt x₀, Draft length K
Output: Generated sequence

while not done:
    # Step 1: Draft phase
    for i = 1 to K:
        Sample x̃ᵢ ~ q(· | x₀, x̃₁, ..., x̃ᵢ₋₁)
        Store q(x̃ᵢ | ...)
    
    # Step 2: Verification phase (single forward pass)
    Compute p(x̃₁ | x₀), p(x̃₂ | x₀, x̃₁), ..., p(x̃ₖ | x₀, ..., x̃ₖ₋₁)
    
    # Step 3: Accept/Reject
    n = 0  # Number of accepted tokens
    for i = 1 to K:
        r ~ Uniform(0, 1)
        if r < min(1, p(x̃ᵢ)/q(x̃ᵢ)):
            Accept x̃ᵢ
            n = n + 1
        else:
            # Sample from residual distribution
            Sample x from norm(max(0, p(·) - q(·)))
            Append x to sequence
            break
    
    if all K tokens accepted:
        # Bonus: sample one more token from p
        Sample x ~ p(· | x₀, x̃₁, ..., x̃ₖ)
        Append x to sequence
    
    Update x₀ with accepted tokens
```

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SpeculativeOutput:
    """Output from speculative decoding step."""
    tokens: torch.Tensor
    num_accepted: int
    num_drafted: int
    
    @property
    def acceptance_rate(self) -> float:
        return self.num_accepted / self.num_drafted if self.num_drafted > 0 else 0.0


class SpeculativeDecoder:
    """
    Speculative decoding for accelerated text generation.
    
    Uses a small draft model to propose tokens, verified by larger target model.
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        draft_length: int = 4,
        temperature: float = 1.0
    ):
        self.target = target_model
        self.draft = draft_model
        self.K = draft_length
        self.temperature = temperature
    
    @torch.no_grad()
    def generate_step(
        self,
        input_ids: torch.Tensor,
        target_cache: Optional[Tuple] = None,
        draft_cache: Optional[Tuple] = None
    ) -> Tuple[SpeculativeOutput, Optional[Tuple], Optional[Tuple]]:
        """
        Single step of speculative decoding.
        
        Returns accepted tokens and updated caches.
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Step 1: Generate K draft tokens
        draft_tokens = []
        draft_probs = []
        current_ids = input_ids
        
        for _ in range(self.K):
            logits, draft_cache = self.draft(current_ids, past_caches=draft_cache)
            probs = F.softmax(logits[:, -1, :] / self.temperature, dim=-1)
            
            # Sample from draft distribution
            token = torch.multinomial(probs, num_samples=1)
            draft_tokens.append(token)
            draft_probs.append(probs.gather(-1, token))
            
            current_ids = token
        
        draft_tokens = torch.cat(draft_tokens, dim=1)  # [batch, K]
        draft_probs = torch.cat(draft_probs, dim=1)    # [batch, K]
        
        # Step 2: Verify all K tokens in parallel with target model
        verify_ids = torch.cat([input_ids, draft_tokens], dim=1)
        target_logits, target_cache = self.target(verify_ids, past_caches=target_cache)
        
        # Get target probabilities for draft tokens
        # target_logits[:, -K-1:-1, :] corresponds to positions before each draft token
        target_probs = F.softmax(target_logits[:, -self.K-1:, :] / self.temperature, dim=-1)
        
        # Step 3: Accept/reject with proper indexing
        accepted_tokens = []
        num_accepted = 0
        
        for i in range(self.K):
            # Target probability for the drafted token
            p = target_probs[:, i, :].gather(-1, draft_tokens[:, i:i+1]).squeeze(-1)
            q = draft_probs[:, i]
            
            # Acceptance probability
            accept_prob = torch.clamp(p / q, max=1.0)
            
            # Sample acceptance
            r = torch.rand(batch_size, device=device)
            accept = r < accept_prob
            
            if accept.all():
                accepted_tokens.append(draft_tokens[:, i:i+1])
                num_accepted += 1
            else:
                # Rejection: sample from residual distribution
                residual = torch.clamp(target_probs[:, i, :] - 
                                       F.softmax(logits[:, -1, :] / self.temperature, dim=-1), 
                                       min=0)
                residual = residual / residual.sum(dim=-1, keepdim=True)
                
                # Handle numerical issues
                if torch.isnan(residual).any():
                    correction = torch.multinomial(target_probs[:, i, :], num_samples=1)
                else:
                    correction = torch.multinomial(residual, num_samples=1)
                
                accepted_tokens.append(correction)
                num_accepted += 1
                break
        else:
            # All K tokens accepted - sample bonus token
            bonus = torch.multinomial(target_probs[:, -1, :], num_samples=1)
            accepted_tokens.append(bonus)
            num_accepted += 1
        
        result_tokens = torch.cat(accepted_tokens, dim=1)
        
        return SpeculativeOutput(
            tokens=result_tokens,
            num_accepted=num_accepted,
            num_drafted=self.K
        ), target_cache, draft_cache
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate tokens using speculative decoding.
        """
        generated = input_ids
        total_accepted = 0
        total_drafted = 0
        num_steps = 0
        
        target_cache = None
        draft_cache = None
        
        while generated.size(1) - input_ids.size(1) < max_new_tokens:
            output, target_cache, draft_cache = self.generate_step(
                generated, target_cache, draft_cache
            )
            
            generated = torch.cat([generated, output.tokens], dim=1)
            total_accepted += output.num_accepted
            total_drafted += output.num_drafted
            num_steps += 1
            
            # Reset caches periodically to avoid memory issues
            if num_steps % 50 == 0:
                target_cache = None
                draft_cache = None
        
        stats = {
            'acceptance_rate': total_accepted / total_drafted if total_drafted > 0 else 0,
            'tokens_per_step': total_accepted / num_steps if num_steps > 0 else 0,
            'speedup_factor': total_accepted / num_steps if num_steps > 0 else 1
        }
        
        return generated[:, :input_ids.size(1) + max_new_tokens], stats


def speculative_sample(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token: torch.Tensor
) -> Tuple[torch.Tensor, bool]:
    """
    Core speculative sampling operation.
    
    Args:
        target_probs: [vocab_size] target model probabilities
        draft_probs: [vocab_size] draft model probabilities  
        draft_token: Proposed token from draft model
        
    Returns:
        (accepted_token, was_accepted)
    """
    p = target_probs[draft_token]
    q = draft_probs[draft_token]
    
    # Accept with probability min(1, p/q)
    if torch.rand(1) < p / q:
        return draft_token, True
    
    # Reject: sample from residual
    residual = torch.clamp(target_probs - draft_probs, min=0)
    residual = residual / residual.sum()
    
    corrected = torch.multinomial(residual, num_samples=1)
    return corrected, False


# Demonstration with mock models
class MockLanguageModel(nn.Module):
    """Simple mock LM for demonstration."""
    
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_size, nhead=4, batch_first=True)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, past_caches=None):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h), None


if __name__ == "__main__":
    print("Speculative Decoding Demo")
    print("=" * 50)
    
    vocab_size = 1000
    
    # Create models (target is larger)
    target = MockLanguageModel(vocab_size, hidden_size=256, num_layers=6)
    draft = MockLanguageModel(vocab_size, hidden_size=128, num_layers=2)
    
    # Create decoder
    decoder = SpeculativeDecoder(
        target_model=target,
        draft_model=draft,
        draft_length=4,
        temperature=1.0
    )
    
    # Generate
    prompt = torch.randint(0, vocab_size, (1, 10))
    output, stats = decoder.generate(prompt, max_new_tokens=50)
    
    print(f"Prompt length: {prompt.size(1)}")
    print(f"Output length: {output.size(1)}")
    print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
    print(f"Tokens per step: {stats['tokens_per_step']:.2f}")
    print(f"Theoretical speedup: {stats['speedup_factor']:.2f}x")
```

## Theoretical Analysis

### Expected Tokens per Step

If the acceptance rate is $\alpha$, expected accepted tokens per step:

$$
\mathbb{E}[\text{tokens}] = \sum_{k=1}^{K} k \cdot \alpha^{k-1}(1-\alpha) + (K+1)\alpha^K
$$

For $K=4$ and $\alpha=0.8$: $\mathbb{E} \approx 3.36$ tokens per step.

### Speedup Analysis

Let:
- $T_t$ = target model forward pass time
- $T_d$ = draft model forward pass time
- $K$ = draft length
- $\alpha$ = acceptance rate

**Without speculation**: Generate $N$ tokens takes $N \cdot T_t$

**With speculation**: 
- Steps needed: $\approx N / \mathbb{E}[\text{tokens}]$
- Time per step: $K \cdot T_d + T_t$ (K draft passes + 1 verify)

**Speedup**:

$$
S = \frac{N \cdot T_t}{\frac{N}{\mathbb{E}[\text{tokens}]} \cdot (K \cdot T_d + T_t)} = \frac{\mathbb{E}[\text{tokens}] \cdot T_t}{K \cdot T_d + T_t}
$$

When $T_d \ll T_t$:

$$
S \approx \mathbb{E}[\text{tokens}]
$$

## Practical Considerations

### Draft Model Selection

| Approach | Pros | Cons |
|----------|------|------|
| Smaller same-family | High acceptance | Still requires separate model |
| Quantized target | Very high acceptance | Limited speedup |
| n-gram / retrieval | No neural compute | Lower acceptance |
| Early exit | Shares parameters | Architecture changes |

### Acceptance Rate Factors

1. **Distribution alignment**: Draft closer to target → higher acceptance
2. **Temperature**: Higher temperature → more uniform → higher acceptance
3. **Domain match**: In-domain draft → higher acceptance
4. **Sequence position**: Later positions often have higher acceptance

### Memory Considerations

- Must keep both models in memory
- KV cache for both models
- Trade-off: memory vs. speed

## Variants

### Medusa

Uses multiple prediction heads on target model:
- No separate draft model
- Predicts multiple future tokens in parallel
- Single model, reduced memory

### SpecInfer

Tree-structured speculation:
- Multiple draft sequences (tree)
- Verify entire tree in one pass
- Higher acceptance with more candidates

### Staged Speculative Decoding

Chain of progressively larger models:
```
Tiny → Small → Medium → Target
```

## Comparison with Other Acceleration Methods

| Method | Speedup | Exact | Memory | Complexity |
|--------|---------|-------|--------|------------|
| Speculative Decoding | 2-3x | ✓ | 2x models | Medium |
| KV Cache | ~Nx | ✓ | O(seq_len) | Low |
| Flash Attention | 2-4x | ✓ | O(N) | Low |
| Quantization | 2-4x | ✗ | 0.25-0.5x | Medium |
| Pruning | Variable | ✗ | <1x | High |

## Summary

Speculative decoding accelerates LLM inference by:

1. **Parallel verification**: Check multiple draft tokens in one forward pass
2. **Exact sampling**: Mathematically guarantees target distribution
3. **Complementary**: Combines with KV cache, Flash Attention, quantization
4. **Trade-off**: Requires draft model, effectiveness depends on alignment

Typical speedups: **2-3x** with well-matched draft models.

## References

1. Leviathan, Y., et al. (2023). "Fast Inference from Transformers via Speculative Decoding." ICML.
2. Chen, C., et al. (2023). "Accelerating Large Language Model Decoding with Speculative Sampling."
3. Cai, T., et al. (2024). "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads."
4. Miao, X., et al. (2023). "SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference."
