# GPT: Generative Pre-trained Transformer

## Introduction

GPT (Generative Pre-trained Transformer) is a decoder-only Transformer architecture designed for autoregressive language modeling. Unlike BERT's bidirectional approach, GPT uses unidirectional (left-to-right) attention, making it naturally suited for text generation tasks.

## Evolution of GPT

| Model | Parameters | Training Data | Context Length |
|-------|------------|---------------|----------------|
| GPT-1 (2018) | 117M | BooksCorpus | 512 |
| GPT-2 (2019) | 1.5B | WebText | 1024 |
| GPT-3 (2020) | 175B | 300B tokens | 2048 |
| GPT-4 (2023) | ~1.8T* | Undisclosed | 8K-128K |

*Estimated from reports

## Architecture

GPT uses a stack of decoder-only Transformer blocks with causal self-attention:

$$
\text{GPT} = \text{TransformerDecoder}^L
$$

### Key Differences from BERT

| Aspect | GPT | BERT |
|--------|-----|------|
| Architecture | Decoder-only | Encoder-only |
| Attention | Causal (unidirectional) | Bidirectional |
| Pre-training | Next token prediction | Masked language modeling |
| Primary use | Generation | Understanding |

### Model Configuration (GPT-2)

| Model | Layers | Heads | $d_{\text{model}}$ | Parameters |
|-------|--------|-------|---------|------------|
| Small | 12 | 12 | 768 | 117M |
| Medium | 24 | 16 | 1024 | 345M |
| Large | 36 | 20 | 1280 | 762M |
| XL | 48 | 25 | 1600 | 1.5B |

## Pre-training Objective

GPT uses standard language modeling (next token prediction):

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_1, \ldots, x_{t-1}; \theta)
$$

Where the probability is computed via softmax over the vocabulary:

$$
P(x_t | x_{<t}) = \text{softmax}(h_t W_e^T)
$$

Here $h_t$ is the hidden state at position $t$ and $W_e$ is the embedding matrix (weight tying).

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, List


class GPTConfig:
    """GPT model configuration."""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: int = None,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner or 4 * n_embd
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(config.n_positions, config.n_positions))
                .view(1, 1, config.n_positions, config.n_positions)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional KV-cache."""
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # Handle cached KV
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        present = (k, v) if use_cache else None
        
        # Attention
        kv_seq_len = k.size(2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = self.bias[:, :, kv_seq_len - seq_len:kv_seq_len, :kv_seq_len]
        attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply to values
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        output = self.c_proj(output)
        output = self.resid_dropout(output)
        
        return output, present


class GPTBlock(nn.Module):
    """Single GPT Transformer block."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            nn.GELU(),
            nn.Linear(config.n_inner, config.n_embd),
            nn.Dropout(config.dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with pre-norm architecture."""
        attn_output, present = self.attn(self.ln_1(x), layer_past, use_cache)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, present


class GPTModel(nn.Module):
    """GPT Language Model."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.h = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # LM head (weight tied)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        past_length = past_key_values[0][0].size(2) if past_key_values else 0
        
        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + seq_len,
                dtype=torch.long, device=device
            ).unsqueeze(0)
        
        # Embeddings
        hidden_states = self.drop(self.wte(input_ids) + self.wpe(position_ids))
        
        # Blocks
        presents = [] if use_cache else None
        for i, block in enumerate(self.h):
            layer_past = past_key_values[i] if past_key_values else None
            hidden_states, present = block(hidden_states, layer_past, use_cache)
            if use_cache:
                presents.append(present)
        
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {'loss': loss, 'logits': logits, 'past_key_values': presents}
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        past_key_values = None
        
        for _ in range(max_new_tokens):
            model_input = input_ids[:, -1:] if past_key_values else input_ids
            
            outputs = self.forward(model_input, past_key_values=past_key_values, use_cache=True)
            logits = outputs['logits'][:, -1, :] / temperature
            past_key_values = outputs['past_key_values']
            
            # Top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Sample
            if do_sample:
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids


# Sampling strategies
class SamplingStrategies:
    """Text generation sampling strategies."""
    
    @staticmethod
    def greedy(logits: torch.Tensor) -> torch.Tensor:
        """Greedy decoding."""
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    @staticmethod
    def temperature_sampling(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Temperature sampling."""
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    @staticmethod
    def top_k_sampling(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> torch.Tensor:
        """Top-k sampling."""
        logits = logits / temperature
        values, indices = torch.topk(logits, k, dim=-1)
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(1, indices, values)
        probs = F.softmax(logits_filtered, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    @staticmethod
    def nucleus_sampling(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
        """Nucleus (top-p) sampling."""
        logits = logits / temperature
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


# Example usage
if __name__ == "__main__":
    config = GPTConfig(vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12)
    model = GPTModel(config)
    
    input_ids = torch.randint(0, config.vocab_size, (2, 64))
    outputs = model(input_ids, labels=input_ids)
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    generated = model.generate(input_ids[:, :10], max_new_tokens=30, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## In-Context Learning

GPT-3 introduced in-context learning where the model adapts to tasks through examples in the prompt:

### Zero-Shot
```
Translate English to French:
sea otter =>
```

### Few-Shot
```
Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
cheese =>
```

## Scaling Laws

GPT-3 demonstrated predictable scaling:

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}
$$

Where $N$ is parameters, suggesting larger models are more sample-efficient.

### Chinchilla Scaling (Hoffmann et al., 2022)

Later work showed that the GPT-3 scaling recipe was suboptimal. The Chinchilla scaling law predicts that the number of training tokens should scale linearly with model parameters:

$$N_{\text{opt}} \approx 20 \cdot D$$

where $N_{\text{opt}}$ is the optimal number of parameters and $D$ is the data budget. This implies GPT-3 (175B parameters, 300B tokens) was significantly undertrained—a compute-optimal model at that FLOP budget would have ~70B parameters trained on ~1.4T tokens.

## Emergent Capabilities

As GPT models scale, qualitatively new capabilities emerge that are not present in smaller models:

| Capability | First Observed | Scale |
|------------|---------------|-------|
| Few-shot learning | GPT-3 (175B) | >10B parameters |
| Chain-of-thought reasoning | ~100B parameters | >60B |
| Code generation | Codex (~12B) | >10B |
| Instruction following | InstructGPT (1.3B+) | With RLHF |

The mechanism behind emergence remains debated—it may reflect a phase transition in the loss landscape or simply improved resolution on evaluation metrics.

## From GPT to ChatGPT: Alignment

GPT models are trained to predict the next token, which doesn't always align with being helpful, harmless, and honest. The path from GPT to ChatGPT involves:

1. **Supervised Fine-Tuning (SFT)**: Train on human-written demonstrations of desired behavior
2. **Reward Model**: Train a model to predict human preferences between outputs
3. **Reinforcement Learning from Human Feedback (RLHF)**: Optimize the policy to maximize the reward model using PPO

This alignment process transforms a next-token predictor into an instruction-following assistant.

## Summary

GPT established the autoregressive language modeling paradigm:

1. **Decoder-Only Architecture**: Simpler than encoder-decoder, scales cleanly
2. **Causal Attention**: Natural for generation, enables KV-caching for efficient inference
3. **Scaling**: Larger models show emergent capabilities not present at smaller scales
4. **In-Context Learning**: Task adaptation without fine-tuning via prompt examples
5. **Weight Tying**: Sharing the embedding matrix $W_e$ between the input embedding layer and the output projection ($P(x_t | x_{<t}) = \text{softmax}(h_t W_e^T)$) reduces parameters and improves consistency

### Why Decoder-Only Became Dominant

GPT's decoder-only approach became the dominant paradigm for large language models because every parameter contributes to every prediction. In an encoder-decoder model, the encoder parameters are idle during generation. Additionally, the unified next-token prediction objective is simple, data-efficient (every position provides a training signal), and naturally supports both understanding and generation at scale.

## References

1. Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training." (GPT-1)
2. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." (GPT-2)
3. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS. (GPT-3)
4. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." (Chinchilla)
5. Ouyang, L., et al. (2022). "Training Language Models to Follow Instructions with Human Feedback." (InstructGPT)

---

## Text Generation with GPT

#### Generation Process

```
Prompt: "The quick brown"
     ↓
Step 1: P(fox|The quick brown) → "fox"
Step 2: P(jumps|The quick brown fox) → "jumps"
Step 3: P(over|...) → "over"
...
```

#### PyTorch Implementation

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

#### Sampling Strategies Comparison

| Strategy | Pros | Cons | Use Case |
|----------|------|------|----------|
| Greedy | Deterministic | Repetitive | Factual output |
| Temperature | Simple control | Can be incoherent | Creative writing |
| Top-k | Limits bad tokens | Fixed cutoff | General use |
| Nucleus (top-p) | Dynamic cutoff | May cut good tokens | Most versatile |
| Beam Search | Optimal under model | Generic output | Translation |
| Contrastive | Diverse, coherent | Slower | High quality |

#### Best Practices

1. **Temperature 0.7-0.9** for creative tasks
2. **Top-p 0.9-0.95** as default
3. **Repetition penalty 1.1-1.3** to reduce loops
4. **Combine top-k and top-p** for best results

#### Summary

GPT generation involves:
1. Process prompt through model
2. Sample next token from distribution
3. Apply sampling strategy (temperature, top-k, top-p)
4. Append token and repeat

#### References

1. Holtzman, A., et al. (2020). "The Curious Case of Neural Text Degeneration."
2. Su, Y., et al. (2022). "A Contrastive Framework for Neural Text Generation."
