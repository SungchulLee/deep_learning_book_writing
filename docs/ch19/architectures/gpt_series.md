# GPT Series: Evolution of Autoregressive Language Models

## Learning Objectives

- Trace the evolution from GPT-1 through GPT-4
- Understand the architectural and training changes across versions
- Analyze what enabled emergent capabilities at scale
- Compare GPT variants with open-source alternatives

## GPT-1: Language Understanding through Pretraining (2018)

### Key Innovation

First demonstration that generative pretraining + discriminative fine-tuning achieves strong results across NLP tasks.

### Architecture

```python
# GPT-1 Configuration
config = {
    'n_layers': 12,
    'n_heads': 12,
    'd_model': 768,
    'vocab_size': 40000,  # BPE
    'context_length': 512,
    'parameters': '117M'
}
```

### Training

- **Data**: BooksCorpus (~7000 books, 800M words)
- **Objective**: Causal language modeling
- **Innovation**: Transfer learning for NLP

### Fine-tuning Approach

```python
def gpt1_finetune_objective(lm_logits, task_logits, labels, lm_weight=0.5):
    """
    GPT-1 combines LM loss with task-specific loss.
    
    Total Loss = Task Loss + λ * LM Loss
    """
    task_loss = cross_entropy(task_logits, labels)
    lm_loss = cross_entropy(lm_logits[:-1], lm_logits[1:])
    
    return task_loss + lm_weight * lm_loss
```

## GPT-2: Zero-Shot Task Transfer (2019)

### Key Innovation

Demonstrated that larger models can perform tasks zero-shot without fine-tuning through prompt formatting.

### Scale Progression

| Variant | Parameters | Layers | Hidden | Heads |
|---------|------------|--------|--------|-------|
| Small | 117M | 12 | 768 | 12 |
| Medium | 345M | 24 | 1024 | 16 |
| Large | 762M | 36 | 1280 | 20 |
| XL | 1.5B | 48 | 1600 | 25 |

### Training Data: WebText

```python
# WebText curation pipeline
def webtext_filter(url, content):
    """
    Reddit-based quality filtering:
    - Links with >= 3 karma from Reddit
    - Deduplicated
    - Removed Wikipedia (evaluation overlap)
    """
    return reddit_karma(url) >= 3 and not is_wikipedia(url)

# Result: 8M documents, 40GB text
```

### Zero-Shot Prompting Discovery

```python
def gpt2_zero_shot_translation():
    """
    GPT-2 discovered that formatting data as prompts enables zero-shot tasks.
    """
    prompt = """
    English: Hello, how are you?
    French: Bonjour, comment allez-vous?
    
    English: The weather is nice today.
    French:"""
    
    # Model completes: " Le temps est beau aujourd'hui."
    return generate(prompt)
```

## GPT-3: In-Context Learning (2020)

### Key Innovation

Scaling to 175B parameters revealed **in-context learning**: performing tasks from examples in the prompt without parameter updates.

### Architecture Details

```python
gpt3_config = {
    'n_layers': 96,
    'n_heads': 96,
    'd_model': 12288,
    'head_dim': 128,  # d_model / n_heads
    'd_ff': 49152,     # 4 * d_model
    'vocab_size': 50257,
    'context_length': 2048,
    'parameters': '175B'
}

# Alternating dense/sparse attention patterns
# Modified initialization for stability at scale
```

### Training Scale

| Resource | Amount |
|----------|--------|
| Training tokens | 300B |
| Compute | ~3.14 × 10²³ FLOPs |
| Training time | ~34 days on 1024 V100s |
| Cost estimate | $4.6M |

### In-Context Learning Paradigms

```python
def in_context_learning_demo():
    """Three paradigms for GPT-3."""
    
    # Zero-shot: Task description only
    zero_shot = """
    Translate English to French:
    cheese =>"""
    
    # One-shot: Single example
    one_shot = """
    Translate English to French:
    sea otter => loutre de mer
    cheese =>"""
    
    # Few-shot: Multiple examples
    few_shot = """
    Translate English to French:
    sea otter => loutre de mer
    peppermint => menthe poivrée
    plush giraffe => girafe en peluche
    cheese =>"""
    
    # Few-shot performance approaches fine-tuned models
```

### GPT-3 Variants

| Model | Parameters | Use Case |
|-------|------------|----------|
| Ada | 350M | Simple tasks, fast |
| Babbage | 1.3B | Moderate complexity |
| Curie | 6.7B | Good balance |
| Davinci | 175B | Best quality |

## InstructGPT / GPT-3.5: Alignment (2022)

### Key Innovation

RLHF (Reinforcement Learning from Human Feedback) to align model outputs with human preferences.

### Training Pipeline

```python
def instructgpt_training_stages():
    """
    Three-stage training process for alignment.
    """
    
    # Stage 1: Supervised Fine-Tuning (SFT)
    # Human-written demonstrations
    sft_data = [
        {"prompt": "Explain quantum computing", 
         "response": "[Human-written explanation]"},
        # ... 13K examples
    ]
    
    # Stage 2: Reward Model Training
    # Human preferences: A > B
    comparison_data = [
        {"prompt": "...", 
         "response_a": "...", 
         "response_b": "...",
         "preference": "a"},  # Human chose A
        # ... 33K comparisons
    ]
    
    # Stage 3: PPO (Proximal Policy Optimization)
    # Optimize policy against reward model
    # with KL penalty to stay close to SFT model
```

### ChatGPT Training Data Scale

| Stage | Data Size |
|-------|-----------|
| Pretraining | ~570GB text |
| SFT | ~13K demonstrations |
| Reward Model | ~33K comparisons |
| PPO | ~31K prompts |

## GPT-4: Multimodal Capabilities (2023)

### Key Innovations

1. **Multimodal input**: Text and images
2. **Longer context**: 8K/32K/128K tokens
3. **Improved reasoning**: Chain-of-thought at scale
4. **Better calibration**: More reliable confidence

### Reported Capabilities

```python
# GPT-4 benchmark performance (approximate)
benchmarks = {
    'MMLU': 86.4,      # vs GPT-3.5: 70.0
    'HellaSwag': 95.3, # vs GPT-3.5: 85.5
    'HumanEval': 67.0, # vs GPT-3.5: 48.1
    'GSM8K': 92.0,     # vs GPT-3.5: 57.1
    'Bar Exam': '90th percentile'
}
```

### Architecture (Rumored/Unofficial)

```python
# Speculated GPT-4 architecture (not confirmed by OpenAI)
gpt4_rumored = {
    'architecture': 'Mixture of Experts (MoE)',
    'total_parameters': '~1.8T',
    'active_parameters': '~220B per forward pass',
    'num_experts': 16,
    'experts_per_token': 2,
    'context_lengths': [8192, 32768, 128000]
}
```

### Vision Capabilities

```python
def gpt4_vision_example():
    """
    GPT-4V processes interleaved text and images.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]
        }
    ]
    
    # Model can:
    # - Describe image contents
    # - Answer questions about images
    # - Perform OCR
    # - Analyze charts/graphs
```

## Evolution Summary

```
GPT-1 (2018)                GPT-2 (2019)
117M params                 1.5B params
Fine-tuning required   →    Zero-shot possible
Single task focus           Multi-task potential
         ↓                        ↓
         
GPT-3 (2020)                GPT-3.5/ChatGPT (2022)
175B params                 ~175B params + RLHF
In-context learning    →    Aligned to preferences
Few-shot master             Conversational AI
         ↓                        ↓
         
GPT-4 (2023)                GPT-4o (2024)
~1T+ params (MoE?)          Multimodal I/O
Multimodal input       →    Native audio/video
Strongest reasoning         Real-time interaction
```

## GPT Architecture Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTBlock(nn.Module):
    """Standard GPT transformer block."""
    
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['d_model'])
        self.ln2 = nn.LayerNorm(config['d_model'])
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config['d_model'], 4 * config['d_model']),
            nn.GELU(),
            nn.Linear(4 * config['d_model'], config['d_model']),
            nn.Dropout(config['dropout'])
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model."""
    
    def __init__(self, config):
        super().__init__()
        
        self.tok_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        self.pos_emb = nn.Embedding(config['context_length'], config['d_model'])
        self.drop = nn.Dropout(config['dropout'])
        
        self.blocks = nn.ModuleList([
            GPTBlock(config) for _ in range(config['n_layers'])
        ])
        
        self.ln_f = nn.LayerNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        
        # Weight tying
        self.tok_emb.weight = self.head.weight
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
```

## Comparison with Open Models

| Aspect | GPT-4 | LLaMA-2 70B | Mistral 7B |
|--------|-------|-------------|------------|
| Parameters | ~1T (MoE) | 70B | 7B |
| Open weights | ✗ | ✓ | ✓ |
| Reasoning | Best | Strong | Good |
| Cost/query | $$$ | Self-host | Self-host |
| Fine-tuning | API only | Full access | Full access |

## Key Takeaways

1. **Scale enables capabilities**: Each GPT version unlocked new abilities
2. **Prompting evolved**: Fine-tune → Zero-shot → Few-shot → In-context
3. **Alignment matters**: RLHF made models useful for consumers
4. **Architecture innovations**: MoE may enable further scaling
5. **Multimodality expands**: Vision, audio, video integration

## References

1. Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training.
2. Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners.
3. Brown, T., et al. (2020). Language Models are Few-Shot Learners.
4. Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback.
5. OpenAI. (2023). GPT-4 Technical Report.
