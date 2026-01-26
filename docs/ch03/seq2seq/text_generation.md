# Text Generation with RNNs

## Introduction

Text generation—producing coherent sequences of text—demonstrates the power of recurrent neural networks as language models. From completing sentences to writing stories, character-level to word-level generation, this section covers the principles and techniques for building text generators.

## Language Modeling Foundation

A language model estimates the probability distribution over sequences:

$$P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^{T} P(w_t | w_1, \ldots, w_{t-1})$$

RNN language models parameterize this conditional probability:

$$P(w_t | w_{1:t-1}) = \text{softmax}(W_o h_t + b_o)$$

Where $h_t$ is the hidden state encoding all previous context.

## Character-Level Generation

### Model Architecture

```python
import torch
import torch.nn as nn

class CharRNN(nn.Module):
    """Character-level RNN for text generation."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden
    
    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)
```

### Data Preparation

```python
class CharDataset(torch.utils.data.Dataset):
    """Dataset for character-level language modeling."""
    
    def __init__(self, text, seq_length):
        self.seq_length = seq_length
        
        # Build vocabulary
        self.chars = sorted(list(set(text)))
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # Encode text
        self.encoded = [self.char2idx[c] for c in text]
    
    def __len__(self):
        return len(self.encoded) - self.seq_length
    
    def __getitem__(self, idx):
        x = torch.tensor(self.encoded[idx:idx + self.seq_length])
        y = torch.tensor(self.encoded[idx + 1:idx + self.seq_length + 1])
        return x, y
```

## Sampling Strategies

### Temperature Sampling

Control randomness with temperature parameter:

$$P(w_i) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

```python
def temperature_sample(logits, temperature=1.0):
    """Sample with temperature scaling."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

### Top-K Sampling

```python
def top_k_sample(logits, k=50, temperature=1.0):
    """Sample from top-k most probable tokens."""
    scaled_logits = logits / temperature
    top_k_logits, top_k_indices = scaled_logits.topk(k, dim=-1)
    probs = torch.softmax(top_k_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    return top_k_indices.gather(-1, sampled_idx).squeeze(-1)
```

### Top-P (Nucleus) Sampling

```python
def top_p_sample(logits, p=0.9, temperature=1.0):
    """Nucleus sampling - sample from top-p cumulative probability."""
    scaled_logits = logits / temperature
    sorted_logits, sorted_indices = scaled_logits.sort(descending=True, dim=-1)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    probs = torch.softmax(sorted_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    
    return sorted_indices.gather(-1, sampled_idx).squeeze(-1)
```

## Complete Generation Function

```python
def generate_text(model, start_string, char2idx, idx2char, length=500,
                  temperature=1.0, top_k=None, top_p=None, device='cpu'):
    """Generate text from a trained character-level model."""
    model.eval()
    
    chars = [char2idx.get(c, 0) for c in start_string]
    input_seq = torch.tensor([chars], device=device)
    hidden = model.init_hidden(1, device)
    
    with torch.no_grad():
        for i in range(len(chars) - 1):
            _, hidden = model(input_seq[:, i:i+1], hidden)
    
    generated = list(start_string)
    current_char = input_seq[:, -1:]
    
    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(current_char, hidden)
            logits = logits[:, -1, :]
            
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

## Evaluation: Perplexity

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i | w_{<i})\right)$$

```python
def evaluate_perplexity(model, dataloader, criterion, device):
    """Compute perplexity on validation/test set."""
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
            hidden = tuple(h.detach() for h in hidden)
    
    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()
```

## Summary

Text generation with RNNs involves:

1. **Language modeling**: Predict next token given history
2. **Sampling strategies**: Temperature, Top-K, Top-P for quality/diversity balance
3. **Evaluation**: Perplexity measures prediction quality

Practical tips:
- Use temperature 0.7-1.0 for balanced output
- Apply repetition penalties for longer generation
- Top-P (0.9) often produces best quality
