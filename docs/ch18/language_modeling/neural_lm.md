# Neural Language Models

## Learning Objectives

By the end of this section, you will be able to:

- Understand the evolution from count-based to neural language models
- Implement feedforward neural language models in PyTorch
- Build RNN-based language models with variable-length context
- Implement LSTM language models to handle long-range dependencies
- Design Transformer-based language models with self-attention
- Compare and select appropriate architectures for different tasks

---

## From N-grams to Neural Models

N-gram models suffer from fundamental limitations: fixed context windows, data sparsity, and no semantic generalization. Neural language models address these by learning **distributed representations** where words are embedded in continuous vector spaces.

### Key Advantages of Neural Language Models

| Aspect | N-gram | Neural |
|--------|--------|--------|
| Context | Fixed, limited | Variable or unlimited |
| Representations | Discrete counts | Continuous embeddings |
| Generalization | None | Semantic similarity |
| Memory | Explicit counts | Learned parameters |
| Smoothing | Explicit techniques | Implicit via embeddings |

---

## Feedforward Neural Language Model

The seminal work of Bengio et al. (2003) introduced neural language models with the following architecture:

### Architecture Overview

```
Input: [w_{t-n+1}, ..., w_{t-1}]  (context words)
         ↓
    [Embedding Layer]  → Lookup word vectors
         ↓
    [Concatenate]      → Join embeddings
         ↓
    [Hidden Layer]     → Non-linear transformation
         ↓
    [Output Layer]     → Scores over vocabulary
         ↓
    [Softmax]          → Probability distribution
         ↓
Output: P(w_t | context)
```

### Mathematical Formulation

Given context words $w_{t-n+1}, \ldots, w_{t-1}$:

1. **Embedding**: $\mathbf{e}_i = C(w_i) \in \mathbb{R}^d$
2. **Concatenation**: $\mathbf{x} = [\mathbf{e}_{t-n+1}; \ldots; \mathbf{e}_{t-1}] \in \mathbb{R}^{(n-1) \cdot d}$
3. **Hidden layer**: $\mathbf{h} = \tanh(\mathbf{W}\mathbf{x} + \mathbf{b})$
4. **Output**: $\mathbf{s} = \mathbf{U}\mathbf{h} + \mathbf{c}$
5. **Softmax**: $P(w_t = v | \text{context}) = \frac{\exp(s_v)}{\sum_{v'} \exp(s_{v'})}$

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


class Vocabulary:
    """Vocabulary management for neural language models."""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.PAD = "<pad>"
        self.UNK = "<unk>"
        self.START = "<s>"
        self.END = "</s>"
        
        # Add special tokens
        for token in [self.PAD, self.UNK, self.START, self.END]:
            self._add(token)
    
    def _add(self, word: str) -> int:
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        return self.word2idx[word]
    
    def build(self, corpus: List[str], min_freq: int = 1):
        """Build vocabulary from corpus."""
        from collections import Counter
        counts = Counter()
        for sentence in corpus:
            counts.update(sentence.lower().split())
        
        for word, count in counts.items():
            if count >= min_freq:
                self._add(word)
        
        print(f"Vocabulary size: {len(self)}")
    
    def encode(self, word: str) -> int:
        return self.word2idx.get(word.lower(), self.word2idx[self.UNK])
    
    def decode(self, idx: int) -> str:
        return self.idx2word[idx]
    
    def __len__(self) -> int:
        return len(self.word2idx)


class FeedforwardLMDataset(Dataset):
    """Dataset for feedforward language model training."""
    
    def __init__(self, corpus: List[str], vocab: Vocabulary, context_size: int):
        self.vocab = vocab
        self.context_size = context_size
        self.examples = []
        
        for sentence in corpus:
            words = sentence.lower().split()
            # Pad with start tokens
            words = [vocab.START] * context_size + words + [vocab.END]
            indices = [vocab.encode(w) for w in words]
            
            # Create (context, target) pairs
            for i in range(context_size, len(indices)):
                context = indices[i - context_size:i]
                target = indices[i]
                self.examples.append((context, target))
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        context, target = self.examples[idx]
        return torch.tensor(context), torch.tensor(target)


class FeedforwardLM(nn.Module):
    """
    Feedforward Neural Language Model (Bengio et al., 2003).
    
    Architecture:
        Embedding → Concatenate → Hidden → Output → Softmax
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        context_size: Number of context words
        hidden_dim: Dimension of hidden layer
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int,
                 context_size: int, hidden_dim: int):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better training."""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch_size, context_size) context word indices
            
        Returns:
            (batch_size, vocab_size) logits
        """
        # Embedding: (batch, context_size) → (batch, context_size, embed_dim)
        embeds = self.embedding(x)
        
        # Flatten: (batch, context_size * embed_dim)
        embeds = embeds.view(x.size(0), -1)
        
        # Hidden layer with tanh activation
        hidden = torch.tanh(self.fc1(embeds))
        
        # Output logits
        logits = self.fc2(hidden)
        
        return logits
    
    def get_next_word_probs(self, context: List[int]) -> torch.Tensor:
        """Get probability distribution over next word."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor([context])
            logits = self(x)
            probs = F.softmax(logits, dim=-1)
        return probs[0]


def train_feedforward_lm(corpus: List[str], context_size: int = 3,
                         embedding_dim: int = 64, hidden_dim: int = 128,
                         epochs: int = 20, batch_size: int = 32,
                         learning_rate: float = 0.001):
    """Train feedforward language model."""
    
    # Build vocabulary
    vocab = Vocabulary()
    vocab.build(corpus)
    
    # Create dataset
    dataset = FeedforwardLMDataset(corpus, vocab, context_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = FeedforwardLM(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        context_size=context_size,
        hidden_dim=hidden_dim
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for contexts, targets in loader:
            logits = model(contexts)
            loss = criterion(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, PPL={perplexity:.2f}")
    
    return model, vocab
```

### Limitations of Feedforward Models

1. **Fixed context window**: Cannot handle arbitrarily long dependencies
2. **No parameter sharing**: Each position has separate weights
3. **Computational cost**: $O(V)$ for output softmax

---

## RNN Language Models

Recurrent Neural Networks address the fixed context limitation by maintaining a **hidden state** that carries information across time steps.

### Architecture

```
For each time step t:
    h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
    y_t = W_hy · h_t + b_y
    P(w_t | w_1,...,w_{t-1}) = softmax(y_t)
```

The hidden state $\mathbf{h}_t$ summarizes the entire history $w_1, \ldots, w_{t-1}$.

### Implementation

```python
class RNNLanguageModel(nn.Module):
    """
    RNN Language Model with variable-length context.
    
    The hidden state carries information from arbitrary history.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of RNN hidden state
        num_layers: Number of stacked RNN layers
        dropout: Dropout probability
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
    
    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None):
        """
        Forward pass through RNN.
        
        Args:
            x: (batch, seq_len) input token indices
            hidden: Optional initial hidden state
            
        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: Final hidden state
        """
        # Embedding: (batch, seq_len, embed_dim)
        embeds = self.dropout(self.embedding(x))
        
        # RNN: (batch, seq_len, hidden_dim)
        output, hidden = self.rnn(embeds, hidden)
        output = self.dropout(output)
        
        # Project to vocabulary
        logits = self.fc(output)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Initialize hidden state with zeros."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)


class RNNLMDataset(Dataset):
    """Dataset for RNN language model (sequence-to-sequence)."""
    
    def __init__(self, corpus: List[str], vocab: Vocabulary, max_len: int = 35):
        self.sequences = []
        
        for sentence in corpus:
            words = sentence.lower().split()
            words = [vocab.START] + words + [vocab.END]
            indices = [vocab.encode(w) for w in words]
            
            # Split into chunks for training
            for i in range(0, len(indices) - 1, max_len):
                seq = indices[i:i + max_len + 1]
                if len(seq) > 1:
                    self.sequences.append(seq)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        # Input: all but last token; Target: all but first token
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])


def collate_sequences(batch):
    """Collate variable-length sequences with padding."""
    inputs, targets = zip(*batch)
    
    # Pad to max length in batch
    max_len = max(len(x) for x in inputs)
    
    padded_inputs = []
    padded_targets = []
    
    for inp, tgt in zip(inputs, targets):
        pad_len = max_len - len(inp)
        if pad_len > 0:
            inp = torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)])
            tgt = torch.cat([tgt, torch.zeros(pad_len, dtype=torch.long)])
        padded_inputs.append(inp)
        padded_targets.append(tgt)
    
    return torch.stack(padded_inputs), torch.stack(padded_targets)
```

### Backpropagation Through Time (BPTT)

RNNs are trained by unrolling the computation graph through time and applying backpropagation. For a sequence of length $T$:

$$\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W}$$

Each gradient term involves products of Jacobians that can lead to **vanishing** or **exploding** gradients.

### Truncated BPTT

For long sequences, we truncate backpropagation to a fixed window while maintaining the forward pass hidden state:

```python
def train_rnn_truncated_bptt(model, data, hidden, seq_len=35):
    """Training with truncated BPTT."""
    model.train()
    
    for i in range(0, data.size(1) - 1, seq_len):
        # Get batch
        seqlen = min(seq_len, data.size(1) - 1 - i)
        inputs = data[:, i:i+seqlen]
        targets = data[:, i+1:i+1+seqlen]
        
        # Detach hidden state from history
        hidden = hidden.detach()
        
        # Forward and backward
        logits, hidden = model(inputs, hidden)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    
    return hidden
```

---

## LSTM Language Models

Long Short-Term Memory networks address the vanishing gradient problem through **gating mechanisms** that control information flow.

### LSTM Equations

$$\mathbf{f}_t = \sigma(\mathbf{W}_f[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(Forget gate)}$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(Input gate)}$$
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \quad \text{(Candidate)}$$
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(Cell update)}$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(Output gate)}$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(Hidden state)}$$

### Implementation

```python
class LSTMLanguageModel(nn.Module):
    """
    LSTM Language Model with gating for long-range dependencies.
    
    The cell state provides a "highway" for gradient flow.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of LSTM hidden state
        num_layers: Number of stacked LSTM layers
        dropout: Dropout probability
        tie_weights: Whether to tie embedding and output weights
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.5, 
                 tie_weights: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Weight tying: embedding and output share weights
        if tie_weights and embedding_dim == hidden_dim:
            self.fc.weight = self.embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        if self.fc.weight is not self.embedding.weight:
            self.fc.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, x: torch.Tensor, hidden: Tuple = None):
        """
        Forward pass through LSTM.
        
        Args:
            x: (batch, seq_len) input tokens
            hidden: Tuple of (h_0, c_0) initial states
            
        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: Tuple of (h_n, c_n) final states
        """
        embeds = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embeds, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int):
        """Initialize hidden and cell states."""
        device = next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)


def train_lstm_lm(corpus: List[str], embedding_dim: int = 256,
                  hidden_dim: int = 512, num_layers: int = 2,
                  epochs: int = 30, batch_size: int = 32,
                  learning_rate: float = 0.001):
    """Train LSTM language model."""
    
    vocab = Vocabulary()
    vocab.build(corpus)
    
    dataset = RNNLMDataset(corpus, vocab)
    loader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=True, collate_fn=collate_sequences)
    
    model = LSTMLanguageModel(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, targets in loader:
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size)
            
            logits, _ = model(inputs, hidden)
            
            # Reshape for loss: (batch * seq_len, vocab_size)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        ppl = torch.exp(torch.tensor(avg_loss))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, PPL={ppl:.2f}")
    
    return model, vocab
```

### AWD-LSTM: Regularized LSTMs

State-of-the-art LSTM language models use aggressive regularization:

1. **Weight dropout**: Dropout on recurrent weights
2. **Embedding dropout**: Dropout on embedding matrix
3. **Locked dropout**: Same dropout mask across time steps
4. **Weight tying**: Share embedding and output weights
5. **Variable-length BPTT**: Randomly sample sequence lengths

---

## Transformer Language Models

Transformers replace recurrence with **self-attention**, enabling parallel training and better long-range modeling.

### Self-Attention Mechanism

Given a sequence of representations $\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_n]$:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

where:
- $\mathbf{Q} = \mathbf{X}\mathbf{W}^Q$ (queries)
- $\mathbf{K} = \mathbf{X}\mathbf{W}^K$ (keys)
- $\mathbf{V} = \mathbf{X}\mathbf{W}^V$ (values)

### Causal Masking

For language modeling, we must prevent attending to future tokens:

$$\text{mask}_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{otherwise} \end{cases}$$

This ensures $P(w_t | w_1, \ldots, w_{t-1})$ only depends on past words.

### Implementation

```python
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention Is All You Need'."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]


class TransformerLM(nn.Module):
    """
    GPT-style Transformer Language Model.
    
    Uses causal (autoregressive) masking for language modeling.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: FFN inner dimension
        dropout: Dropout probability
        max_len: Maximum sequence length
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        
        self.d_model = d_model
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.fc = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
    
    def generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal masking.
        
        Args:
            x: (batch, seq_len) input tokens
            
        Returns:
            (batch, seq_len, vocab_size) logits
        """
        seq_len = x.size(1)
        
        # Embedding with scaling
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Causal mask
        mask = self.generate_causal_mask(seq_len).to(x.device)
        
        # Transformer forward (self-attention only)
        output = self.transformer(x, x, tgt_mask=mask)
        
        # Project to vocabulary
        logits = self.fc(output)
        
        return logits


def train_transformer_lm(corpus: List[str], d_model: int = 256,
                         nhead: int = 4, num_layers: int = 4,
                         epochs: int = 30, batch_size: int = 32):
    """Train Transformer language model."""
    
    vocab = Vocabulary()
    vocab.build(corpus)
    
    dataset = RNNLMDataset(corpus, vocab, max_len=50)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, collate_fn=collate_sequences)
    
    model = TransformerLM(
        vocab_size=len(vocab),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, targets in loader:
            logits = model(inputs)
            
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        ppl = torch.exp(torch.tensor(avg_loss))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, PPL={ppl:.2f}")
    
    return model, vocab
```

---

## Text Generation

Neural language models support various generation strategies:

```python
def generate_text(model, vocab, max_length: int = 50,
                  temperature: float = 1.0, top_k: int = None,
                  top_p: float = None) -> str:
    """
    Generate text from a trained language model.
    
    Args:
        model: Trained LM (supports LSTM or Transformer)
        vocab: Vocabulary object
        max_length: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering (optional)
        top_p: Nucleus sampling threshold (optional)
    """
    model.eval()
    
    # Start with START token
    generated = [vocab.encode(vocab.START)]
    
    # Handle LSTM hidden state
    hidden = None
    if hasattr(model, 'init_hidden'):
        hidden = model.init_hidden(1)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            x = torch.tensor([generated[-50:]])  # Use last 50 tokens
            
            # Forward pass
            if hasattr(model, 'lstm'):
                logits, hidden = model(x, hidden)
            else:
                logits = model(x)
            
            # Get logits for last position
            logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
                logits[indices_to_remove] = float('-inf')
            
            # Apply nucleus (top-p) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    0, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Check for END token
            if next_token == vocab.encode(vocab.END):
                break
            
            generated.append(next_token)
    
    # Decode tokens
    words = [vocab.decode(idx) for idx in generated[1:]]  # Skip START
    return ' '.join(words)
```

---

## Model Comparison

| Aspect | Feedforward | RNN | LSTM | Transformer |
|--------|-------------|-----|------|-------------|
| Context | Fixed window | Unbounded | Unbounded | Full sequence |
| Training | Parallel | Sequential | Sequential | Parallel |
| Long-range deps | Poor | Poor | Good | Excellent |
| Memory | O(window) | O(hidden) | O(hidden) | O(seq²) |
| Modern usage | Rare | Rare | Moderate | Dominant |

### Typical Perplexities (Penn Treebank)

| Model | Parameters | Perplexity |
|-------|------------|------------|
| Feedforward (Bengio) | ~10M | ~140 |
| LSTM (2-layer) | ~20M | ~80-100 |
| AWD-LSTM | ~24M | ~57 |
| Transformer (6-layer) | ~40M | ~60-70 |
| GPT-2 (small) | 117M | ~35 |

---

## Pretrained Language Models

Modern practice leverages pretrained models fine-tuned for specific tasks:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pretrained GPT-2
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Generate text
input_text = "The quick brown fox"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

output = model.generate(
    input_ids,
    max_length=50,
    temperature=0.7,
    top_p=0.95,
    do_sample=True
)

print(tokenizer.decode(output[0]))
```

---

## Summary

Neural language models have evolved from simple feedforward networks to sophisticated Transformer architectures:

1. **Feedforward LMs** introduced continuous word representations but have fixed context
2. **RNN LMs** handle variable-length sequences but suffer from gradient issues
3. **LSTM LMs** address vanishing gradients with gating mechanisms
4. **Transformer LMs** enable parallel training and capture long-range dependencies

Modern large language models (GPT-4, Claude, LLaMA) are scaled-up Transformer language models with billions of parameters.

---

## Exercises

1. **Embedding Visualization**: Train a feedforward LM and visualize word embeddings using t-SNE. Do similar words cluster together?

2. **LSTM vs GRU**: Implement a GRU language model and compare perplexity with LSTM on the same data.

3. **Attention Visualization**: For a Transformer LM, visualize attention patterns. What patterns emerge for different input types?

4. **Generation Quality**: Compare text generated by n-gram, LSTM, and Transformer models on the same prompt.

5. **Fine-tuning**: Fine-tune GPT-2 on a domain-specific corpus (e.g., financial news) and evaluate domain adaptation.

---

## References

1. Bengio, Y., et al. (2003). A neural probabilistic language model. *JMLR*.
2. Mikolov, T., et al. (2010). Recurrent neural network based language model. *INTERSPEECH*.
3. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*.
4. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
5. Merity, S., et al. (2017). Regularizing and optimizing LSTM language models. *ICLR*.
