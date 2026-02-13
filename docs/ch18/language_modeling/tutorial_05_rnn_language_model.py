"""
Tutorial 05: RNN Language Model
=================================

Recurrent Neural Networks (RNNs) overcome the fixed context limitation
of feedforward models by processing sequences of arbitrary length.

Key Concepts:
- Hidden state that carries information across time steps
- Parameter sharing across all time steps
- Ability to model long-range dependencies (in theory)
- Backpropagation Through Time (BPTT)

Mathematical Foundation:
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
P(w_t | w_1,...,w_{t-1}) = softmax(y_t)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple


class RNNLanguageModel(nn.Module):
    """
    RNN-based language model with variable-length context.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings  
        hidden_dim: Dimension of RNN hidden state
        num_layers: Number of RNN layers
        dropout: Dropout probability
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 hidden_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super(RNNLanguageModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer (can be stacked)
        self.rnn = nn.RNN(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output projection layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
    
    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through RNN.
        
        Args:
            x: (batch_size, seq_len) input indices
            hidden: Initial hidden state (optional)
            
        Returns:
            Tuple of (logits, hidden_state)
            logits: (batch_size, seq_len, vocab_size)
            hidden: (num_layers, batch_size, hidden_dim)
        """
        # Get embeddings: (batch, seq_len, embedding_dim)
        embeds = self.embedding(x)
        embeds = self.dropout(embeds)
        
        # RNN forward pass
        # output: (batch, seq_len, hidden_dim)
        # hidden: (num_layers, batch, hidden_dim)
        output, hidden = self.rnn(embeds, hidden)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Project to vocabulary: (batch, seq_len, vocab_size)
        logits = self.fc(output)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Initialize hidden state with zeros."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)


class RNNDataset(Dataset):
    """Dataset for RNN language modeling with sequences."""
    
    def __init__(self, corpus: List[str], vocab, max_seq_len: int = 35):
        """
        Create dataset with sequences.
        
        Args:
            corpus: List of sentences
            vocab: Vocabulary object
            max_seq_len: Maximum sequence length
        """
        self.sequences = []
        
        for sentence in corpus:
            words = sentence.lower().split()
            words = [vocab.START_TOKEN] + words + [vocab.END_TOKEN]
            indices = [vocab.word_to_idx(w) for w in words]
            
            # Split long sequences
            for i in range(0, len(indices) - 1, max_seq_len):
                seq = indices[i:i+max_seq_len+1]
                if len(seq) > 1:  # Need at least input and target
                    self.sequences.append(seq)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return input sequence and target sequence."""
        seq = self.sequences[idx]
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq[1:], dtype=torch.long)
        return input_seq, target_seq


def collate_fn(batch):
    """Collate function for variable-length sequences (padding)."""
    inputs, targets = zip(*batch)
    
    # Find max length in this batch
    max_len = max(len(seq) for seq in inputs)
    
    # Pad sequences
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


def train_rnn_lm(train_corpus: List[str], val_corpus: List[str],
                 vocab, embedding_dim: int = 128, hidden_dim: int = 256,
                 num_layers: int = 2, batch_size: int = 32, 
                 epochs: int = 10, lr: float = 0.001):
    """Train RNN language model."""
    
    print("=" * 70)
    print("Training RNN Language Model")
    print("=" * 70)
    
    # Create datasets
    train_dataset = RNNDataset(train_corpus, vocab)
    val_dataset = RNNDataset(val_corpus, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           collate_fn=collate_fn)
    
    # Initialize model
    model = RNNLanguageModel(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, targets in train_loader:
            # Forward pass
            logits, _ = model(inputs)
            
            # Reshape for loss calculation
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                logits, _ = model(inputs)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                val_loss += criterion(logits, targets).item()
        
        train_ppl = np.exp(total_loss / len(train_loader))
        val_ppl = np.exp(val_loss / len(val_loader))
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train PPL: {train_ppl:.2f} - Val PPL: {val_ppl:.2f}")
    
    return model, train_ppl, val_ppl


def generate_rnn(model, vocab, max_len: int = 20, temperature: float = 1.0):
    """Generate text using RNN model."""
    model.eval()
    
    # Start with START token
    input_idx = vocab.word_to_idx(vocab.START_TOKEN)
    hidden = model.init_hidden(1)
    
    generated = []
    
    with torch.no_grad():
        for _ in range(max_len):
            # Prepare input: (1, 1)
            input_tensor = torch.tensor([[input_idx]], dtype=torch.long)
            
            # Forward pass
            logits, hidden = model(input_tensor, hidden)
            
            # Get probabilities for last time step
            logits = logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=0)
            
            # Sample next word
            next_idx = torch.multinomial(probs, 1).item()
            next_word = vocab.idx_to_word(next_idx)
            
            if next_word == vocab.END_TOKEN:
                break
                
            if next_word not in [vocab.START_TOKEN, vocab.PAD_TOKEN]:
                generated.append(next_word)
            
            input_idx = next_idx
    
    return ' '.join(generated)


if __name__ == "__main__":
    print("""
RNN Language Model Key Points:
================================

Advantages:
- Handles variable-length sequences
- Parameter sharing across time
- Theoretically can model long dependencies

Challenges:
- Vanishing/exploding gradients
- Difficult to capture very long-term dependencies
- Sequential computation (not parallelizable)

Next: LSTM addresses gradient problems!

EXERCISES:
1. Compare RNN with 1, 2, and 3 layers
2. Experiment with different hidden dimensions
3. Implement learning rate scheduling
4. Try different dropout rates
5. Compare with feedforward model from Tutorial 04
6. Implement truncated BPTT for long sequences
    """)
