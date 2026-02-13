"""
Tutorial 06: LSTM Language Model
==================================

Long Short-Term Memory (LSTM) networks address the vanishing gradient
problem of vanilla RNNs, enabling better long-term dependency modeling.

Key Concepts:
- Cell state: long-term memory
- Gates: forget, input, output
- Gradient flow through cell state
- Better at capturing long-range dependencies

LSTM Equations:
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # Input gate  
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  # Candidate cell state
C_t = f_t * C_{t-1} + i_t * C̃_t  # New cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # Output gate
h_t = o_t * tanh(C_t)  # New hidden state
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


class LSTMLanguageModel(nn.Module):
    """LSTM-based language model."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 num_layers=2, dropout=0.2):
        super(LSTMLanguageModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embeds = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embeds, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden


def train_lstm(corpus, vocab, embedding_dim=256, hidden_dim=512,
               num_layers=2, epochs=15, batch_size=64):
    """Train LSTM language model (similar structure to RNN tutorial)."""
    
    from tutorial_05_rnn_language_model import RNNDataset, collate_fn
    
    print("Training LSTM Language Model")
    print("=" * 60)
    
    # Split corpus
    split = int(0.9 * len(corpus))
    train_data = corpus[:split]
    val_data = corpus[split:]
    
    # Create datasets
    train_dataset = RNNDataset(train_data, vocab)
    val_dataset = RNNDataset(val_data, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           collate_fn=collate_fn)
    
    # Model
    model = LSTMLanguageModel(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                logits, _ = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_loss += loss.item()
        
        train_ppl = np.exp(train_loss / len(train_loader))
        val_ppl = np.exp(val_loss / len(val_loader))
        
        print(f"Epoch {epoch+1}: Train PPL={train_ppl:.2f}, Val PPL={val_ppl:.2f}")
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_lstm_lm.pt')
    
    return model


def generate_lstm(model, vocab, prompt=None, max_len=30, temp=1.0):
    """Generate text with LSTM model."""
    model.eval()
    
    if prompt:
        tokens = [vocab.word_to_idx(w) for w in prompt.lower().split()]
    else:
        tokens = [vocab.word_to_idx(vocab.START_TOKEN)]
    
    hidden = None
    generated = list(tokens)
    
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([tokens], dtype=torch.long)
            logits, hidden = model(x, hidden)
            
            logits = logits[0, -1, :] / temp
            probs = torch.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token == vocab.word_to_idx(vocab.END_TOKEN):
                break
            
            generated.append(next_token)
            tokens = [next_token]
    
    return ' '.join([vocab.idx_to_word(i) for i in generated if i != vocab.word_to_idx(vocab.START_TOKEN)])


if __name__ == "__main__":
    print("""
LSTM Language Model
===================

Key Advantages:
1. Addresses vanishing gradient problem
2. Better at long-term dependencies
3. Cell state provides direct gradient path
4. Gates control information flow

Comparison with RNN:
- LSTM: Better long-term memory, more parameters
- RNN: Simpler, faster, fewer parameters
- LSTM typically achieves lower perplexity

Typical Results:
- LSTM (2 layers, 512 hidden): PPL 80-120 on PTB
- RNN (2 layers, 512 hidden): PPL 120-180 on PTB

EXERCISES:
1. Compare LSTM vs RNN on same data
2. Try GRU (simpler variant of LSTM)
3. Experiment with different numbers of layers (1-4)
4. Implement weight tying (share embedding and output weights)
5. Add layer normalization
6. Try bidirectional LSTM
    """)
