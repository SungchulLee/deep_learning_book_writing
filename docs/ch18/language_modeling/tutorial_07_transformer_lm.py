"""
Tutorial 07: Transformer Language Model (GPT-style)
====================================================

Transformers use self-attention mechanisms instead of recurrence,
enabling parallel training and better long-range modeling.

Architecture:
- Token + Positional Embeddings
- Multi-head Self-Attention
- Feed-Forward Networks
- Layer Normalization
- Residual Connections

Key Concepts:
- Attention: Q, K, V matrices
- Scaled Dot-Product Attention
- Causal (autoregressive) masking
- Parallel processing across sequence

Attention Formula:
Attention(Q, K, V) = softmax(QK^T / √d_k) V
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerLanguageModel(nn.Module):
    """GPT-style Transformer language model."""
    
    def __init__(self, vocab_size, d_model=512, nhead=8, 
                 num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer decoder layers (causal)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def forward(self, src):
        """
        Args:
            src: (batch, seq_len) input tokens
        Returns:
            (batch, seq_len, vocab_size) logits
        """
        seq_len = src.size(1)
        
        # Embeddings
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        
        # Causal mask
        mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        
        # Transformer (using as decoder with self-attention)
        # Note: Using dummy memory for nn.TransformerDecoder
        output = self.transformer(src, src, tgt_mask=mask)
        
        # Output projection
        logits = self.fc(output)
        return logits


def train_transformer_lm(corpus, vocab, d_model=256, nhead=4, 
                        num_layers=4, epochs=10):
    """Train transformer language model."""
    
    from tutorial_05_rnn_language_model import RNNDataset, collate_fn
    from torch.utils.data import DataLoader
    import numpy as np
    
    print("Training Transformer Language Model")
    print("=" * 60)
    
    split = int(0.9 * len(corpus))
    train_dataset = RNNDataset(corpus[:split], vocab, max_seq_len=50)
    val_dataset = RNNDataset(corpus[split:], vocab, max_seq_len=50)
    
    train_loader = DataLoader(train_dataset, batch_size=32, 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
    
    model = TransformerLanguageModel(
        vocab_size=len(vocab),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for inputs, targets in train_loader:
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}: "
              f"Train PPL={np.exp(train_loss/len(train_loader)):.2f}, "
              f"Val PPL={np.exp(val_loss/len(val_loader)):.2f}")
    
    return model


if __name__ == "__main__":
    print("""
Transformer Language Models
===========================

Advantages:
1. Parallel training (no sequential dependency)
2. Better long-range dependencies via attention
3. Scales well with data and compute
4. Foundation of modern LLMs (GPT, BERT, etc.)

Architecture Components:
- Self-Attention: Models relationships between all positions
- Multi-Head: Different attention patterns
- Positional Encoding: Provides position information
- Layer Norm: Stabilizes training
- Residual Connections: Enables deep networks

Comparison:
- RNN/LSTM: Sequential, slower training
- Transformer: Parallel, faster, better scaling
- Trade-off: Memory vs. Speed

Modern LLMs (GPT-3, GPT-4) use this architecture with:
- Billions of parameters
- Massive training data
- Advanced optimization techniques

EXERCISES:
1. Visualize attention weights
2. Experiment with different numbers of heads
3. Try different positional encodings (learned vs sinusoidal)
4. Implement relative positional encoding
5. Add pre-layer normalization
6. Compare with LSTM on same data
    """)
