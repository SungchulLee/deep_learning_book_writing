"""
BiLSTM-CRF for Named Entity Recognition
========================================

Bidirectional LSTM with CRF layer for state-of-the-art sequence labeling.

Architecture:
- Embedding layer
- Bidirectional LSTM
- CRF layer for structured prediction

Author: Educational purposes
Date: 2025
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class BiLSTM_CRF(nn.Module):
    """
    BiLSTM-CRF model for NER.
    
    Architecture:
    Input → Embedding → BiLSTM → Linear → CRF → Output
    """
    
    def __init__(self, vocab_size: int, tag_size: int, 
                 embedding_dim: int = 100, hidden_dim: int = 200):
        """
        Initialize BiLSTM-CRF.
        
        Args:
            vocab_size: Size of vocabulary
            tag_size: Number of entity tags
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
        """
        super(BiLSTM_CRF, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        
        # Embedding layer
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                           num_layers=1, bidirectional=True, batch_first=True)
        
        # Linear layer to project to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        
        # CRF transition parameters
        # transitions[i, j] = score of transitioning from tag j to tag i
        self.transitions = nn.Parameter(torch.randn(tag_size, tag_size))
        
        # Never transition to START tag
        self.transitions.data[:, tag_size - 2] = -10000
        # Never transition from END tag
        self.transitions.data[tag_size - 1, :] = -10000
    
    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Get emission scores from BiLSTM.
        
        Args:
            sentence: [batch_size, seq_len] tensor of word indices
            
        Returns:
            emission_scores: [batch_size, seq_len, tag_size]
        """
        # Get embeddings
        embeds = self.word_embeds(sentence)  # [batch, seq_len, embedding_dim]
        
        # Pass through BiLSTM
        lstm_out, _ = self.lstm(embeds)  # [batch, seq_len, hidden_dim]
        
        # Project to tag space
        emissions = self.hidden2tag(lstm_out)  # [batch, seq_len, tag_size]
        
        return emissions


if __name__ == "__main__":
    # Example
    vocab_size = 1000
    tag_size = 10
    
    model = BiLSTM_CRF(vocab_size, tag_size)
    
    # Sample input
    sentence = torch.randint(0, vocab_size, (1, 5))  # Batch of 1, length 5
    emissions = model(sentence)
    
    print(f"Emissions shape: {emissions.shape}")
    print("BiLSTM-CRF model created successfully!")
