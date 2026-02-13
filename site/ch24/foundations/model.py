"""
Character-Level Autoregressive Language Model

This module implements a neural autoregressive model that learns to
generate text one character at a time, where each character is predicted
based on all previous characters in the sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class CharRNN(nn.Module):
    """
    Character-level Recurrent Neural Network for text generation.
    
    This is an autoregressive model that:
    1. Takes a sequence of characters as input
    2. Processes them with an RNN (LSTM)
    3. Predicts the next character
    
    Architecture:
        Embedding -> LSTM -> Linear -> Softmax
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 64,
                 hidden_dim: int = 128,
                 n_layers: int = 2):
        """
        Initialize the character RNN.
        
        Args:
            vocab_size: Number of unique characters in vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Number of hidden units in LSTM
            n_layers: Number of LSTM layers
        """
        super(CharRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Embedding layer: converts character indices to dense vectors
        # Each character gets a learnable embedding vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM: processes sequence of embeddings
        # batch_first=True means input shape is [batch, sequence, features]
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0  # Dropout between LSTM layers
        )
        
        # Output layer: maps LSTM hidden state to vocabulary scores
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, 
                x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length]
               Contains character indices
            hidden: Optional hidden state from previous forward pass
                    Tuple of (h_0, c_0) for LSTM
                    
        Returns:
            output: Logits of shape [batch_size, vocab_size]
                   Scores for each character in vocabulary
            hidden: Updated hidden state
        """
        # Get batch size (number of sequences)
        batch_size = x.size(0)
        
        # 1. Embed characters
        # Input: [batch, sequence_length]
        # Output: [batch, sequence_length, embedding_dim]
        embedded = self.embedding(x)
        
        # 2. Process with LSTM
        # If no hidden state provided, LSTM will initialize with zeros
        # lstm_out: [batch, sequence_length, hidden_dim]
        # hidden: (h_n, c_n) - final hidden and cell states
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # 3. Take only the last time step for prediction
        # We want to predict the next character after the sequence
        # Shape: [batch, hidden_dim]
        last_output = lstm_out[:, -1, :]
        
        # 4. Map to vocabulary scores
        # Shape: [batch, vocab_size]
        output = self.fc(last_output)
        
        return output, hidden
    
    def generate(self, 
                start_sequence: torch.Tensor,
                length: int,
                temperature: float = 1.0) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        This is the key autoregressive generation process:
        1. Start with a seed sequence
        2. Predict next character
        3. Append prediction to sequence
        4. Repeat steps 2-3
        
        Args:
            start_sequence: Seed sequence of character indices [sequence_length]
            length: Number of characters to generate
            temperature: Sampling temperature (higher = more random)
                        1.0 = normal, <1.0 = more conservative, >1.0 = more random
                        
        Returns:
            Generated sequence of character indices
        """
        self.eval()  # Set to evaluation mode
        
        # Start with the seed sequence
        generated = start_sequence.clone().unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            hidden = None  # Start with no hidden state
            
            for _ in range(length):
                # Get prediction for next character
                output, hidden = self.forward(generated, hidden)
                
                # Apply temperature to logits
                # Temperature controls randomness:
                # - Low temp (< 1): more confident, less diverse
                # - High temp (> 1): less confident, more diverse
                output = output / temperature
                
                # Convert logits to probabilities
                probs = F.softmax(output, dim=-1)
                
                # Sample next character from probability distribution
                # This makes generation stochastic (random) rather than deterministic
                next_char = torch.multinomial(probs, 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_char], dim=1)
        
        # Remove batch dimension and return
        return generated.squeeze(0)


class SimpleCharTransformer(nn.Module):
    """
    Simplified Transformer for character-level language modeling.
    
    This is a more modern alternative to RNNs, using self-attention
    instead of recurrence. Transformers are the architecture behind
    models like GPT.
    """
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 max_seq_length: int = 100):
        """
        Initialize the transformer model.
        
        Args:
            vocab_size: Number of unique characters
            embedding_dim: Dimension of embeddings (must be divisible by n_heads)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            max_seq_length: Maximum sequence length (for positional encoding)
        """
        super(SimpleCharTransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Token embedding: converts character indices to vectors
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional embedding: adds position information
        # Transformers don't have inherent order, so we add position info
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output layer
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer.
        
        Args:
            x: Input tensor [batch_size, sequence_length]
            
        Returns:
            Output logits [batch_size, vocab_size]
        """
        batch_size, seq_length = x.shape
        
        # Token embeddings
        token_emb = self.token_embedding(x)
        
        # Position embeddings
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        
        # Combine token and position embeddings
        x = token_emb + pos_emb
        
        # Create causal mask: each position can only attend to previous positions
        # This is crucial for autoregressive modeling!
        mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device), diagonal=1)
        mask = mask.bool()
        
        # Apply transformer with causal mask
        x = self.transformer(x, mask=mask)
        
        # Take last position for prediction
        x = x[:, -1, :]
        
        # Map to vocabulary
        output = self.fc(x)
        
        return output


if __name__ == "__main__":
    """
    Demo: Test the models with dummy data
    """
    
    # Hyperparameters
    vocab_size = 50  # 50 different characters
    batch_size = 16
    seq_length = 20
    
    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    print("=" * 60)
    print("Testing CharRNN")
    print("=" * 60)
    
    # Initialize model
    rnn_model = CharRNN(vocab_size, embedding_dim=32, hidden_dim=64, n_layers=2)
    
    # Count parameters
    n_params = sum(p.numel() for p in rnn_model.parameters())
    print(f"Number of parameters: {n_params:,}")
    
    # Forward pass
    output, hidden = rnn_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output represents scores for {vocab_size} possible next characters")
    
    # Test generation
    seed = torch.randint(0, vocab_size, (10,))
    generated = rnn_model.generate(seed, length=20, temperature=1.0)
    print(f"\nGenerated sequence length: {len(generated)}")
    
    print("\n" + "=" * 60)
    print("Testing SimpleCharTransformer")
    print("=" * 60)
    
    # Initialize transformer
    transformer_model = SimpleCharTransformer(
        vocab_size, 
        embedding_dim=64,  # Must be divisible by n_heads
        n_heads=4,
        n_layers=2
    )
    
    n_params = sum(p.numel() for p in transformer_model.parameters())
    print(f"Number of parameters: {n_params:,}")
    
    # Forward pass
    output = transformer_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\n✓ Both models working correctly!")
