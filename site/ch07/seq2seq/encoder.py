"""
Encoder Module for Sequence-to-Sequence Models
Implements various encoder architectures including LSTM and GRU
"""

import torch
import torch.nn as nn


class BasicEncoder(nn.Module):
    """
    Basic RNN Encoder for Seq2Seq models
    
    Args:
        input_size: Size of input vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_size: Size of hidden state
        num_layers: Number of recurrent layers
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional RNN
        rnn_type: Type of RNN ('LSTM' or 'GRU')
    """
    
    def __init__(self, input_size, embedding_dim, hidden_size, 
                 num_layers=1, dropout=0.1, bidirectional=False, rnn_type='LSTM'):
        super(BasicEncoder, self).__init__()
        
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        
        # Embedding layer
        self.embedding = nn.Embedding(input_size, embedding_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # RNN layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
    def forward(self, input_seq, input_lengths=None):
        """
        Forward pass through encoder
        
        Args:
            input_seq: Input sequence tensor (batch_size, seq_len)
            input_lengths: Actual lengths of sequences (optional)
            
        Returns:
            outputs: All hidden states (batch_size, seq_len, hidden_size * num_directions)
            hidden: Final hidden state
            cell: Final cell state (only for LSTM)
        """
        # Embed input
        embedded = self.embedding(input_seq)
        embedded = self.dropout(embedded)
        
        # Pack padded sequence if lengths provided
        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # Pass through RNN
        if self.rnn_type == 'LSTM':
            outputs, (hidden, cell) = self.rnn(embedded)
        else:  # GRU
            outputs, hidden = self.rnn(embedded)
            cell = None
        
        # Unpack if we packed
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # If bidirectional, combine forward and backward hidden states
        if self.bidirectional:
            # hidden: (num_layers * 2, batch_size, hidden_size)
            # Reshape to (num_layers, batch_size, hidden_size * 2)
            hidden = self._combine_bidirectional(hidden)
            if cell is not None:
                cell = self._combine_bidirectional(cell)
        
        return outputs, hidden, cell
    
    def _combine_bidirectional(self, hidden):
        """Combine forward and backward hidden states"""
        # hidden: (num_layers * 2, batch_size, hidden_size)
        # Output: (num_layers, batch_size, hidden_size * 2)
        num_directions = 2
        batch_size = hidden.size(1)
        
        hidden = hidden.view(self.num_layers, num_directions, batch_size, self.hidden_size)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
        
        return hidden


class ConvEncoder(nn.Module):
    """
    Convolutional Encoder for Seq2Seq models
    Uses 1D convolutions for sequence encoding
    """
    
    def __init__(self, input_size, embedding_dim, hidden_size, 
                 num_layers=3, kernel_size=3, dropout=0.1):
        super(ConvEncoder, self).__init__()
        
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Convolutional layers
        conv_layers = []
        in_channels = embedding_dim
        
        for _ in range(num_layers):
            conv_layers.extend([
                nn.Conv1d(in_channels, hidden_size, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = hidden_size
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
    def forward(self, input_seq):
        """
        Forward pass through convolutional encoder
        
        Args:
            input_seq: Input sequence (batch_size, seq_len)
            
        Returns:
            outputs: Encoded representations (batch_size, seq_len, hidden_size)
        """
        # Embed and transpose for conv1d
        embedded = self.embedding(input_seq)  # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)
        embedded = embedded.transpose(1, 2)  # (batch, embed_dim, seq_len)
        
        # Apply convolutions
        outputs = self.conv_layers(embedded)  # (batch, hidden_size, seq_len)
        outputs = outputs.transpose(1, 2)  # (batch, seq_len, hidden_size)
        
        return outputs, None, None


if __name__ == "__main__":
    # Example usage
    batch_size = 32
    seq_len = 20
    vocab_size = 10000
    embedding_dim = 256
    hidden_size = 512
    
    # Create encoder
    encoder = BasicEncoder(
        input_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
        rnn_type='LSTM'
    )
    
    # Sample input
    input_seq = torch.randint(0, vocab_size, (batch_size, seq_len))
    input_lengths = torch.randint(10, seq_len+1, (batch_size,))
    
    # Forward pass
    outputs, hidden, cell = encoder(input_seq, input_lengths)
    
    print(f"Input shape: {input_seq.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Hidden shape: {hidden.shape}")
    if cell is not None:
        print(f"Cell shape: {cell.shape}")
