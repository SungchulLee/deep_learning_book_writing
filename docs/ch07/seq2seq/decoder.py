"""
Decoder Module for Sequence-to-Sequence Models
Implements various decoder architectures with and without attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicDecoder(nn.Module):
    """
    Basic RNN Decoder for Seq2Seq models
    
    Args:
        output_size: Size of output vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_size: Size of hidden state
        num_layers: Number of recurrent layers
        dropout: Dropout probability
        rnn_type: Type of RNN ('LSTM' or 'GRU')
    """
    
    def __init__(self, output_size, embedding_dim, hidden_size, 
                 num_layers=1, dropout=0.1, rnn_type='LSTM'):
        super(BasicDecoder, self).__init__()
        
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # Embedding layer
        self.embedding = nn.Embedding(output_size, embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # RNN layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_token, hidden, cell=None):
        """
        Forward pass for one time step
        
        Args:
            input_token: Input token (batch_size, 1)
            hidden: Hidden state from previous time step
            cell: Cell state from previous time step (LSTM only)
            
        Returns:
            output: Output predictions (batch_size, output_size)
            hidden: Updated hidden state
            cell: Updated cell state (LSTM only)
        """
        # Embed input token
        embedded = self.embedding(input_token)  # (batch_size, 1, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Pass through RNN
        if self.rnn_type == 'LSTM':
            rnn_output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        else:  # GRU
            rnn_output, hidden = self.rnn(embedded, hidden)
            cell = None
        
        # Generate output prediction
        output = self.fc_out(rnn_output.squeeze(1))  # (batch_size, output_size)
        
        return output, hidden, cell


class AttentionDecoder(nn.Module):
    """
    Decoder with Bahdanau (additive) attention mechanism
    
    Args:
        output_size: Size of output vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_size: Size of decoder hidden state
        encoder_hidden_size: Size of encoder hidden state
        num_layers: Number of recurrent layers
        dropout: Dropout probability
        rnn_type: Type of RNN ('LSTM' or 'GRU')
    """
    
    def __init__(self, output_size, embedding_dim, hidden_size, 
                 encoder_hidden_size, num_layers=1, dropout=0.1, rnn_type='LSTM'):
        super(AttentionDecoder, self).__init__()
        
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # Embedding layer
        self.embedding = nn.Embedding(output_size, embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = BahdanauAttention(hidden_size, encoder_hidden_size)
        
        # RNN layer (input is embedding + context vector)
        rnn_input_size = embedding_dim + encoder_hidden_size
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                rnn_input_size,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                rnn_input_size,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size + encoder_hidden_size + embedding_dim, output_size)
        
    def forward(self, input_token, hidden, encoder_outputs, cell=None, mask=None):
        """
        Forward pass for one time step with attention
        
        Args:
            input_token: Input token (batch_size, 1)
            hidden: Hidden state from previous time step
            encoder_outputs: All encoder outputs (batch_size, src_len, encoder_hidden_size)
            cell: Cell state from previous time step (LSTM only)
            mask: Mask for padding (batch_size, src_len)
            
        Returns:
            output: Output predictions (batch_size, output_size)
            hidden: Updated hidden state
            cell: Updated cell state (LSTM only)
            attention_weights: Attention weights (batch_size, src_len)
        """
        # Embed input token
        embedded = self.embedding(input_token)  # (batch_size, 1, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Calculate attention
        # Use the top layer's hidden state for attention
        query = hidden[-1].unsqueeze(1) if hidden.dim() == 3 else hidden.unsqueeze(1)
        context, attention_weights = self.attention(query, encoder_outputs, mask)
        
        # Concatenate embedded input and context vector
        rnn_input = torch.cat([embedded, context], dim=2)
        
        # Pass through RNN
        if self.rnn_type == 'LSTM':
            rnn_output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        else:  # GRU
            rnn_output, hidden = self.rnn(rnn_input, hidden)
            cell = None
        
        # Concatenate RNN output, context, and embedded input for prediction
        output_input = torch.cat([
            rnn_output.squeeze(1),
            context.squeeze(1),
            embedded.squeeze(1)
        ], dim=1)
        
        # Generate output prediction
        output = self.fc_out(output_input)  # (batch_size, output_size)
        
        return output, hidden, cell, attention_weights.squeeze(1)


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism
    
    Args:
        decoder_hidden_size: Size of decoder hidden state
        encoder_hidden_size: Size of encoder hidden state
    """
    
    def __init__(self, decoder_hidden_size, encoder_hidden_size):
        super(BahdanauAttention, self).__init__()
        
        self.W_decoder = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        self.W_encoder = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.V = nn.Linear(decoder_hidden_size, 1)
        
    def forward(self, query, keys, mask=None):
        """
        Compute attention weights and context vector
        
        Args:
            query: Decoder hidden state (batch_size, 1, decoder_hidden_size)
            keys: Encoder outputs (batch_size, src_len, encoder_hidden_size)
            mask: Padding mask (batch_size, src_len)
            
        Returns:
            context: Context vector (batch_size, 1, encoder_hidden_size)
            attention_weights: Attention weights (batch_size, 1, src_len)
        """
        # Calculate attention scores
        # query: (batch, 1, dec_hidden)
        # keys: (batch, src_len, enc_hidden)
        
        query_transformed = self.W_decoder(query)  # (batch, 1, dec_hidden)
        keys_transformed = self.W_encoder(keys)    # (batch, src_len, dec_hidden)
        
        # Broadcast and add
        scores = self.V(torch.tanh(query_transformed + keys_transformed))  # (batch, src_len, 1)
        scores = scores.squeeze(2)  # (batch, src_len)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Calculate attention weights
        attention_weights = F.softmax(scores, dim=1).unsqueeze(1)  # (batch, 1, src_len)
        
        # Calculate context vector
        context = torch.bmm(attention_weights, keys)  # (batch, 1, encoder_hidden_size)
        
        return context, attention_weights


class LuongAttention(nn.Module):
    """
    Luong (Multiplicative) Attention Mechanism
    
    Args:
        decoder_hidden_size: Size of decoder hidden state
        encoder_hidden_size: Size of encoder hidden state
        attention_type: Type of scoring function ('dot', 'general', or 'concat')
    """
    
    def __init__(self, decoder_hidden_size, encoder_hidden_size, attention_type='general'):
        super(LuongAttention, self).__init__()
        
        self.attention_type = attention_type
        
        if attention_type == 'general':
            self.W = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=False)
        elif attention_type == 'concat':
            self.W = nn.Linear(decoder_hidden_size + encoder_hidden_size, decoder_hidden_size)
            self.V = nn.Linear(decoder_hidden_size, 1, bias=False)
        
    def forward(self, query, keys, mask=None):
        """
        Compute attention weights and context vector
        
        Args:
            query: Decoder hidden state (batch_size, 1, decoder_hidden_size)
            keys: Encoder outputs (batch_size, src_len, encoder_hidden_size)
            mask: Padding mask (batch_size, src_len)
            
        Returns:
            context: Context vector (batch_size, 1, encoder_hidden_size)
            attention_weights: Attention weights (batch_size, 1, src_len)
        """
        if self.attention_type == 'dot':
            # Simple dot product
            scores = torch.bmm(query, keys.transpose(1, 2))  # (batch, 1, src_len)
        elif self.attention_type == 'general':
            # General: query * W * keys^T
            keys_transformed = self.W(keys)  # (batch, src_len, dec_hidden)
            scores = torch.bmm(query, keys_transformed.transpose(1, 2))  # (batch, 1, src_len)
        elif self.attention_type == 'concat':
            # Concat: V * tanh(W * [query; keys])
            src_len = keys.size(1)
            query_expanded = query.expand(-1, src_len, -1)  # (batch, src_len, dec_hidden)
            concat = torch.cat([query_expanded, keys], dim=2)  # (batch, src_len, dec+enc_hidden)
            scores = self.V(torch.tanh(self.W(concat)))  # (batch, src_len, 1)
            scores = scores.transpose(1, 2)  # (batch, 1, src_len)
        
        scores = scores.squeeze(1)  # (batch, src_len)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Calculate attention weights
        attention_weights = F.softmax(scores, dim=1).unsqueeze(1)  # (batch, 1, src_len)
        
        # Calculate context vector
        context = torch.bmm(attention_weights, keys)  # (batch, 1, encoder_hidden_size)
        
        return context, attention_weights


if __name__ == "__main__":
    # Example usage
    batch_size = 32
    src_len = 20
    vocab_size = 10000
    embedding_dim = 256
    hidden_size = 512
    encoder_hidden_size = 512
    
    # Create attention decoder
    decoder = AttentionDecoder(
        output_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        encoder_hidden_size=encoder_hidden_size,
        num_layers=2,
        dropout=0.1,
        rnn_type='LSTM'
    )
    
    # Sample inputs
    input_token = torch.randint(0, vocab_size, (batch_size, 1))
    hidden = torch.randn(2, batch_size, hidden_size)
    cell = torch.randn(2, batch_size, hidden_size)
    encoder_outputs = torch.randn(batch_size, src_len, encoder_hidden_size)
    
    # Forward pass
    output, hidden, cell, attention_weights = decoder(
        input_token, hidden, encoder_outputs, cell
    )
    
    print(f"Input token shape: {input_token.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden shape: {hidden.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
