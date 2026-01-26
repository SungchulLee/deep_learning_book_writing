# Sequence-to-Sequence Models

## Introduction

Sequence-to-Sequence (Seq2Seq) models handle tasks where both input and output are sequences, potentially of different lengths. Introduced by Sutskever et al. (2014), the encoder-decoder architecture revolutionized machine translation and enabled diverse applications from text summarization to conversational AI.

## The Encoder-Decoder Architecture

Seq2Seq consists of two components:

1. **Encoder**: Processes the input sequence and produces a context representation
2. **Decoder**: Generates the output sequence conditioned on the context

```
Input: "How are you?"
         ↓
    [Encoder]
         ↓
    Context Vector
         ↓
    [Decoder]
         ↓
Output: "Comment allez-vous?"
```

## Mathematical Formulation

### Encoder
The encoder processes input sequence $(x_1, x_2, \ldots, x_T)$ and produces hidden states:

$$h_t^{\text{enc}} = \text{EncoderRNN}(x_t, h_{t-1}^{\text{enc}})$$

The final hidden state (or a function of all states) becomes the **context vector**:

$$c = h_T^{\text{enc}} \quad \text{or} \quad c = f(h_1^{\text{enc}}, \ldots, h_T^{\text{enc}})$$

### Decoder
The decoder generates output sequence $(y_1, y_2, \ldots, y_{T'})$ autoregressively:

$$h_t^{\text{dec}} = \text{DecoderRNN}(y_{t-1}, h_{t-1}^{\text{dec}})$$
$$P(y_t | y_{<t}, x) = \text{softmax}(W_o h_t^{\text{dec}} + b_o)$$

The decoder is initialized with the context: $h_0^{\text{dec}} = c$

## Basic Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """LSTM encoder for Seq2Seq."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Source sequence (batch, src_len)
        
        Returns:
            outputs: Encoder hidden states (batch, src_len, hidden)
            (h_n, c_n): Final states (num_layers, batch, hidden)
        """
        embedded = self.dropout(self.embedding(x))
        outputs, (h_n, c_n) = self.lstm(embedded)
        return outputs, (h_n, c_n)


class Decoder(nn.Module):
    """LSTM decoder for Seq2Seq."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden):
        """
        Args:
            x: Target tokens (batch, 1) or (batch, tgt_len)
            hidden: (h, c) from encoder or previous decoder step
        
        Returns:
            output: Logits (batch, seq_len, vocab_size)
            hidden: Updated hidden state
        """
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden


class Seq2Seq(nn.Module):
    """Complete Seq2Seq model."""
    
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src: Source sequence (batch, src_len)
            tgt: Target sequence (batch, tgt_len)
            teacher_forcing_ratio: Probability of using ground truth
        
        Returns:
            outputs: Decoder outputs (batch, tgt_len, vocab_size)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.fc.out_features
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=self.device)
        
        # Encode source
        _, hidden = self.encoder(src)
        
        # First decoder input is <SOS> token
        decoder_input = tgt[:, 0:1]  # (batch, 1)
        
        for t in range(1, tgt_len):
            # Decode one step
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t, :] = output.squeeze(1)
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            if teacher_force:
                decoder_input = tgt[:, t:t+1]
            else:
                decoder_input = output.argmax(dim=-1)
        
        return outputs
```

## Teacher Forcing

During training, the decoder can either use:
1. **Ground truth**: Feed actual target tokens (teacher forcing)
2. **Predictions**: Feed model's own predictions

**Teacher forcing** accelerates training but creates **exposure bias**—the model never learns to recover from its own mistakes.

```python
def train_with_scheduled_sampling(model, src, tgt, epoch, max_epochs):
    """
    Gradually decrease teacher forcing over training.
    """
    # Linear decay from 1.0 to 0.0
    teacher_forcing_ratio = 1.0 - (epoch / max_epochs)
    
    outputs = model(src, tgt, teacher_forcing_ratio)
    return outputs
```

## Inference: Greedy Decoding

```python
def greedy_decode(model, src, max_length, sos_idx, eos_idx):
    """
    Generate sequence using greedy decoding.
    """
    model.eval()
    
    with torch.no_grad():
        # Encode source
        _, hidden = model.encoder(src)
        
        # Start with <SOS>
        decoder_input = torch.tensor([[sos_idx]], device=src.device)
        
        outputs = []
        for _ in range(max_length):
            output, hidden = model.decoder(decoder_input, hidden)
            
            # Greedy selection
            predicted = output.argmax(dim=-1)
            outputs.append(predicted.item())
            
            # Stop at <EOS>
            if predicted.item() == eos_idx:
                break
            
            decoder_input = predicted
    
    return outputs
```

## Inference: Beam Search

Beam search maintains multiple hypotheses for better quality:

```python
def beam_search(model, src, max_length, sos_idx, eos_idx, beam_width=5):
    """
    Generate sequence using beam search.
    """
    model.eval()
    
    with torch.no_grad():
        # Encode source
        _, hidden = model.encoder(src)
        
        # Initialize beam
        # Each hypothesis: (sequence, score, hidden_state)
        beams = [([sos_idx], 0.0, hidden)]
        completed = []
        
        for _ in range(max_length):
            candidates = []
            
            for seq, score, hidden in beams:
                if seq[-1] == eos_idx:
                    completed.append((seq, score))
                    continue
                
                # Decode one step
                decoder_input = torch.tensor([[seq[-1]]], device=src.device)
                output, new_hidden = model.decoder(decoder_input, hidden)
                
                # Get log probabilities
                log_probs = F.log_softmax(output.squeeze(1), dim=-1)
                
                # Get top-k candidates
                topk_probs, topk_ids = log_probs.topk(beam_width)
                
                for prob, idx in zip(topk_probs[0], topk_ids[0]):
                    new_seq = seq + [idx.item()]
                    new_score = score + prob.item()
                    candidates.append((new_seq, new_score, new_hidden))
            
            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]
            
            # Early stopping if all beams completed
            if len(beams) == 0:
                break
        
        # Return best sequence
        all_seqs = completed + [(seq, score) for seq, score, _ in beams]
        all_seqs.sort(key=lambda x: x[1] / len(x[0]), reverse=True)  # Length normalize
        
        return all_seqs[0][0]
```

## Handling Variable Lengths

For batched training with variable-length sequences:

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EncoderWithPacking(nn.Module):
    """Encoder that handles variable-length sequences efficiently."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x, lengths):
        """
        Args:
            x: Padded source sequences (batch, max_len)
            lengths: Actual lengths of each sequence
        """
        embedded = self.embedding(x)
        
        # Pack for efficient computation
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        outputs, (h_n, c_n) = self.lstm(packed)
        
        # Unpack
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        return outputs, (h_n, c_n)
```

## Bidirectional Encoder

Using a bidirectional encoder improves context representation:

```python
class BidirectionalEncoder(nn.Module):
    """Bidirectional LSTM encoder."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Project bidirectional hidden to decoder size
        self.fc_h = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_c = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (h_n, c_n) = self.lstm(embedded)
        
        # Combine forward and backward final states
        # h_n: (num_layers * 2, batch, hidden)
        batch_size = x.size(0)
        
        # Concatenate directions and project
        h_combined = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, hidden*2)
        c_combined = torch.cat([c_n[-2], c_n[-1]], dim=-1)
        
        h_projected = torch.tanh(self.fc_h(h_combined))  # (batch, hidden)
        c_projected = torch.tanh(self.fc_c(c_combined))
        
        return outputs, (h_projected.unsqueeze(0), c_projected.unsqueeze(0))
```

## Training Loop

```python
def train_seq2seq(model, train_loader, optimizer, criterion, device, 
                  clip=1.0, teacher_forcing_ratio=0.5):
    """
    Train Seq2Seq for one epoch.
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (src, tgt) in enumerate(train_loader):
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(src, tgt, teacher_forcing_ratio)
        
        # Reshape for loss computation
        # outputs: (batch, tgt_len, vocab_size)
        # tgt: (batch, tgt_len)
        output_dim = outputs.shape[-1]
        
        # Ignore first token (<SOS>) in target
        outputs = outputs[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = criterion(outputs, tgt)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)
```

## Common Issues and Solutions

### Problem: Information Bottleneck
The context vector must compress the entire source sequence, limiting performance on long sequences.

**Solution**: Use attention mechanisms (next section).

### Problem: Exposure Bias
Model trained with teacher forcing struggles with its own errors at inference.

**Solution**: Scheduled sampling, beam search, or reinforcement learning fine-tuning.

### Problem: Unknown Words
Fixed vocabulary cannot handle all words.

**Solution**: Subword tokenization (BPE, WordPiece) or copy mechanisms.

## Summary

Seq2Seq models enable sequence transduction through:

1. **Encoder**: Compresses source sequence into context
2. **Decoder**: Generates target autoregressively from context
3. **Teacher forcing**: Accelerates training using ground truth

Key components:
- Bidirectional encoders capture full source context
- Beam search improves generation quality
- Packed sequences handle variable lengths efficiently

The fundamental limitation—compressing all information into a fixed-size vector—motivates attention mechanisms covered in the next section.
