"""
Complete Sequence-to-Sequence Models
Combines encoder and decoder into full architectures
"""

import torch
import torch.nn as nn
import random
from encoder import BasicEncoder
from decoder import BasicDecoder, AttentionDecoder


class Seq2Seq(nn.Module):
    """
    Basic Sequence-to-Sequence model without attention
    
    Args:
        encoder: Encoder module
        decoder: Decoder module
        device: Device to run on
    """
    
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5, src_lengths=None):
        """
        Forward pass through the seq2seq model
        
        Args:
            src: Source sequence (batch_size, src_len)
            trg: Target sequence (batch_size, trg_len)
            teacher_forcing_ratio: Probability of using teacher forcing
            src_lengths: Actual lengths of source sequences
            
        Returns:
            outputs: Predictions for each time step (batch_size, trg_len, output_size)
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encode source sequence
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # First input to decoder is <sos> token
        decoder_input = trg[:, 0].unsqueeze(1)
        
        # Decode one token at a time
        for t in range(1, trg_len):
            # Forward pass through decoder
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            
            # Store prediction
            outputs[:, t] = output
            
            # Decide if we use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get predicted token
            top1 = output.argmax(1)
            
            # Next input is either ground truth or predicted token
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs
    
    def generate(self, src, max_len=50, sos_idx=1, eos_idx=2, src_lengths=None):
        """
        Generate sequence using greedy decoding
        
        Args:
            src: Source sequence (batch_size, src_len)
            max_len: Maximum length of generated sequence
            sos_idx: Start of sequence token index
            eos_idx: End of sequence token index
            src_lengths: Actual lengths of source sequences
            
        Returns:
            generated: Generated sequence (batch_size, seq_len)
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # Encode source
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            
            # Start with <sos> token
            decoder_input = torch.full((batch_size, 1), sos_idx, dtype=torch.long).to(self.device)
            
            generated = [decoder_input]
            finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
            
            for _ in range(max_len):
                # Forward pass
                output, hidden, cell = self.decoder(decoder_input, hidden, cell)
                
                # Get predicted token
                predicted = output.argmax(1).unsqueeze(1)
                
                # Update finished sequences
                finished |= (predicted.squeeze(1) == eos_idx)
                
                # Store prediction
                generated.append(predicted)
                
                # Break if all sequences finished
                if finished.all():
                    break
                
                # Next input
                decoder_input = predicted
            
            # Concatenate all predictions
            generated = torch.cat(generated, dim=1)
            
        return generated


class Seq2SeqAttention(nn.Module):
    """
    Sequence-to-Sequence model with attention mechanism
    
    Args:
        encoder: Encoder module
        decoder: Decoder module with attention
        device: Device to run on
        pad_idx: Padding token index
    """
    
    def __init__(self, encoder, decoder, device, pad_idx=0):
        super(Seq2SeqAttention, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx
        
    def create_mask(self, src):
        """Create mask for padding tokens"""
        mask = (src != self.pad_idx)
        return mask
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5, src_lengths=None):
        """
        Forward pass through the seq2seq model with attention
        
        Args:
            src: Source sequence (batch_size, src_len)
            trg: Target sequence (batch_size, trg_len)
            teacher_forcing_ratio: Probability of using teacher forcing
            src_lengths: Actual lengths of source sequences
            
        Returns:
            outputs: Predictions for each time step (batch_size, trg_len, output_size)
            attentions: Attention weights for each time step (batch_size, trg_len, src_len)
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # Tensors to store outputs and attention weights
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device)
        
        # Create mask for source sequence
        mask = self.create_mask(src)
        
        # Encode source sequence
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # First input to decoder is <sos> token
        decoder_input = trg[:, 0].unsqueeze(1)
        
        # Decode one token at a time
        for t in range(1, trg_len):
            # Forward pass through decoder with attention
            output, hidden, cell, attention_weights = self.decoder(
                decoder_input, hidden, encoder_outputs, cell, mask
            )
            
            # Store predictions and attention weights
            outputs[:, t] = output
            attentions[:, t] = attention_weights
            
            # Decide if we use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get predicted token
            top1 = output.argmax(1)
            
            # Next input is either ground truth or predicted token
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs, attentions
    
    def generate(self, src, max_len=50, sos_idx=1, eos_idx=2, src_lengths=None):
        """
        Generate sequence using greedy decoding with attention
        
        Args:
            src: Source sequence (batch_size, src_len)
            max_len: Maximum length of generated sequence
            sos_idx: Start of sequence token index
            eos_idx: End of sequence token index
            src_lengths: Actual lengths of source sequences
            
        Returns:
            generated: Generated sequence (batch_size, seq_len)
            all_attentions: Attention weights for each step (batch_size, seq_len, src_len)
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # Create mask
            mask = self.create_mask(src)
            
            # Encode source
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            
            # Start with <sos> token
            decoder_input = torch.full((batch_size, 1), sos_idx, dtype=torch.long).to(self.device)
            
            generated = [decoder_input]
            all_attentions = []
            finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
            
            for _ in range(max_len):
                # Forward pass
                output, hidden, cell, attention_weights = self.decoder(
                    decoder_input, hidden, encoder_outputs, cell, mask
                )
                
                # Store attention weights
                all_attentions.append(attention_weights.unsqueeze(1))
                
                # Get predicted token
                predicted = output.argmax(1).unsqueeze(1)
                
                # Update finished sequences
                finished |= (predicted.squeeze(1) == eos_idx)
                
                # Store prediction
                generated.append(predicted)
                
                # Break if all sequences finished
                if finished.all():
                    break
                
                # Next input
                decoder_input = predicted
            
            # Concatenate all predictions and attentions
            generated = torch.cat(generated, dim=1)
            all_attentions = torch.cat(all_attentions, dim=1)
            
        return generated, all_attentions
    
    def beam_search(self, src, beam_width=5, max_len=50, sos_idx=1, eos_idx=2, 
                   src_lengths=None, length_penalty=0.6):
        """
        Generate sequence using beam search
        
        Args:
            src: Source sequence (1, src_len) - single sequence only
            beam_width: Number of beams
            max_len: Maximum length of generated sequence
            sos_idx: Start of sequence token index
            eos_idx: End of sequence token index
            src_lengths: Actual length of source sequence
            length_penalty: Length normalization penalty (alpha)
            
        Returns:
            best_sequence: Best generated sequence
            best_score: Score of best sequence
        """
        self.eval()
        
        with torch.no_grad():
            # Encode source
            mask = self.create_mask(src)
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            
            # Initialize beams
            # Each beam: (sequence, score, hidden, cell)
            beams = [(torch.full((1, 1), sos_idx, dtype=torch.long).to(self.device), 
                     0.0, hidden, cell)]
            completed = []
            
            for step in range(max_len):
                candidates = []
                
                for sequence, score, h, c in beams:
                    # Skip if sequence already ended
                    if sequence[0, -1].item() == eos_idx:
                        completed.append((sequence, score, h, c))
                        continue
                    
                    # Get last token
                    last_token = sequence[:, -1].unsqueeze(1)
                    
                    # Forward pass
                    output, new_h, new_c, _ = self.decoder(
                        last_token, h, encoder_outputs, c, mask
                    )
                    
                    # Get top k predictions
                    log_probs = torch.log_softmax(output, dim=1)
                    top_probs, top_indices = log_probs.topk(beam_width)
                    
                    # Create new candidates
                    for prob, idx in zip(top_probs[0], top_indices[0]):
                        new_sequence = torch.cat([sequence, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                        new_score = score + prob.item()
                        candidates.append((new_sequence, new_score, new_h, new_c))
                
                # Select top beam_width candidates
                candidates.sort(key=lambda x: x[1] / (len(x[0][0]) ** length_penalty), reverse=True)
                beams = candidates[:beam_width]
                
                # Stop if all beams completed
                if len(beams) == 0:
                    break
            
            # Add remaining beams to completed
            completed.extend(beams)
            
            # Select best sequence
            if completed:
                best = max(completed, key=lambda x: x[1] / (len(x[0][0]) ** length_penalty))
                return best[0], best[1]
            else:
                return beams[0][0], beams[0][1]


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    input_vocab_size = 10000
    output_vocab_size = 10000
    embedding_dim = 256
    hidden_size = 512
    num_layers = 2
    dropout = 0.1
    
    # Create encoder and decoder
    encoder = BasicEncoder(
        input_size=input_vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=True,
        rnn_type='LSTM'
    ).to(device)
    
    decoder = AttentionDecoder(
        output_size=output_vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size * 2,  # *2 because encoder is bidirectional
        encoder_hidden_size=hidden_size * 2,
        num_layers=num_layers,
        dropout=dropout,
        rnn_type='LSTM'
    ).to(device)
    
    # Create seq2seq model
    model = Seq2SeqAttention(encoder, decoder, device).to(device)
    
    # Sample data
    batch_size = 32
    src_len = 20
    trg_len = 25
    
    src = torch.randint(3, input_vocab_size, (batch_size, src_len)).to(device)
    trg = torch.randint(3, output_vocab_size, (batch_size, trg_len)).to(device)
    
    # Forward pass
    outputs, attentions = model(src, trg, teacher_forcing_ratio=0.5)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {src.shape}")
    print(f"Target shape: {trg.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Attention shape: {attentions.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
