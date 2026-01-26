# Practical Applications of LSTM, GRU, and Seq2Seq Models

## Introduction

LSTM and GRU networks excel at tasks involving sequential data with temporal dependencies. When combined with encoder-decoder architectures and attention mechanisms, they power diverse applications from machine translation to speech recognition. This section covers practical implementations across domains.

---

## Natural Language Processing

### Text Generation

Character-level and word-level language models for creative text generation.

```python
import torch
import torch.nn as nn

class CharLSTM(nn.Module):
    """Character-level language model for text generation."""
    
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=128, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        logits = self.fc(output)
        return logits, hidden
    
    def generate(self, start_chars, char_to_idx, idx_to_char, 
                 length=100, temperature=1.0, device='cpu'):
        """Generate text given starting characters."""
        self.eval()
        
        # Convert start characters to indices
        input_seq = torch.tensor([[char_to_idx[c] for c in start_chars]], 
                                  device=device)
        hidden = None
        generated = list(start_chars)
        
        with torch.no_grad():
            # Process start sequence
            _, hidden = self.forward(input_seq, hidden)
            current_char = input_seq[:, -1:]
            
            for _ in range(length):
                logits, hidden = self.forward(current_char, hidden)
                
                # Apply temperature
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                
                # Sample from distribution
                next_idx = torch.multinomial(probs, num_samples=1)
                next_char = idx_to_char[next_idx.item()]
                generated.append(next_char)
                
                current_char = next_idx
        
        return ''.join(generated)
```

### Sentiment Analysis

```python
class SentimentClassifier(nn.Module):
    """Bidirectional LSTM for sentiment classification."""
    
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256, 
                 num_classes=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, bidirectional=True,
                           batch_first=True, dropout=dropout, num_layers=2)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        return self.fc(context)
```

### Named Entity Recognition

```python
class NERTagger(nn.Module):
    """Sequence labeling for NER with BiLSTM-CRF."""
    
    def __init__(self, vocab_size, tag_size, embed_dim=100, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, bidirectional=True,
                           batch_first=True)
        self.hidden2tag = nn.Linear(hidden_size * 2, tag_size)
        
        # CRF layer for structured prediction
        self.transitions = nn.Parameter(torch.randn(tag_size, tag_size))
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        emissions = self.hidden2tag(lstm_out)
        return emissions
    
    def viterbi_decode(self, emissions):
        """Decode best tag sequence using Viterbi algorithm."""
        seq_len, tag_size = emissions.shape
        
        # Initialize
        scores = emissions[0]
        backpointers = []
        
        for t in range(1, seq_len):
            scores_with_trans = scores.unsqueeze(1) + self.transitions
            best_scores, best_tags = scores_with_trans.max(dim=0)
            scores = best_scores + emissions[t]
            backpointers.append(best_tags)
        
        # Backtrack
        best_path = [scores.argmax().item()]
        for bp in reversed(backpointers):
            best_path.append(bp[best_path[-1]].item())
        
        return list(reversed(best_path))
```

---

## Machine Translation

The original and most prominent seq2seq application.

### Full NMT Architecture

```python
class Attention(nn.Module):
    """Bahdanau-style additive attention."""
    
    def __init__(self, enc_hidden_size, dec_hidden_size, attn_size=256):
        super().__init__()
        self.W_enc = nn.Linear(enc_hidden_size, attn_size, bias=False)
        self.W_dec = nn.Linear(dec_hidden_size, attn_size, bias=False)
        self.v = nn.Linear(attn_size, 1, bias=False)
    
    def forward(self, encoder_outputs, decoder_hidden, mask=None):
        """
        Args:
            encoder_outputs: (batch, src_len, enc_hidden)
            decoder_hidden: (batch, dec_hidden)
            mask: (batch, src_len) - True for padding positions
        """
        src_len = encoder_outputs.size(1)
        
        # Project encoder and decoder states
        enc_proj = self.W_enc(encoder_outputs)  # (batch, src_len, attn_size)
        dec_proj = self.W_dec(decoder_hidden).unsqueeze(1)  # (batch, 1, attn_size)
        
        # Compute attention scores
        scores = self.v(torch.tanh(enc_proj + dec_proj)).squeeze(-1)  # (batch, src_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attn_weights


class NMTModel(nn.Module):
    """Neural Machine Translation with attention."""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=256, 
                 hidden_size=512, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Bridge: project bidirectional to unidirectional
        self.bridge_h = nn.Linear(hidden_size * 2, hidden_size)
        self.bridge_c = nn.Linear(hidden_size * 2, hidden_size)
        
        # Decoder
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.attention = Attention(hidden_size * 2, hidden_size)
        self.decoder = nn.LSTM(
            embed_dim + hidden_size * 2, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        
        # Output
        self.output = nn.Linear(hidden_size + hidden_size * 2 + embed_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, src, src_lengths=None):
        """Encode source sequence."""
        embedded = self.dropout(self.src_embedding(src))
        
        if src_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        outputs, (h_n, c_n) = self.encoder(embedded)
        
        if src_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # Combine bidirectional states for each layer
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)
        c_n = c_n.view(self.num_layers, 2, -1, self.hidden_size)
        
        h_n = torch.cat([h_n[:, 0], h_n[:, 1]], dim=-1)
        c_n = torch.cat([c_n[:, 0], c_n[:, 1]], dim=-1)
        
        h_n = torch.tanh(self.bridge_h(h_n))
        c_n = torch.tanh(self.bridge_c(c_n))
        
        return outputs, (h_n, c_n)
    
    def decode_step(self, tgt_token, hidden, encoder_outputs, src_mask=None):
        """Single decoding step."""
        embedded = self.dropout(self.tgt_embedding(tgt_token))
        
        # Attention
        h_top = hidden[0][-1]  # Top layer hidden state
        context, attn_weights = self.attention(encoder_outputs, h_top, src_mask)
        
        # Decoder input: embedding + context
        decoder_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, hidden = self.decoder(decoder_input, hidden)
        
        # Output projection
        output = torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=-1)
        logits = self.output(output)
        
        return logits, hidden, attn_weights
    
    def forward(self, src, tgt, src_lengths=None, teacher_forcing_ratio=1.0):
        """
        Training forward pass with teacher forcing.
        """
        batch_size, tgt_len = tgt.shape
        
        # Encode
        encoder_outputs, hidden = self.encode(src, src_lengths)
        
        # Create source mask for attention
        src_mask = (src == 0) if src_lengths is not None else None
        
        # Decode
        outputs = []
        input_token = tgt[:, 0:1]  # <SOS> token
        
        for t in range(1, tgt_len):
            logits, hidden, _ = self.decode_step(input_token, hidden, 
                                                  encoder_outputs, src_mask)
            outputs.append(logits)
            
            # Teacher forcing
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tgt[:, t:t+1]
            else:
                input_token = logits.argmax(dim=-1, keepdim=True)
        
        return torch.stack(outputs, dim=1)
```

### Training Considerations for NMT

- **Subword tokenization**: BPE or SentencePiece for open vocabulary
- **Backtranslation**: Augment data by translating target → source
- **Label smoothing**: Prevent overconfident predictions
- **Checkpoint averaging**: Average last N checkpoints for better generalization

---

## Text Summarization

### Abstractive Summarization with Copy Mechanism

```python
class SummarizationModel(nn.Module):
    """Abstractive summarization with pointer-generator network."""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_size, 
                               batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(embed_dim + hidden_size * 2, hidden_size,
                               batch_first=True)
        
        # Attention
        self.attention = Attention(hidden_size * 2, hidden_size)
        
        # Copy mechanism
        self.copy_gate = nn.Linear(hidden_size * 3 + embed_dim, 1)
        self.output = nn.Linear(hidden_size * 3, vocab_size)
    
    def forward(self, src, tgt, src_extended=None, max_oov=0):
        """
        Args:
            src: Source document tokens
            tgt: Target summary tokens  
            src_extended: Source with OOV tokens mapped to extended vocab indices
            max_oov: Maximum number of OOV tokens in batch
        """
        batch_size = src.size(0)
        
        # Encode
        enc_embedded = self.embedding(src)
        enc_outputs, (h_n, c_n) = self.encoder(enc_embedded)
        
        # Initialize decoder state
        h_n = h_n.view(1, 2, batch_size, -1)
        c_n = c_n.view(1, 2, batch_size, -1)
        hidden = (torch.cat([h_n[:, 0], h_n[:, 1]], dim=-1),
                  torch.cat([c_n[:, 0], c_n[:, 1]], dim=-1))
        
        outputs = []
        
        for t in range(tgt.size(1) - 1):
            dec_input = self.embedding(tgt[:, t:t+1])
            
            # Attention
            context, attn_weights = self.attention(enc_outputs, hidden[0][-1])
            
            # Decoder step
            decoder_input = torch.cat([dec_input, context.unsqueeze(1)], dim=-1)
            dec_output, hidden = self.decoder(decoder_input, hidden)
            
            # Copy gate: probability of generating vs copying
            gate_input = torch.cat([dec_output.squeeze(1), context, 
                                    dec_input.squeeze(1)], dim=-1)
            p_gen = torch.sigmoid(self.copy_gate(gate_input))
            
            # Vocabulary distribution
            vocab_input = torch.cat([dec_output.squeeze(1), context], dim=-1)
            vocab_dist = torch.softmax(self.output(vocab_input), dim=-1)
            
            # Final distribution with copy
            if src_extended is not None and max_oov > 0:
                # Extend vocabulary distribution for OOV tokens
                extended_vocab = torch.zeros(batch_size, self.output.out_features + max_oov,
                                            device=src.device)
                extended_vocab[:, :self.output.out_features] = p_gen * vocab_dist
                
                # Add copy probabilities
                copy_dist = (1 - p_gen) * attn_weights
                extended_vocab.scatter_add_(1, src_extended, copy_dist)
                
                outputs.append(extended_vocab)
            else:
                outputs.append(p_gen * vocab_dist + (1 - p_gen) * attn_weights)
        
        return torch.stack(outputs, dim=1)
```

### Extractive Summarization

```python
class ExtractiveSummarizer(nn.Module):
    """Select important sentences from document."""
    
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Sentence encoder
        self.sentence_encoder = nn.LSTM(embed_dim, hidden_size, 
                                        bidirectional=True, batch_first=True)
        
        # Document encoder
        self.document_encoder = nn.LSTM(hidden_size * 2, hidden_size,
                                        bidirectional=True, batch_first=True)
        
        # Sentence selection
        self.selector = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, sentences):
        """
        Args:
            sentences: List of (batch, sent_len) tensors
        """
        sentence_reprs = []
        
        for sent in sentences:
            embedded = self.embedding(sent)
            _, (h_n, _) = self.sentence_encoder(embedded)
            sent_repr = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            sentence_reprs.append(sent_repr)
        
        # Document representation
        doc_input = torch.stack(sentence_reprs, dim=1)
        doc_output, _ = self.document_encoder(doc_input)
        
        # Score each sentence
        combined = torch.cat([doc_input, doc_output], dim=-1)
        scores = self.selector(combined).squeeze(-1)
        
        return torch.sigmoid(scores)
```

---

## Time Series and Finance

### Stock Price Prediction

```python
class StockPredictor(nn.Module):
    """GRU-based stock price predictor with multiple features."""
    
    def __init__(self, input_features=5, hidden_size=64, num_layers=2, 
                 forecast_horizon=5):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_size, num_layers,
                         batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, forecast_horizon)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features) - OHLCV data
        Returns:
            predictions: (batch, forecast_horizon)
        """
        output, _ = self.gru(x)
        return self.fc(output[:, -1, :])


class MultiStepForecaster(nn.Module):
    """Encoder-decoder for multi-step time series forecasting."""
    
    def __init__(self, input_features, hidden_size=64, forecast_steps=10):
        super().__init__()
        self.forecast_steps = forecast_steps
        
        self.encoder = nn.LSTM(input_features, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(1, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, x, target=None):
        """
        Args:
            x: (batch, history_len, features) - historical data
            target: (batch, forecast_steps) - for teacher forcing
        """
        # Encode history
        _, hidden = self.encoder(x)
        
        # Decode future
        predictions = []
        decoder_input = x[:, -1:, 0:1]  # Last value
        
        for t in range(self.forecast_steps):
            output, hidden = self.decoder(decoder_input, hidden)
            pred = self.output(output)
            predictions.append(pred.squeeze(-1))
            
            if target is not None and t < self.forecast_steps - 1:
                decoder_input = target[:, t:t+1].unsqueeze(-1)
            else:
                decoder_input = pred
        
        return torch.cat(predictions, dim=-1)
```

### Anomaly Detection

```python
class AnomalyDetectorLSTM(nn.Module):
    """LSTM autoencoder for time series anomaly detection."""
    
    def __init__(self, input_size, hidden_size=64, latent_size=32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.enc_fc = nn.Linear(hidden_size, latent_size)
        
        # Decoder
        self.dec_fc = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Encode
        _, (hidden, _) = self.encoder(x)
        latent = self.enc_fc(hidden.squeeze(0))
        
        # Decode
        decoder_init = self.dec_fc(latent).unsqueeze(0)
        decoder_input = decoder_init.permute(1, 0, 2).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(decoder_input)
        
        # Reconstruct
        return self.output(decoded)
    
    def compute_anomaly_score(self, x, threshold=None):
        """Compute reconstruction error as anomaly score."""
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = ((x - reconstructed) ** 2).mean(dim=(1, 2))
            
            if threshold is not None:
                is_anomaly = mse > threshold
                return mse, is_anomaly
            return mse
```

---

## Conversational AI

### Dialogue Response Generation

```python
class HierarchicalDialogueModel(nn.Module):
    """Seq2seq for multi-turn dialogue with hierarchical encoding."""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Hierarchical encoder
        self.utterance_encoder = nn.LSTM(
            embed_dim, hidden_size, batch_first=True, bidirectional=True
        )
        self.context_encoder = nn.LSTM(
            hidden_size * 2, hidden_size, batch_first=True
        )
        
        # Decoder with attention
        self.attention = Attention(hidden_size, hidden_size)
        self.decoder = nn.LSTM(
            embed_dim + hidden_size, hidden_size, batch_first=True
        )
        self.output = nn.Linear(hidden_size * 2, vocab_size)
    
    def encode_context(self, turns):
        """
        Encode multi-turn dialogue context.
        
        Args:
            turns: List of (batch, seq_len) tensors for each turn
        """
        turn_representations = []
        
        for turn in turns:
            embedded = self.embedding(turn)
            _, (h_n, _) = self.utterance_encoder(embedded)
            turn_repr = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            turn_representations.append(turn_repr)
        
        # Encode turn sequence
        turn_sequence = torch.stack(turn_representations, dim=1)
        context_output, hidden = self.context_encoder(turn_sequence)
        
        return context_output, hidden
    
    def decode(self, context_output, hidden, response):
        """Generate response tokens."""
        outputs = []
        
        for t in range(response.size(1) - 1):
            token = response[:, t:t+1]
            embedded = self.embedding(token)
            
            # Attention over context turns
            attn_context, _ = self.attention(context_output, hidden[0][-1])
            
            # Decode
            decoder_input = torch.cat([embedded, attn_context.unsqueeze(1)], dim=-1)
            output, hidden = self.decoder(decoder_input, hidden)
            
            # Project to vocabulary
            logits = self.output(torch.cat([output.squeeze(1), attn_context], dim=-1))
            outputs.append(logits)
        
        return torch.stack(outputs, dim=1)


class PersonaDialogue(nn.Module):
    """Dialogue model conditioned on persona description."""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Persona encoder
        self.persona_encoder = nn.LSTM(
            embed_dim, hidden_size, batch_first=True, bidirectional=True
        )
        
        # Context encoder
        self.context_encoder = nn.LSTM(
            embed_dim, hidden_size, batch_first=True, bidirectional=True
        )
        
        # Decoder attends to both persona and context
        self.persona_attention = Attention(hidden_size * 2, hidden_size)
        self.context_attention = Attention(hidden_size * 2, hidden_size)
        
        self.decoder = nn.LSTM(
            embed_dim + hidden_size * 4, hidden_size, batch_first=True
        )
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, persona_sentences, context, response):
        """
        Args:
            persona_sentences: (batch, num_sentences, max_len)
            context: (batch, context_len)
            response: (batch, response_len)
        """
        # Encode persona (concatenate all sentences)
        batch_size = persona_sentences.size(0)
        persona_flat = persona_sentences.view(batch_size, -1)
        persona_embedded = self.embedding(persona_flat)
        persona_output, _ = self.persona_encoder(persona_embedded)
        
        # Encode context
        context_embedded = self.embedding(context)
        context_output, hidden = self.context_encoder(context_embedded)
        
        # Compress hidden for decoder
        hidden = (hidden[0][-2:].mean(0, keepdim=True),
                  hidden[1][-2:].mean(0, keepdim=True))
        
        # Decode with dual attention
        outputs = []
        for t in range(response.size(1) - 1):
            token = response[:, t:t+1]
            embedded = self.embedding(token)
            
            persona_ctx, _ = self.persona_attention(persona_output, hidden[0][-1])
            context_ctx, _ = self.context_attention(context_output, hidden[0][-1])
            
            decoder_input = torch.cat([embedded, persona_ctx.unsqueeze(1), 
                                       context_ctx.unsqueeze(1)], dim=-1)
            output, hidden = self.decoder(decoder_input, hidden)
            
            logits = self.output(output.squeeze(1))
            outputs.append(logits)
        
        return torch.stack(outputs, dim=1)
```

---

## Question Answering

### Reading Comprehension

```python
class QAModel(nn.Module):
    """Generative QA: generate answer from passage and question."""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Question encoder
        self.question_encoder = nn.LSTM(
            embed_dim, hidden_size, batch_first=True, bidirectional=True
        )
        
        # Question-aware passage encoder
        self.passage_attention = Attention(hidden_size * 2, hidden_size * 2)
        self.passage_encoder = nn.LSTM(
            embed_dim + hidden_size * 2, hidden_size, 
            batch_first=True, bidirectional=True
        )
        
        # Answer decoder
        self.answer_attention = Attention(hidden_size * 2, hidden_size)
        self.decoder = nn.LSTM(
            embed_dim + hidden_size * 2, hidden_size, batch_first=True
        )
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, question, passage, answer=None):
        # Encode question
        q_embedded = self.embedding(question)
        q_outputs, _ = self.question_encoder(q_embedded)
        q_summary = q_outputs.mean(dim=1)  # Simple pooling
        
        # Question-aware passage encoding
        p_embedded = self.embedding(passage)
        
        # Compute attention for each passage position
        batch_size, p_len, _ = p_embedded.shape
        q_context = q_summary.unsqueeze(1).expand(-1, p_len, -1)
        p_input = torch.cat([p_embedded, q_context], dim=-1)
        
        p_outputs, hidden = self.passage_encoder(p_input)
        
        # Decode answer
        if answer is not None:
            outputs = []
            for t in range(answer.size(1) - 1):
                token = answer[:, t:t+1]
                embedded = self.embedding(token)
                
                context, _ = self.answer_attention(p_outputs, hidden[0][-1])
                decoder_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
                output, hidden = self.decoder(decoder_input, 
                                             (hidden[0][-1:], hidden[1][-1:]))
                
                logits = self.output(output.squeeze(1))
                outputs.append(logits)
            
            return torch.stack(outputs, dim=1)
        
        return p_outputs, hidden
```

---

## Speech and Audio

### Speech Recognition

```python
class ASRModel(nn.Module):
    """End-to-end speech recognition with attention."""
    
    def __init__(self, input_dim=80, vocab_size=5000, hidden_size=512):
        super().__init__()
        
        # Convolutional frontend for feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Calculate conv output size
        conv_output_size = 32 * (input_dim // 4)
        
        # Audio encoder
        self.encoder = nn.LSTM(
            conv_output_size, hidden_size, num_layers=3,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        
        # Text decoder
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention = Attention(hidden_size * 2, hidden_size)
        self.decoder = nn.LSTM(
            hidden_size * 3, hidden_size, num_layers=2,
            batch_first=True, dropout=0.2
        )
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, audio_features, transcript=None):
        """
        Args:
            audio_features: (batch, time, freq) mel spectrograms
            transcript: (batch, seq_len) text tokens
        """
        batch_size = audio_features.size(0)
        
        # Conv frontend
        x = audio_features.unsqueeze(1)  # (batch, 1, time, freq)
        x = self.conv(x)  # (batch, 32, time/4, freq/4)
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, time/4, 32, freq/4)
        x = x.view(batch_size, x.size(1), -1)  # (batch, time/4, 32*freq/4)
        
        # Encode
        enc_outputs, hidden = self.encoder(x)
        
        # Initialize decoder hidden state
        hidden = (hidden[0][-2:].contiguous(), hidden[1][-2:].contiguous())
        
        if transcript is not None:
            outputs = []
            for t in range(transcript.size(1) - 1):
                token = transcript[:, t:t+1]
                embedded = self.embedding(token)
                
                context, _ = self.attention(enc_outputs, hidden[0][-1])
                decoder_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
                output, hidden = self.decoder(decoder_input, hidden)
                
                logits = self.output(output.squeeze(1))
                outputs.append(logits)
            
            return torch.stack(outputs, dim=1)
        
        return enc_outputs, hidden
```

### Speech Encoder for Downstream Tasks

```python
class SpeechEncoder(nn.Module):
    """Bidirectional LSTM encoder for speech features."""
    
    def __init__(self, input_size=80, hidden_size=256, num_layers=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           bidirectional=True, batch_first=True, dropout=0.2)
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, time_frames, mel_features)
            lengths: actual sequence lengths for packing
        """
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        output, (h_n, c_n) = self.lstm(x)
        
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        return output, (h_n, c_n)
```

---

## Code Generation

### Natural Language to Code

```python
class NL2CodeModel(nn.Module):
    """Generate code from natural language descriptions."""
    
    def __init__(self, nl_vocab_size, code_vocab_size, embed_dim=256, hidden_size=512):
        super().__init__()
        
        # Separate vocabularies for NL and code
        self.nl_embedding = nn.Embedding(nl_vocab_size, embed_dim)
        self.code_embedding = nn.Embedding(code_vocab_size, embed_dim)
        
        self.encoder = nn.LSTM(
            embed_dim, hidden_size, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        
        self.attention = Attention(hidden_size * 2, hidden_size)
        self.decoder = nn.LSTM(
            embed_dim + hidden_size * 2, hidden_size, num_layers=2,
            batch_first=True, dropout=0.3
        )
        
        self.output = nn.Linear(hidden_size, code_vocab_size)
    
    def forward(self, nl_input, code_output=None):
        # Encode natural language
        nl_embedded = self.nl_embedding(nl_input)
        enc_outputs, hidden = self.encoder(nl_embedded)
        
        # Project bidirectional to unidirectional
        hidden = (hidden[0].view(2, 2, -1, hidden[0].size(-1)).mean(1),
                  hidden[1].view(2, 2, -1, hidden[1].size(-1)).mean(1))
        
        if code_output is not None:
            outputs = []
            for t in range(code_output.size(1) - 1):
                token = code_output[:, t:t+1]
                embedded = self.code_embedding(token)
                
                context, _ = self.attention(enc_outputs, hidden[0][-1])
                decoder_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
                output, hidden = self.decoder(decoder_input, hidden)
                
                logits = self.output(output.squeeze(1))
                outputs.append(logits)
            
            return torch.stack(outputs, dim=1)
        
        return enc_outputs, hidden
```

### Code Generation Considerations

- **Syntax constraints**: Ensure generated code is syntactically valid
- **Execution feedback**: Use execution results for reinforcement learning
- **Structured decoding**: Generate AST nodes instead of raw tokens
- **Copy mechanism**: Copy variable names from context

---

## Grammar Error Correction

```python
class GECModel(nn.Module):
    """Grammar Error Correction as seq2seq with high copy bias."""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512):
        super().__init__()
        
        # Shared embedding (source and target are same language)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.encoder = nn.LSTM(
            embed_dim, hidden_size, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        
        self.attention = Attention(hidden_size * 2, hidden_size)
        self.decoder = nn.LSTM(
            embed_dim + hidden_size * 2, hidden_size, num_layers=2,
            batch_first=True, dropout=0.3
        )
        
        # High copy bias since most tokens are unchanged
        self.copy_gate = nn.Linear(hidden_size * 2, 1)
        nn.init.constant_(self.copy_gate.bias, 2.0)  # Bias toward copying
        
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, src, tgt=None):
        # Encode
        embedded = self.embedding(src)
        enc_outputs, hidden = self.encoder(embedded)
        
        # Bridge
        hidden = (hidden[0].view(2, 2, -1, hidden[0].size(-1))[:, -1].contiguous(),
                  hidden[1].view(2, 2, -1, hidden[1].size(-1))[:, -1].contiguous())
        
        if tgt is not None:
            outputs = []
            for t in range(tgt.size(1) - 1):
                token = tgt[:, t:t+1]
                embedded = self.embedding(token)
                
                context, attn_weights = self.attention(enc_outputs, hidden[0][-1])
                decoder_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
                output, hidden = self.decoder(decoder_input, hidden)
                
                # Copy mechanism with high bias
                p_gen = torch.sigmoid(self.copy_gate(
                    torch.cat([output.squeeze(1), context], dim=-1)
                ))
                
                vocab_dist = torch.softmax(self.output(output.squeeze(1)), dim=-1)
                # Combine with copy distribution from attention
                # ...
                
                outputs.append(vocab_dist)
            
            return torch.stack(outputs, dim=1)
        
        return enc_outputs, hidden
```

---

## Application Comparison

| Application | Model Choice | Key Challenges | Special Techniques |
|-------------|--------------|----------------|-------------------|
| Text Generation | LSTM | Coherence, creativity | Temperature sampling |
| Classification | BiLSTM/BiGRU | Full context | Attention pooling |
| Translation | BiLSTM + Attention | Long-range alignment | Subword tokenization |
| Summarization | Pointer-Generator | Faithfulness, compression | Copy mechanism, coverage |
| Time Series | GRU | Noise, non-stationarity | Multi-step decoder |
| Dialogue | Hierarchical LSTM | Context, persona | Turn-level encoding |
| QA | Question-aware encoder | Evidence selection | Passage attention |
| Speech | Deep BiLSTM | Alignment | CTC loss hybrid |
| Code Gen | LSTM + Copy | Syntax validity | Constrained decoding |
| GEC | LSTM + High copy | Most tokens unchanged | Copy bias initialization |

## Evaluation Metrics

| Application | Primary Metrics |
|-------------|-----------------|
| Translation | BLEU, METEOR, chrF |
| Summarization | ROUGE-1/2/L, BERTScore |
| Dialogue | Perplexity, BLEU, human evaluation |
| QA | Exact Match, F1 |
| GEC | GLEU, M², ERRANT |
| Code Gen | Execution accuracy, BLEU |
| ASR | WER, CER |
| Time Series | MSE, MAE, MAPE |
| Anomaly Detection | Precision, Recall, F1, AUC |

---

## Training Best Practices

### Gradient Management

```python
# Essential for stable training
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

### Learning Rate Schedule

```python
# Start with 1e-3 for Adam, reduce on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)
```

### Regularization

```python
# Dropout between layers: 0.2-0.5
# Weight decay: 1e-5 to 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

### Batch Size and Sequence Length

- **Batch size**: 32-128 typical, adjust based on sequence length
- **Truncate or bucket** sequences by length for efficiency
- Use **gradient accumulation** for effective larger batches

### Teacher Forcing Schedule

```python
def get_teacher_forcing_ratio(epoch, total_epochs):
    """Decay teacher forcing over training."""
    return max(0.5, 1.0 - epoch / total_epochs)
```

---

## Summary

LSTM, GRU, and seq2seq models power diverse applications through:

1. **Flexible architecture**: Adapt encoder/decoder for domain
2. **Attention mechanisms**: Handle variable-length alignment
3. **Copy mechanisms**: Enable direct token transfer
4. **Domain-specific decoding**: Constrained generation for structured outputs

### Key Success Factors

- Appropriate tokenization for the domain
- Task-specific training objectives
- Domain-adapted evaluation metrics
- Sufficient training data (or transfer learning)
- Proper regularization and gradient management

---

## References

1. Graves, A. (2012). Supervised Sequence Labelling with Recurrent Neural Networks. *Springer*.

2. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks. *NeurIPS*.

3. Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR*.

4. See, A., Liu, P. J., & Manning, C. D. (2017). Get To The Point: Summarization with Pointer-Generator Networks. *ACL*.

5. Serban, I. V., et al. (2016). Building End-To-End Dialogue Systems Using Generative Models. *AAAI*.

6. Chan, W., et al. (2016). Listen, Attend and Spell: A Neural Network for Large Vocabulary Conversational Speech Recognition. *ICASSP*.
