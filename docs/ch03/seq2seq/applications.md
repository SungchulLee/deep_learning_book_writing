# Applications of Sequence-to-Sequence Models

Sequence-to-sequence models have revolutionized numerous domains requiring the transformation of input sequences into output sequences. This comprehensive document explores key applications, their unique challenges, implementation considerations, and provides deep theoretical insights into why certain architectural choices work for specific domains.

## Introduction

LSTM and GRU networks excel at tasks involving sequential data with temporal dependencies. When combined with encoder-decoder architectures and attention mechanisms, they power diverse applications from machine translation to speech recognition. The key insight underlying all these applications is the **information bottleneck principle**: the encoder compresses variable-length input into a fixed representation that captures task-relevant features, while the decoder reconstructs the appropriate output sequence.

---

## Natural Language Processing Foundations

Before diving into specific applications, we establish foundational NLP models that underpin many seq2seq systems.

### Character-Level Language Models

Character-level models offer fine-grained control over generation and naturally handle out-of-vocabulary words.

```python
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
import math
from collections import Counter


class CharLSTM(nn.Module):
    """
    Character-level language model for text generation.
    
    Key insight: Character-level models capture morphological patterns
    and can generate novel words, making them suitable for creative
    text generation and handling rare words in translation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass returning logits and hidden state."""
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        logits = self.fc(output)
        return logits, hidden
    
    def generate(
        self,
        start_chars: str,
        char_to_idx: Dict[str, int],
        idx_to_char: Dict[int, str],
        length: int = 100,
        temperature: float = 1.0,
        device: torch.device = None
    ) -> str:
        """
        Generate text given starting characters.
        
        Temperature controls randomness:
        - temperature < 1: More deterministic, repetitive
        - temperature = 1: Standard sampling
        - temperature > 1: More random, creative
        """
        device = device or torch.device('cpu')
        self.eval()
        
        input_seq = torch.tensor(
            [[char_to_idx[c] for c in start_chars]], 
            device=device
        )
        hidden = None
        generated = list(start_chars)
        
        with torch.no_grad():
            _, hidden = self.forward(input_seq, hidden)
            current_char = input_seq[:, -1:]
            
            for _ in range(length):
                logits, hidden = self.forward(current_char, hidden)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)
                next_char = idx_to_char[next_idx.item()]
                generated.append(next_char)
                current_char = next_idx
        
        return ''.join(generated)
```

### Sentiment Analysis with Attention

```python
class SentimentClassifier(nn.Module):
    """
    Bidirectional LSTM for sentiment classification.
    
    Deep insight: Attention pooling outperforms simple last-hidden-state
    or mean pooling because sentiment signals can appear anywhere in text.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, bidirectional=True,
            batch_first=True, dropout=dropout, num_layers=2
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        embedded = self.embedding(x)
        
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        lstm_out, _ = self.lstm(embedded)
        
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)
```

### Named Entity Recognition with CRF

```python
class NERTagger(nn.Module):
    """
    Sequence labeling for NER with BiLSTM-CRF.
    
    Deep insight: The CRF layer models label dependencies (e.g., I-PER must
    follow B-PER, not B-LOC). The transition matrix learns valid sequences.
    """
    
    def __init__(self, vocab_size: int, tag_size: int, embed_dim: int = 100, hidden_size: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_size * 2, tag_size)
        self.transitions = nn.Parameter(torch.randn(tag_size, tag_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.hidden2tag(lstm_out)
    
    def viterbi_decode(self, emissions: torch.Tensor) -> List[int]:
        """Decode best tag sequence using Viterbi algorithm."""
        seq_len, tag_size = emissions.shape
        scores = emissions[0]
        backpointers = []
        
        for t in range(1, seq_len):
            scores_with_trans = scores.unsqueeze(1) + self.transitions
            best_scores, best_tags = scores_with_trans.max(dim=0)
            scores = best_scores + emissions[t]
            backpointers.append(best_tags)
        
        best_path = [scores.argmax().item()]
        for bp in reversed(backpointers):
            best_path.append(bp[best_path[-1]].item())
        return list(reversed(best_path))
```

---

## Machine Translation

Machine translation is the canonical seq2seq application.

### Theoretical Foundation

**Why encoder-decoder works**: The encoder builds a language-independent semantic representation, while the decoder generates language-specific surface forms. Attention enables soft alignment between source and target.

### Challenges

| Challenge | Description | Solution |
|-----------|-------------|----------|
| Vocabulary Size | 100K+ words | Subword tokenization (BPE, SentencePiece) |
| Word Order | SOV vs SVO | Attention mechanisms |
| Morphology | Agglutinative languages | Character/subword models |
| Rare Words | Names, terms | Copy mechanisms |

### Complete NMT Implementation

```python
class Attention(nn.Module):
    """Bahdanau-style additive attention."""
    
    def __init__(self, enc_hidden_size: int, dec_hidden_size: int, attn_size: int = 256):
        super().__init__()
        self.W_enc = nn.Linear(enc_hidden_size, attn_size, bias=False)
        self.W_dec = nn.Linear(dec_hidden_size, attn_size, bias=False)
        self.v = nn.Linear(attn_size, 1, bias=False)
    
    def forward(self, encoder_outputs, decoder_hidden, mask=None):
        enc_proj = self.W_enc(encoder_outputs)
        dec_proj = self.W_dec(decoder_hidden).unsqueeze(1)
        scores = self.v(torch.tanh(enc_proj + dec_proj)).squeeze(-1)
        
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class NMTModel(nn.Module):
    """Neural Machine Translation with attention."""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_dim: int = 256,
                 hidden_size: int = 512, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_size, num_layers,
                               batch_first=True, bidirectional=True, dropout=dropout)
        
        # Bridge
        self.bridge_h = nn.Linear(hidden_size * 2, hidden_size)
        self.bridge_c = nn.Linear(hidden_size * 2, hidden_size)
        
        # Decoder
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.attention = Attention(hidden_size * 2, hidden_size)
        self.decoder = nn.LSTM(embed_dim + hidden_size * 2, hidden_size, num_layers,
                               batch_first=True, dropout=dropout)
        self.output = nn.Linear(hidden_size + hidden_size * 2 + embed_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, src, src_lengths=None):
        embedded = self.dropout(self.src_embedding(src))
        if src_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        outputs, (h_n, c_n) = self.encoder(embedded)
        
        if src_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)
        c_n = c_n.view(self.num_layers, 2, -1, self.hidden_size)
        h_n = torch.cat([h_n[:, 0], h_n[:, 1]], dim=-1)
        c_n = torch.cat([c_n[:, 0], c_n[:, 1]], dim=-1)
        h_n = torch.tanh(self.bridge_h(h_n))
        c_n = torch.tanh(self.bridge_c(c_n))
        
        return outputs, (h_n, c_n)
    
    def decode_step(self, tgt_token, hidden, encoder_outputs, src_mask=None):
        embedded = self.dropout(self.tgt_embedding(tgt_token))
        h_top = hidden[0][-1]
        context, attn_weights = self.attention(encoder_outputs, h_top, src_mask)
        decoder_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, hidden = self.decoder(decoder_input, hidden)
        output = torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=-1)
        logits = self.output(output)
        return logits, hidden, attn_weights
    
    def forward(self, src, tgt, src_lengths=None, teacher_forcing_ratio=1.0):
        batch_size, tgt_len = tgt.shape
        encoder_outputs, hidden = self.encode(src, src_lengths)
        src_mask = (src == 0)
        
        outputs = []
        input_token = tgt[:, 0:1]
        
        for t in range(1, tgt_len):
            logits, hidden, _ = self.decode_step(input_token, hidden, encoder_outputs, src_mask)
            outputs.append(logits)
            
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tgt[:, t:t+1]
            else:
                input_token = logits.argmax(dim=-1, keepdim=True)
        
        return torch.stack(outputs, dim=1)


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing prevents overconfidence.
    
    Deep insight: Distributing probability mass to non-targets
    improves BLEU by 0.5-1.0 points. Optimal smoothing ~0.1.
    """
    
    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(predictions, dim=-1)
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.vocab_size - 2))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        smooth_targets[:, self.padding_idx] = 0
        
        non_pad_mask = targets != self.padding_idx
        smooth_targets = smooth_targets * non_pad_mask.unsqueeze(1)
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        return loss.masked_select(non_pad_mask).mean()
```

### Evaluation: BLEU Score

```python
def compute_bleu(references: List[List[str]], hypothesis: List[str],
                 max_n: int = 4, weights: Optional[List[float]] = None) -> float:
    """
    BLEU measures n-gram precision with brevity penalty.
    Correlates well with human judgment at corpus level.
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n
    
    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = Counter(tuple(hypothesis[i:i+n]) for i in range(len(hypothesis) - n + 1))
        
        ref_counts = Counter()
        for ref in references:
            ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref) - n + 1))
            for ngram, count in ref_ngrams.items():
                ref_counts[ngram] = max(ref_counts[ngram], count)
        
        clipped = sum(min(count, ref_counts.get(ngram, 0)) for ngram, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())
        precisions.append(clipped / total if total > 0 else 0)
    
    hyp_len = len(hypothesis)
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda x: abs(x - hyp_len))
    
    if hyp_len > closest_ref_len:
        bp = 1.0
    elif hyp_len > 0:
        bp = math.exp(1 - closest_ref_len / hyp_len)
    else:
        bp = 0
    
    if min(precisions) > 0:
        log_precision = sum(w * math.log(p) for w, p in zip(weights, precisions))
        return bp * math.exp(log_precision)
    return 0.0
```

---

## Text Summarization

### Types and Trade-offs

| Type | Approach | Pros | Cons |
|------|----------|------|------|
| Extractive | Select sentences | Grammatical, faithful | Limited compression |
| Abstractive | Generate new text | Fluent, flexible | Hallucination risk |

### Pointer-Generator Network

```python
class PointerGeneratorNetwork(nn.Module):
    """
    Abstractive summarization with copy mechanism.
    
    Deep insight: Copy mechanism addresses rare words and faithfulness.
    Coverage mechanism prevents repetition by tracking attended positions.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_size: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(embed_dim + hidden_size * 2, hidden_size, batch_first=True)
        
        # Attention with coverage
        self.W_h = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.W_c = nn.Linear(1, hidden_size * 2, bias=False)
        self.v = nn.Linear(hidden_size * 2, 1, bias=False)
        
        # Copy mechanism
        self.copy_gate = nn.Linear(hidden_size * 3 + embed_dim, 1)
        self.output = nn.Linear(hidden_size * 3, vocab_size)
    
    def forward(self, src, tgt, src_extended=None, max_oov=0):
        batch_size = src.size(0)
        
        # Encode
        enc_embedded = self.embedding(src)
        enc_outputs, (h_n, c_n) = self.encoder(enc_embedded)
        
        h_n = h_n.view(1, 2, batch_size, -1)
        c_n = c_n.view(1, 2, batch_size, -1)
        hidden = (torch.cat([h_n[:, 0], h_n[:, 1]], dim=-1),
                  torch.cat([c_n[:, 0], c_n[:, 1]], dim=-1))
        
        coverage = torch.zeros(batch_size, src.size(1), device=src.device)
        outputs, coverage_losses = [], []
        
        for t in range(tgt.size(1) - 1):
            dec_input = self.embedding(tgt[:, t:t+1])
            
            # Attention with coverage
            enc_proj = self.W_h(enc_outputs)
            dec_proj = self.W_s(hidden[0][-1]).unsqueeze(1)
            cov_proj = self.W_c(coverage.unsqueeze(-1))
            scores = self.v(torch.tanh(enc_proj + dec_proj + cov_proj)).squeeze(-1)
            attn_weights = torch.softmax(scores, dim=-1)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
            
            # Coverage loss
            coverage_losses.append(torch.sum(torch.min(attn_weights, coverage), dim=-1))
            coverage = coverage + attn_weights
            
            # Decoder step
            decoder_input = torch.cat([dec_input, context.unsqueeze(1)], dim=-1)
            dec_output, hidden = self.decoder(decoder_input, hidden)
            
            # Copy gate
            gate_input = torch.cat([dec_output.squeeze(1), context, dec_input.squeeze(1)], dim=-1)
            p_gen = torch.sigmoid(self.copy_gate(gate_input))
            
            # Final distribution
            vocab_dist = torch.softmax(self.output(torch.cat([dec_output.squeeze(1), context], dim=-1)), dim=-1)
            
            if src_extended is not None and max_oov > 0:
                extended_vocab = torch.zeros(batch_size, self.vocab_size + max_oov, device=src.device)
                extended_vocab[:, :self.vocab_size] = p_gen * vocab_dist
                extended_vocab.scatter_add_(1, src_extended, (1 - p_gen) * attn_weights)
                outputs.append(extended_vocab)
            else:
                final_dist = p_gen * vocab_dist
                final_dist.scatter_add_(1, src, (1 - p_gen) * attn_weights)
                outputs.append(final_dist)
        
        return torch.stack(outputs, dim=1), torch.stack(coverage_losses, dim=1).mean()
```

### Evaluation: ROUGE Score

```python
def compute_rouge_n(reference: List[str], hypothesis: List[str], n: int = 2) -> Dict[str, float]:
    """ROUGE-N measures recall of n-grams."""
    ref_ngrams = Counter(tuple(reference[i:i+n]) for i in range(len(reference) - n + 1))
    hyp_ngrams = Counter(tuple(hypothesis[i:i+n]) for i in range(len(hypothesis) - n + 1))
    
    overlap = sum(min(hyp_ngrams[ng], ref_ngrams.get(ng, 0)) for ng in hyp_ngrams)
    
    precision = overlap / sum(hyp_ngrams.values()) if hyp_ngrams else 0
    recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_rouge_l(reference: List[str], hypothesis: List[str]) -> Dict[str, float]:
    """ROUGE-L based on longest common subsequence."""
    m, n = len(reference), len(hypothesis)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i-1] == hypothesis[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs = dp[m][n]
    precision = lcs / n if n else 0
    recall = lcs / m if m else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}
```

---

## Dialogue Systems

### Hierarchical Dialogue Model

```python
class HierarchicalDialogueModel(nn.Module):
    """
    Multi-turn dialogue with hierarchical encoding.
    
    Deep insight: Hierarchical encoding separates utterance-level
    understanding from discourse-level flow modeling.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_size: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.utterance_encoder = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.context_encoder = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size, hidden_size)
        self.decoder = nn.LSTM(embed_dim + hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size * 2, vocab_size)
    
    def encode_context(self, turns: List[torch.Tensor]):
        turn_reprs = []
        for turn in turns:
            embedded = self.embedding(turn)
            _, (h_n, _) = self.utterance_encoder(embedded)
            turn_reprs.append(torch.cat([h_n[-2], h_n[-1]], dim=-1))
        
        turn_sequence = torch.stack(turn_reprs, dim=1)
        return self.context_encoder(turn_sequence)
    
    def forward(self, turns: List[torch.Tensor], response: torch.Tensor):
        context_output, hidden = self.encode_context(turns)
        
        outputs = []
        for t in range(response.size(1) - 1):
            token = response[:, t:t+1]
            embedded = self.embedding(token)
            attn_context, _ = self.attention(context_output, hidden[0][-1])
            decoder_input = torch.cat([embedded, attn_context.unsqueeze(1)], dim=-1)
            output, hidden = self.decoder(decoder_input, hidden)
            logits = self.output(torch.cat([output.squeeze(1), attn_context], dim=-1))
            outputs.append(logits)
        
        return torch.stack(outputs, dim=1)
```

---

## Time Series and Financial Applications

### Multi-Step Forecasting

```python
class MultiStepForecaster(nn.Module):
    """
    Encoder-decoder for multi-step forecasting.
    
    Deep insight: Direct multi-step prediction reduces error accumulation
    compared to iterated single-step forecasting.
    """
    
    def __init__(self, input_features: int, hidden_size: int = 64, forecast_steps: int = 10):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.encoder = nn.LSTM(input_features, hidden_size, num_layers=2, batch_first=True)
        self.decoder = nn.LSTM(1, hidden_size, num_layers=2, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None):
        _, hidden = self.encoder(x)
        
        predictions = []
        decoder_input = x[:, -1:, 0:1]
        
        for t in range(self.forecast_steps):
            output, hidden = self.decoder(decoder_input, hidden)
            pred = self.output(output)
            predictions.append(pred.squeeze(-1))
            
            if target is not None and t < self.forecast_steps - 1:
                decoder_input = target[:, t:t+1].unsqueeze(-1)
            else:
                decoder_input = pred
        
        return torch.cat(predictions, dim=-1)


class AnomalyDetectorLSTM(nn.Module):
    """
    LSTM autoencoder for anomaly detection.
    
    Deep insight: Reconstruction error identifies anomalies.
    Normal patterns have low error; unseen patterns have high error.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, latent_size: int = 32):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.enc_fc = nn.Linear(hidden_size, latent_size)
        self.dec_fc = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, input_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        _, (hidden, _) = self.encoder(x)
        latent = self.enc_fc(hidden.squeeze(0))
        decoder_init = self.dec_fc(latent).unsqueeze(0)
        decoder_input = decoder_init.permute(1, 0, 2).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(decoder_input)
        return self.output(decoded)
    
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reconstructed = self.forward(x)
            return ((x - reconstructed) ** 2).mean(dim=(1, 2))
```

---

## Speech Recognition

```python
class ASRModel(nn.Module):
    """
    End-to-end speech recognition.
    
    Deep insight: Conv frontend downsamples (100 frames/sec → ~25) and
    extracts acoustic patterns before recurrent encoding.
    """
    
    def __init__(self, input_dim: int = 80, vocab_size: int = 5000, hidden_size: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.encoder = nn.LSTM(32 * (input_dim // 4), hidden_size, num_layers=3,
                               batch_first=True, bidirectional=True, dropout=0.2)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention = Attention(hidden_size * 2, hidden_size)
        self.decoder = nn.LSTM(hidden_size * 3, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, audio: torch.Tensor, transcript: Optional[torch.Tensor] = None):
        batch_size = audio.size(0)
        x = self.conv(audio.unsqueeze(1))
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, x.size(2), -1)
        
        enc_outputs, hidden = self.encoder(x)
        hidden = (hidden[0][-2:].contiguous(), hidden[1][-2:].contiguous())
        
        if transcript is not None:
            outputs = []
            for t in range(transcript.size(1) - 1):
                embedded = self.embedding(transcript[:, t:t+1])
                context, _ = self.attention(enc_outputs, hidden[0][-1])
                output, hidden = self.decoder(torch.cat([embedded, context.unsqueeze(1)], dim=-1), hidden)
                outputs.append(self.output(output.squeeze(1)))
            return torch.stack(outputs, dim=1)
        return enc_outputs, hidden
```

---

## Code Generation

```python
class NL2CodeModel(nn.Module):
    """
    Natural language to code generation.
    
    Deep insight: Code differs from text - syntax constraints,
    semantic constraints, and execution verification are crucial.
    """
    
    def __init__(self, nl_vocab: int, code_vocab: int, embed_dim: int = 256, hidden_size: int = 512):
        super().__init__()
        self.nl_embedding = nn.Embedding(nl_vocab, embed_dim)
        self.code_embedding = nn.Embedding(code_vocab, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_size, num_layers=2,
                               batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = Attention(hidden_size * 2, hidden_size)
        self.decoder = nn.LSTM(embed_dim + hidden_size * 2, hidden_size,
                               num_layers=2, batch_first=True, dropout=0.3)
        self.output = nn.Linear(hidden_size, code_vocab)
    
    def forward(self, nl_input: torch.Tensor, code_output: Optional[torch.Tensor] = None):
        enc_outputs, hidden = self.encoder(self.nl_embedding(nl_input))
        hidden = (hidden[0].view(2, 2, -1, hidden[0].size(-1)).mean(1),
                  hidden[1].view(2, 2, -1, hidden[1].size(-1)).mean(1))
        
        if code_output is not None:
            outputs = []
            for t in range(code_output.size(1) - 1):
                embedded = self.code_embedding(code_output[:, t:t+1])
                context, _ = self.attention(enc_outputs, hidden[0][-1])
                output, hidden = self.decoder(torch.cat([embedded, context.unsqueeze(1)], dim=-1), hidden)
                outputs.append(self.output(output.squeeze(1)))
            return torch.stack(outputs, dim=1)
        return enc_outputs, hidden


class GrammarErrorCorrection(nn.Module):
    """
    GEC with high copy bias (most tokens unchanged).
    
    Deep insight: Bias initialization (2.0 → sigmoid ≈ 0.88) starts
    model with strong copying prior for efficiency.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_size: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_size, num_layers=2,
                               batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = Attention(hidden_size * 2, hidden_size)
        self.decoder = nn.LSTM(embed_dim + hidden_size * 2, hidden_size,
                               num_layers=2, batch_first=True, dropout=0.3)
        self.copy_gate = nn.Linear(hidden_size * 2, 1)
        nn.init.constant_(self.copy_gate.bias, 2.0)  # High copy bias
        self.output = nn.Linear(hidden_size, vocab_size)
```

---

## Application Comparison

| Application | Model Choice | Key Challenges | Special Techniques |
|-------------|--------------|----------------|-------------------|
| Text Generation | LSTM | Coherence | Temperature sampling |
| Classification | BiLSTM | Full context | Attention pooling |
| Translation | BiLSTM + Attention | Alignment | Subword, label smoothing |
| Summarization | Pointer-Generator | Faithfulness | Copy, coverage |
| Time Series | GRU | Non-stationarity | Multi-step decoder |
| Dialogue | Hierarchical LSTM | Context | Turn-level encoding |
| QA | Question-aware | Evidence selection | Passage attention |
| Speech | Deep BiLSTM | Alignment | Conv frontend, CTC |
| Code Gen | LSTM + Copy | Syntax | Constrained decoding |
| GEC | LSTM + High copy | Few edits | Copy bias init |

## Evaluation Metrics

| Application | Primary Metrics |
|-------------|-----------------|
| Translation | BLEU, METEOR, chrF |
| Summarization | ROUGE-1/2/L, BERTScore |
| Dialogue | Perplexity, human eval |
| QA | Exact Match, F1 |
| GEC | GLEU, M², ERRANT |
| Code Gen | Execution accuracy |
| ASR | WER, CER |
| Time Series | MSE, MAE, MAPE |

---

## Training Best Practices

```python
# Gradient clipping - essential for RNNs
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# Learning rate schedule
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3)

# Regularization
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Teacher forcing decay
def get_teacher_forcing_ratio(epoch, total_epochs):
    return max(0.5, 1.0 - epoch / total_epochs)
```

---

## Summary

Seq2seq models power diverse applications through:

1. **Flexible architecture**: Adapt encoder/decoder for domain
2. **Attention mechanisms**: Variable-length alignment
3. **Copy mechanisms**: Direct token transfer for faithfulness
4. **Domain-specific decoding**: Constrained generation

### Key Success Factors

- Appropriate tokenization (subword for NMT)
- Task-specific objectives (label smoothing, coverage)
- Domain-adapted evaluation metrics
- Proper regularization and gradient management

---

## References

1. Sutskever et al. (2014). Sequence to Sequence Learning. *NeurIPS*.
2. Bahdanau et al. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR*.
3. See et al. (2017). Get To The Point: Summarization with Pointer-Generator Networks. *ACL*.
4. Serban et al. (2016). Building End-To-End Dialogue Systems. *AAAI*.
5. Chan et al. (2016). Listen, Attend and Spell. *ICASSP*.
6. Graves (2012). Supervised Sequence Labelling with Recurrent Neural Networks. *Springer*.
