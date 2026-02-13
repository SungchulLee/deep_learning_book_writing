# BiLSTM for Named Entity Recognition

## Learning Objectives

- Understand why bidirectional context matters for NER
- Implement BiLSTM-based sequence labeling models
- Integrate character-level embeddings for morphological features
- Combine BiLSTM with CRF for optimal performance

## Why BiLSTM for NER?

Named entities require **bidirectional context** for accurate recognition:

- **Left context**: "CEO **Satya Nadella**" - title indicates person
- **Right context**: "**Microsoft** announced" - verb indicates organization  
- **Both directions**: "The **New York** Times" - full phrase needed

## Architecture

```
Word: "Apple"  "Inc"  "announced"  "profits"
        ↓        ↓        ↓           ↓
    [Embedding] [Emb]   [Emb]       [Emb]
        ↓        ↓        ↓           ↓
    →LSTM→   →LSTM→   →LSTM→     →LSTM→   (Forward)
    ←LSTM←   ←LSTM←   ←LSTM←     ←LSTM←   (Backward)
        ↓        ↓        ↓           ↓
    [Concat]  [Concat] [Concat]   [Concat]
        ↓        ↓        ↓           ↓
    [Linear]  [Linear] [Linear]   [Linear]
        ↓        ↓        ↓           ↓
      B-ORG    I-ORG      O           O
```

## Mathematical Formulation

### Forward and Backward LSTMs

**Forward LSTM** processes left-to-right:
$$\overrightarrow{h}_t = \text{LSTM}(x_t, \overrightarrow{h}_{t-1})$$

**Backward LSTM** processes right-to-left:
$$\overleftarrow{h}_t = \text{LSTM}(x_t, \overleftarrow{h}_{t+1})$$

### Concatenated Representation

$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t] \in \mathbb{R}^{2d}$$

### Emission Scores

$$e_t = W_o \cdot h_t + b_o \in \mathbb{R}^{|L|}$$

## PyTorch Implementation

### Basic BiLSTM NER

```python
import torch
import torch.nn as nn
from typing import Optional

class BiLSTMNER(nn.Module):
    """
    BiLSTM for Named Entity Recognition.
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.5,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional doubles this
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_tags)
        
        self.num_tags = num_tags
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        # Embed tokens
        embeds = self.embedding(input_ids)
        embeds = self.dropout(embeds)
        
        # BiLSTM encoding
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embeds, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        else:
            lstm_out, _ = self.lstm(embeds)
        
        lstm_out = self.dropout(lstm_out)
        
        # Classification
        logits = self.classifier(lstm_out)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                logits.view(-1, self.num_tags),
                labels.view(-1)
            )
        
        return {'loss': loss, 'logits': logits}
    
    def predict(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return torch.argmax(outputs['logits'], dim=-1)
```

### BiLSTM with Character Embeddings

Character-level features capture morphology (prefixes, suffixes, capitalization):

```python
class CharLSTM(nn.Module):
    """Character-level LSTM encoder."""
    
    def __init__(
        self,
        char_vocab_size: int,
        char_embedding_dim: int = 25,
        char_hidden_dim: int = 50
    ):
        super().__init__()
        
        self.char_embedding = nn.Embedding(
            char_vocab_size, char_embedding_dim, padding_idx=0
        )
        
        self.char_lstm = nn.LSTM(
            char_embedding_dim,
            char_hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )
    
    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_ids: (batch, max_words, max_chars)
        Returns:
            char_repr: (batch, max_words, char_hidden_dim)
        """
        batch_size, max_words, max_chars = char_ids.shape
        
        # Reshape for processing
        char_ids = char_ids.view(-1, max_chars)
        
        char_embeds = self.char_embedding(char_ids)
        _, (h_n, _) = self.char_lstm(char_embeds)
        
        # Concatenate forward and backward final states
        char_repr = torch.cat([h_n[0], h_n[1]], dim=-1)
        char_repr = char_repr.view(batch_size, max_words, -1)
        
        return char_repr


class BiLSTMCharNER(nn.Module):
    """BiLSTM NER with character-level embeddings."""
    
    def __init__(
        self,
        vocab_size: int,
        char_vocab_size: int,
        num_tags: int,
        word_embedding_dim: int = 100,
        char_embedding_dim: int = 25,
        char_hidden_dim: int = 50,
        hidden_dim: int = 256,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim)
        self.char_encoder = CharLSTM(
            char_vocab_size, char_embedding_dim, char_hidden_dim
        )
        
        # Concatenate word + char embeddings
        lstm_input_dim = word_embedding_dim + char_hidden_dim
        
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_tags)
        self.num_tags = num_tags
    
    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        # Word embeddings
        word_embeds = self.word_embedding(word_ids)
        
        # Character-level representations
        char_repr = self.char_encoder(char_ids)
        
        # Concatenate
        combined = torch.cat([word_embeds, char_repr], dim=-1)
        combined = self.dropout(combined)
        
        # BiLSTM
        lstm_out, _ = self.lstm(combined)
        lstm_out = self.dropout(lstm_out)
        
        # Classify
        logits = self.classifier(lstm_out)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.num_tags), labels.view(-1))
        
        return {'loss': loss, 'logits': logits}
```

### BiLSTM-CRF

```python
class BiLSTMCRF(nn.Module):
    """BiLSTM with CRF layer for NER."""
    
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int = 100,
        hidden_dim: int = 256,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        
        # CRF layer (transitions learned)
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        self.num_tags = num_tags
    
    def _get_emissions(self, input_ids, attention_mask=None):
        embeds = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        return self.hidden2tag(lstm_out)
    
    def forward(self, input_ids, attention_mask, labels):
        emissions = self._get_emissions(input_ids, attention_mask)
        
        # CRF forward: compute log partition and gold score
        gold_score = self._score_sentence(emissions, labels, attention_mask)
        partition = self._forward_algorithm(emissions, attention_mask)
        
        loss = (partition - gold_score).mean()
        return {'loss': loss, 'emissions': emissions}
    
    def _forward_algorithm(self, emissions, mask):
        """Compute log partition function."""
        batch_size, seq_len, num_tags = emissions.shape
        
        # Initialize
        alpha = self.start_transitions + emissions[:, 0]
        
        for t in range(1, seq_len):
            alpha_t = []
            for tag in range(num_tags):
                emit = emissions[:, t, tag].unsqueeze(1)
                trans = self.transitions[tag].unsqueeze(0)
                score = alpha + emit + trans
                alpha_t.append(torch.logsumexp(score, dim=1))
            
            new_alpha = torch.stack(alpha_t, dim=1)
            alpha = torch.where(
                mask[:, t].unsqueeze(1).bool(),
                new_alpha,
                alpha
            )
        
        return torch.logsumexp(alpha + self.end_transitions, dim=1)
    
    def _score_sentence(self, emissions, tags, mask):
        """Compute score of gold sequence."""
        batch_size, seq_len, _ = emissions.shape
        
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        
        for t in range(1, seq_len):
            emit = emissions[:, t].gather(1, tags[:, t:t+1]).squeeze(1)
            trans = self.transitions[tags[:, t], tags[:, t-1]]
            score += (emit + trans) * mask[:, t].float()
        
        # End transitions
        seq_lens = mask.sum(dim=1).long()
        last_tags = tags.gather(1, (seq_lens - 1).unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        
        return score
    
    def decode(self, input_ids, attention_mask):
        """Viterbi decoding."""
        emissions = self._get_emissions(input_ids, attention_mask)
        return self._viterbi_decode(emissions, attention_mask)
    
    def _viterbi_decode(self, emissions, mask):
        """Find best tag sequence."""
        batch_size, seq_len, num_tags = emissions.shape
        
        score = self.start_transitions + emissions[:, 0]
        history = []
        
        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emit = emissions[:, t].unsqueeze(1)
            
            next_score = broadcast_score + self.transitions + broadcast_emit
            next_score, indices = next_score.max(dim=1)
            
            history.append(indices)
            score = torch.where(mask[:, t].unsqueeze(1).bool(), next_score, score)
        
        score += self.end_transitions
        
        # Backtrack
        best_tags = []
        for idx in range(batch_size):
            seq_len_i = mask[idx].sum().int().item()
            _, best_last = score[idx].max(dim=0)
            
            tags = [best_last.item()]
            for hist in reversed(history[:seq_len_i - 1]):
                tags.append(hist[idx, tags[-1]].item())
            tags.reverse()
            best_tags.append(tags)
        
        return best_tags
```

## Comparison: BiLSTM vs Transformers

| Aspect | BiLSTM | Transformer |
|--------|--------|-------------|
| Training data needed | Less | More |
| Training speed | Faster | Slower |
| Inference speed | Slower | Faster (batched) |
| Performance | Good | Best |
| Character features | Easy to add | Subwords handle this |
| Pre-training benefit | GloVe/Word2Vec | BERT/RoBERTa |

## Best Practices

1. **Use pre-trained word embeddings** (GloVe, fastText)
2. **Add character-level features** for morphology
3. **Combine with CRF** for structured prediction
4. **Apply dropout** between layers (0.3-0.5)
5. **Gradient clipping** (max norm 1.0-5.0)
6. **Early stopping** on validation F1

## Summary

BiLSTM models remain a strong baseline for NER:
- Capture bidirectional context effectively
- Work well with limited training data
- Easy to extend with character features
- CRF layer improves boundary detection
