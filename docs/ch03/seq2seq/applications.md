# Sequence-to-Sequence Applications

## Introduction

Sequence-to-sequence models have revolutionized numerous NLP tasks where both input and output are variable-length sequences. This section covers major applications, their specific architectures, and implementation considerations.

## Machine Translation

The original and most prominent seq2seq application.

### Architecture

```python
import torch
import torch.nn as nn

class NMTModel(nn.Module):
    """Neural Machine Translation model."""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, 
                 hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Bidirectional encoder
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.encoder = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Bridge: project bidirectional to unidirectional
        self.bridge_h = nn.Linear(hidden_size * 2, hidden_size)
        self.bridge_c = nn.Linear(hidden_size * 2, hidden_size)
        
        # Decoder with attention
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.decoder = nn.LSTM(
            embed_dim + hidden_size * 2, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        
        # Attention
        self.attention = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
        # Output
        self.output = nn.Linear(hidden_size * 3, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, src):
        embedded = self.dropout(self.src_embedding(src))
        outputs, (h_n, c_n) = self.encoder(embedded)
        
        # Combine bidirectional states
        h_n = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        c_n = torch.cat([c_n[-2], c_n[-1]], dim=-1)
        
        h_n = torch.tanh(self.bridge_h(h_n)).unsqueeze(0)
        c_n = torch.tanh(self.bridge_c(c_n)).unsqueeze(0)
        
        return outputs, (h_n, c_n)
```

### Training Considerations

- **Subword tokenization**: BPE or SentencePiece for open vocabulary
- **Backtranslation**: Augment data by translating target → source
- **Label smoothing**: Prevent overconfident predictions
- **Checkpoint averaging**: Average last N checkpoints

## Text Summarization

Generate concise summaries from longer documents.

### Abstractive Summarization

```python
class SummarizationModel(nn.Module):
    """Abstractive summarization with copy mechanism."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_size, 
                               batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(embed_dim + hidden_size * 2, hidden_size,
                               batch_first=True)
        
        # Copy mechanism
        self.copy_gate = nn.Linear(hidden_size * 3, 1)
        self.output = nn.Linear(hidden_size * 3, vocab_size)
    
    def forward(self, src, tgt, src_extended_vocab=None):
        """
        Args:
            src: Source document tokens
            tgt: Target summary tokens
            src_extended_vocab: Source with OOV tokens mapped to extended vocab
        """
        # Encode
        enc_outputs, hidden = self.encoder(self.embedding(src))
        
        # Decode with attention and copy
        outputs = []
        for t in range(tgt.size(1) - 1):
            dec_input = tgt[:, t:t+1]
            output, hidden, attn_weights = self.decode_step(
                dec_input, hidden, enc_outputs
            )
            
            # Copy mechanism: interpolate between generate and copy
            p_gen = torch.sigmoid(self.copy_gate(output))
            
            vocab_dist = torch.softmax(self.output(output), dim=-1)
            copy_dist = attn_weights  # Copy from source
            
            # Combine distributions
            final_dist = p_gen * vocab_dist
            # Add copy probabilities to extended vocab positions
            # ...
            
            outputs.append(final_dist)
        
        return torch.stack(outputs, dim=1)
```

### Extractive-Abstractive Hybrid

1. Extract salient sentences
2. Rewrite/compress extracted content

## Conversational AI

### Dialogue Response Generation

```python
class DialogueModel(nn.Module):
    """Seq2seq for multi-turn dialogue."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, max_turns=5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Hierarchical encoder
        self.utterance_encoder = nn.LSTM(
            embed_dim, hidden_size, batch_first=True, bidirectional=True
        )
        self.context_encoder = nn.LSTM(
            hidden_size * 2, hidden_size, batch_first=True
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            embed_dim + hidden_size, hidden_size, batch_first=True
        )
        self.output = nn.Linear(hidden_size, vocab_size)
    
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
```

### Persona-Conditioned Generation

```python
class PersonaDialogue(nn.Module):
    """Dialogue model conditioned on persona description."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Persona encoder
        self.persona_encoder = nn.LSTM(
            embed_dim, hidden_size, batch_first=True
        )
        
        # Context encoder
        self.context_encoder = nn.LSTM(
            embed_dim, hidden_size, batch_first=True
        )
        
        # Decoder attends to both persona and context
        self.persona_attention = Attention(hidden_size)
        self.context_attention = Attention(hidden_size)
        
        self.decoder = nn.LSTM(
            embed_dim + hidden_size * 2, hidden_size, batch_first=True
        )
```

## Question Answering

### Reading Comprehension

Generate answers from context passages:

```python
class QAModel(nn.Module):
    """Generative QA: generate answer span or full answer."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Encode question
        self.question_encoder = nn.LSTM(
            embed_dim, hidden_size, batch_first=True, bidirectional=True
        )
        
        # Encode passage with question-aware attention
        self.passage_encoder = nn.LSTM(
            embed_dim + hidden_size * 2, hidden_size, 
            batch_first=True, bidirectional=True
        )
        
        # Decode answer
        self.decoder = nn.LSTM(
            embed_dim + hidden_size * 2, hidden_size, batch_first=True
        )
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, question, passage, answer=None):
        # Encode question
        q_embedded = self.embedding(question)
        q_outputs, q_hidden = self.question_encoder(q_embedded)
        
        # Question-aware passage encoding
        p_embedded = self.embedding(passage)
        # Compute attention between passage and question
        # Concatenate attention output with passage embeddings
        # ...
        
        return answer_logits
```

## Grammar Error Correction

```python
class GECModel(nn.Module):
    """Grammar Error Correction as seq2seq."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        
        # Shared embedding (source and target are same language)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.encoder = nn.LSTM(
            embed_dim, hidden_size, num_layers=2,
            batch_first=True, bidirectional=True
        )
        
        # Decoder with copy mechanism (most tokens copied unchanged)
        self.decoder = nn.LSTM(
            embed_dim + hidden_size * 2, hidden_size, num_layers=2,
            batch_first=True
        )
        
        # High copy bias since most tokens are unchanged
        self.copy_gate = nn.Linear(hidden_size, 1)
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, src, tgt=None):
        # Copy mechanism is crucial here
        # Most tokens should be copied directly
        pass
```

### Training Data Considerations

- Synthetic errors: Inject artificial errors into clean text
- Error type distribution: Match real-world error patterns
- Fluency reward: Ensure corrections are fluent

## Code Generation

### Natural Language to Code

```python
class NL2CodeModel(nn.Module):
    """Generate code from natural language descriptions."""
    
    def __init__(self, nl_vocab_size, code_vocab_size, embed_dim, hidden_size):
        super().__init__()
        
        # Separate vocabularies for NL and code
        self.nl_embedding = nn.Embedding(nl_vocab_size, embed_dim)
        self.code_embedding = nn.Embedding(code_vocab_size, embed_dim)
        
        self.encoder = nn.LSTM(
            embed_dim, hidden_size, batch_first=True, bidirectional=True
        )
        
        # Decoder generates code tokens
        self.decoder = nn.LSTM(
            embed_dim + hidden_size * 2, hidden_size, batch_first=True
        )
        
        self.output = nn.Linear(hidden_size, code_vocab_size)
    
    def forward(self, nl_input, code_output=None):
        # Encode natural language
        nl_embedded = self.nl_embedding(nl_input)
        enc_outputs, hidden = self.encoder(nl_embedded)
        
        # Decode code
        # ...
        pass
```

### Considerations

- **Syntax constraints**: Ensure generated code is syntactically valid
- **Execution feedback**: Use execution results for reinforcement learning
- **Structured decoding**: Generate AST nodes instead of raw tokens

## Speech Recognition (ASR)

Seq2seq for end-to-end speech recognition:

```python
class ASRModel(nn.Module):
    """End-to-end speech recognition."""
    
    def __init__(self, input_dim, vocab_size, hidden_size):
        super().__init__()
        
        # Audio encoder (processes spectrograms)
        self.encoder = nn.LSTM(
            input_dim, hidden_size, num_layers=3,
            batch_first=True, bidirectional=True
        )
        
        # Text decoder
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.decoder = nn.LSTM(
            hidden_size * 3, hidden_size, num_layers=2,
            batch_first=True
        )
        
        self.attention = Attention(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, audio_features, transcript=None):
        """
        Args:
            audio_features: (batch, time, freq) spectrograms
            transcript: (batch, seq_len) text tokens
        """
        enc_outputs, hidden = self.encoder(audio_features)
        # Decode with attention
        # ...
        pass
```

## Application Comparison

| Application | Key Challenges | Special Techniques |
|-------------|----------------|-------------------|
| Translation | Long-range dependencies | Attention, subword tokenization |
| Summarization | Compression, faithfulness | Copy mechanism, coverage |
| Dialogue | Context, persona | Hierarchical encoding |
| QA | Evidence selection | Passage attention |
| GEC | Most tokens unchanged | High copy bias |
| Code Gen | Syntax validity | Constrained decoding |
| ASR | Audio-text alignment | CTC loss hybrid |

## Evaluation Metrics

| Application | Primary Metrics |
|-------------|-----------------|
| Translation | BLEU, METEOR, chrF |
| Summarization | ROUGE, BERTScore |
| Dialogue | Perplexity, human evaluation |
| QA | Exact Match, F1 |
| GEC | GLEU, M² |
| Code Gen | Execution accuracy, BLEU |
| ASR | WER, CER |

## Summary

Seq2seq models power diverse applications through:

1. **Flexible architecture**: Adapt encoder/decoder for domain
2. **Attention mechanisms**: Handle variable-length alignment
3. **Copy mechanisms**: Enable direct token transfer
4. **Domain-specific decoding**: Constrained generation for structured outputs

Key success factors:
- Appropriate tokenization for the domain
- Task-specific training objectives
- Domain-adapted evaluation metrics
- Sufficient training data (or transfer learning)
