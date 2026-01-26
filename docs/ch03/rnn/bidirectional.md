# Bidirectional RNNs

## Introduction

Standard RNNs process sequences in one direction, limiting their understanding to past context. **Bidirectional RNNs** (BiRNNs) process sequences in both forward and backward directions simultaneously, allowing each position to incorporate information from the entire sequence. This architecture proves essential for tasks where future context informs interpretation of current elements.

## Motivation: Why Bidirectionality Matters

### The Fundamental Limitation of Unidirectional Processing

Consider the sentence: "The bank by the river was steep."

A forward-only RNN processing "bank" has only seen "The"—insufficient to distinguish between a financial institution and a riverbank. A bidirectional model sees both the preceding and following context, enabling correct interpretation.

### Linguistic Phenomena Requiring Bidirectional Context

```python
def demonstrate_bidirectional_need():
    """
    Examples where future context is essential for understanding.
    """
    examples = {
        "Word Sense Disambiguation": {
            "sentence": "The bank by the river was steep.",
            "ambiguous_word": "bank",
            "forward_context": "The",
            "backward_context": "river, steep",
            "resolution": "riverbank (not financial institution)"
        },
        "Coreference Resolution": {
            "sentence": "The trophy doesn't fit in the suitcase because it is too big.",
            "ambiguous_word": "it",
            "forward_context": "trophy, suitcase",
            "backward_context": "too big",
            "resolution": "'it' refers to 'trophy' (big things don't fit)"
        },
        "Negation Scope": {
            "sentence": "I don't think this movie is bad at all.",
            "challenge": "Sentiment of 'bad'",
            "forward_context": "don't think",
            "backward_context": "at all",
            "resolution": "Positive sentiment (double negation + intensifier)"
        },
        "Garden Path Sentences": {
            "sentence": "The horse raced past the barn fell.",
            "challenge": "Parse structure",
            "forward_context": "horse raced past",
            "backward_context": "fell",
            "resolution": "'raced past the barn' is a reduced relative clause"
        }
    }
    
    print("Linguistic Phenomena Requiring Bidirectional Context")
    print("=" * 70)
    
    for phenomenon, details in examples.items():
        print(f"\n{phenomenon}:")
        print(f"  Sentence: \"{details['sentence']}\"")
        print(f"  Challenge: {details.get('challenge', details['ambiguous_word'])}")
        print(f"  Forward only sees: {details['forward_context']}")
        print(f"  Backward reveals: {details['backward_context']}")
        print(f"  Resolution: {details['resolution']}")

demonstrate_bidirectional_need()
```

### Tasks That Benefit from Bidirectionality

| Task | Why Bidirectional Helps |
|------|------------------------|
| Named Entity Recognition | "Apple announced..." vs "I ate an apple..." |
| Part-of-Speech Tagging | "record" as noun vs verb depends on context |
| Sentiment Analysis | Negation, intensifiers, sarcasm markers |
| Machine Translation | Word alignment requires full source context |
| Question Answering | Answer span depends on entire passage |
| Semantic Role Labeling | Arguments can appear before or after predicates |

## Architecture Deep Dive

### Conceptual Structure

A BiRNN consists of two independent RNNs processing the sequence in opposite directions:

```
Time:        t=1    t=2    t=3    t=4    t=5
             ───────────────────────────────►

Forward:     h₁→ ─► h₂→ ─► h₃→ ─► h₄→ ─► h₅→
              ↑      ↑      ↑      ↑      ↑
Input:       x₁     x₂     x₃     x₄     x₅
              ↓      ↓      ↓      ↓      ↓
Backward:    h₁← ◄─ h₂← ◄─ h₃← ◄─ h₄← ◄─ h₅←

             ◄───────────────────────────────
Time:        t=5    t=4    t=3    t=2    t=1

Output:     [h₁→;h₁←] [h₂→;h₂←] [h₃→;h₃←] [h₄→;h₄←] [h₅→;h₅←]
              ↓         ↓         ↓         ↓         ↓
Dimension:   2H        2H        2H        2H        2H
```

### Information Flow Analysis

At each position $t$, the bidirectional representation contains:

**Forward hidden state** $\overrightarrow{h}_t$:
- Summarizes information from positions $1, 2, \ldots, t$
- Encodes "what came before"

**Backward hidden state** $\overleftarrow{h}_t$:
- Summarizes information from positions $T, T-1, \ldots, t$
- Encodes "what comes after"

**Combined representation** $h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$:
- Full sequence context available at every position
- Dimension: $2 \times \text{hidden\_size}$

## Mathematical Formulation

### Forward Direction

Standard LSTM/GRU equations proceeding left-to-right:

$$\overrightarrow{h}_t = \text{RNN}_{\text{fwd}}(x_t, \overrightarrow{h}_{t-1}), \quad t = 1, 2, \ldots, T$$

Initial state: $\overrightarrow{h}_0 = \mathbf{0}$ (or learned)

### Backward Direction

Same architecture but processing right-to-left:

$$\overleftarrow{h}_t = \text{RNN}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1}), \quad t = T, T-1, \ldots, 1$$

Initial state: $\overleftarrow{h}_{T+1} = \mathbf{0}$ (or learned)

### Output Combination Strategies

**Concatenation** (most common):
$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t] \in \mathbb{R}^{2d}$$

**Summation** (preserves dimension):
$$h_t = \overrightarrow{h}_t + \overleftarrow{h}_t \in \mathbb{R}^{d}$$

**Average**:
$$h_t = \frac{1}{2}(\overrightarrow{h}_t + \overleftarrow{h}_t) \in \mathbb{R}^{d}$$

**Learned combination**:
$$h_t = W[\overrightarrow{h}_t; \overleftarrow{h}_t] + b \in \mathbb{R}^{d}$$

```python
import torch
import torch.nn as nn

class FlexibleBiRNN(nn.Module):
    """BiRNN with configurable output combination strategy."""
    
    def __init__(self, input_size, hidden_size, combination='concat'):
        super().__init__()
        self.hidden_size = hidden_size
        self.combination = combination
        
        self.forward_rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.backward_rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        if combination == 'learned':
            self.combine_layer = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, x):
        # Forward pass
        out_fwd, _ = self.forward_rnn(x)
        
        # Backward pass
        x_rev = torch.flip(x, dims=[1])
        out_bwd_rev, _ = self.backward_rnn(x_rev)
        out_bwd = torch.flip(out_bwd_rev, dims=[1])
        
        # Combine
        if self.combination == 'concat':
            output = torch.cat([out_fwd, out_bwd], dim=-1)
        elif self.combination == 'sum':
            output = out_fwd + out_bwd
        elif self.combination == 'avg':
            output = (out_fwd + out_bwd) / 2
        elif self.combination == 'learned':
            output = self.combine_layer(torch.cat([out_fwd, out_bwd], dim=-1))
        
        return output
    
    @property
    def output_size(self):
        if self.combination == 'concat':
            return self.hidden_size * 2
        return self.hidden_size
```

## PyTorch Implementation

### Using Built-in Bidirectional Flag

```python
import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    """Bidirectional LSTM with comprehensive output handling."""
    
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch_size, seq_len, input_size)
            lengths: Optional sequence lengths for packing
        
        Returns:
            output: (batch_size, seq_len, hidden_size * 2)
            h_n: (num_layers * 2, batch_size, hidden_size)
            c_n: (num_layers * 2, batch_size, hidden_size)
        """
        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output_packed, (h_n, c_n) = self.lstm(x_packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output_packed, batch_first=True
            )
        else:
            output, (h_n, c_n) = self.lstm(x)
        
        return output, h_n, c_n
    
    def get_final_states(self, h_n):
        """Extract final hidden states from both directions."""
        # h_n: (num_layers * 2, batch, hidden)
        # Layout: [L0_fwd, L0_bwd, L1_fwd, L1_bwd, ...]
        h_fwd = h_n[-2]  # Last layer forward
        h_bwd = h_n[-1]  # Last layer backward
        return torch.cat([h_fwd, h_bwd], dim=-1)


# Example usage
model = BiLSTMModel(input_size=100, hidden_size=128, num_layers=2)
x = torch.randn(32, 50, 100)
output, h_n, c_n = model(x)

print(f"Output shape: {output.shape}")      # (32, 50, 256)
print(f"Hidden shape: {h_n.shape}")         # (4, 32, 128)
final = model.get_final_states(h_n)
print(f"Final states: {final.shape}")       # (32, 256)
```

### Understanding the Output Structure

```python
def analyze_bidirectional_output():
    """
    Detailed analysis of bidirectional LSTM output structure.
    """
    hidden_size = 64
    num_layers = 2
    seq_len = 10
    batch_size = 4
    
    bilstm = nn.LSTM(32, hidden_size, num_layers, 
                     batch_first=True, bidirectional=True)
    x = torch.randn(batch_size, seq_len, 32)
    
    output, (h_n, c_n) = bilstm(x)
    
    print("Bidirectional LSTM Output Analysis")
    print("=" * 60)
    
    print(f"\n1. SEQUENCE OUTPUT: {output.shape}")
    print(f"   Shape: (batch={batch_size}, seq={seq_len}, hidden*2={hidden_size*2})")
    print(f"   - output[:, :, :64]  → Forward direction outputs")
    print(f"   - output[:, :, 64:]  → Backward direction outputs")
    
    # Verify: forward output at t=0 should differ from t=seq_len-1
    # Backward output at t=0 should match backward's final state
    
    print(f"\n2. HIDDEN STATE h_n: {h_n.shape}")
    print(f"   Shape: (num_layers*2={num_layers*2}, batch={batch_size}, hidden={hidden_size})")
    print(f"   Layout:")
    for layer in range(num_layers):
        fwd_idx = layer * 2
        bwd_idx = layer * 2 + 1
        print(f"     Layer {layer}: h_n[{fwd_idx}]=forward, h_n[{bwd_idx}]=backward")
    
    print(f"\n3. CELL STATE c_n: {c_n.shape}")
    print(f"   Same layout as h_n")
    
    print("\n4. IMPORTANT NOTES:")
    print("   - Forward 'final' = state after processing x_T (last token)")
    print("   - Backward 'final' = state after processing x_1 (first token)")
    print("   - For classification, typically use last layer's states:")
    print(f"     final = concat(h_n[-2], h_n[-1])  # Shape: (batch, {hidden_size*2})")
    
    # Demonstrate the relationship
    print("\n5. OUTPUT-STATE CORRESPONDENCE:")
    
    # Forward direction: output at t=-1 should match h_n forward final
    forward_output_final = output[:, -1, :hidden_size]
    forward_h_final = h_n[-2]  # Last layer, forward
    match_fwd = torch.allclose(forward_output_final, forward_h_final, atol=1e-6)
    print(f"   output[:, -1, :H] == h_n[-2]: {match_fwd}")
    
    # Backward direction: output at t=0 should match h_n backward final
    backward_output_final = output[:, 0, hidden_size:]
    backward_h_final = h_n[-1]  # Last layer, backward
    match_bwd = torch.allclose(backward_output_final, backward_h_final, atol=1e-6)
    print(f"   output[:, 0, H:] == h_n[-1]: {match_bwd}")

analyze_bidirectional_output()
```

### Manual Implementation for Understanding

```python
class ManualBidirectionalLSTM(nn.Module):
    """
    Manual implementation to understand the mechanics.
    Equivalent to nn.LSTM(..., bidirectional=True).
    """
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Separate forward and backward LSTMs for each layer
        self.forward_lstms = nn.ModuleList()
        self.backward_lstms = nn.ModuleList()
        
        for layer in range(num_layers):
            # Input size: original for first layer, 2*hidden for subsequent
            layer_input = input_size if layer == 0 else hidden_size * 2
            
            self.forward_lstms.append(
                nn.LSTM(layer_input, hidden_size, 1, batch_first=True)
            )
            self.backward_lstms.append(
                nn.LSTM(layer_input, hidden_size, 1, batch_first=True)
            )
    
    def forward(self, x):
        """
        Process through all layers bidirectionally.
        """
        batch_size = x.size(0)
        
        h_n_list = []
        c_n_list = []
        
        current_input = x
        
        for layer in range(self.num_layers):
            # Forward pass
            out_fwd, (h_fwd, c_fwd) = self.forward_lstms[layer](current_input)
            
            # Backward pass (reverse, process, reverse back)
            x_rev = torch.flip(current_input, dims=[1])
            out_bwd_rev, (h_bwd, c_bwd) = self.backward_lstms[layer](x_rev)
            out_bwd = torch.flip(out_bwd_rev, dims=[1])
            
            # Concatenate outputs for next layer's input
            current_input = torch.cat([out_fwd, out_bwd], dim=-1)
            
            # Store final states
            h_n_list.extend([h_fwd.squeeze(0), h_bwd.squeeze(0)])
            c_n_list.extend([c_fwd.squeeze(0), c_bwd.squeeze(0)])
        
        # Stack final states: (num_layers * 2, batch, hidden)
        h_n = torch.stack(h_n_list, dim=0)
        c_n = torch.stack(c_n_list, dim=0)
        
        return current_input, (h_n, c_n)


# Verify equivalence
def verify_manual_implementation():
    torch.manual_seed(42)
    
    input_size, hidden_size, num_layers = 32, 64, 2
    batch_size, seq_len = 4, 10
    
    # Create both implementations
    manual = ManualBidirectionalLSTM(input_size, hidden_size, num_layers)
    builtin = nn.LSTM(input_size, hidden_size, num_layers, 
                      batch_first=True, bidirectional=True)
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    out_manual, (h_manual, c_manual) = manual(x)
    out_builtin, (h_builtin, c_builtin) = builtin(x)
    
    print("Manual vs Built-in Bidirectional LSTM")
    print(f"Output shapes match: {out_manual.shape == out_builtin.shape}")
    print(f"Hidden shapes match: {h_manual.shape == h_builtin.shape}")
    # Note: Values won't match due to different weight initialization

verify_manual_implementation()
```

## Practical Applications

### Text Classification with BiLSTM

```python
class BiLSTMClassifier(nn.Module):
    """
    Production-ready BiLSTM classifier with multiple pooling strategies.
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes,
                 num_layers=2, dropout=0.3, pooling='last'):
        super().__init__()
        self.pooling = pooling
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Pooling-dependent output dimension
        if pooling in ['last', 'max', 'mean']:
            fc_input = hidden_size * 2
        elif pooling == 'concat_last_max_mean':
            fc_input = hidden_size * 2 * 3
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len) token indices
            lengths: Optional true lengths for masking
        """
        embedded = self.dropout(self.embedding(x))
        
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output_packed, (h_n, _) = self.bilstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output_packed, batch_first=True
            )
        else:
            output, (h_n, _) = self.bilstm(embedded)
        
        # Pooling
        if self.pooling == 'last':
            # Concatenate final states from both directions
            pooled = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        
        elif self.pooling == 'max':
            # Max pooling over sequence
            if lengths is not None:
                # Mask padding positions
                mask = torch.arange(output.size(1)).expand(
                    len(lengths), -1
                ).to(lengths.device) >= lengths.unsqueeze(1)
                output = output.masked_fill(mask.unsqueeze(-1), float('-inf'))
            pooled = output.max(dim=1)[0]
        
        elif self.pooling == 'mean':
            # Mean pooling over sequence
            if lengths is not None:
                mask = torch.arange(output.size(1)).expand(
                    len(lengths), -1
                ).to(lengths.device) < lengths.unsqueeze(1)
                output_sum = (output * mask.unsqueeze(-1).float()).sum(dim=1)
                pooled = output_sum / lengths.unsqueeze(-1).float()
            else:
                pooled = output.mean(dim=1)
        
        elif self.pooling == 'concat_last_max_mean':
            # Combine multiple pooling strategies
            last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            max_pool = output.max(dim=1)[0]
            mean_pool = output.mean(dim=1)
            pooled = torch.cat([last, max_pool, mean_pool], dim=-1)
        
        logits = self.fc(self.dropout(pooled))
        return logits


# Example with different pooling strategies
for pooling in ['last', 'max', 'mean', 'concat_last_max_mean']:
    model = BiLSTMClassifier(
        vocab_size=10000, embed_dim=128, hidden_size=256,
        num_classes=5, pooling=pooling
    )
    x = torch.randint(0, 10000, (32, 100))
    out = model(x)
    print(f"Pooling={pooling:<25} Output: {out.shape}")
```

### Sequence Labeling (NER/POS Tagging)

```python
class BiLSTMTagger(nn.Module):
    """
    BiLSTM for sequence labeling with optional CRF layer.
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_tags,
                 num_layers=2, dropout=0.3, use_crf=False):
        super().__init__()
        self.use_crf = use_crf
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_tags)
        
        if use_crf:
            # CRF transition parameters
            self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
            self.start_transitions = nn.Parameter(torch.randn(num_tags))
            self.end_transitions = nn.Parameter(torch.randn(num_tags))
    
    def forward(self, x, tags=None, mask=None):
        """
        Args:
            x: (batch, seq_len) token indices
            tags: Optional (batch, seq_len) gold tags for training
            mask: Optional (batch, seq_len) attention mask
        
        Returns:
            If training with CRF: negative log likelihood loss
            Otherwise: (batch, seq_len, num_tags) emission scores
        """
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.bilstm(embedded)
        emissions = self.fc(self.dropout(lstm_out))
        
        if self.use_crf and tags is not None:
            # Return CRF loss
            return -self._crf_log_likelihood(emissions, tags, mask)
        
        return emissions
    
    def decode(self, x, mask=None):
        """Viterbi decoding for best tag sequence."""
        emissions = self.forward(x)
        
        if self.use_crf:
            return self._viterbi_decode(emissions, mask)
        else:
            return emissions.argmax(dim=-1)
    
    def _crf_log_likelihood(self, emissions, tags, mask):
        """Compute CRF log likelihood (simplified)."""
        # Full implementation would include forward algorithm
        # and score computation
        pass
    
    def _viterbi_decode(self, emissions, mask):
        """Viterbi algorithm for best path (simplified)."""
        # Full implementation would include dynamic programming
        pass


# Usage
tagger = BiLSTMTagger(
    vocab_size=10000, embed_dim=100, hidden_size=128, 
    num_tags=9, use_crf=False
)

x = torch.randint(0, 10000, (32, 50))
emissions = tagger(x)
predictions = emissions.argmax(dim=-1)
print(f"Emissions: {emissions.shape}")      # (32, 50, 9)
print(f"Predictions: {predictions.shape}")  # (32, 50)
```

### BiLSTM Encoder for Seq2Seq

```python
class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for sequence-to-sequence models.
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=2,
                 dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Bridge: compress bidirectional states for unidirectional decoder
        self.bridge_h = nn.Linear(hidden_size * 2, hidden_size)
        self.bridge_c = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, src, src_lengths=None):
        """
        Encode source sequence.
        
        Returns:
            encoder_outputs: (batch, seq_len, hidden*2) for attention
            decoder_init: ((num_layers, batch, hidden), (num_layers, batch, hidden))
        """
        embedded = self.dropout(self.embedding(src))
        
        if src_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs_packed, (h_n, c_n) = self.bilstm(packed)
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs_packed, batch_first=True
            )
        else:
            encoder_outputs, (h_n, c_n) = self.bilstm(embedded)
        
        # Prepare decoder initial state
        # h_n: (num_layers*2, batch, hidden) -> (num_layers, batch, hidden)
        batch_size = src.size(0)
        
        # Reshape and bridge
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
        c_n = c_n.view(self.num_layers, 2, batch_size, self.hidden_size)
        
        # Concatenate directions and project
        h_combined = torch.cat([h_n[:, 0], h_n[:, 1]], dim=-1)  # (layers, batch, hidden*2)
        c_combined = torch.cat([c_n[:, 0], c_n[:, 1]], dim=-1)
        
        h_init = torch.tanh(self.bridge_h(h_combined))  # (layers, batch, hidden)
        c_init = torch.tanh(self.bridge_c(c_combined))
        
        return encoder_outputs, (h_init, c_init)
```

## Attention Mechanisms with BiLSTM

### Self-Attention Pooling

```python
class BiLSTMWithSelfAttention(nn.Module):
    """
    BiLSTM with self-attention for weighted pooling.
    """
    
    def __init__(self, input_size, hidden_size, num_classes, 
                 attention_heads=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        
        self.bilstm = nn.LSTM(
            input_size, hidden_size, batch_first=True, bidirectional=True
        )
        
        # Multi-head attention
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
            for _ in range(attention_heads)
        ])
        
        self.fc = nn.Linear(hidden_size * 2 * attention_heads, num_classes)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            mask: Optional (batch, seq_len) padding mask
        
        Returns:
            logits: (batch, num_classes)
            attention_weights: List of (batch, seq_len) per head
        """
        output, _ = self.bilstm(x)  # (batch, seq, hidden*2)
        
        contexts = []
        attention_weights = []
        
        for attention in self.attention_layers:
            # Compute attention scores
            scores = attention(output).squeeze(-1)  # (batch, seq)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            
            weights = torch.softmax(scores, dim=-1)
            attention_weights.append(weights)
            
            # Weighted sum
            context = torch.bmm(weights.unsqueeze(1), output).squeeze(1)
            contexts.append(context)
        
        # Concatenate all heads
        combined = torch.cat(contexts, dim=-1)
        logits = self.fc(combined)
        
        return logits, attention_weights


def visualize_attention(model, x, tokens):
    """Visualize attention weights."""
    import matplotlib.pyplot as plt
    
    model.eval()
    with torch.no_grad():
        _, attention_weights = model(x.unsqueeze(0))
    
    fig, axes = plt.subplots(1, len(attention_weights), figsize=(5*len(attention_weights), 4))
    if len(attention_weights) == 1:
        axes = [axes]
    
    for i, (ax, weights) in enumerate(zip(axes, attention_weights)):
        weights_np = weights[0].numpy()
        ax.bar(range(len(tokens)), weights_np)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'Head {i+1}')
    
    plt.tight_layout()
    plt.show()
```

### Cross-Attention for Seq2Seq

```python
class AttentionDecoder(nn.Module):
    """
    Decoder with attention over bidirectional encoder outputs.
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_size, 
                 encoder_hidden_size, attention_type='additive'):
        super().__init__()
        self.attention_type = attention_type
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Decoder LSTM input: embedding + context
        self.lstm = nn.LSTM(
            embed_dim + encoder_hidden_size * 2,  # BiLSTM encoder
            hidden_size, batch_first=True
        )
        
        # Attention
        if attention_type == 'additive':  # Bahdanau
            self.attention_W = nn.Linear(hidden_size, hidden_size)
            self.attention_U = nn.Linear(encoder_hidden_size * 2, hidden_size)
            self.attention_v = nn.Linear(hidden_size, 1)
        elif attention_type == 'multiplicative':  # Luong
            self.attention_W = nn.Linear(hidden_size, encoder_hidden_size * 2)
        
        self.output = nn.Linear(hidden_size + encoder_hidden_size * 2, vocab_size)
    
    def attention(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Compute attention weights over encoder outputs.
        
        Args:
            decoder_hidden: (batch, hidden)
            encoder_outputs: (batch, src_len, encoder_hidden*2)
            mask: Optional (batch, src_len) source padding mask
        
        Returns:
            context: (batch, encoder_hidden*2)
            weights: (batch, src_len)
        """
        batch_size, src_len, _ = encoder_outputs.shape
        
        if self.attention_type == 'additive':
            # Bahdanau attention
            decoder_proj = self.attention_W(decoder_hidden).unsqueeze(1)  # (batch, 1, hidden)
            encoder_proj = self.attention_U(encoder_outputs)  # (batch, src_len, hidden)
            scores = self.attention_v(torch.tanh(decoder_proj + encoder_proj)).squeeze(-1)
        
        elif self.attention_type == 'multiplicative':
            # Luong attention
            decoder_proj = self.attention_W(decoder_hidden).unsqueeze(2)  # (batch, enc_hidden*2, 1)
            scores = torch.bmm(encoder_outputs, decoder_proj).squeeze(-1)  # (batch, src_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, weights
```

## Multi-Layer Bidirectional Networks

### Deep BiLSTM with Residual Connections

```python
class DeepBiLSTM(nn.Module):
    """
    Deep bidirectional LSTM with residual connections and layer normalization.
    """
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Input projection for residual
        self.input_proj = nn.Linear(input_size, hidden_size * 2)
        
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for layer in range(num_layers):
            layer_input = hidden_size * 2  # All layers: bidirectional output
            
            self.layers.append(
                nn.LSTM(layer_input, hidden_size, 1,
                        batch_first=True, bidirectional=True)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_size * 2))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward with residual connections.
        """
        # Project input for residual compatibility
        x = self.input_proj(x)
        
        for i, (lstm, ln) in enumerate(zip(self.layers, self.layer_norms)):
            residual = x
            
            lstm_out, _ = lstm(x)
            lstm_out = ln(lstm_out)
            lstm_out = self.dropout(lstm_out)
            
            # Residual connection
            x = lstm_out + residual
        
        return x


class HighwayBiLSTM(nn.Module):
    """
    BiLSTM with highway connections for better gradient flow.
    """
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_size * 2)
        
        self.layers = nn.ModuleList()
        self.highway_gates = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(
                nn.LSTM(hidden_size * 2, hidden_size, 1,
                        batch_first=True, bidirectional=True)
            )
            self.highway_gates.append(
                nn.Linear(hidden_size * 2, hidden_size * 2)
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.input_proj(x)
        
        for lstm, gate_layer in zip(self.layers, self.highway_gates):
            # Transform
            lstm_out, _ = lstm(x)
            lstm_out = self.dropout(lstm_out)
            
            # Highway gate
            gate = torch.sigmoid(gate_layer(x))
            
            # Highway connection
            x = gate * lstm_out + (1 - gate) * x
        
        return x
```

## Computational Considerations

### Performance Analysis

```python
def analyze_bidirectional_cost():
    """
    Analyze computational cost of bidirectional vs unidirectional.
    """
    import time
    
    configs = [
        {'hidden': 128, 'layers': 1, 'seq': 100},
        {'hidden': 256, 'layers': 2, 'seq': 100},
        {'hidden': 256, 'layers': 2, 'seq': 500},
        {'hidden': 512, 'layers': 3, 'seq': 200},
    ]
    
    batch_size = 32
    input_size = 100
    num_runs = 50
    
    print("Bidirectional vs Unidirectional Performance")
    print("=" * 80)
    print(f"{'Config':<25} {'Uni Params':<12} {'Bi Params':<12} {'Uni Time':<12} {'Bi Time':<12} {'Ratio':<8}")
    print("-" * 80)
    
    for cfg in configs:
        uni = nn.LSTM(input_size, cfg['hidden'], cfg['layers'], batch_first=True)
        bi = nn.LSTM(input_size, cfg['hidden'], cfg['layers'], 
                     batch_first=True, bidirectional=True)
        
        x = torch.randn(batch_size, cfg['seq'], input_size)
        
        # Warmup
        for _ in range(5):
            _ = uni(x)
            _ = bi(x)
        
        # Benchmark
        start = time.time()
        for _ in range(num_runs):
            _ = uni(x)
        uni_time = (time.time() - start) / num_runs * 1000
        
        start = time.time()
        for _ in range(num_runs):
            _ = bi(x)
        bi_time = (time.time() - start) / num_runs * 1000
        
        uni_params = sum(p.numel() for p in uni.parameters())
        bi_params = sum(p.numel() for p in bi.parameters())
        
        config_str = f"h={cfg['hidden']},L={cfg['layers']},s={cfg['seq']}"
        print(f"{config_str:<25} {uni_params:<12,} {bi_params:<12,} "
              f"{uni_time:<12.2f} {bi_time:<12.2f} {bi_time/uni_time:<8.2f}x")
    
    print("\nKey Observations:")
    print("- Bidirectional has 2x parameters")
    print("- Time overhead is ~2x (two sequential passes)")
    print("- Memory overhead is also ~2x")

# analyze_bidirectional_cost()
```

### Memory-Efficient Bidirectional

```python
class MemoryEfficientBiLSTM(nn.Module):
    """
    Memory-efficient bidirectional LSTM using gradient checkpointing.
    """
    
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True
        )
    
    def forward(self, x, use_checkpointing=True):
        if use_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, use_reentrant=False
            )
        return self._forward(x)
    
    def _forward(self, x):
        return self.bilstm(x)
```

## When to Use Bidirectional

### Suitable Tasks

| Task | Suitability | Reason |
|------|-------------|--------|
| Text Classification | ✅ Excellent | Full context improves understanding |
| Named Entity Recognition | ✅ Excellent | Entity type often depends on surrounding context |
| Sentiment Analysis | ✅ Excellent | Negation/intensifiers require full context |
| Machine Translation (Encoder) | ✅ Excellent | Source context needed for alignment |
| Speech Recognition (Offline) | ✅ Good | Future acoustic context helps |
| Question Answering | ✅ Good | Answer depends on entire passage |

### Not Suitable For

| Task | Suitability | Reason |
|------|-------------|--------|
| Language Modeling | ❌ Not applicable | Cannot see future tokens |
| Autoregressive Generation | ❌ Not applicable | Must predict based on past only |
| Real-time Streaming | ❌ Not applicable | Future not available |
| Online Speech Recognition | ❌ Not applicable | Low-latency requirement |

### Decision Framework

```python
def should_use_bidirectional(task_properties):
    """
    Decision helper for bidirectional RNNs.
    
    Args:
        task_properties: dict with keys:
            - 'full_sequence_at_inference': bool
            - 'requires_future_context': bool
            - 'latency_critical': bool
            - 'memory_constrained': bool
    """
    if not task_properties['full_sequence_at_inference']:
        return False, "Bidirectional requires full sequence at inference time"
    
    if task_properties['latency_critical']:
        return False, "Bidirectional doubles computation time"
    
    if task_properties['memory_constrained']:
        return False, "Bidirectional doubles memory usage"
    
    if task_properties['requires_future_context']:
        return True, "Future context will improve performance"
    
    return True, "Default recommendation: try bidirectional first"
```

## Summary

Bidirectional RNNs provide richer sequence representations by processing in both directions:

### Key Properties
- **Output dimension**: $2 \times \text{hidden\_size}$ (concatenation)
- **Parameters**: $2 \times$ unidirectional
- **Computation**: $\sim 2 \times$ slower
- **Memory**: $\sim 2 \times$ more

### When to Use
- Full sequence available at inference
- Task benefits from future context
- Classification, labeling, or encoding tasks
- Not latency-critical

### Best Practices
1. Always use for encoder in seq2seq
2. Combine with attention for best results
3. Use gradient checkpointing for long sequences
4. Consider pooling strategies (last, max, mean, attention)
5. Bridge bidirectional encoder to unidirectional decoder carefully

### Implementation Checklist
- [ ] Set `bidirectional=True` in LSTM/GRU
- [ ] Handle doubled output dimension in subsequent layers
- [ ] Extract final states correctly: `h_n[-2]` (fwd), `h_n[-1]` (bwd)
- [ ] Use packed sequences for variable-length inputs
- [ ] Apply appropriate masking for attention
