# Sequential Recommendations

## Learning Objectives

- Understand why temporal order matters for recommendation quality
- Formulate sequential recommendation as a sequence prediction problem
- Implement session-based recommendations with GRU (GRU4Rec)
- Understand the Self-Attentive Sequential Recommendation (SASRec) architecture
- Connect sequential recommendation to language modeling and Transformer architectures

## Why Sequential Models?

The recommender systems covered in previous sections — matrix factorization, NCF, and hybrid methods — treat user–item interactions as an **unordered set**. They predict ratings from static user and item representations, ignoring the temporal dynamics of user behavior.

In reality, user preferences evolve over time:

- A user who just watched three horror movies is more likely to watch another horror movie than a comedy, regardless of their long-term genre preferences.
- A shopper who added running shoes to their cart is likely browsing athletic gear, not formal wear.
- An investor who just sold technology stocks may be rotating into defensive sectors.

Sequential recommender systems model these temporal dynamics by treating the user's interaction history as an **ordered sequence** and predicting the next interaction.

### From Static to Sequential

| Aspect | Static Models (MF, NCF) | Sequential Models |
|--------|------------------------|-------------------|
| **Input** | User ID, item ID | Ordered interaction sequence |
| **User representation** | Fixed embedding $\mathbf{p}_u$ | Dynamic hidden state $\mathbf{h}_t$ |
| **Captures** | Long-term preferences | Short-term dynamics + long-term |
| **Temporal awareness** | None | Explicit |
| **Architecture** | Embedding + MLP | RNN, Transformer, CNN |

## Problem Formulation

### Sequence Prediction

Given user $u$'s chronologically ordered interaction sequence:

$$\mathcal{S}_u = (s_1^u, s_2^u, \ldots, s_{T_u}^u)$$

where $s_t^u \in \{1, \ldots, n\}$ is the item the user interacted with at time step $t$, the goal is to predict the next item:

$$P(s_{T_u+1}^u = i \mid s_1^u, \ldots, s_{T_u}^u)$$

This is structurally identical to **language modeling** (Chapter 17): predict the next token given preceding tokens. Items play the role of words, and user interaction sequences play the role of sentences.

### Session-Based vs User-Based

**User-based sequential recommendation** maintains the full history of each user across sessions. The model conditions on the user's entire past.

**Session-based recommendation** treats each browsing session independently, without linking sessions to persistent user identities. This is common in e-commerce where many visitors are anonymous. The model must infer preferences from within-session behavior alone.

## GRU4Rec: GRU-Based Session Recommendations

Hidasi et al. (2016) introduced **GRU4Rec**, applying gated recurrent units (Chapter 18) to session-based recommendation. The architecture processes a sequence of item interactions and predicts the next item.

### Architecture

```python
class GRU4Rec(nn.Module):
    """
    GRU-based session recommendation model.
    
    Processes a sequence of item IDs through an embedding layer
    and GRU, then scores all candidate items against the final
    hidden state.
    
    Args:
        num_items:  Number of unique items.
        emb_size:   Item embedding dimension.
        hidden_size: GRU hidden state dimension.
        num_layers: Number of stacked GRU layers.
        dropout:    Dropout rate between GRU layers.
    """
    def __init__(self, num_items, emb_size=64, hidden_size=128,
                 num_layers=1, dropout=0.1):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_size, padding_idx=0)
        self.gru = nn.GRU(
            emb_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.output = nn.Linear(hidden_size, num_items)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, seq, lengths=None):
        """
        Args:
            seq: LongTensor of item IDs, shape (batch, max_len)
            lengths: Optional actual sequence lengths for packing
            
        Returns:
            Logits over all items, shape (batch, num_items)
        """
        x = self.item_emb(seq)          # (batch, max_len, emb_size)
        x = self.dropout(x)
        
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
        
        output, h_n = self.gru(x)       # h_n: (num_layers, batch, hidden_size)
        
        # Use the last hidden state for prediction
        h_last = h_n[-1]                # (batch, hidden_size)
        logits = self.output(h_last)    # (batch, num_items)
        return logits
```

### Training with BPR Loss

For implicit feedback (clicks, purchases), GRU4Rec is typically trained with **Bayesian Personalized Ranking (BPR)** loss or cross-entropy over the item catalog:

$$\mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j)} \log \sigma\bigl(f(u, i) - f(u, j)\bigr)$$

where $i$ is a positive (interacted) item and $j$ is a negative (sampled) item. This pairwise loss learns to rank positive items above negative ones.

Alternatively, using cross-entropy treats next-item prediction as a classification problem:

$$\mathcal{L}_{\text{CE}} = -\sum_{t} \log \frac{\exp(\mathbf{h}_t^\top \mathbf{q}_{s_{t+1}})}{\sum_{j=1}^n \exp(\mathbf{h}_t^\top \mathbf{q}_j)}$$

where $\mathbf{h}_t$ is the hidden state at step $t$ and $\mathbf{q}_j$ is the embedding of item $j$.

### Limitations of RNN-Based Approaches

GRU4Rec suffers from the same limitations as RNNs in NLP:

1. **Sequential computation**: Cannot parallelize across time steps during training.
2. **Long-range dependencies**: Despite gating, GRUs struggle to capture dependencies across very long sequences.
3. **Fixed-size bottleneck**: The entire session history is compressed into a single hidden state vector.

## SASRec: Self-Attentive Sequential Recommendation

Kang and McAuley (2018) proposed **SASRec**, applying the Transformer's self-attention mechanism (Chapter 19) to sequential recommendation. SASRec addresses the limitations of RNN-based models by attending to all previous items simultaneously.

### Architecture

SASRec adapts the Transformer decoder architecture:

```python
class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation.
    
    Uses causal (left-to-right) self-attention over the item
    sequence, analogous to a Transformer decoder for language
    modeling.
    
    Args:
        num_items:  Number of unique items.
        max_len:    Maximum sequence length.
        emb_size:   Item embedding and model dimension.
        num_heads:  Number of attention heads.
        num_layers: Number of Transformer blocks.
        dropout:    Dropout rate.
    """
    def __init__(self, num_items, max_len=50, emb_size=64,
                 num_heads=2, num_layers=2, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, emb_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, emb_size)
        self.emb_dropout = nn.Dropout(dropout)
        
        # Transformer encoder layers with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=emb_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        self.layer_norm = nn.LayerNorm(emb_size)
        self.output = nn.Linear(emb_size, num_items + 1)
        self.max_len = max_len
    
    def forward(self, seq):
        """
        Args:
            seq: LongTensor of item IDs, shape (batch, max_len)
            
        Returns:
            Logits at each position, shape (batch, max_len, num_items+1)
        """
        batch_size, seq_len = seq.shape
        
        # Item + positional embeddings
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.emb_dropout(x)
        
        # Causal mask: prevent attending to future items
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=seq.device
        )
        
        # Padding mask
        padding_mask = (seq == 0)  # True where padded
        
        x = self.transformer(
            x, mask=causal_mask, src_key_padding_mask=padding_mask
        )
        x = self.layer_norm(x)
        logits = self.output(x)     # (batch, max_len, num_items+1)
        return logits
```

### Key Design Decisions

**Causal masking**: Like a language model, SASRec uses a causal (lower-triangular) attention mask to ensure position $t$ can only attend to positions $\leq t$. This prevents information leakage from future items during training.

**Positional embeddings**: Since self-attention is permutation-invariant, learned positional embeddings encode the order of items in the sequence. This is identical to the positional encoding in Transformer language models.

**Shared item embeddings**: The same embedding table is used for both input (encoding the sequence) and output (scoring candidates). This ties the input and output representations, similar to weight tying in language models.

### Connection to Language Modeling

The analogy between sequential recommendation and language modeling is precise:

| Language Modeling | Sequential Recommendation |
|-------------------|--------------------------|
| Vocabulary $V$ | Item catalog $\{1, \ldots, n\}$ |
| Token sequence $(w_1, \ldots, w_T)$ | Interaction sequence $(s_1, \ldots, s_T)$ |
| Next-token prediction $P(w_{t+1} \mid w_{\leq t})$ | Next-item prediction $P(s_{t+1} \mid s_{\leq t})$ |
| Causal Transformer decoder | SASRec |
| Cross-entropy loss | Cross-entropy loss |

This connection means that advances in language modeling — better attention mechanisms, longer context windows, efficient training — translate directly to sequential recommendation.

## Advanced Sequential Architectures

### BERT4Rec

Bidirectional Encoder Representations from Transformers for Recommendation (Sun et al., 2019) adapts BERT's masked language modeling to recommendation. Instead of predicting the next item, it randomly masks items in the sequence and predicts the masked items:

$$\mathcal{L}_{\text{MLM}} = -\sum_{t \in \mathcal{M}} \log P(s_t \mid \mathcal{S}_u \setminus \{s_t\})$$

where $\mathcal{M}$ is the set of masked positions. This allows bidirectional attention (attending to both past and future items), which can capture richer patterns than causal attention alone.

### Incorporating Side Information

Sequential models can be augmented with item features, time intervals, and user features:

- **Time-aware attention**: Modify attention weights based on the time gap between interactions. Items interacted with recently should have higher attention weights.
- **Feature-enriched embeddings**: Concatenate item ID embeddings with content features before feeding into the sequence model.
- **User-conditioned models**: Add a user embedding that modulates the sequence processing, combining static preferences with dynamic behavior.

## Practical Considerations

### Sequence Length and Truncation

Long user histories must be truncated to a fixed maximum length $L$. The choice of $L$ trades off:

- **Short $L$ (e.g., 20)**: Captures only recent behavior. Fast training, but misses long-term patterns.
- **Long $L$ (e.g., 200)**: Captures long-term trends. Slow training, and attention complexity is $O(L^2)$.

In practice, $L = 50$ is a common default that balances these concerns.

### Negative Sampling

For large item catalogs, computing the full softmax over all items is expensive. Common alternatives include sampled softmax (randomly sample $k$ negative items per positive) and in-batch negatives (use other items in the same batch as negatives).

### Evaluation

Sequential models are evaluated with temporal awareness: the model predicts the next item given the history up to that point. Standard metrics include Hit Rate@K (whether the true next item appears in the top-$K$ predictions) and NDCG@K (Section 29.3.1).

## Sequential Recommendations in Finance

Sequential models are naturally suited to financial applications where order matters:

- **Trading pattern analysis**: Model the sequence of trades an investor makes to predict the next likely trade, enabling proactive research or product suggestions.
- **Portfolio rebalancing**: Capture temporal patterns in how investors adjust their portfolios across market regimes.
- **Research consumption**: Model the sequence of analyst reports a portfolio manager reads to recommend the next most relevant report.

## Summary

Sequential recommender systems capture temporal dynamics by modeling user interaction histories as ordered sequences. GRU4Rec applies RNNs to session-based recommendation, while SASRec leverages self-attention for parallel training and better long-range dependency modeling. The deep connection to language modeling means that Transformer-based architectures and training techniques transfer directly to the recommendation domain.

---

## Exercises

1. **Sequence construction**: Using the MovieLens dataset (which includes timestamps), construct chronologically ordered interaction sequences for each user. What is the average sequence length? What fraction of users have sequences longer than 50?

2. **GRU4Rec implementation**: Implement GRU4Rec and train it on MovieLens interaction sequences. Use cross-entropy loss with the next item as the target. Compare Hit Rate@10 against a popularity baseline.

3. **SASRec vs GRU4Rec**: Implement SASRec and compare it against GRU4Rec on the same data. Does self-attention improve over GRU for different sequence lengths? Plot Hit Rate@10 vs maximum sequence length.

4. **Attention visualization**: After training SASRec, extract and visualize the attention weights for a sample user's interaction sequence. Do the attention patterns reveal interpretable temporal patterns (e.g., higher attention on recent items)?

5. **Time-aware extension**: Modify SASRec to incorporate time intervals between interactions. Add a time embedding that encodes the gap $\Delta t = t_{k} - t_{k-1}$ between consecutive interactions. Does this improve prediction quality?

6. **Static + sequential**: Combine a static MF model (for long-term preferences) with a sequential model (for short-term dynamics). Use a simple weighted combination or a learned gating mechanism. Does the hybrid outperform either alone?
