# Conditional Random Fields for Sequence Labeling

## Learning Objectives

By the end of this section, you will be able to:

- Derive the mathematical formulation of Linear-Chain CRFs
- Understand the relationship between CRFs and neural sequence models
- Implement CRF layers in PyTorch with efficient forward algorithms
- Train CRF models using negative log-likelihood loss
- Perform Viterbi decoding for optimal sequence prediction
- Integrate CRF layers with BiLSTM and Transformer architectures

## Introduction

Conditional Random Fields (CRFs) are discriminative probabilistic models for sequence labeling that model the conditional probability $P(\mathbf{Y}|\mathbf{X})$ directly. Unlike independent token classifiers, CRFs capture dependencies between adjacent labels, making them particularly effective for NER where tag transitions follow specific patterns (e.g., I-PER can only follow B-PER or I-PER).

## Mathematical Foundation

### Problem Setup

Given:
- Input sequence: $\mathbf{X} = (x_1, x_2, \ldots, x_n)$
- Output labels: $\mathbf{Y} = (y_1, y_2, \ldots, y_n)$
- Label set: $\mathcal{L} = \{l_1, l_2, \ldots, l_k\}$

### Linear-Chain CRF Model

The conditional probability is defined as:

$$
P(\mathbf{Y}|\mathbf{X}) = \frac{1}{Z(\mathbf{X})} \exp\left(\sum_{i=1}^{n} s(y_i, \mathbf{X}, i) + \sum_{i=1}^{n} t(y_{i-1}, y_i)\right)
$$

Where:
- $s(y_i, \mathbf{X}, i)$: **Emission score** - how likely is label $y_i$ at position $i$ given input
- $t(y_{i-1}, y_i)$: **Transition score** - how likely is transitioning from $y_{i-1}$ to $y_i$
- $Z(\mathbf{X})$: **Partition function** - normalization constant

### Score Function

The total score for a sequence is:

$$
\text{Score}(\mathbf{X}, \mathbf{Y}) = \sum_{i=1}^{n} E_{y_i, i} + \sum_{i=1}^{n} T_{y_{i-1}, y_i}
$$

Where:
- $\mathbf{E} \in \mathbb{R}^{n \times k}$: Emission matrix from neural encoder
- $\mathbf{T} \in \mathbb{R}^{k \times k}$: Transition matrix (learnable parameters)

### Partition Function

The partition function sums over all possible label sequences:

$$
Z(\mathbf{X}) = \sum_{\mathbf{Y}' \in \mathcal{L}^n} \exp\left(\text{Score}(\mathbf{X}, \mathbf{Y}')\right)
$$

Direct computation is $O(k^n)$ - intractable. We use the **forward algorithm**.

### Forward Algorithm

Define forward variables:

$$
\alpha_i(y) = \sum_{\mathbf{Y}_{1:i-1}} \exp\left(\sum_{j=1}^{i} E_{y_j, j} + \sum_{j=2}^{i} T_{y_{j-1}, y_j}\right)
$$

This represents the sum of scores for all partial sequences ending with label $y$ at position $i$.

**Recurrence Relation**:

$$
\alpha_1(y) = \exp(E_{y,1} + T_{\text{START}, y})
$$

$$
\alpha_i(y) = \sum_{y' \in \mathcal{L}} \alpha_{i-1}(y') \cdot \exp(T_{y', y} + E_{y, i})
$$

**Partition Function**:

$$
Z(\mathbf{X}) = \sum_{y \in \mathcal{L}} \alpha_n(y) \cdot \exp(T_{y, \text{END}})
$$

**Complexity**: $O(n \cdot k^2)$ - linear in sequence length, quadratic in label count.

### Log-Space Computation

For numerical stability, compute in log-space:

$$
\log \alpha_i(y) = \text{logsumexp}_{y'}\left(\log \alpha_{i-1}(y') + T_{y', y}\right) + E_{y, i}
$$

Where:
$$
\text{logsumexp}(a_1, \ldots, a_m) = \log\left(\sum_{j=1}^{m} \exp(a_j)\right)
$$

## Loss Function

### Negative Log-Likelihood

The training objective minimizes the negative log-likelihood:

$$
\mathcal{L} = -\log P(\mathbf{Y}^*|\mathbf{X}) = -\text{Score}(\mathbf{X}, \mathbf{Y}^*) + \log Z(\mathbf{X})
$$

Where $\mathbf{Y}^*$ is the ground truth sequence.

### Gradient Computation

The gradient with respect to emission scores:

$$
\frac{\partial \mathcal{L}}{\partial E_{y,i}} = P(y_i = y | \mathbf{X}) - \mathbb{1}[y_i^* = y]
$$

The gradient with respect to transition scores:

$$
\frac{\partial \mathcal{L}}{\partial T_{y', y}} = \sum_{i=2}^{n} P(y_{i-1} = y', y_i = y | \mathbf{X}) - \sum_{i=2}^{n} \mathbb{1}[y_{i-1}^* = y', y_i^* = y]
$$

## Viterbi Decoding

At inference, find the most likely sequence:

$$
\mathbf{Y}^* = \arg\max_{\mathbf{Y}} P(\mathbf{Y}|\mathbf{X}) = \arg\max_{\mathbf{Y}} \text{Score}(\mathbf{X}, \mathbf{Y})
$$

### Viterbi Algorithm

Define:

$$
v_i(y) = \max_{\mathbf{Y}_{1:i-1}} \text{Score}(\mathbf{X}_{1:i}, \mathbf{Y}_{1:i-1}, y)
$$

**Recurrence**:

$$
v_1(y) = E_{y,1} + T_{\text{START}, y}
$$

$$
v_i(y) = \max_{y' \in \mathcal{L}}\left(v_{i-1}(y') + T_{y', y}\right) + E_{y, i}
$$

**Backpointers**:

$$
b_i(y) = \arg\max_{y' \in \mathcal{L}}\left(v_{i-1}(y') + T_{y', y}\right)
$$

**Backtracking**:

$$
y_n^* = \arg\max_{y}\left(v_n(y) + T_{y, \text{END}}\right)
$$

$$
y_i^* = b_{i+1}(y_{i+1}^*) \quad \text{for } i = n-1, \ldots, 1
$$

## PyTorch Implementation

### CRF Layer Module

```python
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class CRF(nn.Module):
    """
    Conditional Random Field layer for sequence labeling.
    
    This implementation supports:
    - Batched computation with padding
    - Log-space forward algorithm for numerical stability
    - Viterbi decoding for inference
    - Transition constraints via masking
    """
    
    def __init__(
        self,
        num_tags: int,
        batch_first: bool = True,
        pad_tag_id: Optional[int] = None
    ):
        """
        Initialize CRF layer.
        
        Args:
            num_tags: Number of tags (including START and END if used)
            batch_first: If True, input is (batch, seq, features)
            pad_tag_id: Tag ID for padding (excluded from loss)
        """
        super().__init__()
        
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.pad_tag_id = pad_tag_id
        
        # Transition matrix: transitions[i, j] = score of j -> i
        # (next_tag, current_tag) indexing for efficient computation
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Start and end transition scores
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters with uniform distribution."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
    
    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.
        
        Args:
            emissions: Emission scores (batch, seq_len, num_tags)
            tags: Ground truth tags (batch, seq_len)
            mask: Boolean mask (batch, seq_len), True for valid positions
            reduction: 'none', 'mean', or 'sum'
            
        Returns:
            Loss value (scalar or per-sample depending on reduction)
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        
        # Compute score of gold sequence
        gold_score = self._compute_score(emissions, tags, mask)
        
        # Compute partition function (log-sum-exp over all sequences)
        partition = self._compute_partition(emissions, mask)
        
        # Negative log-likelihood
        nll = partition - gold_score
        
        if reduction == 'none':
            return nll
        elif reduction == 'mean':
            return nll.mean()
        elif reduction == 'sum':
            return nll.sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute score of gold tag sequence.
        
        Args:
            emissions: (batch, seq_len, num_tags)
            tags: (batch, seq_len)
            mask: (batch, seq_len)
            
        Returns:
            Score for each sequence in batch (batch,)
        """
        batch_size, seq_len, _ = emissions.shape
        
        # Start transition score
        score = self.start_transitions[tags[:, 0]]
        
        # Emission score at first position
        score += emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        
        for i in range(1, seq_len):
            # Transition score: tags[i-1] -> tags[i]
            # Only add for valid positions
            trans_score = self.transitions[tags[:, i], tags[:, i-1]]
            
            # Emission score at position i
            emit_score = emissions[:, i].gather(1, tags[:, i:i+1]).squeeze(1)
            
            # Mask: only update for valid positions
            score += (trans_score + emit_score) * mask[:, i].float()
        
        # End transition score
        # Find last valid position for each sequence
        seq_lengths = mask.sum(dim=1).long()
        last_tags = tags.gather(1, (seq_lengths - 1).unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        
        return score
    
    def _compute_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log partition function using forward algorithm.
        
        Args:
            emissions: (batch, seq_len, num_tags)
            mask: (batch, seq_len)
            
        Returns:
            Log partition function for each sequence (batch,)
        """
        batch_size, seq_len, num_tags = emissions.shape
        
        # Initialize with start transitions + first emissions
        # alpha: (batch, num_tags)
        alpha = self.start_transitions + emissions[:, 0]
        
        for i in range(1, seq_len):
            # Broadcast for all possible transitions
            # alpha_expand: (batch, num_tags, 1)
            # transitions: (num_tags, num_tags) -> broadcast to (batch, num_tags, num_tags)
            # emissions_expand: (batch, 1, num_tags)
            
            alpha_expand = alpha.unsqueeze(2)  # (batch, num_tags, 1)
            emit_scores = emissions[:, i].unsqueeze(1)  # (batch, 1, num_tags)
            
            # Score for transitioning from any tag to any tag
            # (batch, num_tags, num_tags)
            scores = alpha_expand + self.transitions + emit_scores
            
            # Log-sum-exp over previous tags
            new_alpha = torch.logsumexp(scores, dim=1)  # (batch, num_tags)
            
            # Mask: keep old alpha for padded positions
            mask_i = mask[:, i].unsqueeze(1)  # (batch, 1)
            alpha = torch.where(mask_i, new_alpha, alpha)
        
        # Add end transitions and compute final log-sum-exp
        alpha = alpha + self.end_transitions
        return torch.logsumexp(alpha, dim=1)
    
    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Find most likely tag sequence using Viterbi algorithm.
        
        Args:
            emissions: (batch, seq_len, num_tags)
            mask: (batch, seq_len)
            
        Returns:
            List of tag sequences for each item in batch
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool,
                            device=emissions.device)
        
        return self._viterbi_decode(emissions, mask)
    
    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> List[List[int]]:
        """
        Viterbi decoding implementation.
        
        Args:
            emissions: (batch, seq_len, num_tags)
            mask: (batch, seq_len)
            
        Returns:
            List of best tag sequences
        """
        batch_size, seq_len, num_tags = emissions.shape
        
        # Initialize with start transitions
        # score: (batch, num_tags)
        score = self.start_transitions + emissions[:, 0]
        
        # Store backpointers
        history = []
        
        for i in range(1, seq_len):
            # Broadcast for all transitions
            # score: (batch, num_tags, 1)
            # transitions: (num_tags, num_tags)
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[:, i].unsqueeze(1)
            
            # Score for each (prev_tag, current_tag) pair
            # (batch, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission
            
            # Best previous tag for each current tag
            # (batch, num_tags)
            next_score, indices = next_score.max(dim=1)
            
            # Store backpointers
            history.append(indices)
            
            # Mask: keep old scores for padded positions
            mask_i = mask[:, i].unsqueeze(1)
            score = torch.where(mask_i, next_score, score)
        
        # Add end transitions
        score = score + self.end_transitions
        
        # Get best last tag
        seq_lengths = mask.sum(dim=1).long()
        
        # Backtrack
        best_tags_list = []
        
        for idx in range(batch_size):
            # Best tag at last position
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]
            
            # Backtrack through history
            for hist in reversed(history[:seq_lengths[idx] - 1]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())
            
            # Reverse to get forward order
            best_tags.reverse()
            best_tags_list.append(best_tags)
        
        return best_tags_list
```

### BiLSTM-CRF Model

```python
class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF model for sequence labeling.
    
    Architecture:
        Embedding → BiLSTM → Linear → CRF
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.5,
        pad_token_id: int = 0,
        pad_tag_id: int = 0
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Bidirectional doubles hidden size
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Project LSTM output to tag space
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        
        # CRF layer
        self.crf = CRF(num_tags, batch_first=True, pad_tag_id=pad_tag_id)
    
    def _get_emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get emission scores from BiLSTM.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            
        Returns:
            Emission scores (batch, seq_len, num_tags)
        """
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)
        
        # Pack padded sequence for efficient LSTM computation
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embeddings, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        else:
            lstm_out, _ = self.lstm(embeddings)
        
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        
        return emissions
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss for training.
        
        Args:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len)
            attention_mask: (batch, seq_len)
            
        Returns:
            Negative log-likelihood loss
        """
        emissions = self._get_emissions(input_ids, attention_mask)
        
        mask = attention_mask.bool() if attention_mask is not None else None
        loss = self.crf(emissions, labels, mask=mask, reduction='mean')
        
        return loss
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Predict tag sequences using Viterbi decoding.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            
        Returns:
            List of predicted tag sequences
        """
        emissions = self._get_emissions(input_ids, attention_mask)
        
        mask = attention_mask.bool() if attention_mask is not None else None
        return self.crf.decode(emissions, mask=mask)
```

### Transformer-CRF Model

```python
from transformers import AutoModel

class TransformerCRF(nn.Module):
    """
    Transformer (BERT/RoBERTa) with CRF layer for NER.
    """
    
    def __init__(
        self,
        model_name: str,
        num_tags: int,
        dropout: float = 0.1,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        self.transformer = AutoModel.from_pretrained(model_name)
        
        if freeze_encoder:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        hidden_size = self.transformer.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
    
    def _get_emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get emission scores from transformer."""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        
        return emissions
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute CRF loss."""
        emissions = self._get_emissions(input_ids, attention_mask)
        loss = self.crf(emissions, labels, mask=attention_mask.bool())
        return loss
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[List[int]]:
        """Viterbi decode predictions."""
        emissions = self._get_emissions(input_ids, attention_mask)
        return self.crf.decode(emissions, mask=attention_mask.bool())
```

## Transition Constraints

### Applying IOB2 Constraints

```python
def apply_iob2_constraints(
    crf: CRF,
    tag_to_idx: dict,
    penalty: float = -10000.0
):
    """
    Apply IOB2 transition constraints to CRF transitions.
    
    Invalid transitions get large negative scores.
    
    Args:
        crf: CRF layer to modify
        tag_to_idx: Mapping from tag names to indices
        penalty: Score for invalid transitions
    """
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}
    num_tags = len(tag_to_idx)
    
    with torch.no_grad():
        for i in range(num_tags):
            for j in range(num_tags):
                from_tag = idx_to_tag[j]
                to_tag = idx_to_tag[i]
                
                # I-X can only follow B-X or I-X
                if to_tag.startswith('I-'):
                    entity_type = to_tag[2:]
                    valid_prev = {f'B-{entity_type}', f'I-{entity_type}'}
                    
                    if from_tag not in valid_prev:
                        crf.transitions.data[i, j] = penalty
        
        # Constrain start transitions: can't start with I-
        for i in range(num_tags):
            tag = idx_to_tag[i]
            if tag.startswith('I-'):
                crf.start_transitions.data[i] = penalty
```

## Training Loop

```python
def train_ner_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
):
    """
    Training loop for NER model with CRF.
    """
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            loss = model(input_ids, labels, attention_mask)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                predictions = model.predict(input_ids, attention_mask)
                
                # Collect predictions and labels
                for pred, label, mask in zip(
                    predictions,
                    labels.cpu().numpy(),
                    attention_mask.cpu().numpy()
                ):
                    seq_len = mask.sum()
                    all_preds.append(pred[:seq_len])
                    all_labels.append(label[:seq_len].tolist())
        
        # Compute F1 score
        f1 = compute_entity_f1(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Val F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  New best model saved!")
    
    return best_f1
```

## Computational Considerations

### Time Complexity

| Operation | Complexity |
|-----------|------------|
| Forward (partition function) | $O(n \cdot k^2)$ |
| Viterbi decoding | $O(n \cdot k^2)$ |
| Backward pass | $O(n \cdot k^2)$ |

Where $n$ is sequence length and $k$ is number of tags.

### Memory Considerations

- Transition matrix: $O(k^2)$ parameters
- Forward variables: $O(n \cdot k)$ per sequence
- Backpointers for Viterbi: $O(n \cdot k)$

### Batched Computation

The implementations above support batched computation for efficiency:
- Vectorized operations over batch dimension
- Masked computation for variable-length sequences
- GPU-accelerated matrix operations

## Summary

Conditional Random Fields enhance neural sequence labelers by:

1. **Modeling transitions**: Capture label dependencies through learnable transition scores
2. **Global normalization**: Ensure proper probability distribution over all sequences
3. **Structured prediction**: Find optimal sequences via Viterbi algorithm
4. **Constraint enforcement**: Prevent invalid tag transitions through masking

The combination of powerful neural encoders (BiLSTM, Transformers) with CRF decoders remains a strong approach for NER, particularly when label dependencies are important.

## References

1. Lafferty, J., McCallum, A., & Pereira, F. (2001). Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data. *ICML*.

2. Lample, G., et al. (2016). Neural Architectures for Named Entity Recognition. *NAACL-HLT*.

3. Ma, X., & Hovy, E. (2016). End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF. *ACL*.

4. Huang, Z., Xu, W., & Yu, K. (2015). Bidirectional LSTM-CRF Models for Sequence Tagging. *arXiv*.
