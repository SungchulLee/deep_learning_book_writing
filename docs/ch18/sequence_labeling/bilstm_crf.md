# BiLSTM-CRF Architecture

## Learning Objectives

- Understand the complete BiLSTM-CRF pipeline for sequence labeling
- Implement end-to-end training with CRF loss
- Integrate character-level and word-level representations
- Train and evaluate on standard NER benchmarks

---

## Architecture Overview

The BiLSTM-CRF combines the representation power of bidirectional LSTMs with the structured prediction capability of CRFs:

```
Input tokens:    w₁      w₂      w₃      w₄
                  ↓       ↓       ↓       ↓
Character CNN:  [char]  [char]  [char]  [char]
                  ↓       ↓       ↓       ↓
Word Embedding: [emb]   [emb]   [emb]   [emb]
                  ↓       ↓       ↓       ↓
    Concat:     [word;char] → → → →
                  ↓       ↓       ↓       ↓
    BiLSTM:    →h₁→    →h₂→    →h₃→    →h₄→
               ←h₁←    ←h₂←    ←h₃←    ←h₄←
                  ↓       ↓       ↓       ↓
    Linear:     e₁      e₂      e₃      e₄    (emission scores)
                  ↓       ↓       ↓       ↓
    CRF:       [===============================]  (global decoding)
                  ↓       ↓       ↓       ↓
    Output:    B-PER   I-PER     O     B-ORG
```

---

## Mathematical Formulation

### Emission Scores

The BiLSTM produces emission scores for each token-label pair:

$$\mathbf{e}_i = \mathbf{W}_o [\overrightarrow{h}_i ; \overleftarrow{h}_i] + \mathbf{b}_o \in \mathbb{R}^{|\mathcal{L}|}$$

### CRF Objective

The conditional probability of a label sequence is:

$$P(\mathbf{y} | \mathbf{x}) = \frac{\exp\left(\sum_i e_{y_i,i} + \sum_i T_{y_{i-1},y_i}\right)}{\sum_{\mathbf{y}'} \exp\left(\sum_i e_{y'_i,i} + \sum_i T_{y'_{i-1},y'_i}\right)}$$

### Training Loss

$$\mathcal{L} = -\log P(\mathbf{y}^* | \mathbf{x}) = \log Z(\mathbf{x}) - \text{Score}(\mathbf{x}, \mathbf{y}^*)$$

---

## Complete Implementation

See the standalone implementations in:

- [CRF Layer](crf.md) — forward algorithm, Viterbi decoding, batched computation
- [BiLSTM for NER](bilstm_ner.md) — BiLSTM encoder with character embeddings

The full BiLSTM-CRF combines both components. Below is a reference training script:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Training configuration
config = {
    'embedding_dim': 100,
    'char_embedding_dim': 25,
    'hidden_dim': 256,
    'dropout': 0.5,
    'lr': 0.015,
    'momentum': 0.9,
    'epochs': 50,
    'clip_grad': 5.0,
    'lr_decay': 0.05,
}

def train_bilstm_crf(model, train_loader, val_loader, config, device):
    """Train BiLSTM-CRF model with learning rate decay."""
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr'],
        momentum=config['momentum']
    )

    best_f1 = 0.0

    for epoch in range(config['epochs']):
        # Learning rate decay
        lr = config['lr'] / (1 + config['lr_decay'] * epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        f1 = evaluate_ner(model, val_loader, device)
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, F1={f1:.4f}, lr={lr:.5f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_bilstm_crf.pt')

    return best_f1
```

---

## Benchmark Results

| Model | CoNLL-2003 F1 | Params |
|-------|---------------|--------|
| BiLSTM + softmax | 88.7 | ~5M |
| BiLSTM + CRF | 90.9 | ~5M |
| BiLSTM + char-CNN + CRF | 91.2 | ~6M |
| BiLSTM + char-LSTM + CRF | 90.9 | ~6M |
| BERT-base + softmax | 92.4 | 110M |
| BERT-base + CRF | 92.8 | 110M |

The CRF consistently provides a 1–2 F1 point improvement over softmax classification.

---

## Summary

1. BiLSTM-CRF is the classic neural architecture for sequence labeling
2. The CRF layer captures label dependencies that softmax ignores
3. Character-level features (CNN or LSTM) improve handling of morphology and OOV words
4. SGD with gradient clipping and learning rate decay is the standard training recipe
5. While Transformer models now dominate, BiLSTM-CRF remains a strong baseline

---

## References

1. Lample, G., et al. (2016). Neural Architectures for Named Entity Recognition. *NAACL-HLT*.
2. Ma, X., & Hovy, E. (2016). End-to-end Sequence Labeling via BiLSTM-CNNs-CRF. *ACL*.
3. Huang, Z., Xu, W., & Yu, K. (2015). Bidirectional LSTM-CRF Models for Sequence Tagging.
