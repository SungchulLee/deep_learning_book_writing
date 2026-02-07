# Contextual Embeddings

## Overview

Classical word embedding models like Word2Vec and GloVe produce **static embeddings**, where each word receives a single fixed vector representation regardless of context. However, natural language is deeply contextual — a word like "bank" carries entirely different meanings in "river bank" versus "investment bank." Modern advancements address this fundamental limitation through **contextualized embeddings** that dynamically adapt word representations based on surrounding context. This section covers **ELMo**, **BERT**, **GPT**, the **transformer architecture** that underpins them, and the practice of **fine-tuning pre-trained models** for downstream tasks.

## Learning Objectives

By the end of this section, you will:

- Understand the limitations of static embeddings and why contextualization matters
- Explain how ELMo generates context-dependent representations using bidirectional LSTMs
- Describe BERT's masked language model objective and bidirectional transformer architecture
- Distinguish between BERT's bidirectional and GPT's autoregressive approaches
- Understand the self-attention mechanism that enables transformer-based embeddings
- Implement fine-tuning of pre-trained models for classification tasks in PyTorch

## From Static to Contextualized Embeddings

### The Polysemy Problem

Static embeddings assign a single vector to each word. For polysemous words, this creates a fundamental representational bottleneck:

- *"The prisoner sat in a dark cell."* → physical enclosure
- *"The red blood cell carries oxygen."* → biological unit
- *"Plug in your cell phone."* → mobile device

A static embedding $\mathbf{e}_{\text{cell}} \in \mathbb{R}^d$ must represent all three meanings simultaneously, typically settling on an average that poorly serves any individual usage.

**Formal Problem Statement:**

A contextualized embedding function conditions on the surrounding context $C$:

$$f_{\text{context}}(w, C) = \mathbf{h}_w^{(C)} \quad \text{(different vector for each context)}$$

### Timeline of Key Developments

| Year | Model | Key Innovation |
|------|-------|---------------|
| 2013 | Word2Vec | Neural prediction-based static embeddings |
| 2014 | GloVe | Global co-occurrence statistics |
| 2017 | Transformer | Self-attention replaces recurrence |
| 2018 | ELMo | Bidirectional LSTM contextualized embeddings |
| 2018 | GPT | Autoregressive transformer pre-training |
| 2018 | BERT | Bidirectional transformer with masked LM |
| 2019 | GPT-2 | Scaled autoregressive generation |
| 2020 | GPT-3 | In-context learning at scale |

## ELMo (Embeddings from Language Models)

**ELMo** (Peters et al., 2018) was one of the first major models to produce contextualized word embeddings using a deep **bidirectional LSTM**.

### Architecture

ELMo trains a two-layer bidirectional language model. For a sequence of $N$ tokens $(t_1, t_2, \ldots, t_N)$:

**Forward LM:** $p(t_1, \ldots, t_N) = \prod_{k=1}^{N} p(t_k \mid t_1, \ldots, t_{k-1})$

**Backward LM:** $p(t_1, \ldots, t_N) = \prod_{k=1}^{N} p(t_k \mid t_{k+1}, \ldots, t_N)$

The joint objective maximizes log-likelihood in both directions:

$$\mathcal{L} = \sum_{k=1}^{N} \left[ \log p(t_k \mid t_1, \ldots, t_{k-1}; \overrightarrow{\Theta}) + \log p(t_k \mid t_{k+1}, \ldots, t_N; \overleftarrow{\Theta}) \right]$$

### Layer Combination

For each token $t_k$, ELMo computes a task-specific weighted combination of all $L$ biLSTM layers:

$$\text{ELMo}_k^{\text{task}} = \gamma^{\text{task}} \sum_{j=0}^{L} s_j^{\text{task}} \, \mathbf{h}_{k,j}$$

where:

- $\mathbf{h}_{k,0}$: Character-level token embedding (layer 0)
- $\mathbf{h}_{k,j} = [\overrightarrow{\mathbf{h}}_{k,j}; \overleftarrow{\mathbf{h}}_{k,j}]$: Concatenation of forward and backward hidden states at layer $j$
- $s_j^{\text{task}}$: Softmax-normalized layer weights (learned per task)
- $\gamma^{\text{task}}$: Task-specific scalar

!!! info "Layer-wise Linguistic Information"

    Different biLSTM layers capture different levels of linguistic abstraction:
    
    - **Layer 0 (token embeddings):** Morphological features, character patterns
    - **Layer 1 (first biLSTM):** Syntactic information (POS tags, parse structure)
    - **Layer 2 (second biLSTM):** Semantic information (word sense, semantic role)

### PyTorch Implementation

```python
import torch
import torch.nn as nn


class ELMoModel(nn.Module):
    """
    Simplified ELMo-like model using a bidirectional LSTM.
    
    Real ELMo uses character-level CNN for token representations and
    residual connections between LSTM layers. This implementation
    captures the core idea: bidirectional context produces different
    embeddings for the same word in different contexts.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super(ELMoModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Token embedding layer (in real ELMo: character-level CNN)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM: output dim = hidden_dim * 2
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True,
            batch_first=True
        )
        
        # Task-specific layer weights (softmax-normalized)
        self.layer_weights = nn.Parameter(torch.ones(num_layers + 1))
        self.gamma = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        """
        Args:
            x: Token indices of shape (batch_size, seq_length)
        
        Returns:
            Contextualized embeddings of shape (batch_size, seq_length, hidden_dim * 2)
        """
        token_embeddings = self.embedding(x)  # (batch, seq, embed_dim)
        lstm_out, _ = self.lstm(token_embeddings)  # (batch, seq, hidden*2)
        
        # Simplified weighted combination (layer 0 + final LSTM output)
        weights = torch.softmax(self.layer_weights[:2], dim=0)
        
        layer_0 = torch.zeros_like(lstm_out)
        layer_0[:, :, :token_embeddings.size(-1)] = token_embeddings
        
        elmo_repr = self.gamma * (weights[0] * layer_0 + weights[1] * lstm_out)
        return elmo_repr


# ── Demonstrating context sensitivity ──────────────────────────────────
model = ELMoModel(vocab_size=5000, embedding_dim=100, hidden_dim=128)

sentence_a = torch.LongTensor([[45, 23, 78]])   # e.g., "the bank opened"
sentence_b = torch.LongTensor([[12, 23, 90]])    # e.g., "river bank flooded"

elmo_a = model(sentence_a)
elmo_b = model(sentence_b)

# Word at position 1 (index 23) gets DIFFERENT embeddings
emb_a = elmo_a[0, 1, :]
emb_b = elmo_b[0, 1, :]

cosine_sim = torch.nn.functional.cosine_similarity(
    emb_a.unsqueeze(0), emb_b.unsqueeze(0)
)
print(f"Same word, different contexts — cosine similarity: {cosine_sim.item():.4f}")
# Similarity < 1.0, demonstrating context sensitivity
```

## BERT (Bidirectional Encoder Representations from Transformers)

**BERT** (Devlin et al., 2019) replaced LSTMs with the **transformer encoder**, achieving deep bidirectionality through self-attention where every token attends to every other token simultaneously.

### Pre-training Objectives

**1. Masked Language Model (MLM):** Randomly mask 15% of tokens and predict them from context:

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(t_i \mid \mathbf{t}_{\backslash \mathcal{M}}; \theta)$$

Of the 15% selected tokens: 80% replaced with `[MASK]`, 10% replaced with random token, 10% unchanged.

**2. Next Sentence Prediction (NSP):** Predict whether sentence $B$ follows sentence $A$:

$$\mathcal{L}_{\text{NSP}} = -\left[ y \log P(\text{IsNext}) + (1-y) \log P(\text{NotNext}) \right]$$

**Combined:** $\mathcal{L}_{\text{BERT}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$

### Architecture Details

| Parameter | BERT-Base | BERT-Large |
|-----------|-----------|------------|
| Transformer layers ($L$) | 12 | 24 |
| Hidden size ($H$) | 768 | 1024 |
| Attention heads ($A$) | 12 | 16 |
| Total parameters | 110M | 340M |
| Tokenization | WordPiece (30K vocab) | WordPiece (30K vocab) |

### Input Representation

BERT's input sums three embedding types:

$$\mathbf{x}_i = \mathbf{E}_{\text{token}}(t_i) + \mathbf{E}_{\text{segment}}(s_i) + \mathbf{E}_{\text{position}}(i)$$

### PyTorch with Hugging Face

```python
from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# ── Contextualized embeddings ──────────────────────────────────────────
sentence = "The river bank was flooded after the storm."
inputs = tokenizer(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

print(f"Output shape: {last_hidden_states.shape}")
# torch.Size([1, 11, 768]) — (batch, tokens, hidden_size)

# ── Context sensitivity demonstration ──────────────────────────────────
sentence_a = "I deposited money at the bank."
sentence_b = "We walked along the river bank."

inputs_a = tokenizer(sentence_a, return_tensors="pt")
inputs_b = tokenizer(sentence_b, return_tensors="pt")

with torch.no_grad():
    out_a = model(**inputs_a).last_hidden_state
    out_b = model(**inputs_b).last_hidden_state

tokens_a = tokenizer.convert_ids_to_tokens(inputs_a['input_ids'][0])
tokens_b = tokenizer.convert_ids_to_tokens(inputs_b['input_ids'][0])

bank_idx_a = tokens_a.index('bank')
bank_idx_b = tokens_b.index('bank')

bank_emb_a = out_a[0, bank_idx_a, :]
bank_emb_b = out_b[0, bank_idx_b, :]

cosine_sim = torch.nn.functional.cosine_similarity(
    bank_emb_a.unsqueeze(0), bank_emb_b.unsqueeze(0)
)
print(f"'bank' financial vs. river context: cosine sim = {cosine_sim.item():.4f}")
# Typically around 0.6-0.7, showing BERT distinguishes word senses
```

!!! warning "BERT Produces Subword Tokens"

    BERT uses WordPiece tokenization, so words may be split: "unforgettable" → `["un", "##forget", "##table"]`. For word-level embeddings, aggregate subword vectors via mean pooling or use the first subword token.

## GPT (Generative Pre-trained Transformer)

**GPT** (Radford et al., 2018) uses an **autoregressive** approach — predicting the next token based only on preceding tokens:

$$\mathcal{L}_{\text{GPT}} = -\sum_{i=1}^{N} \log P(t_i \mid t_1, \ldots, t_{i-1}; \theta)$$

The model uses **causal (masked) self-attention** where each position attends only to earlier positions:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$

where $M_{ij} = 0$ if $i \geq j$ and $M_{ij} = -\infty$ otherwise.

### BERT vs. GPT

| Aspect | BERT | GPT |
|--------|------|-----|
| **Context** | Bidirectional (full) | Unidirectional (left only) |
| **Architecture** | Transformer encoder | Transformer decoder |
| **Pre-training** | Masked LM + NSP | Autoregressive LM |
| **Self-attention** | Full attention | Causal (masked) attention |
| **Primary strength** | Understanding / classification | Generation / completion |

## The Self-Attention Mechanism

The transformer architecture (Vaswani et al., 2017) underpins both BERT and GPT. For input $\mathbf{X} \in \mathbb{R}^{N \times d}$:

**Step 1 — Compute Query, Key, Value:**

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

**Step 2 — Scaled dot-product attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

The scaling factor $\frac{1}{\sqrt{d_k}}$ prevents dot products from growing too large.

**Multi-head attention** runs $h$ parallel heads:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

### Why Transformers Outperform RNNs for Embeddings

| Property | RNN/LSTM | Transformer |
|----------|----------|-------------|
| **Long-range dependencies** | Degrades with distance | Direct attention to any position |
| **Parallelization** | Sequential | Fully parallelizable |
| **Gradient flow** | Vanishing/exploding risk | Direct paths via residuals |
| **Complexity** | $O(N \cdot d^2)$ per layer | $O(N^2 \cdot d)$ per layer |

## Fine-Tuning Pre-Trained Models

### The Transfer Learning Paradigm

**Phase 1 — Pre-training:** Train on massive unsupervised data (days/weeks on many GPUs)

**Phase 2 — Fine-tuning:** Adapt to a specific task with smaller labeled data (minutes/hours on one GPU)

$$\theta^* = \arg\min_{\theta} \mathcal{L}_{\text{task}}(\theta) \quad \text{initialized at } \theta = \theta_{\text{pre}}$$

### Fine-Tuning Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Feature extraction** | Freeze pre-trained weights, train only task head | Very small labeled dataset |
| **Full fine-tuning** | Update all parameters | Moderate labeled dataset |
| **Gradual unfreezing** | Progressively unfreeze layers top to bottom | Risk of catastrophic forgetting |
| **Discriminative LR** | Lower LR for earlier layers, higher for later | Balance pre-trained knowledge with adaptation |

### PyTorch Fine-Tuning Example

```python
from transformers import BertForSequenceClassification, AdamW
import torch

# Load BERT with classification head
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)

# Discriminative learning rates
optimizer = AdamW([
    {'params': model.bert.parameters(), 'lr': 2e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4},
])

# Training step
input_ids = torch.tensor([
    [101, 2054, 2003, 1996, 2516, 102],
    [101, 2129, 2003, 2115, 2154, 102]
])
attention_mask = torch.ones_like(input_ids)
labels = torch.tensor([0, 1])

model.train()
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

loss = outputs.loss
logits = outputs.logits

loss.backward()
optimizer.step()
optimizer.zero_grad()

print(f"Loss: {loss.item():.4f}")
print(f"Predictions: {torch.argmax(logits, dim=-1)}")
```

### Feature Extraction vs. Fine-Tuning

```python
# Freeze all BERT parameters — train only classification head
model_frozen = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)

for param in model_frozen.bert.parameters():
    param.requires_grad = False

total = sum(p.numel() for p in model_frozen.parameters())
trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)

print(f"Total parameters:     {total:>12,}")
print(f"Trainable parameters: {trainable:>12,}")
# Total:     ~109,483,778
# Trainable: ~1,538  (only the classification head)
```

!!! tip "Fine-Tuning Best Practices"

    - Use a **small learning rate** (2e-5 to 5e-5) — large rates destroy pre-trained features
    - Train for only **2–4 epochs** — more risks overfitting on small datasets
    - Apply **linear warmup** for the learning rate scheduler
    - Use **weight decay** (0.01), excluding bias and LayerNorm parameters
    - Monitor **validation loss** and apply early stopping

## Key Takeaways

!!! success "Main Concepts"

    1. **Static vs. contextualized:** Static models assign one vector per word; contextualized models produce different vectors depending on context
    2. **ELMo** uses bidirectional LSTMs and combines all layer representations with task-specific weights
    3. **BERT** uses transformer encoders with masked language modeling for deep bidirectional contextualization
    4. **GPT** uses transformer decoders with autoregressive pre-training for generation
    5. **Self-attention** enables each token to attend directly to all others, capturing long-range dependencies
    6. **Fine-tuning** adapts pre-trained models to specific tasks with small labeled datasets

!!! tip "Practical Guidelines"

    - Use **BERT** for understanding tasks (classification, NER, QA) where full context is available
    - Use **GPT** for generation tasks (text completion, summarization, dialogue)
    - Start with **feature extraction** when labeled data is scarce, move to **full fine-tuning** with more data
    - Always use a **small learning rate** (2e-5 to 5e-5) when fine-tuning

## References

- Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
- Peters, M. E., et al. (2018). "Deep contextualized word representations." NAACL. *(ELMo)*
- Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training." *(GPT)*
- Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL.
- Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS. *(GPT-3)*
