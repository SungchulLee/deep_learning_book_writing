# BERT: Bidirectional Encoder Representations from Transformers

## Overview

BERT (Devlin et al., 2019) is an **encoder-only** Transformer that revolutionized NLP by introducing bidirectional pre-training. It demonstrated that pre-training on large unlabeled text, then fine-tuning on downstream tasks, yields state-of-the-art results.

## Architecture

BERT uses only the **Transformer encoder stack**:

$$\text{Input} \xrightarrow{\text{Embed + PE}} \mathbf{X} \xrightarrow{\text{Encoder} \times N} \mathbf{H}$$

| Configuration | Layers | Hidden | Heads | Parameters |
|---------------|--------|--------|-------|------------|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

### Key Characteristics

- **Bidirectional attention**: No causal mask—every token sees every other token
- **No generation**: Designed for understanding, not producing text
- **Special tokens**: [CLS] for classification, [SEP] for sentence separation

## Pre-training Objectives

### Masked Language Modeling (MLM)

Randomly mask 15% of tokens and predict them:

- Input: "The [MASK] sat on the [MASK]"
- Target: Predict [MASK] = "cat" and [MASK] = "mat"

**Masking strategy** (for selected 15%):
- 80%: Replace with [MASK]
- 10%: Replace with random token
- 10%: Keep original token

This prevents the model from only learning to recognize [MASK].

### Next Sentence Prediction (NSP)

Given two sentences, predict if B follows A:

- Input: "[CLS] The cat sat. [SEP] It was tired. [SEP]"
- Target: IsNext / NotNext

Later work (RoBERTa) showed NSP provides minimal benefit.

## Input Representation

```
Input:       [CLS]  The    cat   sat   [SEP]  It    was   [SEP]
              ↓      ↓      ↓     ↓      ↓     ↓      ↓     ↓
Token Emb:   E_cls  E_the  E_cat E_sat E_sep E_it  E_was E_sep
              +      +      +     +      +     +      +     +
Segment:     A      A      A     A      A     B      B     B
              +      +      +     +      +     +      +     +
Position:    0      1      2     3      4     5      6     7
```

Total embedding = Token + Segment + Position

## Pre-training Setup

- **Data**: BooksCorpus + English Wikipedia (~3.3B words)
- **Sequence length**: 512 tokens
- **Batch size**: 256
- **Steps**: 1M (about 40 epochs)
- **Optimizer**: Adam with warmup

## Fine-tuning for Downstream Tasks

### Classification

Use [CLS] token representation:

```
[CLS] This movie is great [SEP]
  ↓
Final layer [CLS] embedding
  ↓
Linear classifier → Positive/Negative
```

### Token Classification (NER)

Use each token's representation:

```
[CLS] John  lives in  Paris [SEP]
       ↓     ↓    ↓     ↓
      PER    O    O    LOC
```

### Question Answering

Predict start and end positions of answer span:

```
[CLS] Question [SEP] Context with answer here [SEP]
                            ↑         ↑
                          start      end
```

## PyTorch Usage

```python
from transformers import BertModel, BertTokenizer

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize input
text = "The cat sat on the mat."
inputs = tokenizer(text, return_tensors='pt')

# Forward pass
outputs = model(**inputs)

# Outputs
last_hidden_state = outputs.last_hidden_state  # (1, seq_len, 768)
pooler_output = outputs.pooler_output          # (1, 768) - [CLS] representation
```

## Comparison with GPT

| Aspect | BERT | GPT |
|--------|------|-----|
| Architecture | Encoder-only | Decoder-only |
| Attention | Bidirectional | Causal (left-to-right) |
| Pre-training | MLM + NSP | Autoregressive LM |
| Strengths | Understanding | Generation |
| Use cases | Classification, NER, QA | Text generation, chat |

## BERT Variants

| Model | Changes from BERT |
|-------|-------------------|
| RoBERTa | More data, no NSP, dynamic masking |
| ALBERT | Parameter sharing, sentence order prediction |
| DistilBERT | Knowledge distillation, 40% smaller |
| ELECTRA | Replaced token detection (more efficient) |

## Limitations

1. **Fixed sequence length**: Max 512 tokens
2. **No generation**: Cannot produce text
3. **[MASK] mismatch**: [MASK] token not seen during fine-tuning
4. **Compute-intensive**: Large model, slow inference

## Summary

BERT's contributions:

1. **Bidirectional pre-training**: Context from both directions
2. **Transfer learning paradigm**: Pre-train once, fine-tune many
3. **State-of-the-art results**: Improved 11 NLP benchmarks

BERT showed that deep bidirectional representations capture language understanding better than left-to-right models, though it cannot generate text like decoder models.

## References

- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2019)
- Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019)
