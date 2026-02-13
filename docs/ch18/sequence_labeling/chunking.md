# Chunking (Shallow Parsing)

## Overview

Chunking identifies non-overlapping phrases (chunks) in text without building a full parse tree. It groups tokens into syntactic constituents:

```
[NP The quick brown fox] [VP jumped over] [NP the lazy dog]
```

## Chunk Types

| Chunk | Description | Example |
|-------|-------------|---------|
| NP | Noun Phrase | "the quick brown fox" |
| VP | Verb Phrase | "has been running" |
| PP | Prepositional Phrase | "in the park" |
| ADVP | Adverb Phrase | "very quickly" |
| ADJP | Adjective Phrase | "extremely large" |

## BIO Tagging for Chunks

Chunking uses the same IOB2 tagging scheme as NER:

```
The    B-NP
quick  I-NP
brown  I-NP
fox    I-NP
jumped B-VP
over   B-PP
the    B-NP
lazy   I-NP
dog    I-NP
```

## Implementation

Chunking models are identical in architecture to NER models—only the label set changes. BiLSTM-CRF and Transformer models apply directly.

## CoNLL-2000 Benchmark

| Model | F1 |
|-------|-----|
| SVM | 93.5 |
| BiLSTM-CRF | 95.0 |
| BERT-base | 96.7 |

## Relationship to NER

Chunking and NER share the same sequence labeling framework. In practice, chunking can serve as a preprocessing step for NER by identifying candidate phrase boundaries.

## Summary

1. Chunking is shallow parsing using BIO sequence labeling
2. The same architectures (BiLSTM-CRF, Transformers) apply to both NER and chunking
3. Chunking provides phrase-level structure useful for downstream IE tasks
