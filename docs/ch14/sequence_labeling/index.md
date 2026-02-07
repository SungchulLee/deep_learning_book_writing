# Sequence Labeling

Sequence labeling assigns a label to each token in an input sequence. This section covers the two primary sequence labeling tasks in NLP: Named Entity Recognition (NER) and Part-of-Speech (POS) tagging.

## Core Concepts

- [NER Fundamentals](ner_fundamentals.md) — Task definition, entity types, and evaluation
- [Entity Types and Taxonomies](entity_types.md) — CoNLL, OntoNotes, and domain-specific types
- [BIO Tagging](bio_tagging.md) — IOB2 encoding scheme for sequence labeling
- [NER Evaluation](ner_evaluation.md) — Precision, recall, F1 at entity level

## Classical Methods

- [Rule-Based NER](rule_based_ner.md) — Regex patterns and contextual heuristics
- [Dictionary and Gazetteer Methods](dictionary_ner.md) — Trie-based lookup and fuzzy matching
- [Feature Engineering](feature_engineering.md) — Orthographic, lexical, and contextual features
- [CRF](crf.md) — Conditional Random Fields for structured prediction

## Neural Methods

- [BiLSTM for NER](bilstm_ner.md) — Bidirectional LSTM encoder
- [BiLSTM-CRF](bilstm_crf.md) — Combining BiLSTM with CRF decoding
- [Transformer NER](transformer_ner.md) — Transformer-based sequence labeling
- [BERT for NER](bert_ner.md) — Fine-tuning BERT for token classification
- [Subword Alignment](subword_alignment.md) — Handling subword tokenization for NER

## Advanced Topics

- [Nested NER](nested_ner.md) — Handling overlapping entities
- [Cross-Lingual NER](crosslingual_ner.md) — Transfer across languages
- [Domain Adaptation](domain_adaptation.md) — Adapting NER to new domains
- [Few-Shot NER](fewshot_ner.md) — Learning from minimal examples

## Related Tasks

- [POS Tagging](pos_tagging.md) — Part-of-speech tagging
- [Chunking](chunking.md) — Shallow parsing with BIO tags

## Datasets

- [NER Datasets](ner_datasets.md) — Benchmarks and evaluation resources
