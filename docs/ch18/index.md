# Chapter 18: Natural Language Processing

This chapter covers the core tasks and deep learning architectures in natural language processing, from language modeling and sequence labeling to text classification, information extraction, machine translation, question answering, and summarization. Each section presents the mathematical foundations, traces the evolution from classical to neural approaches, and provides practical implementations.

---

## Language Modeling

Predicting word sequences from n-gram models to neural architectures and modern generation strategies.

- Language Modeling Module -- Module overview covering statistical and neural language models
- Quick Start Guide -- Installation and setup for language modeling examples
- [N-gram Language Models](language_modeling/ngram.md) -- Unigram, bigram, trigram models with MLE and smoothing
- [Neural Language Models](language_modeling/neural_lm.md) -- Feedforward, RNN, and LSTM language models in PyTorch
- Tokenization -- BPE, WordPiece, and SentencePiece algorithms
- Causal vs Masked Language Modeling -- Comparing autoregressive and masked pretraining objectives
- Perplexity -- Standard intrinsic evaluation metric from information theory
- [Sampling Strategies](language_modeling/sampling_strategies.md) -- Greedy, beam search, top-k, top-p, and temperature-based decoding
- Controlled Generation -- Steering outputs via prompt engineering, constrained decoding, and RLHF

---

## Sequence Labeling

Token-level prediction tasks including named entity recognition, POS tagging, and chunking.

- NER Module Overview -- Comprehensive introduction to NER from rule-based to transformer models
- Quick Start Guide -- Installation and quick start for NER examples
- [NER Fundamentals](sequence_labeling/ner_fundamentals.md) -- Definition, role in NLP pipelines, and sequence labeling formulation
- [Entity Types and Taxonomies](sequence_labeling/entity_types.md) -- CoNLL, OntoNotes, and domain-specific entity type systems
- [BIO Tagging Schemes](sequence_labeling/bio_tagging.md) -- IOB, IOB2, and BIOES tagging formats with conversion
- POS Tagging -- Part-of-speech tagging with standard tag sets
- Chunking -- Shallow parsing to identify noun, verb, and other phrase chunks
- Feature Engineering -- Hand-crafted orthographic, lexical, and contextual features for NER
- Rule-Based NER -- Pattern-based entity recognition using regular expressions
- Dictionary and Gazetteer Methods -- Dictionary-based NER with fuzzy matching and gazetteer features
- [Conditional Random Fields](sequence_labeling/crf.md) -- Linear-chain CRF derivation and PyTorch implementation
- [BiLSTM for NER](sequence_labeling/bilstm_ner.md) -- Bidirectional LSTM with character embeddings for sequence labeling
- BiLSTM-CRF -- End-to-end BiLSTM-CRF pipeline for sequence labeling
- Transformer NER -- Fine-tuning BERT and RoBERTa for named entity recognition
- BERT for Token Classification -- BERT fine-tuning with HuggingFace for NER
- Subword Token Alignment -- Aligning subword tokens to word-level labels
- NER Datasets -- Standard benchmarks including CoNLL-2003 and OntoNotes
- [NER Evaluation Metrics](sequence_labeling/ner_evaluation.md) -- Entity-level evaluation with exact and partial match F1
- Nested NER -- Handling overlapping and nested entity annotations
- Cross-Lingual NER -- Transferring NER from high-resource to low-resource languages
- Few-Shot NER -- Recognizing entities with minimal labeled examples
- Domain Adaptation -- Bridging domain shift for NER across different text sources

---

## Text Classification

Document-level classification from bag-of-words baselines to transformer fine-tuning.

- Text Classification Fundamentals -- Task definition, pipeline, and probabilistic formulation
- Bag-of-Words and TF-IDF -- Count-based document representations for classification
- CNN for Text -- TextCNN with 1D convolutions over word embeddings
- RNN for Text -- LSTM/RNN classifiers with sequential processing
- Transformer Classification -- BERT fine-tuning with CLS token for text classification
- Sentiment Analysis -- Binary, fine-grained, and aspect-level sentiment tasks
- Hierarchical Classification -- Taxonomy-based classification with parent-child consistency
- Multi-Label Classification -- Predicting multiple categories per document with BCE loss

---

## Information Extraction

Extracting structured knowledge from unstructured text through relations, events, and knowledge graphs.

- [IE Overview](information_extraction/ie_overview.md) -- The information extraction pipeline and financial applications
- Relation Extraction -- Pipeline and joint approaches to extracting entity relations
- [Event Extraction](information_extraction/event_extraction.md) -- Trigger detection and argument extraction from text
- [Coreference Resolution](information_extraction/coreference.md) -- Mention detection and antecedent linking for document-level IE
- [Open Information Extraction](information_extraction/open_ie.md) -- Schema-free triple extraction for exploratory text mining
- Knowledge Graph Construction -- End-to-end KG pipeline with entity linking and embeddings

---

## Machine Translation

Translating between languages from statistical methods to neural and transformer-based approaches.

- MT Overview -- Evolution from rule-based to neural machine translation
- Statistical MT -- Noisy channel model, IBM alignment, and phrase-based translation
- Neural MT -- Encoder-decoder architecture and the information bottleneck
- Attention in MT -- Additive, multiplicative, and dot-product attention for translation
- [Transformer MT](machine_translation/transformer_mt.md) -- Multi-head self-attention replacing RNN-based translation
- Subword Segmentation -- BPE and SentencePiece for open vocabulary translation
- Multilingual MT -- Single-model many-to-many and zero-shot translation
- Low-Resource MT -- Data augmentation and transfer learning for low-resource languages
- MT Evaluation -- BLEU score computation, neural metrics, and evaluation protocols

---

## Question Answering

Answering questions through extractive, abstractive, and knowledge-based approaches.

- QA Overview -- Taxonomy of QA tasks and financial QA applications
- Reading Comprehension -- Benchmarks and challenges in machine reading
- [Extractive QA](question_answering/extractive_qa.md) -- BERT-based span extraction for question answering
- Span Extraction -- Mathematical formulation of start/end position prediction
- Abstractive QA -- Generating answers that synthesize and paraphrase source text
- Open-Domain QA -- Retriever-reader architecture for corpus-scale QA
- Knowledge-Based QA -- Answering questions by querying structured knowledge bases
- Multi-Hop QA -- Reasoning over multiple evidence pieces to answer questions
- QA Datasets -- SQuAD, Natural Questions, TriviaQA, and other benchmarks

---

## Summarization

Condensing documents through extractive selection and abstractive generation.

- Summarization Overview -- Definition, taxonomy, and comparison of summarization approaches
- Extractive Summarization -- Selecting important sentences from source documents
- Abstractive Summarization -- Generating novel summary text with paraphrasing and fusion
- Seq2Seq Summarization -- Encoder-decoder models with attention for summarization
- Transformer Summarization -- BART, T5, and Pegasus for state-of-the-art summarization
- Multi-Document Summarization -- Summarizing across multiple related documents
- Summarization Evaluation -- ROUGE metrics and evaluation protocols
