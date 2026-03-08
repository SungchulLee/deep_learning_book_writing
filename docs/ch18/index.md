<<<<<<< HEAD
# Chapter Overview

This chapter covers **Divide and Conquer**.

# Reference

[Introduction to Algorithms (CLRS), Chapter 4](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
=======
# Chapter 18: Natural Language Processing

This chapter covers the core tasks and deep learning architectures in natural language processing, from language modeling and sequence labeling to text classification, information extraction, machine translation, question answering, and summarization. Each section presents the mathematical foundations, traces the evolution from classical to neural approaches, and provides practical implementations.

---

## Language Modeling

Predicting word sequences from n-gram models to neural architectures and modern generation strategies.

- [Language Modeling Module](language_modeling/language_modeling_overview.md) -- Module overview covering statistical and neural language models
- [Quick Start Guide](language_modeling/quickstart.md) -- Installation and setup for language modeling examples
- [N-gram Language Models](language_modeling/ngram.md) -- Unigram, bigram, trigram models with MLE and smoothing
- [Neural Language Models](language_modeling/neural_lm.md) -- Feedforward, RNN, and LSTM language models in PyTorch
- [Tokenization](language_modeling/tokenization.md) -- BPE, WordPiece, and SentencePiece algorithms
- [Causal vs Masked Language Modeling](language_modeling/causal_vs_masked.md) -- Comparing autoregressive and masked pretraining objectives
- [Perplexity](language_modeling/perplexity.md) -- Standard intrinsic evaluation metric from information theory
- [Sampling Strategies](language_modeling/sampling_strategies.md) -- Greedy, beam search, top-k, top-p, and temperature-based decoding
- [Controlled Generation](language_modeling/controlled_generation.md) -- Steering outputs via prompt engineering, constrained decoding, and RLHF

---

## Sequence Labeling

Token-level prediction tasks including named entity recognition, POS tagging, and chunking.

- [NER Module Overview](sequence_labeling/named_entity_recognition_overview.md) -- Comprehensive introduction to NER from rule-based to transformer models
- [Quick Start Guide](sequence_labeling/quickstart.md) -- Installation and quick start for NER examples
- [NER Fundamentals](sequence_labeling/ner_fundamentals.md) -- Definition, role in NLP pipelines, and sequence labeling formulation
- [Entity Types and Taxonomies](sequence_labeling/entity_types.md) -- CoNLL, OntoNotes, and domain-specific entity type systems
- [BIO Tagging Schemes](sequence_labeling/bio_tagging.md) -- IOB, IOB2, and BIOES tagging formats with conversion
- [POS Tagging](sequence_labeling/pos_tagging.md) -- Part-of-speech tagging with standard tag sets
- [Chunking](sequence_labeling/chunking.md) -- Shallow parsing to identify noun, verb, and other phrase chunks
- [Feature Engineering](sequence_labeling/feature_engineering.md) -- Hand-crafted orthographic, lexical, and contextual features for NER
- [Rule-Based NER](sequence_labeling/rule_based_ner.md) -- Pattern-based entity recognition using regular expressions
- [Dictionary and Gazetteer Methods](sequence_labeling/dictionary_ner.md) -- Dictionary-based NER with fuzzy matching and gazetteer features
- [Conditional Random Fields](sequence_labeling/crf.md) -- Linear-chain CRF derivation and PyTorch implementation
- [BiLSTM for NER](sequence_labeling/bilstm_ner.md) -- Bidirectional LSTM with character embeddings for sequence labeling
- [BiLSTM-CRF](sequence_labeling/bilstm_crf.md) -- End-to-end BiLSTM-CRF pipeline for sequence labeling
- [Transformer NER](sequence_labeling/transformer_ner.md) -- Fine-tuning BERT and RoBERTa for named entity recognition
- [BERT for Token Classification](sequence_labeling/bert_ner.md) -- BERT fine-tuning with HuggingFace for NER
- [Subword Token Alignment](sequence_labeling/subword_alignment.md) -- Aligning subword tokens to word-level labels
- [NER Datasets](sequence_labeling/ner_datasets.md) -- Standard benchmarks including CoNLL-2003 and OntoNotes
- [NER Evaluation Metrics](sequence_labeling/ner_evaluation.md) -- Entity-level evaluation with exact and partial match F1
- [Nested NER](sequence_labeling/nested_ner.md) -- Handling overlapping and nested entity annotations
- [Cross-Lingual NER](sequence_labeling/crosslingual_ner.md) -- Transferring NER from high-resource to low-resource languages
- [Few-Shot NER](sequence_labeling/fewshot_ner.md) -- Recognizing entities with minimal labeled examples
- [Domain Adaptation](sequence_labeling/domain_adaptation.md) -- Bridging domain shift for NER across different text sources

---

## Text Classification

Document-level classification from bag-of-words baselines to transformer fine-tuning.

- [Text Classification Fundamentals](text_classification/fundamentals.md) -- Task definition, pipeline, and probabilistic formulation
- [Bag-of-Words and TF-IDF](text_classification/bow_tfidf.md) -- Count-based document representations for classification
- [CNN for Text](text_classification/cnn_text.md) -- TextCNN with 1D convolutions over word embeddings
- [RNN for Text](text_classification/rnn_text.md) -- LSTM/RNN classifiers with sequential processing
- [Transformer Classification](text_classification/transformer_classification.md) -- BERT fine-tuning with CLS token for text classification
- [Sentiment Analysis](text_classification/sentiment.md) -- Binary, fine-grained, and aspect-level sentiment tasks
- [Hierarchical Classification](text_classification/hierarchical.md) -- Taxonomy-based classification with parent-child consistency
- [Multi-Label Classification](text_classification/multi_label.md) -- Predicting multiple categories per document with BCE loss

---

## Information Extraction

Extracting structured knowledge from unstructured text through relations, events, and knowledge graphs.

- [IE Overview](information_extraction/ie_overview.md) -- The information extraction pipeline and financial applications
- [Relation Extraction](information_extraction/relation_extraction.md) -- Pipeline and joint approaches to extracting entity relations
- [Event Extraction](information_extraction/event_extraction.md) -- Trigger detection and argument extraction from text
- [Coreference Resolution](information_extraction/coreference.md) -- Mention detection and antecedent linking for document-level IE
- [Open Information Extraction](information_extraction/open_ie.md) -- Schema-free triple extraction for exploratory text mining
- [Knowledge Graph Construction](information_extraction/kg_construction.md) -- End-to-end KG pipeline with entity linking and embeddings

---

## Machine Translation

Translating between languages from statistical methods to neural and transformer-based approaches.

- [MT Overview](machine_translation/mt_overview.md) -- Evolution from rule-based to neural machine translation
- [Statistical MT](machine_translation/statistical_mt.md) -- Noisy channel model, IBM alignment, and phrase-based translation
- [Neural MT](machine_translation/neural_mt.md) -- Encoder-decoder architecture and the information bottleneck
- [Attention in MT](machine_translation/attention_mt.md) -- Additive, multiplicative, and dot-product attention for translation
- [Transformer MT](machine_translation/transformer_mt.md) -- Multi-head self-attention replacing RNN-based translation
- [Subword Segmentation](machine_translation/subword_mt.md) -- BPE and SentencePiece for open vocabulary translation
- [Multilingual MT](machine_translation/multilingual.md) -- Single-model many-to-many and zero-shot translation
- [Low-Resource MT](machine_translation/low_resource.md) -- Data augmentation and transfer learning for low-resource languages
- [MT Evaluation](machine_translation/evaluation.md) -- BLEU score computation, neural metrics, and evaluation protocols

---

## Question Answering

Answering questions through extractive, abstractive, and knowledge-based approaches.

- [QA Overview](question_answering/qa_overview.md) -- Taxonomy of QA tasks and financial QA applications
- [Reading Comprehension](question_answering/reading_comprehension.md) -- Benchmarks and challenges in machine reading
- [Extractive QA](question_answering/extractive_qa.md) -- BERT-based span extraction for question answering
- [Span Extraction](question_answering/span_extraction.md) -- Mathematical formulation of start/end position prediction
- [Abstractive QA](question_answering/abstractive_qa.md) -- Generating answers that synthesize and paraphrase source text
- [Open-Domain QA](question_answering/open_domain.md) -- Retriever-reader architecture for corpus-scale QA
- [Knowledge-Based QA](question_answering/knowledge_qa.md) -- Answering questions by querying structured knowledge bases
- [Multi-Hop QA](question_answering/multi_hop.md) -- Reasoning over multiple evidence pieces to answer questions
- [QA Datasets](question_answering/datasets.md) -- SQuAD, Natural Questions, TriviaQA, and other benchmarks

---

## Summarization

Condensing documents through extractive selection and abstractive generation.

- [Summarization Overview](summarization/summarization_overview.md) -- Definition, taxonomy, and comparison of summarization approaches
- [Extractive Summarization](summarization/extractive.md) -- Selecting important sentences from source documents
- [Abstractive Summarization](summarization/abstractive.md) -- Generating novel summary text with paraphrasing and fusion
- [Seq2Seq Summarization](summarization/seq2seq_summarization.md) -- Encoder-decoder models with attention for summarization
- [Transformer Summarization](summarization/transformer_summarization.md) -- BART, T5, and Pegasus for state-of-the-art summarization
- [Multi-Document Summarization](summarization/multi_document.md) -- Summarizing across multiple related documents
- [Summarization Evaluation](summarization/evaluation.md) -- ROUGE metrics and evaluation protocols
>>>>>>> 96f31bd (...)
