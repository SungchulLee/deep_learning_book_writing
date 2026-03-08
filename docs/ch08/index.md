# Chapter Overview

<<<<<<< HEAD
This chapter covers **Balanced BSTs**.

# Reference

[Introduction to Algorithms (CLRS), Chapters 13-14](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
=======
This chapter provides comprehensive coverage of the Transformer architecture, from its core mechanisms through pretrained language models to vision applications. We develop the mathematical foundations of self-attention, explore how BERT, GPT, and T5 leverage the Transformer for different tasks, and examine the extension of Transformers to computer vision.

---

## 8.1 Transformer Architecture

The core components of the Transformer, from attention mechanisms to the complete encoder-decoder structure.

- [Transformers NLP Overview](transformer_architecture/transformers_nlp_overview.md) -- 10-step learning path from basic attention through Vision Transformers and comparative studies
- [Package Summary](transformer_architecture/package_summary.md) -- Summary of the complete 43-file educational package covering all Transformer topics
- [Attention Review](transformer_architecture/attention_review_overview.md) -- Step 1: RNN-based attention mechanisms as a bridge to self-attention
- [Self-Attention Overview](transformer_architecture/self_attention_overview.md) -- Step 2: The core self-attention mechanism behind Transformers
- [Multi-Head Attention Overview](transformer_architecture/multihead_attention_overview.md) -- Step 3: Parallel attention processing across multiple representation subspaces
- [Positional Encoding Overview](transformer_architecture/positional_encoding_overview.md) -- Step 4: Adding position information to Transformer inputs without recurrence
- [Transformer Architecture](transformer_architecture/transformer_architecture.md) -- Complete architecture overview from historical context through the encoder-decoder design
- [Positional Encoding](transformer_architecture/positional_encoding.md) -- Sinusoidal, learned, and relative positional encoding schemes
- [Encoder-Decoder Structure](transformer_architecture/encoder_decoder.md) -- Encoder-only, decoder-only, and encoder-decoder paradigms with their use cases
- [Masked Self-Attention](transformer_architecture/masked_attention.md) -- Causal masking for autoregressive generation in Transformer decoders
- [Layer Normalization](transformer_architecture/layer_norm.md) -- Pre-norm vs post-norm placement and modern alternatives like RMSNorm

## 8.2 Pretrained Models

Large-scale pretrained Transformer models that define the modern NLP landscape.

- [Transformer Encoder Overview](pretrained_models/transformer_encoder_overview.md) -- Step 5: Building the BERT-style encoder with masked language modeling
- [Transformer Decoder Overview](pretrained_models/transformer_decoder_overview.md) -- Step 6: Building the GPT-style decoder for autoregressive generation
- [BERT Text Classification Overview](pretrained_models/bert_text_classification_overview.md) -- Step 7: Fine-tuning BERT for sentiment analysis and text classification
- [GPT Text Generation Overview](pretrained_models/gpt_text_generation_overview.md) -- Step 8: Generating text with GPT-style models using various sampling strategies
- [BERT](pretrained_models/bert.md) -- Bidirectional Encoder Representations from Transformers with masked language model pretraining
- [GPT](pretrained_models/gpt.md) -- Generative Pre-trained Transformer for autoregressive language modeling from GPT-1 to GPT-4
- [T5](pretrained_models/t5.md) -- Text-to-Text Transfer Transformer that unifies all NLP tasks into a single text generation format
- [Transformer Variants](pretrained_models/variants.md) -- Comparative overview of RNN, CNN, and Transformer architectures

## 8.3 Attention Visualization

Tools and techniques for interpreting what Transformer models learn through attention analysis.

- [Attention Map Visualization](attention_visualization/attention_visualization.md) -- Visualizing encoder self-attention, decoder self-attention, and cross-attention maps
- [Visualization Tools](attention_visualization/visualization_tools.md) -- BertViz, matplotlib heatmaps, and other tools for attention visualization
- [Attention Extraction](attention_visualization/attention_extraction.md) -- Hook-based methods for extracting attention weights from PyTorch models
- [Head and Layer Analysis](attention_visualization/head_layer_analysis.md) -- Analyzing head specialization patterns, attention entropy, and head importance
- [Interpretation Pitfalls](attention_visualization/interpretation_pitfalls.md) -- Why attention is not explanation and how to combine with gradient-based methods

## 8.4 Training and Inference

Practical strategies for training Transformers efficiently and optimizing inference performance.

- [Comparison Study Overview](training_and_inference/comparison_study_overview.md) -- Step 10: Benchmarking Transformers against RNNs and CNNs on speed, accuracy, and memory
- [Training and Inference](training_and_inference/training_inference.md) -- Training with teacher forcing vs autoregressive inference and the fundamental paradigm differences
- [Training Fundamentals](training_and_inference/training_fundamentals.md) -- Loss functions, AdamW optimizer, and gradient accumulation for Transformer training
- [Training Optimization](training_and_inference/training_optimization.md) -- Learning rate schedules, warmup strategies, and regularization for deep attention networks
- [Warmup and Scheduling](training_and_inference/warmup_scheduling.md) -- Linear warmup, Noam schedule, and cosine annealing with warmup
- [Label Smoothing](training_and_inference/label_smoothing.md) -- Soft targets that prevent overconfident predictions and improve generalization
- [Large-Batch Training](training_and_inference/large_batch.md) -- Linear and square root scaling rules, LAMB optimizer, and batch size guidelines
- [Memory-Efficient Training](training_and_inference/memory_efficient.md) -- Mixed precision, gradient checkpointing, and other memory optimization techniques
- [Inference Optimization](training_and_inference/inference_optimization.md) -- KV cache, torch.compile, and strategies for fast autoregressive generation

## 8.5 Transformers for Vision

Extending Transformer architectures from NLP to computer vision tasks.

- [Transformers Vision Overview](transformers_vision/transformers_vision_overview.md) -- Educational resource bridging traditional CNNs with Transformer-based vision architectures
- [Vision Transformer Overview](transformers_vision/vision_transformer_overview.md) -- Step 9: Applying Transformers to image classification with patch embeddings and attention rollout
- [Vision Transformer (ViT)](transformers_vision/vit.md) -- The paradigm shift from convolutions to pure transformer architectures for image understanding
- [ViT Architecture](transformers_vision/vit_architecture.md) -- Standard transformer encoder applied to sequences of image patches
- [Patch Embeddings](transformers_vision/patch_embeddings.md) -- Tokenizing images into patches analogous to word tokenization in NLP
- [Position Embeddings](transformers_vision/position_embeddings.md) -- Encoding spatial information for permutation-invariant self-attention over image patches
- [CLS Token](transformers_vision/cls_token.md) -- Learnable classification token for aggregating global image representation
- [DeiT](transformers_vision/deit.md) -- Data-efficient Image Transformers with improved training strategies and knowledge distillation
- [Swin Transformer](transformers_vision/swin.md) -- Hierarchical architecture with windowed attention for linear complexity on dense prediction tasks
- [Hybrid CNN-Transformer](transformers_vision/hybrid_models.md) -- Combining CNN inductive biases with Transformer global modeling capabilities
- [CNN vs ViT Comparison](transformers_vision/cnn_vs_vit.md) -- Architectural foundations, strengths, and trade-offs between convolutional and attention-based approaches
- [Training Strategies](transformers_vision/training_strategies.md) -- ViT-specific training techniques including heavy augmentation, stochastic depth, and long schedules
- [Attention Visualization](transformers_vision/attention_visualization.md) -- Interpreting ViT predictions through CLS token attention and patch-to-patch attention maps
>>>>>>> 96f31bd (...)
