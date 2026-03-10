# Chapter 9: Self-Supervised Learning


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

Self-supervised learning (SSL) has emerged as the dominant paradigm for learning representations without human annotations. By creating supervision signals from the data itself, SSL methods leverage vast amounts of unlabeled data to learn features that transfer effectively to downstream tasks. This chapter traces the evolution from classical pretext tasks through contrastive learning and masked modeling to modern self-distillation methods.

---

## 9.1 Foundations

Core concepts, motivations, and the theoretical framework underlying self-supervised learning.

- SSL Overview -- The self-supervised learning framework, motivation, and comparison with supervised and semi-supervised paradigms
- Pretext Tasks -- Auxiliary learning objectives that create supervision signals directly from input data structure
- Pretext Task Taxonomy -- Classification of SSL methods into predictive, contrastive, and generative categories
- SSL and Representation Learning -- Desirable representation properties, the collapse problem, and evaluation approaches

## 9.2 Contrastive Learning

Methods that learn representations by pulling positive pairs together and pushing negative pairs apart in embedding space.

- Contrastive Learning Tutorial -- Implementation collection covering SimCLR, MoCo, and MAE with augmentation strategies
- Contrastive Learning -- The InfoNCE framework, positive and negative pairs, and information-theoretic foundations
- SimCLR -- Simple contrastive framework with data augmentation composition and projection head
- MoCo -- Momentum Contrast with dynamic dictionary and queue mechanism for efficient contrastive learning
- BYOL -- Bootstrap Your Own Latent achieving strong representations without negative samples
- SimSiam -- The simplest non-contrastive method using only stop-gradient for collapse prevention
- Barlow Twins -- Learning by making the cross-correlation matrix of embeddings close to identity

## 9.3 Masked Modeling

Reconstruction-based methods that learn representations by predicting masked or corrupted portions of the input.

- Masked Image Modeling Overview -- Framework for self-supervised learning through predicting masked image regions
- Masked Autoencoders (MAE) -- Learning from heavily masked images with an asymmetric encoder-decoder architecture
- BEiT -- BERT-style pre-training for images by predicting discrete visual tokens from a pretrained tokenizer
- Data2Vec -- Unified cross-modal framework predicting latent representations from a momentum teacher
- SimMIM -- Simple masked image modeling with direct pixel prediction and minimal design complexity
- iBOT -- Image BERT pre-training with an online tokenizer learned jointly with the masking objective

## 9.4 Self-Distillation

Teacher-student frameworks where models learn from themselves through knowledge distillation without labels.

- Self-Distillation Overview -- The self-distillation paradigm and its information-theoretic foundations
- [Knowledge Distillation Basics](self_distillation/distillation_basics.md) -- Teacher-student framework fundamentals including dark knowledge and temperature scaling
- [EMA Teacher](self_distillation/ema_teacher.md) -- Exponential moving average teacher mechanism for stable target models in self-distillation
- DINO -- Self-distillation with no labels producing semantically meaningful attention maps from ViTs
- DINOv2 -- Scaling self-distillation to produce universal visual features combining DINO with iBOT objectives

## 9.5 Evaluation

Standard protocols for measuring the quality of self-supervised representations.

- SSL Evaluation Protocols -- Overview of linear probing, k-NN evaluation, and standard evaluation practices
- Linear Probing -- Training a linear classifier on frozen features to directly measure representation quality
- Fine-Tuning Evaluation -- Measuring initialization quality by updating all parameters on downstream tasks
- Transfer Benchmarks -- Standard vision and NLP benchmarks for evaluating cross-domain generalization
