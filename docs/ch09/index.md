# Chapter 10: Self-Supervised Learning

## Overview

Self-supervised learning (SSL) has emerged as the dominant paradigm for learning visual representations without human annotations. By creating supervision signals from the data itself, SSL methods can leverage vast amounts of unlabeled data to learn features that transfer effectively to downstream tasks.

## Chapter Structure

**10.1 Foundations** introduces the SSL framework, motivations, and classical pretext tasks including rotation prediction, jigsaw puzzles, colorization, and inpainting.

**10.2 Contrastive Learning** covers methods that learn by contrasting positive and negative pairs: the InfoNCE framework, SimCLR, MoCo, BYOL, SimSiam, and Barlow Twins—tracing the evolution from large-batch contrastive methods to approaches that eliminate negative samples entirely.

**10.3 Masked Modeling** explores reconstruction-based methods: MAE's masked autoencoder approach, BEiT's discrete token prediction, and Data2Vec's cross-modal framework.

**10.4 Self-Distillation** covers DINO and DINOv2, which learn through a student-teacher framework without labels, producing features with remarkable emergent properties.

**10.5 Evaluation** presents standard evaluation protocols including linear probing, k-NN evaluation, and representation quality metrics.

## Prerequisites

- Convolutional networks and Vision Transformers (Ch 4–5)
- Loss functions and optimisation (Ch 3)
- Transfer learning concepts (Ch 9)
