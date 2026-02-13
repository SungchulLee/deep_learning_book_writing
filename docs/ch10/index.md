# Chapter 9: Transfer Learning

## Overview

Transfer learning uses knowledge from models pretrained on large datasets to improve performance on new tasks, especially when training data is limited. This chapter develops the theoretical foundation, surveys strategies from feature extraction to fine-tuning, and examines domain adaptation techniques for handling distribution shift.

## Chapter Structure

**9.1 Fundamentals** covers the core concepts of transfer learning: why hierarchical features transfer across tasks, mathematical foundations including the domain adaptation bound, and decision frameworks for choosing between feature extraction, fine-tuning, and hybrid strategies.

**9.2 Domain Adaptation** addresses the challenge of distribution mismatch between source and target domains. Topics include unsupervised domain adaptation with MMD, adversarial training with DANN, and practical workflows for assessing and bridging domain gaps.

**9.3 Applications** demonstrates transfer learning in practice across computer vision and natural language processing, with complete PyTorch implementations for both modalities.

## Prerequisites

- Neural network fundamentals (Ch 2–3)
- CNN architectures (Ch 4)
- Optimization and regularization (Ch 5)

## Key Themes

- **Hierarchical feature reuse**: Early layers learn universal features; later layers specialize
- **The transfer-adaptation tradeoff**: More adaptation risks forgetting; less risks underfitting
- **Domain divergence**: Target performance is bounded by source performance plus domain distance
- **Practical decision-making**: Dataset size and domain similarity determine the optimal strategy
