<<<<<<< HEAD
# Chapter Overview

This chapter covers **Advanced Trees**.

# Reference

[Introduction to Algorithms (CLRS), Chapter 14](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
=======
# Chapter 10: Transfer Learning

Transfer learning leverages knowledge from models pretrained on large datasets to improve performance on new tasks, especially when training data is limited. This chapter covers the theoretical foundations, practical fine-tuning strategies, domain adaptation techniques, and real-world applications across computer vision, NLP, and specialized domains.

---

## 10.1 Transfer Learning

Core concepts, strategies, and hands-on examples for adapting pretrained models to downstream tasks.

- [Transfer Learning Overview](transfer_learning/fundamentals.md) -- Why transfer learning works, hierarchical feature transferability, and a decision framework for choosing strategies
- [Transfer Learning Tutorial](transfer_learning/transfer_learning_overview.md) -- PyTorch tutorial package with four progressively challenging transfer learning examples
- [Domain Shift](transfer_learning/domain_shift.md) -- Types of distribution mismatch between source and target domains and their formal definitions
- [Feature Extraction](transfer_learning/feature_extraction.md) -- Using pretrained networks as fixed feature extractors with only a new classifier trained on top
- [Fine-Tuning](transfer_learning/fine_tuning.md) -- The spectrum from frozen feature extraction to full parameter updates with gradual unfreezing
- [Layer Freezing](transfer_learning/layer_freezing.md) -- Selectively controlling which pretrained parameters are updated during transfer
- [Discriminative Learning Rates](transfer_learning/discriminative_lr.md) -- Assigning different learning rates to different layers based on feature transferability
- [Example 1: Basic Feature Extraction](transfer_learning/example_1_basic_overview.md) -- Loading a pretrained ResNet18 and training only the final classification layer on CIFAR-10
- [Example 2: Fine-Tuning](transfer_learning/example_2_fine_tuning_overview.md) -- Selectively unfreezing layers with different learning rates and early stopping
- [Example 3: Custom Datasets](transfer_learning/example_3_custom_dataset_overview.md) -- Applying transfer learning to custom image datasets with class imbalance handling
- [Example 4: Advanced Techniques](transfer_learning/example_4_advanced_overview.md) -- Cosine annealing, mixed precision training, gradient accumulation, and model ensembling

## 10.2 Domain Adaptation

Techniques for bridging the distribution gap between source and target domains.

- [Domain Adaptation](domain_adaptation/domain_adaptation.md) -- The domain adaptation problem, theoretical foundations, and practical techniques
- [Unsupervised Domain Adaptation](domain_adaptation/uda.md) -- Adapting models using labeled source data and unlabeled target data
- [DANN](domain_adaptation/dann.md) -- Domain Adversarial Neural Networks with gradient reversal for domain-invariant features
- [Maximum Mean Discrepancy](domain_adaptation/mmd.md) -- Kernel-based distribution matching for principled domain alignment
- [Multi-Source Domain Adaptation](domain_adaptation/multi_source.md) -- Leveraging multiple source domains with automatic source weighting and selective transfer
- [Self-Training](domain_adaptation/self_training.md) -- Iterative pseudo-labeling of unlabeled target data for progressive domain adaptation

## 10.3 Applications

Practical applications of transfer learning across computer vision, NLP, finance, and specialized domains.

- [Transfer Learning for Computer Vision](applications/transfer_cv.md) -- Standard transfer pipeline with ImageNet-pretrained models for diverse visual tasks
- [Transfer Learning for NLP](applications/transfer_nlp.md) -- Adapting pretrained language models like BERT and GPT to downstream text tasks
- [Cross-Domain Transfer](applications/cross_domain.md) -- Strategies for transferring knowledge across different domains based on domain distance
- [Transfer Learning for Finance](applications/transfer_finance.md) -- Text-based and time series transfer for financial applications including FinBERT and temporal adaptation
- [Transfer Learning for Time Series](applications/transfer_time_series.md) -- Specialized transfer strategies addressing temporal dependencies and non-stationarity
- [Negative Transfer](applications/negative_transfer.md) -- When transfer learning hurts performance and how to diagnose and prevent it
>>>>>>> 96f31bd (...)
