# Chapter 10: Transfer Learning

Transfer learning leverages knowledge from models pretrained on large datasets to improve performance on new tasks, especially when training data is limited. This chapter covers the theoretical foundations, practical fine-tuning strategies, domain adaptation techniques, and real-world applications across computer vision, NLP, and specialized domains.

---

## 10.1 Transfer Learning

Core concepts, strategies, and hands-on examples for adapting pretrained models to downstream tasks.

- Transfer Learning Overview -- Why transfer learning works, hierarchical feature transferability, and a decision framework for choosing strategies
- Transfer Learning Tutorial -- PyTorch tutorial package with four progressively challenging transfer learning examples
- Domain Shift -- Types of distribution mismatch between source and target domains and their formal definitions
- Feature Extraction -- Using pretrained networks as fixed feature extractors with only a new classifier trained on top
- [Fine-Tuning](transfer_learning/fine_tuning.md) -- The spectrum from frozen feature extraction to full parameter updates with gradual unfreezing
- Layer Freezing -- Selectively controlling which pretrained parameters are updated during transfer
- Discriminative Learning Rates -- Assigning different learning rates to different layers based on feature transferability
- Example 1: Basic Feature Extraction -- Loading a pretrained ResNet18 and training only the final classification layer on CIFAR-10
- Example 2: Fine-Tuning -- Selectively unfreezing layers with different learning rates and early stopping
- Example 3: Custom Datasets -- Applying transfer learning to custom image datasets with class imbalance handling
- Example 4: Advanced Techniques -- Cosine annealing, mixed precision training, gradient accumulation, and model ensembling

## 10.2 Domain Adaptation

Techniques for bridging the distribution gap between source and target domains.

- Domain Adaptation -- The domain adaptation problem, theoretical foundations, and practical techniques
- Unsupervised Domain Adaptation -- Adapting models using labeled source data and unlabeled target data
- DANN -- Domain Adversarial Neural Networks with gradient reversal for domain-invariant features
- [Maximum Mean Discrepancy](domain_adaptation/mmd.md) -- Kernel-based distribution matching for principled domain alignment
- Multi-Source Domain Adaptation -- Leveraging multiple source domains with automatic source weighting and selective transfer
- Self-Training -- Iterative pseudo-labeling of unlabeled target data for progressive domain adaptation

## 10.3 Applications

Practical applications of transfer learning across computer vision, NLP, finance, and specialized domains.

- Transfer Learning for Computer Vision -- Standard transfer pipeline with ImageNet-pretrained models for diverse visual tasks
- Transfer Learning for NLP -- Adapting pretrained language models like BERT and GPT to downstream text tasks
- Cross-Domain Transfer -- Strategies for transferring knowledge across different domains based on domain distance
- Transfer Learning for Finance -- Text-based and time series transfer for financial applications including FinBERT and temporal adaptation
- [Transfer Learning for Time Series](applications/transfer_time_series.md) -- Specialized transfer strategies addressing temporal dependencies and non-stationarity
- Negative Transfer -- When transfer learning hurts performance and how to diagnose and prevent it
