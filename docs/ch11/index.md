<<<<<<< HEAD
# Chapter Overview

This chapter covers **Sorting Fundamentals**.

# Reference

[Introduction to Algorithms (CLRS), Chapter 8](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
=======
# Chapter 11: Few-Shot & Zero-Shot Learning

This chapter explores learning paradigms that operate with minimal or no labeled examples per class. Few-shot learning develops algorithms that generalize from just a handful of examples, while zero-shot learning leverages semantic knowledge to recognize classes never seen during training. Together, these approaches address one of deep learning's most significant limitations: the dependence on massive labeled datasets.

---

## Few-Shot Learning

- [Few-Shot Learning Examples](few_shot/few_shot_learning_overview.md) -- Collection of Python implementations covering prototypical networks, matching networks, MAML, and more
- [Few-Shot Learning Fundamentals](few_shot/few_shot_overview.md) -- Problem formulation, formal definitions, and the distinction between training on seen classes and testing on novel classes
- [N-way K-shot Setup](few_shot/nway_kshot.md) -- The standard evaluation protocol structuring tasks around support and query sets with common configurations
- [Episode-Based Training](few_shot/episode_training.md) -- Simulating few-shot conditions during training through episodic sampling of support and query sets
- [Data Augmentation for Few-Shot](few_shot/augmentation.md) -- Feature-space and image-space augmentation strategies tailored to low-data regimes

## Metric Learning

- [Siamese Networks](metric_learning/siamese.md) -- Twin-network architecture for one-shot learning through pairwise similarity comparison
- [Prototypical Networks](metric_learning/prototypical.md) -- Classification via nearest class prototypes (centroids) in a learned embedding space
- [Matching Networks](metric_learning/matching_networks.md) -- Attention-based weighted nearest-neighbour classification with episodic training
- [Relation Networks](metric_learning/relation_networks.md) -- Replacing fixed distance metrics with a learned neural similarity function

## Meta-Learning

- [Overview of Meta-Learning](meta_learning/overview.md) -- Introduction to "learning to learn" paradigms including optimization-based, metric-based, and model-based approaches
- [MAML](meta_learning/maml.md) -- Model-Agnostic Meta-Learning that finds initializations enabling rapid adaptation via a few gradient steps
- [Reptile](meta_learning/reptile.md) -- A simpler first-order meta-learning alternative to MAML using only standard gradient descent
- [Meta-SGD](meta_learning/meta_sgd.md) -- Extension of MAML that additionally learns per-parameter learning rates and update directions
- [Learned Optimizers](meta_learning/learned_optimizers.md) -- Meta-learning the optimization process itself, replacing fixed update rules with learned functions
- [Task Distribution Design](meta_learning/task_distribution.md) -- Principles and strategies for constructing task distributions that ensure meta-learning generalization

## Few-Shot Benchmarks

- [Benchmark Datasets](few_shot_benchmarks/benchmarks.md) -- Survey of standard datasets including Omniglot, mini-ImageNet, tiered-ImageNet, and CUB-200
- [Meta-Dataset](few_shot_benchmarks/meta_dataset.md) -- Large-scale benchmark spanning 10 diverse image domains with variable-way variable-shot episodes
- [Evaluation Protocols](few_shot_benchmarks/evaluation_protocols.md) -- Standard protocols, backbone impact analysis, and common pitfalls in few-shot evaluation

## Zero-Shot Learning

- [Zero-Shot Learning Overview](zero_shot/zero_shot_learning_overview.md) -- Comprehensive overview of ZSL theory, semantic relationships, and transfer from seen to unseen classes
- [Zero-Shot Fundamentals](zero_shot/zero_shot_overview.md) -- Formal problem definition, training/testing protocols, and mathematical foundations for all ZSL methods
- [Attribute-Based Methods](zero_shot/attribute_based.md) -- Direct and indirect attribute prediction approaches using semantic attribute vectors per class
- [Semantic Embedding Methods](zero_shot/embedding_based.md) -- Replacing manual attributes with continuous word embeddings learned from text corpora
- [Zero-Shot Classification](zero_shot/classification.md) -- Visual-semantic embedding models including DeViSE and bilinear compatibility architectures
- [Generalized Zero-Shot Learning](zero_shot/generalized.md) -- Extended setting where test instances may come from either seen or unseen classes
- [CLIP for Zero-Shot Learning](zero_shot/clip_zero_shot.md) -- Leveraging contrastive language-image pretraining for open-vocabulary zero-shot classification
- [Zero-Shot Segmentation](zero_shot/segmentation.md) -- Extending zero-shot learning to dense pixel-level prediction using vision-language models
>>>>>>> 96f31bd (...)
