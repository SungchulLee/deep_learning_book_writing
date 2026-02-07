# Chapter 11: Few-Shot and Zero-Shot Learning

## Overview

Few-shot and zero-shot learning address the challenge of recognising new concepts from minimal or no training examples. These paradigms are critical for deploying deep learning in real-world settings where labeled data is scarce or new classes emerge continuously.

## Chapter Structure

**11.1 Few-Shot Learning** introduces the problem formulation, N-way K-shot episodic setup, and the training/evaluation protocol that distinguishes few-shot from standard classification.

**11.2 Metric Learning** covers distance-based approaches: Siamese Networks for pairwise verification, Matching Networks for attention-based comparison, Prototypical Networks for class-prototype classification, and Relation Networks for learned similarity.

**11.3 Meta-Learning** covers optimisation-based approaches: MAML learns initialisations for rapid adaptation, Reptile provides a simpler first-order alternative, and Meta-SGD extends MAML with learnable per-parameter learning rates.

**11.4 Benchmarks** surveys standard evaluation benchmarks including Omniglot, mini-ImageNet, and tiered-ImageNet.

**11.5 Zero-Shot Learning** addresses recognition without any training examples, leveraging semantic attributes, word embeddings, and vision-language models like CLIP for open-vocabulary recognition.

## Prerequisites

- Neural network fundamentals (Ch 2–3)
- Transfer learning (Ch 9)
- Self-supervised learning concepts (Ch 10)
