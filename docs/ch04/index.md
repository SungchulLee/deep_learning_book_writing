# Chapter 4 — Neural Network Building Blocks

## Overview

A neural network is assembled from a small number of recurring components: activation functions that inject nonlinearity, feedforward layers that learn representations, initialization schemes that set the starting point for optimisation, normalization layers that stabilise training dynamics, and regularization techniques that prevent overfitting. Each component addresses a specific failure mode — vanishing gradients, internal covariate shift, memorisation of noise — and their combined effect determines whether a network trains efficiently or collapses.

This chapter provides a modular, bottom-up treatment. Each section can be read independently, but the components interact and the recommended reading order reflects their dependency structure.

## Chapter Structure

### 4.1 Activation Functions

Activation functions transform linear pre-activations into nonlinear outputs, enabling the network to approximate arbitrary functions. We begin with the classical sigmoid and tanh, trace the shift to ReLU and its variants (Leaky ReLU, PReLU, ELU, SELU), and cover the smooth activations now dominant in modern architectures (GELU, Swish/SiLU, Mish). The section closes with Softmax for classification outputs and a practical selection guide.

**Key takeaway.** The choice of activation function affects gradient flow, training speed, and final accuracy. Modern defaults (ReLU for CNNs, GELU or SiLU for Transformers) have been validated across thousands of experiments, but understanding *why* they work enables informed departures when architectures demand it.

### 4.2 Feedforward Networks

The multi-layer perceptron (MLP) is the simplest and most fundamental deep learning architecture. We formalise the MLP, state the Universal Approximation Theorem and its practical limitations, examine the depth-versus-width tradeoff, derive the forward pass and backpropagation in full generality, and analyse gradient flow through deep networks. A complete CIFAR-10 implementation ties theory to practice.

**Key takeaway.** Depth creates exponentially more efficient representations than width alone, but deeper networks amplify gradient pathologies. Every technique in the subsequent sections — initialization, normalization, regularization — exists to tame these pathologies.

### 4.3 Weight Initialization

Before the first gradient update, the network's weights determine whether activations and gradients remain in a usable numerical range. Poor initialization causes signal to explode or vanish before training begins. We derive the variance conditions that maintain stable forward and backward signal propagation, yielding **Xavier** (Glorot) initialization for symmetric activations and **He** (Kaiming) initialization for ReLU-family networks.

**Key takeaway.** Proper initialization is a necessary condition for training deep networks without normalization layers. Even when normalization is present, good initialization accelerates early-phase convergence.

### 4.4 Normalization Layers

Normalization layers re-centre and re-scale activations at each layer, smoothing the loss landscape and decoupling layers' sensitivity to upstream parameter changes. We cover **Batch Normalization** (including its distinct training and inference modes), **Layer Normalization**, **Instance Normalization**, **Group Normalization**, and **RMSNorm**, with a comparative guide for choosing among them.

**Key takeaway.** Normalization is one of the most impactful additions to deep learning. The choice of method depends on architecture (CNN vs. Transformer), batch size, and whether inference must be independent of batch statistics.

### 4.5 Regularization

Regularization prevents the model from fitting noise in the training data. We cover explicit penalty methods (**L1**, **L2**, **Elastic Net**), stochastic methods (**Dropout**, **DropConnect**), data-space techniques (**Data Augmentation**, **Mixup**, **CutMix**, **Cutout**), output-space smoothing (**Label Smoothing**), training-process controls (**Early Stopping**), and **Noise Injection**. Each technique targets a different source of overfitting; in practice, several are combined.

**Key takeaway.** No single regularizer is universally best. Effective regularization strategies combine complementary techniques — for example, weight decay + dropout + data augmentation — calibrated to the dataset size, model capacity, and task.

## Dependencies and Prerequisites

This chapter assumes familiarity with:

- **Linear algebra**: matrix multiplication, vector norms, eigenvalues (Chapter 1)
- **Calculus**: chain rule, partial derivatives, gradients (Chapter 1)
- **Probability**: expectation, variance, distributions (Chapter 1)
- **PyTorch fundamentals**: tensors, `nn.Module`, autograd (Chapter 3)

## Quantitative Finance Connections

The building blocks in this chapter appear throughout quantitative finance applications:

- **Activation functions** in pricing networks must preserve monotonicity and positivity constraints (e.g., option prices must be non-negative and monotone in spot price).
- **MLP architectures** serve as universal function approximators for implied volatility surfaces, yield curve interpolation, and PnL prediction.
- **Weight initialization** is critical when networks must converge quickly in online learning settings such as real-time market-making.
- **Normalization** stabilises training on financial data where feature scales span many orders of magnitude (e.g., prices vs. volumes vs. returns).
- **Regularization** is essential in low-data regimes common in finance, where models must generalise from limited historical samples without overfitting to regime-specific noise.
