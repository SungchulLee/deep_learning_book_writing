<<<<<<< HEAD
# Chapter Overview

This chapter covers **Arrays and Linked Lists**.

# Reference

[Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
=======
# Chapter 4: NN Building Blocks

This chapter covers the fundamental components used to construct neural networks: activation functions, feedforward network architecture, weight initialization, normalization layers, and regularization techniques. Each topic is developed from mathematical first principles with accompanying PyTorch implementations. Together, these building blocks form the vocabulary for assembling and training any deep learning model.

## Activations

Activation functions introduce nonlinearity into neural networks, enabling them to learn complex patterns.

- [Activation Functions Overview](activations/activation_functions_overview.md) -- Tutorial package overview with progressive examples
- [Activation Overview](activations/activation_overview.md) -- Why nonlinearity is necessary, the collapse proof, and the activation function taxonomy
- [ReLU](activations/relu.md) -- Rectified Linear Unit: the default hidden-layer activation, solving the vanishing gradient problem
- [Sigmoid](activations/sigmoid.md) -- The logistic function for binary output probabilities and gating mechanisms
- [Tanh](activations/tanh.md) -- Hyperbolic tangent for zero-centered bounded activations in recurrent networks
- [Leaky ReLU](activations/leaky_relu.md) -- Non-zero gradient for negative inputs, solving the dead neuron problem
- [PReLU](activations/prelu.md) -- Parametric ReLU with a learnable negative slope optimized during training
- [ELU](activations/elu.md) -- Exponential Linear Unit with smooth saturation for negative inputs
- [SELU](activations/selu.md) -- Scaled ELU enabling self-normalizing networks without explicit normalization layers
- [GELU](activations/gelu.md) -- Gaussian Error Linear Unit, the standard activation for transformer architectures
- [Swish / SiLU](activations/swish.md) -- Self-gating activation used in EfficientNet and modern LLMs via SwiGLU
- [Mish](activations/mish.md) -- Smooth non-monotonic activation adopted in YOLOv4 for object detection
- [Softmax](activations/softmax.md) -- Converting logits to probability distributions for multiclass classification
- [Selection Guide](activations/selection_guide.md) -- Systematic framework for choosing activations by architecture and task

## Feedforward Networks

Multi-layer perceptrons: architecture, forward and backward propagation, and gradient flow.

- [Feedforward Networks Overview](feedforward/feedforward_networks_overview.md) -- Complete tutorial combining mathematical foundations and PyTorch mastery
- [Getting Started](feedforward/getting_started.md) -- Quick-start installation and first steps
- [Quick Reference](feedforward/quick_reference.md) -- Cheat sheet for quick lookups by topic
- [Level 0: Foundations](feedforward/level_0_foundations_overview.md) -- Building neural networks from scratch with NumPy
- [Level 1: PyTorch Basics](feedforward/level_1_pytorch_basics_overview.md) -- Autograd, `nn.Module`, and optimizer usage
- [Level 2: Building Networks](feedforward/level_2_building_networks_overview.md) -- MNIST, multiple architectures, and activation comparisons
- [Level 3: Advanced Techniques](feedforward/level_3_advanced_techniques_overview.md) -- Regularization, normalization, scheduling, and initialization
- [Level 4: Applications](feedforward/level_4_applications_overview.md) -- CIFAR-10, regression, multi-task learning, and deep architectures
- [MLP Architecture](feedforward/mlp_architecture.md) -- Single neuron to full network, parameter counting, and PyTorch implementation
- [Universal Approximation](feedforward/universal_approximation.md) -- The theorem, geometric intuition, and width complexity analysis
- [Depth vs Width](feedforward/depth_vs_width.md) -- Exponential separation results, hierarchical composition, and practical tradeoffs
- [Forward Pass](feedforward/forward_pass.md) -- Propagation equations, numerical trace, computational graph, and complexity analysis
- [Backpropagation](feedforward/backpropagation.md) -- Chain rule derivation, hand computation, general recurrence, and autograd verification
- [Gradient Flow](feedforward/gradient_flow.md) -- Vanishing and exploding gradients, Jacobian analysis, and mitigation strategies

## Weight Initialization

Setting initial weight distributions to maintain stable signal propagation through deep networks.

- [Weight Initialization](weight_initialization/weight_initialization.md) -- Variance conditions, the initialization problem, and practical guidance
- [Xavier (Glorot) Initialization](weight_initialization/xavier_init.md) -- Variance preservation for sigmoid and tanh activations
- [He (Kaiming) Initialization](weight_initialization/he_init.md) -- Adjusted variance for ReLU-family activations in modern architectures

## Normalization

Normalization layers that stabilize and accelerate training by controlling activation distributions.

- [Normalization Layers Overview](normalization/normalization_layers_overview.md) -- Tutorial guide for normalization implementations
- [Normalization Overview](normalization/normalization_overview.md) -- Internal covariate shift, the general framework, and the design space
- [Batch Normalization](normalization/batch_norm.md) -- Normalizing across the batch dimension for each feature or channel
- [Batch Normalization Theory](normalization/batch_norm_theory.md) -- Covariate shift hypothesis, loss-landscape smoothing, and gradient analysis
- [Batch Norm: Training vs Inference](normalization/batch_norm_modes.md) -- Running statistics, mode switching, and common pitfalls
- [Layer Normalization](normalization/layer_norm.md) -- Per-sample normalization across features, the standard for transformers
- [Group Normalization](normalization/group_norm.md) -- Channel-group normalization independent of batch size
- [Instance Normalization](normalization/instance_norm.md) -- Per-sample, per-channel spatial normalization for style transfer and GANs
- [RMSNorm](normalization/rms_norm.md) -- Simplified layer normalization using root mean square, used in LLaMA and Mistral
- [Normalization Comparison](normalization/comparison.md) -- Comprehensive comparison table and guidance on when to use each method

## Regularization

Strategies that constrain learning to improve generalization and prevent overfitting.

- [Regularization Techniques Overview](regularization/regularization_techniques_overview.md) -- Tutorial package with practical implementations
- [Regularization Overview](regularization/regularization_overview.md) -- Bias-variance tradeoff, constrained optimization geometry, and technique taxonomy
- [Dropout](regularization/dropout.md) -- Randomly zeroing activations to prevent co-adaptation of neurons
- [DropConnect](regularization/dropconnect.md) -- Zeroing individual weights for finer-grained stochastic regularization
- [Early Stopping](regularization/early_stopping.md) -- Halting training when validation performance stops improving
- [L1 Regularization](regularization/l1_regularization.md) -- Lasso penalty promoting sparsity and automatic feature selection
- [L2 Regularization](regularization/l2_regularization.md) -- Ridge penalty encouraging small, smooth weight distributions
- [Elastic Net](regularization/elastic_net.md) -- Combined L1 and L2 penalties for stability with correlated features
- [Data Augmentation](regularization/data_augmentation.md) -- Expanding the training set with semantically-preserving transformations
- [Label Smoothing](regularization/label_smoothing.md) -- Soft targets to prevent overconfident predictions and improve calibration
- [Mixup](regularization/mixup.md) -- Training on convex combinations of example pairs for smoother decision boundaries
- [CutMix](regularization/cutmix.md) -- Cutting and pasting image patches with proportional label mixing
- [Cutout](regularization/cutout.md) -- Random rectangular masking to encourage reliance on diverse spatial features
- [Noise Injection](regularization/noise_injection.md) -- Adding random perturbations to inputs, weights, or activations for robustness
>>>>>>> 96f31bd (...)
