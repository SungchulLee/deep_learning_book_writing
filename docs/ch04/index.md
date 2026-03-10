# Chapter 4: NN Building Blocks


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

This chapter covers the fundamental components used to construct neural networks: activation functions, feedforward network architecture, weight initialization, normalization layers, and regularization techniques. Each topic is developed from mathematical first principles with accompanying PyTorch implementations. Together, these building blocks form the vocabulary for assembling and training any deep learning model.

## Activations

Activation functions introduce nonlinearity into neural networks, enabling them to learn complex patterns.

- Activation Functions Overview -- Tutorial package overview with progressive examples
- Activation Overview -- Why nonlinearity is necessary, the collapse proof, and the activation function taxonomy
- ReLU -- Rectified Linear Unit: the default hidden-layer activation, solving the vanishing gradient problem
- Sigmoid -- The logistic function for binary output probabilities and gating mechanisms
- Tanh -- Hyperbolic tangent for zero-centered bounded activations in recurrent networks
- Leaky ReLU -- Non-zero gradient for negative inputs, solving the dead neuron problem
- PReLU -- Parametric ReLU with a learnable negative slope optimized during training
- [ELU](activations/elu.md) -- Exponential Linear Unit with smooth saturation for negative inputs
- SELU -- Scaled ELU enabling self-normalizing networks without explicit normalization layers
- GELU -- Gaussian Error Linear Unit, the standard activation for transformer architectures
- Swish / SiLU -- Self-gating activation used in EfficientNet and modern LLMs via SwiGLU
- Mish -- Smooth non-monotonic activation adopted in YOLOv4 for object detection
- Softmax -- Converting logits to probability distributions for multiclass classification
- Selection Guide -- Systematic framework for choosing activations by architecture and task

## Feedforward Networks

Multi-layer perceptrons: architecture, forward and backward propagation, and gradient flow.

- Feedforward Networks Overview -- Complete tutorial combining mathematical foundations and PyTorch mastery
- Getting Started -- Quick-start installation and first steps
- Quick Reference -- Cheat sheet for quick lookups by topic
- Level 0: Foundations -- Building neural networks from scratch with NumPy
- Level 1: PyTorch Basics -- Autograd, `nn.Module`, and optimizer usage
- Level 2: Building Networks -- MNIST, multiple architectures, and activation comparisons
- Level 3: Advanced Techniques -- Regularization, normalization, scheduling, and initialization
- Level 4: Applications -- CIFAR-10, regression, multi-task learning, and deep architectures
- [MLP Architecture](feedforward/mlp_architecture.md) -- Single neuron to full network, parameter counting, and PyTorch implementation
- [Universal Approximation](feedforward/universal_approximation.md) -- The theorem, geometric intuition, and width complexity analysis
- [Depth vs Width](feedforward/depth_vs_width.md) -- Exponential separation results, hierarchical composition, and practical tradeoffs
- [Forward Pass](feedforward/forward_pass.md) -- Propagation equations, numerical trace, computational graph, and complexity analysis
- [Backpropagation](feedforward/backpropagation.md) -- Chain rule derivation, hand computation, general recurrence, and autograd verification
- [Gradient Flow](feedforward/gradient_flow.md) -- Vanishing and exploding gradients, Jacobian analysis, and mitigation strategies

## Weight Initialization

Setting initial weight distributions to maintain stable signal propagation through deep networks.

- [Weight Initialization](weight_initialization/weight_initialization.md) -- Variance conditions, the initialization problem, and practical guidance
- Xavier (Glorot) Initialization -- Variance preservation for sigmoid and tanh activations
- [He (Kaiming) Initialization](weight_initialization/he_init.md) -- Adjusted variance for ReLU-family activations in modern architectures

## Normalization

Normalization layers that stabilize and accelerate training by controlling activation distributions.

- Normalization Layers Overview -- Tutorial guide for normalization implementations
- Normalization Overview -- Internal covariate shift, the general framework, and the design space
- Batch Normalization -- Normalizing across the batch dimension for each feature or channel
- Batch Normalization Theory -- Covariate shift hypothesis, loss-landscape smoothing, and gradient analysis
- Batch Norm: Training vs Inference -- Running statistics, mode switching, and common pitfalls
- Layer Normalization -- Per-sample normalization across features, the standard for transformers
- Group Normalization -- Channel-group normalization independent of batch size
- Instance Normalization -- Per-sample, per-channel spatial normalization for style transfer and GANs
- [RMSNorm](normalization/rms_norm.md) -- Simplified layer normalization using root mean square, used in LLaMA and Mistral
- Normalization Comparison -- Comprehensive comparison table and guidance on when to use each method

## Regularization

Strategies that constrain learning to improve generalization and prevent overfitting.

- Regularization Techniques Overview -- Tutorial package with practical implementations
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
