# Chapter 2: PyTorch Fundamentals


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

This chapter introduces PyTorch from the ground up, covering tensors, automatic differentiation, gradient descent, and GPU acceleration. By the end of this chapter, you will have a solid command of the core abstractions that underpin every deep learning model built in PyTorch. The chapter also covers integration with scikit-learn and finance applications.

## Intro

Why PyTorch, its ecosystem, and initial GPU configuration.

- Why PyTorch -- Design principles, dynamic computation graphs, and when to choose PyTorch
- PyTorch Ecosystem -- Domain-specific libraries (vision, NLP, audio), training utilities, and deployment tools
- GPU Configuration -- NVIDIA drivers, CUDA setup, Apple Silicon MPS, and GPU verification

## Tensors

The fundamental data structure in PyTorch -- creation, operations, and interoperability.

- Tensors Overview -- Tutorial series guide covering tensor fundamentals through advanced topics
- Tensor Basics -- What tensors are, ranks, shapes, and the relationship to NumPy arrays
- Tensor Creation -- Factory functions, data types, type conversions, and device placement
- Tensor Operations -- Arithmetic, mathematical functions, reductions, statistics, and linear algebra
- Indexing and Slicing -- Basic and advanced indexing, views vs copies, and boolean masking
- NumPy Interoperability -- `from_numpy`, `as_tensor`, `tensor`, memory sharing, and Pandas conversion

## Tensor Attributes

Shape manipulation, memory layout, broadcasting, and dtype/device management.

- Tensor Attributes Overview -- Complete guide to tensor attributes and methods
- Dtype and Device -- Data type selection, precision tradeoffs, and CPU/GPU device transfers
- Broadcasting Rules -- Automatic shape expansion for element-wise operations without copying data
- Shape Manipulation -- Concatenation, stacking, splitting, and squeezing/unsqueezing
- Reshaping and View -- Views vs copies, `view`, `reshape`, `contiguous`, and `flatten`
- [Memory Layout and Strides](tensor_attrs/memory_layout_strides.md) -- Storage-stride model, row-major ordering, and offset calculation
- Memory Management -- Views, clones, in-place operations, and GPU memory management

## Autograd

Automatic differentiation -- the engine behind gradient-based training.

- Gradients Overview -- Tutorial package overview for gradient computation
- Getting Started -- Quick-start guide for autograd fundamentals
- [Gradient Computation](autograd/gradient_computation.md) -- Jacobian matrices, vector-Jacobian products, and forward vs reverse mode AD
- Computational Graphs -- Dynamic graph construction, leaf vs non-leaf tensors, and `retain_graph`
- Backward Pass -- Mechanics of `.backward()`, training loop structure, and gradient clipping
- Gradient Accumulation -- Default accumulation behavior, zeroing gradients, and large-batch simulation
- Detach and No Grad -- `torch.no_grad()`, `detach()`, `requires_grad`, and parameter freezing
- Custom Autograd Functions -- `torch.autograd.Function`, custom forward/backward, and `gradcheck`
- Higher-Order Gradients -- Second derivatives, Hessians, Hessian-vector products, and `create_graph`

## Gradient Descent

The optimization algorithm at the core of all neural network training.

- Gradient Descent Overview -- Tutorial package covering gradient descent from basics to advanced
- Tutorial Guide -- Quick-start guide and package structure
- Level 1: Basics -- First principles, manual implementation, and visualization
- Level 2: Intermediate -- Batch vs mini-batch vs SGD, momentum, and learning rate schedules
- Iterative Refinement -- The optimization problem and gradient descent as iterative improvement
- [Steepest Direction](gradient_descent/steepest_direction.md) -- Why the gradient points in the direction of steepest ascent
- [Learning Rate](gradient_descent/learning_rate.md) -- The role of learning rate, its effects on convergence, and selection strategies
- [Batch, Mini-Batch, and SGD](gradient_descent/batch_minibatch_sgd.md) -- Three gradient descent variants and their tradeoffs
- [Convex vs Non-Convex](gradient_descent/convex_nonconvex.md) -- Convexity definitions, implications for optimization, and deep learning landscapes
- Critical Points -- Local minima, saddle points, plateaus, and the Hessian classification
- Polynomial Sine Fitting -- Six progressively more PyTorch-idiomatic implementations of curve fitting

## Performance

GPU acceleration, mixed precision, device management, and optimization techniques.

- GPU Acceleration -- CPU vs GPU architecture, CUDA cores, and when GPUs help
- Device Management -- `torch.device`, tensor placement, and ensuring operand compatibility
- Mixed Precision -- Float16, bfloat16, and automatic mixed precision training with `torch.cuda.amp`
- Performance Optimization -- Data loading overlap, profiling, and eliminating GPU idle time

## Finance Applications

Domain-specific patterns for quantitative finance with PyTorch.

- Factor Models -- Cross-sectional factor regression and Fama-MacBeth approach
- Credit Scoring -- Imbalanced classification with calibrated probabilities
- Time Series CV -- Temporal cross-validation respecting causality

## PyTorch-Sklearn Integration

Bridging PyTorch models with scikit-learn workflows.

- Skorch -- Wrapping PyTorch modules as sklearn estimators
- Custom Estimators -- Implementing the sklearn estimator interface for PyTorch models
- Hybrid Pipelines -- Combining sklearn preprocessing with PyTorch models
