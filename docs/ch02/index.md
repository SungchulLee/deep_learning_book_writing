<<<<<<< HEAD
# Chapter Overview

This chapter covers **Algorithm Analysis**.

# Reference

[Introduction to Algorithms (CLRS), Chapters 2-4](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
=======
# Chapter 2: PyTorch Fundamentals

This chapter introduces PyTorch from the ground up, covering tensors, automatic differentiation, gradient descent, and GPU acceleration. By the end of this chapter, you will have a solid command of the core abstractions that underpin every deep learning model built in PyTorch. The chapter also covers integration with scikit-learn and finance applications.

## Intro

Why PyTorch, its ecosystem, and initial GPU configuration.

- [Why PyTorch](intro/why_pytorch.md) -- Design principles, dynamic computation graphs, and when to choose PyTorch
- [PyTorch Ecosystem](intro/pytorch_ecosystem.md) -- Domain-specific libraries (vision, NLP, audio), training utilities, and deployment tools
- [GPU Configuration](intro/gpu_configuration.md) -- NVIDIA drivers, CUDA setup, Apple Silicon MPS, and GPU verification

## Tensors

The fundamental data structure in PyTorch -- creation, operations, and interoperability.

- [Tensors Overview](tensors/tensors_overview.md) -- Tutorial series guide covering tensor fundamentals through advanced topics
- [Tensor Basics](tensors/tensor_basics.md) -- What tensors are, ranks, shapes, and the relationship to NumPy arrays
- [Tensor Creation](tensors/tensor_creation.md) -- Factory functions, data types, type conversions, and device placement
- [Tensor Operations](tensors/tensor_operations.md) -- Arithmetic, mathematical functions, reductions, statistics, and linear algebra
- [Indexing and Slicing](tensors/indexing_slicing.md) -- Basic and advanced indexing, views vs copies, and boolean masking
- [NumPy Interoperability](tensors/numpy_interop.md) -- `from_numpy`, `as_tensor`, `tensor`, memory sharing, and Pandas conversion

## Tensor Attributes

Shape manipulation, memory layout, broadcasting, and dtype/device management.

- [Tensor Attributes Overview](tensor_attrs/tensor_attributes_and_methods_overview.md) -- Complete guide to tensor attributes and methods
- [Dtype and Device](tensor_attrs/dtype_device.md) -- Data type selection, precision tradeoffs, and CPU/GPU device transfers
- [Broadcasting Rules](tensor_attrs/broadcasting_rules.md) -- Automatic shape expansion for element-wise operations without copying data
- [Shape Manipulation](tensor_attrs/shape_manipulation.md) -- Concatenation, stacking, splitting, and squeezing/unsqueezing
- [Reshaping and View](tensor_attrs/reshaping_view.md) -- Views vs copies, `view`, `reshape`, `contiguous`, and `flatten`
- [Memory Layout and Strides](tensor_attrs/memory_layout_strides.md) -- Storage-stride model, row-major ordering, and offset calculation
- [Memory Management](tensor_attrs/memory_management.md) -- Views, clones, in-place operations, and GPU memory management

## Autograd

Automatic differentiation -- the engine behind gradient-based training.

- [Gradients Overview](autograd/gradients_overview.md) -- Tutorial package overview for gradient computation
- [Getting Started](autograd/getting_started.md) -- Quick-start guide for autograd fundamentals
- [Gradient Computation](autograd/gradient_computation.md) -- Jacobian matrices, vector-Jacobian products, and forward vs reverse mode AD
- [Computational Graphs](autograd/computational_graphs.md) -- Dynamic graph construction, leaf vs non-leaf tensors, and `retain_graph`
- [Backward Pass](autograd/backward_pass.md) -- Mechanics of `.backward()`, training loop structure, and gradient clipping
- [Gradient Accumulation](autograd/gradient_accumulation.md) -- Default accumulation behavior, zeroing gradients, and large-batch simulation
- [Detach and No Grad](autograd/detach_no_grad.md) -- `torch.no_grad()`, `detach()`, `requires_grad`, and parameter freezing
- [Custom Autograd Functions](autograd/custom_autograd.md) -- `torch.autograd.Function`, custom forward/backward, and `gradcheck`
- [Higher-Order Gradients](autograd/higher_order_gradients.md) -- Second derivatives, Hessians, Hessian-vector products, and `create_graph`

## Gradient Descent

The optimization algorithm at the core of all neural network training.

- [Gradient Descent Overview](gradient_descent/gradient_descent_overview.md) -- Tutorial package covering gradient descent from basics to advanced
- [Tutorial Guide](gradient_descent/tutorial_guide.md) -- Quick-start guide and package structure
- [Level 1: Basics](gradient_descent/level_1_basics_overview.md) -- First principles, manual implementation, and visualization
- [Level 2: Intermediate](gradient_descent/level_2_intermediate_overview.md) -- Batch vs mini-batch vs SGD, momentum, and learning rate schedules
- [Iterative Refinement](gradient_descent/iterative_refinement.md) -- The optimization problem and gradient descent as iterative improvement
- [Steepest Direction](gradient_descent/steepest_direction.md) -- Why the gradient points in the direction of steepest ascent
- [Learning Rate](gradient_descent/learning_rate.md) -- The role of learning rate, its effects on convergence, and selection strategies
- [Batch, Mini-Batch, and SGD](gradient_descent/batch_minibatch_sgd.md) -- Three gradient descent variants and their tradeoffs
- [Convex vs Non-Convex](gradient_descent/convex_nonconvex.md) -- Convexity definitions, implications for optimization, and deep learning landscapes
- [Critical Points](gradient_descent/critical_points.md) -- Local minima, saddle points, plateaus, and the Hessian classification
- [Polynomial Sine Fitting](gradient_descent/polynomial_sine_fitting.md) -- Six progressively more PyTorch-idiomatic implementations of curve fitting

## Performance

GPU acceleration, mixed precision, device management, and optimization techniques.

- [GPU Acceleration](performance/gpu_acceleration.md) -- CPU vs GPU architecture, CUDA cores, and when GPUs help
- [Device Management](performance/device_management.md) -- `torch.device`, tensor placement, and ensuring operand compatibility
- [Mixed Precision](performance/mixed_precision.md) -- Float16, bfloat16, and automatic mixed precision training with `torch.cuda.amp`
- [Performance Optimization](performance/performance_optimization.md) -- Data loading overlap, profiling, and eliminating GPU idle time

## Finance Applications

Domain-specific patterns for quantitative finance with PyTorch.

- [Factor Models](finance/factor_models.md) -- Cross-sectional factor regression and Fama-MacBeth approach
- [Credit Scoring](finance/credit.md) -- Imbalanced classification with calibrated probabilities
- [Time Series CV](finance/time_series_cv.md) -- Temporal cross-validation respecting causality

## PyTorch-Sklearn Integration

Bridging PyTorch models with scikit-learn workflows.

- [Skorch](pytorch/skorch.md) -- Wrapping PyTorch modules as sklearn estimators
- [Custom Estimators](pytorch/custom_estimator.md) -- Implementing the sklearn estimator interface for PyTorch models
- [Hybrid Pipelines](pytorch/hybrid.md) -- Combining sklearn preprocessing with PyTorch models
>>>>>>> 96f31bd (...)
