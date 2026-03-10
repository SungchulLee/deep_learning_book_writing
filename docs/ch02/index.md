# Chapter 2: PyTorch Fundamentals

!!! warning "Incomplete page"
    This is a chapter landing page and does not follow the five-section structure. It serves as a navigation overview for the sections below.

This chapter introduces PyTorch from the ground up, covering tensors, automatic differentiation, gradient descent, and GPU acceleration. By the end, you will have a solid command of the core abstractions that underpin every deep learning model.

## Tensors

The fundamental data structure in PyTorch -- creation, operations, and interoperability.

- Tensor Basics -- What tensors are, ranks, shapes, and the relationship to NumPy arrays
- Tensor Creation -- Factory functions, data types, type conversions, and device placement
- Tensor Operations -- Arithmetic, reductions, statistics, and linear algebra
- Indexing and Slicing -- Basic and advanced indexing, views vs copies, and boolean masking
- NumPy Interoperability -- Memory sharing, conversions, and Pandas integration

## Tensor Attributes

Shape manipulation, memory layout, broadcasting, and dtype/device management.

- Dtype and Device -- Data type selection, precision tradeoffs, and CPU/GPU transfers
- Broadcasting Rules -- Automatic shape expansion for element-wise operations
- Shape Manipulation -- Concatenation, stacking, splitting, and squeezing
- Reshaping and View -- Views vs copies, `view`, `reshape`, `contiguous`, and `flatten`
- [Memory Layout and Strides](tensor_attrs/memory_layout_strides.md) -- Storage-stride model and offset calculation

## Autograd

Automatic differentiation -- the engine behind gradient-based training.

- [Gradient Computation](autograd/gradient_computation.md) -- Jacobian matrices, vector-Jacobian products, and forward vs reverse mode AD
- Computational Graphs -- Dynamic graph construction, leaf vs non-leaf tensors
- Backward Pass -- Mechanics of `.backward()` and gradient clipping
- Gradient Accumulation -- Default accumulation, zeroing gradients, and large-batch simulation
- Custom Autograd Functions -- `torch.autograd.Function` and `gradcheck`

## Gradient Descent

The optimization algorithm at the core of neural network training.

- [Steepest Direction](gradient_descent/steepest_direction.md) -- Why the gradient points in the direction of steepest ascent
- [Learning Rate](gradient_descent/learning_rate.md) -- Effects on convergence and selection strategies
- [Batch, Mini-Batch, and SGD](gradient_descent/batch_minibatch_sgd.md) -- Three variants and their tradeoffs
- [Convex vs Non-Convex](gradient_descent/convex_nonconvex.md) -- Convexity and deep learning landscapes

## Performance

GPU acceleration, mixed precision, and optimization techniques.

- GPU Acceleration -- CPU vs GPU architecture, CUDA cores, and when GPUs help
- Device Management -- `torch.device`, tensor placement, and operand compatibility
- Mixed Precision -- Float16, bfloat16, and automatic mixed precision with `torch.cuda.amp`
