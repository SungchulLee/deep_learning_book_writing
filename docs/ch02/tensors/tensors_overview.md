# PyTorch Tutorial Series for Undergraduates

Welcome to the comprehensive PyTorch tutorial series! This collection of Python scripts takes you from tensor basics to advanced deep learning concepts, with fully commented code and practical examples.

## 📚 Course Structure

This tutorial is organized into 5 progressive levels, each building on the previous one:

### **Level 1: Tensor Fundamentals** (Beginner)
Learn the building blocks of PyTorch - tensors and basic operations.

- `01_scalar_to_tensor.py` - Creating scalar tensors from Python values
- `02_list_to_tensor.py` - Converting Python lists to tensors
- `03_range_to_tensor.py` - Using ranges and sequences
- `04_numpy_array_to_tensor.py` - NumPy-PyTorch conversion
- `05_pandas_to_tensor.py` - Working with Pandas DataFrames
- `06_factory_functions.py` - PyTorch tensor creation utilities
- `07_clone_vs_inplace.py` - Understanding memory and tensor copying

### **Level 2: Tensor Operations** (Beginner-Intermediate)
Master essential tensor manipulations and operations.

- `08_indexing_slicing.py` - Accessing and modifying tensor elements
- `09_reshaping_views.py` - Changing tensor dimensions safely
- `10_arithmetic_operations.py` - Basic math operations on tensors
- `11_broadcasting.py` - Understanding automatic dimension matching
- `12_reduction_operations.py` - Aggregating tensor data (sum, mean, etc.)
- `13_logical_operations.py` - Boolean operations and masking

### **Level 3: Linear Algebra & Advanced Operations** (Intermediate)
Apply mathematical operations essential for machine learning.

- `14_matrix_operations.py` - Matrix multiplication and linear algebra
- `15_eigenvalues_svd.py` - Decompositions and advanced linear algebra
- `16_concatenation_stacking.py` - Combining tensors
- `17_tensor_comparison.py` - Element-wise and tensor comparisons

### **Level 4: Autograd & Neural Network Basics** (Intermediate-Advanced)
Understand automatic differentiation and build simple neural networks.

- `18_autograd_basics.py` - Introduction to automatic differentiation
- `19_gradient_computation.py` - Computing gradients for optimization
- `20_custom_functions.py` - Extending autograd with custom operations
- `21_simple_neural_network.py` - Building a basic feedforward network
- `22_optimization_basics.py` - Optimizers and training loops

### **Level 5: Advanced Topics & Best Practices** (Advanced)
Master GPU computing, performance optimization, and real-world applications.

- `23_gpu_operations.py` - Moving tensors to GPU and CUDA basics
- `24_mixed_precision.py` - Efficient training with mixed precision
- `25_memory_management.py` - Optimizing memory usage
- `26_data_loading.py` - Efficient data pipelines with DataLoader
- `27_model_checkpoint.py` - Saving and loading models
- `28_real_world_example.py` - Complete end-to-end project

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Basic Python programming knowledge
- Understanding of basic mathematics (algebra, calculus helpful but not required)

### Installation

```bash
# Install PyTorch (visit https://pytorch.org for your specific system)
# CPU-only version:
pip install torch torchvision torchaudio

# With CUDA 11.8 (for NVIDIA GPUs):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additional dependencies for some tutorials
pip install numpy pandas matplotlib
```

### How to Use These Tutorials

1. **Start from Level 1** and work your way up sequentially
2. **Run each script** in order - they build on each other
3. **Read the comments** carefully - they explain concepts and gotchas
4. **Experiment** - modify the code and observe the results
5. **Complete exercises** - some files have challenge sections at the end

### Running a Tutorial

```bash
# Navigate to the tutorial directory
cd pytorch_tutorials

# Run any tutorial
python 01_scalar_to_tensor.py
```

## 📖 Learning Path Recommendations

### For Complete Beginners
Start with Level 1, spend 2-3 days on each tutorial, and ensure you understand each concept before moving on.

### For Those with NumPy Experience
You can skim Level 1 (focus on files 06-07) and spend more time on Levels 2-3.

### For ML Practitioners
Focus on Levels 4-5 for PyTorch-specific features like autograd and GPU optimization.

## 🎯 Learning Objectives

By the end of this tutorial series, you will be able to:

- ✅ Create and manipulate PyTorch tensors efficiently
- ✅ Understand tensor operations, broadcasting, and memory management
- ✅ Use automatic differentiation for gradient computation
- ✅ Build and train simple neural networks from scratch
- ✅ Leverage GPU acceleration for faster computation
- ✅ Apply best practices for model training and evaluation
- ✅ Debug common PyTorch issues and optimize performance

## 💡 Tips for Success

1. **Type the code yourself** - don't just copy-paste. This builds muscle memory.
2. **Break things intentionally** - try to create errors to understand the boundaries
3. **Use print statements** - visualize tensor shapes and values frequently
4. **Read error messages** - PyTorch errors are usually informative
5. **Check the PyTorch docs** - https://pytorch.org/docs/stable/index.html
6. **Practice regularly** - 30 minutes daily beats 3 hours once a week

## 🐛 Common Pitfalls

- **Shape mismatches** - Always check tensor shapes before operations
- **Device mismatches** - Ensure all tensors are on the same device (CPU/GPU)
- **In-place operations** - Be careful with operations ending in `_` (e.g., `add_`)
- **Gradient accumulation** - Remember to call `optimizer.zero_grad()`
- **Memory leaks** - Use `.detach()` when you don't need gradients

## 📚 Additional Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **PyTorch Forums**: https://discuss.pytorch.org/
- **Deep Learning Book**: https://www.deeplearningbook.org/
- **CS231n Stanford Course**: http://cs231n.stanford.edu/

## 🤝 Contributing

Found an error or want to suggest improvements? Please:
1. Note the file name and line number
2. Describe the issue or suggestion clearly
3. Provide a minimal example if reporting a bug

## 📝 License

These tutorials are for educational purposes. Feel free to use and modify them for learning.

## 🙏 Acknowledgments

These tutorials draw inspiration from:
- Official PyTorch documentation
- Stanford CS231n course materials
- Fast.ai practical deep learning course
- Various PyTorch community contributions

---

**Happy Learning! 🎉**

Start with `01_scalar_to_tensor.py` and work your way through. Remember: understanding deeply is better than moving quickly!
