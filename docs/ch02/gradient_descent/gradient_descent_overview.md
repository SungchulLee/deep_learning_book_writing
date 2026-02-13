# PyTorch Gradient Descent Tutorial 🚀

A comprehensive, hands-on tutorial on gradient descent optimization using PyTorch, designed for undergraduate students. This package contains progressively challenging examples with detailed comments and explanations.

## 📚 Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Tutorial Structure](#tutorial-structure)
- [Quick Start](#quick-start)
- [Learning Path](#learning-path)
- [Additional Resources](#additional-resources)

## 🎯 Overview

Gradient descent is the cornerstone of modern machine learning and deep learning. This tutorial will guide you through:
- Understanding gradient descent from first principles
- Implementing gradient descent manually (NumPy)
- Leveraging PyTorch's autograd for automatic differentiation
- Exploring variants: SGD, Mini-batch GD, Momentum, Adam, RMSprop
- Applying gradient descent to real-world problems

## 📋 Prerequisites

**Required Knowledge:**
- Basic Python programming
- Elementary calculus (derivatives, chain rule)
- Linear algebra (vectors, matrices)
- Basic understanding of machine learning concepts

**Software Requirements:**
- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- (Optional) Jupyter Notebook for interactive learning

## 🔧 Installation

```bash
# Create a virtual environment (recommended)
python -m venv gd_env
source gd_env/bin/activate  # On Windows: gd_env\Scripts\activate

# Install required packages
pip install torch torchvision numpy matplotlib scikit-learn jupyter
```

## 📁 Tutorial Structure

```
pytorch_gradient_descent_tutorial/
│
├── level_1_basics/              # Start here! 🌱
│   ├── 01_manual_gradient_numpy.py
│   ├── 02_pytorch_autograd_basics.py
│   ├── 03_simple_linear_regression.py
│   └── 04_visualizing_gradient_descent.py
│
├── level_2_intermediate/        # Build your skills 💪
│   ├── 01_batch_vs_minibatch_sgd.py
│   ├── 02_momentum_optimizer.py
│   ├── 03_learning_rate_schedules.py
│   └── 04_polynomial_regression.py
│
├── level_3_advanced/            # Master the techniques 🎓
│   ├── 01_adam_optimizer.py
│   ├── 02_rmsprop_optimizer.py
│   ├── 03_comparing_optimizers.py
│   └── 04_gradient_descent_convergence.py
│
├── level_4_projects/            # Real-world applications 🌟
│   ├── 01_neural_network_mnist.py
│   ├── 02_logistic_regression.py
│   ├── 03_custom_loss_functions.py
│   └── 04_gradient_descent_visualization_3d.py
│
├── utils/                       # Helper functions
│   ├── plotting.py
│   └── datasets.py
│
└── datasets/                    # Sample datasets
    └── README.md
```

## 🚀 Quick Start

### Option 1: Sequential Learning (Recommended for Beginners)
Start from Level 1 and work your way up:

```bash
# Level 1: Understand the basics
cd level_1_basics
python 01_manual_gradient_numpy.py
python 02_pytorch_autograd_basics.py
# ... continue with other files
```

### Option 2: Jump to Your Level
If you already understand basics, jump to the appropriate level:

```bash
# For intermediate learners
cd level_2_intermediate
python 01_batch_vs_minibatch_sgd.py
```

### Option 3: Interactive Learning (Jupyter)
```bash
jupyter notebook
# Open any .py file and run cells interactively
```

## 📖 Learning Path

### Level 1: Basics (2-3 hours) 🌱
**Goal:** Understand what gradient descent is and how it works

1. **01_manual_gradient_numpy.py**
   - Implement gradient descent from scratch
   - Understand the math: gradients, learning rate, iterations
   - Visualize the optimization process

2. **02_pytorch_autograd_basics.py**
   - Learn PyTorch's automatic differentiation
   - Understand computational graphs
   - Compare with manual implementation

3. **03_simple_linear_regression.py**
   - Apply gradient descent to a real problem
   - Understand loss functions and optimization
   - Evaluate model performance

4. **04_visualizing_gradient_descent.py**
   - See gradient descent in action
   - Understand convergence and local minima
   - Experiment with different learning rates

**Key Concepts:** Gradients, learning rate, loss function, forward/backward pass

### Level 2: Intermediate (3-4 hours) 💪
**Goal:** Learn different variants and improve optimization

1. **01_batch_vs_minibatch_sgd.py**
   - Understand batch, mini-batch, and stochastic gradient descent
   - Compare convergence speed and memory usage
   - Learn when to use each variant

2. **02_momentum_optimizer.py**
   - Accelerate convergence with momentum
   - Understand exponential moving averages
   - Escape shallow local minima

3. **03_learning_rate_schedules.py**
   - Implement learning rate decay strategies
   - Understand the importance of learning rate tuning
   - Improve final convergence

4. **04_polynomial_regression.py**
   - Fit non-linear functions
   - Understand feature engineering
   - Prevent overfitting

**Key Concepts:** Mini-batch training, momentum, learning rate scheduling, regularization

### Level 3: Advanced (4-5 hours) 🎓
**Goal:** Master modern optimization algorithms

1. **01_adam_optimizer.py**
   - Implement the Adam optimizer
   - Understand adaptive learning rates
   - Compare with other optimizers

2. **02_rmsprop_optimizer.py**
   - Learn RMSprop optimization
   - Understand adaptive gradient scaling
   - Handle different gradient magnitudes

3. **03_comparing_optimizers.py**
   - Benchmark all optimizers
   - Understand trade-offs
   - Choose the right optimizer for your problem

4. **04_gradient_descent_convergence.py**
   - Analyze convergence properties
   - Understand saddle points and plateaus
   - Debug optimization problems

**Key Concepts:** Adaptive learning rates, momentum variants, optimizer selection

### Level 4: Projects (5-6 hours) 🌟
**Goal:** Apply knowledge to real-world problems

1. **01_neural_network_mnist.py**
   - Build a neural network from scratch
   - Train on MNIST digit dataset
   - Achieve >95% accuracy

2. **02_logistic_regression.py**
   - Implement logistic regression
   - Binary and multi-class classification
   - Understand cross-entropy loss

3. **03_custom_loss_functions.py**
   - Design custom loss functions
   - Implement specialized objectives
   - Solve domain-specific problems

4. **04_gradient_descent_visualization_3d.py**
   - Create beautiful 3D visualizations
   - Understand optimization landscapes
   - Interactive gradient descent visualization

**Key Concepts:** Neural networks, classification, custom objectives, visualization

## 💡 Tips for Success

1. **Run the code yourself** - Don't just read, execute and experiment!
2. **Modify parameters** - Change learning rates, batch sizes, etc. and observe effects
3. **Add print statements** - Understand what's happening at each step
4. **Visualize everything** - Use matplotlib to see loss curves, gradients, etc.
5. **Start simple** - Master Level 1 before moving to advanced topics
6. **Read comments carefully** - They contain important insights and best practices

## 🎓 Learning Outcomes

After completing this tutorial, you will be able to:
- ✅ Explain how gradient descent works mathematically
- ✅ Implement gradient descent from scratch
- ✅ Use PyTorch's autograd for automatic differentiation
- ✅ Choose appropriate optimizers for different problems
- ✅ Tune hyperparameters (learning rate, batch size, etc.)
- ✅ Debug common optimization issues
- ✅ Apply gradient descent to real machine learning problems

## 📚 Additional Resources

**Books:**
- "Deep Learning" by Goodfellow, Bengio, and Courville (Chapter 8)
- "Dive into Deep Learning" - d2l.ai (Free online book)

**Online Courses:**
- PyTorch Official Tutorials: https://pytorch.org/tutorials/
- Stanford CS231n: Neural Networks

**Papers:**
- "Adam: A Method for Stochastic Optimization" - Kingma & Ba
- "On the importance of initialization and momentum in deep learning" - Sutskever et al.

**Interactive Tools:**
- TensorFlow Playground: https://playground.tensorflow.org/
- Distill.pub visualizations: https://distill.pub/

## 🤝 Contributing

Found a bug or have a suggestion? Feel free to:
- Modify the code for your learning
- Add new examples
- Improve documentation
- Share with others!

## 📝 License

This educational material is provided for learning purposes. Feel free to use, modify, and share!

## 🙏 Acknowledgments

Based on excellent resources from:
- PyTorch official tutorials
- Patrick Loeber's PyTorch Tutorial
- Stanford CS231n course materials
- Various open-source implementations

---

**Happy Learning! 🎉**

*Remember: The best way to learn is by doing. Start coding, experiment freely, and don't be afraid to make mistakes!*
