# Maximum Likelihood Estimation (MLE) with PyTorch
## A Complete Tutorial for Undergraduates

Welcome to this comprehensive tutorial on Maximum Likelihood Estimation using PyTorch! This tutorial is designed to take you from basic probability estimation to advanced neural network applications.

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [What is Maximum Likelihood Estimation?](#what-is-mle)
3. [Installation](#installation)
4. [Tutorial Structure](#tutorial-structure)
5. [How to Use This Tutorial](#how-to-use-this-tutorial)
6. [Mathematical Background](#mathematical-background)
7. [References](#references)

---

## 🎯 Introduction

Maximum Likelihood Estimation (MLE) is one of the most fundamental concepts in statistics and machine learning. This tutorial uses PyTorch to help you understand MLE through hands-on examples, progressing from simple to complex applications.

**What you'll learn:**
- The theory behind MLE
- How to implement MLE in PyTorch
- Practical applications in statistics and machine learning
- How gradient descent relates to MLE
- Real-world examples with visualizations

---

## 🤔 What is MLE?

**Maximum Likelihood Estimation** is a method for estimating the parameters of a statistical model. Given a dataset and a probability model, MLE finds the parameter values that make the observed data most probable.

### The Core Idea

Imagine you flip a coin 100 times and get 60 heads. What's the most likely probability of getting heads? Intuitively, you'd say 0.6. That's MLE in action!

### Mathematical Formulation

Given:
- Data: **X** = {x₁, x₂, ..., xₙ}
- Model with parameters: **θ**
- Likelihood function: **L(θ | X)** = P(X | θ)

Goal: Find **θ*** = argmax L(θ | X)

In practice, we maximize the **log-likelihood**:
**ℓ(θ | X)** = log L(θ | X)

This is easier to work with and numerically more stable.

---

## 💻 Installation

### Requirements
- Python 3.8 or higher
- PyTorch 2.0 or higher
- NumPy
- Matplotlib
- SciPy (optional, for some examples)

### Quick Install

```bash
# Create a virtual environment (recommended)
python -m venv mle_env
source mle_env/bin/activate  # On Windows: mle_env\Scripts\activate

# Install required packages
pip install torch numpy matplotlib scipy
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

---

## 📁 Tutorial Structure

The tutorial is organized into three difficulty levels:

### 🟢 Level 1: Easy (`01_easy/`)

Perfect for beginners! Learn the basics with simple examples.

1. **`coin_flip_mle.py`** - Estimate the probability of a biased coin
   - Learn: Basic MLE, Bernoulli distribution
   - Concepts: Likelihood, log-likelihood, visualization

2. **`dice_roll_mle.py`** - Estimate parameters of a loaded die
   - Learn: Categorical distribution, multiple parameters
   - Concepts: Discrete MLE, constraint optimization

### 🟡 Level 2: Medium (`02_medium/`)

Build on the basics with more realistic scenarios.

3. **`linear_regression_mle.py`** - MLE for linear regression
   - Learn: Connection between MLE and least squares
   - Concepts: Continuous distributions, Gaussian noise

4. **`normal_distribution_mle.py`** - Estimate mean and variance
   - Learn: Multi-parameter estimation
   - Concepts: Gaussian distribution, joint optimization

5. **`capture_recapture_mle.py`** - Wildlife population estimation
   - Learn: Real-world application of MLE
   - Concepts: Ecological statistics, hypergeometric distribution

### 🔴 Level 3: Advanced (`03_advanced/`)

Tackle machine learning applications and complex models.

6. **`logistic_regression_mle.py`** - Binary classification with MLE
   - Learn: MLE in classification, cross-entropy loss
   - Concepts: Sigmoid function, gradient descent

7. **`mixture_of_gaussians_mle.py`** - Clustering with EM algorithm
   - Learn: Expectation-Maximization (EM)
   - Concepts: Latent variables, soft clustering

8. **`neural_network_mle.py`** - Deep learning with MLE
   - Learn: How neural networks use MLE
   - Concepts: Backpropagation, custom loss functions

---

## 🚀 How to Use This Tutorial

### For Complete Beginners

1. **Start with Level 1** - Don't skip ahead!
2. **Read the code comments** - Every line is explained
3. **Run the examples** - See the visualizations
4. **Experiment** - Change parameters and see what happens
5. **Do the exercises** - Found at the end of each script

### For Experienced Learners

1. **Review the math** in each script's docstring
2. **Jump to Level 2 or 3** if you're comfortable
3. **Focus on PyTorch implementation** details
4. **Compare with other optimization methods**

### Running an Example

```bash
# Navigate to the tutorial directory
cd mle_pytorch_tutorial

# Run an easy example
python 01_easy/coin_flip_mle.py

# Run with custom parameters (if available)
python 01_easy/coin_flip_mle.py --n_flips 1000 --true_p 0.7

# Run a medium example
python 02_medium/linear_regression_mle.py

# Run an advanced example
python 03_advanced/logistic_regression_mle.py
```

---

## 📖 Mathematical Background

### Why Maximize Likelihood?

The likelihood function tells us how probable our observed data is under different parameter values. By maximizing likelihood, we find the parameters that best explain our data.

### Why Use Log-Likelihood?

1. **Numerical Stability**: Products of small probabilities can underflow
2. **Computational Efficiency**: Sums are faster than products
3. **Mathematical Convenience**: Derivatives are simpler

### The Connection to Loss Functions

In machine learning, we often **minimize loss** instead of **maximize likelihood**. These are equivalent:

```
minimize Loss(θ) ≡ maximize Likelihood(θ)

where: Loss(θ) = -log Likelihood(θ)
```

This is why:
- **Cross-entropy loss** = Negative log-likelihood for classification
- **Mean Squared Error** = Negative log-likelihood for Gaussian regression

### Gradient Descent for MLE

PyTorch's automatic differentiation makes MLE easy:

1. Define your likelihood function
2. Take the negative log (to turn max into min)
3. Use gradient descent to optimize
4. PyTorch handles all the derivatives!

---

## 🎓 Learning Path

```
Start Here → Coin Flip (Easy)
                ↓
            Dice Roll (Easy)
                ↓
        Linear Regression (Medium)
                ↓
        Normal Distribution (Medium)
                ↓
        Capture-Recapture (Medium)
                ↓
        Logistic Regression (Advanced)
                ↓
        Mixture of Gaussians (Advanced)
                ↓
        Neural Networks (Advanced)
```

---

## 💡 Key Takeaways

After completing this tutorial, you'll understand:

1. ✅ What MLE is and why it's important
2. ✅ How to implement MLE in PyTorch
3. ✅ The connection between MLE and machine learning loss functions
4. ✅ How to use gradient descent for parameter estimation
5. ✅ Real-world applications of MLE
6. ✅ How neural networks use MLE principles

---

## 📚 References

### Books
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "Deep Learning" by Goodfellow, Bengio, and Courville

### Online Resources
- [PyTorch Documentation](https://pytorch.org/docs/)
- [StatQuest YouTube Channel](https://www.youtube.com/c/joshstarmer)
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

### Papers
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). "Maximum likelihood from incomplete data via the EM algorithm"

---

## 🤝 Contributing

Found a bug or have a suggestion? Feel free to:
- Modify the code for your learning
- Add comments or explanations
- Create your own MLE examples

---

## 📄 License

This tutorial is provided for educational purposes. Feel free to use and modify for learning.

---

## 🎉 Happy Learning!

Remember: The best way to learn is by doing. Run the code, break it, fix it, and experiment!

**Questions to explore:**
- What happens when you have very little data?
- How does the likelihood surface change with different models?
- When does MLE work well vs. poorly?
- How does regularization relate to MLE?

Start with `01_easy/coin_flip_mle.py` and enjoy your journey into Maximum Likelihood Estimation!
