# 04 Score-Based Methods Leading to Diffusion Models

## Overview

This educational package provides the crucial bridge between **01_Bayesian_Inference** and modern **Diffusion Models**. Score-based methods are the mathematical foundation that connects Bayesian posterior inference, MCMC sampling, and diffusion-based generative models.

## Connection to Previous Material

### From 01_Bayesian_Inference to Score-Based Methods

In Bayesian inference, we learned:
- **Posterior inference**: p(θ|D) ∝ p(D|θ)p(θ)
- **Intractable integrals**: p(D) = ∫ p(D|θ)p(θ) dθ is often impossible to compute
- **Need for sampling**: MCMC methods to sample from posteriors

Score-based methods extend these ideas:
- **Score function**: ∇_x log p(x) - gradient of log-probability
- **No normalization needed**: Work directly with unnormalized distributions
- **Sampling via Langevin dynamics**: Uses score to generate samples
- **Connection to diffusion**: Denoising is Bayesian posterior inference!

## Key Insight: Denoising as Bayesian Inference

The fundamental connection:
```
Denoising noisy data = Bayesian posterior inference

Given: x_noisy = x_clean + noise
Find: p(x_clean | x_noisy)

This is exactly the score function at noise level σ!
```

## Package Structure

### **Part 1: Score Functions Fundamentals** (Modules 01-03)
Building from Bayesian foundations to score-based thinking

1. **01_score_functions_basics.py**
   - Definition: s(x) = ∇_x log p(x)
   - Connection to Bayesian inference
   - Visualization and intuition
   - Simple analytical examples

2. **02_score_matching_theory.py**
   - Why we can't compute scores directly
   - Explicit Score Matching (ESM)
   - Denoising Score Matching (DSM) ⭐ KEY IDEA
   - Connection to Bayesian denoising

3. **03_langevin_dynamics.py**
   - From Metropolis-Hastings to Langevin MCMC
   - Using scores for sampling
   - Convergence and theory
   - 2D demonstrations

### **Part 2: Practical Implementation** (Modules 04-06)
From theory to working code

4. **04_denoising_score_matching.py**
   - Practical DSM implementation
   - Training score networks
   - Swiss roll and toy datasets
   - Debugging and visualization

5. **05_sliced_score_matching.py**
   - Computational efficiency
   - Random projections
   - Comparison with DSM
   - High-dimensional data

6. **06_score_networks.py**
   - Neural architectures for scores
   - Time/noise conditioning
   - Training strategies
   - Best practices

### **Part 3: Multi-Scale and Diffusion** (Modules 07-10)
Bridge to full diffusion models

7. **07_noise_conditional_scores.py**
   - Why multiple noise levels?
   - Noise schedules
   - Annealed Langevin dynamics
   - Connection to forward diffusion process

8. **08_score_sde.py**
   - Continuous-time formulation
   - Variance Exploding (VE) SDE
   - Variance Preserving (VP) SDE
   - Probability flow ODE
   - **Direct connection to DDPM!**

9. **09_image_generation.py**
   - Complete generation pipeline
   - U-Net score architecture
   - Training on MNIST
   - Predictor-corrector sampling

10. **10_diffusion_connection.py**
    - Explicit DDPM formulation
    - Forward process = adding noise
    - Reverse process = learned denoising
    - Complete unification!

## Learning Objectives

After completing this package, you will understand:

1. **Mathematical Foundation**
   - Score functions and their properties
   - Fisher divergence and score matching
   - Stein's identity and integration by parts
   - Connection to Fokker-Planck equations

2. **Sampling Methods**
   - Langevin dynamics from Bayesian perspective
   - Annealed sampling strategies
   - Multi-scale noise schedules
   - Predictor-corrector algorithms

3. **Deep Learning Implementation**
   - Score network architectures
   - Training objectives and losses
   - Noise conditioning techniques
   - Sampling and generation

4. **Connection to Diffusion Models**
   - Forward diffusion as noise schedule
   - Reverse diffusion as learned denoising
   - DDPM training objective
   - SDE vs discrete-time formulations

## Prerequisites

### Mathematical Background
- **From 01_Bayesian_Inference**:
  - Bayes theorem and posterior inference
  - Conjugate priors and likelihoods
  - MAP estimation
  - Basic MCMC concepts

- **Additional Requirements**:
  - Vector calculus (gradients, Jacobians)
  - Stochastic processes (Brownian motion)
  - Basic SDE theory (helpful but not required)

### Programming Skills
- Python 3.7+
- PyTorch fundamentals
- NumPy and Matplotlib
- Basic neural network training

## Installation

```bash
# Required packages
pip install torch torchvision numpy matplotlib scipy

# Optional for advanced features
pip install tqdm tensorboard
```

## Suggested Learning Path

### Week 1: Foundations (8-10 hours)
- **Day 1-2**: Modules 01-02
  - Review Bayesian inference concepts
  - Understand score functions
  - Learn score matching theory

- **Day 3-4**: Module 03
  - Implement Langevin dynamics
  - Connect to MCMC from Bayesian inference
  - Visualize sampling processes

### Week 2: Implementation (10-12 hours)
- **Day 1-2**: Module 04
  - Implement denoising score matching
  - Train first score network
  - Debug and visualize

- **Day 3**: Module 05
  - Explore sliced score matching
  - Compare efficiency

- **Day 4**: Module 06
  - Design score network architectures
  - Understand conditioning strategies

### Week 3: Advanced Topics (12-15 hours)
- **Day 1-2**: Modules 07-08
  - Multi-scale score modeling
  - Continuous-time formulation
  - SDE framework

- **Day 3-4**: Modules 09-10
  - Image generation implementation
  - Connect everything to diffusion models
  - Final project: Generate images!

## Key Concepts

### 1. Score Function
```python
# The score is the gradient of log-probability
score(x) = ∇_x log p(x)

# Properties:
# - Points toward high probability
# - Zero at modes
# - No normalization constant needed
```

### 2. Denoising Score Matching
```python
# Key insight: Learn score by denoising
x_noisy = x + σ * noise
score_σ(x_noisy) ≈ ∇_x log p_σ(x_noisy)

# This is Bayesian inference!
# p(x|x_noisy) ∝ p(x_noisy|x)p(x)
```

### 3. Multi-Scale Modeling
```python
# Use multiple noise levels
σ_1 > σ_2 > ... > σ_L > 0

# Learn score at each level
s_θ(x, σ_i) = ∇_x log p_σi(x)

# Sample by annealing through scales
```

### 4. Connection to Diffusion
```python
# Forward process (DDPM)
q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)

# Reverse process (learned)
p_θ(x_{t-1}|x_t) ∝ N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))

# The score tells us the mean!
∇_x log p(x_t) ≈ -ε_θ(x_t,t) / √(1-ᾱ_t)
```

## Mathematical Deep Dives

Each module includes:
- **Complete derivations** in docstrings
- **Step-by-step proofs** of key theorems
- **Intuitive explanations** with visualizations
- **Connections** to Bayesian inference

## Pedagogical Approach

Following your established style:
- ✅ **Heavily commented code** (~50% comments)
- ✅ **Progressive difficulty** levels
- ✅ **Mathematical rigor** with derivations
- ✅ **Complete examples** that run immediately
- ✅ **Extensive visualizations**
- ✅ **Exercises** at end of each module
- ✅ **Clear connections** between concepts

## Quick Start

```bash
# Navigate to package
cd 04_Score_Based_Methods_To_Diffusion

# Run first module
python 01_score_functions_basics.py

# This will:
# 1. Explain score functions
# 2. Show visualizations
# 3. Connect to Bayesian inference
# 4. Save plots to disk
```

## Assessment and Exercises

Each module includes:
1. **Conceptual questions** - Test understanding
2. **Implementation exercises** - Practice coding
3. **Mathematical problems** - Prove key results
4. **Extension projects** - Go deeper

## Common Questions

### Q: How does this connect to Variational Inference?
A: Both VI and score-based methods avoid computing intractable integrals. VI uses a learned approximate posterior, while score-based methods use learned scores + Langevin dynamics.

### Q: Why multiple noise levels?
A: Low noise = precise but hard to sample. High noise = easy to sample but imprecise. Multiple scales let us start easy and refine.

### Q: Is this the same as DDPM?
A: Yes! DDPM is the discrete-time version. This package shows the continuous-time formulation (SDEs) and connects them.

### Q: How long to train?
A: 2D toy data: ~5 minutes. MNIST: ~2-4 hours. Complex images: days on GPU.

## Further Reading

### Foundational Papers
1. **Score Matching**
   - Hyvärinen (2005): "Estimation of Non-Normalized Statistical Models"
   
2. **Score-Based Models**
   - Song & Ermon (2019): "Generative Modeling by Estimating Gradients of the Data Distribution"
   - Song & Ermon (2020): "Improved Techniques for Training Score-Based Generative Models"
   
3. **Continuous-Time Framework**
   - Song et al. (2021): "Score-Based Generative Modeling through Stochastic Differential Equations"

4. **Diffusion Models**
   - Sohl-Dickstein et al. (2015): "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
   - Ho et al. (2020): "Denoising Diffusion Probabilistic Models"

### Recommended Sequence
1. Review 01_Bayesian_Inference materials
2. Work through this package sequentially
3. Continue to full diffusion model implementation
4. Study score-based SDE theory in depth

## Support and Resources

### Getting Help
- **Code issues**: Check inline comments and docstrings first
- **Math questions**: Review derivations in docstrings
- **Conceptual confusion**: Re-read connection sections
- **Bugs**: Verify PyTorch/Python versions

### Teaching Tips
- **Lecture 1**: Modules 01-02 (Score functions and matching)
- **Lecture 2**: Module 03 (Langevin dynamics)
- **Lecture 3**: Modules 04-06 (Practical implementation)
- **Lecture 4**: Modules 07-08 (Multi-scale and SDEs)
- **Lecture 5**: Modules 09-10 (Full pipeline and diffusion connection)

## License

Educational use only. Suitable for undergraduate/graduate instruction.

## Acknowledgments

This package synthesizes concepts from:
- **Classical statistics**: Fisher, Stein, Hyvärinen
- **Bayesian inference**: Metropolis-Hastings, Langevin MCMC
- **Modern deep learning**: Song, Ho, Dhariwal, Nichol
- **Score-based theory**: Song, Ermon, and collaborators

## Author Notes

Created for comprehensive undergraduate computer science and mathematics education at Yonsei University. Emphasizes mathematical foundations, clear explanations, and progressive skill building from Bayesian inference through modern diffusion models.

**Happy Learning! 🎓📊🚀**

---

**Ready to Begin?**

Start with `01_score_functions_basics.py` and work through sequentially!
