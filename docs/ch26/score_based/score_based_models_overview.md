# Score-Based Generative Models - Comprehensive Python Tutorial

## Overview

This educational package provides a complete introduction to **Score-Based Generative Models**, covering mathematical foundations, implementation techniques, and practical applications. Score-based models represent a powerful approach to generative modeling that has achieved state-of-the-art results in image synthesis, audio generation, and other domains.

## What are Score-Based Generative Models?

Score-based generative models learn to model data distributions by estimating the **score function** - the gradient of the log probability density with respect to the data:

```
score(x) = ∇_x log p(x)
```

Key advantages:
- **No normalization constant needed**: Unlike traditional probabilistic models
- **Flexible architectures**: Can use any neural network architecture
- **High-quality generation**: State-of-the-art sample quality
- **Connection to diffusion models**: Unified framework for understanding generative processes

## Mathematical Foundations

### 1. Score Function
The score function captures the direction of steepest ascent in log-probability space:
- Points toward higher probability regions
- Enables sampling via Langevin dynamics
- Can be estimated without knowing the partition function

### 2. Score Matching
Training objective to learn the score function without computing intractable partition functions:
- **Explicit Score Matching**: Requires computing Hessian (expensive)
- **Denoising Score Matching**: Practical alternative using noise perturbation
- **Sliced Score Matching**: Uses random projections for efficiency

### 3. Langevin Dynamics
Sampling algorithm that uses the score function to generate samples:
```
x_{t+1} = x_t + ε/2 * ∇_x log p(x_t) + √ε * z_t
```
where z_t ~ N(0, I)

### 4. Score-Based Diffusion Models
Modern approach combining score matching with diffusion processes:
- Forward process: Gradually add noise to data
- Reverse process: Learn to denoise using score functions
- Multi-scale: Learn scores at different noise levels

## Package Contents

### Beginner Level (01-03)
**Prerequisites**: Basic Python, NumPy, PyTorch fundamentals

1. **01_score_functions_basics.py**
   - Introduction to score functions
   - Simple 2D examples
   - Visualization of score fields
   - Analytical score computation for simple distributions

2. **02_score_matching_theory.py**
   - Explicit score matching
   - Denoising score matching derivation
   - Fisher divergence
   - Toy examples with Gaussians

3. **03_langevin_dynamics.py**
   - Langevin MCMC sampling
   - Score-based sampling
   - Convergence analysis
   - 2D sampling demonstrations

### Intermediate Level (04-06)
**Prerequisites**: Beginner modules, neural networks, calculus

4. **04_denoising_score_matching.py**
   - Practical score matching implementation
   - Noise perturbation kernels
   - Training score networks
   - 2D Swiss roll example

5. **05_sliced_score_matching.py**
   - Random projection technique
   - Computational efficiency improvements
   - Implementation and comparison
   - High-dimensional toy data

6. **06_score_networks.py**
   - Neural network architectures for scores
   - Time-dependent score networks
   - Noise conditional score networks
   - Training strategies

### Advanced Level (07-10)
**Prerequisites**: Intermediate modules, CNNs, diffusion models basics

7. **07_noise_conditional_scores.py**
   - Multi-scale score modeling
   - Noise level scheduling
   - Annealed Langevin dynamics
   - Image synthesis basics

8. **08_score_sde.py**
   - Score-based SDEs framework
   - Variance Exploding (VE) SDE
   - Variance Preserving (VP) SDE
   - Probability flow ODE

9. **09_image_generation.py**
   - Complete image generation pipeline
   - U-Net score architecture
   - Training on MNIST/CIFAR-10
   - Sampling algorithms (predictor-corrector)

10. **10_advanced_applications.py**
    - Conditional generation
    - Image inpainting
    - Super-resolution
    - Inverse problems

## Learning Path

### Week 1-2: Foundations
- Complete modules 01-03
- Understand score functions conceptually
- Implement basic Langevin sampling
- **Exercises**: Visualize scores for various distributions

### Week 3-4: Score Matching Techniques
- Complete modules 04-06
- Train your first score network
- Compare different score matching objectives
- **Project**: Train score model on 2D datasets

### Week 5-6: Multi-Scale Models
- Complete modules 07-08
- Understand noise conditioning
- Implement annealed sampling
- **Exercises**: Analyze noise schedules

### Week 7-8: Image Generation
- Complete modules 09-10
- Train full image generation model
- Implement advanced sampling
- **Final Project**: Generate high-quality images

## Key Concepts Covered

### Theoretical Foundations
- Score function definition and properties
- Fisher divergence
- Stein's identity
- Score matching objectives
- Fokker-Planck equations

### Practical Techniques
- Denoising score matching
- Sliced score matching
- Annealed Langevin dynamics
- Multi-scale noise conditioning
- Predictor-corrector sampling

### Advanced Topics
- Score-based SDEs
- Continuous-time formulation
- Probability flow ODEs
- Controllable generation
- Inverse problem solving

## Installation Requirements

```bash
pip install torch torchvision numpy matplotlib scipy tqdm
```

**Optional for advanced modules**:
```bash
pip install pytorch-fid lpips tensorboard
```

## Mathematical Prerequisites

- **Linear Algebra**: Vector calculus, gradients, Jacobians
- **Probability**: Gaussian distributions, KL divergence, expectation
- **Calculus**: Partial derivatives, chain rule, integration
- **Optimization**: Gradient descent, stochastic optimization

## PyTorch Prerequisites

- Tensor operations and autograd
- Neural network modules (nn.Module)
- Training loops and optimizers
- Basic CNN architectures

## Dataset Requirements

- **Beginner/Intermediate**: Synthetic 2D datasets (included)
- **Advanced**: MNIST (28x28, ~12MB)
- **Optional**: CIFAR-10 (32x32, ~170MB), CelebA

## Recommended Study Sequence

1. **If new to generative models**:
   - Start with VAE/GAN basics (modules 40-42 in curriculum)
   - Review diffusion models (module 46)
   - Begin score-based models (this package)

2. **If familiar with diffusion models**:
   - Quick review of modules 01-03
   - Focus on modules 04-08 (score perspective)
   - Advanced applications (09-10)

3. **If experienced with deep learning**:
   - Skim modules 01-02 for notation
   - Focus on implementation (04-10)
   - Experiment with custom architectures

## Key Papers and References

### Foundational Papers
1. **"Estimation of Non-Normalized Statistical Models by Score Matching"** (Hyvärinen, 2005)
   - Original score matching paper

2. **"Generative Modeling by Estimating Gradients of the Data Distribution"** (Song & Ermon, 2019)
   - NCSN (Noise Conditional Score Networks)

3. **"Score-Based Generative Modeling through Stochastic Differential Equations"** (Song et al., 2021)
   - Unified SDE framework

### Important Extensions
4. **"Denoising Diffusion Probabilistic Models"** (Ho et al., 2020)
   - Connection to diffusion models

5. **"Sliced Score Matching"** (Song et al., 2019)
   - Efficient score matching variant

## Relationship to Diffusion Models

Score-based models and diffusion models are deeply connected:

| Aspect | Score-Based | Diffusion |
|--------|------------|-----------|
| **Forward Process** | Add noise continuously | Discrete noise schedule |
| **What's Learned** | Score function ∇log p(x) | Noise predictor ε_θ |
| **Sampling** | Langevin dynamics | DDPM/DDIM sampling |
| **Framework** | SDE-based | Discrete steps or SDE |
| **Equivalence** | score = -ε/σ | ε = -σ * score |

**Key Insight**: Score-based models provide the continuous-time view that unifies many discrete diffusion formulations.

## Code Structure and Comments

All code files follow these conventions:

```python
"""
FILE: XX_module_name.py
DIFFICULTY: [Beginner/Intermediate/Advanced]
ESTIMATED TIME: [X hours]
PREREQUISITES: [List of prior modules]

LEARNING OBJECTIVES:
- Objective 1
- Objective 2
- ...

MATHEMATICAL BACKGROUND:
- Key equations and derivations

IMPLEMENTATION NOTES:
- Important design decisions
- Performance considerations
"""

# SECTION 1: IMPORTS AND CONFIGURATION
# Detailed comments on each import

# SECTION 2: MATHEMATICAL FOUNDATIONS
# Theory and equations as comments

# SECTION 3: IMPLEMENTATION
# Heavily commented code with inline math

# SECTION 4: EXAMPLES AND VISUALIZATION
# Demonstrations with plots

# SECTION 5: EXERCISES
# Guided practice problems
```

## Exercises and Projects

Each module includes:
- **Inline exercises**: Small modifications to understand concepts
- **End-of-module projects**: Implementations from scratch
- **Challenge problems**: Advanced extensions

Suggested projects:
1. Compare score matching objectives empirically
2. Implement custom noise schedules
3. Train score model on custom dataset
4. Explore controlled generation techniques
5. Implement accelerated sampling methods

## Common Pitfalls and Tips

### Training Tips
- **Noise schedule**: Start with geometric schedule (σ ∈ [σ_min, σ_max])
- **Learning rate**: Often needs careful tuning per noise level
- **Architecture**: U-Net with attention works well for images
- **Batch size**: Larger batches stabilize training

### Sampling Tips
- **Step size**: Follow ε ∝ σ² relationship
- **Number of steps**: More steps → better quality, slower
- **Annealing**: Start from high noise, gradually reduce
- **Corrector steps**: Add Langevin MCMC for refinement

### Debugging
- **Check score magnitudes**: Should decrease with noise level
- **Visualize samples**: At each noise level during training
- **Monitor losses**: Per noise level separately
- **Test on simple data**: 2D before moving to images

## Performance Benchmarks

Approximate training times (NVIDIA V100):

| Dataset | Model Size | Training Time | FID Score |
|---------|-----------|---------------|-----------|
| 2D Swiss Roll | 3-layer MLP | 5 minutes | N/A |
| MNIST | Small U-Net | 2-4 hours | ~10 |
| CIFAR-10 | Medium U-Net | 1-2 days | ~20-30 |
| CelebA-HQ | Large U-Net | 3-5 days | ~10-15 |

## Extensions and Future Directions

After completing this package, explore:
1. **Consistency models**: Fast sampling variants
2. **Latent score models**: Score-based in latent space
3. **Flow matching**: Alternative continuous formulation
4. **Rectified flows**: Straight trajectory models
5. **Guided diffusion**: Classifier/classifier-free guidance

## Getting Help

- **Issues with code**: Check comments and docstrings first
- **Conceptual questions**: Review mathematical background sections
- **Bug reports**: Ensure dependencies are correct versions
- **Advanced topics**: See papers in references

## Acknowledgments

This educational package synthesizes ideas from:
- Yang Song's work on score-based models
- Jonathan Ho's diffusion model contributions
- Aapo Hyvärinen's score matching theory
- The broader generative modeling community

## License

Educational use only. Code examples provided for learning purposes.

---

## Quick Start

```python
# Install dependencies
pip install torch torchvision numpy matplotlib

# Run first example
python 01_score_functions_basics.py

# Follow along with visualizations
# Read comments carefully
# Complete exercises at end of each file
```

## Contact and Contributions

This is an educational resource for undergraduate CS courses. Corrections and improvements welcome.

**Happy Learning! 🎓**
