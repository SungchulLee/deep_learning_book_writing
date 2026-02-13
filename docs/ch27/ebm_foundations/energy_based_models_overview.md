# Energy-Based Models (EBMs) - Comprehensive Python Tutorial

## Overview

This comprehensive tutorial package covers **Energy-Based Models (EBMs)**, a powerful framework for probabilistic modeling where the probability distribution is defined through an energy function. EBMs have connections to statistical physics, neural networks, and modern generative models.

## What are Energy-Based Models?

Energy-Based Models learn to assign low energy values to likely configurations and high energy to unlikely ones. The probability distribution is defined as:

```
p(x) = exp(-E(x)) / Z
```

where:
- `E(x)` is the energy function
- `Z` is the partition function (normalization constant)

## Module Contents

### Beginner Level

1. **01_ebm_foundations.py**
   - Basic energy function concepts
   - Partition function and probability distributions
   - Simple 1D and 2D energy landscapes
   - Visualization of energy surfaces
   - Duration: 45-60 minutes

2. **02_hopfield_networks.py**
   - Classical energy-based associative memory
   - Binary Hopfield networks
   - Energy function and update rules
   - Pattern storage and retrieval
   - Duration: 60-75 minutes

3. **03_boltzmann_machines.py**
   - Stochastic binary units
   - Boltzmann distribution
   - Gibbs sampling
   - Simple BM implementation
   - Duration: 60-75 minutes

### Intermediate Level

4. **04_restricted_boltzmann_machines.py**
   - RBM architecture (bipartite graph)
   - Contrastive Divergence (CD-k) algorithm
   - Training on MNIST
   - Feature learning and reconstruction
   - Duration: 90-120 minutes

5. **05_contrastive_divergence.py**
   - Detailed CD algorithm implementation
   - Persistent Contrastive Divergence (PCD)
   - Gibbs sampling chains
   - Convergence analysis
   - Duration: 75-90 minutes

6. **06_score_matching.py**
   - Score matching for EBM training
   - Denoising score matching
   - Comparison with maximum likelihood
   - Applications to continuous data
   - Duration: 90-120 minutes

### Advanced Level

7. **07_neural_ebms.py**
   - Deep neural networks as energy functions
   - Modern EBM architectures
   - Training with MCMC and Langevin dynamics
   - Image generation with neural EBMs
   - Duration: 120-150 minutes

8. **08_cooperative_training.py**
   - Cooperative training of generator and EBM
   - Energy-based GANs
   - Joint energy-likelihood models
   - Advanced training techniques
   - Duration: 120-150 minutes

9. **09_ebm_applications.py**
   - Out-of-distribution detection
   - Image denoising and inpainting
   - Adversarial robustness
   - Compositional generation
   - Duration: 90-120 minutes

10. **10_advanced_ebm_topics.py**
    - Connection to diffusion models
    - Flow-based EBMs
    - Latent variable EBMs
    - Modern research directions
    - Duration: 90-120 minutes

## Mathematical Prerequisites

- Probability theory and statistics
- Basic calculus and linear algebra
- Understanding of gradient descent
- Familiarity with neural networks
- MCMC sampling (helpful but not required)

## Python Dependencies

```bash
pip install torch torchvision numpy matplotlib scipy scikit-learn tqdm
```

## Key Concepts Covered

### Theoretical Foundations
- Energy functions and probability distributions
- Partition function and intractability
- Maximum likelihood estimation for EBMs
- Markov Chain Monte Carlo (MCMC) sampling

### Classical Models
- Hopfield Networks
- Boltzmann Machines
- Restricted Boltzmann Machines (RBMs)
- Deep Belief Networks

### Training Methods
- Contrastive Divergence (CD)
- Persistent Contrastive Divergence (PCD)
- Score Matching and Denoising Score Matching
- Maximum Likelihood with MCMC
- Langevin Dynamics

### Modern Approaches
- Neural Energy-Based Models
- Joint Energy-Likelihood Models
- Cooperative Training
- Connections to diffusion models

## Learning Path

### For Beginners
Start with modules 01-03 to understand the foundations:
- Energy landscapes and probability
- Classical architectures (Hopfield, Boltzmann)
- Basic sampling and inference

### For Intermediate Learners
Progress through modules 04-06:
- RBMs and contrastive divergence
- Practical training algorithms
- Score matching techniques

### For Advanced Students
Complete modules 07-10:
- Deep neural EBMs
- Modern training techniques
- Applications and research directions

## Pedagogical Features

- **Progressive Complexity**: Each module builds on previous concepts
- **Extensive Comments**: Every code section includes detailed explanations
- **Mathematical Derivations**: Key equations derived step-by-step
- **Visual Learning**: Comprehensive plotting and visualization
- **Practical Examples**: Real datasets and applications
- **Exercise Problems**: Challenge questions for deeper understanding

## Connections to Other Topics

- **Statistical Physics**: Boltzmann distribution, free energy
- **Generative Models**: GANs, VAEs, Normalizing Flows
- **Diffusion Models**: Score-based generative models
- **MCMC**: Gibbs sampling, Metropolis-Hastings, Langevin dynamics
- **Optimization**: Gradient-based learning in energy landscapes

## Applications

1. **Generative Modeling**: Sampling from learned distributions
2. **Denoising**: Removing noise while preserving structure
3. **Out-of-Distribution Detection**: Identifying anomalies
4. **Feature Learning**: Unsupervised representation learning
5. **Compositional Generation**: Combining multiple concepts
6. **Adversarial Robustness**: Robust classification

## Historical Context

Energy-Based Models have a rich history:
- **1982**: Hopfield Networks introduced
- **1985**: Boltzmann Machines by Hinton & Sejnowski
- **1986**: Restricted Boltzmann Machines
- **2006**: Deep Belief Networks spark deep learning renaissance
- **2005**: Contrastive Divergence algorithm by Hinton
- **2019**: Neural EBMs revival for image generation
- **2020+**: Connections to diffusion models and score-based methods

## Best Practices

1. **Start Simple**: Begin with toy datasets and visualizations
2. **Monitor Training**: Track energy values and sample quality
3. **MCMC Diagnostics**: Check chain mixing and convergence
4. **Regularization**: Use weight decay to prevent divergence
5. **Initialization**: Careful initialization of energy networks
6. **Negative Sampling**: Balance positive and negative examples

## Common Pitfalls

- **Mode Collapse**: MCMC chains stuck in local modes
- **Slow Mixing**: Insufficient MCMC steps for training
- **Partition Function**: Intractability in high dimensions
- **Hyperparameters**: Sensitive to learning rates and MCMC steps
- **Numerical Stability**: Exponential functions can overflow

## References

### Foundational Papers
- Hopfield (1982): Neural networks and physical systems with emergent collective computational abilities
- Hinton & Sejnowski (1986): Learning and relearning in Boltzmann machines
- Hinton (2002): Training products of experts by minimizing contrastive divergence
- Hyvärinen (2005): Estimation of non-normalized statistical models by score matching

### Modern EBMs
- Du & Mordatch (2019): Implicit generation and modeling with energy-based models
- Nijkamp et al. (2019): Learning non-convergent non-persistent short-run MCMC toward energy-based model
- Grathwohl et al. (2020): Your classifier is secretly an energy based model

### Theoretical Foundations
- LeCun et al. (2006): A tutorial on energy-based learning
- Song & Kingma (2021): How to train your energy-based models

## Course Integration

This module fits into the Deep Learning curriculum as:
- **Prerequisites**: Modules 01-20 (Foundations), 23-24 (CNNs)
- **Corequisites**: Module 44 (MCMC), Module 41 (VAE)
- **Leads to**: Module 46 (Diffusion Models), Module 49 (Score-Based Models)

## Estimated Time

- **Total Content**: 15-18 hours of material
- **Suggested Pace**: 2-3 weeks (2-3 lectures + labs)
- **Self-Study**: 3-4 weeks with exercises

## Assessment Ideas

1. Implement RBM for MNIST classification
2. Compare CD-k for different k values
3. Train neural EBM for simple image generation
4. Analyze energy landscapes for different architectures
5. Explore connections to diffusion models

## Getting Started

1. Ensure all dependencies are installed
2. Start with `01_ebm_foundations.py` for basic concepts
3. Progress sequentially through the modules
4. Run each script and examine the outputs
5. Modify hyperparameters to deepen understanding
6. Attempt the exercise problems

## Support and Feedback

For questions or issues:
- Review the inline comments in each module
- Check the mathematical derivations in docstrings
- Experiment with provided visualization tools
- Refer to cited papers for deeper understanding

---

**Author**: Comprehensive Deep Learning Curriculum  
**Target Audience**: Undergraduate/Graduate CS students  
**Last Updated**: November 2025  
**License**: Educational Use
