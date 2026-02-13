# Diffusion Models in PyTorch - Undergraduate Tutorial

## Overview
This educational package provides a comprehensive introduction to diffusion models, designed specifically for undergraduate students learning deep learning and generative AI.

## What are Diffusion Models?
Diffusion models are a class of generative models that learn to create data by reversing a gradual noising process. They have achieved state-of-the-art results in image generation, audio synthesis, and other domains.

## Contents

### 1. Theory (`theory_guide.md`)
- Introduction to diffusion processes
- Forward and reverse diffusion
- Mathematical foundations
- Training objectives
- Sampling procedures

### 2. Code Implementations

#### `simple_diffusion.py`
A minimal implementation of a diffusion model on 2D toy data. Perfect for understanding the core concepts.

#### `mnist_diffusion.py`
A complete diffusion model trained on MNIST digits. Includes training and sampling code.

#### `unet_architecture.py`
Implementation of the U-Net architecture commonly used in diffusion models.

#### `diffusion_utils.py`
Utility functions for noise scheduling, sampling, and visualization.

### 3. Examples

#### `example_train.py`
Script to train the MNIST diffusion model

#### `example_sample.py`
Script to generate samples from a trained model

## Prerequisites

```bash
pip install torch torchvision matplotlib numpy tqdm
```

## Quick Start

1. **Understand the theory**: Read `theory_guide.md`
2. **Run simple example**: `python simple_diffusion.py`
3. **Train on MNIST**: `python example_train.py`
4. **Generate samples**: `python example_sample.py`

## Learning Path

For undergraduates new to diffusion models, we recommend:

1. Start with the theory guide to understand the concepts
2. Run the simple 2D example to see diffusion in action
3. Study the MNIST implementation
4. Experiment with different hyperparameters
5. Try implementing your own improvements

## Key Concepts

- **Forward Process**: Gradually adds noise to data
- **Reverse Process**: Learns to denoise and generate new samples
- **Noise Schedule**: Controls how noise is added over time
- **U-Net**: Neural network architecture for denoising
- **DDPM**: Denoising Diffusion Probabilistic Models
- **Score Matching**: Alternative training objective

## Further Reading

- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- "Denoising Diffusion Implicit Models" (Song et al., 2020)
- "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)

## Exercises

Try these exercises to deepen your understanding:

1. Modify the noise schedule and observe the effects
2. Implement different sampling strategies
3. Experiment with different U-Net architectures
4. Train on different datasets
5. Implement conditional generation

## License
Educational use - MIT License
