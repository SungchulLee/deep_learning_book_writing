# Diffusion Models: Theory Guide

## Introduction

Diffusion models are generative models inspired by non-equilibrium thermodynamics. They learn to generate data by gradually denoising a completely noisy signal. This guide will walk you through the key concepts.

## Intuition: The Big Picture

Imagine you have a clear photograph, and you gradually add random noise to it over many steps until it becomes pure static. A diffusion model learns to reverse this process - it starts with pure noise and gradually removes it to create new images.

## The Forward Process (Adding Noise)

### Definition
The forward process, also called the diffusion process, gradually adds Gaussian noise to data over T timesteps.

Starting with a data point x₀ (e.g., an image), we add noise at each timestep:

```
x_t = √(α̅_t) * x_0 + √(1 - α̅_t) * ε
```

Where:
- `x_t` is the noisy version at timestep t
- `α̅_t` is a coefficient that decreases from 1 to nearly 0
- `ε ~ N(0, I)` is random Gaussian noise
- T is typically 1000 timesteps

### Key Property: Closed Form
We can jump directly to any timestep t without computing all intermediate steps! This makes training efficient.

### Noise Schedule
The noise schedule β_t controls how quickly we add noise:
- β₁ = 0.0001 (start)
- β_T = 0.02 (end)
- Linear or cosine schedules are common

As t increases:
- α̅_t decreases
- x_t becomes more noisy
- At t=T, x_T is essentially pure Gaussian noise

## The Reverse Process (Removing Noise)

### Goal
Learn to reverse the diffusion process: start with noise x_T and gradually denoise it to generate x₀.

### Neural Network
We train a neural network ε_θ to predict the noise that was added at each step:

```
ε_θ(x_t, t) → predicted noise
```

The network takes:
- `x_t`: the noisy image at timestep t
- `t`: the current timestep
- Outputs: predicted noise

### Denoising Step
To go from x_t to x_(t-1), we:

1. Predict the noise: `ε̂ = ε_θ(x_t, t)`
2. Estimate the clean image: `x̂_0 = (x_t - √(1-α̅_t) * ε̂) / √(α̅_t)`
3. Sample x_(t-1) using the predicted x̂_0 and some randomness

## Training Objective

### Simple Loss Function
The training loss is surprisingly simple:

```
L = E[||ε - ε_θ(x_t, t)||²]
```

We:
1. Sample a training image x₀
2. Sample a random timestep t
3. Sample noise ε ~ N(0, I)
4. Create noisy image: x_t = √(α̅_t) * x₀ + √(1 - α̅_t) * ε
5. Train the network to predict ε

### Why This Works
By learning to predict noise at every timestep, the network learns the data distribution. The network essentially learns the "score" (gradient of log probability) of the data distribution.

## Sampling (Generation)

To generate new samples:

```python
# Start with random noise
x_T ~ N(0, I)

# Iteratively denoise
for t from T to 1:
    # Predict noise
    ε̂ = ε_θ(x_t, t)
    
    # Compute mean of x_(t-1)
    μ = compute_mean(x_t, ε̂, t)
    
    # Add randomness (except at t=1)
    if t > 1:
        z ~ N(0, I)
        x_(t-1) = μ + σ_t * z
    else:
        x_0 = μ

return x_0
```

## Mathematical Foundations

### Markov Chain
The forward process is a Markov chain:

```
q(x_t | x_(t-1)) = N(x_t; √(1-β_t) * x_(t-1), β_t * I)
```

### Reverse Process
We learn the reverse distribution:

```
p_θ(x_(t-1) | x_t) = N(x_(t-1); μ_θ(x_t, t), Σ_θ(x_t, t))
```

### Variational Lower Bound
The full objective is derived from maximizing the variational lower bound of the log-likelihood. The simplified version we use in practice is equivalent to denoising score matching.

## Key Hyperparameters

### Number of Timesteps (T)
- Typical: T = 1000
- More steps = smoother transition but slower sampling
- Can use fewer steps during sampling (DDIM)

### Noise Schedule (β_t)
- Linear: β_t increases linearly from β₁ to β_T
- Cosine: smoother schedule, often better results
- Controls how quickly we add/remove noise

### Network Architecture
- U-Net is standard for images
- Includes time embedding to condition on t
- Skip connections help preserve information

### Learning Rate
- Typical: 1e-4 to 2e-4
- Adam optimizer is standard
- EMA (exponential moving average) of weights helps

## Advantages of Diffusion Models

1. **High Quality**: State-of-the-art image generation
2. **Stable Training**: No adversarial training like GANs
3. **Flexible**: Easy to condition or guide generation
4. **Theoretically Grounded**: Solid mathematical foundation

## Disadvantages

1. **Slow Sampling**: Requires many denoising steps
2. **Computational Cost**: Training can be expensive
3. **Fixed Resolution**: Typically trained at one resolution

## Variants and Extensions

### DDIM (Denoising Diffusion Implicit Models)
- Deterministic sampling
- Can skip timesteps for faster generation
- 10-50 steps instead of 1000

### Conditional Generation
- Add class labels or text embeddings
- Classifier guidance or classifier-free guidance
- Powers models like DALL-E 2, Stable Diffusion

### Latent Diffusion
- Work in a compressed latent space
- Much faster than pixel-space diffusion
- Used in Stable Diffusion

### Score-Based Models
- Alternative view: learn the score function
- Continuous time formulation
- Stochastic differential equations (SDEs)

## Practical Tips for Implementation

1. **Start Simple**: Begin with toy datasets (2D data, MNIST)
2. **Noise Schedule**: Use cosine schedule for better results
3. **Time Embedding**: Sinusoidal positional encoding works well
4. **EMA**: Keep exponential moving average of model weights
5. **Validation**: Visualize samples during training
6. **Batch Size**: Larger batches help with stability

## Comparison to Other Generative Models

### vs GANs
- More stable training
- No mode collapse
- Slower sampling
- Better sample quality

### vs VAEs
- Better sample quality
- No posterior collapse
- More expensive to sample
- Different latent structure

### vs Autoregressive Models
- Parallel generation of all pixels/tokens
- More flexible ordering
- Requires many steps
- Different inductive biases

## Conclusion

Diffusion models represent a powerful and theoretically grounded approach to generative modeling. By learning to reverse a gradual noising process, they can generate high-quality samples while avoiding many pitfalls of other approaches.

The key insight is simple: if you can add noise to data, you can learn to remove it, and noise removal is the same as generation!

## Further Study

1. Implement the code examples in this package
2. Experiment with different architectures
3. Read the original DDPM paper (Ho et al., 2020)
4. Explore conditional generation
5. Study recent advances like DDIM and latent diffusion
