# Module 40: Autoencoders

## Overview
This module provides a comprehensive introduction to autoencoders, a fundamental class of neural networks for unsupervised learning and dimensionality reduction. Autoencoders learn efficient data representations by compressing input data into a lower-dimensional latent space and then reconstructing the original input.

## Learning Objectives
By completing this module, students will be able to:
- Understand the architecture and mathematical foundations of autoencoders
- Implement various types of autoencoders (vanilla, denoising, sparse, convolutional)
- Apply autoencoders to dimensionality reduction, denoising, and feature learning
- Understand the relationship between PCA and linear autoencoders
- Visualize and interpret learned latent representations
- Compare autoencoder performance across different architectures

## Prerequisites
- Module 02: Tensors
- Module 04: Gradients
- Module 07: Linear Regression (understanding of loss functions)
- Module 14: Loss Functions
- Module 15: Optimizers
- Module 19: Activation Functions
- Module 20: Feedforward Networks
- Module 23: Convolutional Neural Networks (for convolutional autoencoders)

## Mathematical Background

### Core Concept
An autoencoder consists of two parts:
- **Encoder**: f_θ: X → Z (maps input to latent representation)
- **Decoder**: g_φ: Z → X̂ (reconstructs input from latent representation)

### Objective Function
The autoencoder minimizes reconstruction error:

L(θ, φ) = (1/n) Σᵢ ||xᵢ - g_φ(f_θ(xᵢ))||²

Where:
- xᵢ is the input sample
- f_θ(xᵢ) = z is the latent representation
- g_φ(z) = x̂ is the reconstruction
- || · || is typically L2 (MSE) or L1 (MAE) norm

### Dimensionality Reduction
The latent space Z typically has lower dimension than input space X:
- Input dimension: d_x
- Latent dimension: d_z where d_z < d_x
- Compression ratio: d_x / d_z

### Connection to PCA
Linear autoencoders (with no activation functions) learn the same subspace as PCA when using MSE loss, but may not find the same basis vectors. PCA finds orthogonal principal components ordered by variance.

## Module Structure

### 01_autoencoder_basics.py (Beginner)
- Simple autoencoder architecture
- Training on MNIST dataset
- Basic visualization of reconstructions
- Latent space exploration
- **Estimated time**: 45 minutes

### 02_denoising_autoencoder.py (Beginner-Intermediate)
- Adding noise to inputs
- Learning robust representations
- Comparison with standard autoencoder
- Denoising applications
- **Estimated time**: 40 minutes

### 03_sparse_autoencoder.py (Intermediate)
- L1 regularization on latent activations
- KL divergence sparsity constraint
- Learning sparse representations
- Feature visualization
- **Estimated time**: 50 minutes

### 04_convolutional_autoencoder.py (Intermediate)
- CNN encoder and decoder architecture
- Transposed convolutions for upsampling
- Application to image reconstruction
- Comparing with fully connected autoencoders
- **Estimated time**: 55 minutes

### 05_deep_autoencoder.py (Intermediate-Advanced)
- Stacked autoencoder architecture
- Bottleneck compression
- Hierarchical feature learning
- Visualization of multiple layers
- **Estimated time**: 50 minutes

### 06_autoencoder_applications.py (Advanced)
- Anomaly detection
- Image compression
- Feature extraction for downstream tasks
- Clustering in latent space
- Transfer learning with pretrained encoders
- **Estimated time**: 60 minutes

### 07_autoencoder_analysis.py (Advanced)
- Comparing different architectures
- Capacity vs. reconstruction trade-offs
- Interpolation in latent space
- Manifold learning visualization
- Information-theoretic analysis
- **Estimated time**: 55 minutes

## Key Concepts

### 1. Architecture Components
- **Input Layer**: Accepts original data (e.g., 28×28 images)
- **Encoder**: Progressive dimensionality reduction
- **Bottleneck/Latent Space**: Compressed representation
- **Decoder**: Progressive dimensionality expansion
- **Output Layer**: Same size as input

### 2. Types of Autoencoders

#### Vanilla Autoencoder
- Basic feedforward architecture
- Direct reconstruction of inputs
- No constraints on latent space

#### Denoising Autoencoder (DAE)
- Corrupts input with noise
- Learns to reconstruct clean data
- More robust representations
- Prevents identity mapping

#### Sparse Autoencoder
- Encourages sparse activations in latent layer
- L1 penalty or KL divergence constraint
- Learns interpretable features
- Similar to sparse coding

#### Convolutional Autoencoder
- Uses convolutional layers
- Preserves spatial structure
- Better for image data
- Fewer parameters than fully connected

#### Stacked/Deep Autoencoder
- Multiple encoding/decoding layers
- Hierarchical representations
- Can be pretrained layer-by-layer
- More expressive capacity

### 3. Loss Functions
- **MSE Loss**: L = (1/n) Σ ||x - x̂||²
- **Binary Cross-Entropy**: For binary data (e.g., binarized MNIST)
- **Combined Loss**: Reconstruction + regularization terms

### 4. Applications
- **Dimensionality Reduction**: Alternative to PCA, t-SNE
- **Denoising**: Remove noise from images, audio
- **Anomaly Detection**: High reconstruction error for anomalies
- **Feature Learning**: Pretrain representations for supervised tasks
- **Data Compression**: Lossy compression via learned encoder
- **Data Generation**: Sample from learned latent space (limited)

## Datasets Used
- **MNIST**: Handwritten digits (28×28 grayscale)
- **Fashion-MNIST**: Clothing items (28×28 grayscale)
- **CIFAR-10**: Color images (32×32×3)
- **Synthetic data**: For demonstrating specific concepts

## Implementation Notes

### Common Architectures
```
# Simple Autoencoder for MNIST
Encoder: 784 → 256 → 128 → 64 (latent)
Decoder: 64 → 128 → 256 → 784

# Convolutional Autoencoder
Encoder: 28×28×1 → 14×14×32 → 7×7×64 → flatten → 64
Decoder: 64 → reshape → 7×7×64 → 14×14×32 → 28×28×1
```

### Training Tips
1. **Initialization**: Xavier/He initialization for weights
2. **Learning Rate**: 0.001-0.01 (Adam optimizer)
3. **Batch Size**: 64-256 depending on dataset
4. **Activation Functions**: ReLU for hidden, sigmoid/linear for output
5. **Early Stopping**: Monitor validation reconstruction loss
6. **Regularization**: Dropout, weight decay, or sparsity constraints

### Visualization Techniques
- Original vs. reconstructed images (side by side)
- Latent space scatter plots (2D/3D if possible)
- Interpolation between images in latent space
- Reconstruction error heatmaps
- Feature maps from encoder layers

## Comparison with Other Methods

### Autoencoder vs. PCA
| Aspect | PCA | Autoencoder |
|--------|-----|-------------|
| Linearity | Linear only | Can be nonlinear |
| Optimization | Closed-form (SVD) | Iterative (gradient descent) |
| Basis | Orthogonal | Not necessarily orthogonal |
| Scalability | O(d³) or O(nd²) | Flexible (batch training) |
| Interpretability | High | Lower |

### Autoencoder vs. VAE
| Aspect | Autoencoder | VAE |
|--------|-------------|-----|
| Latent space | Deterministic | Probabilistic |
| Generation | Limited | Natural sampling |
| Regularization | Optional (sparse, denoising) | Built-in (KL divergence) |
| Smoothness | No guarantee | Smooth by design |

## Common Pitfalls
1. **Identity mapping**: Model learns to copy input without compression
   - Solution: Ensure latent dimension < input dimension
   
2. **Poor reconstruction**: Model doesn't learn meaningful features
   - Solution: Adjust architecture depth, width, learning rate
   
3. **Overfitting**: Perfect training reconstruction, poor validation
   - Solution: Add dropout, denoising, or early stopping
   
4. **Dead neurons**: Some latent units never activate
   - Solution: Check activation functions, learning rates, initialization

## Extensions and Advanced Topics
- **Variational Autoencoders (VAE)**: Module 41
- **Adversarial Autoencoders**: Combine with GAN discriminator
- **Sequence-to-Sequence**: Autoencoders for sequential data
- **Graph Autoencoders**: For graph-structured data
- **Contractive Autoencoders**: Penalty on Jacobian of encoder

## References

### Foundational Papers
1. Hinton, G. E., & Salakhutdinov, R. R. (2006). "Reducing the dimensionality of data with neural networks." Science.
2. Vincent, P., et al. (2008). "Extracting and composing robust features with denoising autoencoders." ICML.
3. Rifai, S., et al. (2011). "Contractive auto-encoders: Explicit invariance during feature extraction." ICML.

### Textbooks
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning." MIT Press. Chapter 14.
- Bishop, C. M. (2006). "Pattern Recognition and Machine Learning." Springer. Chapter 12.

### Online Resources
- PyTorch Tutorials: https://pytorch.org/tutorials/
- Deep Learning Book: https://www.deeplearningbook.org/
- Autoencoder Tutorial: https://lilianweng.github.io/posts/2018-08-12-vae/

## Assessment
Students should be able to:
1. ✓ Implement a basic autoencoder from scratch
2. ✓ Train on image datasets and visualize results
3. ✓ Explain the difference between types of autoencoders
4. ✓ Apply autoencoders to real-world problems (denoising, compression)
5. ✓ Analyze latent space representations
6. ✓ Compare autoencoders with linear methods (PCA)

## Next Steps
- **Module 41: Variational Autoencoders** - Add probabilistic latent spaces for generation
- **Module 42: GANs** - Alternative approach to generative modeling
- **Module 53: Transfer Learning** - Use pretrained encoders for downstream tasks
