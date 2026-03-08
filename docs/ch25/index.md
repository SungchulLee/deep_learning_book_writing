# Chapter 25: Generative Adversarial Networks (GAN)

Generative Adversarial Networks leverage a game-theoretic framework where a generator network learns to produce realistic samples by competing against a discriminator network that distinguishes real from fake data. GANs have driven major advances in image synthesis, style transfer, and data augmentation, though their training dynamics present unique challenges. This chapter covers GAN theory, architectures from DCGAN to StyleGAN2, loss functions, training stabilization techniques, conditional generation, evaluation metrics, and applications in finance and beyond.

---

## GAN Foundations

- Introduction to Generative Adversarial Networks -- Overview of the GAN framework with generator-discriminator competition and core intuition.
- [Adversarial Training](gan_foundations/adversarial.md) -- The adversarial framework, historical context, and how competition drives both networks to improve.
- Minimax Objective -- The value function formulating generative modeling as a minimax game with binary cross-entropy.
- [Nash Equilibrium](gan_foundations/nash.md) -- Game-theoretic analysis of GAN convergence and the conditions for equilibrium.

## GAN Architectures

- DCGAN -- Architectural guidelines for stable training of convolutional GANs that remain foundational today.
- Progressive GAN -- Training from low to high resolution progressively for stable high-resolution image generation.
- StyleGAN -- Style-based generation with mapping network and adaptive instance normalization for attribute control.
- StyleGAN2 -- Refinements addressing StyleGAN artifacts through weight demodulation and improved architecture.
- BigGAN -- Scaling up GANs with larger batch sizes and more parameters for state-of-the-art class-conditional generation.

## GAN Training

- Training Dynamics -- How generator and discriminator evolve during training, common failure modes, and monitoring strategies.
- Gradient Penalties -- Enforcing Lipschitz constraints by penalizing discriminator gradients for stable training.
- Spectral Normalization -- Constraining the discriminator's Lipschitz constant by normalizing weight matrices by their spectral norm.
- Two-Timescale Update Rule (TTUR) -- Using different learning rates for generator and discriminator with convergence guarantees.
- Mode Collapse -- Diagnosis and mitigation of the most common GAN failure mode where the generator produces limited variety.

## GAN Losses

- [Original GAN Loss](gan_losses/original.md) -- The original minimax game formulation with binary cross-entropy classification.
- Wasserstein Loss (WGAN) -- Replacing Jensen-Shannon divergence with the Wasserstein distance for stable training and meaningful loss values.
- WGAN-GP -- Improving WGAN by replacing weight clipping with gradient penalty for the Lipschitz constraint.
- Least Squares GAN -- Replacing cross-entropy with mean squared error for more stable gradients and higher quality samples.
- Hinge Loss -- Simple and effective loss used in Spectral Normalization GAN and BigGAN.

## Conditional GAN

- Class-Conditional GAN -- Extending GANs to generate data conditioned on class labels or other attributes.
- Pix2Pix -- Conditional GAN for paired image-to-image translation using U-Net generator and PatchGAN discriminator.
- CycleGAN -- Unpaired image-to-image translation using cycle consistency loss.
- SPADE -- Spatially-adaptive normalization for high-quality image synthesis from semantic segmentation maps.

## GAN Evaluation

- Generative Model Evaluation Overview -- Comprehensive guide to evaluating generative models covering likelihood-based and perceptual metrics.
- [Frechet Inception Distance (FID)](gan_evaluation/fid.md) -- The most widely adopted metric measuring distribution distance in Inception feature space.
- [Inception Score (IS)](gan_evaluation/inception_score.md) -- Scalar metric capturing both quality and diversity of generated images.
- [Precision and Recall](gan_evaluation/precision_recall.md) -- Separately measuring fidelity and diversity to diagnose mode collapse and quality issues.
- Usage Guide -- Practical guide for running evaluation code and examples.
- Quick Reference -- Cheat sheet of evaluation metrics with formulas, ranges, and use cases.

## Finance Applications

- Market Data Generation -- GAN-based scenario generation, market simulation, and synthetic data for financial institutions.
- Tail Risk Modeling -- Modeling extreme events and tail risks where traditional parametric models fail.

## Advanced Topics

- High-Dimensional Data Synthesis -- Extending GANs beyond 2D images to audio, video, 3D shapes, and other complex data.
- GANs in Reinforcement Learning -- Connections between adversarial training and RL including imitation learning and world models.
- Self-Supervised Learning in GANs -- Auxiliary self-supervised tasks for improved training stability, sample quality, and diversity.
- Broader GAN Applications -- Applications in image super-resolution, healthcare, scientific research, and beyond.

## GAN Applications and Training

- Mode Collapse (Practical) -- Practical diagnosis and mitigation strategies for mode collapse in GAN training.

## Adversarial Robustness

- [Adversarial Attacks](adversarial_robustness/attacks.md) -- Fundamentals of adversarial attacks that manipulate models with imperceptible perturbations.
- Defense Mechanisms -- Techniques to defend neural networks against adversarial attacks including adversarial training and certified defenses.
