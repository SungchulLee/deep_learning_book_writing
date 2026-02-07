# Chapter 25: Diffusion Models

Diffusion models learn to generate data by reversing a gradual noising process. Starting from pure Gaussian noise, a trained neural network iteratively denoises to produce structured samples—images, audio, time series, or any continuous data. This chapter develops the theory from first principles and connects it to quantitative finance applications.

## Core Idea

The framework rests on two complementary processes:

$$
\underbrace{x_0 \xrightarrow{\text{add noise}} x_1 \xrightarrow{\text{add noise}} \cdots \xrightarrow{\text{add noise}} x_T}_{\text{Forward process (fixed)}} \qquad \underbrace{x_T \xrightarrow{\text{denoise}} x_{T-1} \xrightarrow{\text{denoise}} \cdots \xrightarrow{\text{denoise}} x_0}_{\text{Reverse process (learned)}}
$$

The forward process is a fixed Markov chain that gradually corrupts data $x_0 \sim p_{\text{data}}$ into isotropic Gaussian noise $x_T \sim \mathcal{N}(0, I)$ over $T$ steps. The reverse process is parameterised by a neural network $\epsilon_\theta(x_t, t)$ that learns to predict and remove the noise added at each step. Generation proceeds by sampling $x_T \sim \mathcal{N}(0, I)$ and running the learned reverse chain.

The training objective reduces to a regression problem: given a noisy sample $x_t$ at a random timestep $t$, predict the noise $\epsilon$ that was added. This simplicity—no adversarial training, no intractable partition functions—yields stable optimisation and excellent sample quality.

## Why Diffusion Models Matter

Diffusion models power state-of-the-art generative systems across domains. Their key advantages over alternative generative frameworks are:

| Property | Diffusion | GANs | VAEs | Normalizing Flows |
|----------|-----------|------|------|-------------------|
| Training stability | Regression loss, no mode collapse | Adversarial, fragile | Stable ELBO | Stable MLE |
| Sample quality | Excellent | Excellent | Often blurry | Good |
| Mode coverage | Full distribution | Can miss modes | Good | Full |
| Sampling speed | Slow (many steps) | Fast (one pass) | Fast | Fast |
| Architecture freedom | Any network | Generator + discriminator | Encoder + decoder | Must be invertible |
| Conditioning | Flexible (text, class, image) | Possible but harder | Natural | Possible |

The architectural freedom is particularly significant: because the denoising network has no invertibility or Jacobian constraints, practitioners can deploy expressive architectures—U-Nets, Transformers, or domain-specific designs—without structural compromises.

## Theoretical Perspectives

This chapter develops diffusion models from four complementary viewpoints, each providing distinct insights:

**Score-based perspective (§25.2).** The score function $\nabla_x \log p(x)$ points toward high-density regions. Learning the score at multiple noise levels enables sampling via Langevin dynamics. Denoising score matching provides a tractable training objective that avoids computing intractable Jacobians.

**Probabilistic perspective (§25.3).** DDPM frames diffusion as a latent variable model optimised via a variational lower bound. The ELBO decomposes into per-timestep KL divergences between Gaussian posteriors, yielding the simplified noise-prediction loss.

**SDE perspective (§25.5).** The continuous-time formulation $dx = f(x,t)\,dt + g(t)\,dW$ unifies discrete diffusion variants under a common framework. Anderson's time-reversal theorem shows the reverse process requires only the score function. The probability flow ODE provides a deterministic sampling alternative with exact likelihood computation.

**Practical perspective (§25.4, §25.6–25.7).** DDIM enables fast deterministic sampling without retraining. Classifier-free guidance controls the fidelity–diversity trade-off. Latent diffusion compresses the computation into a learned latent space, and Transformer-based architectures (DiT) replace U-Nets for improved scaling.

## Comparison with Normalizing Flows

Both diffusion models and normalizing flows are likelihood-based generative models, but they differ fundamentally in how they transform between data and noise.

Normalizing flows learn an **invertible mapping** $x = f_\theta(z)$ with a tractable Jacobian determinant, enabling exact likelihood computation via the change-of-variables formula $\log p(x) = \log p(z) - \log|\det J_f|$. This invertibility constraint limits the function class to coupling layers, autoregressive transforms, or residual flows with spectral normalisation.

Diffusion models learn the reverse of a **fixed noising process** through an unconstrained neural network. The backward mapping is approximate—matching the distribution $q(x_{t-1}|x_t)$ rather than inverting a specific function—but this relaxation permits far more expressive architectures. The probability flow ODE reveals that diffusion models implicitly define a continuous normalizing flow, connecting the two frameworks:

$$\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$

This ODE has the same marginal distributions as the stochastic reverse process but is fully deterministic, enabling exact likelihood computation when needed.

## Chapter Roadmap

| Section | Topic | Key Ideas |
|---------|-------|-----------|
| 25.1 | Foundations | Forward process, reverse process, training objective |
| 25.2 | Score-Based Models | Score function, score matching, denoising score matching |
| 25.3 | DDPM | Fundamentals, noise schedule, training, sampling |
| 25.4 | DDIM | Deterministic sampling, accelerated generation |
| 25.5 | SDE Framework | VP-SDE, VE-SDE, probability flow ODE |
| 25.6 | Conditional Generation | Classifier guidance, classifier-free guidance, text conditioning |
| 25.7 | Advanced Architectures | U-Net, DiT, latent diffusion |
| 25.8 | Evaluation | FID, Inception Score, CLIP Score |
| 25.9 | Finance Applications | Time series generation, scenario generation, synthetic data |

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.
2. Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS*.
3. Song, Y., et al. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR*.
4. Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR*.
5. Ho, J., & Salimans, T. (2022). "Classifier-Free Diffusion Guidance." *NeurIPS Workshop*.
