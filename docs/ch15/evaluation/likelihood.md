# Likelihood-Based Evaluation

## Overview

Likelihood-based evaluation measures how well a generative model assigns probability to real data. Unlike sample-based metrics (FID, IS), likelihood metrics provide a principled, information-theoretic assessment of model quality. This section covers Negative Log-Likelihood (NLL), Bits Per Dimension (BPD), and Perplexity.

!!! info "Learning Objectives"
    By the end of this section, you will be able to:
    
    - Understand the mathematical foundation of likelihood-based metrics
    - Implement NLL, BPD, and Perplexity computations in PyTorch
    - Interpret likelihood metrics and understand their limitations
    - Choose appropriate metrics for different generative models
    - Recognize the likelihood vs. sample quality tradeoff

## Mathematical Foundation

### Likelihood as a Measure of Fit

For a generative model $p_\theta(x)$ and data distribution $p_{\text{data}}(x)$, the **log-likelihood** measures how much probability mass the model assigns to the data:

$$
\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)]
$$

Higher likelihood indicates better fit to the data distribution.

### Connection to Cross-Entropy

The negative log-likelihood equals the cross-entropy between data and model:

$$
\text{NLL} = -\mathcal{L}(\theta) = H(p_{\text{data}}, p_\theta) = -\mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)]
$$

This decomposes as:

$$
H(p_{\text{data}}, p_\theta) = H(p_{\text{data}}) + D_{\text{KL}}(p_{\text{data}} \| p_\theta)
$$

Since $H(p_{\text{data}})$ is constant, minimizing NLL is equivalent to minimizing KL divergence.

### Connection to Optimal Compression

From information theory, the expected code length for encoding data from $p_{\text{data}}$ using a code optimized for $p_\theta$ is:

$$
\mathbb{E}[\text{code length}] = H(p_{\text{data}}, p_\theta)
$$

**Interpretation**: NLL measures how many bits are needed on average to encode real data using the model as a compression scheme.

## Negative Log-Likelihood (NLL)

### Definition

Given a dataset $\mathcal{D} = \{x_1, ..., x_N\}$:

$$
\text{NLL} = -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(x_i)
$$

**Properties:**

- Lower NLL = Better model fit
- NLL ≥ 0 for proper probability distributions
- Theoretically, NLL = 0 only if model perfectly matches data (impossible in practice)

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union


class NLLEvaluator:
    """
    Negative Log-Likelihood evaluator for generative models.
    
    NLL measures how much probability a model assigns to real data.
    Lower NLL indicates better model fit to the data distribution.
    
    Mathematical Definition:
        NLL = -E_{x~p_data}[log p_model(x)]
           = -(1/N) Σ log p_model(x_i)
    """
    
    @staticmethod
    def compute_nll(log_probs: torch.Tensor) -> float:
        """
        Compute NLL from log probabilities.
        
        Args:
            log_probs: Log probabilities for each sample [N]
                       These come from model.log_prob(x)
        
        Returns:
            NLL value (scalar, lower is better)
        """
        # NLL is negative of mean log probability
        nll = -torch.mean(log_probs)
        return nll.item()
    
    @staticmethod
    def compute_nll_with_ci(log_probs: torch.Tensor,
                           confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Compute NLL with confidence interval.
        
        Uses standard error of the mean to estimate uncertainty.
        
        Args:
            log_probs: Log probabilities [N]
            confidence: Confidence level (default 95%)
        
        Returns:
            Tuple of (NLL, lower_bound, upper_bound)
        """
        from scipy import stats
        
        n = len(log_probs)
        nll = -torch.mean(log_probs).item()
        
        # Standard error = std / sqrt(n)
        std = torch.std(log_probs).item()
        se = std / np.sqrt(n)
        
        # Z-score for confidence interval
        alpha = 1 - confidence
        z = stats.norm.ppf(1 - alpha / 2)
        
        # Confidence interval for NLL
        # Note: We negate because NLL = -mean(log_probs)
        lower = nll - z * se
        upper = nll + z * se
        
        return nll, lower, upper
    
    @staticmethod
    def evaluate_model(model,
                      test_data: torch.Tensor,
                      batch_size: int = 64) -> dict:
        """
        Evaluate a generative model using NLL.
        
        Args:
            model: Generative model with .log_prob() method
            test_data: Test data [N, ...]
            batch_size: Batch size for evaluation
        
        Returns:
            Dictionary with NLL and statistics
        """
        model.eval()
        all_log_probs = []
        
        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i+batch_size]
                log_probs = model.log_prob(batch)
                all_log_probs.append(log_probs)
        
        log_probs = torch.cat(all_log_probs)
        
        nll, lower, upper = NLLEvaluator.compute_nll_with_ci(log_probs)
        
        return {
            'nll': nll,
            'nll_lower': lower,
            'nll_upper': upper,
            'mean_log_prob': -nll,
            'n_samples': len(test_data)
        }


# Example: Gaussian model
class GaussianModel(nn.Module):
    """Simple Gaussian model for demonstration."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_sigma = nn.Parameter(torch.zeros(dim))
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability under Gaussian.
        
        log N(x|μ,σ²) = -0.5 * [(x-μ)²/σ² + log(2πσ²)]
        """
        sigma = torch.exp(self.log_sigma)
        
        # Compute log probability
        log_prob = -0.5 * (
            ((x - self.mu) / sigma) ** 2 +
            2 * self.log_sigma +
            np.log(2 * np.pi)
        )
        
        # Sum over dimensions, shape: [batch_size]
        return log_prob.sum(dim=-1)


def demonstrate_nll():
    """Demonstrate NLL computation."""
    print("=" * 70)
    print("Negative Log-Likelihood Demonstration")
    print("=" * 70)
    
    # Generate test data from N(0, 1)
    test_data = torch.randn(1000, 10)
    
    # Model 1: Correct distribution
    model_correct = GaussianModel(dim=10)
    model_correct.mu.data.fill_(0.0)
    model_correct.log_sigma.data.fill_(0.0)  # sigma = 1
    
    # Model 2: Wrong mean
    model_wrong_mean = GaussianModel(dim=10)
    model_wrong_mean.mu.data.fill_(2.0)
    model_wrong_mean.log_sigma.data.fill_(0.0)
    
    # Model 3: Wrong variance
    model_wrong_var = GaussianModel(dim=10)
    model_wrong_var.mu.data.fill_(0.0)
    model_wrong_var.log_sigma.data.fill_(1.0)  # sigma = e ≈ 2.72
    
    evaluator = NLLEvaluator()
    
    print("\nTest data: 1000 samples from N(0, I)")
    print("-" * 50)
    
    for name, model in [("Correct N(0,1)", model_correct),
                        ("Wrong mean N(2,1)", model_wrong_mean),
                        ("Wrong var N(0,e²)", model_wrong_var)]:
        results = evaluator.evaluate_model(model, test_data)
        print(f"\n{name}:")
        print(f"  NLL: {results['nll']:.4f} [{results['nll_lower']:.4f}, {results['nll_upper']:.4f}]")
    
    print("\nNote: Lower NLL = Better fit to data")


demonstrate_nll()
```

## Bits Per Dimension (BPD)

### Why Normalize?

Raw NLL values depend on data dimensionality:

- MNIST (28×28×1 = 784 dimensions): NLL ≈ 1000
- CIFAR-10 (32×32×3 = 3072 dimensions): NLL ≈ 4000
- ImageNet (256×256×3 = 196608 dimensions): NLL ≈ 300000

**BPD normalizes for fair comparison across different data sizes.**

### Definition

$$
\text{BPD} = \frac{\text{NLL}}{D \cdot \ln(2)}
$$

where:
- $D$ is the total dimensionality of the data
- $\ln(2) \approx 0.693$ converts from nats to bits

**Interpretation**: Average number of bits needed to encode one dimension.

### Typical Values for Images

| Model Type | Dataset | BPD |
|------------|---------|-----|
| Uniform (8-bit) | Any | 8.0 |
| PixelCNN++ | CIFAR-10 | ~2.9 |
| Glow | CIFAR-10 | ~3.3 |
| DDPM | CIFAR-10 | ~3.7 |
| Real images | Natural | ~1-2 (estimated) |

### PyTorch Implementation

```python
class BPDCalculator:
    """
    Bits Per Dimension calculator for normalized likelihood comparison.
    
    BPD = NLL / (D × ln(2))
    
    where D is data dimensionality and ln(2) converts nats to bits.
    
    Why BPD?
    1. Normalizes for data dimensionality
    2. Information-theoretic interpretation (bits per pixel)
    3. Enables fair comparison across datasets
    """
    
    @staticmethod
    def nll_to_bpd(nll: float, dimensions: int) -> float:
        """
        Convert NLL to BPD.
        
        Args:
            nll: Negative log-likelihood (in nats)
            dimensions: Total data dimensionality
                       (e.g., 28*28=784 for MNIST, 32*32*3=3072 for CIFAR)
        
        Returns:
            BPD value
        """
        return nll / (dimensions * np.log(2))
    
    @staticmethod
    def bpd_to_nll(bpd: float, dimensions: int) -> float:
        """
        Convert BPD back to NLL.
        
        Args:
            bpd: Bits per dimension
            dimensions: Data dimensionality
        
        Returns:
            NLL value
        """
        return bpd * dimensions * np.log(2)
    
    @staticmethod
    def compute_bpd(log_probs: torch.Tensor, dimensions: int) -> float:
        """
        Compute BPD directly from log probabilities.
        
        Args:
            log_probs: Log probabilities [N]
            dimensions: Data dimensionality
        
        Returns:
            BPD value
        """
        nll = -torch.mean(log_probs).item()
        return nll / (dimensions * np.log(2))
    
    @staticmethod
    def evaluate_image_model(model,
                            images: torch.Tensor,
                            batch_size: int = 64) -> dict:
        """
        Evaluate an image generative model with BPD.
        
        Args:
            model: Generative model with .log_prob() method
            images: Test images [N, C, H, W]
            batch_size: Batch size
        
        Returns:
            Dictionary with BPD and related metrics
        """
        # Get dimensions
        _, c, h, w = images.shape
        dimensions = c * h * w
        
        model.eval()
        all_log_probs = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                log_probs = model.log_prob(batch)
                all_log_probs.append(log_probs)
        
        log_probs = torch.cat(all_log_probs)
        
        nll = -torch.mean(log_probs).item()
        bpd = nll / (dimensions * np.log(2))
        
        return {
            'bpd': bpd,
            'nll': nll,
            'dimensions': dimensions,
            'interpretation': BPDCalculator.interpret_bpd(bpd)
        }
    
    @staticmethod
    def interpret_bpd(bpd: float) -> str:
        """
        Interpret BPD value for natural images.
        
        Args:
            bpd: Bits per dimension
        
        Returns:
            Interpretation string
        """
        if bpd > 8.0:
            return "Worse than uniform (8-bit) - model is wrong"
        elif bpd > 5.0:
            return "Poor - basic compression only"
        elif bpd > 3.5:
            return "Moderate - captures some structure"
        elif bpd > 2.5:
            return "Good - captures significant structure"
        elif bpd > 1.5:
            return "Excellent - approaching optimal compression"
        else:
            return "Outstanding - near-optimal for natural images"


def demonstrate_bpd():
    """Demonstrate BPD computation and comparison."""
    print("=" * 70)
    print("Bits Per Dimension Demonstration")
    print("=" * 70)
    
    # Simulate log probabilities for different scenarios
    # For demonstration, we'll compute what BPD values mean
    
    dimensions = {
        'MNIST': 28 * 28,
        'CIFAR-10': 32 * 32 * 3,
        'ImageNet-256': 256 * 256 * 3
    }
    
    print("\nComparison of dimensionalities:")
    print("-" * 50)
    
    for name, dim in dimensions.items():
        print(f"{name}: {dim:,} dimensions")
    
    print("\nWhat different BPD values mean:")
    print("-" * 50)
    
    bpd_values = [8.0, 5.0, 3.5, 3.0, 2.5]
    
    print(f"{'BPD':>6} | {'MNIST NLL':>12} | {'CIFAR-10 NLL':>14} | {'Interpretation'}")
    print("-" * 70)
    
    for bpd in bpd_values:
        nll_mnist = BPDCalculator.bpd_to_nll(bpd, dimensions['MNIST'])
        nll_cifar = BPDCalculator.bpd_to_nll(bpd, dimensions['CIFAR-10'])
        interp = BPDCalculator.interpret_bpd(bpd)
        print(f"{bpd:>6.1f} | {nll_mnist:>12.1f} | {nll_cifar:>14.1f} | {interp}")
    
    print("\nKey insight: BPD enables fair comparison across different image sizes!")


demonstrate_bpd()
```

## Perplexity

### Definition for Language Models

Perplexity is the standard metric for language models:

$$
\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log p(w_t | w_{<t})\right) = \exp(\text{NLL per token})
$$

where $T$ is the sequence length.

### Intuitive Interpretation

Perplexity represents the **effective vocabulary size** at each position:

- PPL = 100: Model is as uncertain as choosing from 100 equally likely words
- PPL = 10: Model has narrowed down to ~10 likely words
- PPL = 1: Model is perfectly certain (only theoretically achievable)

### Typical Values

| Model | Dataset | Perplexity |
|-------|---------|------------|
| Random | Any | Vocabulary size |
| N-gram | PTB | ~150 |
| LSTM | PTB | ~60 |
| Transformer | PTB | ~25 |
| GPT-2 | WikiText-103 | ~18 |
| GPT-3 | Various | ~15 |

### PyTorch Implementation

```python
class PerplexityCalculator:
    """
    Perplexity calculator for language models.
    
    Perplexity = exp(NLL per token)
               = exp(-1/T Σ log p(w_t | w_{<t}))
    
    Intuition: Effective vocabulary size at each position.
    Lower perplexity = More confident predictions.
    """
    
    @staticmethod
    def compute_perplexity(log_probs: torch.Tensor,
                          lengths: Optional[torch.Tensor] = None) -> float:
        """
        Compute perplexity from token log probabilities.
        
        Args:
            log_probs: Log probabilities [batch, seq_len] or [total_tokens]
            lengths: Optional sequence lengths for variable-length batches
        
        Returns:
            Perplexity value
        """
        if lengths is not None:
            # Variable length sequences
            total_log_prob = 0.0
            total_tokens = 0
            
            for i, length in enumerate(lengths):
                total_log_prob += log_probs[i, :length].sum().item()
                total_tokens += length.item()
            
            avg_nll = -total_log_prob / total_tokens
        else:
            # Fixed length or flattened
            avg_nll = -torch.mean(log_probs).item()
        
        perplexity = np.exp(avg_nll)
        return perplexity
    
    @staticmethod
    def evaluate_language_model(model,
                               input_ids: torch.Tensor,
                               attention_mask: Optional[torch.Tensor] = None,
                               batch_size: int = 16) -> dict:
        """
        Evaluate a language model with perplexity.
        
        Args:
            model: Language model with forward() returning logits
            input_ids: Token IDs [N, seq_len]
            attention_mask: Attention mask [N, seq_len]
            batch_size: Batch size
        
        Returns:
            Dictionary with perplexity and statistics
        """
        model.eval()
        total_log_prob = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(0, len(input_ids), batch_size):
                batch_ids = input_ids[i:i+batch_size]
                
                if attention_mask is not None:
                    batch_mask = attention_mask[i:i+batch_size]
                else:
                    batch_mask = torch.ones_like(batch_ids)
                
                # Forward pass
                logits = model(batch_ids)  # [batch, seq_len, vocab]
                
                # Shift for causal LM: predict next token
                shift_logits = logits[:, :-1, :]
                shift_labels = batch_ids[:, 1:]
                shift_mask = batch_mask[:, 1:]
                
                # Compute log probabilities
                log_probs = torch.log_softmax(shift_logits, dim=-1)
                
                # Gather log probs for actual tokens
                gathered = log_probs.gather(
                    dim=-1,
                    index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)
                
                # Apply mask and accumulate
                masked_log_probs = gathered * shift_mask
                total_log_prob += masked_log_probs.sum().item()
                total_tokens += shift_mask.sum().item()
        
        avg_nll = -total_log_prob / total_tokens
        perplexity = np.exp(avg_nll)
        
        return {
            'perplexity': perplexity,
            'nll_per_token': avg_nll,
            'total_tokens': total_tokens,
            'interpretation': PerplexityCalculator.interpret_perplexity(
                perplexity, vocab_size=model.config.vocab_size if hasattr(model, 'config') else 50000
            )
        }
    
    @staticmethod
    def interpret_perplexity(ppl: float, vocab_size: int = 50000) -> str:
        """
        Interpret perplexity value.
        
        Args:
            ppl: Perplexity value
            vocab_size: Vocabulary size for reference
        
        Returns:
            Interpretation string
        """
        if ppl >= vocab_size:
            return "Random baseline - model learns nothing"
        elif ppl > 200:
            return "Poor - basic patterns only"
        elif ppl > 50:
            return "Moderate - captures some language structure"
        elif ppl > 20:
            return "Good - strong language understanding"
        elif ppl > 10:
            return "Excellent - near state-of-the-art"
        else:
            return "Outstanding - highly confident predictions"


def demonstrate_perplexity():
    """Demonstrate perplexity computation."""
    print("=" * 70)
    print("Perplexity Demonstration")
    print("=" * 70)
    
    vocab_size = 10000
    seq_len = 100
    
    print(f"\nLanguage model with vocabulary size: {vocab_size}")
    print("-" * 50)
    
    # Scenario 1: Random baseline
    print("\nScenario 1: Random Baseline (uniform predictions)")
    random_log_prob = np.log(1.0 / vocab_size)
    random_ppl = np.exp(-random_log_prob)
    print(f"  Log prob per token: {random_log_prob:.4f}")
    print(f"  Perplexity: {random_ppl:.1f}")
    print(f"  Interpretation: Effectively choosing from all {vocab_size} words")
    
    # Scenario 2: Moderate model
    print("\nScenario 2: Moderate Model (~1% probability per token)")
    moderate_log_prob = np.log(0.01)
    moderate_ppl = np.exp(-moderate_log_prob)
    print(f"  Log prob per token: {moderate_log_prob:.4f}")
    print(f"  Perplexity: {moderate_ppl:.1f}")
    print(f"  Interpretation: Effectively choosing from ~{int(moderate_ppl)} words")
    
    # Scenario 3: Good model
    print("\nScenario 3: Good Model (~20% probability per token)")
    good_log_prob = np.log(0.2)
    good_ppl = np.exp(-good_log_prob)
    print(f"  Log prob per token: {good_log_prob:.4f}")
    print(f"  Perplexity: {good_ppl:.1f}")
    print(f"  Interpretation: Effectively choosing from ~{int(good_ppl)} words")
    
    print("\n" + "-" * 50)
    print("Key insight: Lower perplexity = More confident predictions")
    print("Perplexity = 'Effective vocabulary size' at each position")


demonstrate_perplexity()
```

## The Likelihood vs. Sample Quality Tradeoff

### A Critical Limitation

!!! warning "Important"
    **High likelihood does NOT guarantee good samples!**

This is one of the most important insights in generative modeling evaluation.

### Why the Tradeoff Exists

**Case 1: High Likelihood, Poor Samples**

A model can achieve high likelihood by:
- Covering all modes (including unlikely ones)
- Having high variance/uncertainty
- "Playing it safe" with blurry predictions

**Case 2: Low Likelihood, Good Samples**

A model can generate great samples while:
- Missing some modes (mode collapse)
- Being overconfident
- Ignoring rare but valid data points

### Demonstration

```python
def demonstrate_likelihood_sample_tradeoff():
    """
    Demonstrate that high likelihood doesn't mean good samples.
    """
    print("=" * 70)
    print("Likelihood vs. Sample Quality Tradeoff")
    print("=" * 70)
    
    # True distribution: bimodal
    # Mode 1: N(-3, 1), weight 0.5
    # Mode 2: N(+3, 1), weight 0.5
    
    print("\nTrue distribution: Mixture of two Gaussians")
    print("  Mode 1: N(-3, 1) with 50% weight")
    print("  Mode 2: N(+3, 1) with 50% weight")
    
    # Generate test data from true distribution
    n_samples = 1000
    test_data = np.concatenate([
        np.random.randn(n_samples // 2) - 3,
        np.random.randn(n_samples // 2) + 3
    ])
    
    # Model A: Single Gaussian (mode collapse)
    # Only captures one mode but with high precision
    print("\n" + "-" * 50)
    print("Model A: Single Gaussian N(-3, 1)")
    print("  - High quality samples (realistic)")
    print("  - LOW diversity (missing one mode)")
    
    mu_a, sigma_a = -3.0, 1.0
    log_probs_a = -0.5 * ((test_data - mu_a) / sigma_a)**2 - np.log(sigma_a) - 0.5 * np.log(2*np.pi)
    nll_a = -np.mean(log_probs_a)
    
    print(f"  NLL: {nll_a:.4f}")
    
    # Model B: Wide Gaussian (covers both modes but blurry)
    print("\n" + "-" * 50)
    print("Model B: Wide Gaussian N(0, 5)")
    print("  - LOW quality samples (blurry)")
    print("  - High coverage (includes both modes)")
    
    mu_b, sigma_b = 0.0, 5.0
    log_probs_b = -0.5 * ((test_data - mu_b) / sigma_b)**2 - np.log(sigma_b) - 0.5 * np.log(2*np.pi)
    nll_b = -np.mean(log_probs_b)
    
    print(f"  NLL: {nll_b:.4f}")
    
    # Model C: True mixture (ideal)
    print("\n" + "-" * 50)
    print("Model C: True Mixture (ideal)")
    print("  - High quality samples")
    print("  - Full coverage")
    
    # Log probability under mixture
    log_prob_mode1 = -0.5 * ((test_data + 3) ** 2) - 0.5 * np.log(2*np.pi)
    log_prob_mode2 = -0.5 * ((test_data - 3) ** 2) - 0.5 * np.log(2*np.pi)
    log_probs_c = np.logaddexp(log_prob_mode1 + np.log(0.5), log_prob_mode2 + np.log(0.5))
    nll_c = -np.mean(log_probs_c)
    
    print(f"  NLL: {nll_c:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"{'Model':<20} {'NLL':>10} {'Sample Quality':>20} {'Coverage':>15}")
    print("-" * 70)
    print(f"{'A (Mode Collapse)':<20} {nll_a:>10.4f} {'High':>20} {'Low':>15}")
    print(f"{'B (Wide/Blurry)':<20} {nll_b:>10.4f} {'Low':>20} {'High':>15}")
    print(f"{'C (True Mixture)':<20} {nll_c:>10.4f} {'High':>20} {'High':>15}")
    
    print("\n⚠️ Key Insight: Model B has BETTER likelihood than Model A,")
    print("   but Model A produces BETTER samples for mode -3!")
    print("\n→ Always combine likelihood metrics with sample-based metrics (FID, IS)")


demonstrate_likelihood_sample_tradeoff()
```

## When to Use Likelihood Metrics

### Models with Tractable Likelihood

| Model Type | Likelihood Computable? | Recommended Metric |
|------------|------------------------|-------------------|
| VAE | ELBO (lower bound) | ELBO, Reconstruction NLL |
| Normalizing Flows | Exact | NLL, BPD |
| Autoregressive | Exact | NLL, BPD, Perplexity |
| Diffusion | Approximate (via ELBO) | BPD, FID |
| GAN | **No** | FID, IS only |
| Energy-Based | Intractable | FID, other metrics |

### Practical Guidelines

1. **Use BPD for images**: Enables comparison across resolutions
2. **Use Perplexity for text**: Standard metric for language models
3. **Report confidence intervals**: Uncertainty matters
4. **Combine with sample metrics**: FID, IS, Precision/Recall

## Summary

!!! success "Key Takeaways"
    
    1. **NLL measures fit**: Lower NLL = model assigns more probability to data
    
    2. **BPD normalizes for dimensionality**: Enables fair comparison across datasets
    
    3. **Perplexity for language**: Represents "effective vocabulary size"
    
    4. **Critical limitation**: High likelihood ≠ good samples
    
    5. **Best practice**: Always combine with sample-based metrics (FID, IS)

## References

1. Theis, L., van den Oord, A., & Bethge, M. (2016). "A Note on the Evaluation of Generative Models." *ICLR*.

2. Bishop, C. M. (2006). "Pattern Recognition and Machine Learning." Springer.

3. Salimans, T., et al. (2016). "Improved Techniques for Training GANs." *NeurIPS*.

4. Kingma, D. P., & Dhariwal, P. (2018). "Glow: Generative Flow with Invertible 1×1 Convolutions." *NeurIPS*.
