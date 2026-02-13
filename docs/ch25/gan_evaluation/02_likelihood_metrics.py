"""
Module 52.01: Likelihood-Based Metrics
======================================

This module covers the fundamental likelihood-based metrics used to evaluate
generative models: Negative Log-Likelihood (NLL), Bits Per Dimension (BPD),
and Perplexity.

Learning Objectives:
-------------------
1. Understand and implement Negative Log-Likelihood
2. Compute Bits Per Dimension for normalized comparison
3. Calculate Perplexity for language models
4. Interpret likelihood metrics correctly

Key Concepts:
------------
- Likelihood as a probability measure
- Information-theoretic interpretation
- Cross-entropy connection
- Model comparison using likelihood

Mathematical Foundation:
-----------------------
NLL = -E_{x~p_data}[log p_model(x)]
BPD = NLL / (dimensions × log(2))
Perplexity = exp(NLL per token)

Author: Educational AI Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


class NegativeLogLikelihood:
    """
    Negative Log-Likelihood (NLL) implementation and explanation.
    
    Mathematical Definition:
    -----------------------
    For a generative model p_θ(x), the NLL on a dataset D is:
    
        NLL = -1/N * Σ log p_θ(x_i)
    
    where N is the number of samples in D.
    
    Interpretation:
    --------------
    - NLL measures how well the model explains the data
    - Lower NLL = Better fit to data distribution
    - NLL = 0 would mean perfect modeling (impossible in practice)
    - NLL is the negative of the expected log-likelihood
    
    Connection to Cross-Entropy:
    ---------------------------
    NLL is equivalent to the cross-entropy between:
    - True data distribution p_data(x)
    - Model distribution p_θ(x)
    
    H(p_data, p_θ) = -E_{x~p_data}[log p_θ(x)] = NLL
    """
    
    @staticmethod
    def compute(log_probs: torch.Tensor) -> float:
        """
        Compute Negative Log-Likelihood from log probabilities.
        
        Args:
            log_probs: Log probabilities for each sample [n_samples]
        
        Returns:
            NLL as a scalar
        
        Mathematical Steps:
        ------------------
        1. Average log probabilities: (1/N) Σ log p(x_i)
        2. Negate: NLL = -(1/N) Σ log p(x_i)
        """
        # Compute mean log probability
        mean_log_prob = torch.mean(log_probs)
        
        # Negate to get NLL
        nll = -mean_log_prob
        
        return nll.item()
    
    @staticmethod
    def compute_with_variance(log_probs: torch.Tensor) -> Tuple[float, float]:
        """
        Compute NLL with confidence interval.
        
        Args:
            log_probs: Log probabilities [n_samples]
        
        Returns:
            Tuple of (NLL, standard error)
        
        Note:
        ----
        Standard error = std(log_probs) / sqrt(n_samples)
        This gives us uncertainty in our NLL estimate.
        """
        # Compute NLL
        nll = -torch.mean(log_probs).item()
        
        # Compute standard error
        # SE = σ / sqrt(N)
        std = torch.std(log_probs).item()
        n = len(log_probs)
        standard_error = std / np.sqrt(n)
        
        return nll, standard_error


class BitsPerDimension:
    """
    Bits Per Dimension (BPD) metric for normalized likelihood comparison.
    
    Mathematical Definition:
    -----------------------
        BPD = NLL / (D × log(2))
    
    where:
    - D is the dimensionality of the data
    - log(2) converts from nats to bits
    
    Why BPD?
    --------
    1. Normalizes for data dimensionality
       - 28×28 image vs 256×256 image
       - Different length sequences
    
    2. Information-theoretic interpretation
       - Average bits needed to encode one dimension
       - Lower BPD = More efficient compression
    
    3. Enables fair comparison across:
       - Different image resolutions
       - Different sequence lengths
       - Different data modalities
    
    Example Interpretation:
    ----------------------
    BPD = 3.5 means:
    - On average, need 3.5 bits to encode each pixel/dimension
    - For 8-bit images, uniform distribution gives BPD = 8.0
    - Good models achieve BPD < 4.0 for natural images
    """
    
    @staticmethod
    def compute(nll: float, dimensions: int) -> float:
        """
        Compute Bits Per Dimension from NLL.
        
        Args:
            nll: Negative log-likelihood (in nats)
            dimensions: Total number of dimensions in the data
                       (e.g., 28*28=784 for MNIST, 32*32*3=3072 for CIFAR)
        
        Returns:
            BPD value
        
        Mathematical Steps:
        ------------------
        1. NLL is in natural units (nats)
        2. Divide by dimensions to normalize
        3. Convert to bits: divide by log(2) ≈ 0.693
        """
        # Convert from nats to bits: divide by log(2)
        # Then normalize by dimensions
        bpd = nll / (dimensions * np.log(2))
        
        return bpd
    
    @staticmethod
    def compute_from_log_probs(log_probs: torch.Tensor, dimensions: int) -> float:
        """
        Compute BPD directly from log probabilities.
        
        Args:
            log_probs: Log probabilities [n_samples]
            dimensions: Data dimensionality
        
        Returns:
            BPD value
        """
        # Compute NLL first
        nll = -torch.mean(log_probs).item()
        
        # Convert to BPD
        bpd = nll / (dimensions * np.log(2))
        
        return bpd
    
    @staticmethod
    def interpret_bpd(bpd: float, data_type: str = "image") -> str:
        """
        Provide interpretation of BPD value.
        
        Args:
            bpd: Bits per dimension value
            data_type: Type of data (image, text, etc.)
        
        Returns:
            Interpretation string
        """
        if data_type == "image":
            if bpd > 8.0:
                return "Very Poor (worse than random)"
            elif bpd > 5.0:
                return "Poor"
            elif bpd > 3.5:
                return "Moderate"
            elif bpd > 2.0:
                return "Good"
            else:
                return "Excellent"
        else:
            return f"BPD = {bpd:.3f}"


class Perplexity:
    """
    Perplexity metric for language models.
    
    Mathematical Definition:
    -----------------------
        Perplexity = exp(NLL per token)
                   = exp(-1/N Σ log p(x_i))
    
    Intuitive Meaning:
    -----------------
    Perplexity represents the "effective vocabulary size" or the average
    number of tokens the model is uncertain about at each position.
    
    Examples:
    --------
    - Perplexity = 100: Model is as confused as if choosing randomly
                        from 100 equally likely tokens
    - Perplexity = 10:  Model has narrowed down to ~10 likely tokens
    - Perplexity = 1:   Model is perfectly certain (only in theory)
    
    Connection to Cross-Entropy:
    ---------------------------
    Perplexity = 2^(cross-entropy in bits)
    
    Lower perplexity = Better language model
    """
    
    @staticmethod
    def compute(log_probs: torch.Tensor) -> float:
        """
        Compute perplexity from log probabilities.
        
        Args:
            log_probs: Log probabilities per token [n_tokens]
        
        Returns:
            Perplexity value
        
        Mathematical Steps:
        ------------------
        1. Compute mean log probability: (1/N) Σ log p(x_i)
        2. Negate to get NLL: -(1/N) Σ log p(x_i)
        3. Exponentiate: exp(NLL)
        """
        # Compute mean log probability
        mean_log_prob = torch.mean(log_probs)
        
        # Exponentiate the negative: exp(-mean_log_prob)
        perplexity = torch.exp(-mean_log_prob)
        
        return perplexity.item()
    
    @staticmethod
    def compute_per_token(log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token perplexity (useful for analysis).
        
        Args:
            log_probs: Log probabilities [n_tokens]
        
        Returns:
            Per-token perplexities [n_tokens]
        """
        # Per-token: exp(-log_prob)
        per_token_perplexity = torch.exp(-log_probs)
        
        return per_token_perplexity
    
    @staticmethod
    def interpret_perplexity(perplexity: float, vocab_size: int) -> str:
        """
        Interpret perplexity value.
        
        Args:
            perplexity: Perplexity value
            vocab_size: Size of vocabulary
        
        Returns:
            Interpretation string
        """
        # Compare to random baseline
        random_perplexity = vocab_size
        
        if perplexity >= random_perplexity * 0.9:
            quality = "Very Poor (nearly random)"
        elif perplexity >= vocab_size * 0.5:
            quality = "Poor"
        elif perplexity >= vocab_size * 0.2:
            quality = "Moderate"
        elif perplexity >= vocab_size * 0.05:
            quality = "Good"
        else:
            quality = "Excellent"
        
        return f"{quality} (Random baseline: {random_perplexity})"


def demonstrate_nll_computation():
    """
    Demonstrates NLL computation with concrete examples.
    """
    print("=" * 70)
    print("Negative Log-Likelihood (NLL) Demonstration")
    print("=" * 70)
    
    # Example 1: Perfect model (impossible in practice)
    print("\nExample 1: Perfect Model")
    print("-" * 70)
    # If model assigns probability 1.0 to all test samples
    # log(1.0) = 0.0, so NLL = 0.0
    perfect_log_probs = torch.zeros(100)
    nll_perfect = NegativeLogLikelihood.compute(perfect_log_probs)
    print(f"Log probabilities: all 0.0 (prob = 1.0)")
    print(f"NLL: {nll_perfect:.6f}")
    print("Interpretation: Model assigns probability 1 to all samples")
    print("               (Only possible if model memorizes all data)")
    
    # Example 2: Good model
    print("\n Example 2: Good Model")
    print("-" * 70)
    # Log probabilities around -2.0 (prob ≈ 0.135)
    good_log_probs = torch.randn(100) * 0.5 - 2.0
    nll_good, se_good = NegativeLogLikelihood.compute_with_variance(good_log_probs)
    print(f"Mean log probability: {good_log_probs.mean():.4f}")
    print(f"NLL: {nll_good:.4f} ± {se_good:.4f}")
    print(f"Interpretation: Model assigns average probability {np.exp(-nll_good):.4f}")
    
    # Example 3: Poor model
    print("\nExample 3: Poor Model")
    print("-" * 70)
    # Log probabilities around -10.0 (prob ≈ 0.000045)
    poor_log_probs = torch.randn(100) * 1.0 - 10.0
    nll_poor = NegativeLogLikelihood.compute(poor_log_probs)
    print(f"Mean log probability: {poor_log_probs.mean():.4f}")
    print(f"NLL: {nll_poor:.4f}")
    print(f"Interpretation: Model assigns average probability {np.exp(-nll_poor):.6f}")
    print("               (Very low probability = poor model)")
    
    # Comparison
    print("\n" + "=" * 70)
    print("Model Comparison:")
    print("=" * 70)
    print(f"{'Model':<20} {'NLL':<15} {'Avg Probability'}")
    print("-" * 70)
    print(f"{'Perfect':<20} {nll_perfect:<15.4f} {np.exp(-nll_perfect):.6f}")
    print(f"{'Good':<20} {nll_good:<15.4f} {np.exp(-nll_good):.6f}")
    print(f"{'Poor':<20} {nll_poor:<15.4f} {np.exp(-nll_poor):.6f}")
    print("\nLower NLL = Better model")


def demonstrate_bpd_computation():
    """
    Demonstrates BPD computation and comparison across different data dimensions.
    """
    print("\n" + "=" * 70)
    print("Bits Per Dimension (BPD) Demonstration")
    print("=" * 70)
    
    # Scenario: Compare models on different image sizes
    # Model A: MNIST (28×28 = 784 dimensions)
    # Model B: CIFAR (32×32×3 = 3072 dimensions)
    
    # Both models achieve similar NLL per sample
    nll_mnist = 100.0  # arbitrary units
    nll_cifar = 380.0  # arbitrary units
    
    dim_mnist = 28 * 28
    dim_cifar = 32 * 32 * 3
    
    print(f"\nModel A (MNIST):")
    print(f"  Dimensions: {dim_mnist}")
    print(f"  NLL: {nll_mnist:.2f}")
    
    print(f"\nModel B (CIFAR-10):")
    print(f"  Dimensions: {dim_cifar}")
    print(f"  NLL: {nll_cifar:.2f}")
    
    print("\n" + "-" * 70)
    print("Problem: Cannot directly compare NLL across different dimensions!")
    print("-" * 70)
    
    # Compute BPD for fair comparison
    bpd_mnist = BitsPerDimension.compute(nll_mnist, dim_mnist)
    bpd_cifar = BitsPerDimension.compute(nll_cifar, dim_cifar)
    
    print(f"\nSolution: Normalize using BPD")
    print("-" * 70)
    print(f"Model A (MNIST):")
    print(f"  BPD: {bpd_mnist:.4f}")
    print(f"  Quality: {BitsPerDimension.interpret_bpd(bpd_mnist, 'image')}")
    
    print(f"\nModel B (CIFAR-10):")
    print(f"  BPD: {bpd_cifar:.4f}")
    print(f"  Quality: {BitsPerDimension.interpret_bpd(bpd_cifar, 'image')}")
    
    # Information-theoretic interpretation
    print("\n" + "=" * 70)
    print("Information-Theoretic Interpretation:")
    print("=" * 70)
    print(f"\nFor 8-bit images, uniform distribution gives BPD = 8.0")
    print(f"(Each pixel can be one of 256 values, requiring 8 bits)")
    print(f"\nModel A achieves {bpd_mnist:.2f} BPD:")
    print(f"  Compression: {(1 - bpd_mnist/8.0)*100:.1f}% compared to uniform")
    print(f"\nModel B achieves {bpd_cifar:.2f} BPD:")
    print(f"  Compression: {(1 - bpd_cifar/8.0)*100:.1f}% compared to uniform")


def demonstrate_perplexity_computation():
    """
    Demonstrates perplexity computation for language models.
    """
    print("\n" + "=" * 70)
    print("Perplexity Demonstration")
    print("=" * 70)
    
    # Scenario: Language model with vocabulary size 10000
    vocab_size = 10000
    
    print(f"\nLanguage Model with vocabulary size: {vocab_size}")
    print("-" * 70)
    
    # Model 1: Random baseline
    # Each token equally likely: p = 1/vocab_size
    # log p = log(1/vocab_size) = -log(vocab_size)
    random_log_prob = np.log(1.0 / vocab_size)
    random_log_probs = torch.full((1000,), random_log_prob)
    ppl_random = Perplexity.compute(random_log_probs)
    
    print(f"\nModel 1: Random Baseline")
    print(f"  Log prob per token: {random_log_prob:.4f}")
    print(f"  Perplexity: {ppl_random:.1f}")
    print(f"  Interpretation: {Perplexity.interpret_perplexity(ppl_random, vocab_size)}")
    
    # Model 2: Moderate model
    # Average probability ~0.01 (1% chance per token)
    moderate_log_probs = torch.randn(1000) * 0.5 + np.log(0.01)
    ppl_moderate = Perplexity.compute(moderate_log_probs)
    nll_moderate = -moderate_log_probs.mean().item()
    
    print(f"\nModel 2: Moderate Model")
    print(f"  Average log prob: {moderate_log_probs.mean():.4f}")
    print(f"  NLL: {nll_moderate:.4f}")
    print(f"  Perplexity: {ppl_moderate:.1f}")
    print(f"  Interpretation: {Perplexity.interpret_perplexity(ppl_moderate, vocab_size)}")
    
    # Model 3: Good model
    # Average probability ~0.2 (20% chance per token)
    good_log_probs = torch.randn(1000) * 0.3 + np.log(0.2)
    ppl_good = Perplexity.compute(good_log_probs)
    nll_good = -good_log_probs.mean().item()
    
    print(f"\nModel 3: Good Model")
    print(f"  Average log prob: {good_log_probs.mean():.4f}")
    print(f"  NLL: {nll_good:.4f}")
    print(f"  Perplexity: {ppl_good:.1f}")
    print(f"  Interpretation: {Perplexity.interpret_perplexity(ppl_good, vocab_size)}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("Perplexity Comparison:")
    print("=" * 70)
    print(f"{'Model':<20} {'Perplexity':<15} {'Effective Choices'}")
    print("-" * 70)
    print(f"{'Random':<20} {ppl_random:<15.1f} All {vocab_size} tokens")
    print(f"{'Moderate':<20} {ppl_moderate:<15.1f} ~{int(ppl_moderate)} likely tokens")
    print(f"{'Good':<20} {ppl_good:<15.1f} ~{int(ppl_good)} likely tokens")
    print("\nLower perplexity = Better language model")
    print("Perplexity ≈ effective vocabulary size at each position")


def visualize_likelihood_metrics():
    """
    Creates visualizations of likelihood metrics.
    """
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    # Generate synthetic log probabilities for different model qualities
    n_samples = 1000
    
    # Poor model: mean log prob = -10
    poor_log_probs = torch.randn(n_samples) * 2.0 - 10.0
    
    # Moderate model: mean log prob = -5
    moderate_log_probs = torch.randn(n_samples) * 1.5 - 5.0
    
    # Good model: mean log prob = -2
    good_log_probs = torch.randn(n_samples) * 1.0 - 2.0
    
    # Compute metrics
    models = ['Poor', 'Moderate', 'Good']
    log_probs_list = [poor_log_probs, moderate_log_probs, good_log_probs]
    
    nlls = []
    bpds = []
    ppls = []
    
    dimensions = 784  # MNIST size
    
    for log_probs in log_probs_list:
        nll = NegativeLogLikelihood.compute(log_probs)
        bpd = BitsPerDimension.compute(nll, dimensions)
        ppl = Perplexity.compute(log_probs)
        
        nlls.append(nll)
        bpds.append(bpd)
        ppls.append(ppl)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Log probability distributions
    ax = axes[0, 0]
    for i, (model, log_probs) in enumerate(zip(models, log_probs_list)):
        ax.hist(log_probs.numpy(), bins=50, alpha=0.6, label=model,
                density=True)
    ax.set_xlabel('Log Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Log Probability Distributions', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect (log p = 0)')
    
    # Plot 2: NLL comparison
    ax = axes[0, 1]
    bars = ax.bar(models, nlls, color=['#e74c3c', '#f39c12', '#2ecc71'],
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Negative Log-Likelihood', fontsize=12)
    ax.set_title('NLL Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, nll in zip(bars, nlls):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{nll:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: BPD comparison
    ax = axes[1, 0]
    bars = ax.bar(models, bpds, color=['#e74c3c', '#f39c12', '#2ecc71'],
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Bits Per Dimension', fontsize=12)
    ax.set_title('BPD Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.axhline(8.0, color='red', linestyle='--', linewidth=2, label='Random (8 bits)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, bpd in zip(bars, bpds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{bpd:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary table
    table_data = [
        ['Model', 'NLL', 'BPD', 'Perplexity'],
        ['', '', '', ''],
    ]
    
    for i, model in enumerate(models):
        table_data.append([
            model,
            f'{nlls[i]:.2f}',
            f'{bpds[i]:.3f}',
            f'{ppls[i]:.1f}'
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style the table
    for i in range(len(table_data)):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#3498db')
                cell.set_text_props(weight='bold', color='white')
            elif i == 1:  # Separator
                cell.set_facecolor('#ecf0f1')
            else:
                if table_data[i][0] == 'Good':
                    cell.set_facecolor('#d5f4e6')
                elif table_data[i][0] == 'Moderate':
                    cell.set_facecolor('#fef5e7')
                else:
                    cell.set_facecolor('#fadbd8')
    
    ax.set_title('Likelihood Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('/home/claude/likelihood_metrics.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'likelihood_metrics.png'")
    
    return fig


def main():
    """
    Main function demonstrating likelihood metrics.
    """
    print("\n" + "=" * 70)
    print("MODULE 52.02: LIKELIHOOD-BASED METRICS")
    print("=" * 70)
    
    # Demonstrate NLL
    demonstrate_nll_computation()
    
    # Demonstrate BPD
    demonstrate_bpd_computation()
    
    # Demonstrate Perplexity
    demonstrate_perplexity_computation()
    
    # Create visualizations
    visualize_likelihood_metrics()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. Negative Log-Likelihood (NLL):
       - Measures how well model assigns probability to data
       - Lower NLL = Better fit
       - Used as training loss function
       - Can compute confidence intervals with standard error
    
    2. Bits Per Dimension (BPD):
       - Normalizes NLL for fair comparison across dimensions
       - Information-theoretic interpretation
       - BPD = NLL / (dimensions × log(2))
       - Enables comparison: MNIST vs CIFAR vs ImageNet
    
    3. Perplexity:
       - Language model specific metric
       - Perplexity = exp(NLL per token)
       - Intuition: "Effective vocabulary size"
       - Lower perplexity = More confident predictions
    
    4. Metric Selection:
       - VAE, Flows: Use NLL or BPD
       - Language Models: Use Perplexity
       - Image Models: Use BPD for fair comparison
       - Always report confidence intervals
    
    5. Limitations:
       - High likelihood ≠ Good samples
       - Need complementary sample-based metrics
       - Sensitive to model capacity
       - May not correlate with human perception
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
