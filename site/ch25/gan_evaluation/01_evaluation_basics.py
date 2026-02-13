"""
Module 52.01: Evaluation Basics for Generative Models
=====================================================

This module introduces the fundamental concepts of evaluating generative models.
We explore why evaluation matters, different evaluation paradigms, and basic
likelihood computations.

Learning Objectives:
-------------------
1. Understand the challenges of evaluating generative models
2. Distinguish between likelihood and sample quality
3. Implement basic likelihood calculations
4. Recognize evaluation tradeoffs

Key Concepts:
------------
- Likelihood-based evaluation
- Sample-based evaluation
- Quality vs. diversity tradeoffs
- Overfitting in generative models

Author: Educational AI Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class EvaluationParadigms:
    """
    Demonstrates different paradigms for evaluating generative models.
    
    The three main paradigms are:
    1. Likelihood-based: How well does the model assign probability to real data?
    2. Sample-based: How good do the generated samples look?
    3. Task-based: How well do samples perform on downstream tasks?
    """
    
    @staticmethod
    def likelihood_evaluation(model, test_data: torch.Tensor) -> float:
        """
        Evaluate a generative model using likelihood.
        
        Mathematical Foundation:
        ----------------------
        For a generative model p_θ(x), we evaluate:
            L = E_{x~p_data}[log p_θ(x)]
        
        Higher likelihood means the model assigns higher probability
        to the real data distribution.
        
        Args:
            model: Generative model with a log_prob method
            test_data: Real data samples [batch_size, features]
        
        Returns:
            Average log-likelihood
        """
        # Compute log probability for each test sample
        # Shape: [batch_size]
        log_probs = model.log_prob(test_data)
        
        # Return the average log-likelihood
        # This is an empirical estimate of E_{x~p_data}[log p_θ(x)]
        avg_log_likelihood = torch.mean(log_probs).item()
        
        return avg_log_likelihood
    
    @staticmethod
    def sample_quality_evaluation(generated_samples: torch.Tensor,
                                  real_samples: torch.Tensor) -> dict:
        """
        Evaluate generated samples by comparing statistics with real data.
        
        This is a simplified version of sample-based evaluation.
        We compare:
        1. Mean and standard deviation
        2. Distribution shapes (using histogram comparison)
        
        Args:
            generated_samples: Samples from the model [n_samples, features]
            real_samples: Real data samples [n_samples, features]
        
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # 1. Compare first-order statistics (mean)
        mean_diff = torch.mean(torch.abs(
            generated_samples.mean(dim=0) - real_samples.mean(dim=0)
        ))
        metrics['mean_absolute_error'] = mean_diff.item()
        
        # 2. Compare second-order statistics (standard deviation)
        std_diff = torch.mean(torch.abs(
            generated_samples.std(dim=0) - real_samples.std(dim=0)
        ))
        metrics['std_absolute_error'] = std_diff.item()
        
        # 3. Compute correlation if applicable (for multivariate data)
        if generated_samples.shape[1] > 1:
            # Compute correlation matrices
            gen_corr = torch.corrcoef(generated_samples.T)
            real_corr = torch.corrcoef(real_samples.T)
            
            # Compare correlation structures
            corr_diff = torch.mean(torch.abs(gen_corr - real_corr))
            metrics['correlation_error'] = corr_diff.item()
        
        return metrics


class SimpleGaussianModel:
    """
    A simple Gaussian generative model for demonstration.
    
    Mathematical Definition:
    ----------------------
    p_θ(x) = N(x | μ_θ, σ²_θ)
    
    Where θ = {μ_θ, σ_θ} are learnable parameters.
    
    This serves as a pedagogical example because:
    1. We can compute exact likelihood
    2. We can generate exact samples
    3. It illustrates the likelihood vs. sample quality tradeoff
    """
    
    def __init__(self, dim: int = 1):
        """
        Initialize a multivariate Gaussian model.
        
        Args:
            dim: Dimensionality of the data
        """
        # Initialize mean and log standard deviation
        # We use log std for numerical stability and to ensure positivity
        self.mu = nn.Parameter(torch.randn(dim))
        self.log_std = nn.Parameter(torch.zeros(dim))
        
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability under the Gaussian model.
        
        Mathematical Derivation:
        ----------------------
        log N(x | μ, σ²) = -0.5 * [(x-μ)/σ]² - log(σ) - 0.5*log(2π)
        
        Args:
            x: Data points [batch_size, dim]
        
        Returns:
            Log probabilities [batch_size]
        """
        # Get standard deviation from log parameterization
        # This ensures σ > 0
        std = torch.exp(self.log_std)
        
        # Compute the squared Mahalanobis distance: [(x-μ)/σ]²
        # Shape: [batch_size, dim]
        normalized_diff = (x - self.mu) / std
        squared_distance = normalized_diff ** 2
        
        # Compute log probability components
        # -0.5 * sum of squared distances
        mahalanobis_term = -0.5 * torch.sum(squared_distance, dim=1)
        
        # -log(σ) for each dimension
        log_normalization = -torch.sum(self.log_std)
        
        # -0.5 * dim * log(2π)
        constant_term = -0.5 * x.shape[1] * np.log(2 * np.pi)
        
        # Total log probability
        log_prob = mahalanobis_term + log_normalization + constant_term
        
        return log_prob
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generate samples from the Gaussian model.
        
        Algorithm:
        ---------
        x = μ + σ * ε, where ε ~ N(0, I)
        
        Args:
            n_samples: Number of samples to generate
        
        Returns:
            Generated samples [n_samples, dim]
        """
        # Get standard deviation
        std = torch.exp(self.log_std)
        
        # Sample from standard normal
        # Shape: [n_samples, dim]
        epsilon = torch.randn(n_samples, len(self.mu))
        
        # Transform to target distribution
        # Broadcasting handles the addition
        samples = self.mu + std * epsilon
        
        return samples


def demonstrate_likelihood_sample_tradeoff():
    """
    Demonstrates the fundamental tradeoff between likelihood and sample quality.
    
    Key Insight:
    -----------
    A model can have:
    1. High likelihood but poor sample quality (memorization)
    2. Low likelihood but excellent sample quality (missing modes)
    3. Both high likelihood and good samples (ideal case)
    
    This function illustrates case (1): A model that memorizes the training data
    achieves perfect likelihood on training data but generates poor diverse samples.
    """
    print("=" * 70)
    print("Demonstrating Likelihood vs. Sample Quality Tradeoff")
    print("=" * 70)
    
    # Create a bimodal distribution (mixture of two Gaussians)
    # This represents a complex real data distribution
    n_samples = 1000
    mode1 = torch.randn(n_samples // 2, 1) - 3.0  # Left mode
    mode2 = torch.randn(n_samples // 2, 1) + 3.0  # Right mode
    real_data = torch.cat([mode1, mode2], dim=0)
    
    print(f"\nReal data: {n_samples} samples from bimodal distribution")
    print(f"Mode 1 centered at -3.0, Mode 2 centered at +3.0")
    
    # Model A: Fits only one mode (high sample quality, lower likelihood)
    model_a = SimpleGaussianModel(dim=1)
    model_a.mu.data = torch.tensor([-3.0])  # Only captures left mode
    model_a.log_std.data = torch.tensor([0.0])  # log(1.0)
    
    # Model B: Fits both modes poorly (lower sample quality, higher likelihood)
    model_b = SimpleGaussianModel(dim=1)
    model_b.mu.data = torch.tensor([0.0])  # Centered between modes
    model_b.log_std.data = torch.tensor([1.5])  # log(e^1.5) ≈ 4.48 - very wide
    
    # Evaluate likelihoods
    print("\n" + "-" * 70)
    print("Likelihood Evaluation:")
    print("-" * 70)
    
    likelihood_a = EvaluationParadigms.likelihood_evaluation(model_a, real_data)
    likelihood_b = EvaluationParadigms.likelihood_evaluation(model_b, real_data)
    
    print(f"Model A (Single Mode):  Log-Likelihood = {likelihood_a:.4f}")
    print(f"Model B (Wide Gaussian): Log-Likelihood = {likelihood_b:.4f}")
    
    if likelihood_b > likelihood_a:
        print("\n⚠️  Model B has HIGHER likelihood despite missing mode structure!")
        print("    This illustrates that likelihood alone doesn't guarantee")
        print("    good sample quality or mode coverage.")
    
    # Generate and evaluate samples
    print("\n" + "-" * 70)
    print("Sample Quality Evaluation:")
    print("-" * 70)
    
    samples_a = model_a.sample(n_samples)
    samples_b = model_b.sample(n_samples)
    
    metrics_a = EvaluationParadigms.sample_quality_evaluation(samples_a, real_data)
    metrics_b = EvaluationParadigms.sample_quality_evaluation(samples_b, real_data)
    
    print(f"\nModel A (Single Mode):")
    print(f"  Mean Error: {metrics_a['mean_absolute_error']:.4f}")
    print(f"  Std Error:  {metrics_a['std_absolute_error']:.4f}")
    
    print(f"\nModel B (Wide Gaussian):")
    print(f"  Mean Error: {metrics_b['mean_absolute_error']:.4f}")
    print(f"  Std Error:  {metrics_b['std_absolute_error']:.4f}")
    
    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("Model B achieves higher likelihood by spreading probability mass")
    print("across both modes, but its samples are less realistic because they")
    print("often fall between modes where real data doesn't exist.")
    print("\nModel A captures one mode perfectly (realistic samples) but")
    print("achieves lower likelihood because it assigns zero probability to")
    print("the other mode (mode collapse).")
    print("\nThis is why we need BOTH likelihood and sample-based evaluation!")
    print("=" * 70)
    
    return {
        'real_data': real_data,
        'samples_a': samples_a,
        'samples_b': samples_b,
        'model_a': model_a,
        'model_b': model_b
    }


def basic_likelihood_computation():
    """
    Demonstrates basic likelihood computation for different distributions.
    
    We compare:
    1. Gaussian distribution
    2. Mixture of Gaussians
    3. Uniform distribution
    
    This helps build intuition about what likelihood measures.
    """
    print("\n" + "=" * 70)
    print("Basic Likelihood Computation")
    print("=" * 70)
    
    # Generate test data: 100 samples from N(0, 1)
    test_data = torch.randn(100, 1)
    
    print(f"\nTest data: 100 samples from N(0, 1)")
    print(f"Mean: {test_data.mean():.4f}, Std: {test_data.std():.4f}")
    
    # Model 1: Correct distribution N(0, 1)
    model1 = SimpleGaussianModel(dim=1)
    model1.mu.data = torch.tensor([0.0])
    model1.log_std.data = torch.tensor([0.0])  # exp(0) = 1
    
    # Model 2: Wrong mean N(5, 1)
    model2 = SimpleGaussianModel(dim=1)
    model2.mu.data = torch.tensor([5.0])
    model2.log_std.data = torch.tensor([0.0])
    
    # Model 3: Wrong variance N(0, 5)
    model3 = SimpleGaussianModel(dim=1)
    model3.mu.data = torch.tensor([0.0])
    model3.log_std.data = torch.tensor([np.log(5.0)])
    
    # Compute log-likelihoods
    ll1 = model1.log_prob(test_data).mean().item()
    ll2 = model2.log_prob(test_data).mean().item()
    ll3 = model3.log_prob(test_data).mean().item()
    
    print("\n" + "-" * 70)
    print("Model Comparison:")
    print("-" * 70)
    print(f"Model 1 N(0, 1)  - Correct:       {ll1:.4f}")
    print(f"Model 2 N(5, 1)  - Wrong mean:    {ll2:.4f}")
    print(f"Model 3 N(0, 25) - Wrong variance: {ll3:.4f}")
    
    print("\n" + "-" * 70)
    print("Interpretation:")
    print("-" * 70)
    print("Higher log-likelihood means the model better explains the data.")
    print("Model 1 (correct distribution) achieves the highest likelihood.")
    print(f"Likelihood difference (correct vs wrong mean): {ll1 - ll2:.4f}")
    print(f"Likelihood difference (correct vs wrong var):  {ll1 - ll3:.4f}")
    
    # Convert to negative log-likelihood (NLL) - commonly used in practice
    nll1 = -ll1
    nll2 = -ll2
    nll3 = -ll3
    
    print("\n" + "-" * 70)
    print("Negative Log-Likelihood (NLL) - Lower is Better:")
    print("-" * 70)
    print(f"Model 1: {nll1:.4f}")
    print(f"Model 2: {nll2:.4f}")
    print(f"Model 3: {nll3:.4f}")
    print("\nNLL is often used as a loss function for training generative models.")


def visualize_evaluation_concepts():
    """
    Creates visualizations to illustrate evaluation concepts.
    
    This function generates plots showing:
    1. Real data vs. generated samples
    2. Likelihood contours
    3. Sample quality comparisons
    """
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    # Run the tradeoff demonstration
    results = demonstrate_likelihood_sample_tradeoff()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Real data distribution
    ax = axes[0, 0]
    ax.hist(results['real_data'].numpy(), bins=50, density=True, alpha=0.7,
            color='blue', edgecolor='black')
    ax.set_title('Real Data Distribution (Bimodal)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, 'Two clear modes\nat x=-3 and x=+3',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Model A samples (single mode)
    ax = axes[0, 1]
    ax.hist(results['samples_a'].detach().numpy(), bins=50, density=True,
            alpha=0.7, color='green', edgecolor='black', label='Model A Samples')
    ax.hist(results['real_data'].numpy(), bins=50, density=True,
            alpha=0.3, color='blue', edgecolor='black', label='Real Data')
    ax.set_title('Model A: Single Mode (Mode Collapse)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, 'Good sample quality\nbut missing one mode\n(Lower likelihood)',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Plot 3: Model B samples (wide Gaussian)
    ax = axes[1, 0]
    ax.hist(results['samples_b'].detach().numpy(), bins=50, density=True,
            alpha=0.7, color='red', edgecolor='black', label='Model B Samples')
    ax.hist(results['real_data'].numpy(), bins=50, density=True,
            alpha=0.3, color='blue', edgecolor='black', label='Real Data')
    ax.set_title('Model B: Wide Gaussian (Mode Averaging)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, 'Covers both modes\nbut poor sample quality\n(Higher likelihood)',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # Plot 4: Comparison table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create comparison table
    table_data = [
        ['Metric', 'Model A\n(Single Mode)', 'Model B\n(Wide Gaussian)'],
        ['', '', ''],
        ['Likelihood', '⭐⭐', '⭐⭐⭐'],
        ['Sample Quality', '⭐⭐⭐', '⭐'],
        ['Mode Coverage', '⭐', '⭐⭐⭐'],
        ['', '', ''],
        ['Best For:', 'Sample Quality', 'Coverage'],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Style the table
    for i in range(len(table_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif i == 1 or i == 5:  # Separator rows
                cell.set_facecolor('#E8E8E8')
            else:
                cell.set_facecolor('#F5F5F5')
    
    ax.set_title('Evaluation Comparison', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('/home/claude/evaluation_concepts.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'evaluation_concepts.png'")
    
    return fig


def main():
    """
    Main function demonstrating evaluation basics.
    """
    print("\n" + "=" * 70)
    print("MODULE 52.01: EVALUATION BASICS FOR GENERATIVE MODELS")
    print("=" * 70)
    
    # Demonstrate basic likelihood computation
    basic_likelihood_computation()
    
    # Demonstrate likelihood vs. sample quality tradeoff
    demonstrate_likelihood_sample_tradeoff()
    
    # Create visualizations
    visualize_evaluation_concepts()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. Likelihood Measures Probability Assignment:
       - Higher likelihood means better fit to data distribution
       - Can be computed exactly for some models (Gaussians, flows)
       - Used as training objective (maximize likelihood = minimize NLL)
    
    2. Sample Quality Measures Realism:
       - How good do generated samples look?
       - Requires human judgment or learned metrics
       - May not correlate with likelihood
    
    3. The Likelihood-Sample Tradeoff:
       - High likelihood doesn't guarantee good samples
       - Good samples don't guarantee high likelihood
       - Need both types of evaluation
    
    4. Different Evaluation Paradigms:
       - Likelihood-based: Exact probability computations
       - Sample-based: Visual quality and statistics
       - Task-based: Performance on downstream tasks
    
    5. No Single Perfect Metric:
       - Each metric has strengths and weaknesses
       - Use multiple complementary metrics
       - Consider the specific use case
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
