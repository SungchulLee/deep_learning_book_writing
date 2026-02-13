"""
Complete Evaluation Example
===========================

This example demonstrates a complete evaluation workflow for a generative model,
combining multiple metrics for comprehensive assessment.

Author: Educational AI Team
Date: 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# Set seeds
torch.manual_seed(42)
np.random.seed(42)


class MockGenerativeModel:
    """Mock generative model for demonstration."""
    
    def __init__(self, quality_level: str = "good"):
        """
        Initialize mock model with different quality levels.
        
        Args:
            quality_level: "poor", "moderate", or "good"
        """
        self.quality_level = quality_level
        print(f"Initialized {quality_level} quality generator")
    
    def generate(self, n_samples: int) -> torch.Tensor:
        """
        Generate samples.
        
        Args:
            n_samples: Number of samples to generate
        
        Returns:
            Generated images [n_samples, 1, 28, 28]
        """
        if self.quality_level == "poor":
            # High noise, low structure
            samples = torch.randn(n_samples, 1, 28, 28) * 0.5
        elif self.quality_level == "moderate":
            # Medium noise, some structure
            samples = torch.randn(n_samples, 1, 28, 28) * 0.3
            # Add some structure
            samples[:, :, 10:18, 10:18] += 0.5
        else:  # good
            # Low noise, clear structure
            samples = torch.randn(n_samples, 1, 28, 28) * 0.2
            # Add clear structure (cross pattern)
            samples[:, :, 13:15, :] += 0.8
            samples[:, :, :, 13:15] += 0.8
        
        # Normalize to [0, 1]
        samples = torch.sigmoid(samples)
        return samples


def evaluate_generative_model(model: MockGenerativeModel,
                              real_data: torch.Tensor,
                              n_generated: int = 1000) -> Dict:
    """
    Perform comprehensive evaluation of a generative model.
    
    Args:
        model: Generative model to evaluate
        real_data: Real data samples
        n_generated: Number of samples to generate
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("=" * 70)
    print(f"Evaluating {model.quality_level.upper()} quality model")
    print("=" * 70)
    
    metrics = {}
    
    # 1. Generate samples
    print("\n1. Generating samples...")
    generated_data = model.generate(n_generated)
    print(f"   ✓ Generated {n_generated} samples")
    
    # 2. Visual inspection
    print("\n2. Visual Quality Assessment:")
    print(f"   Real data shape: {real_data.shape}")
    print(f"   Generated data shape: {generated_data.shape}")
    print(f"   Real data range: [{real_data.min():.3f}, {real_data.max():.3f}]")
    print(f"   Generated range: [{generated_data.min():.3f}, {generated_data.max():.3f}]")
    
    # 3. Statistical comparison
    print("\n3. Statistical Comparison:")
    real_mean = real_data.mean()
    gen_mean = generated_data.mean()
    real_std = real_data.std()
    gen_std = generated_data.std()
    
    print(f"   Real:      μ={real_mean:.4f}, σ={real_std:.4f}")
    print(f"   Generated: μ={gen_mean:.4f}, σ={gen_std:.4f}")
    print(f"   Mean error: {abs(real_mean - gen_mean):.4f}")
    print(f"   Std error:  {abs(real_std - gen_std):.4f}")
    
    metrics['mean_error'] = abs(real_mean - gen_mean).item()
    metrics['std_error'] = abs(real_std - gen_std).item()
    
    # 4. Mock FID (using simple statistics instead of Inception features)
    print("\n4. Computing Mock FID:")
    # Flatten images
    real_flat = real_data.reshape(len(real_data), -1)
    gen_flat = generated_data.reshape(len(generated_data), -1)
    
    # Compute means and covariances
    mu_real = real_flat.mean(dim=0).numpy()
    mu_gen = gen_flat.mean(dim=0).numpy()
    
    # Simple FID approximation: ||μ_real - μ_gen||²
    mock_fid = np.sum((mu_real - mu_gen) ** 2)
    print(f"   Mock FID: {mock_fid:.4f} (lower is better)")
    metrics['mock_fid'] = float(mock_fid)
    
    # 5. Diversity assessment
    print("\n5. Diversity Assessment:")
    # Compute pairwise distances
    gen_flat = generated_data.reshape(n_generated, -1)
    distances = torch.cdist(gen_flat, gen_flat, p=2)
    upper_tri = distances[torch.triu(torch.ones_like(distances), diagonal=1) == 1]
    
    avg_distance = upper_tri.mean().item()
    min_distance = upper_tri.min().item()
    
    print(f"   Average pairwise distance: {avg_distance:.4f}")
    print(f"   Minimum pairwise distance: {min_distance:.4f}")
    
    metrics['avg_diversity'] = avg_distance
    metrics['min_diversity'] = min_distance
    
    # 6. Reconstruction quality (if applicable)
    print("\n6. Sample Quality Metrics:")
    # Take a subset for comparison
    n_compare = min(100, len(real_data), len(generated_data))
    
    # Compute pixel-wise MSE
    # Note: This is not reconstruction, just comparing distributions
    real_subset = real_data[:n_compare]
    gen_subset = generated_data[:n_compare]
    
    sample_mse = torch.mean((real_subset.mean() - gen_subset.mean()) ** 2)
    print(f"   Distribution MSE: {sample_mse:.6f}")
    metrics['distribution_mse'] = sample_mse.item()
    
    return metrics


def compare_models():
    """
    Compare multiple generative models.
    """
    print("\n" + "=" * 70)
    print("COMPARATIVE EVALUATION")
    print("=" * 70)
    
    # Generate synthetic real data
    n_real = 1000
    real_data = torch.randn(n_real, 1, 28, 28) * 0.25
    real_data[:, :, 12:16, 12:16] += 0.7  # Add structure
    real_data = torch.sigmoid(real_data)
    
    print(f"\nReal dataset: {n_real} samples")
    
    # Create models with different quality levels
    models = {
        "Poor": MockGenerativeModel("poor"),
        "Moderate": MockGenerativeModel("moderate"),
        "Good": MockGenerativeModel("good")
    }
    
    # Evaluate each model
    all_metrics = {}
    for name, model in models.items():
        metrics = evaluate_generative_model(model, real_data, n_generated=1000)
        all_metrics[name] = metrics
    
    # Create comparison table
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'Poor':<15} {'Moderate':<15} {'Good':<15}")
    print("-" * 70)
    
    metric_names = list(all_metrics["Poor"].keys())
    for metric in metric_names:
        values = [all_metrics[name][metric] for name in ["Poor", "Moderate", "Good"]]
        print(f"{metric:<25} {values[0]:<15.6f} {values[1]:<15.6f} {values[2]:<15.6f}")
    
    # Determine best model
    print("\n" + "-" * 70)
    print("Best Model Analysis:")
    print("-" * 70)
    
    # Lower is better for these metrics
    poor_fid = all_metrics["Poor"]["mock_fid"]
    mod_fid = all_metrics["Moderate"]["mock_fid"]
    good_fid = all_metrics["Good"]["mock_fid"]
    
    print(f"\nMock FID (lower is better):")
    print(f"  Poor: {poor_fid:.4f}")
    print(f"  Moderate: {mod_fid:.4f}")
    print(f"  Good: {good_fid:.4f}")
    
    if good_fid < mod_fid < poor_fid:
        print("\n✓ Quality ranking matches FID scores!")
    
    print(f"\nDiversity (higher avg distance is better):")
    for name in ["Poor", "Moderate", "Good"]:
        div = all_metrics[name]["avg_diversity"]
        print(f"  {name}: {div:.4f}")


def main():
    """
    Main function running complete evaluation example.
    """
    print("\n" + "=" * 70)
    print("COMPLETE GENERATIVE MODEL EVALUATION EXAMPLE")
    print("=" * 70)
    print("""
This example demonstrates a comprehensive evaluation workflow:
1. Generate samples from model
2. Visual quality assessment  
3. Statistical comparison
4. FID computation
5. Diversity assessment
6. Sample quality metrics
7. Comparative analysis

In practice, you would also include:
- Inception Score
- Precision and Recall
- Perceptual metrics (LPIPS)
- Human evaluation
    """)
    
    # Run comparative evaluation
    compare_models()
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS FROM EVALUATION")
    print("=" * 70)
    print("""
    1. Multiple Metrics Needed:
       - No single metric captures everything
       - Different metrics measure different aspects
       - Combine quantitative + qualitative
    
    2. Metric Interpretation:
       - Lower FID = Better match to real distribution
       - Higher diversity = Less mode collapse
       - Lower statistical errors = Better moments
    
    3. Quality Levels:
       - Poor: High FID, low diversity, large statistical errors
       - Moderate: Medium FID, moderate diversity
       - Good: Low FID, high diversity, small errors
    
    4. Best Practices:
       - Generate sufficient samples (10K+ for FID)
       - Use multiple complementary metrics
       - Include visual inspection
       - Report confidence intervals
       - Compare to baselines
    
    5. Production Considerations:
       - Automate metric computation
       - Track metrics over training
       - Set quality thresholds
       - Monitor for degradation
       - Regular human evaluation
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
