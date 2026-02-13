"""
Module 52: Fréchet Inception Distance (FID)
==========================================

Comprehensive implementation of FID, one of the most widely used metrics
for evaluating generative models.

Learning Objectives:
-------------------
1. Understand FID mathematical foundation
2. Implement FID from scratch
3. Use pre-trained InceptionV3 for feature extraction
4. Interpret FID scores correctly

Key Formula:
-----------
FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2(Σ_real × Σ_gen)^{1/2})

Author: Educational AI Team  
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from typing import Tuple
import warnings

torch.manual_seed(42)
np.random.seed(42)


class FIDCalculator:
    """
    Fréchet Inception Distance calculator.
    
    Mathematical Foundation:
    -----------------------
    FID measures the distance between two multivariate Gaussians:
    
    Real distribution: X_real ~ N(μ_r, Σ_r)  
    Generated distribution: X_gen ~ N(μ_g, Σ_g)
    
    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})
    
    Why Fréchet Distance?
    --------------------
    1. The Fréchet distance (also called Wasserstein-2 distance) is the
       optimal transport distance for Gaussian distributions
    2. Has a closed-form solution for Gaussians
    3. Sensitive to both mean and covariance differences
    4. Lower FID = distributions are more similar
    
    Why Inception Features?
    ----------------------
    1. InceptionV3 trained on ImageNet captures semantic image features
    2. Pool3 features (2048-dim) represent high-level image content
    3. More robust than pixel-level comparisons
    4. Correlates well with human judgment
    
    Typical FID Values:
    ------------------
    - FID < 10: Excellent (near-perfect generation)
    - FID 10-50: Good quality
    - FID 50-100: Moderate quality  
    - FID > 100: Poor quality
    """
    
    @staticmethod
    def calculate_frechet_distance(mu1: np.ndarray,
                                   sigma1: np.ndarray,
                                   mu2: np.ndarray,
                                   sigma2: np.ndarray,
                                   eps: float = 1e-6) -> float:
        """
        Calculate Fréchet distance between two Gaussians.
        
        Mathematical Derivation:
        -----------------------
        For X ~ N(μ₁, Σ₁) and Y ~ N(μ₂, Σ₂), the Fréchet distance is:
        
        d²_F(X,Y) = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^{1/2})
        
        Breaking down the formula:
        1. ||μ₁ - μ₂||²: Difference in means (first moment)
        2. Tr(Σ₁ + Σ₂): Sum of variances
        3. -2Tr((Σ₁Σ₂)^{1/2}): Covariance overlap term
        
        Args:
            mu1: Mean of first distribution [d]
            sigma1: Covariance matrix of first distribution [d, d]
            mu2: Mean of second distribution [d]
            sigma2: Covariance matrix of second distribution [d, d]
            eps: Small constant for numerical stability
        
        Returns:
            Fréchet distance (scalar)
        """
        # Ensure inputs are numpy arrays
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        assert mu1.shape == mu2.shape, "Means must have same shape"
        assert sigma1.shape == sigma2.shape, "Covariances must have same shape"
        
        # 1. Compute difference in means: ||μ₁ - μ₂||²
        diff = mu1 - mu2
        mean_diff_squared = np.dot(diff, diff)
        
        # 2. Compute matrix square root: (Σ₁Σ₂)^{1/2}
        # This is the most computationally expensive step
        
        # Matrix multiplication: Σ₁ @ Σ₂
        # Shape: [d, d] @ [d, d] = [d, d]
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Handle numerical errors in matrix square root
        # Sometimes sqrtm returns complex numbers due to numerical issues
        if not np.isfinite(covmean).all():
            print(f"WARNING: FID calculation produced non-finite values.")
            print(f"Adding {eps} to diagonal of covariance matrices.")
            # Add small value to diagonal for stability
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # If imaginary component is small, take real part
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                max_imag = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component too large: {max_imag}")
            covmean = covmean.real
        
        # 3. Compute trace term: Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^{1/2})
        trace_term = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        
        # 4. Final FID = ||μ₁ - μ₂||² + Tr(...)
        fid = mean_diff_squared + trace_term
        
        return float(fid)
    
    @staticmethod
    def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance of features.
        
        Args:
            features: Feature vectors [n_samples, feature_dim]
        
        Returns:
            Tuple of (mean [feature_dim], covariance [feature_dim, feature_dim])
        
        Mathematical Notes:
        ------------------
        Mean: μ = (1/N) Σ x_i
        Covariance: Σ = (1/N) Σ (x_i - μ)(x_i - μ)ᵀ
        
        We use rowvar=False to treat each column as a variable
        """
        # Compute mean: average over samples
        # Shape: [feature_dim]
        mu = np.mean(features, axis=0)
        
        # Compute covariance matrix
        # Shape: [feature_dim, feature_dim]
        # rowvar=False: each column is a variable, each row is an observation
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
    
    @staticmethod
    def calculate_fid(real_features: np.ndarray,
                     generated_features: np.ndarray) -> float:
        """
        Calculate FID given real and generated features.
        
        Complete Pipeline:
        -----------------
        1. Extract features from InceptionV3 (done before this function)
        2. Compute statistics (μ, Σ) for real data
        3. Compute statistics (μ, Σ) for generated data
        4. Calculate Fréchet distance
        
        Args:
            real_features: Features from real images [n_real, 2048]
            generated_features: Features from generated images [n_gen, 2048]
        
        Returns:
            FID score (lower is better)
        
        Minimum Sample Sizes:
        --------------------
        - Absolute minimum: 2048 samples (= feature dimension)
        - Recommended: 10,000+ samples for stable estimates
        - More samples = more reliable FID
        """
        print(f"Computing FID with {len(real_features)} real and "
              f"{len(generated_features)} generated samples...")
        
        # Compute statistics for real data
        mu_real, sigma_real = FIDCalculator.compute_statistics(real_features)
        
        # Compute statistics for generated data  
        mu_gen, sigma_gen = FIDCalculator.compute_statistics(generated_features)
        
        # Calculate Fréchet distance
        fid = FIDCalculator.calculate_frechet_distance(
            mu_real, sigma_real, mu_gen, sigma_gen
        )
        
        print(f"✓ FID computed: {fid:.4f}")
        
        return fid


class SimpleInceptionV3Wrapper:
    """
    Simplified InceptionV3 wrapper for educational purposes.
    
    In practice, you would use:
    - torchvision.models.inception_v3(pretrained=True)
    - torch-fidelity library
    - pytorch-fid library
    
    This class demonstrates the key concepts without requiring
    actual InceptionV3 weights.
    """
    
    def __init__(self, feature_dim: int = 2048):
        """
        Initialize with a mock feature extractor.
        
        Args:
            feature_dim: Dimension of feature vectors (2048 for InceptionV3)
        """
        self.feature_dim = feature_dim
        print(f"Initialized mock InceptionV3 with {feature_dim}-dim features")
        print("NOTE: For real FID, use actual InceptionV3 pretrained on ImageNet")
    
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract features from images (mock implementation).
        
        In real implementation:
        1. Preprocess images to InceptionV3 format (299×299, normalized)
        2. Forward pass through InceptionV3
        3. Extract pool3 features (2048-dim)
        4. No gradients needed (eval mode)
        
        Args:
            images: Batch of images [batch_size, C, H, W]
        
        Returns:
            Features [batch_size, feature_dim]
        """
        batch_size = images.shape[0]
        
        # Mock features (in reality, these come from InceptionV3)
        # We create features that have realistic statistical properties
        features = torch.randn(batch_size, self.feature_dim)
        
        # Add image-dependent component (simulate actual feature extraction)
        image_stats = images.mean(dim=[1,2,3], keepdim=True)
        features = features + image_stats * 0.1
        
        return features.numpy()


def demonstrate_fid_computation():
    """
    Demonstrates FID computation with synthetic data.
    """
    print("=" * 70)
    print("Fréchet Inception Distance (FID) Demonstration")
    print("=" * 70)
    
    # Scenario 1: Identical distributions (FID should be ~0)
    print("\nScenario 1: Identical Distributions")
    print("-" * 70)
    
    mu1 = np.array([0.0, 0.0])
    sigma1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    mu2 = np.array([0.0, 0.0])
    sigma2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    fid_identical = FIDCalculator.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"Mean 1: {mu1}")
    print(f"Mean 2: {mu2}")
    print(f"FID: {fid_identical:.6f}")
    print("Interpretation: FID ≈ 0 indicates identical distributions")
    
    # Scenario 2: Different means only
    print("\nScenario 2: Different Means (Same Covariance)")
    print("-" * 70)
    
    mu1 = np.array([0.0, 0.0])
    sigma1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    mu2 = np.array([3.0, 3.0])  # Shifted mean
    sigma2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    fid_mean = FIDCalculator.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"Mean 1: {mu1}")
    print(f"Mean 2: {mu2}")
    print(f"Mean difference norm: {np.linalg.norm(mu1 - mu2):.4f}")
    print(f"FID: {fid_mean:.4f}")
    print("Interpretation: FID increases with mean difference")
    
    # Scenario 3: Different covariances
    print("\nScenario 3: Different Covariances (Same Mean)")
    print("-" * 70)
    
    mu1 = np.array([0.0, 0.0])
    sigma1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    mu2 = np.array([0.0, 0.0])
    sigma2 = np.array([[4.0, 0.0], [0.0, 4.0]])  # Larger variance
    
    fid_cov = FIDCalculator.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"Covariance 1:\n{sigma1}")
    print(f"Covariance 2:\n{sigma2}")
    print(f"FID: {fid_cov:.4f}")
    print("Interpretation: FID sensitive to variance differences")
    
    # Scenario 4: Both different
    print("\nScenario 4: Both Mean and Covariance Different")
    print("-" * 70)
    
    mu1 = np.array([0.0, 0.0])
    sigma1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    mu2 = np.array([2.0, 2.0])
    sigma2 = np.array([[3.0, 0.5], [0.5, 3.0]])  # Different variance + correlation
    
    fid_both = FIDCalculator.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"Mean difference: {np.linalg.norm(mu1 - mu2):.4f}")
    print(f"FID: {fid_both:.4f}")
    print("Interpretation: FID captures both mean and covariance differences")
    
    # Summary
    print("\n" + "=" * 70)
    print("FID Comparison Summary:")
    print("=" * 70)
    print(f"Identical distributions:     FID = {fid_identical:.4f}")
    print(f"Different means:             FID = {fid_mean:.4f}")
    print(f"Different covariances:       FID = {fid_cov:.4f}")
    print(f"Both different:              FID = {fid_both:.4f}")
    print("\nKey Insight: FID increases as distributions become more different")


def demonstrate_fid_with_features():
    """
    Demonstrates FID calculation with feature vectors.
    """
    print("\n" + "=" * 70)
    print("FID with Feature Vectors")
    print("=" * 70)
    
    # Generate synthetic feature vectors
    n_samples = 5000
    feature_dim = 2048
    
    print(f"\nGenerating {n_samples} samples with {feature_dim} features...")
    
    # Real distribution: N(0, I)
    real_features = np.random.randn(n_samples, feature_dim)
    
    # Generated distribution 1: Identical (FID should be low)
    gen1_features = np.random.randn(n_samples, feature_dim)
    
    # Generated distribution 2: Shifted mean
    gen2_features = np.random.randn(n_samples, feature_dim) + 0.5
    
    # Generated distribution 3: Reduced variance (mode collapse indicator)
    gen3_features = np.random.randn(n_samples, feature_dim) * 0.5
    
    # Compute FIDs
    print("\n" + "-" * 70)
    print("FID Comparison:")
    print("-" * 70)
    
    fid1 = FIDCalculator.calculate_fid(real_features, gen1_features)
    fid2 = FIDCalculator.calculate_fid(real_features, gen2_features)
    fid3 = FIDCalculator.calculate_fid(real_features, gen3_features)
    
    print(f"\nGenerator 1 (similar):        FID = {fid1:.2f}")
    print(f"Generator 2 (shifted):        FID = {fid2:.2f}")
    print(f"Generator 3 (mode collapse):  FID = {fid3:.2f}")
    
    print("\n" + "-" * 70)
    print("Interpretation:")
    print("-" * 70)
    print("• Lower FID = Better match to real distribution")
    print("• FID sensitive to both mean shifts and variance changes")
    print("• Reduced variance (mode collapse) increases FID")


def main():
    """
    Main demonstration function.
    """
    print("\n" + "=" * 70)
    print("MODULE 52: FRÉCHET INCEPTION DISTANCE (FID)")
    print("=" * 70)
    
    # Demonstrate basic FID computation
    demonstrate_fid_computation()
    
    # Demonstrate with feature vectors
    demonstrate_fid_with_features()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. FID Definition:
       - Measures distance between real and generated distributions
       - Assumes Gaussian distributions in feature space
       - Lower FID = Better generation quality
    
    2. Mathematical Components:
       - Mean difference: ||μ_r - μ_g||²
       - Covariance term: Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})
       - Closed-form solution for Gaussians
    
    3. Why InceptionV3?
       - Captures semantic image features
       - Pretrained on ImageNet
       - 2048-dimensional pool3 features
       - Better than pixel-space comparisons
    
    4. Sample Size Matters:
       - Minimum: 2048 samples (= feature dimension)
       - Recommended: 10,000+ samples
       - More samples = more stable FID estimates
    
    5. Limitations:
       - Assumes Gaussian distributions (may not hold)
       - Biased by choice of feature extractor
       - Cannot detect all failure modes
       - Should combine with other metrics
    
    6. Typical Values:
       - FID < 10: Excellent quality
       - FID 10-50: Good quality
       - FID 50-100: Moderate quality
       - FID > 100: Poor quality
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
