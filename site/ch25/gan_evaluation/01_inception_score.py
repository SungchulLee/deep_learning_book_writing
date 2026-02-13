"""
Module 52: Inception Score (IS)
================================

Implementation and explanation of Inception Score, a popular metric
for evaluating generative models, especially GANs.

Learning Objectives:
-------------------
1. Understand IS mathematical foundation
2. Implement IS from scratch
3. Interpret IS scores correctly
4. Recognize IS limitations

Key Formula:
-----------
IS = exp(E_x[KL(p(y|x) || p(y))])

where:
- p(y|x): Conditional label distribution (sharpness/quality)
- p(y): Marginal label distribution (diversity)

Author: Educational AI Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import warnings

torch.manual_seed(42)
np.random.seed(42)


class InceptionScore:
    """
    Inception Score calculator.
    
    Mathematical Foundation:
    -----------------------
    IS = exp(E_x[KL(p(y|x) || p(y))])
    
    Breaking down the components:
    
    1. p(y|x): Conditional class distribution
       - Run InceptionV3 on image x
       - Get softmax probabilities over 1000 ImageNet classes
       - Sharp distribution (confident predictions) = high quality
    
    2. p(y): Marginal class distribution  
       - Average of p(y|x) over all generated images
       - Uniform distribution = high diversity
       - Peaked distribution = mode collapse
    
    3. KL(p(y|x) || p(y)): KL divergence
       - Measures how much p(y|x) differs from p(y)
       - High KL = images have confident, diverse predictions
       - Low KL = either low quality or low diversity
    
    4. Exponentiation: Convert from log scale
       - exp(E[KL(...)]) gives final IS
       - Typical range: 1.0 to ~10.0 for ImageNet-like images
    
    Intuition:
    ---------
    Good generative model should produce images that:
    - Are clearly recognizable (high p(y|x) entropy → confident)
    - Cover many classes (uniform p(y) → diverse)
    
    IS captures both via KL divergence.
    
    Limitations:
    -----------
    1. Only works for ImageNet-like images (uses Inception classifier)
    2. Cannot detect overfitting (memorization)
    3. Ignores within-class diversity
    4. Can be fooled by generating one image per class
    5. Sensitive to Inception model choice
    """
    
    @staticmethod
    def calculate_inception_score(probs: np.ndarray,
                                  splits: int = 10) -> Tuple[float, float]:
        """
        Calculate Inception Score from class probabilities.
        
        Args:
            probs: Class probabilities [n_samples, n_classes]
                   Output of InceptionV3 softmax layer
            splits: Number of splits for computing std dev
        
        Returns:
            Tuple of (mean IS, std IS)
        
        Mathematical Steps:
        ------------------
        1. Compute p(y) = (1/N) Σ p(y|x_i) for each x_i
        2. For each split:
           a. Compute KL(p(y|x_i) || p(y)) for each sample
           b. Average: E_x[KL(...)]
           c. Exponentiate: exp(E_x[KL(...)])
        3. Return mean and std over splits
        """
        # Ensure probabilities are valid
        assert np.all(probs >= 0), "Probabilities must be non-negative"
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3), \
            "Probabilities must sum to 1"
        
        n_samples = len(probs)
        
        # Split into batches for computing std
        split_size = n_samples // splits
        scores = []
        
        for i in range(splits):
            # Get split
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < splits - 1 else n_samples
            part = probs[start_idx:end_idx]
            
            # 1. Compute marginal distribution p(y)
            # Average of conditional distributions
            # Shape: [n_classes]
            p_y = np.mean(part, axis=0)
            
            # 2. Compute KL divergence for each sample
            # KL(p(y|x) || p(y)) = Σ p(y|x) * log(p(y|x) / p(y))
            
            # Add small epsilon for numerical stability
            eps = 1e-16
            p_y = p_y + eps
            part = part + eps
            
            # Compute KL divergence
            # Shape: [split_size, n_classes]
            kl_div = part * (np.log(part) - np.log(p_y))
            
            # Sum over classes, average over samples
            # E_x[KL(p(y|x) || p(y))]
            kl_mean = np.mean(np.sum(kl_div, axis=1))
            
            # 3. Exponentiate to get IS
            is_score = np.exp(kl_mean)
            scores.append(is_score)
        
        # Return mean and std over splits
        return float(np.mean(scores)), float(np.std(scores))
    
    @staticmethod
    def interpret_is(is_score: float) -> str:
        """
        Interpret Inception Score value.
        
        Args:
            is_score: IS value
        
        Returns:
            Interpretation string
        
        Typical Ranges (ImageNet):
        -------------------------
        - IS < 2.0: Very poor
        - IS 2.0-5.0: Poor to moderate
        - IS 5.0-8.0: Good
        - IS > 8.0: Excellent
        
        Note: Real images from ImageNet achieve IS ~11.2
        """
        if is_score < 2.0:
            return "Very Poor"
        elif is_score < 5.0:
            return "Poor to Moderate"
        elif is_score < 8.0:
            return "Good"
        else:
            return "Excellent"


def demonstrate_inception_score_intuition():
    """
    Demonstrates IS intuition with toy examples.
    """
    print("=" * 70)
    print("Inception Score Intuition")
    print("=" * 70)
    
    n_samples = 1000
    n_classes = 10  # Simplified (real IS uses 1000 classes)
    
    # Scenario 1: High quality, high diversity (IDEAL)
    print("\nScenario 1: High Quality + High Diversity (IDEAL)")
    print("-" * 70)
    
    # Each image confidently belongs to one class (sharp p(y|x))
    # All classes equally represented (uniform p(y))
    probs1 = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        class_idx = i % n_classes  # Cycle through all classes
        probs1[i, class_idx] = 0.9  # Confident prediction
        probs1[i, :] += 0.1 / n_classes  # Small uniform noise
    
    # Normalize
    probs1 = probs1 / probs1.sum(axis=1, keepdims=True)
    
    is_score1, is_std1 = InceptionScore.calculate_inception_score(probs1)
    print(f"IS: {is_score1:.4f} ± {is_std1:.4f}")
    print(f"Quality: {InceptionScore.interpret_is(is_score1)}")
    print("Explanation: Confident predictions + diverse classes = High IS")
    
    # Scenario 2: Low quality (uncertain predictions)
    print("\nScenario 2: Low Quality (Uncertain Predictions)")
    print("-" * 70)
    
    # Each image has uniform distribution (not confident)
    probs2 = np.ones((n_samples, n_classes)) / n_classes
    
    is_score2, is_std2 = InceptionScore.calculate_inception_score(probs2)
    print(f"IS: {is_score2:.4f} ± {is_std2:.4f}")
    print(f"Quality: {InceptionScore.interpret_is(is_score2)}")
    print("Explanation: p(y|x) = p(y) → KL = 0 → IS = exp(0) = 1.0")
    print("Minimum possible IS = 1.0")
    
    # Scenario 3: Mode collapse (only one class)
    print("\nScenario 3: Mode Collapse (Single Class)")
    print("-" * 70)
    
    # All images classified as class 0
    probs3 = np.zeros((n_samples, n_classes))
    probs3[:, 0] = 0.95
    probs3[:, 1:] = 0.05 / (n_classes - 1)
    
    is_score3, is_std3 = InceptionScore.calculate_inception_score(probs3)
    print(f"IS: {is_score3:.4f} ± {is_std3:.4f}")
    print(f"Quality: {InceptionScore.interpret_is(is_score3)}")
    print("Explanation: Confident predictions but no diversity")
    print("p(y) peaked at one class → Low KL → Low IS")
    
    # Scenario 4: High quality but limited diversity
    print("\nScenario 4: Good Quality but Limited Diversity")
    print("-" * 70)
    
    # Only 3 out of 10 classes represented
    probs4 = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        class_idx = i % 3  # Only classes 0, 1, 2
        probs4[i, class_idx] = 0.9
        probs4[i, 3:] = 0.1 / (n_classes - 3)
    
    probs4 = probs4 / probs4.sum(axis=1, keepdims=True)
    
    is_score4, is_std4 = InceptionScore.calculate_inception_score(probs4)
    print(f"IS: {is_score4:.4f} ± {is_std4:.4f}")
    print(f"Quality: {InceptionScore.interpret_is(is_score4)}")
    print("Explanation: Confident but not covering all classes")
    
    # Summary
    print("\n" + "=" * 70)
    print("IS Comparison:")
    print("=" * 70)
    print(f"High quality + high diversity:  IS = {is_score1:.2f}")
    print(f"Low quality (uncertain):        IS = {is_score2:.2f} (minimum)")
    print(f"Mode collapse (one class):      IS = {is_score3:.2f}")
    print(f"Good quality + limited diversity: IS = {is_score4:.2f}")
    print("\nKey Insight: IS balances quality (sharpness) and diversity")


def demonstrate_is_limitations():
    """
    Demonstrates limitations of Inception Score.
    """
    print("\n" + "=" * 70)
    print("Inception Score Limitations")
    print("=" * 70)
    
    n_samples = 1000
    n_classes = 10
    
    # Limitation 1: Cannot detect memorization
    print("\nLimitation 1: Cannot Detect Memorization/Overfitting")
    print("-" * 70)
    print("A model that perfectly memorizes training images can achieve")
    print("high IS, even though it's not truly generating novel samples.")
    print("\nExample: Generating 100 real images repeatedly")
    print("         → High IS (confident + diverse)")
    print("         → But not creative/generative!")
    
    # Limitation 2: Ignores within-class diversity
    print("\nLimitation 2: Ignores Within-Class Diversity")
    print("-" * 70)
    print("IS only cares about class labels, not visual diversity.")
    print("\nExample: Generating 1000 identical cat images")
    print("         → Still get high IS if classified as 'cat'")
    print("         → But zero visual diversity!")
    
    # Create example: all images classified as same class but IS is high
    probs_same = np.zeros((n_samples, n_classes))
    probs_same[:, 0] = 0.95
    probs_same[:, 1:] = 0.05 / (n_classes - 1)
    is_same, _ = InceptionScore.calculate_inception_score(probs_same)
    
    print(f"\n1000 identical images → IS = {is_same:.2f}")
    print("This should be low but IS doesn't capture it!")
    
    # Limitation 3: Can be fooled
    print("\nLimitation 3: Can Be Fooled by Adversarial Generation")
    print("-" * 70)
    print("Strategy: Generate exactly one image per class")
    print("          → Maximum diversity (uniform p(y))")
    print("          → Confident predictions (sharp p(y|x))")
    print("          → High IS!")
    print("\nBut only 10 unique images for 1000 ImageNet classes is terrible!")
    
    # Limitation 4: Inception-specific
    print("\nLimitation 4: Tied to InceptionV3 Classifier")
    print("-" * 70)
    print("IS depends on InceptionV3's learned representations.")
    print("• Only works well for ImageNet-like natural images")
    print("• May not work for: medical images, satellite imagery,")
    print("  abstract art, non-photorealistic images")
    print("• Different classifiers give different IS values")
    
    print("\n" + "=" * 70)
    print("Recommendation: Use IS alongside other metrics!")
    print("=" * 70)
    print("• Combine with FID (detects mode collapse better)")
    print("• Add precision/recall (measures coverage)")
    print("• Include visual inspection")
    print("• Consider task-specific metrics")


def main():
    """
    Main demonstration function.
    """
    print("\n" + "=" * 70)
    print("MODULE 52: INCEPTION SCORE (IS)")
    print("=" * 70)
    
    # Demonstrate IS intuition
    demonstrate_inception_score_intuition()
    
    # Demonstrate limitations
    demonstrate_is_limitations()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. IS Formula:
       IS = exp(E_x[KL(p(y|x) || p(y))])
       - p(y|x): Conditional distribution (quality/sharpness)
       - p(y): Marginal distribution (diversity)
       - KL divergence balances both
    
    2. What IS Measures:
       - Quality: How confident are the predictions?
       - Diversity: How many classes are covered?
       - High IS = Confident predictions + Diverse samples
    
    3. Typical Values:
       - Minimum: IS = 1.0 (uniform predictions)
       - Good: IS > 5.0
       - Excellent: IS > 8.0
       - Real ImageNet: IS ≈ 11.2
    
    4. Strengths:
       - Single number metric
       - Fast to compute
       - Captures quality-diversity tradeoff
       - Widely used and understood
    
    5. Limitations:
       - Cannot detect memorization/overfitting
       - Ignores within-class diversity
       - Only works for ImageNet-like images
       - Tied to InceptionV3 classifier
       - Can be fooled by adversarial strategies
    
    6. Best Practices:
       - Use splits=10 for computing std dev
       - Report IS ± std
       - Always combine with other metrics (FID, precision/recall)
       - Include visual inspection
       - Consider task-specific evaluation
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
