"""
Module 62.5: Certified Defenses - Advanced Level

This module implements certified robustness defenses, which provide provable
guarantees about model predictions. Unlike empirical defenses (like adversarial
training) that are tested against specific attacks, certified defenses offer
mathematical guarantees that hold for ALL perturbations within a given radius.

MATHEMATICAL BACKGROUND:
=======================

Empirical vs. Certified Robustness:
-----------------------------------
**Empirical robustness**: 
- Test against known attacks (FGSM, PGD, C&W)
- No guarantees against unknown attacks
- May suffer from gradient masking

**Certified robustness**:
- Mathematical proof that prediction is robust
- Guarantees hold for ALL perturbations in ε-ball
- Cannot be fooled by any attack within certified radius

RANDOMIZED SMOOTHING:
====================
Randomized smoothing creates a provably robust classifier by smoothing the
original classifier with Gaussian noise.

Construction:
-------------
Given a base classifier f: R^d → {1,...,k}, construct smoothed classifier g:

    g(x) = argmax_c P(f(x + ε) = c)  where ε ~ N(0, σ²I)

Intuitively:
- Add Gaussian noise to input
- Take majority vote over noisy predictions
- This "smooths out" the classifier

CERTIFICATION THEOREM:
=====================
Cohen et al. (2019) prove:

If for some input x:
    P(f(x + ε) = c_A) ≥ p_A  [probability of top class]
    P(f(x + ε) = c_B) ≤ p_B  [probability of runner-up]

Then g(x) = c_A is certifiably robust within L2 radius:

    R = σ/2 * (Φ^(-1)(p_A) - Φ^(-1)(p_B))

where Φ^(-1) is the inverse CDF of standard normal distribution.

INTUITION:
----------
- p_A is the probability that class c_A wins the majority vote
- p_B is the probability that any other class wins
- If p_A >> p_B (high confidence), then R is large
- R represents the certified robust radius

IMPLEMENTATION:
===============

Two-Stage Process:
------------------
1. **Selection**: Find the predicted class c_A
   - Sample n0 noisy predictions
   - Take majority vote

2. **Certification**: Estimate p_A and certify radius
   - Sample n more noisy predictions
   - Compute confidence intervals for p_A, p_B
   - Calculate certified radius R

Monte Carlo Sampling:
--------------------
We estimate probabilities using Monte Carlo:
    
    P(f(x + ε) = c) ≈ (# times f(x + ε_i) = c) / N

where ε_1, ..., ε_N ~ N(0, σ²I)

Statistical Guarantees:
----------------------
Using Clopper-Pearson confidence intervals, we get probabilistic guarantees:
    
    With probability ≥ 1-α: p_A ≥ p̂_A and p_B ≤ p̂_B

This gives us a certified radius R with probability ≥ 1-α.

KEY PARAMETERS:
===============
1. **σ (sigma)**: Noise standard deviation
   - Larger σ: larger certified radius but lower accuracy
   - Typical: σ ∈ [0.12, 1.0]

2. **n0**: Number of samples for selection
   - Typical: n0 = 100

3. **n**: Number of samples for certification
   - More samples: tighter confidence intervals
   - Typical: n = 10,000 or more

4. **α (alpha)**: Confidence level
   - Typical: α = 0.001 (99.9% confidence)

TRADEOFFS:
==========
1. **Accuracy vs. Certified Radius**:
   - Larger σ: larger R but lower clean accuracy
   - Smaller σ: higher accuracy but smaller R

2. **Computational Cost**:
   - Certification requires many forward passes (N samples)
   - Much slower than standard inference
   - Can be parallelized

3. **L2 vs. L∞**:
   - Randomized smoothing provides L2 certification
   - L∞ certification is more challenging

Author: Educational Materials
Date: November 2025
Difficulty: Advanced
Prerequisites: Probability theory, statistics, confidence intervals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from scipy.stats import norm, binom
from tqdm import tqdm
import math


class RandomizedSmoothing:
    """
    Randomized Smoothing for Certified Robustness
    
    This class implements randomized smoothing, which provides provable L2
    robustness guarantees for any classifier.
    
    Mathematical Formulation:
    -------------------------
    Given base classifier f, construct smoothed classifier:
    
        g(x) = argmax_c E_{ε~N(0,σ²I)}[1{f(x + ε) = c}]
    
    Certification: If P(f(x + ε) = c_A) ≥ p_A, then g(x) is certifiably
    robust within L2 radius:
    
        R = σ/2 * (Φ^(-1)(p_A) - Φ^(-1)(p_B))
    
    where c_B is the runner-up class and Φ^(-1) is inverse standard normal CDF.
    
    Attributes:
    -----------
    base_classifier : nn.Module
        The base classifier to smooth
    sigma : float
        Standard deviation of Gaussian noise
    device : torch.device
        Computation device
    """
    
    def __init__(
        self,
        base_classifier: nn.Module,
        sigma: float = 0.25,
        device: Optional[torch.device] = None
    ):
        """
        Initialize randomized smoothing.
        
        Parameters:
        -----------
        base_classifier : nn.Module
            Base classifier to smooth
            Should output logits (pre-softmax)
        sigma : float, default=0.25
            Standard deviation of Gaussian noise
            Larger σ: more smoothing, larger certified radius
            Smaller σ: less smoothing, higher accuracy
            Typical values: σ ∈ [0.12, 0.25, 0.50, 1.0]
        device : torch.device, optional
            Computation device
        """
        self.base_classifier = base_classifier
        self.sigma = sigma
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set base classifier to evaluation mode
        self.base_classifier.eval()
        self.base_classifier = self.base_classifier.to(self.device)
        
        print(f"Randomized Smoothing Configuration:")
        print(f"  Sigma (σ): {self.sigma}")
        print(f"  Device: {self.device}")
    
    def _sample_noise(
        self,
        x: torch.Tensor,
        num_samples: int,
        batch_size: int = 1000
    ) -> torch.Tensor:
        """
        Generate predictions on noisy samples.
        
        This is the Monte Carlo estimation step. We:
        1. Add Gaussian noise: x + ε where ε ~ N(0, σ²I)
        2. Get predictions: f(x + ε)
        3. Repeat N times
        
        For memory efficiency, we process in batches.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input image (single image)
        num_samples : int
            Number of noisy samples to generate
        batch_size : int, default=1000
            Batch size for processing samples
        
        Returns:
        --------
        counts : torch.Tensor
            Count of predictions for each class
            Shape: (num_classes,)
        """
        with torch.no_grad():
            # Get number of classes from a forward pass
            if not hasattr(self, 'num_classes'):
                test_output = self.base_classifier(x.unsqueeze(0))
                self.num_classes = test_output.size(1)
            
            # Initialize counts for each class
            counts = torch.zeros(self.num_classes, device=self.device)
            
            # Process in batches for memory efficiency
            num_batches = math.ceil(num_samples / batch_size)
            
            for _ in range(num_batches):
                # Determine actual batch size (might be smaller for last batch)
                current_batch_size = min(batch_size, num_samples - len(counts.nonzero()))
                
                # Create batch by repeating input
                batch = x.repeat(current_batch_size, 1, 1, 1)
                
                # Add Gaussian noise: ε ~ N(0, σ²I)
                noise = torch.randn_like(batch) * self.sigma
                noisy_batch = batch + noise
                
                # Get predictions on noisy samples
                outputs = self.base_classifier(noisy_batch)
                predictions = outputs.argmax(dim=1)
                
                # Count predictions for each class
                for pred in predictions:
                    counts[pred] += 1
        
        return counts
    
    def predict(
        self,
        x: torch.Tensor,
        n: int = 1000,
        alpha: float = 0.001,
        batch_size: int = 1000
    ) -> Tuple[int, float]:
        """
        Predict class and certify robustness for a single input.
        
        Two-stage process:
        ------------------
        1. Selection: Find predicted class using n0 samples
        2. Certification: Estimate probabilities and compute radius
        
        Parameters:
        -----------
        x : torch.Tensor
            Input image (single image, shape: (C, H, W))
        n : int, default=1000
            Number of samples for certification
            More samples: tighter confidence intervals
            Typical: n ≥ 10,000 for good guarantees
        alpha : float, default=0.001
            Confidence level (probability of error)
            Typical: α = 0.001 (99.9% confidence)
        batch_size : int, default=1000
            Batch size for processing samples
        
        Returns:
        --------
        prediction : int
            Predicted class label
        radius : float
            Certified L2 radius
            If radius = 0, certification failed
        """
        # Stage 1: Selection (find top class)
        # Use n/10 samples for efficiency
        n_selection = max(100, n // 10)
        counts_selection = self._sample_noise(x, n_selection, batch_size)
        top_class = counts_selection.argmax().item()
        
        # Stage 2: Certification (estimate probabilities)
        counts_cert = self._sample_noise(x, n, batch_size)
        
        # Compute confidence intervals for probabilities
        # Using Clopper-Pearson (exact) binomial confidence interval
        
        # Count for top class
        count_top = counts_cert[top_class].item()
        
        # Lower confidence bound for p_A (probability of top class)
        p_A_lower = self._lower_confidence_bound(count_top, n, alpha)
        
        # Find runner-up class (excluding top class)
        counts_cert[top_class] = -1  # Temporarily remove top class
        runner_up_class = counts_cert.argmax().item()
        count_runner_up = counts_cert[runner_up_class].item()
        
        # Upper confidence bound for p_B (probability of runner-up)
        p_B_upper = self._upper_confidence_bound(count_runner_up, n, alpha)
        
        # Compute certified radius
        # R = σ/2 * (Φ^(-1)(p_A) - Φ^(-1)(p_B))
        if p_A_lower > p_B_upper:
            # Certification succeeded
            radius = self._compute_radius(p_A_lower, p_B_upper)
        else:
            # Certification failed (not enough confidence)
            radius = 0.0
        
        return top_class, radius
    
    def _lower_confidence_bound(
        self,
        count: int,
        n: int,
        alpha: float
    ) -> float:
        """
        Compute lower confidence bound for binomial proportion.
        
        Using Clopper-Pearson method (exact binomial confidence interval).
        
        Mathematical formulation:
        -------------------------
        We observe count successes in n trials.
        True probability p satisfies:
            P(count | p) ≥ α/2
        
        This gives us a lower bound p_lower such that:
            P(p ≥ p_lower) ≥ 1 - α/2
        
        Parameters:
        -----------
        count : int
            Number of successes
        n : int
            Number of trials
        alpha : float
            Significance level
        
        Returns:
        --------
        p_lower : float
            Lower confidence bound
        """
        return binom.ppf(alpha/2, n, count/n) / n if count > 0 else 0.0
    
    def _upper_confidence_bound(
        self,
        count: int,
        n: int,
        alpha: float
    ) -> float:
        """
        Compute upper confidence bound for binomial proportion.
        
        Parameters:
        -----------
        count : int
            Number of successes
        n : int
            Number of trials
        alpha : float
            Significance level
        
        Returns:
        --------
        p_upper : float
            Upper confidence bound
        """
        return binom.ppf(1 - alpha/2, n, count/n) / n if count < n else 1.0
    
    def _compute_radius(
        self,
        p_A: float,
        p_B: float
    ) -> float:
        """
        Compute certified radius from probabilities.
        
        Formula:
        --------
        R = σ/2 * (Φ^(-1)(p_A) - Φ^(-1)(p_B))
        
        where:
        - Φ^(-1) is inverse CDF of standard normal
        - p_A is lower bound on top class probability
        - p_B is upper bound on runner-up probability
        
        Intuition:
        ----------
        - If p_A is close to 1 and p_B close to 0: large R (high confidence)
        - If p_A ≈ p_B: small R (low confidence)
        - σ scales the radius: larger σ = larger R
        
        Parameters:
        -----------
        p_A : float
            Lower bound on top class probability
        p_B : float
            Upper bound on runner-up probability
        
        Returns:
        --------
        radius : float
            Certified L2 radius
        """
        # Compute inverse CDF values
        # norm.ppf is the inverse of standard normal CDF
        if p_A >= 1.0:
            p_A = 0.999999  # Avoid infinity
        if p_B <= 0.0:
            p_B = 0.000001
        
        phi_inv_pA = norm.ppf(p_A)
        phi_inv_pB = norm.ppf(p_B)
        
        # Compute radius
        radius = (self.sigma / 2.0) * (phi_inv_pA - phi_inv_pB)
        
        return max(0.0, radius)  # Ensure non-negative
    
    def certify_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        n: int = 10000,
        alpha: float = 0.001,
        batch_size: int = 1000,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Certify a batch of images and compute metrics.
        
        For each image:
        1. Predict class and certified radius
        2. Check if prediction is correct
        3. Check if certified against attack
        
        Metrics:
        --------
        - Clean accuracy: fraction of correct predictions
        - Certified accuracy at radius r: fraction correct AND certified ≥ r
        
        Parameters:
        -----------
        images : torch.Tensor
            Batch of images
        labels : torch.Tensor
            True labels
        n : int, default=10000
            Number of samples per image
        alpha : float, default=0.001
            Confidence level
        batch_size : int, default=1000
            Batch size for noise sampling
        verbose : bool, default=True
            Print progress
        
        Returns:
        --------
        results : Dict[str, float]
            Certification metrics
        """
        num_images = len(images)
        predictions = []
        radii = []
        
        if verbose:
            pbar = tqdm(range(num_images), desc="Certifying")
        else:
            pbar = range(num_images)
        
        for i in pbar:
            pred, radius = self.predict(images[i], n, alpha, batch_size)
            predictions.append(pred)
            radii.append(radius)
        
        # Convert to tensors
        predictions = torch.tensor(predictions, device=labels.device)
        radii = torch.tensor(radii)
        
        # Compute metrics
        correct = (predictions == labels)
        
        # Clean accuracy
        clean_accuracy = correct.float().mean().item()
        
        # Certified accuracy at different radii
        radius_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        certified_accuracies = {}
        
        for r in radius_levels:
            # Certified: correct prediction AND certified radius ≥ r
            certified = correct & (radii >= r)
            certified_accuracies[f'certified_acc_r={r}'] = certified.float().mean().item()
        
        # Average certified radius (for correctly classified examples)
        avg_radius = radii[correct].mean().item() if correct.any() else 0.0
        
        results = {
            'clean_accuracy': clean_accuracy,
            'avg_certified_radius': avg_radius,
            **certified_accuracies
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print("Certification Results")
            print("=" * 60)
            print(f"Clean Accuracy: {clean_accuracy:.2%}")
            print(f"Avg Certified Radius: {avg_radius:.4f}")
            print("\nCertified Accuracy at different radii:")
            for r in radius_levels:
                key = f'certified_acc_r={r}'
                print(f"  r = {r}: {results[key]:.2%}")
            print("=" * 60)
        
        return results


# Example usage
if __name__ == "__main__":
    """
    Demonstration of certified robustness via randomized smoothing.
    """
    print("=" * 70)
    print("Certified Robustness via Randomized Smoothing")
    print("=" * 70)
    print("\nThis script demonstrates provable robustness guarantees")
    print("using randomized smoothing.")
    print("\nNote: This requires utils.py for data loading and model utilities.")
    print("\nWarning: Certification is computationally expensive!")
    print("=" * 70)
