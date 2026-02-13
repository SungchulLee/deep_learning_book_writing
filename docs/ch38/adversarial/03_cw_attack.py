"""
Module 62.3: Carlini & Wagner (C&W) Attack - Intermediate Level

This module implements the Carlini & Wagner (C&W) attack, one of the strongest
optimization-based adversarial attacks. C&W reformulates adversarial example
generation as an unconstrained optimization problem and is significantly more
powerful than gradient-based methods like FGSM and PGD.

MATHEMATICAL BACKGROUND:
=======================

Unlike FGSM/PGD which directly constrain perturbation magnitude, C&W uses
an unconstrained optimization with a carefully designed objective function.

Problem Formulation:
-------------------
Find δ to solve:
    minimize ||δ||_p + c · f(x + δ)

where:
- ||δ||_p is the perturbation magnitude (p = 2, ∞, or 0)
- c > 0 is a trade-off constant
- f(·) is an objective function that encourages misclassification

The key innovation is the objective function f(·).

LOGIT-BASED OBJECTIVE:
=====================
C&W uses a clever objective based on logits (pre-softmax outputs):

For untargeted attack:
    f(x') = max(max_{i≠t} Z(x')_i - Z(x')_t, -κ)

For targeted attack (target class t'):
    f(x') = max(Z(x')_t - Z(x')_{t'}, -κ)

where:
- Z(x') are the logits (pre-softmax outputs)
- t is the true class
- t' is the target class
- κ ≥ 0 is a confidence parameter (typically 0)

INTUITION:
----------
1. For untargeted: We want max_{i≠t} Z_i > Z_t (any wrong class beats true class)
2. For targeted: We want Z_{t'} > Z_t (target class beats true class)
3. The max(..., -κ) ensures:
   - If already misclassified: f(x') ≤ 0 (attack succeeded)
   - If correctly classified: f(x') > 0 (keep optimizing)
4. κ > 0 adds confidence margin: target must beat true class by κ

CHANGE OF VARIABLES:
===================
To handle box constraints x' ∈ [0, 1], C&W uses change of variables:

    x' = 0.5 * (tanh(w) + 1)

where w is the optimization variable. This ensures:
- tanh(w) ∈ (-1, 1)
- x' ∈ (0, 1) automatically satisfied
- Can optimize w without explicit constraints

Then perturbation is:
    δ = x' - x = 0.5 * (tanh(w) + 1) - x

OPTIMIZATION:
============
C&W uses Adam optimizer to solve:
    minimize_{w} ||0.5*(tanh(w)+1) - x||_p + c · f(0.5*(tanh(w)+1))

Binary search on c:
-------------------
Since we don't know the optimal c value a priori, C&W performs binary search:

1. Initialize: c_low = 0, c_high = large number
2. For each c:
   - Run optimization
   - If attack succeeds: c_high = c (try smaller c)
   - If attack fails: c_low = c (try larger c)
3. Return smallest successful c

This finds the minimal perturbation that achieves misclassification.

KEY DIFFERENCES FROM PGD:
=========================
1. **Optimization-based**: Uses Adam optimizer instead of gradient ascent
2. **Unconstrained**: No explicit projection, uses change of variables
3. **Adaptive**: Binary search finds optimal trade-off parameter
4. **Stronger**: Usually finds smaller perturbations than PGD
5. **Slower**: Requires multiple optimization runs with binary search

NORMS:
======
C&W can minimize different norms:

L2 (most common):
    ||δ||_2 = sqrt(Σ δ_i²)
    Measures overall perturbation energy

L∞:
    ||δ||_∞ = max_i |δ_i|
    Measures maximum per-pixel change

L0 (discrete):
    ||δ||_0 = number of changed pixels
    Measures sparsity of perturbation

Author: Educational Materials
Date: November 2025
Difficulty: Intermediate to Advanced
Prerequisites: PGD (Module 62.2), optimization theory, PyTorch optimizers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, List, Dict, Literal
from tqdm import tqdm


class CarliniWagnerL2:
    """
    Carlini & Wagner (C&W) L2 Attack
    
    This class implements the C&W attack that minimizes the L2 norm of
    perturbations while ensuring misclassification. It's one of the strongest
    known attacks and often finds smaller perturbations than PGD.
    
    Mathematical Formulation:
    -------------------------
    The C&W L2 attack solves:
    
        minimize ||δ||_2² + c · f(x + δ)
    
    where f is the logit-based objective:
        f(x') = max(max_{i≠t} Z(x')_i - Z(x')_t, -κ)
    
    The optimization is performed over transformed variable w:
        x' = 0.5 * (tanh(w) + 1)
    
    Binary search is used to find the optimal constant c.
    
    Attributes:
    -----------
    model : nn.Module
        Neural network to attack
    c : float
        Trade-off constant (found via binary search)
    kappa : float
        Confidence parameter
    learning_rate : float
        Learning rate for Adam optimizer
    max_iter : int
        Maximum optimization iterations
    binary_search_steps : int
        Number of binary search steps for c
    """
    
    def __init__(
        self,
        model: nn.Module,
        c: float = 1.0,
        kappa: float = 0.0,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        binary_search_steps: int = 9,
        initial_const: float = 1e-3,
        device: Optional[torch.device] = None,
        abort_early: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0
    ):
        """
        Initialize the C&W L2 attack.
        
        Parameters:
        -----------
        model : nn.Module
            The neural network to attack
        c : float, default=1.0
            Initial trade-off constant
            Will be tuned via binary search
        kappa : float, default=0.0
            Confidence parameter
            κ = 0: just misclassify
            κ > 0: misclassify with confidence margin κ
            Higher κ = stronger attack but larger perturbation
        learning_rate : float, default=0.01
            Learning rate for Adam optimizer
            Common values: 0.01 (default), 0.001 (careful), 0.1 (aggressive)
        max_iter : int, default=1000
            Maximum iterations for each optimization
            More iterations = stronger attack but slower
            Common values: 100 (fast), 1000 (standard), 10000 (thorough)
        binary_search_steps : int, default=9
            Number of binary search iterations for c
            More steps = better c value but slower
            Each step doubles total optimization time
        initial_const : float, default=1e-3
            Initial value for c in binary search
            Start small to find minimal perturbations
        device : torch.device, optional
            Computation device
        abort_early : bool, default=True
            Stop optimization early if attack succeeds
            Saves time but may miss smaller perturbations
        clip_min : float, default=0.0
            Minimum valid pixel value
        clip_max : float, default=1.0
            Maximum valid pixel value
        """
        self.model = model
        self.c = c
        self.kappa = kappa
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.device = device if device is not None else next(model.parameters()).device
        self.abort_early = abort_early
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        # Set model to evaluation mode
        self.model.eval()
        self.model = self.model.to(self.device)
        
        print(f"C&W L2 Attack Configuration:")
        print(f"  Kappa (κ): {self.kappa}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Max iterations: {self.max_iter}")
        print(f"  Binary search steps: {self.binary_search_steps}")
        print(f"  Early abort: {self.abort_early}")
    
    def _arctanh(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute arctanh (inverse hyperbolic tangent).
        
        This is the inverse of the change of variables transformation.
        Given x ∈ (0, 1), we want to find w such that:
            x = 0.5 * (tanh(w) + 1)
        
        Solving for w:
            2x - 1 = tanh(w)
            w = arctanh(2x - 1)
        
        Mathematical note:
            arctanh(z) = 0.5 * log((1+z)/(1-z))
        
        We add small epsilon to avoid numerical issues at boundaries.
        
        Parameters:
        -----------
        x : torch.Tensor
            Values in (0, 1)
        
        Returns:
        --------
        w : torch.Tensor
            Inverse transformed values
        """
        # Clamp to (epsilon, 1-epsilon) to avoid log(0)
        epsilon = 1e-6
        x = torch.clamp(x, epsilon, 1.0 - epsilon)
        
        # Transform: x ∈ (0,1) → z ∈ (-1,1)
        z = 2 * x - 1
        z = torch.clamp(z, -1 + epsilon, 1 - epsilon)
        
        # Compute arctanh(z)
        w = 0.5 * torch.log((1 + z) / (1 - z))
        
        return w
    
    def _f_objective(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the C&W objective function f(x').
        
        This is the clever part of C&W that encourages misclassification
        using logits (pre-softmax scores).
        
        For untargeted attack:
        ----------------------
        We want to maximize: Z_target - max_{i≠target} Z_i
        Which is equivalent to minimizing:
            f = max(max_{i≠target} Z_i - Z_target, -κ)
        
        When f ≤ 0, the attack has succeeded (with confidence κ).
        
        For targeted attack:
        -------------------
        We want to maximize: Z_target - Z_true
        Which is minimizing:
            f = max(Z_true - Z_target, -κ)
        
        Parameters:
        -----------
        outputs : torch.Tensor
            Model outputs (logits, not probabilities)
        labels : torch.Tensor
            True class labels
        targeted : bool
            Whether this is a targeted attack
        target_labels : torch.Tensor, optional
            Target class labels (for targeted attack)
        
        Returns:
        --------
        f_value : torch.Tensor
            Objective value (shape: (batch_size,))
        """
        batch_size = outputs.size(0)
        num_classes = outputs.size(1)
        
        if targeted:
            # Targeted attack: minimize Z_true - Z_target
            if target_labels is None:
                raise ValueError("target_labels required for targeted attack")
            
            # Get logit for true class
            true_logits = outputs[torch.arange(batch_size), labels]
            # Get logit for target class
            target_logits = outputs[torch.arange(batch_size), target_labels]
            
            # Objective: max(Z_true - Z_target, -κ)
            # We want Z_target > Z_true + κ
            f_value = torch.clamp(true_logits - target_logits, min=-self.kappa)
        else:
            # Untargeted attack: minimize max_{i≠t} Z_i - Z_t
            
            # Get logit for true class
            true_logits = outputs[torch.arange(batch_size), labels]
            
            # Get logits for all other classes
            # Create mask: 1 for true class, 0 for others
            one_hot = F.one_hot(labels, num_classes).bool()
            
            # Set true class logit to -infinity so it's not selected by max
            other_logits = outputs.clone()
            other_logits[one_hot] = float('-inf')
            
            # Get maximum logit among wrong classes
            max_other_logits, _ = torch.max(other_logits, dim=1)
            
            # Objective: max(max_{i≠t} Z_i - Z_t, -κ)
            # We want Z_t < max_{i≠t} Z_i + κ
            f_value = torch.clamp(max_other_logits - true_logits, min=-self.kappa)
        
        return f_value
    
    def _optimize(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        c_value: float,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Perform optimization for a given c value.
        
        This is the core optimization loop that finds adversarial examples.
        
        Algorithm:
        ----------
        1. Initialize w = arctanh(2x - 1)
        2. For iter = 1 to max_iter:
             a. x' = 0.5 * (tanh(w) + 1)  [apply transformation]
             b. Get model outputs on x'
             c. Compute loss = ||δ||_2² + c · f(x')
             d. Backpropagate and update w with Adam
             e. Track best adversarial example found so far
        3. Return best adversarial example
        
        Parameters:
        -----------
        images : torch.Tensor
            Clean images
        labels : torch.Tensor
            True labels
        c_value : float
            Current c value for this optimization
        targeted : bool
            Targeted attack flag
        target_labels : torch.Tensor, optional
            Target labels
        verbose : bool
            Print progress
        
        Returns:
        --------
        best_adv : torch.Tensor
            Best adversarial example found
        best_l2 : torch.Tensor
            L2 distance of best perturbation
        success : bool
            Whether attack succeeded for all examples
        """
        batch_size = images.size(0)
        
        # Initialize w using inverse transformation
        # w = arctanh(2x - 1) so that x = 0.5*(tanh(w) + 1)
        w = self._arctanh(images)
        w = w.to(self.device).detach()
        w.requires_grad = True
        
        # Initialize optimizer for w
        optimizer = optim.Adam([w], lr=self.learning_rate)
        
        # Track best adversarial example for each image
        best_adv = images.clone()
        best_l2 = torch.full((batch_size,), float('inf'), device=self.device)
        best_attack_success = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Optimization loop
        iterator = range(self.max_iter)
        if verbose:
            iterator = tqdm(iterator, desc=f"C&W optimization (c={c_value:.2e})")
        
        for iteration in iterator:
            # Apply transformation: x' = 0.5 * (tanh(w) + 1)
            # This ensures x' ∈ (0, 1) automatically
            adv_images = 0.5 * (torch.tanh(w) + 1)
            
            # Clip to valid range (should be almost redundant due to tanh)
            adv_images = torch.clamp(adv_images, self.clip_min, self.clip_max)
            
            # Get model outputs
            outputs = self.model(adv_images)
            
            # Compute perturbation
            delta = adv_images - images
            
            # L2 loss (squared L2 norm)
            # ||δ||_2² = Σ δ²
            l2_loss = torch.sum(delta.view(batch_size, -1) ** 2, dim=1)
            
            # C&W objective f(x')
            f_loss = self._f_objective(outputs, labels, targeted, target_labels)
            
            # Total loss: ||δ||_2² + c · f(x')
            # c controls trade-off between perturbation size and attack success
            loss = torch.sum(l2_loss + c_value * f_loss)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Check which adversarial examples succeeded
            # Success means: f(x') ≤ 0 (achieved misclassification with confidence κ)
            current_success = (f_loss <= 0)
            
            # Update best adversarial examples
            # For successful attacks, keep the one with smallest L2 distance
            improved_mask = current_success & (l2_loss < best_l2)
            
            if improved_mask.any():
                best_adv[improved_mask] = adv_images[improved_mask].detach()
                best_l2[improved_mask] = l2_loss[improved_mask].detach()
                best_attack_success[improved_mask] = True
            
            # Early abort if all attacks succeeded and we're not looking for smaller perturbations
            if self.abort_early and current_success.all():
                if verbose:
                    print(f"\nEarly abort at iteration {iteration+1}: all attacks succeeded")
                break
        
        # Check overall success (whether all examples were successfully attacked)
        success = best_attack_success.all().item()
        
        return best_adv, best_l2, success
    
    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Generate adversarial examples using C&W attack with binary search.
        
        This method implements the complete C&W attack with binary search on c:
        
        Algorithm:
        ----------
        1. Initialize c_low = 0, c_high = 1e10
        2. For b = 1 to binary_search_steps:
             a. c = (c_low + c_high) / 2
             b. Run optimization with current c
             c. If attack succeeds: c_high = c (try smaller c)
             d. If attack fails: c_low = c (try larger c)
        3. Return best adversarial example from all searches
        
        The binary search finds the smallest c (and thus smallest perturbation)
        that successfully fools the model.
        
        Parameters:
        -----------
        images : torch.Tensor
            Clean images
        labels : torch.Tensor
            True labels
        targeted : bool, default=False
            Targeted attack flag
        target_labels : torch.Tensor, optional
            Target labels for targeted attack
        verbose : bool, default=False
            Print detailed progress
        
        Returns:
        --------
        adv_images : torch.Tensor
            Adversarial images
        """
        # Move to device
        images = images.to(self.device)
        labels = labels.to(self.device)
        if targeted and target_labels is not None:
            target_labels = target_labels.to(self.device)
        
        batch_size = images.size(0)
        
        # Initialize bounds for binary search on c
        # c_low: if attack fails, c wasn't large enough
        # c_high: if attack succeeds, c might be too large
        c_low = torch.zeros(batch_size, device=self.device)
        c_high = torch.full((batch_size,), 1e10, device=self.device)
        
        # Track best adversarial example across all binary search steps
        best_adv = images.clone()
        best_l2 = torch.full((batch_size,), float('inf'), device=self.device)
        
        if verbose:
            print(f"\nStarting binary search with {self.binary_search_steps} steps")
        
        # Binary search on c
        for search_step in range(self.binary_search_steps):
            # Current c value: midpoint of [c_low, c_high]
            # For first iteration, use initial_const
            if search_step == 0:
                c_current = torch.full((batch_size,), self.initial_const, device=self.device)
            else:
                c_current = (c_low + c_high) / 2
            
            if verbose:
                c_min, c_max, c_mean = c_current.min().item(), c_current.max().item(), c_current.mean().item()
                print(f"\n[Step {search_step+1}/{self.binary_search_steps}] "
                      f"c range: [{c_min:.2e}, {c_max:.2e}], mean: {c_mean:.2e}")
            
            # Run optimization with current c
            # Use the mean c value for simplicity (could also optimize per-example)
            c_value = c_current.mean().item()
            adv_images, l2_dist, success = self._optimize(
                images, labels, c_value, targeted, target_labels, verbose=False
            )
            
            # Update best adversarial examples
            improved_mask = l2_dist < best_l2
            if improved_mask.any():
                best_adv[improved_mask] = adv_images[improved_mask]
                best_l2[improved_mask] = l2_dist[improved_mask]
            
            # Update binary search bounds
            # For each example, check if attack succeeded
            with torch.no_grad():
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs, 1)
                
                if targeted:
                    # For targeted: success if predicted == target
                    success_mask = (predicted == target_labels)
                else:
                    # For untargeted: success if predicted != true label
                    success_mask = (predicted != labels)
            
            # Update bounds based on success
            # If succeeded: try smaller c (c_high = c)
            # If failed: try larger c (c_low = c)
            c_high[success_mask] = c_current[success_mask]
            c_low[~success_mask] = c_current[~success_mask]
            
            if verbose:
                success_rate = success_mask.float().mean().item()
                avg_l2 = l2_dist[success_mask].mean().item() if success_mask.any() else float('inf')
                print(f"Success rate: {success_rate:.2%}, Avg L2: {avg_l2:.4f}")
        
        if verbose:
            print(f"\nBinary search complete.")
            final_success = (best_l2 < float('inf')).float().mean().item()
            final_l2 = best_l2[best_l2 < float('inf')].mean().item() if (best_l2 < float('inf')).any() else float('inf')
            print(f"Final success rate: {final_success:.2%}")
            print(f"Final avg L2: {final_l2:.4f}")
        
        return best_adv
    
    def evaluate(
        self,
        clean_images: torch.Tensor,
        labels: torch.Tensor,
        adv_images: torch.Tensor,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate C&W attack effectiveness.
        
        Parameters:
        -----------
        clean_images : torch.Tensor
            Original images
        labels : torch.Tensor
            True labels
        adv_images : torch.Tensor
            Adversarial images
        verbose : bool
            Print results
        
        Returns:
        --------
        metrics : Dict[str, float]
            Evaluation metrics
        """
        with torch.no_grad():
            # Clean accuracy
            clean_outputs = self.model(clean_images.to(self.device))
            _, clean_pred = torch.max(clean_outputs, 1)
            clean_accuracy = (clean_pred == labels.to(self.device)).float().mean().item()
            
            # Adversarial accuracy
            adv_outputs = self.model(adv_images.to(self.device))
            _, adv_pred = torch.max(adv_outputs, 1)
            adv_accuracy = (adv_pred == labels.to(self.device)).float().mean().item()
            
            # Perturbation statistics
            perturbation = (adv_images - clean_images).cpu()
            
            # L2 norm (per-example, then average)
            l2_norms = torch.norm(perturbation.view(len(perturbation), -1), p=2, dim=1)
            l2_mean = l2_norms.mean().item()
            l2_median = l2_norms.median().item()
            
            # L∞ norm
            linf_norm = torch.max(torch.abs(perturbation)).item()
        
        metrics = {
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'attack_success_rate': 1.0 - adv_accuracy,
            'avg_l2_perturbation': l2_mean,
            'median_l2_perturbation': l2_median,
            'max_linf_perturbation': linf_norm,
        }
        
        if verbose:
            print("=" * 60)
            print("C&W L2 Attack Evaluation")
            print("=" * 60)
            print(f"Clean Accuracy: {clean_accuracy:.2%}")
            print(f"Adversarial Accuracy: {adv_accuracy:.2%}")
            print(f"Attack Success Rate: {metrics['attack_success_rate']:.2%}")
            print(f"\nPerturbation Statistics:")
            print(f"  Mean L2: {l2_mean:.4f}")
            print(f"  Median L2: {l2_median:.4f}")
            print(f"  Max L∞: {linf_norm:.6f}")
            print("=" * 60)
        
        return metrics


# Example usage
if __name__ == "__main__":
    """
    Demonstration of C&W attack.
    """
    print("=" * 70)
    print("Carlini & Wagner L2 Attack Demonstration")
    print("=" * 70)
    print("\nThis script demonstrates the C&W attack, a powerful")
    print("optimization-based adversarial attack.")
    print("\nNote: This requires utils.py for data loading and model utilities.")
    print("=" * 70)
