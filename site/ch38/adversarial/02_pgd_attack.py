"""
Module 62.2: Projected Gradient Descent (PGD) Attack - Intermediate Level

This module implements the Projected Gradient Descent (PGD) attack, a stronger
iterative variant of FGSM. PGD is considered one of the strongest first-order
adversarial attacks and is commonly used to evaluate model robustness.

MATHEMATICAL BACKGROUND:
=======================

PGD extends FGSM by applying multiple small gradient steps, projecting back
onto the allowed perturbation set after each step. This makes it much stronger
than the single-step FGSM.

Given a model f with parameters θ, input x, label y, and perturbation budget ε:

PGD solves:
    maximize_{||δ||_∞ ≤ ε} L(θ, x + δ, y)

Algorithm (untargeted):
-----------------------
1. Initialize: x^(0) = x + uniform_noise[-ε, ε]  (random start)
2. For t = 0 to T-1:
       x^(t+1) = Π_{x+S}(x^(t) + α · sign(∇_x L(θ, x^(t), y)))
3. Return x^(T)

where:
- Π_{x+S} is projection onto the ε-ball: Π(z) = clip(z, x-ε, x+ε)
- α is the step size (typically α = ε/num_iter or α = 2.5·ε/num_iter)
- S = {δ : ||δ||_∞ ≤ ε} is the allowed perturbation set
- T is the number of iterations

KEY DIFFERENCES FROM FGSM:
===========================
1. **Multi-step**: PGD takes multiple small steps (typically 10-100)
2. **Random initialization**: Starts from random point in ε-ball
3. **Projection**: After each step, project back to ε-ball
4. **Stronger**: Much better at finding adversarial examples

RANDOM INITIALIZATION:
=====================
Random initialization is crucial for PGD's strength:
- Helps escape poor local optima
- Explores different regions of the ε-ball
- Makes attack less sensitive to initial conditions

Common strategies:
- Uniform: x^(0) ~ Uniform[x-ε, x+ε]
- Gaussian: x^(0) ~ N(x, σ²I), then project to ε-ball

PROJECTION OPERATOR:
===================
The projection Π ensures perturbation stays within ε-ball:

For L∞ norm:
    Π(z)_i = clip(z_i, x_i - ε, x_i + ε)

This is element-wise clipping to the box [x-ε, x+ε].

For L2 norm:
    If ||z - x||_2 ≤ ε: Π(z) = z
    Else: Π(z) = x + ε · (z - x) / ||z - x||_2

STEP SIZE SELECTION:
===================
The step size α controls the trade-off between:
- Larger α: Faster convergence but may overshoot
- Smaller α: More precise but needs more iterations

Common choices:
- α = ε / T (each step is 1/T of total budget)
- α = 2.5 · ε / T (slightly more aggressive)
- α = 2 · ε / T (from Madry et al., 2018)

Author: Educational Materials
Date: November 2025
Difficulty: Intermediate
Prerequisites: FGSM (Module 62.1), iterative optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Literal
from tqdm import tqdm
import copy


class PGD:
    """
    Projected Gradient Descent (PGD) Attack
    
    PGD is an iterative adversarial attack that applies multiple gradient steps
    with projection back onto the allowed perturbation set. It's significantly
    stronger than FGSM and is the de facto standard for evaluating robustness.
    
    Mathematical Formulation:
    -------------------------
    The PGD attack solves the constrained optimization problem:
    
        maximize L(θ, x + δ, y)  subject to ||δ||_p ≤ ε
    
    using projected gradient descent:
    
        x^(t+1) = Π_{x+S}(x^(t) + α · sign(∇_x L(θ, x^(t), y)))
    
    where Π is the projection operator onto the ε-ball.
    
    Attributes:
    -----------
    model : nn.Module
        The neural network to attack
    epsilon : float
        Maximum perturbation magnitude
    alpha : float
        Step size for each iteration
    num_iter : int
        Number of iterations to run
    norm : str
        Norm to use ('linf' or 'l2')
    random_init : bool
        Whether to use random initialization
    loss_fn : nn.Module
        Loss function to maximize
    early_stop : bool
        Whether to stop early if attack succeeds
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_iter: int = 40,
        norm: Literal['linf', 'l2'] = 'linf',
        random_init: bool = True,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        early_stop: bool = False
    ):
        """
        Initialize the PGD attack.
        
        Parameters:
        -----------
        model : nn.Module
            The neural network to attack
        epsilon : float, default=0.03
            Maximum perturbation magnitude (L∞ or L2 norm)
            For CIFAR-10: ε=8/255≈0.031 is standard
        alpha : float, default=0.01
            Step size for each iteration
            Common choice: α = 2.5 * ε / num_iter
            If not specified, defaults to ε / 4
        num_iter : int, default=40
            Number of PGD iterations
            More iterations = stronger attack but slower
            Common values: 10 (fast), 40 (standard), 100 (strong)
        norm : str, default='linf'
            Norm to use: 'linf' (L∞) or 'l2' (L2)
            L∞: all pixels bounded by ε
            L2: total perturbation bounded by ε
        random_init : bool, default=True
            Whether to randomly initialize perturbation
            True: more robust attack (recommended)
            False: starts from original image (like I-FGSM)
        loss_fn : nn.Module, optional
            Loss function to use
        device : torch.device, optional
            Computation device
        clip_min : float, default=0.0
            Minimum valid pixel value
        clip_max : float, default=1.0
            Maximum valid pixel value
        early_stop : bool, default=False
            Stop iteration if attack succeeds
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha if alpha is not None else epsilon / 4.0
        self.num_iter = num_iter
        self.norm = norm
        self.random_init = random_init
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.device = device if device is not None else next(model.parameters()).device
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.early_stop = early_stop
        
        # Set model to evaluation mode
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Validate parameters
        if self.norm not in ['linf', 'l2']:
            raise ValueError(f"norm must be 'linf' or 'l2', got {self.norm}")
        
        # Print configuration
        print(f"PGD Attack Configuration:")
        print(f"  Epsilon (ε): {self.epsilon}")
        print(f"  Alpha (α): {self.alpha}")
        print(f"  Iterations: {self.num_iter}")
        print(f"  Norm: L{self.norm}")
        print(f"  Random init: {self.random_init}")
        print(f"  Early stopping: {self.early_stop}")
    
    def _initialize_perturbation(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Initialize the perturbation.
        
        Two strategies:
        1. Random initialization (recommended):
           - Sample uniformly from ε-ball
           - Helps escape local optima
           - More robust attack
        
        2. Zero initialization:
           - Start from original image
           - Equivalent to iterated FGSM (I-FGSM)
           - Faster but potentially weaker
        
        Mathematical Detail (L∞):
        -------------------------
        Random init: δ^(0) ~ Uniform[-ε, ε] for each dimension
        Then project: x^(0) = clip(x + δ^(0), [clip_min, clip_max])
        
        Parameters:
        -----------
        images : torch.Tensor
            Clean images
        
        Returns:
        --------
        perturbed_images : torch.Tensor
            Initialized adversarial images
        """
        if self.random_init:
            # Random initialization within ε-ball
            if self.norm == 'linf':
                # Uniform random noise in [-ε, +ε]
                # This explores the entire L∞ ball uniformly
                delta = torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
            else:  # l2
                # Random direction, then scale to random radius ≤ ε
                delta = torch.randn_like(images)
                # Normalize to unit sphere
                delta_norm = delta.view(len(delta), -1).norm(p=2, dim=1)
                delta = delta / delta_norm.view(-1, 1, 1, 1)
                # Scale to random radius in [0, ε]
                random_radius = torch.rand(len(delta), device=images.device)
                random_radius = random_radius * self.epsilon
                delta = delta * random_radius.view(-1, 1, 1, 1)
            
            # Apply perturbation and clip to valid range
            perturbed_images = images + delta
            perturbed_images = torch.clamp(perturbed_images, self.clip_min, self.clip_max)
        else:
            # Start from clean images (zero perturbation)
            perturbed_images = images.clone()
        
        return perturbed_images
    
    def _project(
        self,
        perturbed_images: torch.Tensor,
        original_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Project perturbed images back onto the ε-ball around original images.
        
        This is the key operation that keeps perturbations bounded.
        After taking a gradient step, we may have ||δ|| > ε, so we need
        to project back to the feasible set.
        
        L∞ Projection:
        --------------
        For each pixel i:
            δ_i = clip(δ_i, -ε, +ε)
        
        This is element-wise clipping to the box [-ε, +ε].
        
        L2 Projection:
        --------------
        If ||δ||_2 ≤ ε: no projection needed
        Else: δ = ε · δ / ||δ||_2
        
        This scales δ to have exactly norm ε, maintaining direction.
        
        Parameters:
        -----------
        perturbed_images : torch.Tensor
            Current adversarial images
        original_images : torch.Tensor
            Original clean images
        
        Returns:
        --------
        projected_images : torch.Tensor
            Images projected back to ε-ball
        """
        # Compute current perturbation
        delta = perturbed_images - original_images
        
        if self.norm == 'linf':
            # L∞ projection: clip each element to [-ε, +ε]
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        else:  # l2
            # L2 projection: scale to have norm ≤ ε
            batch_size = len(delta)
            # Compute L2 norm for each example
            delta_norm = delta.view(batch_size, -1).norm(p=2, dim=1)
            # Create mask for examples that exceed ε
            exceed_mask = delta_norm > self.epsilon
            # Scale down perturbations that exceed ε
            if exceed_mask.any():
                scale = self.epsilon / delta_norm[exceed_mask]
                delta[exceed_mask] = delta[exceed_mask] * scale.view(-1, 1, 1, 1)
        
        # Apply projected perturbation
        projected_images = original_images + delta
        
        # Also clip to valid pixel range [clip_min, clip_max]
        # This ensures images remain valid (e.g., in [0, 1])
        projected_images = torch.clamp(projected_images, self.clip_min, self.clip_max)
        
        return projected_images
    
    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Generate adversarial examples using PGD.
        
        This is the main PGD algorithm:
        
        Algorithm:
        ----------
        1. Initialize x^(0) (randomly or from x)
        2. For t = 0 to T-1:
             a. Compute loss L(θ, x^(t), y)
             b. Compute gradient g = ∇_x L
             c. Update: x^(t+1) = x^(t) + α · sign(g)  [for L∞]
             d. Project: x^(t+1) = Π(x^(t+1))
             e. Clip to valid range
        3. Return x^(T)
        
        The projection step (d) is crucial: it keeps perturbation within ε-ball.
        
        Parameters:
        -----------
        images : torch.Tensor
            Clean images
        labels : torch.Tensor
            True labels (or target labels for targeted attack)
        targeted : bool, default=False
            Whether to perform targeted attack
        target_labels : torch.Tensor, optional
            Target labels for targeted attack
        verbose : bool, default=False
            Print iteration progress
        
        Returns:
        --------
        adv_images : torch.Tensor
            Adversarial images after PGD iterations
        """
        # Move to device
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Initialize adversarial images
        adv_images = self._initialize_perturbation(images)
        
        # For targeted attack
        if targeted and target_labels is not None:
            target_labels = target_labels.to(self.device)
        
        # Iterative attack
        iterator = range(self.num_iter)
        if verbose:
            iterator = tqdm(iterator, desc="PGD iterations")
        
        for i in iterator:
            # Enable gradient tracking for adversarial images
            adv_images = adv_images.detach().clone()
            adv_images.requires_grad = True
            
            # Forward pass
            outputs = self.model(adv_images)
            
            # Compute loss
            if targeted:
                # For targeted attack: minimize loss w.r.t. target
                loss = -self.loss_fn(outputs, target_labels)
            else:
                # For untargeted attack: maximize loss w.r.t. true label
                loss = self.loss_fn(outputs, labels)
            
            # Backward pass
            self.model.zero_grad()
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            loss.backward()
            
            # Get gradient
            grad = adv_images.grad.data
            
            # Gradient ascent step
            if self.norm == 'linf':
                # For L∞: take step in direction of sign of gradient
                # x^(t+1) = x^(t) + α · sign(∇L)
                adv_images = adv_images + self.alpha * torch.sign(grad)
            else:  # l2
                # For L2: take step in direction of normalized gradient
                # x^(t+1) = x^(t) + α · ∇L / ||∇L||_2
                grad_norm = grad.view(len(grad), -1).norm(p=2, dim=1)
                # Avoid division by zero
                grad_norm = torch.clamp(grad_norm, min=1e-12)
                normalized_grad = grad / grad_norm.view(-1, 1, 1, 1)
                adv_images = adv_images + self.alpha * normalized_grad
            
            # Project back to ε-ball around original images
            adv_images = self._project(adv_images, images)
            
            # Early stopping: if attack succeeds, no need to continue
            if self.early_stop:
                with torch.no_grad():
                    outputs = self.model(adv_images)
                    _, predicted = torch.max(outputs, 1)
                    if targeted:
                        # For targeted: check if all examples reach target
                        if (predicted == target_labels).all():
                            if verbose:
                                print(f"\nEarly stop at iteration {i+1}: all attacks succeeded")
                            break
                    else:
                        # For untargeted: check if all examples are misclassified
                        if (predicted != labels).all():
                            if verbose:
                                print(f"\nEarly stop at iteration {i+1}: all attacks succeeded")
                            break
        
        return adv_images.detach()
    
    def generate_with_restarts(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        num_restarts: int = 5,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        PGD with multiple random restarts.
        
        Multiple random restarts make PGD even stronger by:
        1. Trying different starting points in the ε-ball
        2. Selecting the best adversarial example from all runs
        3. Reducing sensitivity to initialization
        
        Algorithm:
        ----------
        For r = 1 to num_restarts:
            x_adv^(r) = PGD(x, y) with random initialization
        Return x_adv with highest loss (or best fooling)
        
        This is the strongest variant of PGD and is recommended for
        robustness evaluation.
        
        Parameters:
        -----------
        images : torch.Tensor
            Clean images
        labels : torch.Tensor
            True labels
        num_restarts : int, default=5
            Number of random restarts
            More restarts = stronger but slower
            Common values: 1 (no restart), 5 (standard), 10 (strong)
        targeted : bool, default=False
            Targeted attack flag
        target_labels : torch.Tensor, optional
            Target labels for targeted attack
        verbose : bool, default=False
            Print progress
        
        Returns:
        --------
        best_adv_images : torch.Tensor
            Adversarial examples with highest loss from all restarts
        """
        # Ensure random initialization is enabled
        original_random_init = self.random_init
        self.random_init = True
        
        best_adv_images = None
        best_loss = None
        
        for restart in range(num_restarts):
            if verbose:
                print(f"\nRestart {restart + 1}/{num_restarts}")
            
            # Generate adversarial examples with this restart
            adv_images = self.generate(
                images, labels, targeted, target_labels, verbose=False
            )
            
            # Compute loss for these adversarial examples
            with torch.no_grad():
                outputs = self.model(adv_images)
                if targeted:
                    loss = -self.loss_fn(outputs, target_labels).item()
                else:
                    loss = self.loss_fn(outputs, labels).item()
            
            # Keep best adversarial examples (highest loss)
            if best_loss is None or loss > best_loss:
                best_loss = loss
                best_adv_images = adv_images.clone()
            
            if verbose:
                print(f"Loss: {loss:.4f} (best: {best_loss:.4f})")
        
        # Restore original setting
        self.random_init = original_random_init
        
        return best_adv_images
    
    def evaluate(
        self,
        clean_images: torch.Tensor,
        labels: torch.Tensor,
        adv_images: torch.Tensor,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of PGD attack.
        
        Computes various metrics to assess attack effectiveness:
        - Clean accuracy
        - Adversarial accuracy
        - Attack success rate
        - Perturbation statistics (L∞, L2, L1)
        
        Parameters:
        -----------
        clean_images : torch.Tensor
            Original clean images
        labels : torch.Tensor
            True labels
        adv_images : torch.Tensor
            Adversarial images
        verbose : bool, default=True
            Print results
        
        Returns:
        --------
        metrics : Dict[str, float]
            Dictionary of evaluation metrics
        """
        with torch.no_grad():
            # Clean accuracy
            clean_outputs = self.model(clean_images.to(self.device))
            _, clean_pred = torch.max(clean_outputs, 1)
            clean_correct = (clean_pred == labels.to(self.device)).sum().item()
            clean_accuracy = clean_correct / len(labels)
            
            # Adversarial accuracy
            adv_outputs = self.model(adv_images.to(self.device))
            _, adv_pred = torch.max(adv_outputs, 1)
            adv_correct = (adv_pred == labels.to(self.device)).sum().item()
            adv_accuracy = adv_correct / len(labels)
            
            # Attack success rate
            success_rate = 1.0 - adv_accuracy
            
            # Perturbation statistics
            perturbation = (adv_images - clean_images).cpu()
            
            # L∞ norm (maximum absolute change)
            linf_norm = torch.max(torch.abs(perturbation)).item()
            
            # L2 norm (Euclidean distance)
            l2_norms = torch.norm(perturbation.view(len(perturbation), -1), p=2, dim=1)
            l2_norm = l2_norms.mean().item()
            
            # L1 norm (sum of absolute values)
            l1_norms = torch.norm(perturbation.view(len(perturbation), -1), p=1, dim=1)
            l1_norm = l1_norms.mean().item()
        
        metrics = {
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'attack_success_rate': success_rate,
            'avg_linf_perturbation': linf_norm,
            'avg_l2_perturbation': l2_norm,
            'avg_l1_perturbation': l1_norm,
        }
        
        if verbose:
            print("=" * 60)
            print("PGD Attack Evaluation Results")
            print("=" * 60)
            print(f"Configuration:")
            print(f"  Epsilon: {self.epsilon}")
            print(f"  Alpha: {self.alpha}")
            print(f"  Iterations: {self.num_iter}")
            print(f"  Norm: L{self.norm}")
            print(f"\nResults:")
            print(f"  Clean Accuracy: {clean_accuracy:.2%}")
            print(f"  Adversarial Accuracy: {adv_accuracy:.2%}")
            print(f"  Attack Success Rate: {success_rate:.2%}")
            print(f"\nPerturbation Statistics:")
            print(f"  Max L∞: {linf_norm:.6f}")
            print(f"  Avg L2: {l2_norm:.6f}")
            print(f"  Avg L1: {l1_norm:.6f}")
            print("=" * 60)
        
        return metrics


def compare_fgsm_pgd(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    num_iter_list: List[int] = [1, 10, 40, 100],
    device: Optional[torch.device] = None
) -> Dict[int, Dict[str, float]]:
    """
    Compare PGD with different numbers of iterations (including FGSM).
    
    This function demonstrates how PGD strength increases with iterations:
    - 1 iteration = FGSM (baseline)
    - 10 iterations = fast PGD
    - 40 iterations = standard PGD
    - 100 iterations = strong PGD
    
    Parameters:
    -----------
    model : nn.Module
        Model to attack
    images : torch.Tensor
        Clean images
    labels : torch.Tensor
        True labels
    epsilon : float, default=0.03
        Perturbation budget
    num_iter_list : List[int], default=[1, 10, 40, 100]
        List of iteration counts to try
    device : torch.device, optional
        Computation device
    
    Returns:
    --------
    results : Dict[int, Dict[str, float]]
        Results for each iteration count
    """
    if device is None:
        device = next(model.parameters()).device
    
    results = {}
    
    for num_iter in num_iter_list:
        print(f"\n{'='*60}")
        print(f"Testing PGD with {num_iter} iteration(s)")
        print(f"{'='*60}")
        
        # Create PGD attack
        alpha = 2.5 * epsilon / num_iter if num_iter > 1 else epsilon
        attack = PGD(
            model=model,
            epsilon=epsilon,
            alpha=alpha,
            num_iter=num_iter,
            random_init=(num_iter > 1),  # FGSM doesn't use random init
            device=device
        )
        
        # Generate adversarial examples
        adv_images = attack.generate(images, labels)
        
        # Evaluate
        metrics = attack.evaluate(clean_images=images, labels=labels, adv_images=adv_images)
        results[num_iter] = metrics
    
    # Print comparison
    print(f"\n{'='*60}")
    print("PGD Iteration Comparison")
    print(f"{'='*60}")
    print(f"{'Iterations':<12} {'Success Rate':<15} {'Adv Accuracy':<15}")
    print(f"{'-'*60}")
    for num_iter in num_iter_list:
        success = results[num_iter]['attack_success_rate']
        adv_acc = results[num_iter]['adversarial_accuracy']
        print(f"{num_iter:<12} {success:< 15.2%} {adv_acc:<15.2%}")
    print(f"{'='*60}")
    
    return results


# Example usage
if __name__ == "__main__":
    """
    Demonstration of PGD attack.
    
    This example shows:
    1. Basic PGD attack
    2. PGD with multiple restarts
    3. Comparison with different iteration counts
    """
    print("=" * 70)
    print("PGD Attack Demonstration")
    print("=" * 70)
    print("\nThis script demonstrates the Projected Gradient Descent (PGD)")
    print("attack, a strong iterative adversarial attack.")
    print("\nNote: This requires utils.py for data loading and model utilities.")
    print("=" * 70)
