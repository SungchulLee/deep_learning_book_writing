"""
Module 62.1: Fast Gradient Sign Method (FGSM) - Beginner Level

This module implements the Fast Gradient Sign Method (FGSM), one of the simplest
and most influential adversarial attack methods. FGSM demonstrates that neural
networks are vulnerable to small, carefully crafted perturbations.

MATHEMATICAL BACKGROUND:
=======================

The FGSM attack is based on a linearization of the loss function around the
current input. Given a neural network f with parameters θ, input x, and true
label y, we want to find a perturbation δ that maximizes the loss:

    maximize L(θ, x + δ, y)  subject to ||δ||_∞ ≤ ε

FGSM uses a first-order Taylor approximation of the loss:
    
    L(θ, x + δ, y) ≈ L(θ, x, y) + δ^T ∇_x L(θ, x, y)

To maximize this with ||δ||_∞ ≤ ε, we use:
    
    δ = ε · sign(∇_x L(θ, x, y))

This gives the adversarial example:
    
    x_adv = x + ε · sign(∇_x L(θ, x, y))

INTUITION:
==========
- The gradient ∇_x L tells us how to change x to increase the loss
- Taking the sign gives us just the direction (+ or -)
- Multiplying by ε gives us the maximum allowed perturbation in each dimension
- This is a single-step attack that moves in the "steepest ascent" direction

KEY PROPERTIES:
===============
1. Computationally efficient: O(1) - just one gradient computation
2. Generates adversarial examples in a single step
3. Often sufficient to fool many models
4. Serves as a starting point for stronger attacks

Author: Educational Materials
Date: November 2025
Difficulty: Beginner
Prerequisites: PyTorch basics, backpropagation, CNNs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm


class FGSM:
    """
    Fast Gradient Sign Method (FGSM) Attack
    
    This class implements the FGSM attack, which generates adversarial examples
    by taking a single gradient step in the direction that maximizes the loss.
    
    Mathematical Formulation:
    -------------------------
    Given:
        - Model f(x; θ) with parameters θ
        - Input x with true label y
        - Loss function L(θ, x, y)
        - Perturbation budget ε
    
    The FGSM attack computes:
        x_adv = x + ε · sign(∇_x L(θ, x, y))
    
    where:
        - ∇_x L is the gradient of loss w.r.t. input
        - sign(·) returns +1, 0, or -1 for each element
        - ε controls the perturbation magnitude
    
    Attributes:
    -----------
    model : nn.Module
        The neural network to attack
    epsilon : float
        Maximum perturbation magnitude (L∞ norm)
    loss_fn : nn.Module
        Loss function to maximize (default: CrossEntropyLoss)
    device : torch.device
        Device for computation (CPU or GPU)
    clip_min : float
        Minimum valid pixel value (default: 0.0)
    clip_max : float
        Maximum valid pixel value (default: 1.0)
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.3,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0
    ):
        """
        Initialize the FGSM attack.
        
        Parameters:
        -----------
        model : nn.Module
            The neural network to attack. Should be in evaluation mode.
        epsilon : float, default=0.3
            Maximum L∞ perturbation. For images normalized to [0,1],
            ε=0.3 means each pixel can change by at most 0.3.
            Common values: 0.03 (subtle), 0.1 (moderate), 0.3 (strong)
        loss_fn : nn.Module, optional
            Loss function to use. If None, uses CrossEntropyLoss.
        device : torch.device, optional
            Device for computation. If None, uses model's device.
        clip_min : float, default=0.0
            Minimum value for output (usually 0 for images)
        clip_max : float, default=1.0
            Maximum value for output (usually 1 for normalized images)
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.device = device if device is not None else next(model.parameters()).device
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        # Set model to evaluation mode (important for batchnorm, dropout)
        self.model.eval()
        
        # Move model to device if needed
        self.model = self.model.to(self.device)
        
    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate adversarial examples using FGSM.
        
        This method implements the core FGSM algorithm:
        1. Compute the loss for the original inputs
        2. Compute gradient of loss w.r.t. inputs
        3. Generate perturbation: δ = ε · sign(∇_x L)
        4. Create adversarial example: x_adv = x + δ
        5. Clip to valid range [clip_min, clip_max]
        
        Parameters:
        -----------
        images : torch.Tensor
            Clean images of shape (batch_size, channels, height, width)
            Should be in range [clip_min, clip_max]
        labels : torch.Tensor
            True labels of shape (batch_size,)
        targeted : bool, default=False
            If True, generate examples to be classified as target_labels
            If False, generate examples to be misclassified (any wrong class)
        target_labels : torch.Tensor, optional
            Target labels for targeted attack. Required if targeted=True.
        
        Returns:
        --------
        adv_images : torch.Tensor
            Adversarial images of same shape as input
            Guaranteed to satisfy ||x_adv - x||_∞ ≤ ε
        
        Mathematical Details:
        ---------------------
        For untargeted attack (targeted=False):
            We maximize loss: δ = ε · sign(∇_x L(f(x), y))
            This pushes the prediction away from the true label
        
        For targeted attack (targeted=True):
            We minimize loss: δ = -ε · sign(∇_x L(f(x), y_target))
            This pulls the prediction toward the target label
        """
        # Move inputs to device
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Create a copy for perturbation (detach from computation graph)
        # We need to track gradients w.r.t. images, not model parameters
        images_adv = images.clone().detach()
        
        # Enable gradient computation for images
        # This is crucial: we need ∇_x L, not ∇_θ L
        images_adv.requires_grad = True
        
        # Forward pass: compute model predictions
        outputs = self.model(images_adv)
        
        # Compute loss
        # For targeted attack, use target_labels; for untargeted, use true labels
        if targeted:
            if target_labels is None:
                raise ValueError("target_labels must be provided for targeted attack")
            target_labels = target_labels.to(self.device)
            loss = self.loss_fn(outputs, target_labels)
        else:
            loss = self.loss_fn(outputs, labels)
        
        # Backward pass: compute gradient of loss w.r.t. input images
        # This gives us ∇_x L(θ, x, y)
        self.model.zero_grad()  # Clear any existing gradients
        if images_adv.grad is not None:
            images_adv.grad.zero_()
        loss.backward()
        
        # Extract gradients
        # images_adv.grad has shape (batch_size, channels, height, width)
        grad = images_adv.grad.data
        
        # Generate perturbation using the sign of gradients
        # For untargeted attack: δ = ε · sign(∇_x L)
        # For targeted attack: δ = -ε · sign(∇_x L) (minimize loss toward target)
        if targeted:
            perturbation = -self.epsilon * torch.sign(grad)
        else:
            perturbation = self.epsilon * torch.sign(grad)
        
        # Create adversarial examples
        # x_adv = x + δ
        images_adv = images + perturbation
        
        # Clip to valid range to ensure adversarial examples are valid images
        # This is important: we want x_adv ∈ [clip_min, clip_max]
        images_adv = torch.clamp(images_adv, self.clip_min, self.clip_max)
        
        # Detach from computation graph (no longer need gradients)
        images_adv = images_adv.detach()
        
        return images_adv
    
    def generate_with_budget_search(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        epsilon_values: List[float],
        target_success_rate: float = 0.9
    ) -> Tuple[torch.Tensor, float]:
        """
        Find the smallest ε that achieves target attack success rate.
        
        This method performs a search over different epsilon values to find
        the minimal perturbation needed to achieve a desired success rate.
        
        Algorithm:
        ----------
        1. Try each epsilon value in ascending order
        2. Generate adversarial examples for each epsilon
        3. Compute attack success rate
        4. Return first epsilon that achieves target_success_rate
        
        This is useful for understanding the model's robustness: smaller
        epsilon means the model is more vulnerable.
        
        Parameters:
        -----------
        images : torch.Tensor
            Clean images
        labels : torch.Tensor
            True labels
        epsilon_values : List[float]
            List of epsilon values to try (should be sorted ascending)
        target_success_rate : float, default=0.9
            Desired attack success rate (0 to 1)
        
        Returns:
        --------
        best_adv_images : torch.Tensor
            Adversarial images with smallest successful epsilon
        best_epsilon : float
            Smallest epsilon achieving target success rate
        """
        best_epsilon = None
        best_adv_images = None
        
        for eps in epsilon_values:
            # Temporarily set epsilon
            original_epsilon = self.epsilon
            self.epsilon = eps
            
            # Generate adversarial examples
            adv_images = self.generate(images, labels)
            
            # Evaluate success rate
            success_rate = self.compute_success_rate(images, labels, adv_images)
            
            # Restore original epsilon
            self.epsilon = original_epsilon
            
            # Check if we achieved target success rate
            if success_rate >= target_success_rate:
                best_epsilon = eps
                best_adv_images = adv_images
                break
        
        if best_epsilon is None:
            print(f"Warning: Could not achieve success rate {target_success_rate}")
            print(f"Using largest epsilon: {epsilon_values[-1]}")
            self.epsilon = epsilon_values[-1]
            best_adv_images = self.generate(images, labels)
            best_epsilon = epsilon_values[-1]
            self.epsilon = original_epsilon
        
        return best_adv_images, best_epsilon
    
    def compute_success_rate(
        self,
        clean_images: torch.Tensor,
        labels: torch.Tensor,
        adv_images: torch.Tensor
    ) -> float:
        """
        Compute the attack success rate.
        
        Success rate is the fraction of adversarial examples that are
        misclassified (for untargeted attack) or classified as target
        (for targeted attack).
        
        Mathematical Definition:
        ------------------------
        Success rate = (1/n) * Σ I[f(x_adv) ≠ y]
        
        where:
        - n is the number of examples
        - I[·] is the indicator function (1 if true, 0 if false)
        - f(x_adv) is the model's prediction on adversarial example
        - y is the true label
        
        Parameters:
        -----------
        clean_images : torch.Tensor
            Original clean images (not used, kept for API consistency)
        labels : torch.Tensor
            True labels
        adv_images : torch.Tensor
            Adversarial images
        
        Returns:
        --------
        success_rate : float
            Fraction of successful attacks (0 to 1)
        """
        with torch.no_grad():  # No gradients needed for evaluation
            # Get predictions on adversarial examples
            outputs = self.model(adv_images.to(self.device))
            _, predicted = torch.max(outputs, 1)
            
            # Count misclassifications
            # Success means prediction != true label
            successful_attacks = (predicted != labels.to(self.device)).sum().item()
            
            # Compute success rate
            success_rate = successful_attacks / len(labels)
        
        return success_rate
    
    def evaluate(
        self,
        clean_images: torch.Tensor,
        labels: torch.Tensor,
        adv_images: torch.Tensor,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of attack effectiveness.
        
        This method computes various metrics to assess the attack:
        1. Clean accuracy: accuracy on original images
        2. Adversarial accuracy: accuracy on perturbed images
        3. Attack success rate: fraction of successful attacks
        4. Average L∞ perturbation: maximum absolute change
        5. Average L2 perturbation: Euclidean distance
        
        Parameters:
        -----------
        clean_images : torch.Tensor
            Original clean images
        labels : torch.Tensor
            True labels
        adv_images : torch.Tensor
            Adversarial images
        verbose : bool, default=True
            If True, print results
        
        Returns:
        --------
        metrics : Dict[str, float]
            Dictionary containing all computed metrics
        """
        with torch.no_grad():
            # Evaluate on clean images
            clean_outputs = self.model(clean_images.to(self.device))
            _, clean_pred = torch.max(clean_outputs, 1)
            clean_correct = (clean_pred == labels.to(self.device)).sum().item()
            clean_accuracy = clean_correct / len(labels)
            
            # Evaluate on adversarial images
            adv_outputs = self.model(adv_images.to(self.device))
            _, adv_pred = torch.max(adv_outputs, 1)
            adv_correct = (adv_pred == labels.to(self.device)).sum().item()
            adv_accuracy = adv_correct / len(labels)
            
            # Compute attack success rate
            success_rate = 1.0 - adv_accuracy
            
            # Compute perturbation statistics
            perturbation = (adv_images - clean_images).cpu()
            
            # L∞ norm: maximum absolute change across all pixels
            linf_norm = torch.max(torch.abs(perturbation)).item()
            
            # L2 norm: Euclidean distance
            # We compute per-example L2 norm, then average
            l2_norms = torch.norm(perturbation.view(len(perturbation), -1), p=2, dim=1)
            l2_norm = l2_norms.mean().item()
            
            # L0 norm: number of changed pixels (informally)
            # Count pixels where |change| > threshold
            threshold = 1e-5
            l0_norm = (torch.abs(perturbation) > threshold).sum().item() / len(labels)
        
        # Compile metrics
        metrics = {
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'attack_success_rate': success_rate,
            'avg_linf_perturbation': linf_norm,
            'avg_l2_perturbation': l2_norm,
            'avg_l0_perturbation': l0_norm,
        }
        
        if verbose:
            print("=" * 60)
            print("FGSM Attack Evaluation Results")
            print("=" * 60)
            print(f"Epsilon (ε): {self.epsilon}")
            print(f"Clean Accuracy: {clean_accuracy:.2%}")
            print(f"Adversarial Accuracy: {adv_accuracy:.2%}")
            print(f"Attack Success Rate: {success_rate:.2%}")
            print(f"Avg L∞ Perturbation: {linf_norm:.6f}")
            print(f"Avg L2 Perturbation: {l2_norm:.6f}")
            print(f"Avg L0 Perturbation: {l0_norm:.2f} pixels")
            print("=" * 60)
        
        return metrics


def visualize_attack(
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: Optional[List[str]] = None,
    num_examples: int = 5,
    epsilon: float = 0.3,
    save_path: Optional[str] = None
):
    """
    Visualize adversarial examples and perturbations.
    
    This function creates a visualization showing:
    - Original clean images with true labels
    - Adversarial images with predicted labels
    - Magnified perturbations (amplified for visibility)
    
    The visualization helps understand what FGSM is doing to the images.
    
    Parameters:
    -----------
    clean_images : torch.Tensor
        Original images of shape (batch_size, C, H, W)
    adv_images : torch.Tensor
        Adversarial images
    labels : torch.Tensor
        True labels
    predictions : torch.Tensor
        Model predictions on adversarial images
    class_names : List[str], optional
        Names of classes for labeling
    num_examples : int, default=5
        Number of examples to visualize
    epsilon : float, default=0.3
        Epsilon value used (for title)
    save_path : str, optional
        If provided, save figure to this path
    """
    # Convert tensors to numpy for matplotlib
    clean_np = clean_images[:num_examples].cpu().numpy()
    adv_np = adv_images[:num_examples].cpu().numpy()
    labels_np = labels[:num_examples].cpu().numpy()
    pred_np = predictions[:num_examples].cpu().numpy()
    
    # Compute perturbations
    perturbations = adv_np - clean_np
    
    # Create figure with 3 rows: clean, adversarial, perturbation (magnified)
    fig, axes = plt.subplots(3, num_examples, figsize=(3*num_examples, 9))
    
    if num_examples == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(num_examples):
        # Transpose from (C, H, W) to (H, W, C) for matplotlib
        # Handle both grayscale and RGB
        if clean_np.shape[1] == 1:  # Grayscale
            clean_img = clean_np[i, 0]
            adv_img = adv_np[i, 0]
            pert_img = perturbations[i, 0]
            cmap = 'gray'
        else:  # RGB
            clean_img = np.transpose(clean_np[i], (1, 2, 0))
            adv_img = np.transpose(adv_np[i], (1, 2, 0))
            pert_img = np.transpose(perturbations[i], (1, 2, 0))
            cmap = None
        
        # Row 1: Clean images
        axes[0, i].imshow(clean_img, cmap=cmap)
        true_label = class_names[labels_np[i]] if class_names else labels_np[i]
        axes[0, i].set_title(f'Clean\nTrue: {true_label}', fontsize=10)
        axes[0, i].axis('off')
        
        # Row 2: Adversarial images
        axes[1, i].imshow(adv_img, cmap=cmap)
        pred_label = class_names[pred_np[i]] if class_names else pred_np[i]
        color = 'red' if pred_np[i] != labels_np[i] else 'green'
        axes[1, i].set_title(f'Adversarial\nPred: {pred_label}', 
                            fontsize=10, color=color)
        axes[1, i].axis('off')
        
        # Row 3: Perturbations (magnified by 10x for visibility)
        # Normalize perturbation for better visualization
        pert_magnified = pert_img * 10
        pert_magnified = np.clip(pert_magnified + 0.5, 0, 1)  # Center around 0.5
        
        axes[2, i].imshow(pert_magnified, cmap='RdBu_r')  # Red-Blue colormap
        axes[2, i].set_title(f'Perturbation\n(10× magnified)', fontsize=10)
        axes[2, i].axis('off')
    
    plt.suptitle(f'FGSM Attack Visualization (ε = {epsilon})', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


# Example usage and demonstrations
if __name__ == "__main__":
    """
    Demonstration of FGSM attack on CIFAR-10 dataset.
    
    This example shows:
    1. Loading a pretrained model
    2. Creating FGSM attack
    3. Generating adversarial examples
    4. Evaluating attack effectiveness
    5. Visualizing results
    """
    print("=" * 70)
    print("FGSM Attack Demonstration")
    print("=" * 70)
    print("\nThis script demonstrates the Fast Gradient Sign Method (FGSM)")
    print("attack on the CIFAR-10 dataset using a pretrained ResNet-18 model.")
    print("\nNote: This requires utils.py for data loading and model utilities.")
    print("=" * 70)
