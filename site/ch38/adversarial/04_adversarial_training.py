"""
Module 62.4: Adversarial Training - Advanced Level

This module implements adversarial training, the most effective defense against
adversarial attacks. Adversarial training augments the training set with
adversarial examples, making the model robust to perturbations.

MATHEMATICAL BACKGROUND:
=======================

Standard Training:
-----------------
Standard empirical risk minimization (ERM):
    
    min_θ E_{(x,y)~D}[L(θ, x, y)]

This optimizes for accuracy on the training distribution but doesn't account
for adversarial perturbations.

Adversarial Training (Robust Optimization):
------------------------------------------
Adversarial training solves a min-max optimization problem:

    min_θ E_{(x,y)~D}[ max_{||δ||≤ε} L(θ, x + δ, y) ]

Inner maximization: Find worst-case perturbation (attack)
Outer minimization: Train model to be robust against it

This is a robust optimization problem that seeks to minimize the worst-case loss
within the ε-ball around each training example.

INTERPRETATION:
==============
1. **Inner max**: For each training example, find the adversarial perturbation
   that maximizes the loss (strongest attack within budget ε)
2. **Outer min**: Update model parameters to minimize the worst-case loss
3. **Result**: Model learns to be robust within ε-ball of training data

PRACTICAL IMPLEMENTATION:
=========================
Since the inner maximization is intractable, we approximate it using PGD:

Algorithm (PGD Adversarial Training):
-------------------------------------
For each epoch:
    For each mini-batch (x, y):
        1. Generate adversarial examples:
           x_adv = PGD(x, y, ε, α, K)  [K-step PGD]
        
        2. Compute loss on adversarial examples:
           L_adv = L(θ, x_adv, y)
        
        3. Update parameters:
           θ ← θ - η∇_θ L_adv

This is called "PGD-based adversarial training" or "PGD-AT".

KEY HYPERPARAMETERS:
===================
1. **ε (epsilon)**: Perturbation budget for training
   - Determines the robustness level
   - Common: ε = 8/255 ≈ 0.031 for CIFAR-10
   
2. **K (PGD steps)**: Number of PGD iterations
   - More steps = stronger training-time attack
   - Common: K = 10 (training), K = 20 (evaluation)
   
3. **α (step size)**: PGD step size
   - Typically α = 2ε/K or α = 2.5ε/K

VARIANTS OF ADVERSARIAL TRAINING:
=================================

1. **Standard PGD-AT** (Madry et al., 2018):
   - Use PGD-generated adversarial examples
   - Strongest but computationally expensive

2. **TRADES** (Zhang et al., 2019):
   - Theoretically Principled Trade-off between Robustness and Accuracy
   - Balances clean accuracy and robust accuracy
   - Loss: L_nat + β·KL(f(x)||f(x_adv))
   
3. **MART** (Wang et al., 2020):
   - Misclassification Aware adversarial Training
   - Focuses on misclassified examples
   - Boosted CE on misclassified, KL divergence on correctly classified

4. **Fast AT** (Wong et al., 2020):
   - Single-step FGSM for efficiency
   - Much faster but slightly less robust
   
5. **Free AT** (Shafahi et al., 2019):
   - Reuse gradients from backpropagation
   - Same cost as standard training

ACCURACY-ROBUSTNESS TRADEOFF:
==============================
Adversarial training typically causes a drop in clean accuracy:
- Standard training: ~95% clean accuracy, ~0% robust accuracy
- Adversarial training: ~85% clean accuracy, ~50% robust accuracy

This tradeoff is fundamental and well-documented. TRADES attempts to mitigate
it by explicitly balancing the two objectives.

PRACTICAL CONSIDERATIONS:
========================
1. **Computational cost**: 7-10× slower than standard training (due to PGD)
2. **Overfitting**: Need careful regularization and data augmentation
3. **Catastrophic overfitting**: Loss may suddenly spike; use early stopping
4. **Evaluation**: Must evaluate with strong attacks (PGD, AutoAttack)

Author: Educational Materials
Date: November 2025
Difficulty: Advanced
Prerequisites: PGD (Module 62.2), neural network training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple, Callable
from tqdm import tqdm
import copy


class AdversarialTrainer:
    """
    Adversarial Training for Robust Models
    
    This class implements adversarial training using PGD-generated adversarial
    examples. The model is trained to minimize the worst-case loss within an
    ε-ball around each training example.
    
    Mathematical Formulation:
    -------------------------
    Solves the robust optimization problem:
    
        min_θ E_{(x,y)}[ max_{||δ||≤ε} L(θ, x + δ, y) ]
    
    Approximate inner max using K-step PGD:
    
        x_adv = PGD(x, y, ε, α, K)
        θ ← θ - η∇_θ L(θ, x_adv, y)
    
    Attributes:
    -----------
    model : nn.Module
        Model to train
    epsilon : float
        Perturbation budget
    alpha : float
        PGD step size
    num_iter : int
        Number of PGD iterations
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.031,
        alpha: float = 0.007,
        num_iter: int = 10,
        device: Optional[torch.device] = None,
        loss_fn: Optional[nn.Module] = None,
        norm: str = 'linf'
    ):
        """
        Initialize adversarial trainer.
        
        Parameters:
        -----------
        model : nn.Module
            Model to train robustly
        epsilon : float, default=0.031
            Perturbation budget for adversarial training
            Standard: ε = 8/255 ≈ 0.031 for CIFAR-10
        alpha : float, default=0.007
            Step size for PGD during training
            Typically α = 2ε/K where K is num_iter
        num_iter : int, default=10
            Number of PGD iterations during training
            Fewer iterations for speed, more for strength
            Training: 7-10, Evaluation: 20-100
        device : torch.device, optional
            Computation device
        loss_fn : nn.Module, optional
            Loss function (default: CrossEntropyLoss)
        norm : str, default='linf'
            Norm for perturbation ('linf' or 'l2')
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.norm = norm
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        print(f"Adversarial Training Configuration:")
        print(f"  Epsilon (ε): {self.epsilon}")
        print(f"  Alpha (α): {self.alpha}")
        print(f"  PGD iterations: {self.num_iter}")
        print(f"  Norm: L{self.norm}")
        print(f"  Device: {self.device}")
    
    def _generate_adversarial_examples(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        random_init: bool = True
    ) -> torch.Tensor:
        """
        Generate adversarial examples using PGD.
        
        This is the inner maximization in adversarial training.
        We use PGD to find approximate worst-case perturbations.
        
        Algorithm:
        ----------
        1. Initialize: δ^(0) ~ Uniform[-ε, ε] (random) or δ^(0) = 0
        2. For t = 1 to K:
             δ^(t) = Π_ε(δ^(t-1) + α·sign(∇_δ L(θ, x+δ^(t-1), y)))
        3. Return x + δ^(K)
        
        where Π_ε is projection to ε-ball.
        
        Parameters:
        -----------
        images : torch.Tensor
            Clean images
        labels : torch.Tensor
            True labels
        random_init : bool, default=True
            Whether to use random initialization for PGD
        
        Returns:
        --------
        adv_images : torch.Tensor
            Adversarial examples
        """
        # Set model to evaluation mode for attack generation
        # (BatchNorm/Dropout should be in eval mode during attack)
        self.model.eval()
        
        # Initialize perturbation
        if random_init:
            # Random initialization within ε-ball
            delta = torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
            delta = delta.to(self.device)
        else:
            delta = torch.zeros_like(images).to(self.device)
        
        # Require gradients for delta
        delta.requires_grad = True
        
        # PGD iterations
        for _ in range(self.num_iter):
            # Compute adversarial images
            adv_images = images + delta
            
            # Forward pass
            outputs = self.model(adv_images)
            
            # Compute loss
            loss = self.loss_fn(outputs, labels)
            
            # Backward pass
            self.model.zero_grad()
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()
            
            # Gradient ascent step
            if self.norm == 'linf':
                delta_grad = delta.grad.detach()
                delta = delta + self.alpha * torch.sign(delta_grad)
                # Project to ε-ball
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            else:  # l2
                delta_grad = delta.grad.detach()
                # Normalize gradient
                grad_norm = delta_grad.view(len(delta_grad), -1).norm(p=2, dim=1)
                grad_norm = torch.clamp(grad_norm, min=1e-12)
                normalized_grad = delta_grad / grad_norm.view(-1, 1, 1, 1)
                delta = delta + self.alpha * normalized_grad
                # Project to ε-ball
                delta_norm = delta.view(len(delta), -1).norm(p=2, dim=1)
                scale = torch.clamp(delta_norm / self.epsilon, min=1.0)
                delta = delta / scale.view(-1, 1, 1, 1)
            
            # Clip to valid image range [0, 1]
            delta = torch.clamp(images + delta, 0, 1) - images
            delta = delta.detach()
            delta.requires_grad = True
        
        # Return adversarial examples
        adv_images = images + delta.detach()
        adv_images = torch.clamp(adv_images, 0, 1)
        
        # Set model back to training mode
        self.model.train()
        
        return adv_images
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch using adversarial training.
        
        This implements the outer minimization in adversarial training.
        For each batch:
        1. Generate adversarial examples (inner max)
        2. Compute loss on adversarial examples
        3. Update model parameters (outer min)
        
        Algorithm:
        ----------
        For each batch (x, y):
            # Inner maximization (approximate with PGD)
            x_adv = PGD(x, y, ε, α, K)
            
            # Compute adversarial loss
            L_adv = L(θ, x_adv, y)
            
            # Outer minimization
            θ ← θ - η∇_θ L_adv
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        optimizer : optim.Optimizer
            Optimizer for model parameters
        epoch : int
            Current epoch number
        verbose : bool, default=True
            Print progress
        
        Returns:
        --------
        metrics : Dict[str, float]
            Training metrics (loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        if verbose:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        else:
            pbar = train_loader
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Generate adversarial examples
            adv_images = self._generate_adversarial_examples(images, labels)
            
            # Forward pass on adversarial examples
            outputs = self.model(adv_images)
            loss = self.loss_fn(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            if verbose:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{avg_loss:.3f}',
                    'acc': f'{accuracy:.2f}%'
                })
        
        # Compute epoch metrics
        metrics = {
            'train_loss': total_loss / len(train_loader),
            'train_accuracy': correct / total
        }
        
        return metrics
    
    def evaluate(
        self,
        test_loader: DataLoader,
        attack_epsilon: Optional[float] = None,
        attack_iterations: int = 20,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on clean and adversarial examples.
        
        This provides both clean accuracy and robust accuracy, which are
        the two key metrics for adversarial robustness.
        
        Parameters:
        -----------
        test_loader : DataLoader
            Test data loader
        attack_epsilon : float, optional
            Epsilon for evaluation attack (default: same as training)
        attack_iterations : int, default=20
            Number of PGD iterations for evaluation
            Should be more than training (e.g., 20-100 vs 7-10)
        verbose : bool, default=True
            Print results
        
        Returns:
        --------
        metrics : Dict[str, float]
            Clean accuracy, robust accuracy, etc.
        """
        if attack_epsilon is None:
            attack_epsilon = self.epsilon
        
        self.model.eval()
        
        clean_correct = 0
        robust_correct = 0
        total = 0
        
        if verbose:
            pbar = tqdm(test_loader, desc="Evaluating")
        else:
            pbar = test_loader
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Clean accuracy
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                clean_correct += predicted.eq(labels).sum().item()
                
                total += labels.size(0)
        
        # Generate adversarial examples for robust evaluation
        # Temporarily adjust attack parameters
        original_epsilon = self.epsilon
        original_num_iter = self.num_iter
        self.epsilon = attack_epsilon
        self.num_iter = attack_iterations
        
        if verbose:
            pbar = tqdm(test_loader, desc="Robust evaluation")
        else:
            pbar = test_loader
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Generate adversarial examples
            adv_images = self._generate_adversarial_examples(images, labels)
            
            # Robust accuracy
            with torch.no_grad():
                outputs = self.model(adv_images)
                _, predicted = outputs.max(1)
                robust_correct += predicted.eq(labels).sum().item()
        
        # Restore original parameters
        self.epsilon = original_epsilon
        self.num_iter = original_num_iter
        
        # Compute metrics
        clean_accuracy = clean_correct / total
        robust_accuracy = robust_correct / total
        
        metrics = {
            'clean_accuracy': clean_accuracy,
            'robust_accuracy': robust_accuracy,
            'accuracy_drop': clean_accuracy - robust_accuracy
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print("Evaluation Results")
            print("=" * 60)
            print(f"Clean Accuracy: {clean_accuracy:.2%}")
            print(f"Robust Accuracy (ε={attack_epsilon}): {robust_accuracy:.2%}")
            print(f"Accuracy Drop: {metrics['accuracy_drop']:.2%}")
            print("=" * 60)
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        save_path: Optional[str] = None,
        eval_frequency: int = 1
    ) -> Dict[str, List[float]]:
        """
        Complete adversarial training loop.
        
        This is the main training function that coordinates:
        1. Training for multiple epochs
        2. Periodic evaluation
        3. Model checkpointing
        4. Learning rate scheduling
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data
        test_loader : DataLoader
            Test data
        epochs : int
            Number of training epochs
        optimizer : optim.Optimizer, optional
            Optimizer (default: SGD with momentum)
        scheduler : lr_scheduler, optional
            Learning rate scheduler
        save_path : str, optional
            Path to save best model
        eval_frequency : int, default=1
            Evaluate every N epochs
        
        Returns:
        --------
        history : Dict[str, List[float]]
            Training history (losses, accuracies)
        """
        # Default optimizer: SGD with momentum
        if optimizer is None:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=0.1,
                momentum=0.9,
                weight_decay=5e-4
            )
        
        # Default scheduler: Multi-step LR decay
        if scheduler is None:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(0.5*epochs), int(0.75*epochs)],
                gamma=0.1
            )
        
        # Track training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'clean_accuracy': [],
            'robust_accuracy': []
        }
        
        best_robust_acc = 0.0
        
        print(f"\nStarting adversarial training for {epochs} epochs")
        print(f"Training with ε={self.epsilon}, PGD steps={self.num_iter}")
        print("=" * 60)
        
        for epoch in range(1, epochs + 1):
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, optimizer, epoch)
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_accuracy'].append(train_metrics['train_accuracy'])
            
            # Evaluate
            if epoch % eval_frequency == 0:
                eval_metrics = self.evaluate(test_loader, verbose=False)
                history['clean_accuracy'].append(eval_metrics['clean_accuracy'])
                history['robust_accuracy'].append(eval_metrics['robust_accuracy'])
                
                print(f"Epoch {epoch}/{epochs}:")
                print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
                print(f"  Train Acc: {train_metrics['train_accuracy']:.2%}")
                print(f"  Clean Acc: {eval_metrics['clean_accuracy']:.2%}")
                print(f"  Robust Acc: {eval_metrics['robust_accuracy']:.2%}")
                
                # Save best model
                if save_path and eval_metrics['robust_accuracy'] > best_robust_acc:
                    best_robust_acc = eval_metrics['robust_accuracy']
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  → Saved best model (robust acc: {best_robust_acc:.2%})")
            
            # Update learning rate
            scheduler.step()
            
            print("-" * 60)
        
        print(f"\nTraining complete!")
        print(f"Best robust accuracy: {best_robust_acc:.2%}")
        
        return history


class TRADESTrainer(AdversarialTrainer):
    """
    TRADES: Trade-off between Robustness and Accuracy
    
    TRADES explicitly balances clean accuracy and robust accuracy by using
    a combination of natural loss and KL divergence.
    
    Loss Function:
    --------------
    L_TRADES = L_CE(f(x), y) + β·KL(f(x) || f(x_adv))
    
    where:
    - L_CE is standard cross-entropy on clean examples
    - KL is KL divergence between clean and adversarial predictions
    - β controls the trade-off
    
    The KL term encourages predictions to be similar on clean and adversarial
    examples, promoting local smoothness.
    
    Reference: Zhang et al., "Theoretically Principled Trade-off between
               Robustness and Accuracy" (ICML 2019)
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.031,
        alpha: float = 0.007,
        num_iter: int = 10,
        beta: float = 6.0,
        device: Optional[torch.device] = None,
        norm: str = 'linf'
    ):
        """
        Initialize TRADES trainer.
        
        Parameters:
        -----------
        beta : float, default=6.0
            Trade-off parameter between natural and robust loss
            Larger β: more emphasis on robustness
            Smaller β: more emphasis on clean accuracy
        """
        super().__init__(model, epsilon, alpha, num_iter, device, None, norm)
        self.beta = beta
        print(f"  TRADES β: {self.beta}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        TRADES training for one epoch.
        
        Loss: L_natural + β·KL(f(x) || f(x_adv))
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        if verbose:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} (TRADES)")
        else:
            pbar = train_loader
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward on clean examples
            logits_clean = self.model(images)
            loss_natural = F.cross_entropy(logits_clean, labels)
            
            # Generate adversarial examples
            adv_images = self._generate_adversarial_examples(images, labels)
            
            # Forward on adversarial examples
            logits_adv = self.model(adv_images)
            
            # KL divergence between clean and adversarial predictions
            # KL(P||Q) where P=f(x), Q=f(x_adv)
            loss_robust = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_clean, dim=1),
                reduction='batchmean'
            )
            
            # Total TRADES loss
            loss = loss_natural + self.beta * loss_robust
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            total_loss += loss.item()
            _, predicted = logits_clean.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if verbose:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{avg_loss:.3f}',
                    'acc': f'{accuracy:.2f}%'
                })
        
        metrics = {
            'train_loss': total_loss / len(train_loader),
            'train_accuracy': correct / total
        }
        
        return metrics


# Example usage
if __name__ == "__main__":
    """
    Demonstration of adversarial training.
    """
    print("=" * 70)
    print("Adversarial Training Demonstration")
    print("=" * 70)
    print("\nThis script demonstrates adversarial training for robust models.")
    print("\nNote: This requires utils.py for data loading and model utilities.")
    print("=" * 70)
