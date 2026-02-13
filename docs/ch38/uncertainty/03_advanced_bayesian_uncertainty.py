"""
Module 63.3: Bayesian Neural Networks & Advanced Uncertainty (Advanced Level)

This script covers advanced Bayesian approaches to uncertainty quantification,
including Bayesian Neural Networks (BNNs), variational inference, and modern
practical approximations like SWAG.

Topics:
    1. Bayesian Neural Networks fundamentals
    2. Bayes by Backprop algorithm
    3. Variational Inference for BNNs
    4. Stochastic Weight Averaging Gaussian (SWAG)
    5. Laplace Approximation
    6. Comparison with other methods

Mathematical Background:
    
    Bayesian Framework:
        Prior: p(w)
        Likelihood: p(D|w) = ∏ p(y_i|x_i,w)
        Posterior: p(w|D) = p(D|w)p(w) / p(D)  [intractable]
        Predictive: p(y*|x*,D) = ∫ p(y*|x*,w) p(w|D) dw
    
    Variational Inference:
        Goal: Approximate p(w|D) with simpler q(w|θ)
        Minimize: KL(q(w|θ) || p(w|D))
        ELBO: log p(D) ≥ E_q[log p(D|w)] - KL(q(w|θ) || p(w))
    
    Bayes by Backprop (Blundell et al., 2015):
        q(w|θ) = N(μ, σ²)  (Gaussian posterior)
        θ = {μ, σ} learned via SGD
        Loss: -E_q[log p(D|w)] + KL(q(w|θ) || p(w))
    
    SWAG (Maddox et al., 2019):
        Track mean and covariance during SGD
        Approximate posterior as Gaussian
        Very practical and effective

Learning Objectives:
    - Understand Bayesian deep learning principles
    - Implement Bayes by Backprop
    - Apply SWAG for uncertainty
    - Compare Bayesian vs frequentist approaches
    - Recognize computational trade-offs

Prerequisites:
    - Module 63.1-63.2: Basic and Intermediate Uncertainty
    - Module 45: Bayesian Neural Networks (theory)
    - Strong understanding of probability theory
    - Variational inference basics

Time: 4-5 hours
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from collections import defaultdict
from tqdm import tqdm
import math


# ============================================================================
# PART 1: BAYESIAN NEURAL NETWORK WITH BAYES BY BACKPROP
# ============================================================================

class GaussianPrior:
    """
    Gaussian prior distribution: p(w) = N(0, σ²)
    
    Used as prior over network weights in Bayesian NNs.
    Typically σ² = 1 (standard normal prior).
    """
    
    def __init__(self, sigma: float = 1.0):
        """
        Initialize Gaussian prior.
        
        Args:
            sigma: Standard deviation of prior
        """
        self.sigma = sigma
        self.sigma2 = sigma ** 2
    
    def log_prob(self, w: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability: log p(w) = -w²/(2σ²) - log(σ√(2π))
        
        Args:
            w: Weight tensor
        
        Returns:
            Log probability
        """
        log_prob = -0.5 * (w ** 2) / self.sigma2 - 0.5 * math.log(2 * math.pi * self.sigma2)
        return log_prob.sum()


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with learnable weight distributions.
    
    Instead of deterministic weights W, we have:
        W ~ q(W|μ, σ) = N(μ, σ²)
    
    where μ and σ are learned parameters.
    
    Forward pass:
        1. Sample W from q(W|μ, σ): W = μ + σ ⊙ ε, where ε ~ N(0, I)
        2. Compute output: y = Wx + b
    
    This is the core of Bayes by Backprop (Blundell et al., 2015).
    """
    
    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        """
        Initialize Bayesian linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension  
            prior_sigma: Standard deviation of weight prior
        """
        super(BayesianLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Variational parameters for weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Variational parameters for bias
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # Prior
        self.prior = GaussianPrior(prior_sigma)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialize variational parameters.
        
        μ ~ U(-√(1/n), √(1/n))  (similar to standard initialization)
        ρ ~ U(-3, -5)  (ensures σ starts small)
        """
        stdv = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.bias_mu.data.uniform_(-stdv, stdv)
        
        # Initialize rho to give small initial sigma
        self.weight_rho.data.uniform_(-5, -4)
        self.bias_rho.data.uniform_(-5, -4)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Forward pass with weight sampling.
        
        Reparameterization trick:
            σ = log(1 + exp(ρ))  (softplus ensures σ > 0)
            W = μ + σ ⊙ ε, where ε ~ N(0, I)
        
        Args:
            x: Input tensor (batch_size, in_features)
            sample: If True, sample weights; if False, use mean weights
        
        Returns:
            Output tensor (batch_size, out_features)
        """
        if sample:
            # Compute sigma using softplus: σ = log(1 + exp(ρ))
            weight_sigma = F.softplus(self.weight_rho)
            bias_sigma = F.softplus(self.bias_rho)
            
            # Sample weights using reparameterization trick
            # W = μ + σ ⊙ ε, where ε ~ N(0, 1)
            weight_epsilon = torch.randn_like(weight_sigma)
            bias_epsilon = torch.randn_like(bias_sigma)
            
            weight = self.weight_mu + weight_sigma * weight_epsilon
            bias = self.bias_mu + bias_sigma * bias_epsilon
        else:
            # Use mean weights (no sampling)
            weight = self.weight_mu
            bias = self.bias_mu
        
        # Standard linear transformation
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence: KL(q(w|θ) || p(w))
        
        For Gaussian q and p:
            KL(N(μ_q, σ²_q) || N(μ_p, σ²_p)) = 
                log(σ_p/σ_q) + (σ²_q + (μ_q - μ_p)²)/(2σ²_p) - 1/2
        
        Returns:
            KL divergence (scalar)
        """
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        
        # KL for weights
        kl_weight = self._kl_gaussian(
            self.weight_mu, weight_sigma,
            0.0, self.prior.sigma
        )
        
        # KL for biases
        kl_bias = self._kl_gaussian(
            self.bias_mu, bias_sigma,
            0.0, self.prior.sigma
        )
        
        return kl_weight + kl_bias
    
    def _kl_gaussian(self, mu_q: torch.Tensor, sigma_q: torch.Tensor,
                     mu_p: float, sigma_p: float) -> torch.Tensor:
        """
        KL divergence between two Gaussians.
        
        Args:
            mu_q, sigma_q: Variational posterior parameters
            mu_p, sigma_p: Prior parameters
        
        Returns:
            KL divergence
        """
        kl = torch.log(sigma_p / sigma_q) + \
             (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2 * sigma_p ** 2) - 0.5
        
        return kl.sum()


class BayesianNN(nn.Module):
    """
    Full Bayesian Neural Network using Bayes by Backprop.
    
    Architecture:
        - Bayesian linear layers with weight distributions
        - ReLU activations
        - Classification output
    
    Training Loss (ELBO):
        L = -E_q[log p(D|w)] + (1/N) * KL(q(w|θ) || p(w))
    
    where N is number of data points (for proper scaling).
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, prior_sigma: float = 1.0):
        """
        Initialize Bayesian NN.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            prior_sigma: Prior standard deviation
        """
        super(BayesianNN, self).__init__()
        
        # Build Bayesian layers
        self.layers = nn.ModuleList()
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(BayesianLinear(prev_dim, hidden_dim, prior_sigma))
            prev_dim = hidden_dim
        
        # Output layer
        self.layers.append(BayesianLinear(prev_dim, output_dim, prior_sigma))
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Forward pass through Bayesian network.
        
        Args:
            x: Input tensor
            sample: If True, sample weights; if False, use mean
        
        Returns:
            Output logits
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x, sample=sample))
        
        # Output layer (no activation)
        x = self.layers[-1](x, sample=sample)
        
        return x
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Total KL divergence across all layers.
        
        Returns:
            Sum of KL divergences
        """
        kl_sum = 0.0
        for layer in self.layers:
            kl_sum += layer.kl_divergence()
        return kl_sum
    
    def predict_with_uncertainty(self, x: torch.Tensor, 
                                 n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bayesian prediction with uncertainty.
        
        Sample multiple weight configurations and aggregate predictions.
        This gives us the predictive distribution: p(y|x,D)
        
        Args:
            x: Input tensor
            n_samples: Number of posterior samples
        
        Returns:
            mean_probs: Mean predicted probabilities
            epistemic_uncertainty: Epistemic uncertainty
        """
        self.eval()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Sample weights from posterior
                logits = self.forward(x, sample=True)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Compute statistics
        mean_probs = torch.mean(predictions, dim=0)
        variance = torch.var(predictions, dim=0)
        epistemic_uncertainty = torch.mean(variance, dim=1)
        
        return mean_probs, epistemic_uncertainty


def train_bayesian_nn(model: BayesianNN, train_loader: DataLoader,
                      epochs: int = 10, lr: float = 0.001,
                      n_train: int = None) -> List[float]:
    """
    Train Bayesian NN using Bayes by Backprop.
    
    Loss function (negative ELBO):
        L = -log p(y|X,w) + (1/N) * KL(q(w|θ) || p(w))
    
    The KL term is scaled by 1/N to balance with log-likelihood.
    
    Args:
        model: Bayesian neural network
        train_loader: Training data loader
        epochs: Number of epochs
        lr: Learning rate
        n_train: Number of training samples (for KL scaling)
    
    Returns:
        List of training losses
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Determine number of training samples
    if n_train is None:
        n_train = len(train_loader.dataset)
    
    losses = []
    
    print(f"\nTraining Bayesian NN for {epochs} epochs...")
    print(f"Dataset size: {n_train}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_x, batch_y in pbar:
            # Flatten if needed
            if len(batch_x.shape) > 2:
                batch_x = batch_x.view(batch_x.size(0), -1)
            
            # Forward pass (sample weights)
            logits = model(batch_x, sample=True)
            
            # Negative log-likelihood (data fit term)
            nll = F.cross_entropy(logits, batch_y)
            
            # KL divergence (complexity penalty, scaled by dataset size)
            kl = model.kl_divergence() / n_train
            
            # Total loss (negative ELBO)
            loss = nll + kl
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'nll': f'{nll.item():.4f}',
                'kl': f'{kl.item():.4f}'
            })
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
    
    return losses


def demonstrate_bayesian_nn():
    """
    Demonstrate Bayesian Neural Network on MNIST.
    """
    print("=" * 70)
    print("PART 1: Bayesian Neural Networks (Bayes by Backprop)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Load MNIST
    print("\nLoading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Use subset for faster training
    train_subset, _ = random_split(train_dataset, [5000, 55000])
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    
    print(f"Training samples: {len(train_subset)}")
    
    # Create Bayesian NN
    print("\nCreating Bayesian Neural Network...")
    bnn = BayesianNN(
        input_dim=28*28,
        hidden_dims=[256, 128],
        output_dim=10,
        prior_sigma=1.0
    )
    
    print(f"Model parameters: {sum(p.numel() for p in bnn.parameters()):,}")
    print("Each weight is represented by (μ, ρ) → distribution")
    
    # Train
    losses = train_bayesian_nn(
        bnn, train_loader, epochs=10, lr=0.001, n_train=len(train_subset)
    )
    
    # Test uncertainty estimation
    print("\n" + "=" * 70)
    print("Bayesian Uncertainty Estimation")
    print("=" * 70)
    
    # Get test batch
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_subset, _ = random_split(test_dataset, [100, 9900])
    test_loader = DataLoader(test_subset, batch_size=100)
    
    test_x, test_y = next(iter(test_loader))
    test_x_flat = test_x.view(test_x.size(0), -1)
    
    # Get predictions
    mean_probs, uncertainty = bnn.predict_with_uncertainty(test_x_flat, n_samples=100)
    predictions = torch.argmax(mean_probs, dim=1)
    
    # Analyze
    correct = (predictions == test_y).numpy()
    uncertainty = uncertainty.numpy()
    
    print(f"Accuracy: {correct.mean() * 100:.2f}%")
    print(f"Avg uncertainty (correct): {uncertainty[correct].mean():.6f}")
    print(f"Avg uncertainty (incorrect): {uncertainty[~correct].mean():.6f}")
    
    return bnn, losses


# ============================================================================
# PART 2: STOCHASTIC WEIGHT AVERAGING GAUSSIAN (SWAG)
# ============================================================================

class SWAGModel:
    """
    Stochastic Weight Averaging Gaussian (Maddox et al., 2019).
    
    Key Idea:
        - Run SGD and collect weight snapshots
        - Fit Gaussian distribution to snapshots
        - Use for uncertainty estimation
    
    Algorithm:
        1. Train model with SGD
        2. After convergence, collect weight snapshots every K iterations
        3. Compute mean: μ = (1/T) Σ w_t
        4. Compute covariance (low-rank approximation)
        5. Sample from N(μ, Σ) for predictions
    
    Advantages:
        - Very simple to implement
        - Works with standard training
        - No modification to architecture
        - Computationally efficient
    """
    
    def __init__(self, base_model: nn.Module, max_num_models: int = 20):
        """
        Initialize SWAG.
        
        Args:
            base_model: Base neural network
            max_num_models: Maximum number of models to store
        """
        self.base_model = base_model
        self.max_num_models = max_num_models
        
        # Storage for weight statistics
        self.mean = {}
        self.sq_mean = {}
        self.cov_mat_sqrt = []
        
        self.n_models = 0
    
    def collect_model(self, model: nn.Module):
        """
        Collect weight snapshot from current model.
        
        Updates running statistics for mean and covariance.
        
        Args:
            model: Current model with weights to collect
        """
        # Get current weights as vector
        w = self._flatten_params(model)
        
        if self.n_models == 0:
            # Initialize
            self.mean = {name: param.data.clone() 
                        for name, param in model.named_parameters()}
            self.sq_mean = {name: param.data.clone() ** 2 
                           for name, param in model.named_parameters()}
        else:
            # Update running mean
            for name, param in model.named_parameters():
                self.mean[name] = (self.mean[name] * self.n_models + param.data) / (self.n_models + 1)
                self.sq_mean[name] = (self.sq_mean[name] * self.n_models + param.data ** 2) / (self.n_models + 1)
        
        # Update deviation matrix (for low-rank covariance)
        if self.n_models < self.max_num_models:
            dev = []
            for name, param in model.named_parameters():
                dev.append((param.data - self.mean[name]).view(-1))
            self.cov_mat_sqrt.append(torch.cat(dev))
        
        self.n_models += 1
    
    def _flatten_params(self, model: nn.Module) -> torch.Tensor:
        """
        Flatten all model parameters into single vector.
        
        Args:
            model: Neural network
        
        Returns:
            Flattened parameter vector
        """
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)
    
    def _unflatten_params(self, vector: torch.Tensor, model: nn.Module):
        """
        Unflatten vector back into model parameters.
        
        Args:
            vector: Flattened parameter vector
            model: Model to load parameters into
        """
        offset = 0
        for param in model.parameters():
            numel = param.numel()
            param.data = vector[offset:offset + numel].view(param.shape)
            offset += numel
    
    def sample(self, scale: float = 1.0, block: bool = False) -> nn.Module:
        """
        Sample model from SWAG posterior.
        
        Samples: w ~ N(μ_SWA, (σ²/2) * I + (1/2K) * Σ)
        
        Args:
            scale: Scaling factor for variance
            block: If True, use block diagonal approximation
        
        Returns:
            Model with sampled weights
        """
        # Create new model instance
        sampled_model = type(self.base_model)(
            *[getattr(self.base_model, attr) 
              for attr in ['input_dim', 'hidden_dims', 'output_dim'] 
              if hasattr(self.base_model, attr)]
        )
        
        # Sample from posterior
        for name, param in sampled_model.named_parameters():
            # Diagonal variance
            var = torch.clamp(self.sq_mean[name] - self.mean[name] ** 2, min=1e-30)
            
            # Sample
            eps = torch.randn_like(self.mean[name])
            param.data = self.mean[name] + scale * torch.sqrt(var) * eps
        
        # Add low-rank component
        if len(self.cov_mat_sqrt) > 0:
            # Sample from low-rank Gaussian
            z = torch.randn(len(self.cov_mat_sqrt))
            low_rank_sample = sum([z[i] * self.cov_mat_sqrt[i] 
                                  for i in range(len(self.cov_mat_sqrt))])
            low_rank_sample = low_rank_sample / math.sqrt(2 * (len(self.cov_mat_sqrt) - 1))
            
            # Add to parameters
            self._unflatten_params(
                self._flatten_params(sampled_model) + scale * low_rank_sample,
                sampled_model
            )
        
        return sampled_model
    
    def predict_with_uncertainty(self, x: torch.Tensor, 
                                n_samples: int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty using SWAG samples.
        
        Args:
            x: Input tensor
            n_samples: Number of posterior samples
        
        Returns:
            mean_probs: Mean predictions
            uncertainty: Epistemic uncertainty
        """
        predictions = []
        
        for _ in range(n_samples):
            # Sample model from posterior
            model = self.sample(scale=1.0)
            model.eval()
            
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions)
        mean_probs = torch.mean(predictions, dim=0)
        variance = torch.var(predictions, dim=0)
        uncertainty = torch.mean(variance, dim=1)
        
        return mean_probs, uncertainty


def demonstrate_swag():
    """
    Demonstrate SWAG for uncertainty estimation.
    """
    print("\n" + "=" * 70)
    print("PART 2: Stochastic Weight Averaging Gaussian (SWAG)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Load data
    print("\nPreparing data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    train_subset, _ = random_split(train_dataset, [5000, 55000])
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    
    # Create base model
    print("\nCreating base model for SWAG...")
    
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(28*28, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    
    model = SimpleNN()
    
    # Train normally first
    print("\nPhase 1: Normal SGD training (5 epochs)...")
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.view(batch_x.size(0), -1)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")
    
    # Initialize SWAG and collect models
    print("\nPhase 2: Collecting weight snapshots for SWAG...")
    swag = SWAGModel(model, max_num_models=20)
    
    # Continue training and collect snapshots
    n_snapshots = 20
    snapshot_freq = 5  # Collect every 5 batches
    
    model.train()
    batch_count = 0
    
    for _ in range(3):  # Additional epochs for collection
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.view(batch_x.size(0), -1)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Collect snapshot
            batch_count += 1
            if batch_count % snapshot_freq == 0 and swag.n_models < n_snapshots:
                swag.collect_model(model)
                print(f"Collected snapshot {swag.n_models}/{n_snapshots}")
    
    print(f"\nSWAG: Collected {swag.n_models} weight snapshots")
    print("Ready for uncertainty estimation!")
    
    return swag


# ============================================================================
# PART 3: COMPARISON OF BAYESIAN METHODS
# ============================================================================

def compare_bayesian_methods():
    """
    Compare different Bayesian uncertainty methods.
    """
    print("\n" + "=" * 70)
    print("PART 3: Comparison of Bayesian Methods")
    print("=" * 70)
    
    print("""
    Method Comparison:
    
    1. Bayes by Backprop:
       ✓ Principled variational inference
       ✓ Learns weight distributions
       ✗ Requires architecture modification
       ✗ Slower convergence
       ✗ More hyperparameters (prior)
       
    2. SWAG:
       ✓ Works with standard training
       ✓ No architecture changes needed
       ✓ Fast and practical
       ✗ Gaussian approximation
       ✗ Requires weight snapshot storage
       
    3. Laplace Approximation:
       ✓ Post-hoc (after training)
       ✓ Theoretical foundation
       ✗ Expensive Hessian computation
       ✗ Quadratic approximation
    
    Trade-offs:
    - Accuracy: Bayes by Backprop > SWAG > Laplace
    - Speed: SWAG > Laplace > Bayes by Backprop
    - Simplicity: SWAG > Laplace > Bayes by Backprop
    - Memory: Laplace ≈ Bayes by Backprop < SWAG
    
    Recommendation:
    - Research: Bayes by Backprop
    - Production: SWAG
    - Quick experiments: Laplace
    """)


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_weight_distributions(bnn: BayesianNN, layer_idx: int = 0):
    """
    Visualize learned weight distributions in Bayesian NN.
    
    Args:
        bnn: Bayesian neural network
        layer_idx: Which layer to visualize
    """
    layer = bnn.layers[layer_idx]
    
    # Get weight statistics
    mu = layer.weight_mu.detach().cpu().numpy().flatten()
    sigma = F.softplus(layer.weight_rho).detach().cpu().numpy().flatten()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Weight means
    axes[0].hist(mu, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('Weight Mean (μ)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Distribution of Weight Means (Layer {layer_idx})')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[0].legend()
    
    # Plot 2: Weight standard deviations
    axes[1].hist(sigma, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('Weight Std (σ)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Distribution of Weight Std Devs (Layer {layer_idx})')
    
    # Plot 3: Signal-to-noise ratio
    snr = np.abs(mu) / (sigma + 1e-8)
    axes[2].hist(snr, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[2].set_xlabel('Signal-to-Noise Ratio (|μ|/σ)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Weight Signal-to-Noise Ratio')
    axes[2].axvline(1, color='red', linestyle='--', linewidth=2, label='SNR=1')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('bayesian_weights.png', dpi=150, bbox_inches='tight')
    print("\nWeight distribution visualization saved as 'bayesian_weights.png'")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function demonstrating advanced Bayesian uncertainty.
    """
    print("\n" + "="*70)
    print("MODULE 63.3: BAYESIAN NEURAL NETWORKS & ADVANCED UNCERTAINTY")
    print("="*70)
    
    # Part 1: Bayesian NN with Bayes by Backprop
    bnn, losses = demonstrate_bayesian_nn()
    
    # Part 2: SWAG
    swag = demonstrate_swag()
    
    # Part 3: Comparison
    compare_bayesian_methods()
    
    # Visualizations
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)
    visualize_weight_distributions(bnn, layer_idx=0)
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. Bayesian Deep Learning:
       - Represents weights as distributions, not point estimates
       - Provides principled uncertainty quantification
       - Enables better calibration and robustness
    
    2. Bayes by Backprop:
       - Variational inference for BNNs
       - Learns μ and σ for each weight
       - Balances data fit (NLL) and complexity (KL)
    
    3. SWAG:
       - Practical Bayesian approximation
       - Works with standard SGD training
       - Captures weight uncertainty from snapshots
    
    4. When to Use What:
       - High-stakes applications: Bayes by Backprop
       - Production systems: SWAG
       - Quick prototyping: MC Dropout
       - Best performance: Ensembles + SWAG
    
    5. Theoretical Foundations:
       - All methods approximate p(w|D)
       - Trade-off: accuracy vs computational cost
       - Calibration should always be evaluated
    """)
    
    print("\nNext Steps:")
    print("  → Try 04_calibration_evaluation.py for calibration methods")
    print("  → Experiment with prior strengths in BNN")
    print("  → Compare computational costs")
    print("  → Apply to your domain-specific problems")


if __name__ == "__main__":
    main()
