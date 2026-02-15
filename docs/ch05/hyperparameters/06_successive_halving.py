"""
Successive Halving and Hyperband for Hyperparameter Optimization

This module implements two modern hyperparameter optimization algorithms:
1. Successive Halving: Trains many configurations on small budgets, progressively
   eliminating bad ones and increasing the budget for survivors.
2. Hyperband: Runs successive halving with different trade-off parameters to
   balance exploration vs. exploitation.

These algorithms are more efficient than random search and grid search because
they use early stopping to discard unpromising configurations.

Educational purpose: Chapter 5 - Deep Learning Optimization & Hyperparameter Tuning
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Dict, Tuple
import random


class SimpleMLPConfig:
    """Configuration for a simple MLP model."""
    def __init__(self, learning_rate: float, batch_size: int, hidden_dim: int = 64):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.id = random.randint(0, 1000000)

    def __repr__(self):
        return f"Config(lr={self.learning_rate:.2e}, bs={self.batch_size}, id={self.id})"


class SimpleMLP(nn.Module):
    """Simple two-layer MLP for classification."""
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def create_synthetic_data(n_samples: int = 1000, input_dim: int = 20, num_classes: int = 2):
    """
    Create synthetic classification dataset.

    Args:
        n_samples: Total number of samples
        input_dim: Feature dimension
        num_classes: Number of output classes

    Returns:
        DataLoader for training and validation sets
    """
    # Generate random features and labels
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))

    # Split into train (70%) and validation (30%)
    train_size = int(0.7 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    return X_train, y_train, X_val, y_val


def train_config_for_budget(
    config: SimpleMLPConfig,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    device: str = 'cpu'
) -> float:
    """
    Train a model with the given configuration for a specified number of epochs.

    Args:
        config: Configuration with learning rate, batch size, etc.
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of epochs to train
        device: Device to train on

    Returns:
        Final validation accuracy
    """
    # Create model and optimizer
    model = SimpleMLP(hidden_dim=config.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Create data loader with config's batch size
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val.to(device))
        val_preds = val_logits.argmax(dim=1)
        val_accuracy = (val_preds == y_val.to(device)).float().mean().item()

    return val_accuracy


class SuccessiveHalvingScheduler:
    """
    Successive Halving: Start with many configurations at small budget, keep top 1/eta,
    increase budget by eta factor, repeat.

    Algorithm:
    1. Sample n configurations
    2. Allocate resource budget r_min to each
    3. Evaluate all, keep top ceil(n / eta) configurations
    4. Increase budget: r_min *= eta
    5. Repeat until 1 configuration remains

    Args:
        eta: Factor by which to reduce/increase (typically 2 or 3)
        r_min: Minimum budget (e.g., epochs)
        r_max: Maximum budget
    """

    def __init__(self, eta: float = 2.0, r_min: int = 1, r_max: int = 32):
        self.eta = eta
        self.r_min = r_min
        self.r_max = r_max
        self.s_max = math.floor(math.log(r_max / r_min) / math.log(eta))

    def run(
        self,
        configs: List[SimpleMLPConfig],
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        device: str = 'cpu',
        verbose: bool = True
    ) -> Tuple[SimpleMLPConfig, float]:
        """
        Run successive halving on the given configurations.

        Args:
            configs: List of configurations to search
            X_train, y_train, X_val, y_val: Training and validation data
            device: Device to train on
            verbose: Whether to print progress

        Returns:
            Best configuration and its validation accuracy
        """
        remaining_configs = configs[:]
        budget = self.r_min

        stage = 0
        while len(remaining_configs) > 1:
            if verbose:
                print(f"\nStage {stage}: {len(remaining_configs)} configs, budget={budget} epochs")

            # Evaluate all remaining configs with current budget
            scores = []
            for config in remaining_configs:
                acc = train_config_for_budget(
                    config, X_train, y_train, X_val, y_val,
                    epochs=budget, device=device
                )
                scores.append((config, acc))
                if verbose:
                    print(f"  {config} -> accuracy={acc:.4f}")

            # Sort by score (descending)
            scores.sort(key=lambda x: x[1], reverse=True)

            # Keep top 1/eta configs
            num_keep = max(1, int(len(remaining_configs) / self.eta))
            remaining_configs = [config for config, _ in scores[:num_keep]]

            if verbose:
                print(f"  Keeping top {num_keep} config(s)")

            # Increase budget for next stage
            budget = int(budget * self.eta)
            budget = min(budget, self.r_max)
            stage += 1

        # Final evaluation of best config
        best_config = remaining_configs[0]
        final_acc = train_config_for_budget(
            best_config, X_train, y_train, X_val, y_val,
            epochs=self.r_max, device=device
        )

        if verbose:
            print(f"\nBest config: {best_config}")
            print(f"Final accuracy at r_max={self.r_max}: {final_acc:.4f}")

        return best_config, final_acc


class Hyperband:
    """
    Hyperband: Runs successive halving with different trade-off parameters.

    Hyperband runs multiple successive halving brackets, each with a different
    R (max budget) and n (initial number of configs) pair. This balances:
    - Many configs with small budgets (broad search)
    - Few configs with large budgets (deep search)

    Args:
        eta: Factor for successive halving (typically 2 or 3)
        R: Maximum resource per configuration (total budget)
        B: Total budget (epochs * num_configs) to spend
    """

    def __init__(self, eta: float = 2.0, R: int = 81, B: int = 5):
        self.eta = eta
        self.R = R
        self.B = B  # Budget per bracket
        self.s_max = math.floor(math.log(R) / math.log(eta))

    def run(
        self,
        configs: List[SimpleMLPConfig],
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        device: str = 'cpu',
        verbose: bool = True
    ) -> Tuple[SimpleMLPConfig, float]:
        """
        Run Hyperband (multiple successive halving brackets).

        Args:
            configs: Pool of configurations to sample from
            X_train, y_train, X_val, y_val: Training and validation data
            device: Device to train on
            verbose: Whether to print progress

        Returns:
            Best configuration found and its accuracy
        """
        best_config = None
        best_acc = 0.0

        # Iterate through successive halving brackets (s parameter)
        for s in range(self.s_max, -1, -1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Hyperband Bracket s={s}")
                print(f"{'='*60}")

            # Compute bracket-specific parameters
            n = math.ceil((self.B / self.R) * (self.s_max + 1) / (s + 1))
            r_min = self.R / (self.eta ** s)

            # Sample n configs for this bracket
            bracket_configs = random.sample(configs, min(int(n), len(configs)))

            # Run successive halving on this bracket
            scheduler = SuccessiveHalvingScheduler(
                eta=self.eta,
                r_min=max(1, int(r_min)),
                r_max=self.R
            )
            bracket_best_config, bracket_best_acc = scheduler.run(
                bracket_configs, X_train, y_train, X_val, y_val,
                device=device, verbose=verbose
            )

            # Update global best
            if bracket_best_acc > best_acc:
                best_acc = bracket_best_acc
                best_config = bracket_best_config

        return best_config, best_acc


def main():
    """
    Demo: Run Successive Halving and Hyperband on synthetic data.
    Shows how these algorithms efficiently find good hyperparameters.
    """
    print("Deep Learning Hyperparameter Optimization: Successive Halving & Hyperband")
    print("=" * 70)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Create synthetic dataset
    print("\nGenerating synthetic data...")
    X_train, y_train, X_val, y_val = create_synthetic_data(n_samples=2000, input_dim=20)
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

    # Generate random configurations to search
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
    batch_sizes = [16, 32, 64, 128]
    configs = [
        SimpleMLPConfig(lr=lr, batch_size=bs)
        for lr in learning_rates
        for bs in batch_sizes
    ]
    print(f"\nSearching over {len(configs)} configurations")

    # Run Successive Halving
    print("\n" + "="*70)
    print("SUCCESSIVE HALVING")
    print("="*70)
    sh_scheduler = SuccessiveHalvingScheduler(eta=2.0, r_min=1, r_max=16)
    sh_best_config, sh_best_acc = sh_scheduler.run(
        configs, X_train, y_train, X_val, y_val, verbose=True
    )

    # Run Hyperband
    print("\n" + "="*70)
    print("HYPERBAND")
    print("="*70)
    hyperband = Hyperband(eta=2.0, R=16, B=5)
    hb_best_config, hb_best_acc = hyperband.run(
        configs, X_train, y_train, X_val, y_val, verbose=False
    )

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Successive Halving best: {sh_best_config} with accuracy {sh_best_acc:.4f}")
    print(f"Hyperband best: {hb_best_config} with accuracy {hb_best_acc:.4f}")


if __name__ == "__main__":
    main()
