"""
Visualization of Gradient Descent Paths on 2D Loss Landscapes

This module implements various 2D test functions commonly used to visualize
and understand optimizer behavior. We compare how different optimizers navigate
the loss surface: SGD, SGD+Momentum, Adam, and RMSprop.

This helps understand why certain optimizers perform better on different
landscape geometries, and how momentum and adaptive learning rates affect
convergence paths.

Educational purpose: Chapter 5 - Understanding Optimizer Dynamics
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import math


# ============================================================================
# 2D Test Functions for Optimizer Visualization
# ============================================================================

def rosenbrock(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    A classic test function with a narrow valley.
    Global minimum at (1, 1) with f = 0.
    """
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def beale(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Beale function: f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2
    Multiple local minima and a complex landscape.
    Global minimum at (3, 0.5) with f = 0.
    """
    term1 = (1.5 - x + x * y) ** 2
    term2 = (2.25 - x + x * y ** 2) ** 2
    term3 = (2.625 - x + x * y ** 3) ** 2
    return term1 + term2 + term3


def himmelblau(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Himmelblau's function: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    Has four equal local minima and complex landscape structure.
    Global minima at (3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)
    """
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def rastrigin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Rastrigin function: f(x,y) = 20 + x^2 + y^2 - 10*cos(2πx) - 10*cos(2πy)
    Many local minima; highly multimodal landscape.
    Global minimum at (0, 0) with f = 0.
    """
    A = 10
    return A * 2 + x ** 2 + y ** 2 - A * torch.cos(2 * math.pi * x) - A * torch.cos(2 * math.pi * y)


# ============================================================================
# Generic Training Function with Path Tracking
# ============================================================================

def train_2d(
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer_fn: Callable,
    x_init: float = -1.0,
    y_init: float = -1.0,
    num_steps: int = 1000,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a 2D point using the specified optimizer and loss function.
    Track the path taken by the optimizer across the landscape.

    Args:
        loss_fn: Function that takes (x, y) tensors and returns loss
        optimizer_fn: Function that takes [x, y] params and returns optimizer
        x_init, y_init: Initial position
        num_steps: Number of optimization steps
        device: Device to use ('cpu' or 'cuda')

    Returns:
        Tuple of (x_path, y_path) numpy arrays recording the optimization path
    """
    # Initialize parameters
    x = torch.tensor(x_init, dtype=torch.float32, device=device, requires_grad=True)
    y = torch.tensor(y_init, dtype=torch.float32, device=device, requires_grad=True)

    # Create optimizer
    optimizer = optimizer_fn([x, y])

    # Track path
    x_path = [x.detach().cpu().numpy()]
    y_path = [y.detach().cpu().numpy()]

    # Optimization loop
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = loss_fn(x, y)
        loss.backward()
        optimizer.step()

        # Record position
        x_path.append(x.detach().cpu().numpy())
        y_path.append(y.detach().cpu().numpy())

    return np.array(x_path), np.array(y_path)


# ============================================================================
# Visualization
# ============================================================================

def plot_optimizer_comparison(
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    loss_name: str,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    x_init: float = -1.0,
    y_init: float = -1.0,
    num_steps: int = 1000
):
    """
    Compare optimizer trajectories on a 2D loss landscape.

    Args:
        loss_fn: The loss function to visualize
        loss_name: Name of the loss function (for title)
        x_range, y_range: Ranges for the contour plot
        x_init, y_init: Starting position
        num_steps: Number of optimization steps
    """
    # Generate grid for contour plot
    x_grid = np.linspace(x_range[0], x_range[1], 200)
    y_grid = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Compute loss on grid
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float()
        Y_tensor = torch.from_numpy(Y).float()
        Z = loss_fn(X_tensor, Y_tensor).numpy()

    # Define optimizers to compare
    optimizers = {
        'SGD': lambda params: torch.optim.SGD(params, lr=0.01),
        'SGD+Momentum': lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
        'Adam': lambda params: torch.optim.Adam(params, lr=0.1),
        'RMSprop': lambda params: torch.optim.RMSprop(params, lr=0.01),
    }

    # Colors for each optimizer
    colors = {
        'SGD': 'blue',
        'SGD+Momentum': 'green',
        'Adam': 'red',
        'RMSprop': 'orange',
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot contour map (log scale for better visibility)
    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)

    # Train each optimizer and plot trajectory
    for opt_name, opt_fn in optimizers.items():
        x_path, y_path = train_2d(
            loss_fn, opt_fn, x_init=x_init, y_init=y_init, num_steps=num_steps
        )

        # Plot path
        ax.plot(x_path, y_path, 'o-', color=colors[opt_name], label=opt_name,
                markersize=3, alpha=0.7, linewidth=1.5)

        # Mark start and end
        ax.plot(x_path[0], y_path[0], 's', color=colors[opt_name], markersize=8)
        ax.plot(x_path[-1], y_path[-1], '*', color=colors[opt_name], markersize=15)

    # Labels and legend
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'{loss_name} Function - Optimizer Trajectories\n(Square=Start, Star=End)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """
    Demo: Visualize different optimizers on multiple 2D landscapes.
    Shows how optimizer choice affects convergence paths and behavior.
    """
    print("2D Loss Landscape Visualization for Different Optimizers")
    print("=" * 70)

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Define test functions with appropriate ranges
    test_functions = [
        (rosenbrock, 'Rosenbrock', (-2, 3), (-1, 4), -1.5, 2.5),
        (beale, 'Beale', (-4.5, 4.5), (-4.5, 4.5), -2, -2),
        (himmelblau, 'Himmelblau', (-5, 5), (-5, 5), 0, 0),
        (rastrigin, 'Rastrigin', (-5.12, 5.12), (-5.12, 5.12), -3, -3),
    ]

    # Create subplots for all functions
    fig_list = []
    for loss_fn, loss_name, x_range, y_range, x_init, y_init in test_functions:
        print(f"\nVisualizing {loss_name} function...")
        fig = plot_optimizer_comparison(
            loss_fn, loss_name, x_range, y_range, x_init, y_init, num_steps=500
        )
        fig_list.append((loss_name, fig))

    # Display all figures
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"Generated {len(fig_list)} optimizer comparison plots:")
    for name, _ in fig_list:
        print(f"  - {name}")

    print("\nKey observations:")
    print("  - SGD: Direct paths but may oscillate at narrow valleys (Rosenbrock)")
    print("  - Momentum: Faster convergence due to acceleration in consistent directions")
    print("  - Adam: Adaptive learning rates handle different scales well (Rastrigin)")
    print("  - RMSprop: Balances speed and stability across varied landscapes")

    plt.show()


if __name__ == "__main__":
    main()
