"""
FILE: 03_langevin_dynamics.py
DIFFICULTY: Beginner
ESTIMATED TIME: 2 hours
PREREQUISITES: 01_score_functions_basics.py, 02_score_matching_theory.py

LEARNING OBJECTIVES:
    1. Understand Langevin MCMC sampling
    2. Implement score-based sampling algorithms
    3. Visualize sampling trajectories
    4. Understand the role of step size and noise

MATHEMATICAL BACKGROUND:
    Langevin Dynamics is an MCMC algorithm that uses gradients to sample from a distribution.
    
    Update rule: x_{t+1} = x_t + ε/2 * ∇log p(x_t) + √ε * z_t
    
    where z_t ~ N(0, I) is standard Gaussian noise.
    
    When ε→0 and t→∞, samples converge to p(x).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def langevin_sampling(score_fn, x_init, n_steps=1000, step_size=0.01):
    """
    Sample from a distribution using Langevin dynamics.
    
    Args:
        score_fn: Function computing ∇log p(x)
        x_init: Initial position, shape (n_samples, dim)
        n_steps: Number of Langevin steps
        step_size: Step size ε
    
    Returns:
        samples: Final samples, shape (n_samples, dim)
        trajectory: All intermediate positions, shape (n_steps, n_samples, dim)
    """
    x = x_init.clone()
    trajectory = [x.clone()]
    
    for step in range(n_steps):
        # Compute score
        score = score_fn(x)
        
        # Langevin update
        noise = torch.randn_like(x)
        x = x + (step_size / 2) * score + np.sqrt(step_size) * noise
        
        trajectory.append(x.clone())
    
    return x, torch.stack(trajectory)


def visualize_sampling_trajectories(trajectory, true_pdf=None, xlim=(-3,3), ylim=(-3,3)):
    """Visualize how samples evolve during Langevin sampling."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot true density if available
    if true_pdf is not None:
        x = np.linspace(xlim[0], xlim[1], 100)
        y = np.linspace(ylim[0], ylim[1], 100)
        X, Y = np.meshgrid(x, y)
        grid = np.stack([X.flatten(), Y.flatten()], axis=1)
        Z = true_pdf(grid).reshape(X.shape)
        ax.contourf(X, Y, Z, levels=20, cmap='Blues', alpha=0.3)
    
    # Plot trajectories
    traj_np = trajectory.numpy()
    n_steps, n_samples, _ = traj_np.shape
    
    for i in range(min(n_samples, 10)):  # Plot first 10 trajectories
        ax.plot(traj_np[:, i, 0], traj_np[:, i, 1], 
               alpha=0.3, linewidth=0.5)
        ax.scatter(traj_np[0, i, 0], traj_np[0, i, 1],
                  c='red', s=50, marker='o', label='Start' if i==0 else '')
        ax.scatter(traj_np[-1, i, 0], traj_np[-1, i, 1],
                  c='green', s=50, marker='*', label='End' if i==0 else '')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Langevin Sampling Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


if __name__ == "__main__":
    print("Langevin Dynamics Demo")
    print("=" * 80)
    
    # Demo: Sample from 2D Gaussian
    def gaussian_score(x):
        return -x  # Score of N(0, I)
    
    # Initialize samples
    x_init = torch.randn(100, 2) * 3  # Start from broad distribution
    
    # Run Langevin sampling
    samples, trajectory = langevin_sampling(
        lambda x: torch.tensor(gaussian_score(x.numpy()), dtype=torch.float32),
        x_init,
        n_steps=500,
        step_size=0.1
    )
    
    # Visualize
    visualize_sampling_trajectories(
        trajectory,
        true_pdf=lambda x: np.exp(-0.5 * np.sum(x**2, axis=1)) / (2*np.pi)
    )
    plt.savefig('/home/claude/demo_langevin.png', dpi=150, bbox_inches='tight')
    print("Saved demo_langevin.png")
    
    print(f"\nFinal sample statistics:")
    print(f"Mean: {samples.mean(dim=0).numpy()}")
    print(f"Std: {samples.std(dim=0).numpy()}")
    print("\nExpected: Mean=[0, 0], Std=[1, 1]")
