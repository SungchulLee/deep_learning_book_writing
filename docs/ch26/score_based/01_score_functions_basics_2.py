"""
FILE: 01_score_functions_basics.py
DIFFICULTY: Beginner
ESTIMATED TIME: 2-3 hours
PREREQUISITES: 
    - Basic Python programming
    - NumPy fundamentals
    - Basic probability (Gaussian distributions)
    - Matplotlib for visualization

LEARNING OBJECTIVES:
    1. Understand what a score function is mathematically
    2. Compute score functions analytically for simple distributions
    3. Visualize score functions as vector fields
    4. Understand the connection between scores and probability gradients
    5. Implement score computation for basic distributions

MATHEMATICAL BACKGROUND:
    The score function is defined as the gradient of the log probability density:
    
    s(x) = ∇_x log p(x)
    
    Key properties:
    - Points toward regions of higher probability
    - Does not require knowing the normalization constant Z
    - For p(x) = exp(f(x))/Z, we have s(x) = ∇_x f(x)
    - Related to Fisher Information: I = E[||s(x)||²]
    
    For a Gaussian N(μ, Σ):
    - log p(x) = -1/2 (x-μ)ᵀ Σ⁻¹ (x-μ) + constant
    - s(x) = -Σ⁻¹(x - μ)
    
    For a mixture of Gaussians:
    - s(x) = Σᵢ wᵢ(x) * sᵢ(x)
    - where wᵢ(x) = p(i|x) are the posterior weights

IMPLEMENTATION NOTES:
    - We use PyTorch for automatic differentiation
    - Visualizations use matplotlib's quiver plots for vector fields
    - Focus on 2D examples for intuition before generalizing
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math


# ============================================================================
# SECTION 1: SCORE FUNCTION FUNDAMENTALS
# ============================================================================

def compute_gaussian_score_analytical(x, mu, sigma):
    """
    Compute the score function for a Gaussian distribution analytically.
    
    For a Gaussian N(μ, σ²), the score function is:
        s(x) = ∇_x log p(x) = -(x - μ)/σ²
    
    This points toward the mean μ with strength proportional to distance
    and inversely proportional to variance σ².
    
    Args:
        x: Input points, shape (N, D) where N=number of points, D=dimension
        mu: Mean vector, shape (D,)
        sigma: Standard deviation (scalar or shape (D,))
    
    Returns:
        score: Score function values, shape (N, D)
    """
    # Convert to numpy if needed
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()
    
    # Compute score: s(x) = -(x - μ)/σ²
    # This is the gradient of log probability
    score = -(x - mu) / (sigma ** 2)
    
    return score


def compute_gaussian_score_autograd(x, mu, sigma):
    """
    Compute the score function for a Gaussian using automatic differentiation.
    
    This demonstrates the connection between:
    - Score function: s(x) = ∇_x log p(x)
    - Automatic differentiation: PyTorch's autograd
    
    Args:
        x: Input tensor with requires_grad=True, shape (N, D)
        mu: Mean tensor, shape (D,)
        sigma: Standard deviation (scalar)
    
    Returns:
        score: Score function values, shape (N, D)
    """
    # Ensure x requires gradient
    x = x.clone().detach().requires_grad_(True)
    
    # Compute log probability (up to constant)
    # log p(x) = -1/(2σ²) ||x - μ||² + constant
    log_prob = -0.5 * torch.sum((x - mu) ** 2, dim=1) / (sigma ** 2)
    
    # Compute gradient of log_prob with respect to x
    # This is the score function by definition
    score = torch.autograd.grad(
        outputs=log_prob.sum(),  # Sum for batched computation
        inputs=x,
        create_graph=True  # Allow second-order derivatives if needed
    )[0]
    
    return score


def gaussian_mixture_score(x, mus, sigmas, weights):
    """
    Compute score function for a Gaussian Mixture Model.
    
    For a mixture p(x) = Σᵢ πᵢ N(x|μᵢ, σᵢ²), the score is:
    
    s(x) = ∇_x log p(x) = Σᵢ p(i|x) * sᵢ(x)
    
    where:
    - p(i|x) = πᵢ N(x|μᵢ, σᵢ²) / p(x) is the posterior probability
    - sᵢ(x) = -(x - μᵢ)/σᵢ² is the score of component i
    
    Args:
        x: Input points, shape (N, D)
        mus: Means of components, list of length K with shapes (D,)
        sigmas: Standard deviations, list of length K
        weights: Mixture weights πᵢ, array of shape (K,), must sum to 1
    
    Returns:
        score: Mixture score function, shape (N, D)
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    K = len(mus)  # Number of mixture components
    N, D = x.shape
    
    # Initialize arrays
    log_probs = torch.zeros(N, K)  # Log probabilities for each component
    scores = torch.zeros(N, D, K)   # Scores for each component
    
    # Compute log probability and score for each component
    for i in range(K):
        mu_i = torch.as_tensor(mus[i], dtype=torch.float32)
        sigma_i = sigmas[i]
        
        # Log probability of component i
        # log N(x|μᵢ, σᵢ²) = -1/(2σᵢ²)||x-μᵢ||² - D/2*log(2πσᵢ²)
        diff = x - mu_i
        log_probs[:, i] = (
            -0.5 * torch.sum(diff ** 2, dim=1) / (sigma_i ** 2)
            - D / 2 * np.log(2 * np.pi * sigma_i ** 2)
            + np.log(weights[i])
        )
        
        # Score of component i: sᵢ(x) = -(x - μᵢ)/σᵢ²
        scores[:, :, i] = -diff / (sigma_i ** 2)
    
    # Compute posterior probabilities p(i|x) using log-sum-exp trick
    # p(i|x) = exp(log p(x, i)) / Σⱼ exp(log p(x, j))
    log_sum = torch.logsumexp(log_probs, dim=1, keepdim=True)  # log p(x)
    posterior = torch.exp(log_probs - log_sum)  # p(i|x)
    
    # Weighted sum of component scores: s(x) = Σᵢ p(i|x) * sᵢ(x)
    mixture_score = torch.sum(
        scores * posterior.unsqueeze(1),  # Broadcast posterior
        dim=2
    )
    
    return mixture_score.numpy()


# ============================================================================
# SECTION 2: VISUALIZATION UTILITIES
# ============================================================================

def plot_score_field_2d(score_fn, xlim=(-3, 3), ylim=(-3, 3), n_points=20, 
                        title="Score Function Vector Field", figsize=(10, 8)):
    """
    Visualize a 2D score function as a vector field using quiver plot.
    
    The arrows point in the direction of increasing log probability,
    showing how samples would move under score-based dynamics.
    
    Args:
        score_fn: Function that takes (N, 2) array and returns (N, 2) scores
        xlim: X-axis limits
        ylim: Y-axis limits
        n_points: Number of grid points per dimension
        title: Plot title
        figsize: Figure size
    """
    # Create grid of points
    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid for batch computation
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    # Compute scores at all grid points
    scores = score_fn(grid_points)
    
    # Reshape for plotting
    U = scores[:, 0].reshape(X.shape)
    V = scores[:, 1].reshape(X.shape)
    
    # Compute magnitude for coloring
    magnitude = np.sqrt(U**2 + V**2)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Quiver plot with color indicating magnitude
    plt.quiver(X, Y, U, V, magnitude, cmap='viridis', 
               scale=50, width=0.003, alpha=0.8)
    
    plt.colorbar(label='Score Magnitude')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_probability_and_score(pdf_fn, score_fn, xlim=(-3, 3), ylim=(-3, 3),
                               n_points=100, figsize=(15, 5)):
    """
    Create side-by-side visualization of probability density and score field.
    
    This helps understand the relationship:
    - High probability regions → scores point inward
    - Low probability regions → scores point toward modes
    
    Args:
        pdf_fn: Function computing probability density p(x)
        score_fn: Function computing score ∇log p(x)
        xlim, ylim: Axis limits
        n_points: Grid resolution
        figsize: Figure size
    """
    # Create fine grid for probability density
    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    # Compute probability density
    density = pdf_fn(grid_points).reshape(X.shape)
    
    # Create coarser grid for score field
    n_arrows = 20
    x_arrows = np.linspace(xlim[0], xlim[1], n_arrows)
    y_arrows = np.linspace(ylim[0], ylim[1], n_arrows)
    X_arrows, Y_arrows = np.meshgrid(x_arrows, y_arrows)
    arrow_points = np.stack([X_arrows.flatten(), Y_arrows.flatten()], axis=1)
    
    # Compute scores
    scores = score_fn(arrow_points)
    U = scores[:, 0].reshape(X_arrows.shape)
    V = scores[:, 1].reshape(X_arrows.shape)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Probability density
    im1 = axes[0].contourf(X, Y, density, levels=20, cmap='Blues')
    axes[0].set_xlabel('x₁')
    axes[0].set_ylabel('x₂')
    axes[0].set_title('Probability Density p(x)')
    axes[0].axis('equal')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot 2: Score field
    magnitude = np.sqrt(U**2 + V**2)
    axes[1].quiver(X_arrows, Y_arrows, U, V, magnitude, 
                   cmap='viridis', scale=50, width=0.003)
    axes[1].set_xlabel('x₁')
    axes[1].set_ylabel('x₂')
    axes[1].set_title('Score Function ∇log p(x)')
    axes[1].axis('equal')
    
    # Plot 3: Overlay
    axes[2].contourf(X, Y, density, levels=20, cmap='Blues', alpha=0.5)
    axes[2].quiver(X_arrows, Y_arrows, U, V, magnitude,
                   cmap='Reds', scale=50, width=0.003, alpha=0.7)
    axes[2].set_xlabel('x₁')
    axes[2].set_ylabel('x₂')
    axes[2].set_title('Overlay: Density + Score')
    axes[2].axis('equal')
    
    plt.tight_layout()


# ============================================================================
# SECTION 3: EXAMPLE DISTRIBUTIONS
# ============================================================================

class GaussianDistribution2D:
    """
    2D Gaussian distribution with analytical score function.
    
    This class demonstrates:
    1. How to compute scores analytically
    2. The relationship between covariance and score behavior
    3. Isotropic vs anisotropic cases
    """
    
    def __init__(self, mu, sigma):
        """
        Initialize 2D Gaussian.
        
        Args:
            mu: Mean vector, shape (2,)
            sigma: Standard deviation (scalar for isotropic, 
                   array of shape (2,) for diagonal covariance)
        """
        self.mu = np.array(mu, dtype=np.float32)
        
        if np.isscalar(sigma):
            self.sigma = np.array([sigma, sigma], dtype=np.float32)
        else:
            self.sigma = np.array(sigma, dtype=np.float32)
        
        # Covariance matrix (diagonal)
        self.cov = np.diag(self.sigma ** 2)
        self.cov_inv = np.diag(1 / (self.sigma ** 2))
    
    def pdf(self, x):
        """Compute probability density."""
        x = np.atleast_2d(x)
        diff = x - self.mu
        
        # Normalization constant
        norm = 1 / (2 * np.pi * np.prod(self.sigma))
        
        # Exponent: -1/2 (x-μ)ᵀ Σ⁻¹ (x-μ)
        exponent = -0.5 * np.sum(diff ** 2 / (self.sigma ** 2), axis=1)
        
        return norm * np.exp(exponent)
    
    def score(self, x):
        """
        Compute score function analytically.
        
        For diagonal covariance Σ = diag(σ₁², σ₂²):
        s(x) = -Σ⁻¹(x - μ) = -[(x₁-μ₁)/σ₁², (x₂-μ₂)/σ₂²]
        """
        x = np.atleast_2d(x)
        return -(x - self.mu) / (self.sigma ** 2)


class GaussianMixture2D:
    """
    2D Gaussian Mixture Model.
    
    Demonstrates:
    1. Multi-modal distributions
    2. Complex score functions
    3. Mixture score computation
    """
    
    def __init__(self, mus, sigmas, weights):
        """
        Initialize Gaussian mixture.
        
        Args:
            mus: List of K mean vectors, each shape (2,)
            sigmas: List of K standard deviations (scalar or array)
            weights: Array of shape (K,), mixture weights summing to 1
        """
        self.mus = [np.array(mu, dtype=np.float32) for mu in mus]
        self.sigmas = sigmas
        self.weights = np.array(weights, dtype=np.float32)
        
        assert np.abs(self.weights.sum() - 1.0) < 1e-6, "Weights must sum to 1"
        
        # Create individual Gaussian components
        self.components = [
            GaussianDistribution2D(mu, sigma)
            for mu, sigma in zip(mus, sigmas)
        ]
    
    def pdf(self, x):
        """Compute mixture density: p(x) = Σᵢ πᵢ N(x|μᵢ, σᵢ²)"""
        x = np.atleast_2d(x)
        density = np.zeros(len(x))
        
        for i, component in enumerate(self.components):
            density += self.weights[i] * component.pdf(x)
        
        return density
    
    def score(self, x):
        """
        Compute mixture score function.
        
        Uses the formula: s(x) = Σᵢ p(i|x) * sᵢ(x)
        where p(i|x) is the posterior probability of component i.
        """
        return gaussian_mixture_score(x, self.mus, self.sigmas, self.weights)


# ============================================================================
# SECTION 4: DEMONSTRATIONS AND EXAMPLES
# ============================================================================

def demo_single_gaussian():
    """Demonstrate score function for a single 2D Gaussian."""
    print("=" * 80)
    print("DEMO 1: Single 2D Gaussian Distribution")
    print("=" * 80)
    
    # Create isotropic Gaussian centered at origin
    dist = GaussianDistribution2D(mu=[0, 0], sigma=1.0)
    
    # Test analytical vs autograd computation
    test_points = torch.tensor([
        [0.0, 0.0],   # At mean
        [1.0, 0.0],   # Right of mean
        [0.0, 1.0],   # Above mean
        [1.0, 1.0],   # Diagonal
    ])
    
    analytical_scores = dist.score(test_points.numpy())
    autograd_scores = compute_gaussian_score_autograd(
        test_points, 
        mu=torch.tensor([0.0, 0.0]),
        sigma=1.0
    ).numpy()
    
    print("\nTest points and their scores:")
    print("-" * 80)
    for i, pt in enumerate(test_points):
        print(f"Point: {pt.numpy()}")
        print(f"  Analytical score: {analytical_scores[i]}")
        print(f"  Autograd score:   {autograd_scores[i]}")
        print(f"  Difference:       {np.abs(analytical_scores[i] - autograd_scores[i]).max():.2e}")
        print()
    
    # Visualize
    plot_probability_and_score(
        dist.pdf,
        dist.score,
        title="Single Gaussian: N(0, I)"
    )
    plt.savefig('/home/claude/demo1_single_gaussian.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: demo1_single_gaussian.png")


def demo_anisotropic_gaussian():
    """Demonstrate score function for anisotropic Gaussian."""
    print("\n" + "=" * 80)
    print("DEMO 2: Anisotropic 2D Gaussian Distribution")
    print("=" * 80)
    
    # Create Gaussian with different variances in each direction
    dist = GaussianDistribution2D(mu=[1, -1], sigma=[2.0, 0.5])
    
    print(f"\nMean: {dist.mu}")
    print(f"Standard deviations: {dist.sigma}")
    print(f"Covariance matrix:\n{dist.cov}")
    
    # The score will be stronger (larger magnitude) in the direction
    # with smaller variance (y-direction in this case)
    test_points = np.array([
        [1.0, -1.0],   # At mean
        [2.0, -1.0],   # Shifted in x (large variance)
        [1.0, 0.0],    # Shifted in y (small variance)
    ])
    
    scores = dist.score(test_points)
    
    print("\nObserve how score magnitude relates to variance:")
    print("-" * 80)
    for i, (pt, score) in enumerate(zip(test_points, scores)):
        print(f"Point: {pt}, Score: {score}, Magnitude: {np.linalg.norm(score):.3f}")
    
    print("\nKey insight: Score is inversely proportional to variance!")
    print("  - Large variance → small score magnitude")
    print("  - Small variance → large score magnitude")
    
    # Visualize
    plot_probability_and_score(
        dist.pdf,
        dist.score,
        xlim=(-4, 6),
        ylim=(-3, 1),
        title="Anisotropic Gaussian"
    )
    plt.savefig('/home/claude/demo2_anisotropic.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: demo2_anisotropic.png")


def demo_gaussian_mixture():
    """Demonstrate score function for a Gaussian mixture."""
    print("\n" + "=" * 80)
    print("DEMO 3: Gaussian Mixture Model (3 components)")
    print("=" * 80)
    
    # Create a mixture with 3 components
    mus = [[-2, 0], [2, 0], [0, 2]]
    sigmas = [0.5, 0.5, 0.5]
    weights = [0.3, 0.3, 0.4]
    
    mixture = GaussianMixture2D(mus, sigmas, weights)
    
    print("\nMixture components:")
    for i, (mu, sigma, weight) in enumerate(zip(mus, sigmas, weights)):
        print(f"  Component {i+1}: μ={mu}, σ={sigma}, π={weight}")
    
    # Test score at various points
    test_points = np.array([
        [-2, 0],   # Center of component 1
        [2, 0],    # Center of component 2
        [0, 2],    # Center of component 3
        [0, 0],    # Between components
        [0, 1],    # Near component 3
    ])
    
    scores = mixture.score(test_points)
    densities = mixture.pdf(test_points)
    
    print("\nScore behavior at different locations:")
    print("-" * 80)
    for pt, score, dens in zip(test_points, scores, densities):
        print(f"Point: {pt}, Density: {dens:.4f}, Score: {score}, Mag: {np.linalg.norm(score):.3f}")
    
    print("\nKey insight for mixtures:")
    print("  - At mode centers: score ≈ 0 (no gradient)")
    print("  - Between modes: score points toward nearest mode")
    print("  - Score is weighted combination of component scores")
    
    # Visualize
    plot_probability_and_score(
        mixture.pdf,
        mixture.score,
        xlim=(-4, 4),
        ylim=(-2, 4),
        title="Gaussian Mixture (3 components)"
    )
    plt.savefig('/home/claude/demo3_mixture.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: demo3_mixture.png")


def demo_score_properties():
    """Demonstrate key properties of score functions."""
    print("\n" + "=" * 80)
    print("DEMO 4: Key Properties of Score Functions")
    print("=" * 80)
    
    # Property 1: Score is zero at modes
    print("\nProperty 1: Score is zero at distribution modes")
    print("-" * 80)
    
    dist = GaussianDistribution2D(mu=[0, 0], sigma=1.0)
    score_at_mode = dist.score(np.array([[0, 0]]))
    print(f"Score at mode (μ=[0,0]): {score_at_mode[0]}")
    print("Explanation: At the peak, log p(x) has zero gradient")
    
    # Property 2: Score points toward higher probability
    print("\nProperty 2: Score points toward higher probability regions")
    print("-" * 80)
    
    test_point = np.array([[2, 2]])
    score = dist.score(test_point)
    toward_mode = -test_point[0] / np.linalg.norm(test_point)  # Unit vector toward mode
    score_direction = score[0] / np.linalg.norm(score)
    
    print(f"Test point: {test_point[0]}")
    print(f"Score: {score[0]}")
    print(f"Direction toward mode: {toward_mode}")
    print(f"Score direction: {score_direction}")
    print(f"Alignment (dot product): {np.dot(toward_mode, score_direction):.4f}")
    
    # Property 3: No normalization constant needed
    print("\nProperty 3: Score doesn't require normalization constant")
    print("-" * 80)
    print("For p(x) = exp(f(x))/Z:")
    print("  log p(x) = f(x) - log Z")
    print("  ∇log p(x) = ∇f(x) - ∇log Z = ∇f(x)")
    print("  (since Z is constant w.r.t. x)")
    print("\nThis is crucial for tractable learning!")
    
    # Property 4: Scale invariance
    print("\nProperty 4: Score changes with temperature")
    print("-" * 80)
    
    # Compare scores at different temperatures (variance)
    dist_cold = GaussianDistribution2D(mu=[0, 0], sigma=0.5)  # Low temp
    dist_hot = GaussianDistribution2D(mu=[0, 0], sigma=2.0)   # High temp
    
    test_pt = np.array([[1, 1]])
    score_cold = dist_cold.score(test_pt)[0]
    score_hot = dist_hot.score(test_pt)[0]
    
    print(f"Point: {test_pt[0]}")
    print(f"Score (σ=0.5): {score_cold}, Magnitude: {np.linalg.norm(score_cold):.3f}")
    print(f"Score (σ=2.0): {score_hot}, Magnitude: {np.linalg.norm(score_hot):.3f}")
    print("\nLow temperature → sharper distribution → larger score magnitudes")


# ============================================================================
# SECTION 5: EXERCISES
# ============================================================================

def exercises():
    """
    Guided exercises for understanding score functions.
    
    Complete these to reinforce your understanding!
    """
    print("\n" + "=" * 80)
    print("EXERCISES")
    print("=" * 80)
    
    print("""
    EXERCISE 1: Implement score for 1D Laplace distribution
    --------------------------------------------------------
    The Laplace distribution is p(x) = (1/2b)exp(-|x-μ|/b)
    
    Tasks:
    a) Derive the score function analytically
    b) Implement score_laplace(x, mu, b) function
    c) Plot the score function for μ=0, b=1
    d) Compare with Gaussian score
    
    Hint: The derivative of |x| is sign(x) for x≠0
    
    EXERCISE 2: Multimodal mixture exploration
    -----------------------------------------
    Create a 4-component Gaussian mixture:
    - Components at corners of a square
    - Equal weights and variances
    - Plot score field and identify interesting regions
    
    Questions:
    a) What happens to the score between modes?
    b) Where are the score magnitudes largest?
    c) Can you identify saddle points?
    
    EXERCISE 3: Score function verification
    --------------------------------------
    For any distribution, verify that:
    E[s(x)] = 0
    
    That is, the expected score is zero.
    
    Tasks:
    a) Prove this theoretically using integration by parts
    b) Verify numerically for a Gaussian mixture
    c) Explain intuitively why this must be true
    
    Hint: ∫ ∇p(x) dx = ∇∫ p(x) dx = ∇(1) = 0
    
    EXERCISE 4: Temperature and scores
    --------------------------------
    Investigate how "temperature" affects scores:
    
    For a distribution p_T(x) ∝ p(x)^(1/T):
    a) Show that s_T(x) = s(x)/T
    b) Create visualizations at T = 0.5, 1.0, 2.0
    c) Explain implications for sampling
    
    EXERCISE 5: Fisher Information
    ----------------------------
    The Fisher Information is I = E[||s(x)||²]
    
    Tasks:
    a) Compute I for a 2D Gaussian N(0, σ²I)
    b) How does I change with σ?
    c) Compute I for the 3-component mixture from Demo 3
    d) Explain relationship between I and "difficulty" of learning
    """)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("SCORE FUNCTIONS BASICS - DEMONSTRATIONS")
    print("=" * 80)
    print("\nThis module introduces score functions through examples and visualizations.")
    print("Follow along with the code and comments to build intuition.")
    print("\n")
    
    # Run demonstrations
    demo_single_gaussian()
    demo_anisotropic_gaussian()
    demo_gaussian_mixture()
    demo_score_properties()
    
    # Show exercises
    exercises()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATIONS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review the generated plots")
    print("  2. Experiment with different parameters")
    print("  3. Complete the exercises")
    print("  4. Move on to 02_score_matching_theory.py")
    print("\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create output directory
    import os
    os.makedirs('/home/claude', exist_ok=True)
    
    # Run main demonstrations
    main()
    
    # Show all plots
    plt.show()
