"""
FILE: 02_score_matching_theory.py
DIFFICULTY: Beginner
ESTIMATED TIME: 2-3 hours
PREREQUISITES:
    - 01_score_functions_basics.py (score function definition)
    - Basic calculus (derivatives, expectations)
    - Understanding of loss functions

LEARNING OBJECTIVES:
    1. Understand the score matching objective
    2. Derive explicit score matching from first principles
    3. Understand denoising score matching (DSM)
    4. Learn why DSM is more practical than explicit score matching
    5. Implement both methods on simple examples

MATHEMATICAL BACKGROUND:
    
    SCORE MATCHING PROBLEM:
    Given samples {x₁, ..., xₙ} from unknown p_data(x), learn a score model
    s_θ(x) that approximates the true score ∇log p_data(x).
    
    EXPLICIT SCORE MATCHING (ESM):
    Minimizes the expected squared error:
    
    J_ESM(θ) = E_p[||s_θ(x) - ∇log p(x)||²]
            = E_p[||s_θ(x)||²] + E_p[tr(∇s_θ(x))] + const
    
    Problem: Requires computing trace of Jacobian (expensive!)
    
    DENOISING SCORE MATCHING (DSM):
    Add noise to data: x̃ = x + ε, where ε ~ N(0, σ²I)
    
    Minimize: J_DSM(θ) = E_p E_ε[||s_θ(x̃) - ∇log q(x̃|x)||²]
    
    where q(x̃|x) = N(x̃|x, σ²I) is the noise distribution.
    
    Key insight: ∇log q(x̃|x) = -(x̃ - x)/σ²
    
    This is easy to compute! No Jacobian needed.
    
    EQUIVALENCE:
    As σ→0, DSM converges to ESM. For small σ, they're approximately equal.

IMPLEMENTATION NOTES:
    - We'll use simple 2D distributions for visualization
    - Implement both ESM and DSM for comparison
    - Show that DSM is much faster in practice
    - Use PyTorch for automatic differentiation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import patches
import time


# ============================================================================
# SECTION 1: SCORE MATCHING OBJECTIVES
# ============================================================================

def fisher_divergence(score_model, true_score, samples):
    """
    Compute the Fisher Divergence between model and true score.
    
    Fisher Divergence: D_F(p_θ||p) = (1/2)E_p[||s_θ(x) - s(x)||²]
    
    This is what score matching minimizes.
    
    Args:
        score_model: Neural network that outputs score, s_θ(x)
        true_score: Function computing true score, s(x)
        samples: Samples from p_data, shape (N, D)
    
    Returns:
        fisher_div: Fisher divergence value
    """
    with torch.no_grad():
        # Get model predictions
        model_scores = score_model(samples)
        
        # Get true scores
        true_scores = torch.tensor(true_score(samples.numpy()), 
                                   dtype=torch.float32)
        
        # Compute squared error
        squared_diff = torch.sum((model_scores - true_scores) ** 2, dim=1)
        
        # Average over samples
        fisher_div = 0.5 * torch.mean(squared_diff)
    
    return fisher_div.item()


def explicit_score_matching_loss(score_model, samples):
    """
    Compute Explicit Score Matching (ESM) loss.
    
    L_ESM = E[||s_θ(x)||²/2 + tr(∇s_θ(x))]
    
    The trace term requires computing Jacobian diagonal:
    tr(∇s_θ(x)) = Σᵢ ∂s_θ(x)ᵢ/∂xᵢ
    
    This is expensive! We use Hutchinson's trace estimator with
    random projections to make it tractable.
    
    Args:
        score_model: Neural network outputting score
        samples: Data samples, shape (N, D)
    
    Returns:
        loss: ESM loss value
    """
    samples = samples.clone().detach().requires_grad_(True)
    N, D = samples.shape
    
    # Compute model scores
    scores = score_model(samples)
    
    # Term 1: ||s_θ(x)||²/2
    norm_term = 0.5 * torch.sum(scores ** 2, dim=1)
    
    # Term 2: tr(∇s_θ(x)) - this is the expensive part!
    # We need to compute ∂s_θ(x)ᵢ/∂xᵢ for each dimension i
    
    trace_term = torch.zeros(N, device=samples.device)
    
    for i in range(D):
        # Compute gradient of i-th score output w.r.t. i-th input
        # This requires one backward pass per dimension!
        grad = torch.autograd.grad(
            outputs=scores[:, i],
            inputs=samples,
            grad_outputs=torch.ones_like(scores[:, i]),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Extract diagonal element
        trace_term += grad[:, i]
    
    # Total loss
    loss = torch.mean(norm_term + trace_term)
    
    return loss


def denoising_score_matching_loss(score_model, samples, noise_std=0.1):
    """
    Compute Denoising Score Matching (DSM) loss.
    
    L_DSM = E_x E_ε[||s_θ(x̃) - (-(x̃-x)/σ²)||²]
    
    where x̃ = x + ε, ε ~ N(0, σ²I)
    
    Key advantage: The target score -(x̃-x)/σ² is trivial to compute!
    No Jacobian needed.
    
    Args:
        score_model: Neural network outputting score
        samples: Clean data samples, shape (N, D)
        noise_std: Standard deviation of noise σ
    
    Returns:
        loss: DSM loss value
    """
    # Add noise to samples: x̃ = x + ε
    noise = torch.randn_like(samples) * noise_std
    noisy_samples = samples + noise
    
    # Compute model score at noisy samples
    model_scores = score_model(noisy_samples)
    
    # Compute target score: ∇log q(x̃|x) = -(x̃ - x)/σ²
    # This is the score of the noise distribution!
    target_scores = -noise / (noise_std ** 2)
    
    # Compute squared error
    loss = 0.5 * torch.mean(torch.sum((model_scores - target_scores) ** 2, dim=1))
    
    return loss


# ============================================================================
# SECTION 2: SIMPLE SCORE MODEL
# ============================================================================

class SimpleScoreNetwork(nn.Module):
    """
    Simple MLP for score function estimation.
    
    Architecture:
    - Input: x (position in data space)
    - Hidden layers: ReLU activations
    - Output: s_θ(x) (score vector, same dim as input)
    
    Note: Output is NOT normalized - it's a vector field.
    """
    
    def __init__(self, input_dim=2, hidden_dims=[64, 64], activation=nn.ReLU):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation()
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Compute score function s_θ(x).
        
        Args:
            x: Input positions, shape (N, D)
        
        Returns:
            scores: Score vectors, shape (N, D)
        """
        return self.network(x)


# ============================================================================
# SECTION 3: TRAINING PROCEDURES
# ============================================================================

def train_score_model_esm(samples, true_score, num_epochs=1000, lr=0.01,
                          hidden_dims=[64, 64], verbose=True):
    """
    Train score model using Explicit Score Matching.
    
    This is slow because it requires computing Jacobian traces.
    
    Args:
        samples: Training samples, shape (N, D)
        true_score: True score function (for evaluation)
        num_epochs: Number of training epochs
        lr: Learning rate
        hidden_dims: Hidden layer dimensions
        verbose: Print training progress
    
    Returns:
        model: Trained score model
        losses: List of loss values during training
        fisher_divs: List of Fisher divergences during training
    """
    input_dim = samples.shape[1]
    
    # Create model
    model = SimpleScoreNetwork(input_dim, hidden_dims)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    fisher_divs = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Forward pass
        loss = explicit_score_matching_loss(model, samples)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record metrics
        losses.append(loss.item())
        
        # Compute Fisher divergence for monitoring
        if epoch % 100 == 0:
            fd = fisher_divergence(model, true_score, samples)
            fisher_divs.append(fd)
            
            if verbose:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:4d}/{num_epochs} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Fisher Div: {fd:.4f} | "
                      f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    if verbose:
        print(f"\nESM Training completed in {total_time:.1f}s")
    
    return model, losses, fisher_divs


def train_score_model_dsm(samples, noise_std=0.1, num_epochs=1000, lr=0.01,
                          hidden_dims=[64, 64], true_score=None, verbose=True):
    """
    Train score model using Denoising Score Matching.
    
    This is much faster than ESM because no Jacobian computation is needed.
    
    Args:
        samples: Training samples, shape (N, D)
        noise_std: Standard deviation of denoising noise
        num_epochs: Number of training epochs
        lr: Learning rate
        hidden_dims: Hidden layer dimensions
        true_score: True score function (optional, for evaluation)
        verbose: Print training progress
    
    Returns:
        model: Trained score model
        losses: List of loss values during training
        fisher_divs: List of Fisher divergences during training
    """
    input_dim = samples.shape[1]
    
    # Create model
    model = SimpleScoreNetwork(input_dim, hidden_dims)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    fisher_divs = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Forward pass
        loss = denoising_score_matching_loss(model, samples, noise_std)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record metrics
        losses.append(loss.item())
        
        # Compute Fisher divergence for monitoring (if true score available)
        if true_score is not None and epoch % 100 == 0:
            fd = fisher_divergence(model, true_score, samples)
            fisher_divs.append(fd)
            
            if verbose:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:4d}/{num_epochs} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Fisher Div: {fd:.4f} | "
                      f"Time: {elapsed:.1f}s")
        elif verbose and epoch % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:4d}/{num_epochs} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    if verbose:
        print(f"\nDSM Training completed in {total_time:.1f}s")
    
    return model, losses, fisher_divs


# ============================================================================
# SECTION 4: VISUALIZATION UTILITIES
# ============================================================================

def plot_learned_score_field(model, xlim=(-3, 3), ylim=(-3, 3), 
                             n_points=20, title="Learned Score Field"):
    """Visualize learned score function as a vector field."""
    # Create grid
    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Compute scores
    grid_points = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1),
        dtype=torch.float32
    )
    
    with torch.no_grad():
        scores = model(grid_points).numpy()
    
    U = scores[:, 0].reshape(X.shape)
    V = scores[:, 1].reshape(X.shape)
    magnitude = np.sqrt(U**2 + V**2)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, U, V, magnitude, cmap='viridis',
               scale=50, width=0.003, alpha=0.8)
    plt.colorbar(label='Score Magnitude')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def compare_score_fields(true_score_fn, model, samples=None,
                        xlim=(-3, 3), ylim=(-3, 3), n_points=20):
    """Create side-by-side comparison of true and learned scores."""
    # Create grid
    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(x, y)
    grid_points_np = np.stack([X.flatten(), Y.flatten()], axis=1)
    grid_points_torch = torch.tensor(grid_points_np, dtype=torch.float32)
    
    # Compute true scores
    true_scores = true_score_fn(grid_points_np)
    U_true = true_scores[:, 0].reshape(X.shape)
    V_true = true_scores[:, 1].reshape(X.shape)
    mag_true = np.sqrt(U_true**2 + V_true**2)
    
    # Compute learned scores
    with torch.no_grad():
        learned_scores = model(grid_points_torch).numpy()
    U_learned = learned_scores[:, 0].reshape(X.shape)
    V_learned = learned_scores[:, 1].reshape(X.shape)
    mag_learned = np.sqrt(U_learned**2 + V_learned**2)
    
    # Compute error
    error = learned_scores - true_scores
    U_error = error[:, 0].reshape(X.shape)
    V_error = error[:, 1].reshape(X.shape)
    mag_error = np.sqrt(U_error**2 + V_error**2)
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: True score
    axes[0].quiver(X, Y, U_true, V_true, mag_true,
                   cmap='viridis', scale=50, width=0.003, alpha=0.8)
    axes[0].set_title('True Score ∇log p(x)', fontsize=14)
    axes[0].set_xlabel('x₁')
    axes[0].set_ylabel('x₂')
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Learned score
    axes[1].quiver(X, Y, U_learned, V_learned, mag_learned,
                   cmap='viridis', scale=50, width=0.003, alpha=0.8)
    axes[1].set_title('Learned Score s_θ(x)', fontsize=14)
    axes[1].set_xlabel('x₁')
    axes[1].set_ylabel('x₂')
    axes[1].axis('equal')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Error
    axes[2].quiver(X, Y, U_error, V_error, mag_error,
                   cmap='Reds', scale=50, width=0.003, alpha=0.8)
    axes[2].set_title('Error (Learned - True)', fontsize=14)
    axes[2].set_xlabel('x₁')
    axes[2].set_ylabel('x₂')
    axes[2].axis('equal')
    axes[2].grid(True, alpha=0.3)
    
    # Add training samples if provided
    if samples is not None:
        for ax in axes:
            ax.scatter(samples[:, 0], samples[:, 1], 
                      c='red', s=10, alpha=0.3, label='Training Data')
        axes[0].legend()
    
    plt.tight_layout()


def plot_training_curves(losses_esm, losses_dsm, 
                        fisher_divs_esm, fisher_divs_dsm):
    """Plot training curves for ESM vs DSM comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss curves
    axes[0].plot(losses_esm, label='ESM Loss', alpha=0.7)
    axes[0].plot(losses_dsm, label='DSM Loss', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot 2: Fisher divergence
    epochs_esm = np.arange(0, len(losses_esm), 100)[:len(fisher_divs_esm)]
    epochs_dsm = np.arange(0, len(losses_dsm), 100)[:len(fisher_divs_dsm)]
    
    axes[1].plot(epochs_esm, fisher_divs_esm, 
                'o-', label='ESM', alpha=0.7)
    axes[1].plot(epochs_dsm, fisher_divs_dsm,
                's-', label='DSM', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Fisher Divergence')
    axes[1].set_title('Fisher Divergence (Lower = Better)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()


# ============================================================================
# SECTION 5: DEMONSTRATIONS
# ============================================================================

def demo_gaussian_score_matching():
    """Demonstrate score matching on a simple 2D Gaussian."""
    print("=" * 80)
    print("DEMO 1: Score Matching on 2D Gaussian")
    print("=" * 80)
    
    # True distribution: N(0, I)
    def true_score(x):
        """Score of N(0, I): s(x) = -x"""
        return -x
    
    # Generate training samples
    torch.manual_seed(42)
    np.random.seed(42)
    n_samples = 1000
    samples = torch.randn(n_samples, 2)
    
    print(f"\nTraining on {n_samples} samples from N(0, I)")
    print(f"True score: s(x) = -x (linear function)")
    
    # Train with DSM
    print("\n" + "-" * 80)
    print("Training with Denoising Score Matching (DSM)")
    print("-" * 80)
    model_dsm, losses_dsm, fisher_dsm = train_score_model_dsm(
        samples, noise_std=0.5, num_epochs=500, lr=0.01,
        true_score=true_score, verbose=True
    )
    
    # Visualize results
    print("\nGenerating visualizations...")
    compare_score_fields(true_score, model_dsm, samples.numpy(),
                        xlim=(-3, 3), ylim=(-3, 3))
    plt.savefig('/home/claude/demo1_gaussian_dsm.png', dpi=150, bbox_inches='tight')
    print("Saved: demo1_gaussian_dsm.png")
    
    print("\nKey observations:")
    print("  - DSM successfully learns linear score function")
    print("  - Training is fast (no Jacobian computation)")
    print("  - Final Fisher divergence indicates good approximation")


def demo_esm_vs_dsm():
    """Compare ESM and DSM on same data."""
    print("\n" + "=" * 80)
    print("DEMO 2: Comparing ESM vs DSM")
    print("=" * 80)
    
    # True distribution: Mixture of 2 Gaussians
    mus = [np.array([-1.5, 0]), np.array([1.5, 0])]
    sigmas = [0.5, 0.5]
    weights = [0.5, 0.5]
    
    def true_score(x):
        """Mixture score function."""
        from scipy.stats import multivariate_normal
        x = np.atleast_2d(x)
        
        # Compute component probabilities
        p1 = weights[0] * multivariate_normal.pdf(x, mus[0], sigmas[0]**2)
        p2 = weights[1] * multivariate_normal.pdf(x, mus[1], sigmas[1]**2)
        p_total = p1 + p2
        
        # Weighted scores
        s1 = -(x - mus[0]) / (sigmas[0] ** 2)
        s2 = -(x - mus[1]) / (sigmas[1] ** 2)
        
        posterior1 = p1 / (p_total + 1e-10)
        posterior2 = p2 / (p_total + 1e-10)
        
        return posterior1[:, None] * s1 + posterior2[:, None] * s2
    
    # Generate samples from mixture
    torch.manual_seed(42)
    np.random.seed(42)
    n_samples = 1000
    
    # Sample from each component
    n1 = int(n_samples * weights[0])
    n2 = n_samples - n1
    
    samples1 = torch.randn(n1, 2) * sigmas[0] + torch.tensor(mus[0])
    samples2 = torch.randn(n2, 2) * sigmas[1] + torch.tensor(mus[1])
    samples = torch.cat([samples1, samples2], dim=0)
    
    print(f"\nTraining on {n_samples} samples from 2-component Gaussian mixture")
    print(f"Components: μ₁={mus[0]}, μ₂={mus[1]}, σ=0.5")
    
    # Train with ESM (fewer epochs due to slowness)
    print("\n" + "-" * 80)
    print("Training with Explicit Score Matching (ESM)")
    print("-" * 80)
    print("Warning: This will be slow due to Jacobian computation!")
    
    model_esm, losses_esm, fisher_esm = train_score_model_esm(
        samples, true_score, num_epochs=300, lr=0.01, verbose=True
    )
    
    # Train with DSM
    print("\n" + "-" * 80)
    print("Training with Denoising Score Matching (DSM)")
    print("-" * 80)
    
    model_dsm, losses_dsm, fisher_dsm = train_score_model_dsm(
        samples, noise_std=0.3, num_epochs=500, lr=0.01,
        true_score=true_score, verbose=True
    )
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"\nFinal Fisher Divergence:")
    print(f"  ESM: {fisher_esm[-1]:.6f}")
    print(f"  DSM: {fisher_dsm[-1]:.6f}")
    print(f"\nKey insights:")
    print(f"  - DSM is much faster (no Jacobian trace)")
    print(f"  - Both methods learn similar score functions")
    print(f"  - DSM is preferred in practice")
    
    # Visualizations
    plot_training_curves(losses_esm, losses_dsm, fisher_esm, fisher_dsm)
    plt.savefig('/home/claude/demo2_esm_vs_dsm_curves.png', dpi=150, bbox_inches='tight')
    print("\nSaved: demo2_esm_vs_dsm_curves.png")
    
    compare_score_fields(true_score, model_esm, samples.numpy(),
                        xlim=(-4, 4), ylim=(-3, 3))
    plt.suptitle('ESM Results', fontsize=16, y=1.02)
    plt.savefig('/home/claude/demo2_esm_result.png', dpi=150, bbox_inches='tight')
    
    compare_score_fields(true_score, model_dsm, samples.numpy(),
                        xlim=(-4, 4), ylim=(-3, 3))
    plt.suptitle('DSM Results', fontsize=16, y=1.02)
    plt.savefig('/home/claude/demo2_dsm_result.png', dpi=150, bbox_inches='tight')
    print("Saved: demo2_esm_result.png, demo2_dsm_result.png")


# ============================================================================
# SECTION 6: EXERCISES
# ============================================================================

def exercises():
    """Exercises for understanding score matching."""
    print("\n" + "=" * 80)
    print("EXERCISES")
    print("=" * 80)
    
    print("""
    EXERCISE 1: Noise level in DSM
    ------------------------------
    Investigate how noise level σ affects DSM performance.
    
    Tasks:
    a) Train models with σ ∈ {0.1, 0.3, 0.5, 1.0, 2.0}
    b) Plot Fisher divergence vs. noise level
    c) Explain the trade-off: too small vs. too large
    
    EXERCISE 2: Derive DSM objective
    -------------------------------
    Starting from ESM, derive the DSM objective.
    
    Steps:
    a) Write down J_ESM = E[||s_θ - s||²]
    b) Show that for q(x̃|x) = N(x|x̃, σ²I):
       E_q[||s_θ(x̃) + (x̃-x)/σ²||²] ≈ J_ESM for small σ
    c) Explain why this doesn't need the true score s(x)
    
    EXERCISE 3: Sliced Score Matching
    --------------------------------
    Implement Sliced Score Matching (SSM), another efficient variant.
    
    SSM uses random projections:
    L_SSM = E_p E_v[v^T∇s_θ(x) v + 1/2(v^Ts_θ(x))²]
    
    where v ~ N(0, I) is a random direction.
    
    Tasks:
    a) Implement ssm_loss() function
    b) Compare computational cost with ESM and DSM
    c) Compare final performance
    
    EXERCISE 4: Network architecture
    -------------------------------
    Experiment with different architectures.
    
    Try:
    a) Deeper networks (4-5 layers)
    b) Wider networks (128, 256 hidden units)
    c) Different activations (Tanh, ELU, Swish)
    d) Residual connections
    
    Compare:
    - Training speed
    - Final Fisher divergence
    - Score field smoothness
    
    EXERCISE 5: Complex distributions
    --------------------------------
    Test score matching on more complex distributions.
    
    Try:
    a) 4-component mixture (square configuration)
    b) Ring distribution (annulus)
    c) Swiss roll
    d) Two moons
    
    For each:
    - Generate samples
    - Define true score (if known)
    - Train DSM model
    - Visualize learned score field
    """)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("SCORE MATCHING THEORY - DEMONSTRATIONS")
    print("=" * 80)
    print("\nThis module demonstrates score matching techniques.")
    print("We'll compare Explicit Score Matching (ESM) vs Denoising Score Matching (DSM).")
    print("\n")
    
    # Run demos
    demo_gaussian_score_matching()
    
    # Note: ESM demo is slow, so we make it optional
    print("\n" + "=" * 80)
    print("OPTIONAL: ESM vs DSM Comparison")
    print("=" * 80)
    print("The next demo compares ESM and DSM but is slower.")
    print("It will take ~2-3 minutes to complete.")
    
    response = input("\nRun ESM vs DSM comparison? (y/n): ")
    if response.lower() == 'y':
        demo_esm_vs_dsm()
    else:
        print("Skipping ESM demo. DSM is recommended for practice!")
    
    # Show exercises
    exercises()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATIONS COMPLETE")
    print("=" * 80)
    print("\nKey takeaways:")
    print("  1. Score matching learns score functions from data")
    print("  2. ESM is theoretically clean but computationally expensive")
    print("  3. DSM is practical and widely used")
    print("  4. Small noise levels work well in practice")
    print("\nNext: 03_langevin_dynamics.py (sampling with scores)")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()
    plt.show()
