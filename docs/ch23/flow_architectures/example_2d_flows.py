"""
2D Normalizing Flows: Complete Visualization and Training Examples

================================================================================
EDUCATIONAL PURPOSE
================================================================================
This script demonstrates normalizing flows on 2D toy datasets, providing
immediate visual feedback on what flows learn and how they transform data.

Perfect for:
- Understanding flow transformations visually
- Debugging flow implementations
- Experimenting with different architectures
- Building intuition before tackling high-dimensional problems

KEY LEARNING OUTCOMES:
1. See how flows progressively transform Gaussian → complex distribution
2. Understand the role of number of layers and hidden dimensions
3. Visualize training dynamics and convergence
4. Compare generated samples with real data
5. Observe how different datasets require different model capacities

PREREQUISITES:
- Understanding of flow_utils.py (base concepts)
- Familiarity with coupling_flows.py (RealNVP architecture)
- Basic knowledge of training neural networks
- Understanding of 2D visualization

WHAT THIS SCRIPT DOES:
1. Generates 2D toy datasets (moons, circles, spirals, checkerboards)
2. Trains a RealNVP flow model to match the distribution
3. Creates comprehensive visualizations:
   - Original data distribution
   - Training loss curve
   - Latent → data transformation
   - Learned probability density
   - Evolution through flow layers
   - Generated vs real data comparison

================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from tqdm import tqdm

from flow_utils import BaseDistribution, FlowSequence, visualize_2d_transformation, visualize_2d_density
from coupling_flows import build_realnvp_model


def generate_toy_data(dataset_name: str = 'moons', n_samples: int = 2000) -> torch.Tensor:
    """
    Generate 2D toy datasets for testing and visualizing flows.
    
    ============================================================================
    AVAILABLE DATASETS
    ============================================================================
    
    1. MOONS (Two Crescent Shapes)
       - Classic benchmark for non-linear methods
       - Tests model's ability to capture curved structures
       - Difficulty: EASY-MEDIUM
       - Good for: Initial testing, quick experiments
    
    2. CIRCLES (Two Concentric Circles)
       - Tests radial symmetry modeling
       - Requires model to separate based on distance from origin
       - Difficulty: MEDIUM
       - Good for: Understanding radial transformations
    
    3. SPIRAL (Two Intertwined Spirals)
       - Most challenging 2D distribution
       - Requires many layers to model well
       - Tests long-range dependencies
       - Difficulty: HARD
       - Good for: Stress-testing model capacity
    
    4. CHECKERBOARD (Grid Pattern)
       - Tests multi-modal distribution learning
       - Requires capturing regular patterns
       - Difficulty: MEDIUM-HARD
       - Good for: Understanding mode coverage
    
    ============================================================================
    DATASET CHARACTERISTICS
    ============================================================================
    
    All datasets are:
    - Normalized to roughly fit in [-4, 4] × [-4, 4]
    - Add small Gaussian noise for smoothness
    - Balanced between classes/modes
    - Designed to be learnable with 4-8 coupling layers
    
    ============================================================================
    
    Args:
        dataset_name (str): One of ['moons', 'circles', 'spiral', 'checkerboard']
        n_samples (int): Number of 2D points to generate
    
    Returns:
        torch.Tensor: 2D points, shape (n_samples, 2)
    
    Example Usage:
        >>> # Generate different datasets
        >>> moons = generate_toy_data('moons', n_samples=2000)
        >>> circles = generate_toy_data('circles', n_samples=2000)
        >>> 
        >>> # Plot to visualize
        >>> plt.scatter(moons[:, 0], moons[:, 1])
        >>> plt.show()
    
    Difficulty Recommendations:
        - Starting out? Try 'moons' with 4-6 layers
        - Want a challenge? Try 'spiral' with 8-12 layers
        - Testing capacity? Try all datasets with same architecture
    """
    
    if dataset_name == 'moons':
        # ==================== TWO MOONS ====================
        # Two half-moon shapes facing each other
        # Generated using sklearn's make_moons
        data, _ = make_moons(n_samples=n_samples, noise=0.05)
        
        # Note: noise=0.05 adds Gaussian noise to make smooth distribution
        # Lower noise = sharper moons (harder to model)
        # Higher noise = smoother moons (easier to model)
    
    elif dataset_name == 'circles':
        # ==================== TWO CIRCLES ====================
        # Two concentric circles with different radii
        # Tests radial/circular structure modeling
        data, _ = make_circles(n_samples=n_samples, noise=0.05, factor=0.5)
        
        # factor=0.5 means inner circle has radius 0.5 of outer circle
    
    elif dataset_name == 'spiral':
        # ==================== TWO SPIRALS ====================
        # Most challenging: two spirals wrapping around each other
        
        # Generate half the samples for each spiral
        n = n_samples // 2
        
        # Spiral parameter: angle increases with radius
        theta = np.sqrt(np.random.rand(n)) * 2 * np.pi
        
        # First spiral: grows outward counterclockwise
        r_a = 2 * theta + np.pi
        data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
        
        # Second spiral: grows outward clockwise (negative angle)
        r_b = -2 * theta - np.pi
        data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
        
        # Combine both spirals
        data = np.concatenate([data_a, data_b])
        
        # Add noise for smooth distribution
        data = data + np.random.randn(*data.shape) * 0.2
    
    elif dataset_name == 'checkerboard':
        # ==================== CHECKERBOARD ====================
        # Grid of squares with alternating density
        # Tests multi-modal distribution learning
        
        # Random x coordinates
        x1 = np.random.rand(n_samples) * 4 - 2
        
        # y coordinates depend on which square we're in
        x2_ = np.random.rand(n_samples) - np.random.randint(0, 2, n_samples) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        
        # Combine and scale
        data = np.concatenate([x1[:, None], x2[:, None]], axis=1)
        data = data * 2
    
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Choose from: 'moons', 'circles', 'spiral', 'checkerboard'"
        )
    
    # Convert to PyTorch tensor
    return torch.tensor(data, dtype=torch.float32)


def plot_data(data: torch.Tensor, filename: str = 'data.png', title: str = 'Data'):
    """
    Plot 2D data points.
    
    Simple utility to visualize 2D datasets before training.
    
    Args:
        data (torch.Tensor): 2D points, shape (n_samples, 2)
        filename (str): Where to save the plot
        title (str): Plot title
    
    Visualization Tips:
        - Blue points = data samples
        - Alpha=0.5 for transparency (see density)
        - Equal aspect ratio shows true geometric relationships
        - Grid helps assess scale and position
    """
    plt.figure(figsize=(6, 6))
    
    # Scatter plot with transparency
    plt.scatter(data[:, 0].numpy(), data[:, 1].numpy(), 
               alpha=0.5, s=10, color='blue')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    
    # Set consistent limits for comparison across plots
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    
    # Equal aspect ratio so circles look like circles
    plt.axis('equal')
    
    # Grid for reference
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    
    print(f"✓ Saved {filename}")


def train_flow_2d(data: torch.Tensor, n_layers: int = 6, hidden_dim: int = 64,
                 n_epochs: int = 500, batch_size: int = 256, lr: float = 1e-3):
    """
    Train a normalizing flow on 2D data with progress tracking.
    
    ============================================================================
    TRAINING PROCESS
    ============================================================================
    
    What happens during training:
    1. Sample batch of real data
    2. Compute log p(x) for the batch
       - Map x → z through inverse
       - Evaluate log p(z) + log|det|
    3. Minimize -log p(x) (maximize likelihood)
    4. Update parameters via gradient descent
    5. Repeat for all batches (one epoch)
    6. Repeat for multiple epochs until convergence
    
    ============================================================================
    HYPERPARAMETER TUNING GUIDE
    ============================================================================
    
    n_layers (Number of Coupling Layers):
        - Too few: Can't model complex distributions
        - Too many: Slower, might overfit
        - Sweet spot for 2D: 4-8 layers
        - For spirals: 8-12 layers
        - For simple moons/circles: 4-6 layers
    
    hidden_dim (Network Capacity):
        - Too small: Limited expressiveness
        - Too large: Slower, might overfit
        - Sweet spot for 2D: 64-128
        - Can go to 256 for very complex distributions
    
    n_epochs (Training Duration):
        - Too few: Underfitting
        - Too many: Overfitting (less common with flows)
        - Sweet spot: 300-500 for 2D
        - Watch the loss curve to decide
    
    batch_size:
        - Smaller: Noisier gradients, might help escape local minima
        - Larger: Smoother gradients, more stable
        - Sweet spot for 2D: 128-256
        - For 2000 samples: 256 works well
    
    lr (Learning Rate):
        - Too small: Very slow convergence
        - Too large: Unstable training, might diverge
        - Sweet spot: 1e-3 to 1e-4
        - Use learning rate decay for best results
    
    ============================================================================
    MONITORING TRAINING
    ============================================================================
    
    Signs of Good Training:
        ✓ Loss steadily decreases
        ✓ Progress bar shows improving loss
        ✓ No NaN or Inf values
        ✓ Final loss is negative (flows typically have negative log-likelihood)
    
    Signs of Problems:
        ✗ Loss oscillates wildly → Reduce learning rate
        ✗ Loss increases → Learning rate too high
        ✗ Loss stuck at start → Model too simple or bug in code
        ✗ NaN values → Numerical instability, check Jacobian computation
    
    ============================================================================
    
    Args:
        data (torch.Tensor): Training data, shape (n_samples, 2)
        n_layers (int): Number of coupling layers in model
        hidden_dim (int): Hidden dimension for coupling networks
        n_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate
    
    Returns:
        Tuple containing:
            model: Trained flow model
            losses: List of training losses (one per epoch)
    
    Example Usage:
        >>> # Standard training
        >>> data = generate_toy_data('moons', n_samples=2000)
        >>> model, losses = train_flow_2d(
        ...     data,
        ...     n_layers=6,
        ...     hidden_dim=64,
        ...     n_epochs=500,
        ...     batch_size=256,
        ...     lr=1e-3
        ... )
        >>> 
        >>> # Quick test with fewer epochs
        >>> model, losses = train_flow_2d(
        ...     data,
        ...     n_layers=4,
        ...     n_epochs=100,  # Faster
        ...     lr=5e-3  # Higher LR for faster convergence
        ... )
    """
    # ==================== DEVICE SETUP ====================
    # Use GPU if available for faster training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # ==================== MODEL CONSTRUCTION ====================
    print("\nBuilding model...")
    model = build_realnvp_model(
        dim=2,  # 2D data
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        use_batchnorm=True  # Usually helps training
    )
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}")
    print(f"  Number of layers: {len(model.flows)}")
    
    # ==================== OPTIMIZER SETUP ====================
    # Adam is the standard choice for flows
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Optional: Learning rate scheduler for better convergence
    # Uncomment to use:
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    # ==================== DATA LOADER ====================
    # PyTorch DataLoader handles batching and shuffling
    dataset = TensorDataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle each epoch
        num_workers=0  # Keep 0 for compatibility
    )
    
    print(f"\nDataset: {len(data)} samples")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # ==================== TRAINING LOOP ====================
    print(f"\nTraining for {n_epochs} epochs...")
    print("=" * 60)
    
    losses = []  # Store loss history
    
    # Progress bar for nice output
    pbar = tqdm(range(n_epochs), desc="Training")
    
    for epoch in pbar:
        epoch_loss = 0.0
        n_batches = 0
        
        # Iterate over batches
        for batch in dataloader:
            # Extract data (DataLoader returns tuple)
            batch = batch[0].to(device)
            
            # ==================== FORWARD PASS ====================
            # Compute negative log-likelihood
            log_prob = model.log_prob(batch)
            loss = -log_prob.mean()  # Negative for minimization
            
            # ==================== BACKWARD PASS ====================
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()        # Compute gradients
            
            # Optional: Gradient clipping for stability
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()       # Update parameters
            
            # ==================== BOOKKEEPING ====================
            epoch_loss += loss.item()
            n_batches += 1
        
        # Average loss for this epoch
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        # Update progress bar
        pbar.set_description(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}")
        
        # Optional: Update learning rate
        # scheduler.step()
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Final loss: {losses[-1]:.4f}")
    
    return model, losses


def visualize_flow_evolution(data: torch.Tensor, model: nn.Module, 
                            n_samples: int = 500):
    """
    Visualize how samples evolve through each layer of the flow.
    
    ============================================================================
    PURPOSE
    ============================================================================
    
    This visualization is INCREDIBLY INSIGHTFUL! It shows:
    1. How each layer progressively transforms the distribution
    2. Which layers do the most work
    3. Whether layers are redundant (no change)
    4. The gradual transformation from Gaussian → target distribution
    
    ============================================================================
    INTERPRETATION GUIDE
    ============================================================================
    
    What to Look For:
    
    FIRST PLOT (Base):
        - Should be circular Gaussian blob
        - Centered at origin
        - This never changes (base distribution)
    
    INTERMEDIATE PLOTS:
        - Each plot shows output of one layer
        - Watch for progressive refinement
        - Early layers: Coarse transformations
        - Later layers: Fine details
    
    FINAL PLOT:
        - Should match target distribution
        - Compare with original data
        - If very different → Need more layers or capacity
    
    Common Patterns:
        ✓ Smooth progression → Good architecture
        ✓ Big changes in some layers → These layers are important
        ✗ No change across many layers → Might be redundant
        ✗ Sudden chaotic change → Numerical instability
    
    ============================================================================
    
    Args:
        data (torch.Tensor): Original training data (for comparison)
        model (nn.Module): Trained flow model
        n_samples (int): Number of samples to track through layers
    
    Output:
        Saves 'flow_evolution.png' showing transformation at each layer
    
    Example Usage:
        >>> model, losses = train_flow_2d(data, n_layers=8)
        >>> visualize_flow_evolution(data, model, n_samples=500)
        >>> # Open flow_evolution.png to see the transformation!
    """
    # Set to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    print("\nGenerating flow evolution visualization...")
    
    # ==================== SAMPLE FROM BASE ====================
    z = model.base_dist.sample(n_samples, device=device)
    
    # ==================== TRACK THROUGH LAYERS ====================
    intermediates = [z.cpu()]  # Start with base samples
    x = z
    
    with torch.no_grad():  # No gradients needed for visualization
        # Apply each flow layer and store result
        for i, flow in enumerate(model.flows):
            x, _ = flow.forward(x)
            intermediates.append(x.cpu())
            print(f"  Layer {i+1}/{len(model.flows)} processed")
    
    # ==================== CREATE PLOT ====================
    n_plots = len(intermediates)
    n_cols = min(5, n_plots)  # Max 5 columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    
    # Handle case of single row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each intermediate transformation
    for idx, intermediate in enumerate(intermediates):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        intermediate = intermediate.numpy()
        
        # Scatter plot
        ax.scatter(intermediate[:, 0], intermediate[:, 1], 
                  alpha=0.5, s=5, color='blue')
        
        # Title
        if idx == 0:
            ax.set_title('Base Distribution\n(Gaussian)', fontweight='bold')
        elif idx == len(intermediates) - 1:
            ax.set_title(f'Final Output\n(Layer {idx})', fontweight='bold')
        else:
            ax.set_title(f'After Layer {idx}')
        
        # Formatting
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x₁', fontsize=8)
        ax.set_ylabel('x₂', fontsize=8)
    
    # Hide extra subplots
    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Flow Evolution: Gaussian → Target Distribution', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('flow_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved flow_evolution.png")
    print("  Compare how distribution changes through layers!")


def compare_distributions(data: torch.Tensor, model: nn.Module, 
                         n_samples: int = 2000):
    """
    Side-by-side comparison of real data vs generated samples.
    
    ============================================================================
    PURPOSE
    ============================================================================
    
    The ultimate test: Do generated samples look like real data?
    
    This visualization answers:
    1. Did the model learn the data distribution?
    2. Are generated samples diverse?
    3. Are there any obvious artifacts or failures?
    4. Does the model cover all modes of the distribution?
    
    ============================================================================
    QUALITY ASSESSMENT
    ============================================================================
    
    Signs of GOOD Generation:
        ✓ Left and right plots look similar
        ✓ Generated samples spread similarly to real data
        ✓ All modes of distribution are covered
        ✓ No obvious geometric distortions
    
    Signs of PROBLEMS:
        ✗ Generated samples are clustered (mode collapse)
        ✗ Missing parts of the distribution
        ✗ Obvious artifacts or strange patterns
        ✗ Very different density in regions
    
    Common Issues and Solutions:
        - Mode collapse → More layers, different architecture
        - Missing modes → Train longer, check loss convergence
        - Artifacts → Check for numerical instability
        - Low diversity → Might need more capacity
    
    ============================================================================
    
    Args:
        data (torch.Tensor): Original training data
        model (nn.Module): Trained flow model
        n_samples (int): Number of samples to generate for comparison
    
    Output:
        Saves 'comparison.png' with side-by-side plots
    
    Tip:
        Look at this plot to judge if training was successful!
    """
    # Set to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    print("\nGenerating samples for comparison...")
    
    # ==================== GENERATE SAMPLES ====================
    samples = model.sample(n_samples, device=device).cpu()
    
    print(f"  Generated {n_samples} samples")
    
    # ==================== CREATE COMPARISON PLOT ====================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # LEFT: Original data
    axes[0].scatter(data[:, 0].numpy(), data[:, 1].numpy(), 
                   alpha=0.5, s=10, color='blue')
    axes[0].set_title('Original Data', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('x₁')
    axes[0].set_ylabel('x₂')
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # RIGHT: Generated samples
    axes[1].scatter(samples[:, 0].detach().numpy(), samples[:, 1].detach().numpy(), 
                   alpha=0.5, s=10, color='red')
    axes[1].set_title('Generated Samples', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('x₁')
    axes[1].set_ylabel('x₂')
    axes[1].set_xlim(-4, 4)
    axes[1].set_ylim(-4, 4)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150)
    plt.close()
    
    print("✓ Saved comparison.png")
    print("  Blue (left) = real data, Red (right) = generated")
    print("  Compare to assess model quality!")


def main():
    """
    Main function: Complete 2D normalizing flow demonstration.
    
    ============================================================================
    WHAT THIS SCRIPT DOES
    ============================================================================
    
    1. Configuration: Set dataset and hyperparameters
    2. Data Generation: Create toy 2D dataset
    3. Training: Train normalizing flow model
    4. Visualization: Create comprehensive plots
       - Original data
       - Training loss curve
       - Latent ↔ data transformation
       - Learned probability density
       - Evolution through flow layers
       - Generated vs real comparison
    5. Metrics: Report final statistics
    
    ============================================================================
    EXPERIMENT GUIDE
    ============================================================================
    
    Beginner Experiments:
        1. Run with defaults → Learn what to expect
        2. Change dataset → See how different shapes require different capacity
        3. Vary n_layers → Understand layer count vs expressiveness
        4. Vary n_epochs → See convergence behavior
    
    Intermediate Experiments:
        1. Compare different architectures on same data
        2. Find minimum layers needed for each dataset
        3. Experiment with learning rates
        4. Try with/without batch normalization
    
    Advanced Experiments:
        1. Implement new toy datasets
        2. Add learning rate scheduling
        3. Try different coupling architectures
        4. Measure metrics like MMD or FID
    
    ============================================================================
    """
    print("=" * 70)
    print("2D NORMALIZING FLOWS: INTERACTIVE DEMONSTRATION")
    print("=" * 70)
    print("\nThis script trains a normalizing flow on 2D toy data")
    print("and creates comprehensive visualizations of the results.")
    print()
    
    # ==================== CONFIGURATION ====================
    print("[CONFIGURATION]")
    print("-" * 70)
    
    # Dataset selection
    # Try: 'moons', 'circles', 'spiral', 'checkerboard'
    dataset_name = 'moons'
    n_samples = 2000
    
    # Model architecture
    n_layers = 6        # Number of coupling layers (4-12 typical)
    hidden_dim = 64     # Hidden dimension in networks (64-256 typical)
    
    # Training hyperparameters
    n_epochs = 500      # Training epochs (300-500 typical for 2D)
    batch_size = 256    # Batch size (128-256 typical)
    lr = 1e-3          # Learning rate (1e-3 to 1e-4 typical)
    
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {n_samples}")
    print(f"Architecture: {n_layers} coupling layers × {hidden_dim} hidden dim")
    print(f"Training: {n_epochs} epochs, batch size {batch_size}, lr {lr}")
    print()
    
    # ==================== STEP 1: GENERATE DATA ====================
    print("[STEP 1] Generating Data")
    print("-" * 70)
    
    data = generate_toy_data(dataset_name, n_samples)
    print(f"✓ Generated {n_samples} samples from '{dataset_name}' distribution")
    print(f"  Data shape: {data.shape}")
    print(f"  Data range: [{data.min():.2f}, {data.max():.2f}]")
    
    # Plot original data
    plot_data(data, 'original_data.png', 
             f'Original {dataset_name.title()} Data ({n_samples} samples)')
    
    # ==================== STEP 2: TRAIN FLOW ====================
    print("\n[STEP 2] Training Normalizing Flow")
    print("-" * 70)
    
    model, losses = train_flow_2d(
        data,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr
    )
    
    # ==================== STEP 3: TRAINING LOSS PLOT ====================
    print("\n[STEP 3] Analyzing Training")
    print("-" * 70)
    
    plt.figure(figsize=(10, 4))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Negative Log-Likelihood', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Annotate key points
    min_loss_epoch = np.argmin(losses)
    plt.annotate(f'Min: {losses[min_loss_epoch]:.4f}',
                xy=(min_loss_epoch, losses[min_loss_epoch]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150)
    plt.close()
    
    print("✓ Saved training_loss.png")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Best loss: {losses[min_loss_epoch]:.4f} (epoch {min_loss_epoch})")
    
    # ==================== STEP 4: VISUALIZATIONS ====================
    print("\n[STEP 4] Creating Visualizations")
    print("-" * 70)
    
    device = next(model.parameters()).device
    model.eval()  # Set to evaluation mode
    
    # Visualization 1: Transformation (latent → data)
    print("\n  [4.1] Latent ↔ Data Transformation")
    visualize_2d_transformation(
        model,
        n_points=2000,
        xlim=(-4, 4),
        ylim=(-4, 4),
        filename='transformation.png'
    )
    
    # Visualization 2: Learned density
    print("\n  [4.2] Learned Probability Density")
    visualize_2d_density(
        model,
        xlim=(-4, 4),
        ylim=(-4, 4),
        n_grid=200,
        filename='learned_density.png'
    )
    
    # Visualization 3: Evolution through layers
    print("\n  [4.3] Evolution Through Layers")
    visualize_flow_evolution(data, model, n_samples=500)
    
    # Visualization 4: Real vs generated comparison
    print("\n  [4.4] Real vs Generated Comparison")
    compare_distributions(data, model, n_samples=2000)
    
    # ==================== STEP 5: FINAL METRICS ====================
    print("\n[STEP 5] Computing Final Metrics")
    print("-" * 70)
    
    with torch.no_grad():
        data_device = data.to(device)
        
        # Log probability on training data
        log_prob = model.log_prob(data_device)
        avg_log_prob = log_prob.mean().item()
        std_log_prob = log_prob.std().item()
        
        print(f"Average log p(x) on training data: {avg_log_prob:.4f}")
        print(f"Std log p(x): {std_log_prob:.4f}")
        
        # Generate samples and check their quality
        samples = model.sample(1000, device=device)
        sample_log_prob = model.log_prob(samples)
        
        print(f"\nAverage log p(x) on generated samples: {sample_log_prob.mean().item():.4f}")
        print("  (Should be similar to training data log prob)")
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 70)
    
    print("\nGenerated Files:")
    print("  1. original_data.png        - Input data distribution")
    print("  2. training_loss.png        - Training convergence curve")
    print("  3. transformation.png       - Latent → Data transformation")
    print("  4. learned_density.png      - Probability density heatmap")
    print("  5. flow_evolution.png       - Evolution through each layer")
    print("  6. comparison.png           - Real data vs generated samples")
    
    print("\nNext Steps:")
    print("  • Open the PNG files to see results")
    print("  • Try different datasets by changing 'dataset_name'")
    print("  • Experiment with n_layers and hidden_dim")
    print("  • Compare results across different architectures")
    
    print("\nRecommended Experiments:")
    print("  1. Train on 'spiral' with 8-12 layers")
    print("  2. Compare 4 vs 8 layers on same dataset")
    print("  3. Try with/without batch normalization")
    print("  4. Test different learning rates")
    
    print("\n" + "=" * 70)
    print("Happy learning! 🎓")
    print("=" * 70)


# ============================================================================
# MODULE EXECUTION
# ============================================================================
if __name__ == "__main__":
    """
    Run the complete 2D normalizing flow demonstration.
    
    This is the entry point when running the script directly:
        python example_2d_flows.py
    
    The main() function will:
    1. Generate 2D toy data
    2. Train a normalizing flow model
    3. Create comprehensive visualizations
    4. Report metrics and results
    
    All outputs are saved as PNG files in the current directory.
    """
    main()
