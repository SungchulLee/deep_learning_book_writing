"""
Coupling Flows (RealNVP Style): Advanced Normalizing Flow Architectures

================================================================================
EDUCATIONAL PURPOSE
================================================================================
This module implements coupling-based normalizing flows, which are among the
most powerful and widely-used flow architectures. These are the flows that
enable high-quality image generation and density estimation.

KEY CONCEPTS COVERED:
1. Coupling layers (RealNVP architecture)
2. Affine transformations conditioned on input
3. Alternating masks for full expressiveness
4. Batch normalization as a flow
5. Checkerboard patterns for image data

PREREQUISITES:
- Complete understanding of flow_utils.py
- Comfortable with neural networks (MLPs, CNNs)
- Understanding of conditional transformations
- Basic knowledge of batch normalization

WHY COUPLING FLOWS?
- Efficient: O(d) Jacobian computation instead of O(d³)
- Expressive: Can model complex distributions
- Invertible: Analyt analytical inverses (unlike planar flows)
- Scalable: Works for high-dimensional data

LEARNING PATH:
1. Understand CouplingLayer (the core building block)
2. See how AlternatingCouplingLayer ensures all dimensions transform
3. Learn about BatchNorm as a flow
4. Explore image-specific patterns (CheckerboardCouplingLayer)
5. Build complete models with build_realnvp_model()

================================================================================
"""

import torch
import torch.nn as nn
from typing import Tuple
from flow_utils import Flow


class CouplingLayer(Flow):
    """
    Affine Coupling Layer: The Heart of RealNVP
    
    ============================================================================
    CONCEPTUAL OVERVIEW
    ============================================================================
    The coupling layer is a clever trick that solves a major problem:
    
    PROBLEM: Computing Jacobian determinants is usually O(d³) - very expensive!
    SOLUTION: Split input and only transform part of it
    
    The Magic of Coupling:
        1. Split input: [x_a, x_b]  (e.g., first half and second half)
        2. Keep one part: y_a = x_a  (identity transform, no change)
        3. Transform other conditioned on first: y_b = x_b * s(x_a) + t(x_a)
        
    Where s() and t() are neural networks that take x_a as input.
    
    ============================================================================
    WHY THIS IS BRILLIANT
    ============================================================================
    
    1. EFFICIENT JACOBIAN:
       Because y_a = x_a (no change), the Jacobian is triangular:
       
       J = [I    0  ]
           [*  diag(s)]
       
       Determinant of triangular matrix = product of diagonal = exp(sum(log s))
       This is O(d) instead of O(d³)!
    
    2. EASY INVERSION:
       Forward:  y_b = x_b * s + t
       Inverse:  x_b = (y_b - t) / s
       
       Just subtract and divide! No iterative solver needed.
    
    3. EXPRESSIVENESS:
       The networks s() and t() can be arbitrarily complex (deep networks)
       They learn to transform the data in powerful ways
    
    ============================================================================
    MATHEMATICAL DETAILS
    ============================================================================
    
    Forward Pass:
        Given input x = [x_a, x_b] (split by mask)
        1. Compute scale: s(x_a)
        2. Compute translation: t(x_a)
        3. Transform: y_b = x_b ⊙ exp(s(x_a)) + t(x_a)
        4. Output: y = [x_a, y_b]
    
    Why exp(s)?
        - s can be any real number
        - exp(s) is always positive (needed for invertibility)
        - Numerically stable
    
    Jacobian:
        ∂y/∂x = [I      0    ]  (block triangular)
                [∂y_b/∂x_a  diag(exp(s))]
    
    Log Determinant:
        log|det(J)| = sum(s(x_a))
        
    This is the KEY INSIGHT that makes coupling flows fast!
    
    ============================================================================
    DESIGN CHOICES
    ============================================================================
    
    1. MASK SELECTION:
       - Common: First half / second half
       - Images: Checkerboard pattern
       - Rule: MUST alternate masks in sequence
    
    2. NETWORK ARCHITECTURE:
       - Typically: 2-3 layer MLPs with ReLU
       - Hidden dim: 64-256 typical
       - Output activation for s: Tanh (bounded for stability)
    
    3. STABILITY TRICKS:
       - Bound s with tanh
       - Scale tanh output (e.g., multiply by 2)
       - Initialize networks carefully
    
    ============================================================================
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64, mask: torch.Tensor = None):
        """
        Initialize a coupling layer.
        
        Args:
            dim (int): Input dimensionality (total size of x)
            
            hidden_dim (int): Hidden dimension for neural networks
                             Larger = more expressive but slower
                             Typical values: 64, 128, 256
            
            mask (torch.Tensor): Binary mask indicating which dimensions to transform
                                Shape: (dim,)
                                0 = keep unchanged (x_a)
                                1 = transform (x_b)
                                If None, transforms second half by default
        
        Network Architecture:
            Both s and t networks have the structure:
                Input (masked dims) → Linear + ReLU → Linear + ReLU → Output
                
        Example:
            >>> # Transform second half based on first half
            >>> layer = CouplingLayer(dim=10, hidden_dim=64)
            >>> 
            >>> # Custom mask: transform odd indices
            >>> mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            >>> layer = CouplingLayer(dim=10, hidden_dim=64, mask=mask)
        """
        super().__init__()
        
        # ==================== MASK SETUP ====================
        if mask is None:
            # Default behavior: split in half
            # First half unchanged, second half transformed
            mask = torch.zeros(dim)
            mask[dim // 2:] = 1  # Second half marked with 1
        
        # Register as buffer (not a parameter, but moves with model to GPU)
        self.register_buffer('mask', mask)
        
        # ==================== DIMENSION CALCULATIONS ====================
        # Count how many dimensions are in each part
        mask_sum = int(mask.sum().item())         # Number of 1s (part to keep)
        output_dim = dim - mask_sum               # Number of 0s (part to transform)
        
        # IMPORTANT: Networks take the MASKED part as input
        # and output transformations for the UNMASKED part
        
        # ==================== SCALE NETWORK ====================
        # Outputs log(scale) for numerical stability
        # We'll exponentiate later: s = exp(log_s)
        self.scale_net = nn.Sequential(
            # First layer: masked dims → hidden
            nn.Linear(mask_sum, hidden_dim),
            nn.ReLU(),
            
            # Middle layer: hidden → hidden (learn complex patterns)
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            # Output layer: hidden → unmasked dims
            nn.Linear(hidden_dim, output_dim),
            
            # CRITICAL: Tanh bounds output to [-1, 1]
            # This prevents extreme scale values that cause numerical issues
            nn.Tanh()
        )
        
        # ==================== TRANSLATION NETWORK ====================
        # Outputs translation (shift) values
        # No output activation - translation can be any value
        self.translation_net = nn.Sequential(
            nn.Linear(mask_sum, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            # No activation here - translation can be any real number
        )
        
        # ==================== SCALE FACTOR ====================
        # Since tanh outputs [-1, 1], we scale by this factor
        # log_s ∈ [-2, 2] means s ∈ [exp(-2), exp(2)] ≈ [0.135, 7.39]
        # This is a reasonable range for scaling factors
        self.scale_factor = 2.0
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation: z → x (latent to data)
        
        This is the GENERATIVE direction for sampling.
        
        Process:
            1. Split input according to mask
            2. Extract the part that conditions the transformation (z_a)
            3. Compute scale s(z_a) and translation t(z_a)
            4. Transform the other part: z_b → x_b
            5. Combine: x = [z_a, x_b]
            6. Compute log determinant
        
        Args:
            z (torch.Tensor): Latent samples, shape (batch_size, dim)
        
        Returns:
            Tuple containing:
                x (torch.Tensor): Transformed samples, shape (batch_size, dim)
                log_det (torch.Tensor): Log determinant, shape (batch_size,)
        
        Example:
            >>> layer = CouplingLayer(dim=4, hidden_dim=16)
            >>> z = torch.randn(10, 4)  # 10 samples
            >>> x, log_det = layer.forward(z)
            >>> print(x.shape, log_det.shape)
            torch.Size([10, 4]) torch.Size([10])
        """
        # ==================== STEP 1: SPLIT INPUT ====================
        # Separate into masked (unchanged) and unmasked (to transform)
        z_masked = z * self.mask              # Keep these dims
        z_unmasked = z * (1 - self.mask)      # Transform these dims
        
        # ==================== STEP 2: EXTRACT CONDITIONER ====================
        # Get the part that will condition the transformation
        # self.mask.bool() converts 0/1 to False/True for indexing
        z_a = z_masked[:, self.mask.bool()]   # Shape: (batch_size, mask_sum)
        
        # ==================== STEP 3: COMPUTE SCALE & TRANSLATION ====================
        # Pass conditioner through networks
        log_s = self.scale_net(z_a) * self.scale_factor   # Log scale
        t = self.translation_net(z_a)                     # Translation
        
        # Convert log scale to actual scale
        # This ensures s is always positive
        s = torch.exp(log_s)  # Shape: (batch_size, output_dim)
        
        # ==================== STEP 4: TRANSFORM ====================
        # Apply affine transformation to unmasked part
        # Formula: x_b = z_b * s + t
        x_unmasked = z_unmasked.clone()
        x_unmasked[:, (~self.mask.bool())] = z[:, (~self.mask.bool())] * s + t
        
        # ==================== STEP 5: COMBINE ====================
        # Masked part unchanged, unmasked part transformed
        x = z_masked + x_unmasked
        
        # ==================== STEP 6: LOG DETERMINANT ====================
        # For coupling: log|det(J)| = sum of log scales
        # This is the magic that makes coupling flows efficient!
        log_det = log_s.sum(dim=-1)  # Sum over dimensions, one value per sample
        
        return x, log_det
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation: x → z (data to latent)
        
        This is the INFERENCE direction for computing probabilities.
        
        Process:
            1. Split input according to mask
            2. Extract the conditioner (x_a = z_a, unchanged)
            3. Compute scale s(x_a) and translation t(x_a) (same as forward!)
            4. Inverse transform: x_b → z_b = (x_b - t) / s
            5. Combine: z = [x_a, z_b]
            6. Compute log determinant (negative of forward)
        
        Key Insight:
            The inverse is ANALYTICAL - just subtract and divide!
            No iterative solver needed (unlike planar flows).
        
        Args:
            x (torch.Tensor): Data samples, shape (batch_size, dim)
        
        Returns:
            Tuple containing:
                z (torch.Tensor): Latent samples, shape (batch_size, dim)
                log_det (torch.Tensor): Log determinant, shape (batch_size,)
        
        Mathematical Note:
            Forward: y_b = x_b * s + t
            Inverse: x_b = (y_b - t) / s
            
            Jacobian determinant inverts:
            log|det(J_inv)| = -log|det(J_forward)|
        
        Example:
            >>> layer = CouplingLayer(dim=4)
            >>> x = torch.randn(10, 4)
            >>> z, log_det = layer.inverse(x)
            >>> # Now we can compute log p(x) = log p(z) + log_det
        """
        # ==================== STEP 1: SPLIT INPUT ====================
        x_masked = x * self.mask
        x_unmasked = x * (1 - self.mask)
        
        # ==================== STEP 2: EXTRACT CONDITIONER ====================
        # Conditioner is the same as in forward!
        x_a = x_masked[:, self.mask.bool()]
        
        # ==================== STEP 3: COMPUTE SCALE & TRANSLATION ====================
        # Use SAME networks as forward pass
        log_s = self.scale_net(x_a) * self.scale_factor
        t = self.translation_net(x_a)
        s = torch.exp(log_s)
        
        # ==================== STEP 4: INVERSE TRANSFORM ====================
        # Formula: z_b = (x_b - t) / s
        # This is the inverse of: x_b = z_b * s + t
        z_unmasked = x_unmasked.clone()
        z_unmasked[:, (~self.mask.bool())] = (x[:, (~self.mask.bool())] - t) / s
        
        # ==================== STEP 5: COMBINE ====================
        z = x_masked + z_unmasked
        
        # ==================== STEP 6: LOG DETERMINANT ====================
        # Inverse determinant is negative of forward
        # log|det(J^-1)| = -log|det(J)|
        log_det = -log_s.sum(dim=-1)
        
        return z, log_det


class AlternatingCouplingLayer(nn.Module):
    """
    Alternating Coupling: Ensuring Full Expressiveness
    
    ============================================================================
    THE PROBLEM WITH SINGLE COUPLING LAYERS
    ============================================================================
    
    A single coupling layer has a limitation:
    - Part of input NEVER CHANGES (the masked part)
    - Only the unmasked part gets transformed
    
    Example with single layer:
        Input:  [x₁, x₂, x₃, x₄]
        Mask:   [ 0,  0,  1,  1]  (0 = keep, 1 = transform)
        Output: [x₁, x₂, y₃, y₄]  ← x₁, x₂ never changed!
    
    This means the model can't fully learn the distribution!
    
    ============================================================================
    THE SOLUTION: ALTERNATE MASKS
    ============================================================================
    
    Use TWO coupling layers with OPPOSITE masks:
    
    Layer 1 - Mask: [0, 0, 1, 1]
        Input:  [x₁, x₂, x₃, x₄]
        Output: [x₁, x₂, y₃, y₄]  ← x₃, x₄ transformed
    
    Layer 2 - Mask: [1, 1, 0, 0]  (opposite!)
        Input:  [x₁, x₂, y₃, y₄]
        Output: [z₁, z₂, y₃, y₄]  ← Now x₁, x₂ transformed too!
    
    Final: ALL dimensions have been transformed!
    
    ============================================================================
    DESIGN PATTERN
    ============================================================================
    
    In practice, we use MANY alternating coupling layers:
        [Coupling₁, Coupling₂, Coupling₃, Coupling₄, ...]
         mask=[0,1]  mask=[1,0]  mask=[0,1]  mask=[1,0]
    
    This creates a deep, expressive transformation where:
    - Every dimension influences every other dimension
    - Complex dependencies can be learned
    - Still maintains O(d) Jacobian computation!
    
    ============================================================================
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64):
        """
        Initialize alternating coupling layers.
        
        This is a convenience wrapper that creates two coupling layers
        with complementary masks, ensuring all dimensions get transformed.
        
        Args:
            dim (int): Input dimensionality
            hidden_dim (int): Hidden dimension for coupling networks
        
        Architecture Created:
            Layer 1: Transform second half based on first half
            Layer 2: Transform first half based on second half
            
        Example:
            >>> # Simple way to ensure full transformation
            >>> alternating = AlternatingCouplingLayer(dim=10, hidden_dim=64)
            >>> 
            >>> # This is equivalent to:
            >>> layer1 = CouplingLayer(dim=10, hidden_dim=64, mask=[0,0,0,0,0,1,1,1,1,1])
            >>> layer2 = CouplingLayer(dim=10, hidden_dim=64, mask=[1,1,1,1,1,0,0,0,0,0])
        """
        super().__init__()
        
        # ==================== FIRST MASK ====================
        # Transform second half
        mask1 = torch.zeros(dim)
        mask1[dim // 2:] = 1  # Ones in second half
        
        # ==================== SECOND MASK ====================
        # Transform first half (opposite of first mask)
        mask2 = 1 - mask1  # Invert the mask
        
        # ==================== CREATE LAYERS ====================
        self.coupling1 = CouplingLayer(dim, hidden_dim, mask1)
        self.coupling2 = CouplingLayer(dim, hidden_dim, mask2)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward through both coupling layers.
        
        Process:
            z → Coupling 1 → h → Coupling 2 → x
            
        Log determinants add:
            log|det(total)| = log|det(layer1)| + log|det(layer2)|
        
        Args:
            z: Input samples
        
        Returns:
            x: Transformed samples
            log_det: Combined log determinant
        """
        # Apply first coupling
        x, log_det1 = self.coupling1.forward(z)
        
        # Apply second coupling to result
        x, log_det2 = self.coupling2.forward(x)
        
        # Log determinants add in log space
        return x, log_det1 + log_det2
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse through both coupling layers in REVERSE order.
        
        Process:
            x → Coupling 2⁻¹ → h → Coupling 1⁻¹ → z
            
        Critical: Apply inverses in reverse order!
        
        Args:
            x: Transformed samples
        
        Returns:
            z: Original samples
            log_det: Combined log determinant
        """
        # IMPORTANT: Reverse order for inverse!
        # Last applied forward is first applied inverse
        z, log_det2 = self.coupling2.inverse(x)
        z, log_det1 = self.coupling1.inverse(z)
        
        return z, log_det1 + log_det2


class BatchNorm(Flow):
    """
    Batch Normalization as a Flow Layer
    
    ============================================================================
    CONCEPTUAL OVERVIEW
    ============================================================================
    
    Batch normalization can be viewed as a normalizing flow!
    
    Standard Batch Norm (in neural networks):
        1. Normalize: (x - mean) / std
        2. Scale and shift: γ * normalized + β
    
    As a Flow:
        - Forward: Normalize and scale (latent → data)
        - Inverse: Denormalize (data → latent)
        - Jacobian: Diagonal matrix (efficient!)
    
    ============================================================================
    WHY USE BATCH NORM IN FLOWS?
    ============================================================================
    
    Benefits:
        1. TRAINING STABILITY
           - Prevents internal covariate shift
           - Makes deeper flows trainable
           - Reduces sensitivity to initialization
        
        2. EXPRESSIVENESS
           - Acts as learned data-dependent transformation
           - Helps model different scales across dimensions
           - Can normalize heavy-tailed distributions
        
        3. EFFICIENCY
           - Diagonal Jacobian → O(d) determinant
           - Fast forward and inverse
           - Minimal parameter count
    
    Common Usage Pattern:
        [CouplingLayer, BatchNorm, CouplingLayer, BatchNorm, ...]
        
    The batch norm between couplings helps training!
    
    ============================================================================
    MATHEMATICAL DETAILS
    ============================================================================
    
    Forward (Training):
        1. Compute batch statistics:
           μ = mean(z)
           σ² = var(z)
        
        2. Normalize:
           z_norm = (z - μ) / √(σ² + ε)
        
        3. Scale and shift:
           x = exp(log_γ) * z_norm + β
        
        4. Update running statistics (exponential moving average)
    
    Forward (Evaluation):
        Use running statistics instead of batch statistics
    
    Inverse:
        x_norm = (x - β) / exp(log_γ)
        z = x_norm * √(σ² + ε) + μ
    
    Log Determinant:
        log|det(J)| = Σᵢ log_γᵢ - 0.5 * Σᵢ log(σᵢ² + ε)
    
    ============================================================================
    """
    
    def __init__(self, dim: int, momentum: float = 0.1, eps: float = 1e-5):
        """
        Initialize batch normalization flow layer.
        
        Args:
            dim (int): Number of dimensions
            
            momentum (float): Momentum for running statistics update
                             Higher = faster adaptation to recent batches
                             Typical: 0.1
                             
            eps (float): Small constant for numerical stability
                        Prevents division by zero
                        Typical: 1e-5
        
        Learned Parameters:
            - log_γ (log scale): Initialized to 0 → γ = 1 (no scaling initially)
            - β (shift): Initialized to 0 (no shift initially)
        
        Running Statistics (not learned, but updated during training):
            - running_mean: Exponential moving average of means
            - running_var: Exponential moving average of variances
        
        Example:
            >>> bn = BatchNorm(dim=10)
            >>> z = torch.randn(32, 10)  # Batch of 32 samples
            >>> x, log_det = bn.forward(z)
        """
        super().__init__()
        
        self.momentum = momentum
        self.eps = eps
        
        # ==================== LEARNABLE PARAMETERS ====================
        # We learn log(γ) instead of γ for numerical stability
        # exp is always positive, avoiding negative scales
        self.log_gamma = nn.Parameter(torch.zeros(dim))
        
        # Shift parameter (can be any value)
        self.beta = nn.Parameter(torch.zeros(dim))
        
        # ==================== RUNNING STATISTICS ====================
        # These are NOT learned, but updated during training
        # Used during evaluation (when batch statistics aren't reliable)
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward: Normalize and scale.
        
        Behavior differs between training and evaluation:
        - Training: Use batch statistics, update running statistics
        - Evaluation: Use running statistics (batch might be small/unreliable)
        
        Args:
            z (torch.Tensor): Input samples, shape (batch_size, dim)
        
        Returns:
            Tuple containing:
                x (torch.Tensor): Normalized samples
                log_det (torch.Tensor): Log determinant
        
        Mathematical Flow:
            z → normalize → scale → shift → x
        """
        if self.training:
            # ==================== TRAINING MODE ====================
            # Use batch statistics
            mean = z.mean(dim=0)  # Mean over batch
            var = z.var(dim=0)     # Variance over batch
            
            # Update running statistics using exponential moving average
            # new_running = (1 - momentum) * old + momentum * current
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # ==================== EVALUATION MODE ====================
            # Use accumulated running statistics
            mean = self.running_mean
            var = self.running_var
        
        # ==================== NORMALIZE ====================
        # Standardize to mean=0, var=1
        # Add eps to avoid division by zero
        x_normalized = (z - mean) / torch.sqrt(var + self.eps)
        
        # ==================== SCALE AND SHIFT ====================
        # Apply learned affine transformation
        gamma = torch.exp(self.log_gamma)  # Convert log scale to scale
        x = gamma * x_normalized + self.beta
        
        # ==================== LOG DETERMINANT ====================
        # Jacobian is diagonal with entries γ / √(σ² + ε)
        # log|det| = Σ log(γ) - 0.5 * Σ log(σ² + ε)
        log_det = self.log_gamma.sum() - 0.5 * torch.log(var + self.eps).sum()
        
        # Expand to batch size (same log_det for all samples in batch)
        log_det = log_det.expand(z.shape[0])
        
        return x, log_det
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse: Unscale and denormalize.
        
        This reverses the normalization process:
            x → unshift → unscale → denormalize → z
        
        Args:
            x (torch.Tensor): Normalized samples
        
        Returns:
            Tuple containing:
                z (torch.Tensor): Original samples
                log_det (torch.Tensor): Log determinant (negative of forward)
        
        Note:
            We ALWAYS use running statistics for inverse
            (Even in training mode, for consistency)
        """
        # Use running statistics for inverse
        if self.training:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # ==================== UNSCALE ====================
        # Reverse: γ * x_norm + β → x_norm
        gamma = torch.exp(self.log_gamma)
        x_normalized = (x - self.beta) / gamma
        
        # ==================== DENORMALIZE ====================
        # Reverse: (z - μ) / σ → z
        z = x_normalized * torch.sqrt(var + self.eps) + mean
        
        # ==================== LOG DETERMINANT ====================
        # Inverse determinant is negative of forward
        log_det = -self.log_gamma.sum() + 0.5 * torch.log(var + self.eps).sum()
        log_det = log_det.expand(x.shape[0])
        
        return z, log_det


def build_realnvp_model(dim: int, n_layers: int = 4, hidden_dim: int = 64,
                       use_batchnorm: bool = True) -> nn.Module:
    """
    Build a complete RealNVP-style flow model.
    
    ============================================================================
    MODEL ARCHITECTURE
    ============================================================================
    
    This function constructs a full normalizing flow model with:
    - Multiple coupling layers (alternating masks)
    - Optional batch normalization between layers
    - Base Gaussian distribution
    
    Default Pattern:
        [Coupling (mask1), BatchNorm, Coupling (mask2), BatchNorm, ...]
        
    The alternating masks ensure all dimensions get transformed.
    
    ============================================================================
    HYPERPARAMETER GUIDANCE
    ============================================================================
    
    dim: Dimensionality of data
        - 2D toy data: dim=2
        - MNIST: dim=784 (28×28)
        - CIFAR-10: dim=3072 (32×32×3)
    
    n_layers: Number of coupling layers
        - 2D toy: 4-8 layers
        - Images: 20-40 layers
        - More layers = more expressive but slower
    
    hidden_dim: Hidden dimension in coupling networks
        - Small data: 64-128
        - Large data: 256-512
        - Larger = more capacity but slower
    
    use_batchnorm: Whether to use batch norm
        - Usually True for better training
        - False if small batches or already normalized data
    
    ============================================================================
    EXAMPLE USAGE
    ============================================================================
    
    For 2D visualization:
        >>> model = build_realnvp_model(
        ...     dim=2, 
        ...     n_layers=6,
        ...     hidden_dim=64,
        ...     use_batchnorm=True
        ... )
        >>> 
        >>> # Train on toy data
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> train(model, data, optimizer)
        >>> 
        >>> # Sample new points
        >>> samples = model.sample(100)
        >>> 
        >>> # Evaluate log probability
        >>> log_probs = model.log_prob(data)
    
    For Image Generation:
        >>> model = build_realnvp_model(
        ...     dim=784,  # MNIST
        ...     n_layers=20,
        ...     hidden_dim=256,
        ...     use_batchnorm=True
        ... )
    
    ============================================================================
    
    Args:
        dim (int): Input dimensionality
        n_layers (int): Number of coupling layers
        hidden_dim (int): Hidden dimension for networks
        use_batchnorm (bool): Whether to add batch norm layers
    
    Returns:
        FlowSequence: Complete flow model ready for training
    """
    from flow_utils import FlowSequence, BaseDistribution
    
    # List to accumulate flow layers
    flows = []
    
    # Build alternating coupling layers
    for i in range(n_layers):
        # ==================== CREATE MASK ====================
        # Alternate between two complementary masks
        if i % 2 == 0:
            # Even layers: transform second half
            mask = torch.zeros(dim)
            mask[dim // 2:] = 1
        else:
            # Odd layers: transform first half
            mask = torch.zeros(dim)
            mask[:dim // 2] = 1
        
        # ==================== ADD COUPLING LAYER ====================
        flows.append(CouplingLayer(dim, hidden_dim, mask))
        
        # ==================== ADD BATCH NORM ====================
        # Optionally add batch norm after coupling
        # This helps training stability and expressiveness
        if use_batchnorm:
            flows.append(BatchNorm(dim))
    
    # ==================== CREATE COMPLETE MODEL ====================
    # Combine all flows with a Gaussian base distribution
    base_dist = BaseDistribution(dim)
    return FlowSequence(flows, base_dist)


class CheckerboardCouplingLayer(Flow):
    """
    Checkerboard Coupling for Spatial Data (Images)
    
    ============================================================================
    MOTIVATION
    ============================================================================
    
    For images, simple "split in half" masks are suboptimal:
    - Top half vs bottom half? Loses spatial structure
    - Left half vs right half? Same problem
    - Channels? Only works if you have many channels
    
    Solution: CHECKERBOARD PATTERN
    
    Checkerboard Mask (for 4×4 image):
        1 0 1 0
        0 1 0 1
        1 0 1 0
        0 1 0 1
    
    Benefits:
        - Preserves spatial locality
        - Every pixel has neighbors from both groups
        - Better inductive bias for images
    
    ============================================================================
    HOW IT WORKS
    ============================================================================
    
    Same principle as regular coupling, but:
    1. Mask is checkerboard pattern (not channel/spatial split)
    2. Networks are CONVOLUTIONAL (not fully connected)
    3. Works on 4D tensors: (batch, channels, height, width)
    
    Forward:
        - Masked pixels stay same (checkerboard pattern)
        - Unmasked pixels transformed based on masked pixels
        - CNNs capture spatial relationships
    
    ============================================================================
    ARCHITECTURE CHOICES
    ============================================================================
    
    Network Design:
        - Use 3×3 convolutions (local spatial context)
        - Use 1×1 convolutions (channel mixing)
        - Padding=1 to keep spatial dimensions
    
    Typical Pattern:
        Conv3×3 → ReLU → Conv1×1 → ReLU → Conv3×3
        
    This is more parameter-efficient than fully connected for images!
    
    ============================================================================
    """
    
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        """
        Initialize checkerboard coupling for images.
        
        Args:
            in_channels (int): Number of input channels
                              RGB images: 3
                              Grayscale: 1
                              Feature maps: any value
            
            hidden_channels (int): Number of hidden channels in networks
                                  Typical: 64-256
        
        Network Architecture:
            Both networks follow:
                Input → Conv3×3 + ReLU → Conv1×1 + ReLU → Conv3×3 → Output
            
        Example:
            >>> # For RGB images
            >>> layer = CheckerboardCouplingLayer(in_channels=3, hidden_channels=128)
            >>> 
            >>> # For grayscale images
            >>> layer = CheckerboardCouplingLayer(in_channels=1, hidden_channels=64)
        """
        super().__init__()
        self.in_channels = in_channels
        
        # ==================== SCALE NETWORK (CONVOLUTIONAL) ====================
        self.scale_net = nn.Sequential(
            # First conv: spatial context
            # 3×3 kernel with padding=1 keeps spatial size
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            
            # Middle conv: channel mixing
            # 1×1 kernel for efficient channel interactions
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            
            # Output conv: back to input channels
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1),
            
            # Tanh for bounded, stable scales
            nn.Tanh()
        )
        
        # ==================== TRANSLATION NETWORK ====================
        self.translation_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
            # No output activation for translation
        )
        
        # Scale factor for tanh output
        self.scale_factor = 2.0
    
    def _get_mask(self, height: int, width: int, device: str) -> torch.Tensor:
        """
        Create checkerboard mask for spatial dimensions.
        
        Pattern:
            1 0 1 0 1 0 ...
            0 1 0 1 0 1 ...
            1 0 1 0 1 0 ...
            0 1 0 1 0 1 ...
            ...
        
        Args:
            height (int): Image height
            width (int): Image width
            device (str): Device for tensor
        
        Returns:
            torch.Tensor: Binary mask, shape (1, 1, height, width)
        """
        # Create mask filled with zeros
        mask = torch.zeros(1, 1, height, width, device=device)
        
        # Set 1s in checkerboard pattern
        # Even rows, even columns
        mask[:, :, ::2, ::2] = 1
        # Odd rows, odd columns
        mask[:, :, 1::2, 1::2] = 1
        
        return mask
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation for 4D image tensors.
        
        Args:
            z (torch.Tensor): Latent images, shape (batch, channels, height, width)
        
        Returns:
            Tuple containing:
                x (torch.Tensor): Transformed images, same shape as z
                log_det (torch.Tensor): Log determinant, shape (batch,)
        
        Example:
            >>> layer = CheckerboardCouplingLayer(in_channels=3)
            >>> z = torch.randn(16, 3, 32, 32)  # 16 RGB images, 32×32
            >>> x, log_det = layer.forward(z)
            >>> x.shape
            torch.Size([16, 3, 32, 32])
        """
        b, c, h, w = z.shape
        
        # Get checkerboard mask
        mask = self._get_mask(h, w, z.device)
        
        # Masked part (stays same)
        z_masked = z * mask
        
        # Compute transformations using CNNs
        log_s = self.scale_net(z_masked) * self.scale_factor
        t = self.translation_net(z_masked)
        s = torch.exp(log_s)
        
        # Transform unmasked part
        x = z_masked + (1 - mask) * (z * s + t)
        
        # Log determinant (sum over unmasked pixels)
        log_det = (log_s * (1 - mask)).view(b, -1).sum(dim=1)
        
        return x, log_det
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation for images.
        
        Args:
            x (torch.Tensor): Transformed images
        
        Returns:
            Tuple containing:
                z (torch.Tensor): Original images
                log_det (torch.Tensor): Log determinant
        """
        b, c, h, w = x.shape
        
        mask = self._get_mask(h, w, x.device)
        
        x_masked = x * mask
        
        log_s = self.scale_net(x_masked) * self.scale_factor
        t = self.translation_net(x_masked)
        s = torch.exp(log_s)
        
        # Inverse transform
        z = x_masked + (1 - mask) * ((x - t) / s)
        
        log_det = -(log_s * (1 - mask)).view(b, -1).sum(dim=1)
        
        return z, log_det


# ============================================================================
# TESTING AND VERIFICATION
# ============================================================================
if __name__ == "__main__":
    """
    Comprehensive tests to verify coupling layer implementation.
    
    These tests check:
    1. Forward and inverse are true inverses
    2. Log determinants sum to zero (forward + inverse)
    3. Model can forward pass and sample
    4. Shapes are correct
    
    Run this to verify everything works before using in your projects!
    """
    print("=" * 70)
    print("TESTING COUPLING FLOWS IMPLEMENTATION")
    print("=" * 70)
    
    # Configuration
    dim = 10
    batch_size = 32
    
    # ==================== TEST 1: SINGLE COUPLING LAYER ====================
    print("\n[TEST 1] Single Coupling Layer")
    print("-" * 70)
    
    layer = CouplingLayer(dim, hidden_dim=64)
    
    # Forward pass
    z = torch.randn(batch_size, dim)
    x, log_det_forward = layer.forward(z)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {z.shape}")
    print(f"  Output shape: {x.shape}")
    print(f"  Log det shape: {log_det_forward.shape}")
    
    # Inverse pass
    z_reconstructed, log_det_inverse = layer.inverse(x)
    
    print(f"✓ Inverse pass successful")
    
    # Check reconstruction
    reconstruction_error = (z - z_reconstructed).abs().max().item()
    print(f"✓ Reconstruction error: {reconstruction_error:.6f}")
    assert reconstruction_error < 1e-5, "Reconstruction failed!"
    
    # Check log determinants sum to zero
    log_det_sum = (log_det_forward + log_det_inverse).abs().max().item()
    print(f"✓ Log det sum: {log_det_sum:.6f}")
    assert log_det_sum < 1e-5, "Log determinant check failed!"
    
    # ==================== TEST 2: COMPLETE FLOW MODEL ====================
    print("\n[TEST 2] Complete RealNVP Model")
    print("-" * 70)
    
    model = build_realnvp_model(dim, n_layers=4, hidden_dim=64)
    
    print(f"✓ Model built with {len(model.flows)} layers")
    
    # Forward through model
    x, log_det = model.forward(z)
    print(f"✓ Model forward pass successful")
    
    # Inverse through model
    z_rec, log_det_inv = model.inverse(x)
    
    # Check reconstruction
    model_error = (z - z_rec).abs().max().item()
    print(f"✓ Model reconstruction error: {model_error:.6f}")
    assert model_error < 1e-4, "Model reconstruction failed!"
    
    # ==================== TEST 3: SAMPLING ====================
    print("\n[TEST 3] Sampling from Model")
    print("-" * 70)
    
    samples = model.sample(16)
    print(f"✓ Generated {samples.shape[0]} samples")
    print(f"  Sample shape: {samples.shape}")
    
    # ==================== TEST 4: LOG PROBABILITY ====================
    print("\n[TEST 4] Log Probability Computation")
    print("-" * 70)
    
    log_prob = model.log_prob(samples)
    print(f"✓ Log probability computed")
    print(f"  Shape: {log_prob.shape}")
    print(f"  Mean log prob: {log_prob.mean().item():.4f}")
    print(f"  Std log prob: {log_prob.std().item():.4f}")
    
    # ==================== TEST 5: BATCH NORM FLOW ====================
    print("\n[TEST 5] Batch Normalization Flow")
    print("-" * 70)
    
    bn = BatchNorm(dim)
    bn.train()  # Set to training mode
    
    x_bn, log_det_bn = bn.forward(z)
    z_bn, log_det_bn_inv = bn.inverse(x_bn)
    
    bn_error = (z - z_bn).abs().max().item()
    print(f"✓ BatchNorm reconstruction error: {bn_error:.6f}")
    
    # ==================== ALL TESTS PASSED ====================
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    print("\nYour coupling flows implementation is working correctly!")
    print("You can now use these components to build and train flow models.")
    print("\nNext steps:")
    print("  1. Try example_2d_flows.py for 2D visualization")
    print("  2. Experiment with different architectures")
    print("  3. Apply to your own datasets")
