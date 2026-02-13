"""
Normalizing Flows: Fundamental Utilities and Base Classes

================================================================================
EDUCATIONAL PURPOSE
================================================================================
This module provides the foundational building blocks for normalizing flows.
It's designed for undergraduate students learning about generative models
and normalizing flows for the first time.

KEY CONCEPTS COVERED:
1. Base distributions (where we start sampling from)
2. Flow transformations (how we change the distribution)
3. Jacobian determinants (tracking volume changes)
4. Composing multiple transformations
5. Maximum likelihood training

PREREQUISITES:
- Understanding of probability distributions
- Basic neural networks (PyTorch)
- Multivariable calculus (Jacobians)
- Linear algebra fundamentals

LEARNING PATH:
Start with BaseDistribution → Flow → FlowSequence → Simple examples
Then move to more complex transformations in coupling_flows.py

================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class BaseDistribution:
    """
    Base Distribution: The Starting Point for Normalizing Flows
    
    ============================================================================
    CONCEPTUAL OVERVIEW
    ============================================================================
    In normalizing flows, we need a simple distribution that we can:
    1. Easily sample from (generate random numbers)
    2. Easily compute probabilities for
    
    We typically use a standard Gaussian (normal distribution) because:
    - It has a known probability density function
    - We can sample from it efficiently
    - It's centered at zero with unit variance
    
    Think of this as our "raw material" that we'll transform into
    more complex distributions through learned transformations.
    
    ============================================================================
    MATHEMATICAL FOUNDATION
    ============================================================================
    For a standard Gaussian in d dimensions:
    
    p(z) = (1/√(2π))^d × exp(-||z||²/2)
    
    Taking the log:
    log p(z) = -d/2 × log(2π) - ||z||²/2
             = -1/2 × (z² + log(2π)) summed over all dimensions
    
    This simple form makes it easy to compute probabilities!
    
    ============================================================================
    """
    
    def __init__(self, dim: int):
        """
        Initialize the base distribution.
        
        Args:
            dim (int): Dimensionality of the distribution
                      For images: might be height × width × channels
                      For tabular data: number of features
                      For 2D toy problems: typically 2
        
        Example:
            >>> base = BaseDistribution(dim=2)  # For 2D visualization
            >>> base = BaseDistribution(dim=784)  # For MNIST (28×28)
        """
        self.dim = dim
    
    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Sample from the standard Gaussian distribution.
        
        This is like rolling dice, but instead of 1-6, we get numbers
        from a bell curve centered at 0.
        
        Args:
            n_samples (int): How many samples to generate
            device (str): 'cpu' or 'cuda' - where to create the tensor
        
        Returns:
            torch.Tensor: Random samples, shape (n_samples, dim)
                         Each row is one sample from the Gaussian
        
        Mathematical Detail:
            We use torch.randn which implements the Box-Muller transform
            to generate normally distributed random numbers.
        
        Example:
            >>> base = BaseDistribution(dim=2)
            >>> samples = base.sample(100)  # Get 100 2D points
            >>> samples.shape
            torch.Size([100, 2])
            >>> samples.mean()  # Should be close to 0
            tensor(0.0234)
            >>> samples.std()   # Should be close to 1
            tensor(0.9876)
        """
        return torch.randn(n_samples, self.dim, device=device)
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of samples under the standard Gaussian.
        
        WHY LOG PROBABILITY?
        - Probabilities can be very small (like 1e-100), causing numerical issues
        - Taking logs converts multiplication to addition (more stable)
        - Logs are monotonic: higher log prob = higher probability
        
        Args:
            z (torch.Tensor): Samples to evaluate, shape (batch_size, dim)
        
        Returns:
            torch.Tensor: Log probabilities, shape (batch_size,)
                         One log probability value per sample
        
        Mathematical Derivation:
            For standard Gaussian: p(z) = (2π)^(-d/2) × exp(-z²/2)
            
            Taking log of both sides:
            log p(z) = log[(2π)^(-d/2)] + log[exp(-z²/2)]
                     = -d/2 × log(2π) - z²/2
                     = -0.5 × (z² + log(2π))  [per dimension]
            
            For multidimensional: sum over all dimensions
        
        Example:
            >>> base = BaseDistribution(dim=2)
            >>> z = torch.tensor([[0.0, 0.0],    # At the mean
            ...                   [3.0, 3.0]])   # Far from mean
            >>> log_probs = base.log_prob(z)
            >>> log_probs
            tensor([-1.8379, -10.8379])  # Second point has lower probability
        """
        # For each element: -0.5 × (z² + log(2π))
        # The log(2π) constant is approximately 1.8379
        log_prob = -0.5 * (z ** 2 + np.log(2 * np.pi))
        
        # Sum across dimensions to get total log probability for each sample
        # Shape: (batch_size, dim) → (batch_size,)
        return log_prob.sum(dim=-1)


class Flow(nn.Module):
    """
    Flow: Abstract Base Class for Invertible Transformations
    
    ============================================================================
    CONCEPTUAL OVERVIEW
    ============================================================================
    A "flow" is a special type of transformation that:
    
    1. Is INVERTIBLE (one-to-one mapping)
       - We can go from z → x (forward)
       - We can go from x → z (inverse)
       - No information is lost
    
    2. Has a TRACTABLE JACOBIAN
       - We can efficiently compute how volumes change
       - This is crucial for computing probabilities
    
    Think of flows like this:
    - Forward direction: Takes simple distribution → complex distribution
    - Inverse direction: Takes complex distribution → simple distribution
    
    ============================================================================
    WHY INVERTIBILITY MATTERS
    ============================================================================
    For training and sampling:
    
    TRAINING (Inverse):
        Real data x → Flow⁻¹ → Latent z → Compute log p(z)
        We need inverse to evaluate probabilities of real data
    
    SAMPLING (Forward):
        Sample z ~ N(0,I) → Flow → Generated data x
        We need forward to generate new samples
    
    ============================================================================
    THE CHANGE OF VARIABLES FORMULA
    ============================================================================
    This is the mathematical heart of normalizing flows!
    
    If z ~ p(z) and x = f(z), then:
    
        p(x) = p(z) |det(∂z/∂x)|
    
    In log space (more stable):
    
        log p(x) = log p(z) + log |det(∂z/∂x)|
    
    The Jacobian determinant log |det(∂z/∂x)| tells us:
    - How much the transformation stretches/shrinks volumes
    - Positive det = volume expansion
    - Negative det = volume contraction
    
    ============================================================================
    """
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation: Latent space → Data space (z → x)
        
        This is the GENERATIVE direction - used for sampling.
        
        Args:
            z (torch.Tensor): Samples from latent space, shape (batch_size, dim)
                             These come from our simple base distribution
        
        Returns:
            Tuple containing:
                x (torch.Tensor): Transformed samples in data space, shape (batch_size, dim)
                                 These should look like real data
                
                log_det (torch.Tensor): Log absolute determinant of Jacobian, shape (batch_size,)
                                       Tracks how volumes change
        
        Conceptual Flow:
            Simple Gaussian z → [Neural Network Transformation] → Complex data x
            
        Example Usage:
            >>> flow = SomeFlow(dim=2)
            >>> z = torch.randn(10, 2)  # 10 samples from Gaussian
            >>> x, log_det = flow.forward(z)  # Transform to data space
            >>> x.shape
            torch.Size([10, 2])
            >>> log_det.shape
            torch.Size([10])
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation: Data space → Latent space (x → z)
        
        This is the INFERENCE direction - used for computing probabilities.
        
        Args:
            x (torch.Tensor): Samples from data space, shape (batch_size, dim)
                             These are typically our training data
        
        Returns:
            Tuple containing:
                z (torch.Tensor): Transformed samples in latent space, shape (batch_size, dim)
                                 Should follow our base distribution
                
                log_det (torch.Tensor): Log absolute determinant of Jacobian, shape (batch_size,)
                                       For inverse, this has opposite sign of forward
        
        Conceptual Flow:
            Complex data x → [Inverse Transformation] → Simple Gaussian z
            
        Mathematical Note:
            If forward has Jacobian J_f, inverse has Jacobian J_f⁻¹
            det(J_f⁻¹) = 1/det(J_f)
            log|det(J_f⁻¹)| = -log|det(J_f)|
            
        Example Usage:
            >>> flow = SomeFlow(dim=2)
            >>> x = torch.tensor([[1.0, 2.0]])  # Real data point
            >>> z, log_det = flow.inverse(x)  # Map to latent space
            >>> # Now we can compute log p(x) = log p(z) + log_det
        """
        raise NotImplementedError("Subclasses must implement inverse()")


class FlowSequence(nn.Module):
    """
    Flow Sequence: Composing Multiple Transformations
    
    ============================================================================
    CONCEPTUAL OVERVIEW
    ============================================================================
    A single flow transformation is often not expressive enough to model
    complex distributions. Instead, we COMPOSE multiple flows:
    
        z → Flow₁ → Flow₂ → Flow₃ → ... → x
    
    Each flow adds a bit more complexity, gradually transforming the
    simple Gaussian into a rich, complex distribution.
    
    Think of it like:
    - Flow 1: Stretches the Gaussian
    - Flow 2: Rotates it
    - Flow 3: Adds some curvature
    - Flow 4: Creates more complex patterns
    - ... and so on
    
    ============================================================================
    MATHEMATICAL COMPOSITION
    ============================================================================
    For composed transformations f₁, f₂, ..., fₙ:
    
    Forward: x = fₙ(...f₂(f₁(z)))
    
    The log determinant rule:
        log|det(∂x/∂z)| = Σᵢ log|det(∂fᵢ/∂fᵢ₋₁)|
    
    We simply SUM the log determinants! This is why we work in log space.
    
    Inverse: z = f₁⁻¹(f₂⁻¹(...fₙ⁻¹(x)))
    
    We apply inverses in REVERSE order (like taking off nested clothes).
    
    ============================================================================
    DESIGN PRINCIPLES
    ============================================================================
    1. Alternating Patterns: Often alternate different types of flows
       Example: [Coupling, BatchNorm, Coupling, BatchNorm, ...]
    
    2. Increasing Complexity: Earlier flows can be simpler, later more complex
    
    3. Number of Flows: Typical ranges:
       - 2D toy problems: 4-8 flows
       - Image generation: 20-40 flows
       - More flows = more expressive but slower
    
    ============================================================================
    """
    
    def __init__(self, flows: list, base_dist: BaseDistribution):
        """
        Initialize a sequence of flow transformations.
        
        Args:
            flows (list): List of Flow objects to compose
                         These will be applied in order during forward pass
                         Example: [CouplingLayer(), BatchNorm(), CouplingLayer()]
            
            base_dist (BaseDistribution): The starting distribution (typically Gaussian)
        
        Design Pattern:
            flows = [
                CouplingLayer(dim, hidden_dim=64),
                BatchNorm(dim),
                CouplingLayer(dim, hidden_dim=64),
                BatchNorm(dim),
            ]
            model = FlowSequence(flows, BaseDistribution(dim))
        """
        super().__init__()
        
        # Use ModuleList so PyTorch knows these are trainable parameters
        self.flows = nn.ModuleList(flows)
        
        # Store the base distribution
        self.base_dist = base_dist
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform samples through all flows: z → x
        
        This is the GENERATIVE direction used for sampling.
        
        Process:
            1. Start with z from base distribution
            2. Apply Flow₁: z → h₁, accumulate log_det₁
            3. Apply Flow₂: h₁ → h₂, accumulate log_det₂
            4. Continue through all flows
            5. Final output: x, total_log_det
        
        Args:
            z (torch.Tensor): Latent samples from base distribution, shape (batch_size, dim)
        
        Returns:
            Tuple containing:
                x (torch.Tensor): Final transformed samples, shape (batch_size, dim)
                log_det_sum (torch.Tensor): Sum of all log determinants, shape (batch_size,)
        
        Mathematical Detail:
            log|det(∂x/∂z)| = Σᵢ log|det(∂fᵢ/∂fᵢ₋₁)|
            
        Example:
            >>> flows = [Flow1(), Flow2(), Flow3()]
            >>> model = FlowSequence(flows, BaseDistribution(2))
            >>> z = model.base_dist.sample(100)
            >>> x, log_det = model.forward(z)
            >>> # Now x contains 100 generated samples
        """
        # Initialize log determinant accumulator
        # Start with zeros for each sample in the batch
        log_det_sum = torch.zeros(z.shape[0], device=z.device)
        
        # Start with z, progressively transform it
        x = z
        
        # Apply each flow in sequence
        for flow in self.flows:
            # Transform: current → next
            x, log_det = flow.forward(x)
            
            # Accumulate log determinants
            # This is the key insight: log determinants ADD in log space
            log_det_sum += log_det
        
        return x, log_det_sum
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform samples back through all flows: x → z
        
        This is the INFERENCE direction used for computing probabilities.
        
        Process:
            1. Start with x (real data)
            2. Apply Flowₙ⁻¹: x → hₙ₋₁, get log_detₙ
            3. Apply Flowₙ₋₁⁻¹: hₙ₋₁ → hₙ₋₂, get log_detₙ₋₁
            4. Continue in REVERSE order
            5. Final output: z, total_log_det
        
        Args:
            x (torch.Tensor): Data samples to transform, shape (batch_size, dim)
        
        Returns:
            Tuple containing:
                z (torch.Tensor): Latent samples, shape (batch_size, dim)
                log_det_sum (torch.Tensor): Sum of all log determinants, shape (batch_size,)
        
        Critical Note:
            We apply inverses in REVERSE order!
            If forward is: z → f₁ → f₂ → f₃ → x
            Then inverse is: x → f₃⁻¹ → f₂⁻¹ → f₁⁻¹ → z
            
        Example:
            >>> model = FlowSequence([Flow1(), Flow2()], BaseDistribution(2))
            >>> x = torch.randn(100, 2)  # Some data
            >>> z, log_det = model.inverse(x)
            >>> # Now z should look like Gaussian samples
        """
        # Initialize log determinant accumulator
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        
        # Start with x, progressively transform backwards to z
        z = x
        
        # Apply each flow's inverse in REVERSE order
        # reversed() is crucial here!
        for flow in reversed(self.flows):
            # Transform: current → previous
            z, log_det = flow.inverse(z)
            
            # Accumulate log determinants
            log_det_sum += log_det
        
        return z, log_det_sum
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of data samples.
        
        This is THE KEY EQUATION for normalizing flows!
        
        Mathematical Formula:
            log p(x) = log p(z) + log|det(∂z/∂x)|
            
        Where:
            - z = f⁻¹(x) is the latent representation
            - log p(z) is the base distribution log probability
            - log|det(∂z/∂x)| accounts for volume changes
        
        Intuition:
            1. Map data x back to latent space: x → z
            2. Evaluate how likely z is under base distribution
            3. Adjust for volume changes during transformation
            4. Result: likelihood of x under learned distribution
        
        Args:
            x (torch.Tensor): Data samples to evaluate, shape (batch_size, dim)
        
        Returns:
            torch.Tensor: Log probabilities, shape (batch_size,)
                         Higher values = more likely under learned distribution
        
        Training Usage:
            >>> model = FlowSequence([...], BaseDistribution(2))
            >>> real_data = load_data()
            >>> log_probs = model.log_prob(real_data)
            >>> loss = -log_probs.mean()  # Negative log-likelihood
            >>> loss.backward()  # Optimize to maximize likelihood
        
        Generation Usage:
            >>> # Check quality of generated samples
            >>> generated = model.sample(100)
            >>> log_probs = model.log_prob(generated)
            >>> # Higher log_probs indicate better quality
        """
        # Step 1: Transform data to latent space
        # This gives us z and the change-of-variables correction
        z, log_det = self.inverse(x)
        
        # Step 2: Evaluate probability under base distribution
        log_pz = self.base_dist.log_prob(z)
        
        # Step 3: Apply change of variables formula
        # log p(x) = log p(z) + log|det(∂z/∂x)|
        return log_pz + log_det
    
    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Generate new samples from the learned distribution.
        
        This is how we CREATE NEW DATA after training!
        
        Process:
            1. Sample z from simple base distribution (Gaussian)
            2. Transform z through all flows
            3. Output x should resemble training data
        
        Args:
            n_samples (int): Number of samples to generate
            device (str): Device to generate on ('cpu' or 'cuda')
        
        Returns:
            torch.Tensor: Generated samples, shape (n_samples, dim)
        
        Example After Training:
            >>> # Train model on images
            >>> model = FlowSequence([...], BaseDistribution(784))
            >>> train(model, mnist_data)
            >>> 
            >>> # Generate new images
            >>> new_images = model.sample(16)  # Generate 16 new images
            >>> # new_images should look like realistic MNIST digits
        
        Quality Check:
            After training, samples should:
            - Resemble training data distribution
            - Be diverse (not all the same)
            - Have high log probability under model
        """
        # Step 1: Sample from simple base distribution
        z = self.base_dist.sample(n_samples, device=device)
        
        # Step 2: Transform through all flows
        # We discard log_det since we don't need it for sampling
        x, _ = self.forward(z)
        
        return x


class AffineTransform(Flow):
    """
    Affine Transformation: The Simplest Flow
    
    ============================================================================
    CONCEPTUAL OVERVIEW
    ============================================================================
    An affine transformation is the simplest possible flow:
    
        y = scale × x + shift
    
    This is like:
    - Stretching/shrinking each dimension (scale)
    - Moving the center (shift)
    
    Example in 1D:
        If scale=2 and shift=3:
        Input: [0, 1, 2] → Output: [3, 5, 7]
    
    ============================================================================
    MATHEMATICAL PROPERTIES
    ============================================================================
    Forward:
        y = exp(log_scale) × x + shift
        
    Why exp(log_scale)?
        - We learn log_scale instead of scale
        - This ensures scale is always positive
        - More numerically stable for optimization
    
    Jacobian:
        ∂y/∂x = diag(exp(log_scale))
        
    Log Determinant:
        log|det(∂y/∂x)| = sum(log_scale)
        
    This is SUPER EFFICIENT - just sum the parameters!
    
    ============================================================================
    USE CASES
    ============================================================================
    1. Data normalization: Center and scale input data
    2. Simple baseline: Compare more complex flows against this
    3. Initial layer: Sometimes used as first transformation
    4. Debugging: Easy to verify correctness
    
    ============================================================================
    """
    
    def __init__(self, dim: int):
        """
        Initialize an affine transformation.
        
        Args:
            dim (int): Dimensionality of the data
        
        Learned Parameters:
            - log_scale: Controls stretching/compression per dimension
            - shift: Controls translation per dimension
        
        Initialization:
            - log_scale starts at 0 (scale = 1, no change)
            - shift starts at 0 (no translation)
        """
        super().__init__()
        
        # Learnable parameters
        # We learn log(scale) for numerical stability
        # Initial scale = exp(0) = 1 (no change)
        self.log_scale = nn.Parameter(torch.zeros(dim))
        
        # Initial shift = 0 (no translation)
        self.shift = nn.Parameter(torch.zeros(dim))
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation: y = scale * z + shift
        
        Args:
            z (torch.Tensor): Input samples, shape (batch_size, dim)
        
        Returns:
            Tuple containing:
                x (torch.Tensor): Transformed samples
                log_det (torch.Tensor): Log determinant
        
        Example:
            >>> transform = AffineTransform(dim=2)
            >>> z = torch.tensor([[0., 0.], [1., 1.]])
            >>> x, log_det = transform.forward(z)
        """
        # Convert log_scale to actual scale (always positive)
        scale = torch.exp(self.log_scale)
        
        # Apply affine transformation
        # Broadcasting: scale and shift are applied element-wise
        x = scale * z + self.shift
        
        # Compute log determinant
        # For diagonal Jacobian: det = product of diagonal
        # log(det) = sum of log(diagonal) = sum(log_scale)
        log_det = self.log_scale.sum().expand(z.shape[0])
        
        return x, log_det
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation: z = (x - shift) / scale
        
        Args:
            x (torch.Tensor): Transformed samples, shape (batch_size, dim)
        
        Returns:
            Tuple containing:
                z (torch.Tensor): Original samples
                log_det (torch.Tensor): Log determinant (negative of forward)
        """
        # Convert log_scale to actual scale
        scale = torch.exp(self.log_scale)
        
        # Apply inverse affine transformation
        z = (x - self.shift) / scale
        
        # Log determinant of inverse
        # det(J_inv) = 1/det(J_forward)
        # log(det(J_inv)) = -log(det(J_forward))
        log_det = -self.log_scale.sum().expand(x.shape[0])
        
        return z, log_det


class PlanarFlow(Flow):
    """
    Planar Flow: Adding Curvature to the Transformation
    
    ============================================================================
    CONCEPTUAL OVERVIEW
    ============================================================================
    Planar flows add non-linearity through a "planar" transformation:
    
        f(z) = z + u × tanh(w^T z + b)
    
    Think of it as:
    - w^T z + b: A linear projection (like a decision boundary)
    - tanh: Squashes values between -1 and 1
    - u: Direction of transformation
    - Result: Bends the space along a "plane"
    
    ============================================================================
    GEOMETRIC INTUITION
    ============================================================================
    Imagine a flat 2D plane:
    1. The term w^T z + b defines a line in the plane
    2. tanh creates smooth transitions around this line
    3. u controls how much we "push" perpendicular to the line
    
    This creates a smooth, curved transformation that can:
    - Create ridges or valleys in the distribution
    - Bend Gaussian distributions into curved shapes
    
    ============================================================================
    MATHEMATICAL DETAILS
    ============================================================================
    Forward:
        f(z) = z + u × tanh(w^T z + b)
    
    Jacobian:
        ∂f/∂z = I + u × ψ^T
        where ψ = (1 - tanh²(w^T z + b)) × w
    
    Log Determinant:
        log|det(∂f/∂z)| = log|1 + u^T ψ|
        
    This uses the matrix determinant lemma for efficiency!
    
    ============================================================================
    LIMITATIONS
    ============================================================================
    1. No analytical inverse: Need iterative solver
       (This is why coupling flows are often preferred)
    
    2. Limited expressiveness: Single planar flow is quite simple
       (Need many flows to model complex distributions)
    
    3. Mainly useful for: Low-dimensional problems (2D, 3D)
    
    ============================================================================
    """
    
    def __init__(self, dim: int):
        """
        Initialize a planar flow transformation.
        
        Args:
            dim (int): Dimensionality of the space
        
        Learned Parameters:
            - weight (w): Defines the "plane" direction, shape (dim,)
            - bias (b): Shifts the plane, shape (1,)
            - u: Transformation direction, shape (dim,)
        
        Initialization:
            - weight: Random normal (creates varied initial planes)
            - bias: Zero (plane passes through origin)
            - u: Random normal (random transformation direction)
        """
        super().__init__()
        
        # Parameters for the planar transformation
        # w: defines the hyperplane in input space
        self.weight = nn.Parameter(torch.randn(dim))
        
        # b: offset for the hyperplane
        self.bias = nn.Parameter(torch.zeros(1))
        
        # u: direction of transformation
        self.u = nn.Parameter(torch.randn(dim))
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation through planar flow.
        
        Process:
            1. Compute linear projection: w^T z + b
            2. Apply tanh non-linearity
            3. Scale by u and add to z
            4. Compute Jacobian determinant
        
        Args:
            z (torch.Tensor): Input samples, shape (batch_size, dim)
        
        Returns:
            Tuple containing:
                x (torch.Tensor): Transformed samples
                log_det (torch.Tensor): Log determinant
        
        Numerical Stability:
            - tanh keeps values bounded
            - We add small epsilon (1e-8) to avoid log(0)
        """
        # Step 1: Compute linear projection w^T z + b
        # For each sample: dot product with weight vector plus bias
        # Shape: (batch_size, dim) × (dim,) → (batch_size, 1)
        linear = torch.sum(self.weight * z, dim=-1, keepdim=True) + self.bias
        
        # Step 2: Apply transformation
        # f(z) = z + u × tanh(w^T z + b)
        # The keepdim=True ensures broadcasting works correctly
        x = z + self.u * torch.tanh(linear)
        
        # Step 3: Compute Jacobian determinant
        # This is the tricky part!
        
        # Derivative of tanh: d/dx tanh(x) = 1 - tanh²(x)
        # This is called the "tanh derivative" or "sech²"
        psi = (1 - torch.tanh(linear) ** 2) * self.weight
        
        # Using matrix determinant lemma:
        # det(I + u × ψ^T) = 1 + u^T ψ
        det = 1 + torch.sum(psi * self.u, dim=-1)
        
        # Take log and absolute value for numerical stability
        # Add epsilon to avoid log(0)
        log_det = torch.log(torch.abs(det) + 1e-8)
        
        return x, log_det
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation: x → z
        
        IMPORTANT LIMITATION:
        Planar flows do NOT have a closed-form inverse!
        
        To invert, you would need to solve:
            z = x - u × tanh(w^T z + b)
            
        This requires iterative methods (like Newton's method or fixed-point iteration).
        
        For most applications, we avoid planar flows in favor of coupling flows,
        which DO have efficient analytical inverses.
        
        Args:
            x: Transformed samples
        
        Raises:
            NotImplementedError: This method requires an iterative solver
        
        If You Need This:
            Implement fixed-point iteration or Newton-Raphson:
            
            def inverse_iterative(x, n_iters=10):
                z = x  # Initial guess
                for _ in range(n_iters):
                    linear = torch.sum(weight * z, dim=-1, keepdim=True) + bias
                    z = x - u * torch.tanh(linear)
                return z
        """
        raise NotImplementedError(
            "Planar flow inverse requires iterative solver. "
            "Consider using coupling flows instead, which have analytical inverses."
        )


def visualize_2d_transformation(flow_model: nn.Module, n_points: int = 1000,
                               xlim: tuple = (-3, 3), ylim: tuple = (-3, 3),
                               filename: str = 'transformation.png'):
    """
    Visualize how a flow transforms a 2D distribution.
    
    ============================================================================
    PURPOSE
    ============================================================================
    This function helps you SEE what normalizing flows do!
    
    It creates a side-by-side comparison:
    - Left: Samples from base Gaussian (latent space z)
    - Right: After transformation through flow (data space x)
    
    This visualization is crucial for understanding:
    - How flows warp space
    - Whether training is working
    - What patterns the flow has learned
    
    ============================================================================
    INTERPRETATION GUIDE
    ============================================================================
    BEFORE TRAINING:
        - Left: Nice Gaussian blob
        - Right: Slightly warped Gaussian (random initialization)
    
    DURING TRAINING:
        - Right side gradually morphs to match data distribution
        - Watch for increasing complexity and structure
    
    AFTER TRAINING:
        - Left: Still Gaussian (never changes)
        - Right: Should match target distribution (e.g., moons, circles)
    
    TROUBLESHOOTING:
        - Right side unchanged? → Model not training
        - Right side chaotic? → Learning rate too high
        - Right side clumped? → Not enough layers/capacity
    
    ============================================================================
    
    Args:
        flow_model (nn.Module): Trained flow model to visualize
        n_points (int): Number of points to sample and plot
        xlim (tuple): X-axis limits for both plots
        ylim (tuple): Y-axis limits for both plots
        filename (str): Where to save the visualization
    
    Example Usage:
        >>> model = build_realnvp_model(dim=2, n_layers=6)
        >>> train(model, data)  # Train on 2D data
        >>> visualize_2d_transformation(model, n_points=2000)
        >>> # Open transformation.png to see the result!
    """
    # Set model to evaluation mode (disables dropout, batch norm updates)
    flow_model.eval()
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Step 1: Sample from base Gaussian distribution
        z = flow_model.base_dist.sample(n_points)
        
        # Step 2: Transform through the flow
        x, _ = flow_model.forward(z)
        
        # Step 3: Move to CPU for plotting (if on GPU)
        z = z.cpu().numpy()
        x = x.cpu().numpy()
    
    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # LEFT PLOT: Latent space (always Gaussian)
    axes[0].scatter(z[:, 0], z[:, 1], alpha=0.5, s=10)
    axes[0].set_title('Latent Space (z) - Base Gaussian')
    axes[0].set_xlabel('z₁')
    axes[0].set_ylabel('z₂')
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[0].set_aspect('equal')  # Equal aspect ratio
    axes[0].grid(True, alpha=0.3)
    
    # RIGHT PLOT: Data space (transformed distribution)
    axes[1].scatter(x[:, 0], x[:, 1], alpha=0.5, s=10, color='red')
    axes[1].set_title('Data Space (x) - Transformed Distribution')
    axes[1].set_xlabel('x₁')
    axes[1].set_ylabel('x₂')
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved visualization to {filename}")
    print(f"  Compare left (Gaussian) vs right (learned distribution)")


def visualize_2d_density(flow_model: nn.Module, xlim: tuple = (-3, 3),
                        ylim: tuple = (-3, 3), n_grid: int = 200,
                        filename: str = 'density.png'):
    """
    Visualize the learned probability density of a 2D flow.
    
    ============================================================================
    PURPOSE
    ============================================================================
    While visualize_2d_transformation shows samples, this function shows the
    PROBABILITY DENSITY - how likely different regions are under the model.
    
    This creates a heatmap where:
    - Bright regions = High probability (model thinks data should be here)
    - Dark regions = Low probability (model thinks data is unlikely here)
    
    ============================================================================
    INTERPRETATION GUIDE
    ============================================================================
    WELL-TRAINED MODEL:
        - Peaks align with data clusters
        - Smooth, continuous probability landscape
        - Clear separation between modes
    
    POORLY-TRAINED MODEL:
        - Flat everywhere (hasn't learned structure)
        - Peaks in wrong places
        - Extremely sharp peaks (overfitting)
    
    COMPARISON WITH DATA:
        Overlay this with scatter plot of real data to verify:
        - High probability where data is dense
        - Low probability where data is sparse
    
    ============================================================================
    COMPUTATIONAL NOTE
    ============================================================================
    This computes probability at EVERY point on a 200×200 grid = 40,000 points!
    For high-resolution or high-dimensional problems, this can be slow.
    
    Time complexity: O(n_grid² × model_complexity)
    
    ============================================================================
    
    Args:
        flow_model (nn.Module): Trained flow model
        xlim (tuple): X-axis limits for density grid
        ylim (tuple): Y-axis limits for density grid
        n_grid (int): Resolution of density grid (n_grid × n_grid points)
        filename (str): Where to save the visualization
    
    Example Usage:
        >>> model = build_realnvp_model(dim=2, n_layers=6)
        >>> train(model, moons_data)
        >>> visualize_2d_density(model)
        >>> # The plot should show two crescent-shaped high-probability regions!
    """
    # Set to evaluation mode
    flow_model.eval()
    
    # Step 1: Create a grid of points covering the entire space
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(x, y)  # Create 2D grid
    
    # Step 2: Flatten grid into list of (x, y) points
    # Shape: (n_grid*n_grid, 2)
    points = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1), 
        dtype=torch.float32
    )
    
    # Step 3: Compute log probability at each grid point
    with torch.no_grad():
        log_prob = flow_model.log_prob(points)
        
        # Convert log probability to actual probability
        # prob = exp(log_prob)
        prob = torch.exp(log_prob).cpu().numpy()
    
    # Step 4: Reshape back to grid for plotting
    prob = prob.reshape(n_grid, n_grid)
    
    # Step 5: Create filled contour plot
    plt.figure(figsize=(8, 7))
    
    # contourf creates smooth colored regions
    # levels=50 gives fine gradations
    plt.contourf(X, Y, prob, levels=50, cmap='viridis')
    
    # Add colorbar to show probability scale
    plt.colorbar(label='Probability Density')
    
    plt.title('Learned Probability Density p(x)')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved density visualization to {filename}")
    print(f"  Bright = high probability, Dark = low probability")


def train_flow(flow_model: nn.Module, dataloader, optimizer, 
              n_epochs: int = 100, device: str = 'cpu') -> list:
    """
    Train a normalizing flow model using maximum likelihood estimation.
    
    ============================================================================
    CONCEPTUAL OVERVIEW
    ============================================================================
    Training Objective: Maximize likelihood of training data
    
    In other words: Adjust model parameters so that real data gets
    HIGH probability under the learned distribution.
    
    Loss Function:
        Loss = -mean(log p(x))  for x in training data
        
    We MINIMIZE negative log-likelihood (= MAXIMIZE likelihood)
    
    ============================================================================
    TRAINING ALGORITHM
    ============================================================================
    For each epoch:
        For each batch of data:
            1. Compute log p(x) for batch
               - Map x → z through inverse
               - Evaluate log p(z) + log|det|
            
            2. Compute loss = -mean(log p(x))
               - Negative because we minimize
               - Mean over batch for stable gradients
            
            3. Backpropagate and update parameters
               - Compute gradients: ∂Loss/∂θ
               - Update: θ ← θ - lr × ∇Loss
    
    ============================================================================
    MONITORING TRAINING
    ============================================================================
    HEALTHY TRAINING:
        - Loss steadily decreases
        - Eventually plateaus
        - No wild oscillations
    
    PROBLEMS TO WATCH FOR:
        - Loss increases → Learning rate too high
        - Loss oscillates → Reduce learning rate or batch size
        - Loss stuck → Model too simple or data too complex
        - Loss → NaN → Numerical instability (gradient explosion)
    
    ============================================================================
    HYPERPARAMETER TIPS
    ============================================================================
    Learning Rate:
        - Start: 1e-3 or 1e-4
        - Lower if training unstable
        - Can use learning rate scheduling
    
    Batch Size:
        - Larger = more stable but slower
        - Smaller = faster but noisier
        - Typical: 64-256 for small datasets
    
    Epochs:
        - 2D toy data: 100-500 epochs
        - Images: 50-100 epochs
        - Monitor loss, stop when converged
    
    ============================================================================
    
    Args:
        flow_model (nn.Module): The flow model to train
        dataloader: PyTorch DataLoader with training data
        optimizer: PyTorch optimizer (e.g., Adam)
        n_epochs (int): Number of training epochs
        device (str): 'cpu' or 'cuda'
    
    Returns:
        list: Training losses for each epoch
    
    Example Usage:
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> 
        >>> # Prepare data
        >>> data = generate_toy_data('moons', n_samples=2000)
        >>> dataset = TensorDataset(data)
        >>> dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        >>> 
        >>> # Build model
        >>> model = build_realnvp_model(dim=2, n_layers=6)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> 
        >>> # Train!
        >>> losses = train_flow(model, dataloader, optimizer, n_epochs=500)
        >>> 
        >>> # Plot training curve
        >>> plot_training_loss(losses)
    """
    # Set model to training mode
    # This enables things like dropout and batch norm updates
    flow_model.train()
    
    # List to store loss values for plotting
    losses = []
    
    # Training loop
    for epoch in range(n_epochs):
        epoch_loss = 0.0  # Accumulator for this epoch
        n_batches = 0     # Count batches
        
        # Iterate over batches
        for batch in dataloader:
            # Handle different dataloader formats
            if isinstance(batch, (tuple, list)):
                batch = batch[0]  # Extract data (ignore labels if present)
            
            # Move batch to device (CPU or GPU)
            batch = batch.to(device)
            
            # Flatten if data has more than 2 dimensions
            # E.g., (batch_size, 28, 28) → (batch_size, 784) for MNIST
            if batch.dim() > 2:
                batch = batch.view(batch.shape[0], -1)
            
            # ==================== FORWARD PASS ====================
            # Compute log probability of batch under current model
            # This involves:
            #   1. Inverse transform: x → z
            #   2. Evaluate base log prob: log p(z)
            #   3. Add Jacobian correction: log|det|
            log_prob = flow_model.log_prob(batch)
            
            # ==================== COMPUTE LOSS ====================
            # Negative log-likelihood loss
            # We want HIGH log probability, so we minimize NEGATIVE log prob
            loss = -log_prob.mean()
            
            # ==================== BACKWARD PASS ====================
            # Clear previous gradients
            optimizer.zero_grad()
            
            # Compute gradients via backpropagation
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # ==================== BOOKKEEPING ====================
            epoch_loss += loss.item()
            n_batches += 1
        
        # Compute average loss for this epoch
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        # Print progress periodically
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    return losses


def plot_training_loss(losses: list, filename: str = 'training_loss.png'):
    """
    Plot the training loss curve.
    
    ============================================================================
    PURPOSE
    ============================================================================
    Visualizing the loss curve helps you understand:
    - Is training working? (Loss should decrease)
    - Has training converged? (Loss plateaus)
    - Are there problems? (Oscillations, divergence)
    
    ============================================================================
    INTERPRETATION GUIDE
    ============================================================================
    GOOD TRAINING CURVE:
        - Smooth decrease
        - Eventually flattens
        - No sudden spikes
        Example: \___  (starts high, drops, levels off)
    
    PROBLEMS:
        - Flat line → Model not learning (check learning rate, model capacity)
        - Increasing → Diverging (reduce learning rate)
        - Oscillating → Unstable (reduce learning rate or batch size)
        - Sharp drops then flat → Needs more epochs or better initialization
    
    ============================================================================
    
    Args:
        losses (list): List of loss values from training
        filename (str): Where to save the plot
    
    Example Usage:
        >>> losses = train_flow(model, dataloader, optimizer, n_epochs=500)
        >>> plot_training_loss(losses)
        >>> # Check training_loss.png to verify training worked
    """
    plt.figure(figsize=(10, 5))
    
    # Plot loss vs epoch
    plt.plot(losses, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Negative Log-Likelihood', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    
    # Add grid for easier reading
    plt.grid(True, alpha=0.3)
    
    # Annotate final loss
    final_loss = losses[-1]
    plt.annotate(f'Final: {final_loss:.4f}',
                xy=(len(losses)-1, final_loss),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, bbox=dict(boxstyle='round,pad=0.5', 
                                      facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved loss plot to {filename}")
    print(f"  Final loss: {final_loss:.4f}")


# ============================================================================
# MODULE SUMMARY AND NEXT STEPS
# ============================================================================
"""
CONGRATULATIONS! You've completed the fundamental building blocks!

WHAT YOU'VE LEARNED:
✓ BaseDistribution: Where we sample from (Gaussian)
✓ Flow: Abstract class for transformations
✓ FlowSequence: Composing multiple flows
✓ AffineTransform: Simplest possible flow
✓ PlanarFlow: Adding non-linearity
✓ Visualization tools for 2D flows
✓ Training pipeline for maximum likelihood

NEXT STEPS:
1. Study coupling_flows.py for more powerful transformations
   - Coupling layers (RealNVP)
   - Batch normalization flows
   - Checkerboard patterns for images

2. Run example_2d_flows.py to see everything in action
   - Train on toy 2D datasets
   - Visualize transformations
   - Understand training dynamics

3. Experiment with your own data!
   - Start with 2D toy problems
   - Try different architectures
   - Visualize what you learn

RECOMMENDED READING:
- "Normalizing Flows for Probabilistic Modeling" (Papamakarios et al., 2019)
- "Density Estimation Using Real NVP" (Dinh et al., 2017)
- "Variational Inference with Normalizing Flows" (Rezende & Mohamed, 2015)

TIPS FOR SUCCESS:
- Always visualize! Use the provided visualization tools
- Start simple: 2D data, few layers
- Monitor training loss carefully
- Experiment with hyperparameters
- Check invertibility: x → z → x should give back x

Happy learning! 🎓
"""
