# Neural Architecture Search

## Overview

Neural Architecture Search (NAS) automates the design of neural network architectures, replacing manual architecture engineering with algorithmic optimization. NAS discovers architectures that are often more efficient than human-designed networks, making it a powerful tool for model compression and deployment optimization.

## Motivation

Hand-designed architectures involve countless design decisions: number of layers, layer widths, kernel sizes, skip connections, activation functions, and normalization strategies. NAS systematically explores this space to find architectures optimized for specific objectives such as accuracy, latency, or model size.

## Search Space Design

### Common Search Spaces

```python
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import random

# Define operation candidates for each layer
OPERATIONS = {
    'conv_3x3': lambda C: nn.Conv2d(C, C, 3, padding=1),
    'conv_5x5': lambda C: nn.Conv2d(C, C, 5, padding=2),
    'sep_conv_3x3': lambda C: nn.Sequential(
        nn.Conv2d(C, C, 3, padding=1, groups=C),
        nn.Conv2d(C, C, 1)
    ),
    'dil_conv_3x3': lambda C: nn.Conv2d(C, C, 3, padding=2, dilation=2),
    'max_pool_3x3': lambda C: nn.MaxPool2d(3, stride=1, padding=1),
    'avg_pool_3x3': lambda C: nn.AvgPool2d(3, stride=1, padding=1),
    'skip_connect': lambda C: nn.Identity(),
    'none': lambda C: Zero(C),
}

class Zero(nn.Module):
    """Zero operation for NAS (output zeros)."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
    
    def forward(self, x):
        return torch.zeros_like(x)
```

### One-Shot NAS with Weight Sharing

```python
class SearchCell(nn.Module):
    """NAS cell with differentiable architecture parameters."""
    
    def __init__(self, channels: int, operations: Dict):
        super().__init__()
        self.ops = nn.ModuleDict({
            name: op_fn(channels) for name, op_fn in operations.items()
        })
        # Architecture parameters (learnable)
        self.alphas = nn.Parameter(
            torch.randn(len(operations)) * 0.01
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softmax over architecture parameters
        weights = torch.softmax(self.alphas, dim=0)
        
        # Weighted sum of all operations
        output = sum(
            w * op(x) for w, (name, op) in zip(weights, self.ops.items())
        )
        return output
    
    def get_best_op(self) -> str:
        """Return the operation with highest weight."""
        idx = self.alphas.argmax().item()
        return list(self.ops.keys())[idx]


class DARTSNetwork(nn.Module):
    """DARTS-style differentiable NAS."""
    
    def __init__(self, num_cells: int = 8, channels: int = 16,
                 num_classes: int = 10):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        self.cells = nn.ModuleList([
            SearchCell(channels, OPERATIONS) for _ in range(num_cells)
        ])
        
        self.classifier = nn.Linear(channels, num_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
    
    def architecture_parameters(self):
        """Return architecture parameters for separate optimization."""
        return [cell.alphas for cell in self.cells]
    
    def weight_parameters(self):
        """Return weight parameters (excluding architecture params)."""
        arch_params = set(id(p) for p in self.architecture_parameters())
        return [p for p in self.parameters() if id(p) not in arch_params]
    
    def derive_architecture(self) -> List[str]:
        """Extract the discovered architecture."""
        return [cell.get_best_op() for cell in self.cells]
```

## Hardware-Aware NAS

Incorporate hardware constraints (latency, memory) into the search objective:

```python
class LatencyPredictor:
    """Predict inference latency for architecture candidates."""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.latency_cache = {}
    
    def measure_latency(self, module: nn.Module, input_shape: Tuple,
                       num_runs: int = 100) -> float:
        """Measure actual inference latency."""
        import time
        x = torch.randn(1, *input_shape).to(self.device)
        module = module.to(self.device).eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                module(x)
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                module(x)
                times.append(time.perf_counter() - start)
        
        return sum(times) / len(times) * 1000  # ms


def hardware_aware_loss(logits, targets, architecture_params,
                       latency_predictor, target_latency_ms=10.0,
                       lambda_latency=0.1):
    """Combined loss: accuracy + latency penalty."""
    import torch.nn.functional as F
    
    # Task loss
    task_loss = F.cross_entropy(logits, targets)
    
    # Latency penalty (differentiable approximation)
    predicted_latency = sum(
        torch.softmax(alpha, dim=0).sum() for alpha in architecture_params
    )
    latency_penalty = torch.relu(predicted_latency - target_latency_ms)
    
    return task_loss + lambda_latency * latency_penalty
```

## Low-Rank Factorization as Architecture Optimization

Low-rank factorization can be viewed as an architecture transformation that replaces large layers with smaller equivalents. This connects NAS with matrix decomposition approaches:

# Low-Rank Factorization

## Overview

Low-rank factorization compresses neural networks by decomposing weight matrices into products of smaller matrices, exploiting the observation that trained weight matrices often have low effective rank. This reduces both storage requirements and computational cost while preserving most of the model's representational capacity.

## Mathematical Foundation

### Matrix Rank and Approximation

A matrix $\mathbf{W} \in \mathbb{R}^{m \times n}$ can be approximated by a low-rank factorization:

$$\mathbf{W} \approx \mathbf{U}\mathbf{V}^T$$

where:
- $\mathbf{U} \in \mathbb{R}^{m \times r}$
- $\mathbf{V} \in \mathbb{R}^{n \times r}$
- $r \ll \min(m, n)$ is the rank

**Parameter reduction:**
- Original: $m \times n$ parameters
- Factorized: $m \times r + n \times r = r(m + n)$ parameters
- Reduction factor: $\frac{mn}{r(m+n)}$

### Singular Value Decomposition (SVD)

For a weight matrix $\mathbf{W} \in \mathbb{R}^{m \times n}$ with rank $r$, the Singular Value Decomposition (SVD) gives:

$$\mathbf{W} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

where:
- $\mathbf{U} \in \mathbb{R}^{m \times m}$ contains left singular vectors (orthonormal)
- $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$ contains singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$
- $\mathbf{V} \in \mathbb{R}^{n \times n}$ contains right singular vectors (orthonormal)

### Truncated SVD (Low-Rank Approximation)

The best rank-$k$ approximation (in Frobenius norm) is obtained by keeping only the top $k$ singular values:

$$\mathbf{W}_k = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$

where $\mathbf{U}_k \in \mathbb{R}^{m \times k}$, $\mathbf{\Sigma}_k \in \mathbb{R}^{k \times k}$, $\mathbf{V}_k \in \mathbb{R}^{n \times k}$.

**Eckart-Young-Mirsky Theorem:**
$$\mathbf{W}_k = \arg\min_{\text{rank}(\mathbf{A}) \leq k} \|\mathbf{W} - \mathbf{A}\|_F$$

**Approximation error:**
$$\|\mathbf{W} - \mathbf{W}_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$$

### Compression Ratio

**Original storage:** $mn$ parameters

**Factorized storage:** $mk + k + kn = k(m + n + 1) \approx k(m + n)$

**Compression ratio:**
$$\rho = \frac{mn}{k(m + n)}$$

For compression, we need $k < \frac{mn}{m + n}$.

**Example:** For $\mathbf{W} \in \mathbb{R}^{1024 \times 1024}$:
- Original: 1,048,576 parameters
- With $k = 64$: 131,136 parameters (8× compression)
- With $k = 128$: 262,272 parameters (4× compression)

## Factorizing Linear Layers

### Basic SVD Decomposition

```python
import torch
import torch.nn as nn
from typing import Tuple, Dict, List


def svd_decomposition(weight: torch.Tensor, 
                      rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose weight matrix using truncated SVD.
    
    Args:
        weight: Weight matrix to decompose (m x n)
        rank: Target rank for decomposition
        
    Returns:
        U, V: Factor matrices such that weight ≈ U @ V.T
    """
    # Perform SVD
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    
    # Truncate to target rank
    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = Vh[:rank, :]
    
    # Absorb singular values into U
    U_scaled = U_r * S_r.unsqueeze(0)
    
    # Reconstruction: weight ≈ U_scaled @ V_r
    return U_scaled, V_r.T


def compute_reconstruction_error(original: torch.Tensor, 
                                 reconstructed: torch.Tensor) -> float:
    """Compute relative reconstruction error."""
    error = torch.norm(original - reconstructed) / torch.norm(original)
    return error.item()


# Example: Analyze compression vs error tradeoff
def analyze_rank_tradeoff(W: torch.Tensor):
    """Analyze error and compression at various ranks."""
    print("Rank Analysis:")
    print("-" * 60)
    
    for rank in [16, 32, 64, 128]:
        U, V = svd_decomposition(W, rank)
        W_approx = U @ V.T
        
        error = compute_reconstruction_error(W, W_approx)
        original_params = W.numel()
        factored_params = U.numel() + V.numel()
        compression = original_params / factored_params
        
        print(f"Rank {rank:3d}: Error={error:.4f}, "
              f"Compression={compression:.2f}x")
```

### FactorizedLinear Layer

```python
class FactorizedLinear(nn.Module):
    """
    Linear layer decomposed into two smaller layers.
    
    Original: y = Wx + b  (W is m x n)
    Factored: y = U(Vx) + b  (U is m x r, V is r x n)
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 rank: int, bias: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Factored layers
        self.V = nn.Linear(in_features, rank, bias=False)
        self.U = nn.Linear(rank, out_features, bias=bias)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize to approximate identity mapping scaled down."""
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.U.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.U(self.V(x))
    
    @classmethod
    def from_linear(cls, linear_layer: nn.Linear, rank: int) -> 'FactorizedLinear':
        """
        Create factorized layer from existing linear layer.
        
        Uses SVD to find optimal factorization.
        """
        W = linear_layer.weight.data
        b = linear_layer.bias.data if linear_layer.bias is not None else None
        
        # SVD decomposition
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        
        # Truncate and distribute singular values evenly
        U_r = U[:, :rank] * S[:rank].sqrt().unsqueeze(0)
        V_r = Vh[:rank, :] * S[:rank].sqrt().unsqueeze(1)
        
        # Create factorized layer
        factorized = cls(
            linear_layer.in_features,
            linear_layer.out_features,
            rank,
            bias=linear_layer.bias is not None
        )
        
        factorized.U.weight.data = U_r
        factorized.V.weight.data = V_r
        if b is not None:
            factorized.U.bias.data = b
        
        return factorized


def svd_decompose_linear(layer: nn.Linear,
                         rank: int) -> Tuple[nn.Linear, nn.Linear]:
    """
    Decompose a Linear layer using truncated SVD.
    
    W ≈ U_k * Σ_k * V_k^T = A * B
    
    Original: y = Wx + b
    Decomposed: y = A(Bx) + b
    
    Args:
        layer: Linear layer to decompose
        rank: Target rank for approximation
        
    Returns:
        Two Linear layers (first, second) where output = second(first(x))
    """
    W = layer.weight.data  # (out_features, in_features)
    b = layer.bias.data if layer.bias is not None else None
    
    # SVD decomposition
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
    # Truncate to rank k
    U_k = U[:, :rank]  # (out_features, rank)
    S_k = S[:rank]     # (rank,)
    V_k = Vh[:rank, :]  # (rank, in_features)
    
    # Create factorized layers
    # First layer: x -> intermediate (rank dimensions)
    first = nn.Linear(layer.in_features, rank, bias=False)
    first.weight.data = V_k  # (rank, in_features)
    
    # Second layer: intermediate -> output
    second = nn.Linear(rank, layer.out_features, bias=b is not None)
    second.weight.data = U_k @ torch.diag(S_k)  # (out_features, rank)
    if b is not None:
        second.bias.data = b
    
    return first, second
```

## Factorizing Convolutional Layers

### Spatial Decomposition (Separable Convolutions)

Decompose $k \times k$ convolution into $1 \times k$ and $k \times 1$ convolutions:

```python
class SeparableConv2d(nn.Module):
    """
    Spatially separable convolution.
    
    Replaces k×k convolution with 1×k followed by k×1.
    
    Original: O(C_in × C_out × k × k × H × W)
    Separable: O(C_in × C_out × 2k × H × W)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        
        # 1×k convolution
        self.conv_h = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            bias=False
        )
        
        # k×1 convolution
        self.conv_v = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_h(x)
        x = self.conv_v(x)
        return x
```

### Depthwise Separable Convolution (MobileNet Style)

Separate spatial and channel mixing:

```python
class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise separable convolution (MobileNet style).
    
    1. Depthwise: One k×k filter per input channel
    2. Pointwise: 1×1 convolution to mix channels
    
    Computation reduction: k²/(k² + C_out) ≈ 1/k² for large C_out
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        
        # Depthwise convolution (groups=in_channels)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Key: separate filter per channel
            bias=False
        )
        
        # Pointwise convolution (1×1)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# Compare parameters
def compare_conv_variants():
    in_ch, out_ch, kernel = 64, 128, 3
    
    standard = nn.Conv2d(in_ch, out_ch, kernel, padding=1)
    separable = DepthwiseSeparableConv2d(in_ch, out_ch, kernel, padding=1)
    
    std_params = sum(p.numel() for p in standard.parameters())
    sep_params = sum(p.numel() for p in separable.parameters())
    
    print(f"Standard Conv: {std_params:,} parameters")
    print(f"Depthwise Separable: {sep_params:,} parameters")
    print(f"Reduction: {std_params/sep_params:.1f}x")
```

### Channel-wise SVD Decomposition

```python
def decompose_conv_channel(conv: nn.Conv2d,
                           rank: int) -> nn.Sequential:
    """
    Decompose Conv2d using channel-wise factorization.
    
    Original: C_in -> C_out with k×k kernel
    Decomposed:
    1. C_in -> rank with k×k kernel
    2. rank -> C_out with 1×1 kernel (pointwise)
    """
    W = conv.weight.data  # (C_out, C_in, k_h, k_w)
    
    # Reshape to (C_out, C_in * k_h * k_w)
    W_mat = W.view(conv.out_channels, -1)
    
    # Apply SVD
    U, S, Vh = torch.linalg.svd(W_mat, full_matrices=False)
    
    # Truncate
    U_k = U[:, :rank]  # (C_out, rank)
    S_k = S[:rank]
    Vh_k = Vh[:rank, :]  # (rank, C_in * k_h * k_w)
    
    # Create factorized convolutions
    conv1 = nn.Conv2d(
        conv.in_channels, rank,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=False
    )
    conv1.weight.data = (Vh_k @ torch.diag(S_k ** 0.5)).view(
        rank, conv.in_channels, *conv.kernel_size
    )
    
    conv2 = nn.Conv2d(rank, conv.out_channels, kernel_size=1,
                      bias=conv.bias is not None)
    conv2.weight.data = (U_k @ torch.diag(S_k ** 0.5)).view(
        conv.out_channels, rank, 1, 1
    )
    if conv.bias is not None:
        conv2.bias.data = conv.bias.data
    
    return nn.Sequential(conv1, conv2)
```

### Tucker Decomposition

For higher-order tensors, Tucker decomposition provides a generalized low-rank approximation:

$$\mathcal{K} \approx \mathcal{G} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \times_3 \mathbf{U}^{(3)} \times_4 \mathbf{U}^{(4)}$$

```python
class TuckerConv2d(nn.Module):
    """
    Tucker decomposition for convolutional layers.
    
    Decomposes C_in × C_out × k × k into:
    1. 1×1 conv: C_in → r_in (compress input channels)
    2. r_in × r_out × k × k conv (smaller core)
    3. 1×1 conv: r_out → C_out (expand output channels)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 rank_in: int, rank_out: int, stride: int = 1, 
                 padding: int = 0, bias: bool = True):
        super().__init__()
        
        # Compress input channels
        self.compress = nn.Conv2d(in_channels, rank_in, 1, bias=False)
        
        # Core convolution (reduced rank)
        self.core = nn.Conv2d(
            rank_in, rank_out, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        
        # Expand output channels
        self.expand = nn.Conv2d(rank_out, out_channels, 1, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compress(x)
        x = self.core(x)
        x = self.expand(x)
        return x


def tucker_decompose_conv(conv: nn.Conv2d,
                          ranks: Tuple[int, int]) -> nn.Sequential:
    """
    Tucker decomposition for Conv2d layers using tensorly.
    
    Decomposes into three convolutions:
    1. 1x1 conv: C_in -> rank[1]
    2. kxk conv: rank[1] -> rank[0]
    3. 1x1 conv: rank[0] -> C_out
    """
    try:
        import tensorly as tl
        from tensorly.decomposition import partial_tucker
        tl.set_backend('pytorch')
    except ImportError:
        raise ImportError("Tucker decomposition requires tensorly: pip install tensorly")
    
    W = conv.weight.data
    core, factors = partial_tucker(W, modes=[0, 1], rank=ranks, init='svd')
    
    conv_input = nn.Conv2d(conv.in_channels, ranks[1], kernel_size=1, bias=False)
    conv_input.weight.data = factors[1].t().unsqueeze(-1).unsqueeze(-1)
    
    conv_spatial = nn.Conv2d(ranks[1], ranks[0], kernel_size=conv.kernel_size,
                             stride=conv.stride, padding=conv.padding, bias=False)
    conv_spatial.weight.data = core
    
    conv_output = nn.Conv2d(ranks[0], conv.out_channels, kernel_size=1,
                            bias=conv.bias is not None)
    conv_output.weight.data = factors[0].unsqueeze(-1).unsqueeze(-1)
    if conv.bias is not None:
        conv_output.bias.data = conv.bias.data
    
    return nn.Sequential(conv_input, conv_spatial, conv_output)
```

## Automatic Rank Selection

### Energy-Based Selection

```python
def select_rank_by_energy(weight: torch.Tensor,
                          energy_threshold: float = 0.95) -> int:
    """
    Select rank that preserves given fraction of energy (squared Frobenius norm).
    
    Args:
        weight: Weight matrix
        energy_threshold: Fraction of energy to preserve (0.95 = 95%)
        
    Returns:
        Optimal rank
    """
    _, S, _ = torch.linalg.svd(weight.view(weight.size(0), -1), 
                               full_matrices=False)
    
    # Compute cumulative energy
    total_energy = (S ** 2).sum()
    cumulative_energy = (S ** 2).cumsum(0) / total_energy
    
    # Find rank that achieves threshold
    rank = (cumulative_energy < energy_threshold).sum().item() + 1
    
    return rank


def analyze_singular_values(weight: torch.Tensor) -> Dict:
    """
    Analyze singular value distribution to determine optimal rank.
    """
    _, S, _ = torch.linalg.svd(weight.view(weight.size(0), -1), 
                               full_matrices=False)
    S = S.cpu().numpy()
    
    # Cumulative energy (explained variance)
    total_energy = (S ** 2).sum()
    cumulative_energy = (S ** 2).cumsum() / total_energy
    
    # Find rank for various energy thresholds
    thresholds = [0.9, 0.95, 0.99, 0.999]
    ranks_for_threshold = {}
    for thresh in thresholds:
        rank = (cumulative_energy < thresh).sum() + 1
        ranks_for_threshold[f'{thresh:.1%}_energy'] = int(rank)
    
    return {
        'singular_values': S,
        'cumulative_energy': cumulative_energy,
        'ranks_for_threshold': ranks_for_threshold,
        'effective_rank': int((S > S[0] * 0.01).sum()),
        'condition_number': S[0] / S[-1] if S[-1] > 0 else float('inf')
    }


def analyze_layer_ranks(model: nn.Module) -> List[Dict]:
    """
    Analyze effective ranks of all layers in a model.
    """
    results = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data
            _, S, _ = torch.linalg.svd(W, full_matrices=False)
            
            # Compute effective rank metrics
            total_energy = (S ** 2).sum()
            cumulative = (S ** 2).cumsum(0) / total_energy
            
            rank_95 = (cumulative < 0.95).sum().item() + 1
            rank_99 = (cumulative < 0.99).sum().item() + 1
            full_rank = min(W.shape)
            
            results.append({
                'layer': name,
                'shape': tuple(W.shape),
                'full_rank': full_rank,
                'rank_95': rank_95,
                'rank_99': rank_99,
                'compression_95': full_rank / rank_95
            })
    
    return results
```

## Model-Level Factorization

### Factorize Entire Model

```python
class LowRankModel(nn.Module):
    """
    Apply low-rank factorization to a pre-trained model.
    """
    
    def __init__(self, model: nn.Module, rank_ratio: float = 0.5,
                 min_rank: int = 8):
        super().__init__()
        self.model = self._factorize_model(model, rank_ratio, min_rank)
    
    def _factorize_model(self, model: nn.Module, rank_ratio: float,
                         min_rank: int) -> nn.Module:
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                max_rank = min(module.in_features, module.out_features)
                rank = max(min_rank, int(max_rank * rank_ratio))
                if rank < max_rank:
                    first, second = svd_decompose_linear(module, rank)
                    setattr(model, name, nn.Sequential(first, second))
            elif isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
                max_rank = min(module.in_channels, module.out_channels)
                rank = max(min_rank, int(max_rank * rank_ratio))
                if rank < max_rank:
                    decomposed = decompose_conv_channel(module, rank)
                    setattr(model, name, decomposed)
            else:
                self._factorize_model(module, rank_ratio, min_rank)
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def factorize_model(model: nn.Module, rank_ratio: float = 0.5, 
                    min_rank: int = 8) -> nn.Module:
    """
    Apply low-rank factorization to all linear layers.
    
    Args:
        model: PyTorch model
        rank_ratio: Fraction of full rank to use (0.5 = 50%)
        min_rank: Minimum rank to use
        
    Returns:
        Factorized model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Compute target rank
            full_rank = min(module.weight.shape)
            target_rank = max(min_rank, int(full_rank * rank_ratio))
            
            # Create factorized version
            factorized = FactorizedLinear.from_linear(module, target_rank)
            setattr(model, name, factorized)
            
            print(f"Factorized {name}: {module.weight.shape} → rank {target_rank}")
        
        elif len(list(module.children())) > 0:
            # Recursively process nested modules
            factorize_model(module, rank_ratio, min_rank)
    
    return model
```

### Fine-Tuning After Factorization

```python
def finetune_factorized_model(model: nn.Module,
                              train_loader: torch.utils.data.DataLoader,
                              val_loader: torch.utils.data.DataLoader,
                              epochs: int = 10,
                              lr: float = 1e-4,
                              device: str = 'cpu') -> Tuple[nn.Module, float]:
    """
    Fine-tune model after factorization to recover accuracy.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        best_accuracy = max(best_accuracy, accuracy)
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
              f"Accuracy={accuracy*100:.2f}%")
    
    return model, best_accuracy
```

## LoRA: Low-Rank Adaptation

LoRA (Low-Rank Adaptation) adds trainable low-rank matrices to frozen pretrained weights, enabling efficient fine-tuning of large models:

$$y = Wx + BAx$$

where $W$ is frozen and $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$ are trainable.

```python
class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer.
    
    Keeps original weights frozen and adds trainable low-rank updates:
    y = Wx + (BA)x
    
    Where W is frozen and B, A are trainable low-rank matrices.
    """
    
    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 16):
        super().__init__()
        
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha  # Scaling factor
        
        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False
        
        # Low-rank adaptation matrices
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Initialize A with small random values, B with zeros
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.scaling = alpha / rank
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output
        original_out = self.original(x)
        
        # LoRA contribution: x @ A.T @ B.T
        lora_out = x @ self.lora_A.T @ self.lora_B.T
        
        return original_out + self.scaling * lora_out
    
    def merge_weights(self):
        """Merge LoRA weights into original for inference."""
        with torch.no_grad():
            delta_W = self.scaling * (self.lora_B @ self.lora_A)
            self.original.weight.data += delta_W
    
    def get_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return self.lora_A.numel() + self.lora_B.numel()


def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16,
               target_modules: List[str] = None) -> nn.Module:
    """
    Apply LoRA to linear layers in model.
    
    Args:
        model: Model to modify
        rank: LoRA rank
        alpha: Scaling factor
        target_modules: List of module names to target (None = all Linear)
        
    Returns:
        Model with LoRA applied
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if target_modules is None or name in target_modules:
                setattr(model, name, LoRALinear(module, rank, alpha))
        elif len(list(module.children())) > 0:
            apply_lora(module, rank, alpha, target_modules)
    return model


def count_lora_params(model: nn.Module) -> Dict[str, int]:
    """Count trainable vs frozen parameters in LoRA model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    return {
        'trainable': trainable,
        'frozen': frozen,
        'total': trainable + frozen,
        'trainable_ratio': trainable / (trainable + frozen)
    }
```

## Trade-offs and Limitations

### Compression vs Accuracy

| Compression Ratio | Typical Accuracy Drop | Recovery Difficulty |
|-------------------|----------------------|---------------------|
| 2× | < 0.5% | Easy (brief fine-tuning) |
| 4× | 0.5-2% | Moderate |
| 8× | 2-5% | Challenging |
| 16× | 5%+ | Very challenging |

### When to Use Low-Rank Factorization

**Good candidates:**
- Large fully connected layers with many parameters
- Convolutional layers with many channels
- Layers with naturally low-rank weight distributions
- Fine-tuning scenarios (LoRA)

**Poor candidates:**
- Small layers (overhead may exceed savings)
- 1×1 convolutions (already efficient)
- First/last layers (often critical for accuracy)

### Computational Overhead

Factorization introduces additional operations:
- One matrix multiply becomes two sequential multiplies
- Memory access patterns may be less efficient
- GPU utilization may decrease for small intermediate ranks

## Summary

Low-rank factorization reduces model size and computation:

1. **SVD decomposition**: Optimal approximation, good for post-training compression
2. **Spatial separation**: Decompose convolutions spatially (1×k and k×1)
3. **Depthwise separable**: Standard for efficient architectures (MobileNet)
4. **Tucker decomposition**: Combined channel and spatial reduction
5. **LoRA**: Efficient fine-tuning of large pre-trained models

Key recommendations:
- Analyze layer ranks before deciding compression ratio
- Use energy-based rank selection (95-99% energy retention)
- Fine-tune after factorization to recover accuracy
- Combine with other techniques (quantization, pruning) for maximum compression
- Consider LoRA for efficient adaptation of large models

## References

1. Denton, E., et al. "Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation." NeurIPS 2014.
2. Jaderberg, M., et al. "Speeding up Convolutional Neural Networks with Low Rank Expansions." BMVC 2014.
3. Kim, Y., et al. "Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications." ICLR 2016.
4. Howard, A., et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." arXiv 2017.
5. Hu, E., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
6. Lebedev, V., et al. "Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition." ICLR 2015.


---

## Combined Compression Pipelines

NAS is most effective when combined with other compression techniques in a unified pipeline:

# Combined Compression Pipelines

## Overview

Maximum model compression is achieved by combining multiple techniques: pruning, quantization, knowledge distillation, and low-rank factorization. The key challenge is determining the optimal order and configuration of these methods, as they interact in complex ways.

## Compression Pipeline Design

### Technique Interactions

Different compression methods have synergistic and antagonistic interactions:

| Combination | Interaction | Notes |
|-------------|-------------|-------|
| Pruning → Quantization | Synergistic | Sparse weights have narrower distributions |
| Quantization → Pruning | Neutral/Negative | Quantized weights harder to rank by magnitude |
| Distillation → Pruning | Synergistic | Student architecture designed for sparsity |
| Distillation → Quantization | Synergistic | Distillation can compensate for quantization loss |
| Low-rank → Quantization | Synergistic | Factorized layers often have simpler distributions |

### Optimal Ordering

Based on empirical results, the recommended order is:

```
1. Knowledge Distillation (optional, if teacher available)
       ↓
2. Pruning (structured preferred for speedup)
       ↓  
3. Fine-tuning (recover from pruning)
       ↓
4. Low-Rank Factorization (optional)
       ↓
5. Quantization-Aware Training or PTQ
       ↓
6. Final calibration and export
```

**Rationale:**
- Distillation first: Student architecture optimized for downstream compression
- Pruning before quantization: Narrower weight distributions quantize better
- Fine-tuning between steps: Recover accuracy at each stage
- Quantization last: Benefits from optimized weight distributions

## PyTorch Implementation

### Complete Compression Pipeline

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quant
import copy
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CompressionConfig:
    """Configuration for compression pipeline."""
    # Distillation
    use_distillation: bool = True
    temperature: float = 4.0
    alpha: float = 0.5
    distillation_epochs: int = 20
    
    # Pruning
    target_sparsity: float = 0.7
    pruning_method: str = 'structured'  # 'unstructured', 'structured'
    pruning_epochs: int = 10
    
    # Low-rank
    use_low_rank: bool = False
    rank_ratio: float = 0.5
    
    # Quantization
    quantization_method: str = 'qat'  # 'ptq', 'qat'
    qat_epochs: int = 10
    
    # General
    learning_rate: float = 1e-3
    device: str = 'cpu'


class CompressionPipeline:
    """
    Unified compression pipeline combining multiple techniques.
    
    Supports:
    - Knowledge Distillation (from teacher)
    - Structured/Unstructured Pruning
    - Low-Rank Factorization
    - Post-Training and Quantization-Aware Training
    """
    
    def __init__(self,
                 student: nn.Module,
                 teacher: Optional[nn.Module] = None,
                 config: Optional[CompressionConfig] = None):
        """
        Args:
            student: Model to compress
            teacher: Optional teacher model for distillation
            config: Compression configuration
        """
        self.student = student
        self.teacher = teacher
        self.config = config or CompressionConfig()
        
        # Track compression stages
        self.history = {
            'stage': [],
            'accuracy': [],
            'size_mb': [],
            'sparsity': []
        }
    
    def compress(self,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 calibration_loader: Optional[torch.utils.data.DataLoader] = None
                 ) -> nn.Module:
        """
        Execute full compression pipeline.
        
        Args:
            train_loader: Training data
            test_loader: Test data for evaluation
            calibration_loader: Calibration data for PTQ (uses train_loader if None)
            
        Returns:
            Compressed model
        """
        device = self.config.device
        model = self.student.to(device)
        
        # Log initial state
        self._log_state('initial', model, test_loader)
        
        # Stage 1: Knowledge Distillation
        if self.config.use_distillation and self.teacher is not None:
            print("\n" + "="*60)
            print("STAGE 1: Knowledge Distillation")
            print("="*60)
            model = self._distillation_stage(model, train_loader, test_loader)
            self._log_state('after_distillation', model, test_loader)
        
        # Stage 2: Pruning
        print("\n" + "="*60)
        print("STAGE 2: Pruning")
        print("="*60)
        model = self._pruning_stage(model, train_loader, test_loader)
        self._log_state('after_pruning', model, test_loader)
        
        # Stage 3: Low-Rank Factorization (optional)
        if self.config.use_low_rank:
            print("\n" + "="*60)
            print("STAGE 3: Low-Rank Factorization")
            print("="*60)
            model = self._low_rank_stage(model, train_loader, test_loader)
            self._log_state('after_low_rank', model, test_loader)
        
        # Stage 4: Quantization
        print("\n" + "="*60)
        print("STAGE 4: Quantization")
        print("="*60)
        
        cal_loader = calibration_loader or train_loader
        
        if self.config.quantization_method == 'qat':
            model = self._qat_stage(model, train_loader, test_loader)
        else:
            model = self._ptq_stage(model, cal_loader)
        
        self._log_state('final', model, test_loader)
        
        # Print summary
        self._print_summary()
        
        return model
    
    def _distillation_stage(self,
                            model: nn.Module,
                            train_loader: torch.utils.data.DataLoader,
                            test_loader: torch.utils.data.DataLoader) -> nn.Module:
        """Execute distillation training."""
        device = self.config.device
        model = model.to(device)
        teacher = self.teacher.to(device)
        teacher.eval()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.distillation_epochs
        )
        
        T = self.config.temperature
        alpha = self.config.alpha
        
        for epoch in range(self.config.distillation_epochs):
            model.train()
            epoch_loss = 0.0
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                with torch.no_grad():
                    teacher_logits = teacher(data)
                
                optimizer.zero_grad()
                student_logits = model(data)
                
                # Distillation loss
                hard_loss = F.cross_entropy(student_logits, target)
                soft_loss = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction='batchmean'
                ) * (T ** 2)
                
                loss = alpha * hard_loss + (1 - alpha) * soft_loss
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                acc = self._evaluate(model, test_loader)
                print(f"Distillation Epoch {epoch+1}/{self.config.distillation_epochs}, "
                      f"Loss: {epoch_loss/len(train_loader):.4f}, Acc: {acc*100:.2f}%")
        
        return model
    
    def _pruning_stage(self,
                       model: nn.Module,
                       train_loader: torch.utils.data.DataLoader,
                       test_loader: torch.utils.data.DataLoader) -> nn.Module:
        """Execute pruning with fine-tuning."""
        device = self.config.device
        model = model.to(device)
        
        if self.config.pruning_method == 'structured':
            model = self._structured_pruning(model, train_loader, test_loader)
        else:
            model = self._unstructured_pruning(model, train_loader, test_loader)
        
        return model
    
    def _structured_pruning(self,
                            model: nn.Module,
                            train_loader: torch.utils.data.DataLoader,
                            test_loader: torch.utils.data.DataLoader) -> nn.Module:
        """Apply structured (filter) pruning."""
        import torch.nn.utils.prune as prune
        
        # Calculate amount to achieve target sparsity
        amount = self.config.target_sparsity
        
        # Apply structured pruning to conv layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
        
        # Fine-tune
        model = self._fine_tune(model, train_loader, test_loader, 
                               self.config.pruning_epochs)
        
        # Make pruning permanent
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass
        
        return model
    
    def _unstructured_pruning(self,
                              model: nn.Module,
                              train_loader: torch.utils.data.DataLoader,
                              test_loader: torch.utils.data.DataLoader) -> nn.Module:
        """Apply unstructured (weight) pruning."""
        import torch.nn.utils.prune as prune
        
        amount = self.config.target_sparsity
        
        # Global unstructured pruning
        parameters_to_prune = [
            (module, 'weight') 
            for module in model.modules() 
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
        
        # Fine-tune with mask maintenance
        masks = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight_mask'):
                    masks[name] = module.weight_mask.clone()
        
        model = self._fine_tune(model, train_loader, test_loader,
                               self.config.pruning_epochs, masks=masks)
        
        # Make pruning permanent
        for module, _ in parameters_to_prune:
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass
        
        return model
    
    def _low_rank_stage(self,
                        model: nn.Module,
                        train_loader: torch.utils.data.DataLoader,
                        test_loader: torch.utils.data.DataLoader) -> nn.Module:
        """Apply low-rank factorization."""
        # Replace large linear layers with factorized versions
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                max_rank = min(module.in_features, module.out_features)
                rank = max(8, int(max_rank * self.config.rank_ratio))
                
                if rank < max_rank and module.in_features > 64:
                    # Factorize
                    first, second = self._svd_factorize_linear(module, rank)
                    
                    # Replace in model
                    parts = name.split('.')
                    parent = model
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, parts[-1], nn.Sequential(first, second))
        
        # Fine-tune
        model = self._fine_tune(model, train_loader, test_loader, 5)
        
        return model
    
    def _svd_factorize_linear(self, layer: nn.Linear, rank: int) -> Tuple[nn.Linear, nn.Linear]:
        """Factorize linear layer using SVD."""
        W = layer.weight.data
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        
        U_k = U[:, :rank]
        S_k = S[:rank]
        V_k = Vh[:rank, :]
        
        first = nn.Linear(layer.in_features, rank, bias=False)
        first.weight.data = V_k
        
        second = nn.Linear(rank, layer.out_features, bias=layer.bias is not None)
        second.weight.data = U_k @ torch.diag(S_k)
        if layer.bias is not None:
            second.bias.data = layer.bias.data
        
        return first, second
    
    def _qat_stage(self,
                   model: nn.Module,
                   train_loader: torch.utils.data.DataLoader,
                   test_loader: torch.utils.data.DataLoader) -> nn.Module:
        """Quantization-Aware Training."""
        device = self.config.device
        model = model.to(device)
        model.train()
        
        # Prepare for QAT
        model.qconfig = quant.get_default_qat_qconfig('fbgemm')
        quant.prepare_qat(model, inplace=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate * 0.1)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.config.qat_epochs):
            model.train()
            
            # Freeze observers after half the epochs
            if epoch >= self.config.qat_epochs // 2:
                model.apply(quant.disable_observer)
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 2 == 0:
                acc = self._evaluate(model, test_loader)
                print(f"QAT Epoch {epoch+1}/{self.config.qat_epochs}, Acc: {acc*100:.2f}%")
        
        # Convert to quantized model
        model.eval()
        model_quantized = quant.convert(model.cpu(), inplace=False)
        
        return model_quantized
    
    def _ptq_stage(self,
                   model: nn.Module,
                   calibration_loader: torch.utils.data.DataLoader) -> nn.Module:
        """Post-Training Quantization."""
        model.eval()
        
        # Dynamic quantization (simpler, works for most cases)
        model_quantized = quant.quantize_dynamic(
            model.cpu(),
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        return model_quantized
    
    def _fine_tune(self,
                   model: nn.Module,
                   train_loader: torch.utils.data.DataLoader,
                   test_loader: torch.utils.data.DataLoader,
                   epochs: int,
                   masks: Optional[Dict] = None) -> nn.Module:
        """Fine-tune model with optional mask maintenance."""
        device = self.config.device
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate * 0.1)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Maintain masks if provided
                if masks:
                    with torch.no_grad():
                        for name, module in model.named_modules():
                            if name in masks and hasattr(module, 'weight'):
                                module.weight.data *= masks[name].to(device)
            
            if (epoch + 1) % 2 == 0:
                acc = self._evaluate(model, test_loader)
                print(f"Fine-tune Epoch {epoch+1}/{epochs}, Acc: {acc*100:.2f}%")
        
        return model
    
    def _evaluate(self, model: nn.Module, 
                  test_loader: torch.utils.data.DataLoader) -> float:
        """Evaluate model accuracy."""
        device = self.config.device
        model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / total
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def _get_sparsity(self, model: nn.Module) -> float:
        """Calculate model sparsity."""
        total, zeros = 0, 0
        for param in model.parameters():
            if param.dim() > 1:
                total += param.numel()
                zeros += (param.data.abs() < 1e-8).sum().item()
        return zeros / total if total > 0 else 0.0
    
    def _log_state(self, stage: str, model: nn.Module,
                   test_loader: torch.utils.data.DataLoader):
        """Log compression state."""
        acc = self._evaluate(model, test_loader)
        size = self._get_model_size(model)
        sparsity = self._get_sparsity(model)
        
        self.history['stage'].append(stage)
        self.history['accuracy'].append(acc)
        self.history['size_mb'].append(size)
        self.history['sparsity'].append(sparsity)
        
        print(f"\n[{stage}] Accuracy: {acc*100:.2f}%, "
              f"Size: {size:.2f} MB, Sparsity: {sparsity*100:.1f}%")
    
    def _print_summary(self):
        """Print compression summary."""
        print("\n" + "="*70)
        print("COMPRESSION PIPELINE SUMMARY")
        print("="*70)
        print(f"\n{'Stage':<25} {'Accuracy':<12} {'Size (MB)':<12} {'Sparsity':<12}")
        print("-" * 70)
        
        for i, stage in enumerate(self.history['stage']):
            print(f"{stage:<25} {self.history['accuracy'][i]*100:>8.2f}%   "
                  f"{self.history['size_mb'][i]:>8.2f}     "
                  f"{self.history['sparsity'][i]*100:>8.1f}%")
        
        # Compute overall compression
        initial_size = self.history['size_mb'][0]
        final_size = self.history['size_mb'][-1]
        initial_acc = self.history['accuracy'][0]
        final_acc = self.history['accuracy'][-1]
        
        print("\n" + "="*70)
        print("OVERALL METRICS")
        print("="*70)
        print(f"Compression Ratio:    {initial_size/final_size:.1f}×")
        print(f"Size Reduction:       {(1 - final_size/initial_size)*100:.1f}%")
        print(f"Accuracy Change:      {(final_acc - initial_acc)*100:+.2f}%")
        print("="*70)
```

## Usage Example

```python
# Create models
teacher = LargeTeacherModel()  # Pre-trained teacher
student = SmallStudentModel()   # Student to compress

# Load pre-trained teacher
teacher.load_state_dict(torch.load('teacher.pth'))

# Configure compression
config = CompressionConfig(
    use_distillation=True,
    temperature=4.0,
    alpha=0.5,
    distillation_epochs=20,
    target_sparsity=0.7,
    pruning_method='structured',
    pruning_epochs=10,
    use_low_rank=False,
    quantization_method='qat',
    qat_epochs=10,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Create and run pipeline
pipeline = CompressionPipeline(student, teacher, config)
compressed_model = pipeline.compress(train_loader, test_loader)

# Export
torch.save(compressed_model.state_dict(), 'compressed_model.pth')
```

## Best Practices

### Configuration Guidelines

| Scenario | Distillation | Pruning | Low-Rank | Quantization |
|----------|-------------|---------|----------|--------------|
| Maximum compression | ✓ | Structured 80% | ✓ | QAT INT8 |
| Balanced | ✓ | Structured 50% | ✗ | QAT INT8 |
| Quick deployment | ✗ | ✗ | ✗ | PTQ INT8 |
| Mobile/Edge | ✓ | Structured 70% | ✗ | QAT INT8/INT4 |
| Server | ✓ | Unstructured 90% | ✓ | FP16 |

### Debugging Pipeline Issues

1. **Accuracy drops too much at one stage**: Reduce that stage's aggressiveness
2. **Quantization fails**: Check for unsupported operations; use dynamic quantization
3. **No speedup from pruning**: Switch to structured pruning
4. **Distillation not helping**: Verify teacher accuracy; adjust temperature

## References

1. Polino, A., et al. "Model Compression via Distillation and Quantization." ICLR 2018.
2. Han, S., et al. "Deep Compression." ICLR 2016.
3. Cheng, Y., et al. "A Survey of Model Compression and Acceleration." IEEE Signal Processing 2020.
