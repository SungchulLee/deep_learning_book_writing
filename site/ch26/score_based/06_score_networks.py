"""
FILE: 06_score_networks.py
DIFFICULTY: Intermediate
ESTIMATED TIME: 2-3 hours
PREREQUISITES: 04_denoising_score_matching.py

LEARNING OBJECTIVES:
    1. Design effective score network architectures
    2. Implement noise-conditional score networks
    3. Understand normalization and activation choices
    4. Learn best practices for score modeling

MATHEMATICAL BACKGROUND:
    Score networks must output vector fields that represent ∇log p(x).
    
    Key design considerations:
    1. No final activation (outputs can be any real value)
    2. Often conditioned on noise level σ
    3. Should be Lipschitz continuous for stability
    4. Larger capacity for complex distributions
"""

import torch
import torch.nn as nn


class NoiseConditionalScoreNetwork(nn.Module):
    """
    Score network that conditions on noise level.
    
    Architecture: s_θ(x, σ) outputs the score of the distribution p_σ(x).
    This is crucial for multi-scale score modeling.
    """
    
    def __init__(self, data_dim=2, hidden_dims=[128, 128, 128],
                 sigma_encoding_dim=32):
        super().__init__()
        
        # Noise level encoding (sinusoidal positional encoding)
        self.sigma_encoding_dim = sigma_encoding_dim
        
        # Main network
        layers = []
        input_dim = data_dim + sigma_encoding_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.GroupNorm(min(8, h_dim//8), h_dim),  # Normalization
                nn.SiLU(),  # Smooth activation
            ])
            input_dim = h_dim
        
        layers.append(nn.Linear(input_dim, data_dim))
        self.network = nn.Sequential(*layers)
    
    def sigma_embedding(self, sigma):
        """
        Sinusoidal embedding of noise level.
        
        Similar to Transformer positional encodings.
        """
        half_dim = self.sigma_encoding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=sigma.device) * -emb)
        emb = sigma[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, x, sigma):
        """
        Forward pass conditioned on noise level.
        
        Args:
            x: Data points, shape (N, D)
            sigma: Noise levels, shape (N,) or scalar
        
        Returns:
            score: Score vectors, shape (N, D)
        """
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor([sigma] * len(x), device=x.device)
        elif sigma.dim() == 0:
            sigma = sigma.repeat(len(x))
        
        # Encode noise level
        sigma_emb = self.sigma_embedding(sigma)
        
        # Concatenate with data
        x_cond = torch.cat([x, sigma_emb], dim=-1)
        
        # Pass through network
        return self.network(x_cond)


class ResidualBlock(nn.Module):
    """Residual block for deeper score networks."""
    
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GroupNorm(8, dim),
        )
        self.activation = nn.SiLU()
    
    def forward(self, x):
        return self.activation(x + self.block(x))


class DeepScoreNetwork(nn.Module):
    """Deep score network with residual connections."""
    
    def __init__(self, data_dim=2, hidden_dim=128, n_blocks=4):
        super().__init__()
        
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, data_dim)
    
    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


def demo_architectures():
    """Compare different score network architectures."""
    print("Score Network Architectures Demo")
    print("=" * 80)
    
    x = torch.randn(32, 2)
    
    # Test basic network
    basic_net = nn.Sequential(
        nn.Linear(2, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 2)
    )
    
    # Test noise-conditional network
    cond_net = NoiseConditionalScoreNetwork(data_dim=2)
    
    # Test deep network
    deep_net = DeepScoreNetwork(data_dim=2, hidden_dim=64, n_blocks=3)
    
    print("\nArchitecture comparison:")
    print(f"Basic network parameters: {sum(p.numel() for p in basic_net.parameters()):,}")
    print(f"Conditional network parameters: {sum(p.numel() for p in cond_net.parameters()):,}")
    print(f"Deep network parameters: {sum(p.numel() for p in deep_net.parameters()):,}")
    
    # Test forward pass
    basic_out = basic_net(x)
    cond_out = cond_net(x, sigma=0.5)
    deep_out = deep_net(x)
    
    print(f"\nOutput shapes:")
    print(f"Basic: {basic_out.shape}")
    print(f"Conditional: {cond_out.shape}")
    print(f"Deep: {deep_out.shape}")
    
    print("\n✓ All architectures tested successfully!")


if __name__ == "__main__":
    demo_architectures()
