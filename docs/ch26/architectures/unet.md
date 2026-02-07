# U-Net Denoiser Architecture

The **U-Net** is the standard neural network architecture for diffusion models. It predicts the noise added to data at each timestep.

## Overview

The U-Net is an encoder-decoder architecture with:
1. **Encoder**: Progressively downsamples and extracts features
2. **Bottleneck**: Processes at lowest resolution
3. **Decoder**: Progressively upsamples with skip connections
4. **Conditioning**: Timestep and optional class/text embeddings

## Architecture Diagram

```
Input x_t (C×H×W)
    │
    ▼ Conv
┌───────────────────┐
│  Encoder Block 1  │──────────────────────────────┐
│  (C×H×W)          │                              │ Skip
└───────────────────┘                              │
    │ Downsample                                   │
    ▼                                              │
┌───────────────────┐                              │
│  Encoder Block 2  │─────────────────────┐        │
│  (2C×H/2×W/2)     │                     │ Skip   │
└───────────────────┘                     │        │
    │ Downsample                          │        │
    ▼                                     │        │
┌───────────────────┐                     │        │
│  Encoder Block 3  │────────────┐        │        │
│  (4C×H/4×W/4)     │            │ Skip   │        │
└───────────────────┘            │        │        │
    │ Downsample                 │        │        │
    ▼                            │        │        │
┌───────────────────┐            │        │        │
│    Bottleneck     │            │        │        │
│  (8C×H/8×W/8)     │            │        │        │
└───────────────────┘            │        │        │
    │ Upsample                   │        │        │
    ▼                            ▼        │        │
┌───────────────────┐     ┌──────────┐    │        │
│  Decoder Block 3  │◄────│  Concat  │    │        │
│  (4C×H/4×W/4)     │     └──────────┘    │        │
└───────────────────┘                     │        │
    │ Upsample                            ▼        │
    ▼                              ┌──────────┐    │
┌───────────────────┐              │  Concat  │    │
│  Decoder Block 2  │◄─────────────┘          │    │
│  (2C×H/2×W/2)     │                         │    │
└───────────────────┘                         │    │
    │ Upsample                                ▼    │
    ▼                                  ┌──────────┐│
┌───────────────────┐                  │  Concat  ││
│  Decoder Block 1  │◄─────────────────┘          ││
│  (C×H×W)          │◄────────────────────────────┘│
└───────────────────┘                              │
    │                                              │
    ▼ Conv                                         │
Output ε_θ (C×H×W)
```

## Core Components

### ResNet Block

```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)
```

### Timestep Embedding

Sinusoidal position encoding (like Transformers):

```python
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
```

### Self-Attention Block

```python
class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h*w).transpose(1, 2)  # (B, H*W, C)
        x_norm = self.norm(x.view(b, c, -1)).view(b, c, h*w).transpose(1, 2)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        return x + attn_out.transpose(1, 2).view(b, c, h, w)
```

### Cross-Attention for Conditioning

```python
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)
    
    def forward(self, x, context):
        # x: image features (B, H*W, C)
        # context: text embeddings (B, seq_len, context_dim)
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = q.view(q.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        out = attn @ v
        
        # Reshape back
        out = out.transpose(1, 2).reshape(x.shape[0], -1, q.shape[2] * self.head_dim)
        return self.to_out(out)
```

## Complete U-Net Implementation

```python
class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        channel_mult=[1, 2, 4, 8],
        num_heads=8,
        context_dim=768,  # For text conditioning
    ):
        super().__init__()
        
        # Time embedding
        time_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Input convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        ch = model_channels
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.encoder.append(ResBlock(ch, out_ch, time_dim))
                ch = out_ch
            
            # Attention at certain resolutions
            if 64 // (2 ** level) in attention_resolutions:
                self.encoder_attns.append(SelfAttention(ch, num_heads))
            else:
                self.encoder_attns.append(nn.Identity())
            
            # Downsample (except last level)
            if level < len(channel_mult) - 1:
                self.encoder.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResBlock(ch, ch, time_dim),
            SelfAttention(ch, num_heads),
            ResBlock(ch, ch, time_dim),
        ])
        
        # Decoder (mirror of encoder)
        self.decoder = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        
        for level, mult in enumerate(reversed(channel_mult)):
            for i in range(num_res_blocks + 1):
                skip_ch = model_channels * mult if i == 0 else 0
                out_ch = model_channels * mult
                self.decoder.append(ResBlock(ch + skip_ch, out_ch, time_dim))
                ch = out_ch
            
            if 64 // (2 ** (len(channel_mult) - 1 - level)) in attention_resolutions:
                self.decoder_attns.append(SelfAttention(ch, num_heads))
            else:
                self.decoder_attns.append(nn.Identity())
            
            if level < len(channel_mult) - 1:
                self.decoder.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
        
        # Output
        self.output = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )
    
    def forward(self, x, t, context=None):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        h = self.input_conv(x)
        skips = [h]
        
        for layer in self.encoder:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
            skips.append(h)
        
        # Bottleneck
        for layer in self.bottleneck:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # Decoder with skip connections
        for layer in self.decoder:
            if isinstance(layer, ResBlock):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        return self.output(h)
```

## Key Design Choices

### Normalization

- **GroupNorm** instead of BatchNorm (works with small batches)
- Typically 32 groups

### Activation

- **SiLU/Swish** ($x \cdot \sigma(x)$) throughout
- Smooth, non-monotonic

### Attention Placement

- Only at low resolutions (8×8, 16×16)
- Self-attention is O(n²) in spatial dimension
- Cross-attention for conditioning

### Channel Progression

Typical multipliers: `[1, 2, 4, 8]`
- Base channels: 128-256
- Maximum channels: 512-1024

## Summary

The U-Net for diffusion models combines:
1. **Hierarchical structure** with skip connections
2. **Timestep conditioning** via sinusoidal embeddings
3. **Attention mechanisms** for global context
4. **ResNet blocks** for stable training

This architecture effectively learns to predict noise across all timesteps, enabling high-quality generation through the reverse diffusion process.
