# PixelCNN: Autoregressive Image Generation

## Introduction

PixelCNN represents a landmark achievement in autoregressive generative modeling, extending the chain rule factorization to image data. Introduced by van den Oord et al. (2016), PixelCNN models the joint distribution of image pixels by predicting each pixel conditioned on all previously generated pixels. The key innovation lies in using **masked convolutions** to enforce the autoregressive property while maintaining the computational efficiency of convolutional neural networks.

## The Autoregressive Image Model

### Raster Scan Ordering

For an image with height $H$ and width $W$, we have $n = H \times W$ pixels. PixelCNN adopts a **raster scan ordering**: pixels are generated left-to-right within each row, and rows are processed top-to-bottom.

For pixel at position $(i, j)$, the conditioning set includes:
- All pixels in rows above row $i$
- All pixels to the left of position $j$ in row $i$

The joint distribution factorizes as:

$$P(\mathbf{x}) = \prod_{i=1}^{H} \prod_{j=1}^{W} P(x_{i,j} | x_{<(i,j)})$$

where $x_{<(i,j)}$ denotes all pixels preceding $(i,j)$ in raster scan order.

### Color Channels

For RGB images, each pixel has three channels $(R, G, B)$. PixelCNN factorizes within each pixel as well:

$$P(x_{i,j}) = P(R_{i,j} | x_{<(i,j)}) \cdot P(G_{i,j} | x_{<(i,j)}, R_{i,j}) \cdot P(B_{i,j} | x_{<(i,j)}, R_{i,j}, G_{i,j})$$

This captures correlations between color channels within each pixel.

## Masked Convolutions

### The Masking Problem

Standard convolutions have symmetric receptive fields—each output depends on input values both before and after the current position. For autoregressive modeling, we need convolutions that only look at "previous" pixels.

### Mask Types

PixelCNN uses two types of masks:

**Type A Mask (First Layer Only):**
- Excludes the current pixel entirely
- Ensures the prediction for pixel $(i,j)$ cannot see $x_{i,j}$

**Type B Mask (Subsequent Layers):**
- Includes the current pixel
- Allows information from the current pixel's features (but not its value) to propagate

For a $k \times k$ kernel centered at position $(i,j)$:

```
Type A Mask (k=5):          Type B Mask (k=5):
1 1 1 1 1                   1 1 1 1 1
1 1 1 1 1                   1 1 1 1 1
1 1 0 0 0                   1 1 1 0 0
0 0 0 0 0                   0 0 0 0 0
0 0 0 0 0                   0 0 0 0 0
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

class MaskedConv2d(nn.Conv2d):
    """
    Masked convolution for autoregressive image modeling.
    
    Ensures each pixel only depends on pixels above and to the left
    (raster scan order).
    
    Args:
        mask_type: 'A' for first layer (excludes center), 
                   'B' for subsequent layers (includes center)
        *args, **kwargs: Standard Conv2d arguments
    """
    
    def __init__(
        self, 
        mask_type: Literal['A', 'B'],
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        **kwargs
    ):
        # Ensure odd kernel size for symmetric mask construction
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        # Padding to maintain spatial dimensions
        padding = kernel_size // 2
        super().__init__(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=padding, 
            **kwargs
        )
        
        self.mask_type = mask_type
        
        # Create and register the mask
        self.register_buffer('mask', self._create_mask())
    
    def _create_mask(self) -> torch.Tensor:
        """
        Create the autoregressive mask.
        
        Returns:
            Mask tensor of shape [out_channels, in_channels, k, k]
        """
        k = self.kernel_size[0]
        mask = torch.ones(self.out_channels, self.in_channels, k, k)
        
        center = k // 2
        
        # Zero out everything below center row
        mask[:, :, center + 1:, :] = 0
        
        # Zero out right side of center row
        if self.mask_type == 'A':
            # Exclude center pixel
            mask[:, :, center, center:] = 0
        else:  # Type B
            # Include center pixel
            mask[:, :, center, center + 1:] = 0
        
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply masked convolution."""
        # Apply mask to weights
        self.weight.data *= self.mask
        return super().forward(x)


class MaskedConv2dColor(nn.Module):
    """
    Masked convolution with color channel dependencies.
    
    For RGB images, ensures:
    - R can see: previous pixels
    - G can see: previous pixels + current R
    - B can see: previous pixels + current R,G
    """
    
    def __init__(
        self,
        mask_type: Literal['A', 'B'],
        in_channels: int,
        out_channels: int,
        kernel_size: int
    ):
        super().__init__()
        
        assert in_channels % 3 == 0 and out_channels % 3 == 0
        
        self.mask_type = mask_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Standard spatial masked conv
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=padding,
            bias=True
        )
        
        # Create and register spatial + channel mask
        self.register_buffer('mask', self._create_mask(kernel_size))
    
    def _create_mask(self, k: int) -> torch.Tensor:
        """Create mask enforcing both spatial and channel dependencies."""
        mask = torch.ones(self.out_channels, self.in_channels, k, k)
        center = k // 2
        
        # Spatial masking (same as before)
        mask[:, :, center + 1:, :] = 0
        
        # For center row, need careful channel handling
        in_per_channel = self.in_channels // 3
        out_per_channel = self.out_channels // 3
        
        for out_c in range(3):  # R=0, G=1, B=2
            for in_c in range(3):
                out_start = out_c * out_per_channel
                out_end = (out_c + 1) * out_per_channel
                in_start = in_c * in_per_channel
                in_end = (in_c + 1) * in_per_channel
                
                if self.mask_type == 'A':
                    # Type A: strictly previous channels
                    if in_c >= out_c:
                        mask[out_start:out_end, in_start:in_end, center, center:] = 0
                else:
                    # Type B: include current channel
                    if in_c > out_c:
                        mask[out_start:out_end, in_start:in_end, center, center:] = 0
                    elif in_c == out_c:
                        mask[out_start:out_end, in_start:in_end, center, center + 1:] = 0
        
        # Zero out below center (spatial)
        mask[:, :, center + 1:, :] = 0
        
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.conv.weight.data *= self.mask
        return self.conv(x)
```

## PixelCNN Architecture

### Basic Architecture

The original PixelCNN consists of:
1. Initial Type A masked convolution
2. Stack of Type B masked convolution blocks
3. Output convolutions predicting pixel distributions

```python
class PixelCNN(nn.Module):
    """
    Original PixelCNN architecture for grayscale images.
    
    Predicts categorical distribution over 256 intensity values.
    """
    
    def __init__(
        self,
        n_channels: int = 64,
        n_layers: int = 7,
        kernel_size: int = 7,
        n_classes: int = 256  # 8-bit grayscale
    ):
        super().__init__()
        
        self.n_classes = n_classes
        
        # Initial layer: Type A (can't see current pixel)
        self.input_conv = MaskedConv2d(
            'A', 
            in_channels=1, 
            out_channels=n_channels,
            kernel_size=kernel_size
        )
        
        # Hidden layers: Type B (can see current pixel features)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                MaskedConv2d(
                    'B',
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=kernel_size
                ),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(n_layers)
        ])
        
        # Output: predict distribution over pixel values
        self.output_conv = nn.Sequential(
            MaskedConv2d('B', n_channels, n_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            MaskedConv2d('B', n_channels, n_classes, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for each pixel's intensity distribution.
        
        Args:
            x: Input image [batch, 1, H, W], values in [0, 1]
            
        Returns:
            Logits [batch, n_classes, H, W]
        """
        # Initial masked convolution
        h = F.relu(self.input_conv(x))
        
        # Hidden layers with residual connections
        for layer in self.hidden_layers:
            h = h + layer(h)
        
        # Output logits
        logits = self.output_conv(h)
        
        return logits
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            x: Images with values in [0, 1]
        """
        # Get logits
        logits = self.forward(x)
        
        # Convert continuous values to class indices
        targets = (x * (self.n_classes - 1)).long().squeeze(1)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, targets)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self, 
        n_samples: int, 
        height: int, 
        width: int,
        temperature: float = 1.0,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Generate images autoregressively.
        
        Args:
            n_samples: Number of images to generate
            height, width: Image dimensions
            temperature: Sampling temperature
            device: Device to generate on
            
        Returns:
            Generated images [n_samples, 1, height, width]
        """
        self.eval()
        
        # Initialize with zeros
        samples = torch.zeros(n_samples, 1, height, width, device=device)
        
        # Generate pixel by pixel
        for i in range(height):
            for j in range(width):
                # Get prediction for all pixels
                logits = self.forward(samples)
                
                # Extract logits for current position
                logits_ij = logits[:, :, i, j] / temperature
                
                # Sample from categorical distribution
                probs = F.softmax(logits_ij, dim=1)
                pixel_class = torch.multinomial(probs, 1)
                
                # Convert back to [0, 1] range
                samples[:, 0, i, j] = pixel_class.squeeze(1).float() / (self.n_classes - 1)
        
        return samples
```

### The Blind Spot Problem

The original PixelCNN has a **blind spot**: the receptive field doesn't fully cover the context region. For pixel $(i,j)$, some pixels in the upper-right quadrant are not reachable through stacked masked convolutions.

This occurs because:
- Vertical context can only flow straight down
- There's no mechanism for information to flow diagonally

## Gated PixelCNN

### Addressing the Blind Spot

Gated PixelCNN (van den Oord et al., 2016) solves the blind spot problem by separating **vertical** and **horizontal** processing streams:

- **Vertical stack**: Processes all rows above the current row
- **Horizontal stack**: Processes the current row, receives information from vertical stack

```python
class GatedBlock(nn.Module):
    """
    Gated activation unit used in Gated PixelCNN.
    
    Applies: tanh(W_f * x) ⊙ sigmoid(W_g * x)
    where ⊙ is element-wise multiplication.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Outputs 2x channels: half for tanh, half for sigmoid gate
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=1)
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        # Split into tanh and gate portions
        h_tanh = torch.tanh(h[:, :self.out_channels])
        h_gate = torch.sigmoid(h[:, self.out_channels:])
        return h_tanh * h_gate


class VerticalMaskedConv(nn.Module):
    """Vertical stack: sees all pixels above current row."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        # Vertical conv: k x k kernel, but only top half + center row
        # Pad top to maintain causal structure
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size=(kernel_size // 2 + 1, kernel_size),
            padding=(kernel_size // 2, kernel_size // 2)
        )
        # Remove bottom padding to enforce causality
        self.crop = kernel_size // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        # Crop bottom to maintain causality
        if self.crop > 0:
            h = h[:, :, :-self.crop, :]
        return h


class HorizontalMaskedConv(nn.Module):
    """Horizontal stack: sees pixels to the left in current row."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        mask_type: Literal['A', 'B'] = 'B'
    ):
        super().__init__()
        self.mask_type = mask_type
        
        # Horizontal conv: 1 x k kernel
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2)
        )
        
        # Create mask
        k = kernel_size
        mask = torch.ones(out_channels, in_channels, 1, k)
        center = k // 2
        
        if mask_type == 'A':
            mask[:, :, :, center:] = 0
        else:
            mask[:, :, :, center + 1:] = 0
        
        self.register_buffer('mask', mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.conv.weight.data *= self.mask
        return self.conv(x)


class GatedPixelCNNBlock(nn.Module):
    """
    Single block of Gated PixelCNN.
    
    Processes vertical and horizontal streams with gated activations.
    """
    
    def __init__(
        self, 
        n_channels: int, 
        kernel_size: int = 3,
        mask_type: Literal['A', 'B'] = 'B'
    ):
        super().__init__()
        
        # Vertical stack processing
        self.vertical_conv = VerticalMaskedConv(
            n_channels, 2 * n_channels, kernel_size
        )
        
        # Vertical to horizontal connection
        self.v_to_h = nn.Conv2d(2 * n_channels, 2 * n_channels, kernel_size=1)
        
        # Horizontal stack processing
        self.horizontal_conv = HorizontalMaskedConv(
            n_channels, 2 * n_channels, kernel_size, mask_type
        )
        
        # Output projection
        self.output_conv = nn.Conv2d(n_channels, n_channels, kernel_size=1)
        
        self.n_channels = n_channels
    
    def forward(
        self, 
        v_input: torch.Tensor, 
        h_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process both stacks.
        
        Args:
            v_input: Vertical stack input [batch, channels, H, W]
            h_input: Horizontal stack input [batch, channels, H, W]
            
        Returns:
            (vertical_output, horizontal_output)
        """
        # Vertical stack
        v = self.vertical_conv(v_input)
        v_out = self._gated_activation(v)
        
        # Horizontal stack (receives from vertical)
        h = self.horizontal_conv(h_input) + self.v_to_h(v)
        h_gated = self._gated_activation(h)
        h_out = self.output_conv(h_gated)
        
        # Residual connection on horizontal stack
        h_out = h_out + h_input
        
        return v_out, h_out
    
    def _gated_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated activation: tanh(x1) * sigmoid(x2)"""
        x_tanh = torch.tanh(x[:, :self.n_channels])
        x_gate = torch.sigmoid(x[:, self.n_channels:])
        return x_tanh * x_gate


class GatedPixelCNN(nn.Module):
    """
    Complete Gated PixelCNN architecture.
    """
    
    def __init__(
        self,
        n_channels: int = 64,
        n_layers: int = 7,
        kernel_size: int = 5,
        n_classes: int = 256
    ):
        super().__init__()
        
        self.n_classes = n_classes
        
        # Initial convolutions (Type A - can't see current pixel)
        self.v_input = VerticalMaskedConv(1, n_channels, kernel_size)
        self.h_input = HorizontalMaskedConv(1, n_channels, kernel_size, 'A')
        
        # Stack of gated blocks
        self.layers = nn.ModuleList([
            GatedPixelCNNBlock(n_channels, kernel_size, 'B')
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 1),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_classes, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for pixel distributions."""
        # Initial processing
        v = self.v_input(x)
        h = self.h_input(x)
        
        # Process through gated blocks
        for layer in self.layers:
            v, h = layer(v, h)
        
        # Output logits
        logits = self.output(h)
        
        return logits
```

## PixelCNN++

### Improvements Over Original

PixelCNN++ (Salimans et al., 2017) introduces several enhancements:

1. **Discretized Logistic Mixture**: More flexible output distribution
2. **Downsampling**: Multi-resolution processing for efficiency
3. **Skip connections**: Better gradient flow
4. **Dropout**: Regularization

### Discretized Logistic Mixture

Instead of a 256-way categorical, PixelCNN++ uses a mixture of logistic distributions:

$$P(x | \pi, \mu, s) = \sum_{i=1}^{K} \pi_i \left[ \sigma\left(\frac{x + 0.5 - \mu_i}{s_i}\right) - \sigma\left(\frac{x - 0.5 - \mu_i}{s_i}\right) \right]$$

where $\sigma$ is the sigmoid function. This parameterization:
- Requires fewer parameters (3K vs 256)
- Naturally handles continuous underlying values
- Is differentiable for training

```python
class DiscretizedLogisticMixture(nn.Module):
    """
    Discretized mixture of logistics for modeling pixel intensities.
    
    More efficient than categorical: predicts K mixture components
    instead of 256 classes.
    """
    
    def __init__(self, n_mixtures: int = 10):
        super().__init__()
        self.n_mixtures = n_mixtures
    
    def forward(
        self, 
        x: torch.Tensor, 
        params: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log-likelihood of x under mixture model.
        
        Args:
            x: Pixel values in [-1, 1], shape [batch, 1, H, W]
            params: Mixture parameters [batch, 3*K, H, W]
                   (K logit weights, K means, K log-scales)
        
        Returns:
            Log-likelihood [batch, 1, H, W]
        """
        K = self.n_mixtures
        
        # Extract parameters
        logit_weights = params[:, :K]        # Mixture weights (logits)
        means = params[:, K:2*K]              # Means
        log_scales = params[:, 2*K:3*K]       # Log scales
        
        # Clamp log scales for stability
        log_scales = log_scales.clamp(min=-7.0)
        
        # Center x around means
        x = x.unsqueeze(1).expand_as(means)  # [batch, K, H, W]
        centered = x - means
        
        # Compute CDF at x + 0.5 and x - 0.5
        inv_scales = torch.exp(-log_scales)
        plus_cdf = torch.sigmoid((centered + 1/255) * inv_scales)
        minus_cdf = torch.sigmoid((centered - 1/255) * inv_scales)
        
        # Probability mass in bin
        cdf_delta = plus_cdf - minus_cdf
        
        # Handle edge cases (x near -1 or 1)
        log_pdf_mid = centered * inv_scales - log_scales - 2 * F.softplus(centered * inv_scales)
        
        # Edge handling
        log_cdf_plus = torch.where(
            x > 0.999,
            -F.softplus(-centered * inv_scales),
            torch.log(plus_cdf.clamp(min=1e-12))
        )
        log_one_minus_cdf = torch.where(
            x < -0.999,
            -F.softplus(centered * inv_scales),
            torch.log((1 - minus_cdf).clamp(min=1e-12))
        )
        
        # Log probability for middle of distribution
        log_probs = torch.where(
            cdf_delta > 1e-5,
            torch.log(cdf_delta.clamp(min=1e-12)),
            log_pdf_mid - torch.log(torch.tensor(127.5))
        )
        
        # Combine with mixture weights
        log_probs = log_probs + F.log_softmax(logit_weights, dim=1)
        
        # Log-sum-exp over mixture components
        return torch.logsumexp(log_probs, dim=1, keepdim=True)
    
    def sample(self, params: torch.Tensor) -> torch.Tensor:
        """
        Sample from the mixture model.
        
        Args:
            params: Mixture parameters [batch, 3*K, H, W]
            
        Returns:
            Samples [batch, 1, H, W] in [-1, 1]
        """
        K = self.n_mixtures
        batch, _, H, W = params.shape
        
        # Extract parameters
        logit_weights = params[:, :K]
        means = params[:, K:2*K]
        log_scales = params[:, 2*K:3*K].clamp(min=-7.0)
        
        # Sample mixture component
        weights = F.softmax(logit_weights, dim=1)
        # Reshape for multinomial: [batch * H * W, K]
        weights_flat = weights.permute(0, 2, 3, 1).reshape(-1, K)
        component = torch.multinomial(weights_flat, 1).squeeze(-1)
        
        # Gather parameters for selected components
        component = component.reshape(batch, H, W)
        idx = component.unsqueeze(1)  # [batch, 1, H, W]
        
        mu = means.gather(1, idx).squeeze(1)
        log_s = log_scales.gather(1, idx).squeeze(1)
        
        # Sample from logistic
        u = torch.rand_like(mu).clamp(1e-5, 1 - 1e-5)
        x = mu + torch.exp(log_s) * (torch.log(u) - torch.log(1 - u))
        
        # Clamp to valid range
        return x.clamp(-1, 1).unsqueeze(1)
```

## Training and Generation

### Training Procedure

```python
def train_pixelcnn(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    n_epochs: int,
    learning_rate: float = 1e-3,
    device: str = 'cuda'
):
    """
    Train PixelCNN model.
    
    Args:
        model: PixelCNN model
        train_loader: Training data loader
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            # Forward pass
            loss = model.compute_loss(images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")


def evaluate_bpd(model: nn.Module, test_loader: torch.utils.data.DataLoader, device: str = 'cuda'):
    """
    Evaluate model using bits-per-dimension (BPD).
    
    Lower BPD indicates better density estimation.
    """
    model.eval()
    total_bpd = 0
    n_samples = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            batch_size, channels, height, width = images.shape
            
            # Compute negative log-likelihood
            nll = model.compute_loss(images)
            
            # Convert to bits per dimension
            # BPD = NLL / (log(2) * n_dimensions)
            n_dims = channels * height * width
            bpd = nll / (torch.log(torch.tensor(2.0)) * n_dims)
            
            total_bpd += bpd.item() * batch_size
            n_samples += batch_size
    
    return total_bpd / n_samples
```

### Generation with Class Conditioning

For conditional generation, we can incorporate class labels:

```python
class ConditionalPixelCNN(nn.Module):
    """
    Class-conditional PixelCNN.
    
    Generates images conditioned on class labels.
    """
    
    def __init__(
        self,
        n_channels: int = 64,
        n_layers: int = 7,
        n_classes: int = 256,
        n_conditions: int = 10  # e.g., MNIST digits
    ):
        super().__init__()
        
        self.n_pixel_classes = n_classes
        
        # Class embedding
        self.class_embed = nn.Embedding(n_conditions, n_channels)
        
        # Base PixelCNN layers (same as before)
        self.input_conv = MaskedConv2d('A', 1, n_channels, kernel_size=7)
        
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                MaskedConv2d('B', n_channels, n_channels, kernel_size=7),
                nn.BatchNorm2d(n_channels),
                nn.ReLU()
            )
            for _ in range(n_layers)
        ])
        
        self.output_conv = nn.Sequential(
            MaskedConv2d('B', n_channels, n_channels, 1),
            nn.ReLU(),
            MaskedConv2d('B', n_channels, n_classes, 1)
        )
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with class conditioning.
        
        Args:
            x: Images [batch, 1, H, W]
            labels: Class labels [batch]
            
        Returns:
            Logits [batch, n_classes, H, W]
        """
        # Get class embedding
        class_emb = self.class_embed(labels)  # [batch, n_channels]
        class_emb = class_emb.unsqueeze(-1).unsqueeze(-1)  # [batch, n_channels, 1, 1]
        
        # Initial conv
        h = F.relu(self.input_conv(x))
        
        # Add class conditioning
        h = h + class_emb
        
        # Hidden layers
        for layer in self.hidden_layers:
            h = h + layer(h + class_emb)
        
        return self.output_conv(h)
    
    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        height: int,
        width: int,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Generate images conditioned on class labels."""
        self.eval()
        batch_size = labels.shape[0]
        
        samples = torch.zeros(batch_size, 1, height, width, device=device)
        
        for i in range(height):
            for j in range(width):
                logits = self.forward(samples, labels)
                probs = F.softmax(logits[:, :, i, j], dim=1)
                pixel = torch.multinomial(probs, 1)
                samples[:, 0, i, j] = pixel.squeeze(1).float() / (self.n_pixel_classes - 1)
        
        return samples
```

## Applications and Extensions

### Image Inpainting

PixelCNN's autoregressive structure makes it naturally suited for inpainting:

```python
@torch.no_grad()
def inpaint(model, image, mask, device='cpu'):
    """
    Inpaint masked regions autoregressively.
    
    Args:
        model: Trained PixelCNN
        image: Original image [1, 1, H, W]
        mask: Binary mask [1, 1, H, W], 1 = known, 0 = missing
    """
    model.eval()
    result = image.clone().to(device)
    mask = mask.to(device)
    H, W = image.shape[-2:]
    
    # Generate missing pixels in raster order
    for i in range(H):
        for j in range(W):
            if mask[0, 0, i, j] == 0:  # Missing pixel
                logits = model(result)
                probs = F.softmax(logits[:, :, i, j], dim=1)
                pixel = torch.multinomial(probs, 1)
                result[0, 0, i, j] = pixel.float() / (model.n_classes - 1)
    
    return result
```

### Density Estimation for Anomaly Detection

The tractable likelihood enables anomaly detection:

```python
def detect_anomalies(model, images, threshold):
    """
    Detect anomalies using likelihood threshold.
    
    Args:
        model: Trained PixelCNN
        images: Test images
        threshold: NLL threshold for anomaly
        
    Returns:
        Boolean mask indicating anomalies
    """
    model.eval()
    
    with torch.no_grad():
        # Compute per-image NLL
        logits = model(images)
        targets = (images * (model.n_classes - 1)).long().squeeze(1)
        
        # Per-pixel cross-entropy
        nll = F.cross_entropy(logits, targets, reduction='none')
        
        # Sum over pixels for total NLL
        nll_per_image = nll.sum(dim=(1, 2))
    
    # Flag as anomaly if NLL exceeds threshold
    return nll_per_image > threshold
```

## Computational Considerations

### Generation Speed

The main limitation of PixelCNN is slow generation. For a $H \times W$ image:
- Each pixel requires a full forward pass
- Total: $H \times W$ forward passes
- For 28×28 MNIST: 784 passes
- For 256×256 images: 65,536 passes

### Acceleration Strategies

1. **Caching**: Store activations that don't change between positions
2. **Fast PixelCNN**: Restructure computation for parallelism
3. **Multi-scale**: Generate at low resolution, then upsample

```python
class FastPixelCNNCache:
    """
    Cache for accelerating PixelCNN generation.
    
    Stores intermediate activations that can be reused.
    """
    
    def __init__(self):
        self.vertical_cache = {}
        self.current_row = -1
    
    def update(self, row: int, vertical_output: torch.Tensor):
        """Cache vertical stack output for current row."""
        self.vertical_cache[row] = vertical_output
        self.current_row = row
    
    def get_vertical(self, row: int) -> torch.Tensor:
        """Retrieve cached vertical output."""
        return self.vertical_cache.get(row)
```

## Summary

PixelCNN demonstrates that autoregressive modeling can scale to high-dimensional image data through clever architectural design:

1. **Masked convolutions** enforce the autoregressive property efficiently
2. **Gated activations** and **vertical/horizontal stacks** address the blind spot problem
3. **Discretized logistics mixtures** provide flexible output distributions
4. **Conditional generation** enables controlled synthesis

While generation remains sequential, PixelCNN's tractable likelihood makes it valuable for density estimation, compression, and anomaly detection applications.

## References

1. van den Oord, A., Kalchbrenner, N., & Kavukcuoglu, K. (2016). Pixel Recurrent Neural Networks. *ICML*.
2. van den Oord, A., et al. (2016). Conditional Image Generation with PixelCNN Decoders. *NeurIPS*.
3. Salimans, T., Karpathy, A., Chen, X., & Kingma, D. P. (2017). PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood. *ICLR*.
4. Chen, X., et al. (2018). PixelSNAIL: An Improved Autoregressive Generative Model. *ICML*.

---

## Exercises

1. **Mask Visualization**: Implement a function to visualize the effective receptive field of a PixelCNN at each layer.

2. **Blind Spot Analysis**: Train a basic PixelCNN and a Gated PixelCNN on the same dataset. Compare their ability to capture diagonal dependencies.

3. **Conditional Generation**: Extend the ConditionalPixelCNN to handle continuous conditioning variables (e.g., latent codes from an encoder).

4. **Speed Optimization**: Implement caching for the vertical stack to accelerate row-by-row generation.
