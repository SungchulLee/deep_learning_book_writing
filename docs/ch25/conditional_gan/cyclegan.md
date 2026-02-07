# CycleGAN

CycleGAN (Zhu et al., 2017) enables unpaired image-to-image translation using cycle consistency loss.

## Problem: Unpaired Translation

Unlike Pix2Pix (paired data), CycleGAN works with unpaired datasets:
- Horses ↔ Zebras
- Summer ↔ Winter
- Photos ↔ Monet paintings

## Architecture

### Two Generators, Two Discriminators

```
Domain X (e.g., horses)     Domain Y (e.g., zebras)
    x ----G_XY----> ŷ
    x <---G_YX---- ŷ (cycle)
    
    y ----G_YX----> x̂
    y <---G_XY---- x̂ (cycle)
```

### Cycle Consistency Loss

Key insight: Translation should be reversible.

$$\mathcal{L}_{cyc}(G_{XY}, G_{YX}) = \mathbb{E}_x[\|G_{YX}(G_{XY}(x)) - x\|_1] + \mathbb{E}_y[\|G_{XY}(G_{YX}(y)) - y\|_1]$$

```python
def cycle_consistency_loss(G_XY, G_YX, real_X, real_Y):
    # X -> Y -> X
    fake_Y = G_XY(real_X)
    cycled_X = G_YX(fake_Y)
    loss_cycle_X = F.l1_loss(cycled_X, real_X)
    
    # Y -> X -> Y
    fake_X = G_YX(real_Y)
    cycled_Y = G_XY(fake_X)
    loss_cycle_Y = F.l1_loss(cycled_Y, real_Y)
    
    return loss_cycle_X + loss_cycle_Y
```

### Identity Loss (Optional)

Preserve color when already in target domain:

```python
def identity_loss(G_XY, G_YX, real_X, real_Y):
    # G_XY should be identity on Y
    same_Y = G_XY(real_Y)
    loss_identity_Y = F.l1_loss(same_Y, real_Y)
    
    # G_YX should be identity on X
    same_X = G_YX(real_X)
    loss_identity_X = F.l1_loss(same_X, real_X)
    
    return loss_identity_X + loss_identity_Y
```

## Full Objective

$$\mathcal{L} = \mathcal{L}_{GAN}(G_{XY}, D_Y) + \mathcal{L}_{GAN}(G_{YX}, D_X) + \lambda_{cyc}\mathcal{L}_{cyc} + \lambda_{id}\mathcal{L}_{id}$$

```python
def cyclegan_loss(G_XY, G_YX, D_X, D_Y, real_X, real_Y, lambda_cyc=10, lambda_id=5):
    # GAN losses
    fake_Y = G_XY(real_X)
    loss_GAN_XY = mse_loss(D_Y(fake_Y), ones)
    
    fake_X = G_YX(real_Y)
    loss_GAN_YX = mse_loss(D_X(fake_X), ones)
    
    # Cycle loss
    loss_cycle = cycle_consistency_loss(G_XY, G_YX, real_X, real_Y)
    
    # Identity loss
    loss_identity = identity_loss(G_XY, G_YX, real_X, real_Y)
    
    # Total generator loss
    G_loss = loss_GAN_XY + loss_GAN_YX + lambda_cyc * loss_cycle + lambda_id * loss_identity
    
    return G_loss
```

## Generator Architecture

ResNet-based generator (9 residual blocks for 256×256):

```python
class ResNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=9):
        super().__init__()
        
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        
        # Downsampling
        for i in range(2):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf*mult, ngf*mult*2, 3, 2, 1),
                nn.InstanceNorm2d(ngf*mult*2),
                nn.ReLU(True)
            ]
        
        # Residual blocks
        mult = 2 ** 2
        for i in range(n_blocks):
            model += [ResidualBlock(ngf * mult)]
        
        # Upsampling
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [
                nn.ConvTranspose2d(ngf*mult, ngf*mult//2, 3, 2, 1, 1),
                nn.InstanceNorm2d(ngf*mult//2),
                nn.ReLU(True)
            ]
        
        # Output
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
```

## Training Tips

1. **Use LSGAN loss** (MSE instead of BCE)
2. **Image buffer** to stabilize D training
3. **Instance normalization** (not batch)
4. **Learning rate decay** after 100 epochs

## Summary

| Component | Value |
|-----------|-------|
| Generators | 2 (G_XY, G_YX) |
| Discriminators | 2 (D_X, D_Y) |
| λ_cycle | 10 |
| λ_identity | 5 |
| Normalization | Instance Norm |

## Complete Inference Example

The following provides a production-ready CycleGAN inference script using a pretrained ResNet-based generator for horse-to-zebra translation.

### ResNet Generator Architecture

```python
"""
CycleGAN Inference: Horse → Zebra Style Transfer
==================================================

This script demonstrates CycleGAN inference using a pretrained ResNet-based
generator. CycleGAN learns unpaired image-to-image translation by training
two generators (G: X→Y, F: Y→X) and two discriminators with cycle-consistency
loss:

    L_cycle = ||F(G(x)) - x||₁ + ||G(F(y)) - y||₁

Architecture Overview:
    - Generator: ResNet with 9 residual blocks
    - Normalization: InstanceNorm2d (style transfer standard)
    - Activation: ReLU (hidden), Tanh (output → [-1, 1])
    - Downsampling: Strided convolutions (2×)
    - Upsampling: Transposed convolutions (2×)

Reference:
    Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent
    Adversarial Networks", ICCV 2017.

Source:
    https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/p1ch2/3_cyclegan.ipynb
"""

import argparse
import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# =============================================================================
# 1. Configuration
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CycleGAN Horse→Zebra Inference")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--image_path", type=str, default="./image/horse.jpg",
                        help="Path to input horse image")
    parser.add_argument("--model_path", type=str, default="./model/horse2zebra_0.4.0.pth",
                        help="Path to pretrained generator weights")
    parser.add_argument("--output_path", type=str, default="./output/result.png",
                        help="Path to save output comparison image")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Resize input image to this size")
    args = parser.parse_args()
    return args


# =============================================================================
# 2. Download Utilities
# =============================================================================

def download_file(url: str, save_path: str, description: str = "file") -> None:
    """Download a file from URL if it does not already exist locally.

    Args:
        url: Remote URL to download from.
        save_path: Local path to save the downloaded file.
        description: Human-readable description for logging.
    """
    if os.path.exists(save_path):
        print(f"[INFO] {description} already exists: {save_path}")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"[INFO] Downloading {description} from {url}...")
    response = requests.get(url)

    if response.status_code == 200:
        if description == "image":
            image = Image.open(BytesIO(response.content))
            image.save(save_path)
        else:
            with open(save_path, "wb") as f:
                f.write(response.content)
        print(f"[INFO] Saved {description} to {save_path}")
    else:
        raise RuntimeError(
            f"Failed to download {description}. HTTP status: {response.status_code}"
        )


def download_assets(args):
    """Download the horse image and pretrained generator weights."""
    download_file(
        url="https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/data/p1ch2/horse.jpg?raw=true",
        save_path=args.image_path,
        description="image",
    )
    download_file(
        url="https://github.com/deep-learning-with-pytorch/dlwpt-code/raw/master/data/p1ch2/horse2zebra_0.4.0.pth",
        save_path=args.model_path,
        description="model weights",
    )


# =============================================================================
# 3. Data Loading
# =============================================================================

def load_image(image_path: str, image_size: int = 256) -> torch.Tensor:
    """Load and preprocess an image for CycleGAN inference.

    The generator expects input in [0, 1] range (no explicit normalization
    to [-1, 1] is applied here; the model handles this internally).

    Args:
        image_path: Path to the input image file.
        image_size: Target spatial resolution.

    Returns:
        Tensor of shape (1, 3, H, W) in [0, 1] range.
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path)
    tensor = transform(img)
    batch = tensor.unsqueeze(0)  # (1, 3, H, W)
    return batch


# =============================================================================
# 4. Model Architecture
# =============================================================================

class ResNetBlock(nn.Module):
    """Residual block with two 3×3 convolutions and InstanceNorm.

    Architecture:
        x → ReflectionPad → Conv3×3 → InstanceNorm → ReLU
          → ReflectionPad → Conv3×3 → InstanceNorm → (+x) → out

    The skip connection preserves spatial dimensions and enables
    gradient flow through deep generator networks.

    Args:
        dim: Number of input/output channels.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class ResNetGenerator(nn.Module):
    """CycleGAN generator based on ResNet architecture.

    Architecture (for default n_blocks=9):
        Input (3 channels)
        → ReflectionPad(3) → Conv7×7 → InstanceNorm → ReLU       [ngf=64]
        → Conv3×3(s=2) → InstanceNorm → ReLU                      [128]
        → Conv3×3(s=2) → InstanceNorm → ReLU                      [256]
        → 9 × ResNetBlock                                          [256]
        → ConvTranspose3×3(s=2) → InstanceNorm → ReLU             [128]
        → ConvTranspose3×3(s=2) → InstanceNorm → ReLU             [64]
        → ReflectionPad(3) → Conv7×7 → Tanh                       [3]
        Output (3 channels, range [-1, 1])

    Key design choices:
        - ReflectionPadding reduces boundary artifacts vs zero-padding
        - InstanceNorm (not BatchNorm) for style transfer tasks
        - Tanh output constrains pixel values to [-1, 1]

    Args:
        input_nc: Number of input channels (default: 3 for RGB).
        output_nc: Number of output channels (default: 3 for RGB).
        ngf: Base number of generator filters (default: 64).
        n_blocks: Number of residual blocks (default: 9).
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        n_blocks: int = 9,
    ):
        assert n_blocks >= 0
        super().__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        # --- Initial convolution block ---
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]

        # --- Downsampling (2 stages: 64→128→256) ---
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2,
                          kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]

        # --- Residual blocks (at 256 channels) ---
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        # --- Upsampling (2 stages: 256→128→64) ---
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2,
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=True),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(True),
            ]

        # --- Output convolution ---
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the generator.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W) in [-1, 1].
        """
        return self.model(x)


# =============================================================================
# 5. Inference and Visualization
# =============================================================================

def run_inference(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """Run generator inference and convert output to displayable image.

    The generator outputs values in [-1, 1]. We rescale to [0, 1] for display.

    Args:
        model: Pretrained CycleGAN generator.
        input_tensor: Input image tensor of shape (1, 3, H, W).

    Returns:
        NumPy array of shape (H, W, 3) in [0, 255] uint8 range.
    """
    with torch.no_grad():
        output = model(input_tensor)

    # Rescale from [-1, 1] → [0, 1]
    output_rescaled = (output.squeeze(0) + 1.0) / 2.0
    output_image = transforms.ToPILImage()(output_rescaled)
    return np.array(output_image)


def visualize_results(
    input_tensor: torch.Tensor,
    output_image: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Display input and generated images side by side.

    Args:
        input_tensor: Original input tensor of shape (1, 3, H, W).
        output_image: Generated image as NumPy array (H, W, 3).
        save_path: If provided, save the figure to this path.
    """
    fig, (ax_input, ax_output) = plt.subplots(1, 2, figsize=(10, 5))

    # Input image: tensor (1, 3, H, W) → (H, W, 3)
    input_np = input_tensor.squeeze(0).permute(1, 2, 0).numpy()
    ax_input.imshow(input_np)
    ax_input.set_title("Input (Horse)")
    ax_input.axis("off")

    # Generated image
    ax_output.imshow(output_image)
    ax_output.set_title("Generated (Zebra)")
    ax_output.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Result saved to {save_path}")

    plt.show()


# =============================================================================
# 6. Main
# =============================================================================

def main():
    args = parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Download assets
    download_assets(args)

    # Load model
    model = ResNetGenerator()
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    print(f"[INFO] Generator loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load input image
    input_tensor = load_image(args.image_path, args.image_size)
    print(f"[INFO] Input tensor shape: {input_tensor.shape}")

    # Run inference
    output_image = run_inference(model, input_tensor)

    # Visualize
    visualize_results(input_tensor, output_image, save_path=args.output_path)


if __name__ == "__main__":
    main()

```

This inference script demonstrates the complete CycleGAN pipeline from loading a pretrained model to generating translated images. The ResNet-based generator with 9 residual blocks and InstanceNorm is the standard architecture for 256×256 image translation tasks.
