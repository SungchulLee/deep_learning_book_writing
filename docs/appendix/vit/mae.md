# MAE

MAE was introduced in the 2021 paper "Masked Autoencoders Are Scalable Vision Learners." - Mask most image patches (e.g., 75%)   - Encode only visible patches   - Lightweight decoder reconstructs masked patches.

This implementation provides a concise, educational reference for MAE. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
MAE - Masked Autoencoders
Paper: "Masked Autoencoders Are Scalable Vision Learners" (2021)
Authors: Kaiming He et al.
Key idea:
  - Mask most image patches (e.g., 75%)
  - Encode only visible patches
  - Lightweight decoder reconstructs masked patches

File: appendix/vit/mae.py
Note: Educational implementation (encoder-decoder structure).
"""

import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================


class MAE(nn.Module):
    """
    Masked Autoencoder with ViT-style encoder and decoder.
    """
    def __init__(self, embed_dim=768, decoder_dim=512, num_patches=196):
        super().__init__()

        # Encoder processes visible patches only
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=12, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # Decoder reconstructs masked patches
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=8, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=8)

        self.enc_to_dec = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.head = nn.Linear(decoder_dim, embed_dim)

    def forward(self, x, mask):
        """
        x   : (B, N, D) patch embeddings
        mask: (B, N) True for masked patches
        """
        # Keep only visible patches
        visible = x[~mask].view(x.size(0), -1, x.size(-1))

        # Encode visible patches
        enc = self.encoder(visible)

        # Project to decoder dimension
        dec_input = self.enc_to_dec(enc)

        # Append mask tokens for reconstruction
        num_masked = mask.sum(dim=1).max()
        mask_tokens = self.mask_token.expand(x.size(0), num_masked, -1)

        dec_input = torch.cat([dec_input, mask_tokens], dim=1)

        # Decode
        dec = self.decoder(dec_input)

        # Predict reconstructed patch embeddings
        recon = self.head(dec)
        return recon


if __name__ == "__main__":
    pass```

## Discussion

The `MAE` class encapsulates the model architecture using PyTorch's `nn.Module` interface. The `forward` method defines the computational graph, allowing PyTorch's autograd system to handle gradient computation automatically during training. This modular design makes it straightforward to modify individual components or integrate the model into larger pipelines.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `MAE` with the default initialization. Break down the count by layer, including both weights and biases.

??? success "Solution to Exercise 1"
    For each `nn.Linear(in_features, out_features)`, there are `in_features * out_features` weight parameters plus `out_features` bias parameters (unless `bias=False`). For `nn.Conv2d(in_c, out_c, k)`, there are `in_c * out_c * k * k` weight parameters plus `out_c` bias parameters. For `nn.Embedding(num, dim)`, there are `num * dim` parameters. Sum across all layers. You can verify with `sum(p.numel() for p in model.parameters())`.

---

**Exercise 2.**
Add input validation to the main function or class to check that inputs have the expected shape and dtype. Raise informative error messages for invalid inputs.

??? success "Solution to Exercise 2"
    At the start of the `forward` method (or relevant function), add checks like: `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'` and `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. For shape validation, check critical dimensions: `B, C, H, W = x.shape; assert C == self.expected_channels`. Informative error messages significantly speed up debugging and make the code more robust for reuse.

---

**Exercise 3.**
Describe two potential failure modes of this implementation and explain how you would diagnose and fix each one.

??? success "Solution to Exercise 3"
    Common failure modes include: (1) **Vanishing/exploding gradients** -- diagnosed by monitoring gradient norms (`torch.nn.utils.clip_grad_norm_` or logging `param.grad.norm()` per layer). Fix with gradient clipping, better initialization (Xavier/Kaiming), or architectural changes (residual connections, normalization). (2) **Overfitting** -- diagnosed when training loss decreases but validation loss increases. Fix with regularization (dropout, weight decay, data augmentation) or reducing model capacity. Always monitor both training and validation metrics to catch these issues early.

---

**Exercise 4.**
Extend `MAE` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = MAE(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
