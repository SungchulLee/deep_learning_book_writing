# WAV2VEC

WAV2VEC was introduced in the 2020 paper "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." Contrastive learning for speech, quantization, transformer encoder.

This implementation provides a concise, educational reference for WAV2VEC. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
'''
Wav2Vec 2.0 - Self-Supervised Learning for Speech Recognition
Paper: "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (2020)
Key: Contrastive learning for speech, quantization, transformer encoder
'''
import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================

class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 512, 10, stride=5, bias=False),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, 3, stride=2, bias=False),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, 3, stride=2, bias=False),
            nn.GroupNorm(512, 512),
            nn.GELU(),
        )
    
    def forward(self, x):
        return self.conv_layers(x)

class Wav2Vec2(nn.Module):
    def __init__(self, d_model=768, n_layers=12, n_heads=12):
        super().__init__()
        self.feature_extractor = FeatureEncoder()
        self.feature_projection = nn.Linear(512, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.quantizer = nn.Linear(d_model, 320)
    
    def forward(self, x):
        # x: [batch, 1, time]
        features = self.feature_extractor(x)
        features = features.transpose(1, 2)
        features = self.feature_projection(features)
        
        context = self.transformer(features)
        
        quantized = self.quantizer(context)
        
        return context, quantized

if __name__ == "__main__":
    model = Wav2Vec2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(2, 1, 16000)
    print(f"Input: {x.shape}")```

## Discussion

The implementation defines 2 classes (`FeatureEncoder`, `Wav2Vec2`) that work together to form the complete transformer architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Trace the tensor shapes through the `FeatureEncoder` forward pass. For a batch of 4 input samples with the default parameters, write down the shape after each major operation (convolution, pooling, linear layer).

??? success "Solution to Exercise 1"
    Start from the input shape and apply each layer sequentially. For each `Conv2d(in_c, out_c, k)`, the spatial dimensions change as $H_{\text{out}} = H_{\text{in}} - k + 1$ (without padding) or remain the same (with `padding=k//2`). Pooling with kernel 2 halves spatial dimensions. Linear layers transform the last dimension. Track the batch dimension throughout: it remains unchanged. Write each intermediate shape as $(B, C, H, W)$ for convolutional layers and $(B, F)$ after flattening.

---

**Exercise 2.**
Modify the architecture to accept RGB images of size $64 \times 64$ (input shape: $3 \times 64 \times 64$). Update all layer dimensions accordingly and verify the model runs without errors.

??? success "Solution to Exercise 2"
    Change the first convolutional layer's `in_channels` from the current value to 3. Recalculate the spatial dimensions after each convolution and pooling layer using the formulas $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$. Update the first linear layer's `in_features` to match the flattened output of the last convolutional/pooling layer. Verify with: `model = FeatureEncoder(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

**Exercise 3.**
Compare the number of parameters and FLOPs between a standard convolution and a depthwise separable convolution for the same input/output dimensions. When is the computational saving most significant?

??? success "Solution to Exercise 3"
    A standard `Conv2d(C_in, C_out, k)` has $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$ parameters. A depthwise separable convolution splits this into: (1) depthwise: $C_{{\text{{in}}}} \times k^2$ parameters (one filter per input channel), and (2) pointwise: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$ parameters (1x1 conv). The ratio of parameters is approximately $1/C_{{\text{{out}}}} + 1/k^2$. For $k=3$ and $C_{{\text{{out}}}}=256$, this is about $8{-}9\times$ fewer parameters. The savings are most significant when both $C_{{\text{{out}}}}$ and $k$ are large.

---

**Exercise 4.**
Extend `FeatureEncoder` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = FeatureEncoder(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
