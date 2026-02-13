# Autoregressive Image Generation

## Overview

Autoregressive image generation extends the pixel-by-pixel approach of PixelCNN to modern transformer architectures, achieving state-of-the-art results through scaling and tokenization.

## Token-Based Approaches

Instead of predicting raw pixels, modern methods first tokenize images into discrete tokens using a learned codebook:

### VQGAN + Transformer

1. **Encode**: VQGAN encoder maps image patches to discrete tokens
2. **Model**: autoregressive transformer predicts token sequence
3. **Decode**: VQGAN decoder maps tokens back to pixels

```python
# Conceptual pipeline
image_tokens = vqgan.encode(image)  # (B, H*W) discrete tokens
logits = transformer(image_tokens[:, :-1])  # predict next token
loss = F.cross_entropy(logits.view(-1, codebook_size), 
                       image_tokens[:, 1:].view(-1))
```

### DALL-E (v1)

Uses a dVAE (discrete VAE) to tokenize images into a 32×32 grid of 8192 possible tokens, then trains a 12B parameter transformer to model the joint distribution of text tokens and image tokens.

## Masked Image Modeling as AR

Models like MaskGIT use a non-autoregressive approach at test time but can be viewed through the AR lens:

1. Start with all tokens masked
2. Predict all tokens simultaneously
3. Keep the most confident predictions, re-mask the rest
4. Repeat until all tokens are generated

This parallel decoding is much faster than sequential AR while maintaining competitive quality.

## Comparison with Diffusion Models

| Aspect | Autoregressive | Diffusion |
|--------|---------------|-----------|
| Sampling steps | $N$ (tokens) | $T$ (noise levels) |
| Exact likelihood | Yes | Lower bound |
| Text conditioning | Natural (shared sequence) | Cross-attention |
| Resolution scaling | Quadratic in tokens | Linear in pixels |
| Current SOTA | Competitive | Leading |
