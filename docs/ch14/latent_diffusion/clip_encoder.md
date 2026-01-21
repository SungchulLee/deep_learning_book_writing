# CLIP Text Encoder

## Introduction

**CLIP** (Contrastive Language-Image Pre-training) provides text understanding for text-to-image generation.

## Role in Stable Diffusion

1. **Tokenize** text prompt
2. **Embed** tokens to vectors
3. **Transform** via transformer layers
4. **Output** sequence of embeddings for cross-attention

## Architecture

- Model: ViT-L/14 text encoder
- Vocabulary: 49,408 tokens (BPE)
- Context length: 77 tokens
- Embedding dimension: 768

## Usage

```python
# Tokenization
tokens = tokenizer(
    prompt,
    max_length=77,
    padding="max_length",
    truncation=True
)

# Encoding
text_embeddings = text_encoder(tokens)[0]  # [batch, 77, 768]
```

## Prompt Engineering

CLIP's training affects how prompts work:
- Trained on (image, caption) pairs from web
- Responds well to descriptive captions
- Quality modifiers: "highly detailed", "4k", "trending on artstation"

## Summary

CLIP bridges natural language to visual semantics, enabling text-to-image generation.

## Navigation

- **Previous**: [U-Net Denoiser](unet_denoiser.md)
- **Next**: [Diffusion vs Flows](../comparisons/diffusion_vs_flows.md)
