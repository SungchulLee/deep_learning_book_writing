# Efficient LLM Methods Overview

## Learning Objectives

- Understand why parameter-efficient fine-tuning (PEFT) is necessary
- Classify PEFT methods by approach
- Select the appropriate method for different scenarios

## Why PEFT?

Full fine-tuning of LLMs is prohibitively expensive:

| Model | Parameters | Full FT Memory (fp16) | Optimizer States | Total |
|-------|-----------|----------------------|-----------------|-------|
| LLaMA-7B | 7B | 14 GB | 42 GB | 56 GB |
| LLaMA-13B | 13B | 26 GB | 78 GB | 104 GB |
| LLaMA-70B | 70B | 140 GB | 420 GB | 560 GB |

PEFT methods update only a small fraction of parameters, reducing memory and compute by 10-1000x.

## PEFT Taxonomy

### Additive Methods

Add new trainable parameters while freezing the original model:

- **Adapters**: Small bottleneck layers inserted between transformer layers
- **Prefix Tuning**: Learnable prefix tokens prepended to keys and values
- **Prompt Tuning**: Learnable soft prompt embeddings

### Reparameterization Methods

Reparameterize weight updates as low-rank:

- **LoRA**: Low-rank decomposition of weight updates
- **QLoRA**: LoRA with quantized base model

### Selective Methods

Update only specific existing parameters:

- **BitFit**: Train only bias terms
- **Layer freezing**: Train only last N layers

## Quick Comparison

| Method | Trainable Params | Memory | Quality | Speed |
|--------|-----------------|--------|---------|-------|
| Full FT | 100% | Very high | Best | Slow |
| LoRA | 0.1-1% | Low | Near-full FT | Fast |
| QLoRA | 0.1-1% | Very low | Near-LoRA | Fast |
| Adapters | 1-5% | Moderate | Good | Moderate |
| Prefix Tuning | <1% | Low | Good | Fast |
| Prompt Tuning | <0.01% | Minimal | Moderate | Fastest |
| BitFit | <0.1% | Low | Moderate | Fast |

## Decision Guide

- **Limited GPU memory**: QLoRA
- **Best quality**: LoRA (rank 16-64) or full fine-tuning
- **Multi-task deployment**: Adapters or LoRA (swap modules per task)
- **Minimal changes**: BitFit or prompt tuning
- **API-only access**: Prompt tuning (no weight access needed)

## References

1. Hu, E., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*.
2. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS*.
3. Houlsby, N., et al. (2019). "Parameter-Efficient Transfer Learning for NLP." *ICML*.
