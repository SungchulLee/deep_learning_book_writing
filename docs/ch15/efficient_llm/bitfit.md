# BitFit: Bias-Only Fine-Tuning

## Learning Objectives

- Understand bias-only fine-tuning and why it works
- Implement BitFit for LLM adaptation
- Compare BitFit with other PEFT methods

## Core Concept

**BitFit** (Zaken et al., 2022) fine-tunes **only the bias terms** of a pre-trained model, freezing all weight matrices:

$$\theta_{\text{trainable}} = \{b_i \mid b_i \text{ is a bias term in the model}\}$$

For a typical transformer, bias terms constitute less than **0.1%** of total parameters.

## Why Bias-Only Works

Bias terms control the **activation thresholds** of neurons. Adjusting biases effectively shifts the decision boundaries without altering the learned feature representations. This is sufficient for many downstream tasks because:

1. Pre-trained features are already highly expressive
2. Task adaptation often requires only recalibrating feature importance
3. Bias adjustments act as a form of output distribution shift

## Implementation

```python
def apply_bitfit(model):
    """Freeze all parameters except biases."""
    trainable_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'bias' in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    ratio = trainable_params / total_params * 100
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({ratio:.3f}%)")
    return model


# Example
# model = AutoModelForSequenceClassification.from_pretrained("llama-7b")
# model = apply_bitfit(model)
# Trainable: ~6.6M / 7B (0.09%)
```

## Results

On GLUE benchmark (with BERT-base):

| Method | Trainable Params | Avg Score |
|--------|-----------------|-----------|
| Full Fine-Tuning | 100% | 84.2 |
| BitFit | 0.08% | 82.8 |
| Random Subset (0.08%) | 0.08% | 73.1 |

BitFit achieves 98.3% of full fine-tuning performance with 1000x fewer trainable parameters.

## When to Use BitFit

- **Extremely low compute**: When even LoRA is too expensive
- **Quick prototyping**: Fast iteration on task-specific adaptations
- **Multi-task serving**: Minimal per-task storage overhead
- **Baseline**: As a lower bound for PEFT method comparison

## References

1. Zaken, E., et al. (2022). "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models." *ACL*.
