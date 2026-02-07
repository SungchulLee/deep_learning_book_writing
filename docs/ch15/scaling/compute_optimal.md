# Compute-Optimal Training

## Learning Objectives

- Formalize the compute budget and its relationship to model size and data
- Derive the compute-optimal allocation between parameters and tokens
- Apply compute-optimal principles to practical training decisions

## The Compute Budget

Training an LLM requires a compute budget $C$ measured in FLOPs. For a transformer with $N$ parameters trained on $D$ tokens:

$$C \approx 6ND$$

This accounts for both the forward pass (~$2ND$ FLOPs) and the backward pass (~$4ND$ FLOPs).

## Kaplan Scaling Laws (2020)

Kaplan et al. (OpenAI) proposed that loss scales as a power law:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}$$

Key finding: **for a fixed compute budget, it is more efficient to scale model size than data**. This led to the "large model, less data" paradigm (e.g., GPT-3 at 175B parameters on only 300B tokens).

Kaplan allocation: $N_{\text{opt}} \propto C^{0.73}$, $D_{\text{opt}} \propto C^{0.27}$.

## Chinchilla Scaling Laws (2022)

Hoffmann et al. (DeepMind) challenged Kaplan, showing **model size and data should scale equally**:

$$N_{\text{opt}} \propto C^{0.50}, \quad D_{\text{opt}} \propto C^{0.50}$$

Practical rule: **training tokens should be approximately 20x the number of parameters**.

| Model | Params (N) | Tokens (D) | D/N Ratio | Status |
|-------|-----------|-----------|-----------|--------|
| GPT-3 | 175B | 300B | 1.7x | Under-trained |
| Gopher | 280B | 300B | 1.1x | Severely under-trained |
| Chinchilla | 70B | 1.4T | 20x | Compute-optimal |
| LLaMA | 65B | 1.4T | 21.5x | Near-optimal |

## Practical Compute Planning

```python
import math


def compute_optimal_allocation(compute_budget_flops: float) -> dict:
    # Chinchilla optimal: C = 6 * N * D = 6 * N * 20N = 120 * N^2
    n_opt = math.sqrt(compute_budget_flops / 120)
    d_opt = 20 * n_opt

    return {
        "optimal_params": n_opt,
        "optimal_tokens": d_opt,
        "params_billions": n_opt / 1e9,
        "tokens_trillions": d_opt / 1e12,
        "tokens_per_param": d_opt / n_opt,
    }


# Example: 1000 A100s for 30 days
# 1000 * 312 TFLOPS * 0.4 MFU * 30 days * 86400 sec/day
budget = 1000 * 312e12 * 0.4 * 30 * 86400
result = compute_optimal_allocation(budget)
print(f"Optimal: {result['params_billions']:.1f}B params, "
      f"{result['tokens_trillions']:.2f}T tokens")
```

## Inference-Aware Scaling

Chinchilla-optimal training ignores **inference cost**. For deployment:

$$\text{Total Cost} = C_{\text{train}} + n_{\text{queries}} \cdot C_{\text{inference}}(N)$$

LLaMA explicitly "over-trained" smaller models (up to 140x tokens for the 7B model) because smaller models are cheaper to serve.

## References

1. Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models." *arXiv*.
2. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." *NeurIPS*.
3. Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models."
