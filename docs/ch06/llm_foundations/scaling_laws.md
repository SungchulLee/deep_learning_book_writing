# Scaling Laws for Large Language Models

## Learning Objectives

- Understand the empirical scaling laws governing LLM performance
- Derive compute-optimal training configurations
- Apply Chinchilla scaling to determine model and data sizes
- Analyze the trade-offs between parameters, data, and compute

## Introduction

Scaling laws describe the predictable relationship between model performance and three key factors: model size (parameters), dataset size (tokens), and compute budget (FLOPs). These empirical laws enable researchers to predict performance before training and optimize resource allocation.

## The Kaplan Scaling Laws (OpenAI, 2020)

### Power Law Relationships

Performance (measured as cross-entropy loss $L$) follows power laws:

**Model Size Scaling**:
$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

**Data Size Scaling**:
$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

**Compute Scaling**:
$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

Where $N_c$, $D_c$, $C_c$ are critical constants and $N$, $D$, $C$ represent parameters, data tokens, and compute FLOPs respectively.

### Combined Scaling Law

When scaling all factors simultaneously:

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D}\right]^{\alpha_D}$$

### Key Findings

1. **Smooth power laws**: Loss decreases predictably with scale
2. **Model size dominates**: Larger models are more sample-efficient
3. **Compute budget**: For fixed compute, prefer larger models trained on less data

## Chinchilla Scaling Laws (DeepMind, 2022)

### Revised Optimal Allocation

Hoffmann et al. found the original scaling laws underestimated data requirements:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

With fitted parameters:
- $E \approx 1.69$ (irreducible loss)
- $A \approx 406.4$, $\alpha \approx 0.34$
- $B \approx 410.7$, $\beta \approx 0.28$

### Compute-Optimal Training

For a compute budget $C \approx 6ND$ (forward + backward passes):

**Optimal parameters**:
$$N_{opt} \propto C^{0.50}$$

**Optimal tokens**:
$$D_{opt} \propto C^{0.50}$$

**Key insight**: Parameters and data should scale equally with compute.

### Chinchilla vs Gopher

| Model | Parameters | Training Tokens | Compute |
|-------|------------|-----------------|---------|
| Gopher | 280B | 300B | ~$5 \times 10^{23}$ |
| Chinchilla | 70B | 1.4T | ~$5 \times 10^{23}$ |

Chinchilla (4× smaller, 4× more data) outperformed Gopher on most benchmarks.

## Practical Application

### Estimating Required Compute

```python
def estimate_training_flops(
    num_parameters: int,
    num_tokens: int,
    flops_per_token_per_param: float = 6.0
) -> float:
    """
    Estimate total training FLOPs.
    
    The factor of 6 accounts for:
    - Forward pass: 2 FLOPs per parameter per token
    - Backward pass: 4 FLOPs per parameter per token
    
    Args:
        num_parameters: Model parameter count
        num_tokens: Training tokens
        flops_per_token_per_param: FLOPs multiplier (default 6)
    
    Returns:
        Total FLOPs for training
    """
    return flops_per_token_per_param * num_parameters * num_tokens


def chinchilla_optimal_config(
    compute_budget_flops: float,
    flops_per_token_per_param: float = 6.0
) -> dict:
    """
    Calculate compute-optimal model and data size.
    
    Based on Chinchilla scaling: N ≈ D for optimal allocation.
    
    Args:
        compute_budget_flops: Available compute in FLOPs
        
    Returns:
        Dictionary with optimal parameters and tokens
    """
    # C = 6 * N * D, and N ≈ D for optimal
    # So C = 6 * N^2, thus N = sqrt(C/6)
    
    optimal_n = (compute_budget_flops / flops_per_token_per_param) ** 0.5
    optimal_d = optimal_n  # Equal scaling
    
    return {
        'optimal_parameters': int(optimal_n),
        'optimal_tokens': int(optimal_d),
        'tokens_per_parameter': optimal_d / optimal_n
    }


# Example: 10^24 FLOPs budget
config = chinchilla_optimal_config(1e24)
print(f"Optimal parameters: {config['optimal_parameters'] / 1e9:.1f}B")
print(f"Optimal tokens: {config['optimal_tokens'] / 1e12:.2f}T")
```

### Predicting Loss

```python
import numpy as np

def predict_loss_chinchilla(
    num_parameters: float,
    num_tokens: float,
    E: float = 1.69,
    A: float = 406.4,
    alpha: float = 0.34,
    B: float = 410.7,
    beta: float = 0.28
) -> float:
    """
    Predict training loss using Chinchilla scaling law.
    
    L(N, D) = E + A/N^α + B/D^β
    """
    return E + A / (num_parameters ** alpha) + B / (num_tokens ** beta)


# Compare different configurations
configs = [
    ("7B, 1T tokens", 7e9, 1e12),
    ("13B, 1T tokens", 13e9, 1e12),
    ("7B, 2T tokens", 7e9, 2e12),
    ("70B, 1.4T tokens", 70e9, 1.4e12),
]

for name, n, d in configs:
    loss = predict_loss_chinchilla(n, d)
    print(f"{name}: predicted loss = {loss:.3f}")
```

## Beyond Chinchilla: Recent Findings

### LLaMA Scaling Philosophy

Meta's LLaMA models prioritize **inference efficiency**:
- Train smaller models on more data than Chinchilla-optimal
- 7B model trained on 1T+ tokens (vs ~200B Chinchilla-optimal)
- Better inference cost for deployed models

### Emergent Scaling Behaviors

Some capabilities show **discontinuous** improvements:

```
Performance
    │
    │                    ╭──── Emergent capability
    │                   ╱
    │          ────────╯
    │    ─────╯
    │───╯
    └────────────────────────── Scale (log)
```

These include:
- Chain-of-thought reasoning
- In-context learning
- Code generation

### Data Quality Scaling

Recent work suggests data quality scales differently:

$$L(N, D, Q) = E + \frac{A}{N^\alpha} + \frac{B}{(D \cdot Q)^\beta}$$

Where $Q$ represents data quality (filtering, deduplication).

## Scaling Law Limitations

### What Scaling Laws Don't Capture

1. **Capability thresholds**: Some abilities emerge suddenly
2. **Task-specific performance**: Different tasks scale differently  
3. **Architecture effects**: Laws derived for transformers specifically
4. **Data distribution**: Quality and diversity matter beyond quantity
5. **Fine-tuning dynamics**: Laws focus on pretraining

### Extrapolation Risks

```python
def scaling_uncertainty(
    predicted_loss: float,
    extrapolation_factor: float,
    uncertainty_per_order: float = 0.05
) -> tuple:
    """
    Estimate uncertainty when extrapolating scaling laws.
    
    Uncertainty grows with extrapolation distance.
    """
    log_extrapolation = np.log10(extrapolation_factor)
    relative_uncertainty = uncertainty_per_order * log_extrapolation
    
    lower = predicted_loss * (1 - relative_uncertainty)
    upper = predicted_loss * (1 + relative_uncertainty)
    
    return lower, upper
```

## Summary

| Scaling Law | Key Insight | Optimal Ratio (N:D) |
|-------------|-------------|---------------------|
| Kaplan (2020) | Larger models more efficient | ~10:1 (favor parameters) |
| Chinchilla (2022) | Data equally important | ~1:1 (balanced) |
| LLaMA (2023) | Inference cost matters | ~1:20+ (favor data) |

## Key Equations

$$\boxed{L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}}$$

$$\boxed{C_{opt} = 6 \cdot N_{opt} \cdot D_{opt}, \quad N_{opt} \approx D_{opt}}$$

## References

1. Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. *arXiv:2001.08361*.
2. Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. *arXiv:2203.15556*.
3. Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv:2302.13971*.
