# Chinchilla Scaling Laws

## Learning Objectives

- Understand the Chinchilla loss parametrization
- Compare Chinchilla predictions against empirical results
- Implement scaling law prediction for planning purposes

## The Chinchilla Loss Function

Hoffmann et al. (2022) proposed:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

where:

- $E$: Irreducible loss (entropy of natural language, $\approx 1.69$ nats)
- $A/N^\alpha$: Approximation error (reduced by increasing model size)
- $B/D^\beta$: Estimation error (reduced by increasing data)

### Fitted Parameters

| Parameter | Value | Interpretation |
|-----------|-------|---------------|
| $E$ | 1.69 | Irreducible entropy of text |
| $A$ | 406.4 | Model size scaling coefficient |
| $\alpha$ | 0.34 | Model size scaling exponent |
| $B$ | 410.7 | Data scaling coefficient |
| $\beta$ | 0.28 | Data scaling exponent |

## Deriving Optimal Allocation

For fixed compute $C = 6ND$, minimize $L(N, D)$ via Lagrange multipliers:

$$\frac{\partial L}{\partial N} = \lambda \cdot 6D, \quad \frac{\partial L}{\partial D} = \lambda \cdot 6N$$

This yields:

$$N_{\text{opt}} = G \cdot \left(\frac{C}{6}\right)^{a}, \quad D_{\text{opt}} = G^{-1} \cdot \left(\frac{C}{6}\right)^{b}$$

where $a = \frac{\beta}{\alpha + \beta} \approx 0.46$ and $b = \frac{\alpha}{\alpha + \beta} \approx 0.54$.

## Chinchilla vs. Gopher

| Property | Gopher | Chinchilla |
|----------|--------|-----------|
| Parameters | 280B | 70B |
| Training tokens | 300B | 1.4T |
| Training FLOPs | ~$5 \times 10^{23}$ | ~$5 \times 10^{23}$ |
| D/N ratio | 1.07x | 20x |
| MMLU (5-shot) | 60.0% | 67.6% |
| HellaSwag | 79.2% | 80.8% |
| Inference cost | 4x higher | 1x (baseline) |

## Implementation

```python
import numpy as np


class ChinchillaScalingLaw:
    def __init__(self, E=1.69, A=406.4, alpha=0.34, B=410.7, beta=0.28):
        self.E = E
        self.A = A
        self.alpha = alpha
        self.B = B
        self.beta = beta

    def predict_loss(self, N: float, D: float) -> float:
        return self.E + self.A / (N ** self.alpha) + self.B / (D ** self.beta)

    def optimal_allocation(self, C: float) -> dict:
        a = self.beta / (self.alpha + self.beta)
        b = self.alpha / (self.alpha + self.beta)
        G = (self.alpha * self.A / (self.beta * self.B)) ** (1 / (self.alpha + self.beta))

        N_opt = G * (C / 6) ** a
        D_opt = (C / 6) / N_opt

        return {
            "N_opt": N_opt,
            "D_opt": D_opt,
            "predicted_loss": self.predict_loss(N_opt, D_opt),
            "tokens_per_param": D_opt / N_opt,
        }


law = ChinchillaScalingLaw()
budget = 6 * 70e9 * 1.4e12
result = law.optimal_allocation(budget)
print(f"Optimal: {result['N_opt']/1e9:.1f}B params, {result['D_opt']/1e12:.2f}T tokens")
print(f"Predicted loss: {result['predicted_loss']:.3f}")
```

## Limitations

1. **Architecture dependence**: Derived for dense transformers; MoE models may differ
2. **Data quality not modeled**: Assumes homogeneous data quality
3. **Task-specific scaling**: Downstream tasks may scale differently than LM loss
4. **Post-training effects**: RLHF and alignment shift the effective frontier

## References

1. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." *NeurIPS*.
2. Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models."
