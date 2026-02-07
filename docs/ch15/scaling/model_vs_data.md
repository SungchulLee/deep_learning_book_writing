# Model Size vs. Data Trade-offs

## Learning Objectives

- Analyze the trade-off between model size and training data volume
- Identify three scaling regimes and their practical implications
- Understand data repetition effects
- Apply scaling insights to deployment decisions

## Three Scaling Regimes

### 1. Under-Trained Regime ($D/N < 10$)

The model has more capacity than the data can support. Symptoms: overfitting, poor generalization, wasted parameters.

Examples: GPT-3 (175B params, 300B tokens, $D/N = 1.7$), Gopher (280B params, 300B tokens, $D/N = 1.1$)

### 2. Chinchilla-Optimal Regime ($D/N \approx 20$)

Minimizes training loss for fixed compute. Model and data are balanced.

Examples: Chinchilla (70B params, 1.4T tokens), LLaMA-65B (65B params, 1.4T tokens)

### 3. Over-Trained Regime ($D/N > 50$)

Not compute-optimal for training loss but can be optimal for deployment:

$$\text{Total Cost} = C_{\text{train}}(N, D) + K \cdot C_{\text{inference}}(N)$$

Examples: LLaMA-7B ($D/N = 143$), Mistral-7B ($D/N \approx 285$)

## Practical Implications

| Regime | When to Use | Finance Application |
|--------|------------|-------------------|
| Under-trained | Never (unless compute-constrained) | - |
| Chinchilla-optimal | Research, one-off analysis | Large-scale research models |
| Over-trained | Production deployment | Real-time trading signals, API services |

## Data Repetition Effects

Muennighoff et al. (2023) found:

1. **Diminishing returns**: Each additional epoch provides less benefit
2. **Quality sensitivity**: Higher-quality data tolerates more repetition
3. **Practical guideline**: Degradation noticeable after ~4 epochs on web data

## Scaling Decisions for Finance

```python
def recommend_model_size(
    queries_per_day: int,
    latency_requirement_ms: float,
    accuracy_priority: str,
) -> dict:
    if accuracy_priority == "high" and latency_requirement_ms > 5000:
        return {"size": "70B+", "regime": "chinchilla-optimal",
                "note": "Use for deep analysis, report generation"}
    elif queries_per_day > 100_000 or latency_requirement_ms < 500:
        return {"size": "7-13B", "regime": "over-trained",
                "note": "Over-train on domain data; quantize for deployment"}
    else:
        return {"size": "13-34B", "regime": "slightly over-trained",
                "note": "Good balance for most finance workloads"}
```

## Key Takeaways

1. **Chinchilla-optimal is not deployment-optimal**: Over-training smaller models is more cost-effective for serving
2. **Data quality matters more than scaling laws suggest**: Curated financial text shifts the optimal allocation
3. **The frontier moves toward smaller, better-trained models**: Mistral-7B and Phi demonstrate architecture + data quality compensating for fewer parameters
4. **For finance**: High query volume and low latency strongly favor the over-trained regime

## References

1. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models."
2. Muennighoff, N., et al. (2023). "Scaling Data-Constrained Language Models."
3. Sardana, N., et al. (2023). "Beyond Chinchilla-Optimal: Accounting for Inference."
