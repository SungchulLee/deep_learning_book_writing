# Few-Shot Prompting

## Learning Objectives

- Formalize few-shot (in-context) learning
- Implement effective few-shot prompts for financial tasks
- Understand key findings about example selection and ordering

## Formulation

Few-shot prompting includes $k$ input-output demonstrations before the query:

$$\hat{y} = \text{LLM}\left((x_1, y_1), (x_2, y_2), \ldots, (x_k, y_k), x_{\text{query}}\right)$$

The model performs **in-context learning**: adapting its behavior based on the demonstrations without gradient updates.

## Implementation

```python
def build_few_shot_prompt(examples, query, instruction=""):
    prompt_parts = []
    if instruction:
        prompt_parts.append(instruction + "\n")

    for inp, out in examples:
        prompt_parts.append(f"Input: {inp}")
        prompt_parts.append(f"Output: {out}\n")

    prompt_parts.append(f"Input: {query}")
    prompt_parts.append("Output:")
    return "\n".join(prompt_parts)


# Financial sentiment classification
examples = [
    ("Revenue beat estimates by 15%, raising full-year guidance", "BULLISH"),
    ("Company announced 20% workforce reduction amid declining sales", "BEARISH"),
    ("Q3 results in line with expectations, maintaining guidance", "NEUTRAL"),
]

prompt = build_few_shot_prompt(
    examples=examples,
    query="Free cash flow surged 45% YoY driven by margin expansion",
    instruction="Classify the financial statement as BULLISH, BEARISH, or NEUTRAL.",
)
```

## Key Findings

### Example Order Matters

Lu et al. (2022) showed that example order significantly affects performance. Recency bias means the model weighs later examples more heavily.

### Optimal Number of Examples

| Model Size | Optimal k | Notes |
|-----------|----------|-------|
| 7B | 3-5 | Limited context window |
| 13B | 5-8 | Good balance |
| 70B+ | 1-3 | Often sufficient |

### Example Selection Strategies

1. **Random**: Simple baseline, often surprisingly effective
2. **Similarity-based**: Select examples most similar to the query via embeddings
3. **Diverse**: Ensure coverage of different patterns and edge cases

## References

1. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS*.
2. Lu, Y., et al. (2022). "Fantastically Ordered Prompts and Where to Find Them." *ACL*.
