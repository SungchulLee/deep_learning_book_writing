# Chain-of-Thought Prompting

## Learning Objectives

- Understand chain-of-thought (CoT) reasoning and when it helps
- Implement zero-shot and few-shot CoT prompting
- Analyze why CoT improves performance on reasoning tasks

## Core Idea

**Chain-of-Thought (CoT) prompting** (Wei et al., 2022) elicits intermediate reasoning steps before the final answer:

$$x \xrightarrow{\text{standard}} y \quad \text{vs.} \quad x \xrightarrow{\text{CoT}} r_1, r_2, \ldots, r_n, y$$

## Zero-Shot CoT

Kojima et al. (2022) discovered that appending **"Let's think step by step"** dramatically improves reasoning:

```
Q: A company's stock price was $150. It dropped 20% on Monday,
   then rose 25% on Tuesday. What is the final price?

A: Let's think step by step.
   Step 1: Starting price = $150
   Step 2: After 20% drop: $150 × 0.80 = $120
   Step 3: After 25% rise: $120 × 1.25 = $150
   Final price: $150
```

## Few-Shot CoT

Provide demonstrations with explicit reasoning chains:

```python
cot_examples = [
    {
        "question": "A bond has face value $1000, coupon 5%, trades at $950. "
                    "What is the current yield?",
        "reasoning": "Annual coupon = $1000 × 5% = $50. "
                     "Current yield = $50 / $950 = 5.26%.",
        "answer": "5.26%"
    },
]
```

## When CoT Helps

CoT provides the largest improvements for:

1. **Multi-step arithmetic**: Sequential calculations
2. **Logical reasoning**: Syllogisms, deductive chains
3. **Word problems**: Natural language to math translation

CoT provides **minimal benefit** for simple classification or pattern matching.

## Key Results

| Benchmark | Standard | CoT | Improvement |
|-----------|---------|-----|-------------|
| GSM8K (math) | 56.5% | 74.4% | +17.9pp |
| SVAMP (math) | 68.9% | 79.0% | +10.1pp |
| StrategyQA | 65.4% | 73.2% | +7.8pp |

*PaLM 540B results (Wei et al., 2022)*

## Mathematical Analysis

For a multi-step problem requiring $n$ operations with per-step accuracy $p$:

- **Direct prediction**: Accuracy $\approx p^n$
- **CoT**: Accuracy $\approx p$ per step with self-correction

For $p = 0.95$, $n = 5$: direct $\approx 0.77$, CoT $\approx 0.95$ per step.

## References

1. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in LLMs." *NeurIPS*.
2. Kojima, T., et al. (2022). "Large Language Models are Zero-Shot Reasoners." *NeurIPS*.
