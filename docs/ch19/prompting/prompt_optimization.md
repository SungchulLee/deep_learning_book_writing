# Prompt Optimization

## Learning Objectives

- Understand automated prompt optimization methods
- Compare discrete vs. continuous prompt search
- Apply optimization techniques to improve financial prompts

## Why Optimize Prompts?

Manual prompt engineering is time-consuming, subjective, and suboptimal. Automated methods can systematically search for better prompts.

## Automated Prompt Engineering (APE)

Zhou et al. (2023) proposed using LLMs to generate and evaluate prompts:

1. **Generate**: Use an LLM to propose candidate instructions from examples
2. **Evaluate**: Score each candidate on a validation set
3. **Select**: Choose the highest-scoring prompt

```python
def automatic_prompt_engineering(llm, task_examples, n_candidates=20):
    meta_prompt = (
        "I have a task with these input-output examples:\n"
        + format_examples(task_examples[:5])
        + "\nGenerate a clear instruction that would produce these outputs."
    )

    candidates = [llm(meta_prompt, temperature=0.9) for _ in range(n_candidates)]

    # Evaluate each candidate
    scores = []
    for prompt in candidates:
        correct = sum(
            1 for inp, expected in task_examples[5:]
            if llm(f"{prompt}\n\nInput: {inp}\nOutput:").strip() == expected
        )
        scores.append(correct)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return candidates[best_idx], scores[best_idx]
```

## OPRO: Optimization by PROmpting

Yang et al. (2023) use LLMs as optimizers: present the LLM with previous prompts and their scores, ask it to propose a better one.

```
Previous prompts and scores:
- "Classify the sentiment" -> accuracy: 72%
- "Determine if the text is positive or negative" -> accuracy: 78%
- "Rate the sentiment as positive, negative, or neutral" -> accuracy: 81%

Generate a new instruction that would score higher than all previous ones:
```

## Comparison of Methods

| Method | Search Space | Requires Training | Cost |
|--------|-------------|------------------|------|
| Manual engineering | Discrete (text) | No | Human time |
| APE | Discrete (text) | No | LLM calls |
| OPRO | Discrete (text) | No | LLM calls |
| Prompt tuning | Continuous (embeddings) | Yes | GPU compute |
| Prefix tuning | Continuous (embeddings) | Yes | GPU compute |

Discrete methods (APE, OPRO) are model-agnostic and work with API-only access. Continuous methods require model weight access but can be more effective.

## References

1. Zhou, Y., et al. (2023). "Large Language Models Are Human-Level Prompt Engineers." *ICLR*.
2. Yang, C., et al. (2023). "Large Language Models as Optimizers." *arXiv*.
