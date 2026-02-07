# Prompting Techniques Overview

## Learning Objectives

- Understand the paradigm shift from fine-tuning to prompting
- Identify the anatomy of effective prompts
- Classify prompting strategies by complexity and use case

## The Prompting Paradigm

Traditional NLP required task-specific fine-tuning for each downstream task. LLMs introduced a fundamentally different paradigm: **prompting**—conditioning the model's behavior through natural language instructions without updating any parameters.

Formally, instead of learning task-specific parameters $\theta_{\text{task}}$, we construct an input $\text{prompt}(x)$ such that:

$$\hat{y} = \text{LLM}(\text{prompt}(x); \theta_{\text{pretrained}})$$

This eliminates the need for labeled training data and task-specific model copies.

## Prompt Anatomy

A well-structured prompt typically contains four components:

1. **System instruction**: Role definition and behavioral constraints
2. **Context**: Background information relevant to the task
3. **Examples** (optional): Input-output demonstrations
4. **Query**: The specific input to process

```
[System] You are a financial analyst specializing in equity research.
         Always cite specific metrics and provide confidence levels.

[Context] Company: NVIDIA (NVDA)
          Q3 FY2025 Revenue: $35.1B (+94% YoY)
          Data Center Revenue: $30.8B (+112% YoY)

[Query] Provide a brief assessment of NVIDIA's growth trajectory.
```

## Prompting Strategy Taxonomy

| Strategy | Key Idea | Best For |
|----------|---------|---------|
| Zero-shot | Direct instruction, no examples | Simple classification, extraction |
| Few-shot | Include input-output examples | Format-sensitive tasks, rare patterns |
| Chain-of-Thought | Elicit step-by-step reasoning | Math, logic, multi-step analysis |
| Tree-of-Thought | Explore multiple reasoning paths | Complex planning, optimization |
| Self-Consistency | Majority vote over multiple paths | Arithmetic, factual questions |
| ReAct | Interleave reasoning and actions | Tool use, information gathering |

## References

1. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS*.
2. Liu, P., et al. (2023). "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in NLP."
