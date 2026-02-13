# Few-Shot NER

## Overview

Few-shot NER aims to recognize entities in new domains or for new entity types with minimal labeled examples (typically 5-50 per type).

## Approaches

### Prototype Networks for NER

Learn a metric space where entities cluster by type:

$$P(y | x) = \frac{\exp(-d(f(x), \mathbf{c}_y))}{\sum_{y'} \exp(-d(f(x), \mathbf{c}_{y'}))}$$

where $\mathbf{c}_y$ is the prototype (mean representation) of support examples for type $y$.

### Prompt-Based NER

Frame NER as template filling with language models:

```
Input: "Goldman Sachs reported earnings."
Prompt: "Goldman Sachs is a [MASK] entity."
→ Model predicts: "organization"
```

### In-Context Learning

Provide examples in the prompt for LLM-based few-shot NER:

```
Extract entities from the text.
Example: "Apple launched iPhone" → Apple: ORG, iPhone: PRODUCT
Example: "Biden visited Paris" → Biden: PER, Paris: LOC
Text: "Tesla acquired SolarCity" → ?
```

## Benchmarks

| Method | Few-NERD 5-shot F1 | Few-NERD 10-shot F1 |
|--------|-------------------|---------------------|
| ProtoBERT | 40.1 | 47.3 |
| NNShot | 25.7 | 31.2 |
| CONTaiNER | 55.3 | 61.8 |
| GPT-4 (in-context) | ~50 | ~55 |

## Summary

1. Few-shot NER is essential for rapid deployment in new domains
2. Prototype networks provide a principled metric learning approach
3. Prompt-based methods leverage pretrained LM knowledge
4. In-context learning with LLMs offers annotation-free adaptation
