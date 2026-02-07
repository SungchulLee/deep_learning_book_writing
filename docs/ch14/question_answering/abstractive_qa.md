# Abstractive QA

## Overview

Abstractive QA generates answers in natural language rather than extracting spans, enabling answers that synthesize, paraphrase, or reason beyond the source text.

## Architecture

Sequence-to-sequence models with encoder-decoder structure:

$$P(\mathbf{a} | q, c) = \prod_{t=1}^{T} P(a_t | a_{<t}, q, c)$$

## Approaches

### Generative Models

Fine-tune T5, BART, or GPT for answer generation:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

input_text = f"question: {question} context: {context}"
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=100)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Retrieval-Augmented Generation (RAG)

Combine retrieval with generation:
1. Retrieve relevant passages using dense retrieval
2. Condition generator on retrieved passages
3. Generate answer from combined evidence

## Datasets

| Dataset | Description |
|---------|-------------|
| NarrativeQA | Answers from book/movie summaries |
| ELI5 | Long-form explanatory answers |
| MS MARCO | Web-sourced natural language answers |

## Summary

1. Abstractive QA generates fluent answers beyond span extraction
2. T5 and BART are effective base models for generative QA
3. RAG combines retrieval precision with generation flexibility
