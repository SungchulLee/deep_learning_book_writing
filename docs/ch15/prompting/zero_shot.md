# Zero-Shot Prompting

## Learning Objectives

- Formalize zero-shot prompting and its theoretical basis
- Apply zero-shot prompting to financial NLP tasks
- Understand when zero-shot is sufficient vs. when few-shot is needed

## Formulation

In zero-shot prompting, the model receives only a task description and input—no demonstrations:

$$\hat{y} = \text{LLM}(\text{instruction} \oplus x)$$

The model must rely entirely on knowledge acquired during pretraining.

## Examples

### Classification

```
Classify the following headline as BULLISH, BEARISH, or NEUTRAL.

Headline: "Fed signals potential rate cuts in Q2 amid cooling inflation data"

Classification:
```

### Extraction

```
Extract all company ticker symbols mentioned in the following text.
Return as a JSON array.

Text: "Analysts upgraded MSFT and GOOGL while downgrading META
after reviewing Q3 earnings reports."

Tickers:
```

### Summarization

```
Summarize the following earnings call excerpt in 2-3 sentences,
focusing on forward guidance.

[Earnings call text...]

Summary:
```

## When Zero-Shot Works Well

- **Common tasks**: Sentiment analysis, summarization, translation
- **Well-defined outputs**: Classification with clear categories
- **Large models**: GPT-4 class models handle most tasks zero-shot
- **Standard formats**: Tasks that resemble common internet text patterns

## When Zero-Shot Falls Short

- **Unusual output formats**: Custom JSON schemas, specific table formats
- **Domain-specific conventions**: Financial statement formatting, legal citations
- **Ambiguous instructions**: Tasks where expected behavior isn't obvious
- **Small models**: Models <13B parameters often need examples

## References

1. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS*.
2. Kojima, T., et al. (2022). "Large Language Models are Zero-Shot Reasoners." *NeurIPS*.
