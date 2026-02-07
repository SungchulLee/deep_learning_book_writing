# Transformer Summarization

## Overview

Pretrained Transformer models dominate modern summarization. BART, T5, and Pegasus achieve state-of-the-art results.

## Key Models

### BART (Lewis et al., 2020)

Denoising autoencoder pretrained with sentence shuffling and text infilling:

```python
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

inputs = tokenizer(article, max_length=1024, return_tensors="pt", truncation=True)
summary_ids = model.generate(
    inputs["input_ids"], num_beams=4,
    max_length=150, min_length=40, length_penalty=2.0,
)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

### T5

Treats summarization as text-to-text: `"summarize: {document}"` → summary

### Pegasus (Zhang et al., 2020)

Pre-trained with Gap Sentence Generation (GSG): mask important sentences and predict them. Specifically designed for summarization.

## Benchmark Results

| Model | CNN/DailyMail ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|----------------------|---------|---------|
| BART-large | 44.16 | 21.28 | 40.90 |
| T5-large | 43.52 | 21.55 | 40.69 |
| Pegasus | 44.17 | 21.47 | 41.11 |

## Financial Summarization

Summarizing earnings calls, SEC filings, and analyst reports:

- Long document challenge (10-K filings: 50-100 pages)
- Domain vocabulary (financial terminology)
- Numerical accuracy is critical (revenue figures, growth rates)
- FinBERT + summarization models for financial text

## Summary

1. BART and Pegasus are the leading models for summarization
2. Pre-training objectives aligned with summarization improve performance
3. Financial summarization requires handling long documents and preserving numerical accuracy
