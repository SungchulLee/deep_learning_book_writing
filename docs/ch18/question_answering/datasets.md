# QA Datasets

## Overview

| Dataset | Task | Size | Answer Type |
|---------|------|------|-------------|
| SQuAD 1.1 | Extractive | 100K | Span |
| SQuAD 2.0 | Extractive + unanswerable | 150K | Span / No answer |
| Natural Questions | Open-domain | 307K | Short/long answer |
| TriviaQA | Open-domain | 650K | Span |
| HotpotQA | Multi-hop | 113K | Span + supporting facts |
| MS MARCO | Open-domain | 1M | Natural language |
| CoQA | Conversational | 127K | Free-form |
| QuAC | Conversational | 100K | Span |

## Financial QA

| Dataset | Domain | Description |
|---------|--------|-------------|
| FinQA | Financial reports | Numerical reasoning over tables |
| ConvFinQA | Financial | Conversational financial QA |
| TAT-QA | Table + text | Hybrid table-text QA |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Exact Match (EM) | Prediction exactly matches gold answer |
| F1 | Token-level overlap between prediction and gold |
| ROUGE-L | Longest common subsequence for abstractive answers |

## Summary

1. SQuAD established the standard for extractive QA evaluation
2. Financial QA datasets require numerical reasoning and table understanding
3. EM and F1 are standard metrics for extractive QA; ROUGE-L for abstractive
