# Text Summarization Overview

## Definition

Text summarization automatically produces a condensed version of a document that captures the most important information.

## Taxonomy

| Type | Method | Output |
|------|--------|--------|
| Extractive | Select sentences | Subset of original sentences |
| Abstractive | Generate new text | Novel sentences capturing key points |
| Hybrid | Extract then rewrite | Refined selected content |

## Formal Definition

### Extractive

Given document $D = \{s_1, \ldots, s_n\}$, select subset $S \subseteq D$ with $|S| \leq k$:

$$S^* = \arg\max_{S \subseteq D, |S| \leq k} \text{Quality}(S)$$

### Abstractive

Generate summary $\mathbf{y}$:

$$\mathbf{y}^* = \arg\max_{\mathbf{y}} P(\mathbf{y} | D)$$

## Applications

- News article summarization
- Meeting transcript summarization
- Scientific paper summarization
- Financial report summarization (earnings calls, SEC filings)
- Legal document summarization

## Summary

1. Extractive methods select important sentences; abstractive methods generate new text
2. Modern approaches increasingly use abstractive generation
3. Evaluation remains challenging due to the subjective nature of summarization quality
