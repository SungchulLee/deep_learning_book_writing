# Abstractive Summarization

## Overview

Abstractive summarization generates novel text that captures the key information from source documents, allowing paraphrasing, compression, and fusion of information.

## Approaches

### Sequence-to-Sequence

Encode document, decode summary:

$$P(\mathbf{y} | \mathbf{x}) = \prod_{t=1}^{T} P(y_t | y_{<t}, \mathbf{x})$$

### Copy Mechanism (Pointer-Generator)

Combine generation with copying from source (See et al., 2017):

$$P(w) = p_{\text{gen}} \cdot P_{\text{vocab}}(w) + (1 - p_{\text{gen}}) \cdot \sum_{i: x_i = w} \alpha_i$$

where $p_{\text{gen}}$ is a learned switch between generating and copying.

## Challenges

1. **Hallucination**: Generating facts not in the source
2. **Repetition**: Repeating phrases or sentences
3. **Coverage**: Missing important information
4. **Faithfulness**: Contradicting the source

## Summary

1. Abstractive methods generate fluent, concise summaries
2. Copy mechanisms reduce hallucination by allowing verbatim copying
3. Hallucination detection and faithfulness metrics are active research areas
