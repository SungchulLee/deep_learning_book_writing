# Conditional Random Fields

## Overview

Conditional Random Fields (CRFs) are discriminative undirected graphical models that model the conditional distribution $P(Y \mid X)$ directly, rather than the joint $P(X, Y)$. They are the undirected counterpart to discriminative classifiers and are widely used in sequence labeling tasks.

## Linear-Chain CRF

The most common CRF variant models sequences:

$$P(y_1, \ldots, y_T \mid x) = \frac{1}{Z(x)} \prod_{t=1}^{T} \psi_t(y_t, y_{t-1}, x)$$

$$\psi_t(y_t, y_{t-1}, x) = \exp\left(\sum_k \lambda_k f_k(y_t, y_{t-1}, x, t)\right)$$

where $f_k$ are feature functions and $\lambda_k$ are learned weights.

## CRF vs HMM

| Aspect | HMM | CRF |
|--------|-----|-----|
| Model type | Generative | Discriminative |
| Models | $P(X, Y)$ | $P(Y \mid X)$ |
| Features | $P(x_t \mid y_t)$ only | Arbitrary features of entire $x$ |
| Independence | $x_t$ depends only on $y_t$ | No assumption on $x$ |
| Label bias | Not present | Not present |

The CRF's ability to use arbitrary features of the entire input sequence (not just the current position) is its key advantage.

## Training

Maximum conditional likelihood:

$$\mathcal{L}(\lambda) = \sum_{i} \log P(y^{(i)} \mid x^{(i)}; \lambda) - \frac{\alpha}{2}\|\lambda\|^2$$

The gradient requires computing marginals $P(y_t, y_{t-1} \mid x)$, which is done efficiently via the forward-backward algorithm.

## Neural CRF

In modern NLP, CRFs are combined with neural feature extractors:

$$\text{BiLSTM} \rightarrow \text{emission scores} \rightarrow \text{CRF layer} \rightarrow \text{label sequence}$$

The BiLSTM computes rich features; the CRF layer ensures globally consistent label sequences (e.g., I-PER cannot follow B-LOC).

## Applications

CRFs are used in NER, POS tagging, semantic segmentation (2D CRF), and any task requiring structured output with label dependencies.
