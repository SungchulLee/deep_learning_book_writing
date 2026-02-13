# Connection to LLMs

## Overview

Large language models are autoregressive models. Understanding this connection reveals why techniques developed for image AR models transfer to LLMs and vice versa.

## Shared Framework

Both LLMs and autoregressive image models:

1. Decompose the joint distribution via the chain rule
2. Use transformer architectures with causal masking
3. Train with next-token prediction (cross-entropy loss)
4. Sample sequentially via ancestral sampling

The only difference is the modality of the tokens: text tokens (from a vocabulary) vs. image tokens (from a codebook) vs. audio tokens.

## Techniques That Transfer

| Technique | Origin | Transfers To |
|-----------|--------|-------------|
| Causal attention masking | NLP | Images, audio |
| KV caching | NLP | All AR models |
| Top-k / nucleus sampling | NLP | Image generation |
| Classifier-free guidance | Images | Text generation |
| Tokenization (BPE, VQVAE) | NLP / Vision | Cross-modal |

## Multimodal Autoregressive Models

The shared framework enables multimodal generation by interleaving tokens from different modalities:

$$p(\text{text}, \text{image}) = \prod_i p(t_i \mid t_{<i})$$

where $t_i$ can be a text token or an image token. Models like Chameleon, Gemini, and GPT-4o use this unified approach.

## Scaling Laws

Scaling laws for autoregressive models appear consistent across modalities:

$$L(N) \propto N^{-\alpha}$$

where $L$ is the loss, $N$ is the number of parameters, and $\alpha \approx 0.07$ for text (Kaplan et al., 2020). Similar power-law relationships hold for image and audio AR models, suggesting a universal principle.
