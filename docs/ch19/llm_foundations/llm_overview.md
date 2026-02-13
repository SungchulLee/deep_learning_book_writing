# LLM Overview

## Learning Objectives

- Define large language models and their distinguishing characteristics
- Trace the evolution from GPT-1 to modern open-source models
- Understand the core autoregressive generation paradigm
- Identify key LLM capabilities and limitations

## What is a Large Language Model?

A **Large Language Model (LLM)** is a neural network—typically based on the transformer decoder architecture—trained on massive text corpora to model the probability distribution over sequences of tokens. Formally, given a sequence of tokens $x_1, x_2, \ldots, x_{t-1}$, the model estimates:

$$P(x_t \mid x_1, x_2, \ldots, x_{t-1}) = \text{softmax}(W_h \cdot h_t + b)$$

where $h_t$ is the hidden state produced by the transformer at position $t$, and $W_h$ is the output projection matrix (often tied to the input embedding matrix).

The term "large" is not precisely defined but generally implies:

- **Parameter count**: $\geq 1$ billion parameters
- **Training data**: Hundreds of billions to trillions of tokens
- **Compute budget**: $\geq 10^{21}$ FLOPs

## Historical Evolution

| Model | Year | Parameters | Training Tokens | Key Innovation |
|-------|------|-----------|----------------|----------------|
| GPT-1 | 2018 | 117M | ~5B | Unsupervised pretraining + supervised fine-tuning |
| GPT-2 | 2019 | 1.5B | ~10B | Zero-shot task transfer |
| GPT-3 | 2020 | 175B | 300B | In-context learning, few-shot prompting |
| PaLM | 2022 | 540B | 780B | Pathways system, chain-of-thought |
| LLaMA | 2023 | 7-65B | 1.0-1.4T | Open weights, compute-efficient training |
| LLaMA 2 | 2023 | 7-70B | 2T | RLHF alignment, extended context |
| Mixtral | 2024 | 8x7B | ~2T | Mixture-of-experts efficiency |
| LLaMA 3 | 2024 | 8-405B | 15T+ | Massive data scaling |

## Architecture Families

Modern LLMs fall into three architectural families:

### Decoder-Only (Autoregressive)

The dominant paradigm. Each token attends only to previous tokens via causal masking. Examples: GPT series, LLaMA, Mistral, Falcon.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

where $M$ is the causal mask ($M_{ij} = -\infty$ for $j > i$).

### Encoder-Only (Bidirectional)

Full bidirectional attention, typically trained with masked language modeling. Examples: BERT, RoBERTa, DeBERTa.

### Encoder-Decoder

Encoder processes input bidirectionally; decoder generates output autoregressively with cross-attention. Examples: T5, BART, Flan-T5.

## Core Capabilities

LLMs exhibit several remarkable capabilities that emerge at scale:

1. **In-Context Learning**: Performing tasks from natural language descriptions and examples without parameter updates
2. **Instruction Following**: Executing complex multi-step instructions
3. **Reasoning**: Multi-step logical and mathematical reasoning (especially with chain-of-thought)
4. **Code Generation**: Writing, debugging, and explaining code
5. **Knowledge Retrieval**: Recalling factual information absorbed during pretraining

## Limitations

- **Hallucination**: Generating plausible but factually incorrect content
- **Knowledge Cutoff**: No awareness of events after training data collection
- **Context Window**: Fixed maximum sequence length (though expanding rapidly)
- **Reasoning Brittleness**: Failures on novel reasoning patterns outside training distribution
- **Computational Cost**: Inference cost scales linearly with sequence length (quadratically for attention without optimization)

## Relevance to Quantitative Finance

LLMs are increasingly deployed in finance for:

- **Document Understanding**: Parsing complex regulatory filings (10-K, 10-Q, proxy statements)
- **Sentiment Analysis**: Extracting market sentiment from news, social media, and analyst reports
- **Structured Extraction**: Converting unstructured financial text to structured data (JSON, tables)
- **Research Automation**: Generating investment memos and risk assessments
- **Code Generation**: Writing quantitative models, backtesting scripts, and data pipelines

## References

1. Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training."
2. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners."
3. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS*.
4. Chowdhery, A., et al. (2022). "PaLM: Scaling Language Modeling with Pathways." *arXiv*.
5. Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models."
