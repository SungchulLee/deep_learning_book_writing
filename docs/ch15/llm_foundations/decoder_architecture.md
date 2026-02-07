# 6.11.3 Architecture and Mechanism

## Transformer Decoder Architecture

At the heart of ChatGPT lies the **Transformer decoder** architecture (Vaswani et al., 2017), specifically the decoder-only variant used in all GPT models. Unlike the original encoder-decoder Transformer designed for sequence-to-sequence tasks, the GPT family uses only the decoder stack with **causal (masked) self-attention**, ensuring that the prediction for position $t$ depends only on tokens at positions $\leq t$.

!!! note "Cross-Reference"
    For the full Transformer architecture — including multi-head attention, positional encoding, and layer normalization — see [Chapter 3: Transformer Architecture](../../ch03/index.md). This section focuses on the aspects most relevant to understanding ChatGPT's generation mechanism.

### Input Processing: Tokenization

Before any computation, raw text is converted into a sequence of **tokens** using a subword tokenization algorithm. GPT models use **Byte Pair Encoding (BPE)** (Sennrich et al., 2016), which iteratively merges the most frequent character pairs to build a vocabulary of subword units.

Given an input string, the tokenizer produces a sequence of token IDs $(x_1, x_2, \ldots, x_T)$, where each $x_t \in \{1, 2, \ldots, |\mathcal{V}|\}$ indexes into the vocabulary $\mathcal{V}$. Each token ID is then mapped to a dense vector via an **embedding matrix** $\mathbf{W}_e \in \mathbb{R}^{|\mathcal{V}| \times d}$:

$$
\mathbf{e}_t = \mathbf{W}_e[x_t] + \mathbf{p}_t
$$

where $\mathbf{p}_t \in \mathbb{R}^d$ is the positional encoding for position $t$ and $d$ is the model dimension. GPT models use **learned positional embeddings** rather than the sinusoidal encodings of the original Transformer.

### Causal Self-Attention

The core computational unit is the **causal multi-head self-attention** mechanism. For a single attention head with query, key, and value projections $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d_k}$, the attention computation at position $t$ is:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})_t = \sum_{s=1}^{t} \alpha_{ts} \, \mathbf{v}_s
$$

where the attention weights are:

$$
\alpha_{ts} = \frac{\exp\!\big(\mathbf{q}_t^\top \mathbf{k}_s / \sqrt{d_k}\big)}{\sum_{s'=1}^{t} \exp\!\big(\mathbf{q}_t^\top \mathbf{k}_{s'} / \sqrt{d_k}\big)}
$$

The **causal mask** restricts the summation to $s \leq t$, preventing the model from attending to future tokens. This is implemented by setting $\alpha_{ts} = 0$ for $s > t$ (equivalently, adding $-\infty$ to the logits before the softmax for masked positions).

With $H$ attention heads, the multi-head output is:

$$
\text{MHA}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \, \mathbf{W}_O
$$

where $\mathbf{W}_O \in \mathbb{R}^{d \times d}$ is the output projection matrix.

### Transformer Block

Each Transformer block consists of:

1. **Causal multi-head self-attention** with residual connection and layer normalization
2. **Position-wise feed-forward network (FFN)** with residual connection and layer normalization

$$
\begin{aligned}
\mathbf{h}' &= \text{LayerNorm}\!\big(\mathbf{h} + \text{MHA}(\mathbf{h})\big) \\
\mathbf{h}'' &= \text{LayerNorm}\!\big(\mathbf{h}' + \text{FFN}(\mathbf{h}')\big)
\end{aligned}
$$

The FFN typically uses a **GELU** activation:

$$
\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \, \text{GELU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
$$

where $\mathbf{W}_1 \in \mathbb{R}^{d \times 4d}$ and $\mathbf{W}_2 \in \mathbb{R}^{4d \times d}$. The inner dimension is conventionally $4 \times$ the model dimension.

!!! info "GPT-2/3 Architecture Specifications"

    | Component | GPT-2 (Large) | GPT-3 |
    |-----------|--------------|-------|
    | Layers $N$ | 36 | 96 |
    | Model dimension $d$ | 1280 | 12288 |
    | Attention heads $H$ | 20 | 96 |
    | Head dimension $d_k$ | 64 | 128 |
    | FFN inner dimension | 5120 | 49152 |
    | Parameters | 774M | 175B |

## Response Generation

### Autoregressive Decoding

ChatGPT generates responses **autoregressively**: given the dialogue context (system prompt + conversation history + current user input) as a prefix, the model samples one token at a time, appending each sampled token to the context before generating the next.

At each step $\ell$, the model computes a probability distribution over the vocabulary:

$$
p_\theta(w_\ell \mid w_{<\ell}) = \text{softmax}\!\big(\mathbf{W}_e^\top \, \mathbf{h}_\ell^{(N)}\big)
$$

where $\mathbf{h}_\ell^{(N)}$ is the hidden state at position $\ell$ from the final Transformer layer and $\mathbf{W}_e$ is the (tied) embedding matrix.

### Decoding Strategies

The choice of decoding strategy critically affects response quality:

**Greedy Decoding** selects the highest-probability token at each step:

$$
w_\ell = \arg\max_{w \in \mathcal{V}} \; p_\theta(w \mid w_{<\ell})
$$

This is deterministic but tends to produce repetitive and generic text.

**Temperature Sampling** introduces controlled randomness by scaling the logits before softmax:

$$
p_\tau(w \mid w_{<\ell}) = \frac{\exp\!\big(z_w / \tau\big)}{\sum_{w'} \exp\!\big(z_{w'} / \tau\big)}
$$

where $z_w$ are the raw logits and $\tau > 0$ is the **temperature** parameter. As $\tau \to 0$, sampling approaches greedy decoding; as $\tau \to \infty$, sampling approaches uniform random selection. Typical values for conversational use are $\tau \in [0.7, 1.0]$.

**Top-$k$ Sampling** (Fan et al., 2018) restricts sampling to the $k$ most probable tokens:

$$
p_{\text{top-}k}(w \mid w_{<\ell}) \propto \begin{cases} p_\theta(w \mid w_{<\ell}) & \text{if } w \in \mathcal{V}_k \\ 0 & \text{otherwise} \end{cases}
$$

where $\mathcal{V}_k$ contains the top-$k$ tokens by probability.

**Nucleus (Top-$p$) Sampling** (Holtzman et al., 2020) dynamically selects the smallest set of tokens whose cumulative probability exceeds a threshold $p$:

$$
\mathcal{V}_p = \min \left\{ \mathcal{S} \subseteq \mathcal{V} : \sum_{w \in \mathcal{S}} p_\theta(w \mid w_{<\ell}) \geq p \right\}
$$

This adapts the effective vocabulary size to the model's confidence at each step — using fewer tokens when the model is confident and more when it is uncertain. In practice, ChatGPT uses a combination of temperature scaling and nucleus sampling.

!!! tip "Practical Guidance"
    For factual Q&A tasks, lower temperatures ($\tau \approx 0.2$–$0.5$) and smaller $p$ values produce more focused, deterministic responses. For creative tasks (story writing, brainstorming), higher temperatures ($\tau \approx 0.8$–$1.0$) and larger $p$ values encourage diversity.

## Context Window and Attention Patterns

A critical architectural constraint is the **context window** — the maximum number of tokens the model can process in a single forward pass. For GPT-3, this was 2,048 tokens; GPT-4 extended it to 8,192 or 32,768 tokens (depending on the variant), with later versions supporting up to 128K tokens.

The context window determines how much dialogue history the model can "see" when generating a response. For multi-turn conversations, the entire dialogue is concatenated into a single sequence:

$$
[\texttt{system}] \;\|\; [u_1] \;\|\; [r_1] \;\|\; [u_2] \;\|\; [r_2] \;\|\; \cdots \;\|\; [u_t]
$$

where $\|$ denotes concatenation. When this sequence exceeds the context window, earlier turns must be truncated or summarized, which can lead to loss of contextual information in very long conversations.

The causal attention mechanism allows every token in the response to attend to *all* preceding tokens in the context — including the system prompt, all previous user messages, and all previous assistant responses. This is what enables ChatGPT to maintain coherence across multi-turn dialogues, reference earlier parts of the conversation, and follow system-level instructions throughout.

---

**Next:** [6.11.4 Training Process](training_process.md)
