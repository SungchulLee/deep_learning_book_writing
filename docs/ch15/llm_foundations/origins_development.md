# 6.11.2 Understanding ChatGPT — Origins and Development

## The GPT Series

ChatGPT is a conversational AI model developed by OpenAI as part of the broader **Generative Pre-trained Transformer (GPT)** series. The GPT series represents a sustained effort to scale autoregressive language models — both in parameter count and training data — to achieve increasingly general language capabilities.

### GPT-1 (2018)

GPT-1 introduced the paradigm of **unsupervised pre-training followed by supervised fine-tuning**. The model used a 12-layer Transformer decoder with 117M parameters, pre-trained on the BooksCorpus dataset (~7,000 unpublished books). The key insight was that a language model trained to predict the next token could develop internal representations useful for downstream NLP tasks (classification, entailment, similarity) with minimal task-specific architecture changes.

The pre-training objective is the standard **causal language modeling (CLM)** loss:

$$
\mathcal{L}_{\text{CLM}}(\theta) = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$

where $x_1, \ldots, x_T$ is a token sequence and $\theta$ denotes model parameters.

### GPT-2 (2019)

GPT-2 scaled to 1.5B parameters and was trained on **WebText**, a dataset of ~8 million web pages filtered for quality. The key contribution was demonstrating **zero-shot task transfer**: a sufficiently large language model could perform tasks (summarization, translation, question answering) without any fine-tuning, simply by conditioning on appropriate prompts.

### GPT-3 (2020)

GPT-3 represented a massive leap to **175 billion parameters**, trained on a filtered blend of Common Crawl, WebText, books, and Wikipedia. It demonstrated strong **few-shot** and **in-context learning** capabilities — the ability to perform new tasks given only a few examples in the prompt, without any gradient updates.

The scaling behavior follows an empirical **power law** relationship between model size $N$, dataset size $D$, compute budget $C$, and loss $L$:

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}
$$

where $N_c, D_c$ are critical scale parameters and $\alpha_N, \alpha_D$ are empirical exponents. These scaling laws, studied systematically by Kaplan et al. (2020) and later refined by Hoffmann et al. (2022, "Chinchilla"), provide the theoretical motivation for continued model scaling. See [Section 6.1: Scaling Laws](../language_models/fundamentals.md) for a detailed treatment.

### GPT-4 (2023)

GPT-4 introduced **multimodal capabilities** (text + image inputs) and further improvements in reasoning, instruction following, and factual accuracy. While OpenAI did not disclose the exact architecture or parameter count, GPT-4 demonstrated qualitative improvements in handling nuanced queries, sustaining long dialogues, and performing complex multi-step reasoning tasks.

!!! info "GPT Evolution Summary"

    | Model  | Year | Parameters | Key Innovation |
    |--------|------|-----------|----------------|
    | GPT-1  | 2018 | 117M      | Pre-train + fine-tune paradigm |
    | GPT-2  | 2019 | 1.5B      | Zero-shot transfer via prompting |
    | GPT-3  | 2020 | 175B      | Few-shot in-context learning |
    | GPT-4  | 2023 | Undisclosed | Multimodal inputs, improved reasoning |

## From GPT to ChatGPT

The GPT base models are trained purely on next-token prediction and are not inherently optimized for *dialogue*. They may produce outputs that are factually plausible but unhelpful, verbose, evasive, or misaligned with user intent. **ChatGPT** bridges this gap through a three-stage **alignment pipeline**:

1. **Supervised Fine-Tuning (SFT):** Human annotators write high-quality dialogue responses for a set of prompts. The model is fine-tuned on these demonstrations using the standard CLM loss.

2. **Reward Model Training:** Human annotators rank multiple model outputs for the same prompt. These rankings are used to train a **reward model** $R_\phi(x, y)$ that scores response quality. The ranking data is converted to pairwise comparisons via the **Bradley-Terry model**:

    $$
    P(y_w \succ y_l \mid x) = \sigma\big(R_\phi(x, y_w) - R_\phi(x, y_l)\big)
    $$

    where $y_w$ is the preferred response, $y_l$ is the dispreferred response, and $\sigma$ is the sigmoid function.

3. **RLHF via PPO:** The SFT model is further optimized using **Proximal Policy Optimization (PPO)** to maximize the reward while staying close to the SFT policy $\pi_{\text{SFT}}$:

    $$
    \mathcal{L}_{\text{RLHF}}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_\theta(\cdot \mid x)} \Big[ R_\phi(x, y) - \beta \, D_{\text{KL}}\big(\pi_\theta(\cdot \mid x) \;\|\; \pi_{\text{SFT}}(\cdot \mid x)\big) \Big]
    $$

    The KL penalty (controlled by $\beta > 0$) prevents the policy from deviating too far from the SFT baseline, which would lead to reward hacking — generating outputs that exploit the reward model without genuine quality improvement.

!!! note "Cross-Reference"
    The full mathematical treatment of RLHF, including the derivation of the PPO clipped objective and practical training considerations, is provided in [Section 6.5: Alignment and RLHF](../alignment/rlhf.md).

## Key Design Principles

Several design principles distinguish ChatGPT from the base GPT models:

**Helpfulness.** The model is trained to be genuinely useful — providing detailed, accurate, and actionable responses rather than generic or evasive text.

**Harmlessness.** Through RLHF and content filtering, the model is trained to refuse harmful requests, avoid generating toxic content, and flag uncertainty rather than fabricating information.

**Honesty.** The alignment process encourages the model to acknowledge limitations, express uncertainty when appropriate, and avoid confidently stating false information.

These three objectives (sometimes called the "HHH" criteria) formalize the alignment goals and serve as the guiding principles for the reward model training data.

---

**Next:** [6.11.3 Architecture and Mechanism](architecture.md)
