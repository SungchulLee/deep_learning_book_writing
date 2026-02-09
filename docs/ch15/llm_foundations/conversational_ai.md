# Conversational AI

## Overview

**Conversational AI** refers to systems that engage in multi-turn, natural language dialogue with users — interpreting intent, maintaining context across exchanges, and generating fluent, contextually appropriate responses. From early rule-based chatbots to modern LLM-powered assistants, the field has undergone a fundamental transformation: what was once painstakingly engineered through dialogue trees and pattern matching is now learned end-to-end from massive text corpora and refined through human feedback.

This section traces the evolution of conversational AI, formalizes the dialogue generation problem as autoregressive conditional generation, covers the key architectural and training innovations that make modern dialogue systems possible, and connects to the broader LLM ecosystem — [alignment via RLHF](../alignment/rlhf.md), [prompting techniques](../prompting/prompting_overview.md), [retrieval-augmented generation](../rag/rag_overview.md), and [agentic capabilities](../agents/agent_overview.md) — that transforms a base language model into a useful conversational assistant.

---

## 1. Historical Development

### 1.1 Era 1: Rule-Based Systems (1960s–1990s)

The earliest conversational agent was **ELIZA** (Weizenbaum, 1966), a program that mimicked psychotherapist dialogue using keyword pattern matching and scripted transformations. Despite having no understanding whatsoever, ELIZA demonstrated that even simple pattern-response rules could create a compelling illusion of comprehension — the so-called "ELIZA effect."

Subsequent systems extended the rule-based approach:

- **PARRY** (Colby, 1972): Simulated a paranoid patient with more sophisticated state tracking
- **A.L.I.C.E.** (Wallace, 1995): Used AIML (Artificial Intelligence Markup Language) with thousands of hand-authored patterns
- **Commercial IVR systems**: Deployed in call centers with rigid dialogue trees and slot-filling grammars

The core limitation of all rule-based systems was the same: they could not learn from data, could not generalize beyond their programmed rules, and scaled only through manual engineering effort.

### 1.2 Era 2: Statistical and Neural Approaches (2000s–2016)

Machine learning enabled systems to learn conversational patterns from data:

- **Statistical dialogue systems**: Used probabilistic models for dialogue state tracking (belief tracking over slot-value pairs) and policy optimization via reinforcement learning
- **Sequence-to-sequence models** (Sutskever et al., 2014): Framed conversation as a translation problem — encoding the input utterance and decoding a response
- **Attention-augmented Seq2Seq** (Bahdanau et al., 2015): Improved by allowing the decoder to focus on relevant parts of the input at each generation step
- **Memory networks** (Sukhbaatar et al., 2015): Introduced explicit memory modules for multi-turn reasoning

These approaches improved fluency and could learn from dialogue corpora, but suffered from generic responses ("I don't know"), poor long-range coherence, and limited world knowledge.

### 1.3 Era 3: Large Language Models (2017–Present)

The Transformer architecture and large-scale pretraining changed the paradigm entirely:

| System | Year | Key Innovation |
|--------|------|---------------|
| GPT | 2018 | Autoregressive pretraining + task fine-tuning |
| GPT-2 | 2019 | Scaled unsupervised LM; zero-shot task transfer |
| Meena / BlenderBot | 2020 | Dialogue-specific training at scale |
| GPT-3 | 2020 | Few-shot learning via in-context examples |
| ChatGPT | 2022 | RLHF alignment for dialogue quality and safety |
| GPT-4 | 2023 | Multimodal input; improved reasoning and instruction following |
| Claude, Gemini, LLaMA-Chat | 2023–24 | Diverse alignment approaches; open-weight dialogue models |

The critical insight of this era: a sufficiently large language model pretrained on diverse text already contains the "knowledge" for conversation — the challenge shifts from building dialogue capabilities to **aligning** the model's behavior with human preferences. This alignment process is covered in [RLHF](../alignment/rlhf.md), [DPO](../alignment/dpo.md), and the broader [alignment pipeline](../alignment/training_pipeline.md).

---

## 2. Formal Framework

### 2.1 Dialogue as Conditional Generation

A conversational system maps a dialogue history to a response. Let $\mathcal{V}$ denote the vocabulary, and define the dialogue history at turn $t$ as:

$$H_t = (u_1, r_1, u_2, r_2, \ldots, u_t)$$

where $u_i$ is the user utterance and $r_i$ is the system response at turn $i$. The conversational model defines a conditional distribution over responses:

$$p_\theta(r_t \mid H_t)$$

For autoregressive models, the response $r_t = (w_1, w_2, \ldots, w_L)$ is generated token by token via the chain rule:

$$p_\theta(r_t \mid H_t) = \prod_{\ell=1}^{L} p_\theta(w_\ell \mid H_t, w_{1:\ell-1})$$

This is the same [next-token prediction](next_token_prediction.md) objective used in pretraining, applied at inference time with the dialogue history as the prompt. The [decoder architecture](decoder_architecture.md) processes the concatenated history through causal self-attention, ensuring each token attends only to preceding tokens.

### 2.2 System Prompt and Instruction Following

In practice, the dialogue history is prepended with a **system prompt** $s$ that defines the assistant's behavior:

$$p_\theta(r_t \mid s, H_t) = \prod_{\ell=1}^{L} p_\theta(w_\ell \mid s, H_t, w_{1:\ell-1})$$

The system prompt is a form of [zero-shot prompting](../prompting/zero_shot.md) that steers the model's persona, tone, and capabilities without modifying weights. Fine-tuning the model to follow system prompts reliably is part of the [alignment training pipeline](../alignment/training_pipeline.md).

### 2.3 Dialogue State

Beyond the raw text history, many systems maintain an implicit or explicit **dialogue state** $b_t$ that summarizes the conversation:

$$b_t = f(H_t)$$

In classical task-oriented dialogue, $b_t$ is a structured belief state over slot-value pairs (e.g., `destination=London`, `date=tomorrow`). In LLM-based systems, the dialogue state is implicitly encoded in the model's hidden representations across the context window, with no explicit state tracking module. The context window limit — typically 4K to 128K+ tokens depending on the model (see [tokenization and scale](tokenization_scale.md)) — determines how much history the model can condition on.

---

## 3. From Base LLM to Conversational Assistant

A pretrained language model is not inherently conversational — it is trained to predict the next token in web text, which includes dialogue but also code, articles, and many other formats. Converting a base LLM into a useful conversational assistant requires a multi-stage pipeline:

```
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│ Pretraining│────▶│ Supervised │────▶│  Alignment │────▶│  Deployed   │
│  (Base LM) │     │ Fine-Tuning│     │   (RLHF)   │     │ Assistant  │
└────────────┘     └────────────┘     └────────────┘     └────────────┘
     Next-token         Dialogue           Human              + RAG
     prediction         examples          feedback            + Tools
```

### Stage 1: Pretraining

The base language model learns general language competence, factual knowledge, and reasoning patterns from a large text corpus. This is covered in [pretraining objectives](pretraining_objectives.md) and [training data](training_data.md).

### Stage 2: Supervised Fine-Tuning (SFT)

The model is fine-tuned on curated dialogue examples — human-written conversations demonstrating helpful, accurate, and safe responses. This teaches the model the conversational format: when to ask clarifying questions, how to structure responses, and how to follow instructions.

### Stage 3: Alignment

SFT alone produces a model that mimics the training dialogues but may still generate harmful, dishonest, or unhelpful content. [RLHF](../alignment/rlhf.md) refines the model using human preference data:

1. A [reward model](../alignment/reward_modeling.md) is trained on human comparisons of response quality
2. The dialogue model is optimized via [PPO](../alignment/ppo_llm.md) or [DPO](../alignment/dpo.md) to maximize the reward while staying close to the SFT model

This pipeline is what transforms GPT-4-base (a next-token predictor) into ChatGPT (a conversational assistant).

### Stage 4: Augmentation

The deployed assistant is further enhanced with capabilities beyond the base model:

- **[Retrieval-Augmented Generation](../rag/rag_overview.md)**: Grounds responses in retrieved documents, reducing hallucination
- **[Tool Use](../agents/tool_use.md)**: Enables the model to call external APIs, execute code, or query databases
- **[Function Calling](../agents/function_calling.md)**: Structured interface for invoking external tools
- **[Multi-Agent Systems](../agents/multi_agent.md)**: Orchestrates multiple specialized agents for complex tasks

---

## 4. Key Components of Modern Dialogue Systems

### 4.1 Context Management

The finite context window is the primary bottleneck for long conversations. Strategies for managing context include:

| Strategy | Approach | Tradeoff |
|----------|----------|----------|
| Full history | Include all turns in the prompt | Hits context limit in long conversations |
| Sliding window | Keep only the most recent $k$ turns | Loses early context |
| Summarization | Compress old turns into a summary | Lossy but scalable |
| Retrieval | Index past turns, retrieve relevant ones | Requires retrieval infrastructure |
| KV cache | Cache key-value pairs for efficient reuse | Memory-efficient; see [KV Cache](../inference/kv_cache.md) |

### 4.2 Decoding Strategies

How tokens are sampled from $p_\theta(w_\ell \mid \cdot)$ critically affects response quality:

| Strategy | Mechanism | Effect on Dialogue |
|----------|-----------|-------------------|
| Greedy | $\arg\max$ at each step | Deterministic, repetitive |
| Temperature sampling | Scale logits by $1/T$ before softmax | $T < 1$: focused; $T > 1$: creative |
| Top-$k$ | Sample from top $k$ tokens | Bounds randomness |
| Nucleus (top-$p$) | Sample from smallest set with cumulative prob $\geq p$ | Adaptive vocabulary size |
| Beam search | Track top-$B$ sequences | Better for short, precise outputs |

For open-ended dialogue, **nucleus sampling** with $p \in [0.9, 0.95]$ and temperature $T \in [0.7, 1.0]$ is the standard choice, balancing coherence with diversity.

### 4.3 Safety and Guardrails

Conversational systems must handle adversarial inputs, requests for harmful content, and edge cases:

- **Input classifiers**: Detect harmful or off-topic queries before generation
- **Output filters**: Screen generated responses for policy violations
- **[Constitutional AI](../alignment/constitutional.md)**: Self-critique and revision guided by a set of principles
- **Refusal training**: Teaching the model to decline harmful requests while remaining helpful for benign ones

---

## 5. Task-Oriented vs. Open-Domain Dialogue

### 5.1 Task-Oriented Dialogue

Goal: complete a specific task (booking a flight, querying a database, troubleshooting a device).

| Component | Function |
|-----------|----------|
| Natural Language Understanding (NLU) | Extract intent and entities from user utterance |
| Dialogue State Tracking (DST) | Maintain belief state over slots |
| Policy | Decide next action (ask, confirm, execute) |
| Natural Language Generation (NLG) | Produce response text |

Modern LLM-based task-oriented systems collapse these components into a single model prompted with the task schema, using [function calling](../agents/function_calling.md) to interface with backends.

### 5.2 Open-Domain Dialogue

Goal: sustain engaging, coherent conversation on any topic.

Key challenges:

- **Consistency**: Maintaining a stable persona and avoiding contradictions across turns
- **Specificity**: Generating informative responses rather than generic platitudes
- **Engagement**: Asking questions, showing interest, and adapting tone to the user
- **Groundedness**: Stating facts accurately rather than hallucinating

The shift from specialized dialogue systems to general-purpose LLMs has blurred this distinction: a single aligned LLM like ChatGPT handles both task-oriented and open-domain dialogue through different prompting strategies.

---

## 6. Evaluation

### 6.1 Automatic Metrics

| Metric | What It Measures | Limitations |
|--------|-----------------|-------------|
| Perplexity | Model's surprise at reference responses | Doesn't capture response quality |
| BLEU / ROUGE | $n$-gram overlap with references | Poor correlation with dialogue quality |
| BERTScore | Semantic similarity via embeddings | Better than $n$-gram but still limited |
| Distinct-$n$ | Ratio of unique $n$-grams | Measures diversity, not quality |

### 6.2 Human Evaluation

Human evaluation remains the gold standard for dialogue quality. Common dimensions:

- **Helpfulness**: Does the response address the user's need?
- **Harmlessness**: Is the response free of harmful content?
- **Honesty**: Is the response factually accurate and calibrated in uncertainty?
- **Coherence**: Does the response follow logically from the conversation?
- **Fluency**: Is the response grammatically correct and natural?

Pairwise comparison (A/B testing) between models is more reliable than absolute ratings. The [alignment section](../alignment/alignment_overview.md) covers how human evaluations are used to train reward models.

### 6.3 Benchmarks

| Benchmark | Focus | Format |
|-----------|-------|--------|
| MMLU | Factual knowledge across domains | Multiple choice |
| MT-Bench | Multi-turn instruction following | Open-ended, GPT-4 judged |
| AlpacaEval | Instruction following quality | Pairwise vs. reference model |
| Chatbot Arena | Open-ended dialogue | ELO rating from human pairwise comparisons |
| HumanEval | Code generation | Functional correctness |

For broader evaluation context, see [evaluation metrics](evaluation_metrics.md).

---

## 7. Prompting for Dialogue

The quality of conversational responses depends heavily on how the model is prompted. Key techniques from [prompting](../prompting/prompting_overview.md) that apply to dialogue:

| Technique | Application to Dialogue | Reference |
|-----------|------------------------|-----------|
| System prompts | Define assistant persona, tone, constraints | [Zero-Shot](../prompting/zero_shot.md) |
| Few-shot examples | Demonstrate desired response format | [Few-Shot](../prompting/few_shot.md) |
| Chain-of-thought | Encourage step-by-step reasoning before answering | [CoT](../prompting/chain_of_thought.md) |
| Self-consistency | Sample multiple responses and select the most common | [Self-Consistency](../prompting/self_consistency.md) |
| ReAct | Interleave reasoning and tool use | [ReAct](../agents/react.md) |

---

## 8. Finance Applications

Conversational AI has significant applications in quantitative finance and financial services:

| Application | Approach | Notes |
|-------------|----------|-------|
| Financial research assistant | RAG over earnings calls, SEC filings, research reports | Grounds answers in source documents; reduces hallucination |
| Trading desk copilot | LLM agent with market data API access | Real-time portfolio queries, scenario analysis |
| Client-facing advisory | Aligned dialogue model with compliance guardrails | Regulatory constraints on financial advice |
| Code generation for quant | Conversational coding assistant for Python/pandas/PyTorch | See [LLM applications](../agents/llm_applications.md) |
| Anomaly explanation | Dialogue interface over monitoring dashboards | Natural language queries about model drift, P&L attribution |
| Document Q&A | RAG over prospectuses, term sheets, legal contracts | [Document chunking](../rag/chunking.md) + [reranking](../rag/reranking.md) |

Financial conversational AI requires particular attention to **hallucination prevention** (incorrect financial data can have material consequences), **auditability** (responses should be traceable to source documents), and **regulatory compliance** (financial advice is regulated in most jurisdictions).

---

## 9. Key Takeaways

1. **Conversational AI has shifted from engineering to learning**: modern systems generate responses by sampling from learned distributions over tokens, conditioned on dialogue history, rather than following scripted rules.

2. **The base LLM is necessary but not sufficient**: converting a pretrained language model into a useful assistant requires supervised fine-tuning, alignment via RLHF or DPO, and augmentation with retrieval and tool use.

3. **Context management is the primary bottleneck** for long conversations — strategies include sliding windows, summarization, retrieval, and efficient KV caching.

4. **Alignment determines usability**: the same base model can produce helpful or harmful responses depending on the alignment procedure. RLHF and constitutional AI are the current standard approaches.

5. **Evaluation remains challenging**: automatic metrics correlate poorly with perceived quality; human evaluation through pairwise comparison is the gold standard.

6. **Finance applications** require grounding, auditability, and compliance guardrails beyond what general-purpose conversational AI provides.

---

## Exercises

### Exercise 1: Dialogue Probability

Given a vocabulary of size $|\mathcal{V}| = 50{,}000$ and a response length of $L = 100$ tokens, how many possible responses exist? Why does this make beam search with large beam widths impractical, and why is sampling preferred for dialogue?

### Exercise 2: Temperature Effects

Using any LLM API, generate 10 responses to the same prompt at temperatures $T \in \{0.1, 0.5, 0.7, 1.0, 1.5\}$. Measure diversity (Distinct-2) and coherence (self-rated 1–5). At what temperature is the diversity-coherence tradeoff optimal for dialogue?

### Exercise 3: Context Window Saturation

Simulate a 50-turn conversation where each turn averages 100 tokens. At what turn does a model with a 4K context window lose access to the initial system prompt? Design a summarization strategy and compare response quality with and without it.

### Exercise 4: RAG for Financial Q&A

Build a simple RAG pipeline that retrieves relevant paragraphs from a set of earnings call transcripts and generates answers. Compare response factuality with and without retrieval on 20 financial questions.

### Exercise 5: Alignment Comparison

Compare responses from a base LLM and its RLHF-aligned variant on 10 ambiguous or sensitive financial questions. Rate each on helpfulness, harmlessness, and honesty. How does alignment change the response distribution?

---

## References

1. Weizenbaum, J. (1966). ELIZA — A Computer Program for the Study of Natural Language Communication Between Man and Machine. *Communications of the ACM*, 9(1), 36–45.
2. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks. *Advances in Neural Information Processing Systems (NeurIPS)*.
3. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*.
4. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving Language Understanding by Generative Pre-Training. *OpenAI Technical Report*.
5. Brown, T. B., Mann, B., Ryder, N., Subbiah, M., et al. (2020). Language Models Are Few-Shot Learners. *Advances in Neural Information Processing Systems (NeurIPS)*.
6. Ouyang, L., Wu, J., Jiang, X., Almeida, D., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *Advances in Neural Information Processing Systems (NeurIPS)*.
7. Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. *Advances in Neural Information Processing Systems (NeurIPS)*.
8. Bai, Y., Jones, A., Ndousse, K., Askell, A., et al. (2022). Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. *arXiv preprint arXiv:2204.05862*.
9. Lewis, P., Perez, E., Piktus, A., Petroni, F., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems (NeurIPS)*.
10. Roller, S., Dinan, E., Goyal, N., Ju, D., et al. (2021). Recipes for Building an Open-Domain Chatbot. *Proceedings of the 16th Conference of the EACL*.
