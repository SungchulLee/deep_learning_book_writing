# Evaluation Metrics

## Overview

Evaluating large language models is uniquely challenging because LLMs are used for an enormous range of tasks — from factual question answering to creative writing, code generation to multi-turn dialogue — and no single metric captures quality across all of them. This section covers the three pillars of LLM evaluation: **intrinsic metrics** that measure language modeling quality, **task-specific metrics** that assess performance on particular capabilities, and **human evaluation** protocols that capture the nuanced quality dimensions that automatic metrics miss. We also cover the growing role of **LLM-as-judge** evaluation and the major **benchmarks** used to compare models.

The evaluation methods here apply to base language models (assessed during [pretraining](pretraining_objectives.md)), to instruction-tuned models (assessed on [prompting](../prompting/prompting_overview.md) tasks), and to aligned assistants (assessed on the [alignment](../alignment/alignment_overview.md) objectives of helpfulness, harmlessness, and honesty).

---

## 1. Intrinsic Metrics

Intrinsic metrics evaluate the language model's core capability: predicting text.

### 1.1 Perplexity

**Perplexity (PPL)** is the standard intrinsic metric for language models. It measures how well the model predicts a held-out test set, defined as the exponentiated average negative log-likelihood:

$$\text{PPL} = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})\right)$$

where $T$ is the total number of tokens in the test set.

**Interpretation**: Perplexity equals the effective vocabulary size the model is "choosing from" at each step. A perplexity of 20 means the model is, on average, as uncertain as if choosing uniformly among 20 equally likely tokens. Lower is better.

**Relationship to cross-entropy**: Perplexity is $2^H$ where $H$ is the cross-entropy in bits per token, or equivalently $e^{H'}$ where $H'$ is cross-entropy in nats. The [scaling laws](../scaling/scaling_overview.md) predict that perplexity decreases as a power law with model size and training data.

**Bits per byte (BPB)**: Because perplexity depends on the tokenizer (models with larger vocabularies tend to have lower perplexity per token), **bits per byte** provides a tokenizer-independent comparison:

$$\text{BPB} = \frac{\text{total cross-entropy (bits)}}{\text{total bytes in test set}}$$

This normalizes by the raw data size rather than the number of tokens. BPB is used in the [Chinchilla](../scaling/chinchilla.md) scaling analysis for fair comparison across models with different tokenizers (see [Tokenization and Scale](tokenization_scale.md)).

### 1.2 Limitations of Perplexity

| Limitation | Explanation |
|-----------|-------------|
| Doesn't measure task quality | Low perplexity ≠ helpful, safe, or accurate responses |
| Tokenizer-dependent | Different tokenizers produce different perplexity values for the same model quality |
| Distribution-dependent | Perplexity on one test set may not predict performance on another domain |
| Not comparable across architectures | Decoder-only, encoder-only, and encoder-decoder models have different likelihood formulations |

Perplexity is essential for monitoring [pretraining](pretraining_objectives.md) progress and validating [scaling laws](../scaling/model_vs_data.md), but insufficient for evaluating the downstream usefulness of an LLM.

---

## 2. Reference-Based Text Metrics

These metrics compare generated text against one or more reference outputs. They were developed for tasks like machine translation and summarization where reference answers exist.

### 2.1 BLEU

**BLEU** (Bilingual Evaluation Understudy; Papineni et al., 2002) measures $n$-gram precision between generated output $\hat{y}$ and reference $y$:

$$\text{BLEU} = \text{BP} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

where $p_n$ is the modified $n$-gram precision (each $n$-gram counted at most as many times as it appears in the reference), $w_n = 1/N$ are uniform weights, and BP is the **brevity penalty**:

$$\text{BP} = \begin{cases} 1 & \text{if } |\hat{y}| \geq |y| \\ \exp(1 - |y|/|\hat{y}|) & \text{if } |\hat{y}| < |y| \end{cases}$$

### 2.2 ROUGE

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation; Lin, 2004) emphasizes recall. The most common variants:

| Variant | Measures | Best For |
|---------|----------|----------|
| ROUGE-1 | Unigram overlap | Content coverage |
| ROUGE-2 | Bigram overlap | Fluency + content |
| ROUGE-L | Longest common subsequence | Sequence-level structure |

$$\text{ROUGE-L} = \frac{(1 + \beta^2) \cdot P_{\text{lcs}} \cdot R_{\text{lcs}}}{R_{\text{lcs}} + \beta^2 \cdot P_{\text{lcs}}}$$

where $R_{\text{lcs}} = \text{LCS}(\hat{y}, y) / |y|$ and $P_{\text{lcs}} = \text{LCS}(\hat{y}, y) / |\hat{y}|$.

### 2.3 BERTScore

**BERTScore** (Zhang et al., 2020) computes semantic similarity in a learned embedding space, addressing the lexical mismatch problem of BLEU and ROUGE:

$$P_{\text{BERT}} = \frac{1}{|\hat{y}|}\sum_{i} \max_{j}\cos(\hat{\mathbf{e}}_i, \mathbf{e}_j), \quad R_{\text{BERT}} = \frac{1}{|y|}\sum_{j} \max_{i}\cos(\hat{\mathbf{e}}_i, \mathbf{e}_j)$$

$$F_{\text{BERT}} = 2 \cdot \frac{P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}$$

BERTScore correlates more strongly with human judgments than $n$-gram metrics because semantically equivalent but lexically different outputs receive high scores.

### 2.4 When Reference-Based Metrics Fail

All reference-based metrics share a fundamental limitation for LLM evaluation: **many valid outputs exist for any given prompt**. In open-ended generation, dialogue, or creative tasks, comparing against a single reference penalizes valid alternative responses. This is why LLM evaluation has shifted toward reference-free approaches (LLM-as-judge, human pairwise comparison).

---

## 3. Task-Specific Evaluation

### 3.1 Knowledge and Reasoning Benchmarks

| Benchmark | Tasks | Format | What It Measures |
|-----------|-------|--------|-----------------|
| **MMLU** | 57 subjects (STEM, humanities, social sciences) | 4-way multiple choice | Breadth of factual knowledge |
| **ARC** | Grade-school science questions | Multiple choice | Scientific reasoning |
| **HellaSwag** | Sentence completion | Multiple choice | Commonsense reasoning |
| **WinoGrande** | Pronoun disambiguation | Binary choice | Commonsense + coreference |
| **TruthfulQA** | Questions that elicit common misconceptions | Open-ended + multiple choice | Truthfulness, resistance to common errors |
| **GSM8K** | Grade-school math word problems | Open-ended (numeric answer) | Multi-step mathematical reasoning |
| **MATH** | Competition-level mathematics | Open-ended (formal answer) | Advanced mathematical reasoning |
| **BBH** (BIG-Bench Hard) | 23 challenging tasks | Mixed | Diverse reasoning capabilities |

These benchmarks evaluate the model after [pretraining](pretraining_objectives.md) and are the primary comparison point in the [scaling laws](../scaling/scaling_overview.md) literature. Performance on many of these tasks shows [emergent abilities](../scaling/emergent_abilities.md) — sudden jumps in capability at critical model scales.

### 3.2 Code Generation

| Benchmark | Format | Metric |
|-----------|--------|--------|
| **HumanEval** | Python function completion | pass@$k$ (fraction of problems solved in $k$ attempts) |
| **MBPP** | Python function from description | pass@$k$ |
| **SWE-bench** | Real GitHub issues → patches | Resolved rate |

The **pass@$k$** metric samples $k$ completions and checks if any passes the test suite:

$$\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

where $n$ is the total number of samples and $c$ is the number that pass. This accounts for the stochastic nature of sampling (see decoding strategies in [Decoder Architecture](decoder_architecture.md)).

### 3.3 Instruction Following

| Benchmark | Format | Evaluation Method |
|-----------|--------|------------------|
| **MT-Bench** | 80 multi-turn questions across 8 categories | GPT-4 as judge (1–10 score) |
| **AlpacaEval** | 805 instructions | GPT-4 pairwise comparison vs. reference |
| **IFEval** | Verifiable instruction constraints | Automatic (did the output follow formatting rules?) |
| **Arena-Hard** | 500 challenging user queries | GPT-4 pairwise comparison |

These benchmarks evaluate instruction-tuned and aligned models — the output of the [alignment pipeline](../alignment/training_pipeline.md). They use **LLM-as-judge** evaluation (Section 5) rather than reference matching.

### 3.4 Safety and Alignment

| Benchmark | What It Tests |
|-----------|--------------|
| **ToxiGen** | Toxic content generation across demographics |
| **BBQ** | Social bias in question answering |
| **XSTest** | Exaggerated safety refusals (over-refusal) |
| **HarmBench** | Resistance to adversarial attacks |

Safety evaluation is integral to the [alignment](../alignment/alignment_overview.md) process. See [Constitutional AI](../alignment/constitutional.md) for how safety principles are formalized.

---

## 4. Human Evaluation

### 4.1 Evaluation Dimensions

Human evaluation captures qualities that automatic metrics miss. The standard dimensions:

| Dimension | Definition | Discriminative Power |
|-----------|-----------|---------------------|
| **Helpfulness** | Does the response accomplish the user's goal? | High — primary differentiator |
| **Harmlessness** | Is the response free of harmful content? | High for safety-critical applications |
| **Honesty** | Is the response factually accurate and appropriately uncertain? | High — captures hallucination |
| **Coherence** | Is the response logically consistent with the conversation? | Medium — modern LLMs rarely fail here |
| **Fluency** | Is the response grammatically correct and natural? | Low — modern LLMs almost always succeed |

The "HHH" criteria (helpfulness, harmlessness, honesty) from Askell et al. (2021) are the guiding framework for [RLHF](../alignment/rlhf.md) and [reward model](../alignment/reward_modeling.md) training.

### 4.2 Evaluation Protocols

**Likert scale rating**: Annotators rate each response on a 1–5 scale per dimension. Simple to implement but suffers from inter-annotator variance and scale calibration drift.

**Pairwise comparison**: Annotators see two responses (from different models) side by side and select which is better. More reliable than absolute rating because comparative judgments are cognitively easier. Pairwise preferences are aggregated using the Bradley-Terry model:

$$P(y_A \succ y_B \mid x) = \frac{\exp(R_A)}{\exp(R_A) + \exp(R_B)}$$

which is the same model used to train [reward models](../alignment/reward_modeling.md) from human preferences.

**Elo rating**: Platforms like **Chatbot Arena** (Zheng et al., 2023) use crowdsourced pairwise comparisons to compute Elo ratings:

$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}, \quad R_A' = R_A + K(S_A - E_A)$$

where $R_A, R_B$ are current ratings, $E_A$ is the expected win probability, $S_A \in \{0, 0.5, 1\}$ is the outcome, and $K$ is the update step size. As of 2024, Chatbot Arena is widely considered the most reliable public ranking of LLM quality due to its large-scale, blind, user-driven evaluation.

### 4.3 Inter-Annotator Agreement

Human evaluation requires measuring agreement to ensure reliability. Common metrics:

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| Cohen's $\kappa$ | $\kappa = (p_o - p_e)/(1 - p_e)$ | Chance-corrected agreement between 2 raters |
| Fleiss' $\kappa$ | Generalization to $n$ raters | Multi-rater agreement |
| Krippendorff's $\alpha$ | Handles missing data, multiple scales | Most robust general measure |

Typical values for dialogue quality annotation: $\kappa \in [0.4, 0.7]$ (moderate to substantial agreement). Pairwise comparison protocols generally achieve higher agreement than Likert scales.

---

## 5. LLM-as-Judge

### 5.1 Motivation

Human evaluation is expensive (∼$1–5 per evaluation), slow (days to weeks), and difficult to scale. **LLM-as-judge** uses a strong model (typically GPT-4) to evaluate outputs from other models, enabling fast, cheap, and reproducible evaluation.

### 5.2 Approaches

| Approach | Prompt Structure | Output |
|----------|-----------------|--------|
| **Pointwise scoring** | "Rate this response 1–10 on helpfulness" | Single score |
| **Pairwise comparison** | "Which response is better: A or B?" | Winner + explanation |
| **Reference-guided** | "Compare the response to this reference answer" | Score relative to reference |
| **Rubric-based** | "Evaluate using these specific criteria: ..." | Structured scores per criterion |

### 5.3 Reliability

LLM-as-judge correlates well with human evaluation (typically $r > 0.8$ for GPT-4 judging on MT-Bench) but has known biases:

| Bias | Description | Mitigation |
|------|------------|-----------|
| **Position bias** | Prefers the first response in A/B comparisons | Swap positions and average |
| **Verbosity bias** | Prefers longer, more detailed responses | Include length control in rubric |
| **Self-preference** | GPT-4 may prefer GPT-4 outputs | Use diverse judges |
| **Style over substance** | Prefers well-formatted outputs regardless of accuracy | Include factuality criteria explicitly |

### 5.4 Best Practices

- Use **pairwise comparison** (more reliable than pointwise scoring)
- **Swap positions** (present A-B and B-A, require consistent judgment)
- Provide a **detailed rubric** with specific criteria
- Use **multiple judge models** when possible
- **Calibrate** against human judgments on a held-out subset
- Report **confidence intervals** from bootstrap resampling

---

## 6. RAG Evaluation

[Retrieval-augmented generation](../rag/rag_overview.md) requires evaluating both the retrieval component and the generation component:

### 6.1 Retrieval Metrics

| Metric | Definition | What It Measures |
|--------|-----------|-----------------|
| Recall@$k$ | Fraction of relevant documents in top-$k$ results | Coverage |
| MRR | $1/\text{rank of first relevant result}$ | Ranking quality |
| NDCG@$k$ | Normalized discounted cumulative gain | Graded relevance |

### 6.2 End-to-End RAG Metrics

| Metric | Definition |
|--------|-----------|
| **Faithfulness** | Is the answer supported by the retrieved documents? |
| **Answer relevance** | Does the answer address the original question? |
| **Context relevance** | Are the retrieved passages relevant to the question? |

The RAGAS framework (Shahul et al., 2023) provides automated evaluation of these dimensions using LLM-as-judge. For full treatment, see [RAG Evaluation](../rag/evaluation.md).

---

## 7. Production Metrics

In deployed systems, model quality is ultimately measured by user behavior:

### 7.1 Explicit Feedback

| Signal | Collection Method | Strength |
|--------|------------------|----------|
| Thumbs up/down | In-interface buttons | Direct but sparse; binary |
| Star ratings | Post-interaction survey | More granular; low response rate |
| Free-text feedback | Comment boxes | Rich but unstructured |
| Escalation to human | Transfer requests | Strong negative signal |

### 7.2 Implicit Feedback

| Signal | What It Indicates |
|--------|------------------|
| Session length | Engagement (but ambiguous: could indicate confusion) |
| Return rate | Overall satisfaction |
| Task completion rate | Functional quality |
| Regeneration rate | Dissatisfaction with specific responses |
| Copy/paste rate | Perceived output utility |
| Abandonment rate | Failure to engage |

### 7.3 Feedback Integration

Production feedback drives model improvement through:

- **[Reward model](../alignment/reward_modeling.md) updates**: Real user preferences supplement annotator data
- **Active learning**: Prioritize uncertain or low-rated interactions for human review
- **Continued fine-tuning**: Retrain on newly collected high-quality interactions
- **Failure analysis**: Systematic review of low-rated responses to identify patterns

---

## 8. Evaluation Strategy by Use Case

| Use Case | Primary Metrics | Secondary Metrics | Reference |
|----------|----------------|-------------------|-----------|
| Pretraining validation | Perplexity, BPB | MMLU (zero-shot), HellaSwag | [Pretraining Objectives](pretraining_objectives.md) |
| Scaling law analysis | Perplexity, BPB | Downstream benchmarks | [Scaling Laws](../scaling/scaling_overview.md) |
| Instruction tuning | MT-Bench, AlpacaEval | IFEval, Arena-Hard | [Prompting](../prompting/prompting_overview.md) |
| Alignment | Chatbot Arena Elo, human pairwise | ToxiGen, XSTest, TruthfulQA | [Alignment](../alignment/alignment_overview.md) |
| Code generation | HumanEval pass@$k$, SWE-bench | MBPP | [LLM Applications](../agents/llm_applications.md) |
| RAG | RAGAS (faithfulness, relevance), recall@$k$ | BERTScore, human eval | [RAG Evaluation](../rag/evaluation.md) |
| Production deployment | User satisfaction, task completion | Regeneration rate, retention | [Conversational AI](conversational_ai.md) |

---

## 9. Finance-Specific Evaluation

| Dimension | Metric / Approach | Why It Matters |
|-----------|------------------|----------------|
| Numerical accuracy | Exact match on extracted figures | Financial data errors have material consequences |
| Temporal reasoning | Correct date/period attribution | "Q3 2024 revenue" vs. "Q3 2023 revenue" confusion |
| Source attribution | Citation accuracy against retrieved documents | Auditability for regulatory compliance |
| Hallucination rate | Factual verification against known databases | Critical for advisory and research applications |
| Bias auditing | Demographic parity across customer segments | Fair lending and treatment compliance |
| Latency | End-to-end response time | Real-time trading and client-facing applications |

Financial LLM evaluation should combine standard benchmarks with **domain-specific test suites** covering financial terminology, numerical reasoning, regulatory knowledge, and market-specific factual accuracy.

---

## 10. Key Takeaways

1. **No single metric captures LLM quality**: evaluation requires combining intrinsic metrics (perplexity), task benchmarks (MMLU, HumanEval), and human/LLM-as-judge evaluation.

2. **Perplexity measures modeling quality, not usefulness**: it is essential for pretraining and scaling law analysis but tells us nothing about helpfulness, safety, or alignment.

3. **Reference-based metrics (BLEU, ROUGE) are ill-suited for LLMs**: open-ended generation has many valid outputs, making reference comparison misleading. BERTScore partially addresses this via semantic similarity.

4. **Human pairwise comparison is the gold standard** for overall quality assessment. Elo-based systems like Chatbot Arena provide the most reliable public model rankings.

5. **LLM-as-judge scales evaluation** but requires careful bias mitigation (position swapping, rubric design, multi-judge aggregation). It correlates well with human judgment when properly implemented.

6. **Evaluation strategy should match the use case**: pretraining → perplexity; instruction tuning → MT-Bench; alignment → Chatbot Arena; RAG → faithfulness metrics; production → user behavior signals.

---

## Exercises

### Exercise 1: Perplexity and Bits Per Byte

A model achieves a test perplexity of 8.5 using a BPE tokenizer that averages 1.3 tokens per word and 4.5 bytes per token. Compute (a) the cross-entropy in bits per token, (b) the bits per byte, and (c) the equivalent perplexity for a character-level model (1 byte = 1 token). Why is BPB a fairer comparison metric than perplexity?

### Exercise 2: BERTScore vs. BLEU

Consider the reference "The Federal Reserve raised interest rates by 25 basis points" and two candidate responses: (a) "The Fed increased rates by a quarter percentage point" and (b) "Interest rates basis points Federal Reserve raised 25." Predict which has higher BLEU and which has higher BERTScore. Verify computationally using the `bert_score` and `sacrebleu` Python packages.

### Exercise 3: LLM-as-Judge Bias

Design an experiment to measure position bias in GPT-4 as a judge. Take 50 response pairs, evaluate each in both orderings (A-B and B-A), and compute the disagreement rate. What disagreement rate would you expect from a perfectly unbiased judge?

### Exercise 4: Chatbot Arena Elo

Two models have Elo ratings $R_A = 1200$ and $R_B = 1100$. Compute (a) the expected win probability for each, (b) the updated ratings after model A wins with $K = 32$, and (c) how many consecutive wins model B would need to surpass model A's rating.

### Exercise 5: Financial Evaluation Suite

Design a 20-question evaluation suite for testing an LLM's financial knowledge. Include questions covering: numerical extraction from earnings statements, temporal reasoning about fiscal periods, regulatory knowledge (e.g., Basel III requirements), and market terminology. For each question, specify the evaluation criterion (exact match, semantic equivalence, or factual accuracy).

---

## References

1. Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A Method for Automatic Evaluation of Machine Translation. *Proceedings of the 40th Annual Meeting of the ACL*.
2. Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *Text Summarization Branches Out*.
3. Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. *Proceedings of the 8th International Conference on Learning Representations (ICLR)*.
4. Hendrycks, D., Burns, C., Basart, S., et al. (2021). Measuring Massive Multitask Language Understanding. *Proceedings of the 9th International Conference on Learning Representations (ICLR)*.
5. Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems (NeurIPS)*.
6. Chen, M., Tworek, J., Jun, H., et al. (2021). Evaluating Large Language Models Trained on Code. *arXiv preprint arXiv:2107.03374*.
7. Askell, A., Bai, Y., Chen, A., et al. (2021). A General Language Assistant as a Laboratory for Alignment. *arXiv preprint arXiv:2112.00861*.
8. Shahul, E., James, J., Thirunavukarasu, A., & Aman, D. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv preprint arXiv:2309.15217*.
9. Li, X., Zhang, T., Dubois, Y., et al. (2023). AlpacaEval: An Automatic Evaluator of Instruction-Following Models. *GitHub Repository*.
