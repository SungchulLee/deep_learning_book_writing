# 6.11.7 Evaluation and Metrics

Evaluating conversational AI requires both **automatic metrics** that scale to large test sets and **human evaluation** that captures nuanced quality dimensions. This section formalizes the key metrics and discusses their strengths and limitations.

## Automatic Metrics

### Perplexity

**Perplexity (PPL)** is the most widely used intrinsic metric for language models. It measures how well the model predicts a held-out test set and is defined as the exponentiated average negative log-likelihood:

$$
\text{PPL} = \exp\!\left( -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) \right)
$$

where $T$ is the total number of tokens in the test set. Equivalently, perplexity equals $2^{H}$ where $H$ is the cross-entropy in bits per token.

**Interpretation.** Perplexity can be understood as the effective vocabulary size the model is "choosing from" at each step. A perplexity of 20 means the model is, on average, as uncertain as if it were choosing uniformly among 20 equally likely tokens. Lower perplexity indicates better predictive performance.

**Limitations.** Perplexity measures language modeling quality but does not directly capture conversational qualities like helpfulness, relevance, or safety. A model with low perplexity may still produce unhelpful or harmful responses — it simply means the model assigns high probability to the test data.

### BLEU Score

**BLEU** (Bilingual Evaluation Understudy; Papineni et al., 2002) measures the $n$-gram overlap between a generated response $\hat{y}$ and a reference response $y$:

$$
\text{BLEU} = \text{BP} \cdot \exp\!\left( \sum_{n=1}^{N} w_n \log p_n \right)
$$

where $p_n$ is the modified $n$-gram precision (counting each $n$-gram at most as many times as it appears in the reference), $w_n = 1/N$ are uniform weights, and BP is the **brevity penalty** that discourages overly short outputs:

$$
\text{BP} = \begin{cases} 1 & \text{if } |\hat{y}| \geq |y| \\ \exp\!\big(1 - |y| / |\hat{y}|\big) & \text{if } |\hat{y}| < |y| \end{cases}
$$

**Limitations.** BLEU was designed for machine translation where there are well-defined reference outputs. In open-ended dialogue, many valid responses exist for a single prompt, making reference-based metrics inherently limited. BLEU also ignores semantic similarity — it measures surface-level $n$-gram overlap rather than meaning.

### BERTScore

**BERTScore** (Zhang et al., 2020) addresses the semantic limitation of BLEU by computing similarity in a learned embedding space. Given the generated token embeddings $\{\hat{\mathbf{e}}_i\}$ and reference token embeddings $\{\mathbf{e}_j\}$ from a pre-trained model (typically BERT), the precision and recall are:

$$
P_{\text{BERT}} = \frac{1}{|\hat{y}|} \sum_{i} \max_{j} \cos(\hat{\mathbf{e}}_i, \mathbf{e}_j), \quad R_{\text{BERT}} = \frac{1}{|y|} \sum_{j} \max_{i} \cos(\hat{\mathbf{e}}_i, \mathbf{e}_j)
$$

The F1 score combines both:

$$
F_{\text{BERT}} = 2 \cdot \frac{P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}
$$

BERTScore correlates more strongly with human judgments than BLEU for dialogue evaluation, because semantically similar but lexically different responses receive high scores.

### ROUGE

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation; Lin, 2004) emphasizes recall rather than precision. The most common variant, ROUGE-L, uses the **longest common subsequence (LCS)**:

$$
\text{ROUGE-L} = \frac{(1 + \beta^2) \cdot R_{\text{lcs}} \cdot P_{\text{lcs}}}{R_{\text{lcs}} + \beta^2 \cdot P_{\text{lcs}}}
$$

where $R_{\text{lcs}} = \text{LCS}(\hat{y}, y) / |y|$ and $P_{\text{lcs}} = \text{LCS}(\hat{y}, y) / |\hat{y}|$. ROUGE is most useful for evaluating summarization-like tasks rather than open-ended dialogue.

## Human Evaluation

Automatic metrics capture only surface-level quality. Comprehensive evaluation of conversational AI requires human judgment along multiple dimensions.

### Evaluation Dimensions

The standard dimensions for human evaluation of dialogue systems are:

**Relevance.** Does the response directly address the user's query? Relevance failures include off-topic responses, misinterpretation of the question, or generic answers that do not engage with the specific content of the prompt.

**Coherence.** Is the response internally consistent and logically connected to the conversation history? Coherence failures include contradicting earlier statements, introducing non-sequiturs, or losing track of the conversational thread.

**Fluency.** Is the response grammatically correct and natural-sounding? Modern LLMs like ChatGPT rarely have fluency issues, so this dimension is less discriminative than for earlier systems.

**Helpfulness.** Does the response actually help the user accomplish their goal? This is the most holistic dimension and subsumes relevance, accuracy, and actionability.

**Safety.** Does the response avoid generating harmful, biased, or misleading content? Safety evaluation is typically performed through **red-teaming** — adversarial testing designed to elicit failure modes.

### Evaluation Protocols

**Likert scale rating.** Annotators rate each response on a 1–5 scale for each dimension. This is simple to implement but suffers from inter-annotator variance and scale calibration issues.

**Pairwise comparison.** Annotators are shown two responses (from different models or model versions) and asked which is better. This is more reliable than absolute rating because comparative judgments are easier for humans than absolute scoring. The Bradley-Terry model (see [Section 6.11.4](training_process.md)) can aggregate pairwise comparisons into global rankings.

**Elo rating systems.** Inspired by chess ratings, platforms like Chatbot Arena (Zheng et al., 2023) use crowdsourced pairwise comparisons to compute Elo ratings for different models. Each comparison updates the rating:

$$
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}, \quad R_A' = R_A + K(S_A - E_A)
$$

where $R_A, R_B$ are current ratings, $E_A$ is the expected win probability, $S_A \in \{0, 0.5, 1\}$ is the outcome, and $K$ is the update step size. This approach produces robust model rankings from noisy pairwise data.

## User Feedback and Continuous Improvement

### Feedback Collection

In production deployments, feedback is gathered through:

- **Explicit signals.** Thumbs up/down ratings, satisfaction surveys, escalation-to-human requests.
- **Implicit signals.** Session length, return visits, task completion rates, abandonment rates.
- **Error reports.** User-flagged incorrect or inappropriate responses.

### Feedback Integration

Collected feedback drives model improvement through several mechanisms:

**Active learning.** Identifying interactions where the model is most uncertain or where user feedback indicates failure, and prioritizing these for human annotation and inclusion in future training data.

**Online RLHF.** Incorporating real user preference signals into reward model updates, enabling the model to adapt to the actual distribution of user queries and preferences rather than relying solely on initial annotator data.

**Model retraining.** Periodic full retraining or continued fine-tuning on newly collected data to address emerging topics, changing user expectations, and identified failure modes.

!!! tip "Evaluation Best Practice"
    No single metric captures all aspects of conversational quality. Best practice combines automatic metrics (perplexity for language modeling quality, BERTScore for semantic accuracy) with structured human evaluation (pairwise comparisons on relevance, helpfulness, and safety) and production metrics (user satisfaction, task completion, retention).

---

**Next:** [6.11.8 Future Directions](future_directions.md)
