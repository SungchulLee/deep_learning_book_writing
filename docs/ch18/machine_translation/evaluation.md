# Machine Translation Evaluation

## Learning Objectives

- Understand BLEU score computation and its limitations
- Compare string-based and neural evaluation metrics
- Select appropriate metrics for different MT evaluation scenarios

## Why Evaluation Is Hard

Translation is inherently many-to-one: multiple valid translations exist for any source sentence. Evaluation must account for paraphrases, word order variation, and stylistic choices.

## BLEU (Bilingual Evaluation Understudy)

The most widely used MT metric (Papineni et al., 2002). BLEU measures n-gram overlap between the system output and reference translations.

### Modified n-gram Precision

$$p_n = \frac{\sum_{\text{n-gram} \in \hat{\mathbf{y}}} \min\left(\text{Count}(\text{n-gram}, \hat{\mathbf{y}}), \max_r \text{Count}(\text{n-gram}, \mathbf{y}_r)\right)}{\sum_{\text{n-gram} \in \hat{\mathbf{y}}} \text{Count}(\text{n-gram}, \hat{\mathbf{y}})}$$

The clipping prevents inflating precision by repeating common words.

### Brevity Penalty

BLEU penalizes overly short translations:

$$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ \exp(1 - r/c) & \text{if } c \leq r \end{cases}$$

where $c$ is the total output length and $r$ is the effective reference length.

### Final Score

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Typically $N=4$ with uniform weights $w_n = 1/4$, yielding BLEU-4.

### BLEU Limitations

- No credit for synonyms or paraphrases
- Sensitive to tokenization choices
- Corpus-level metric (unreliable at sentence level)
- Does not distinguish between content and function word errors

## METEOR

Addresses BLEU limitations by incorporating:

- **Exact match**: Same as BLEU
- **Stem match**: "running" matches "runs"
- **Synonym match**: "big" matches "large" (via WordNet)
- **Paraphrase match**: Phrase-level paraphrase tables

METEOR also penalizes fragmentation — preferring contiguous matches.

## Character-Level Metrics

### chrF

Character n-gram F-score, robust across languages (especially agglutinative ones like Finnish, Turkish):

$$\text{chrF}_\beta = (1 + \beta^2) \frac{\text{chrP} \cdot \text{chrR}}{\beta^2 \cdot \text{chrP} + \text{chrR}}$$

## Neural Evaluation Metrics

### BERTScore

Uses contextualized BERT embeddings to compute semantic similarity:

$$\text{BERTScore} = \frac{1}{|\hat{\mathbf{y}}|} \sum_{\hat{y}_i \in \hat{\mathbf{y}}} \max_{y_j \in \mathbf{y}} \cos(\mathbf{h}_{\hat{y}_i}, \mathbf{h}_{y_j})$$

### COMET

Trained on human quality judgments (DA scores), COMET takes source, hypothesis, and reference as input. It correlates significantly better with human judgment than string-based metrics.

```python
from comet import download_model, load_from_checkpoint

model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
data = [{"src": "Hello world", "mt": "Bonjour le monde", "ref": "Bonjour le monde"}]
scores = model.predict(data, batch_size=8)
```

## Metric Comparison

| Metric | Correlation with Human | Handles Paraphrases | Speed |
|--------|----------------------|---------------------|-------|
| BLEU | Moderate | No | Fast |
| METEOR | Good | Partially | Moderate |
| chrF | Good | No | Fast |
| BERTScore | High | Yes | Slow |
| COMET | Highest | Yes | Slow |

## Best Practices

1. Report BLEU for comparability with prior work
2. Include at least one neural metric (COMET or BERTScore) for reliability
3. Use SacreBLEU for reproducible BLEU computation
4. Always specify tokenization scheme and number of references

## References

1. Papineni, K., et al. (2002). BLEU: A Method for Automatic Evaluation of MT. *ACL*.
2. Banerjee, S., & Lavie, A. (2005). METEOR: An Automatic Metric for MT Evaluation. *ACL Workshop*.
3. Rei, R., et al. (2022). COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task. *WMT*.
