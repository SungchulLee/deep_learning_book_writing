# Summarization Evaluation

## ROUGE Metrics

Recall-Oriented Understudy for Gisting Evaluation (Lin, 2004):

### ROUGE-N

N-gram overlap between generated and reference summaries:

$$\text{ROUGE-N} = \frac{\sum_{s \in \text{Ref}} \sum_{\text{gram}_n \in s} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{s \in \text{Ref}} \sum_{\text{gram}_n \in s} \text{Count}(\text{gram}_n)}$$

- **ROUGE-1**: Unigram overlap (content coverage)
- **ROUGE-2**: Bigram overlap (fluency indicator)

### ROUGE-L

Longest Common Subsequence (LCS):

$$\text{ROUGE-L} = F_\beta = \frac{(1 + \beta^2) \cdot R_{lcs} \cdot P_{lcs}}{R_{lcs} + \beta^2 \cdot P_{lcs}}$$

## Computation

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

scores = scorer.score(
    "The cat sat on the mat.",
    "A cat was sitting on a mat."
)
for metric, score in scores.items():
    print(f"{metric}: P={score.precision:.3f} R={score.recall:.3f} F={score.fmeasure:.3f}")
```

## Beyond ROUGE

| Metric | Measures | Correlation with Human |
|--------|----------|----------------------|
| ROUGE | N-gram overlap | Moderate |
| BERTScore | Semantic similarity | High |
| BARTScore | Generation likelihood | High |
| QuestEval | QA-based factual consistency | High |
| SummaC | NLI-based faithfulness | High |

## Faithfulness Evaluation

Measure whether the summary is factually consistent with the source:

1. **NLI-based**: Check if source entails summary sentences
2. **QA-based**: Generate questions from summary, verify answers match source
3. **Fact extraction**: Extract triples from both, compare overlap

## Summary

1. ROUGE is the standard automatic metric but has known limitations
2. ROUGE-1 measures content coverage; ROUGE-2 indicates fluency
3. Semantic metrics (BERTScore) and faithfulness metrics (SummaC) provide complementary evaluation
4. Human evaluation remains essential for assessing summary quality
