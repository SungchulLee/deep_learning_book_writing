# RAG Evaluation

## Learning Objectives

- Evaluate retrieval quality with standard IR metrics
- Assess generation quality for RAG systems
- Use RAG-specific evaluation frameworks

## Retrieval Metrics

### Recall@k

Proportion of relevant documents retrieved in top-$k$:

$$\text{Recall@k} = \frac{|\text{Relevant} \cap \text{Retrieved@k}|}{|\text{Relevant}|}$$

### Mean Reciprocal Rank (MRR)

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

where $\text{rank}_i$ is the position of the first relevant document for query $i$.

### NDCG@k (Normalized Discounted Cumulative Gain)

$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}, \quad \text{DCG@k} = \sum_{i=1}^{k} \frac{2^{r_i} - 1}{\log_2(i + 1)}$$

## Generation Metrics

### Faithfulness

Does the answer only contain information from the retrieved context? Measures hallucination.

### Relevance

Does the answer address the question?

### Correctness

Is the answer factually correct?

## RAG-Specific Frameworks

### RAGAS

Evaluates four dimensions without ground truth:

```python
# RAGAS evaluation dimensions
dimensions = {
    "faithfulness": "Is the answer grounded in the context?",
    "answer_relevancy": "Does the answer address the question?",
    "context_precision": "Are retrieved docs relevant?",
    "context_recall": "Are all needed docs retrieved?",
}
```

### Implementation Sketch

```python
def evaluate_rag(
    questions: list,
    ground_truth: list,
    rag_pipeline,
    retrieval_evaluator,
    generation_evaluator,
):
    results = {"retrieval": [], "generation": []}

    for q, gt in zip(questions, ground_truth):
        # Get RAG output
        retrieved_docs = rag_pipeline.retriever.search(q, k=10)
        answer = rag_pipeline(q)

        # Evaluate retrieval
        results["retrieval"].append({
            "recall@5": compute_recall(retrieved_docs[:5], gt["relevant_docs"]),
            "mrr": compute_mrr(retrieved_docs, gt["relevant_docs"]),
        })

        # Evaluate generation
        results["generation"].append({
            "faithfulness": check_faithfulness(answer, retrieved_docs),
            "correctness": compare_to_ground_truth(answer, gt["answer"]),
        })

    return aggregate_results(results)
```

## Financial RAG Evaluation Challenges

- **Numeric precision**: Financial answers require exact numbers
- **Temporal sensitivity**: Answers must reference correct time periods
- **Multi-document synthesis**: Complex queries require combining information
- **Regulatory accuracy**: Compliance queries demand zero hallucination

## References

1. Es, S., et al. (2024). "RAGAS: Automated Evaluation of Retrieval Augmented Generation." *arXiv*.
2. Saad-Falcon, J., et al. (2024). "ARES: An Automated Evaluation Framework for RAG Systems." *arXiv*.
