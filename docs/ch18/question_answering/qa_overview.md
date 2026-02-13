# Question Answering Overview

## Learning Objectives

- Understand the taxonomy of QA tasks
- Distinguish between extractive, abstractive, and generative QA
- Identify financial QA applications

## Task Taxonomy

Question answering systems take a natural language question and return an answer. The task varies significantly depending on the answer source and format:

### By Answer Format

| Type | Answer Source | Example |
|------|--------------|---------|
| Extractive | Span from context | "Who founded Tesla?" -> "Elon Musk" (from passage) |
| Abstractive | Generated text | "Summarize the quarterly results" -> novel sentence |
| Boolean | Yes/No | "Did revenue increase?" -> "Yes" |
| List | Multiple items | "Name the board members" -> [list] |
| Numerical | Computed value | "What is the P/E ratio?" -> "25.3" |

### By Knowledge Source

| Type | Source | Challenge |
|------|--------|-----------|
| Reading Comprehension | Given passage | Understanding & reasoning |
| Open-Domain | Large corpus | Retrieval + comprehension |
| Knowledge-Based | Structured KB | Semantic parsing |
| Conversational | Dialog history | Context tracking |

### By Reasoning Complexity

- **Single-hop**: Answer found in one passage
- **Multi-hop**: Requires reasoning across multiple documents
- **Discrete reasoning**: Requires counting, sorting, arithmetic (e.g., DROP)
- **Commonsense**: Requires world knowledge not stated in text

## General Architecture

Most modern QA systems follow a retriever-reader pattern:

```
Question
  → Retriever (find relevant passages)
    → Reader (extract or generate answer)
      → Answer
```

For reading comprehension tasks, the retriever is unnecessary since the context is provided.

## Financial QA Applications

- **Earnings call QA**: "What was the gross margin this quarter?"
- **SEC filing analysis**: "What are the key risk factors mentioned?"
- **Market research**: "Which companies are expanding in Southeast Asia?"
- **Compliance**: "Does this contract contain a change-of-control clause?"

## Evaluation

- **Exact Match (EM)**: Binary — predicted answer exactly matches gold answer
- **F1 Score**: Token-level overlap between predicted and gold answer
- **ROUGE-L**: Longest common subsequence overlap (for abstractive QA)

## References

1. Rajpurkar, P., et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. *EMNLP*.
2. Chen, D., et al. (2017). Reading Wikipedia to Answer Open-Domain Questions. *ACL*.
3. Zhu, F., et al. (2021). Retrieving and Reading: A Comprehensive Survey on Open-Domain QA. *arXiv*.
