# Open-Domain QA

## Overview

Open-domain QA answers questions without a pre-specified context document, requiring the system to find relevant evidence from a large corpus (e.g., Wikipedia).

## Architecture: Retriever-Reader

```
Question → Retriever → Top-k passages → Reader → Answer
```

### Dense Passage Retrieval (DPR)

$$\text{sim}(q, p) = \mathbf{E}_Q(q)^T \mathbf{E}_P(p)$$

Encode questions and passages independently with BERT, then use dot-product similarity for retrieval.

```python
from transformers import DPRQuestionEncoder, DPRContextEncoder

q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
p_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
```

### Reader

Apply extractive or abstractive QA model to each retrieved passage, then aggregate answers.

## Key Systems

| System | Retriever | Reader |
|--------|-----------|--------|
| DrQA | TF-IDF | BiLSTM |
| DPR + Reader | Dense (BERT) | BERT extractive |
| RAG | Dense (DPR) | BART generative |
| FiD | Dense (DPR) | T5 (all passages concatenated) |

## Summary

1. Open-domain QA requires both retrieval and comprehension
2. Dense retrieval (DPR) outperforms sparse retrieval (TF-IDF/BM25) for most tasks
3. Fusion-in-Decoder (FiD) processes multiple passages jointly for better answer synthesis
