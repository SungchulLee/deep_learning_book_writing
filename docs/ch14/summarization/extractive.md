# Extractive Summarization

## Overview

Extractive summarization selects the most important sentences from the source document to form a summary.

## Sentence Scoring Methods

### TF-IDF Based

Score sentences by the sum of TF-IDF weights of their constituent words:

$$\text{score}(s) = \sum_{w \in s} \text{TF-IDF}(w)$$

### TextRank

Graph-based ranking inspired by PageRank:

1. Build sentence similarity graph (nodes = sentences, edges = cosine similarity)
2. Run PageRank to compute sentence importance
3. Select top-$k$ ranked sentences

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def textrank(sentence_embeddings, num_sentences=3, damping=0.85, max_iter=100):
    """TextRank algorithm for extractive summarization."""
    sim_matrix = cosine_similarity(sentence_embeddings)
    np.fill_diagonal(sim_matrix, 0)
    
    # Normalize
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    sim_matrix = sim_matrix / row_sums
    
    # Power iteration
    n = len(sentence_embeddings)
    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * sim_matrix.T @ scores
        if np.abs(new_scores - scores).sum() < 1e-6:
            break
        scores = new_scores
    
    top_indices = scores.argsort()[-num_sentences:]
    return sorted(top_indices)  # Maintain document order
```

### BERT-based Extraction

Use BERT embeddings for sentence scoring:

```python
from transformers import AutoModel, AutoTokenizer
import torch

class BertExtractiveSummarizer:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_sentence_embeddings(self, sentences):
        embeddings = []
        for sent in sentences:
            inputs = self.tokenizer(sent, return_tensors="pt", 
                                     truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze())
        return torch.stack(embeddings)
```

## Redundancy Handling

### Maximal Marginal Relevance (MMR)

Balance relevance and diversity:

$$\text{MMR} = \arg\max_{s_i \in D \setminus S} \left[\lambda \cdot \text{Rel}(s_i) - (1 - \lambda) \cdot \max_{s_j \in S} \text{Sim}(s_i, s_j)\right]$$

## Summary

1. Extractive methods are simpler and preserve factual accuracy
2. TextRank provides an unsupervised graph-based approach
3. BERT embeddings improve sentence scoring quality
4. MMR addresses redundancy in multi-sentence extraction
