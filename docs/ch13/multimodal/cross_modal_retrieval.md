# Cross-Modal Retrieval

## Learning Objectives

By the end of this section, you will be able to:

- Understand image-to-text and text-to-image retrieval tasks
- Implement efficient retrieval using pre-computed embeddings
- Evaluate retrieval performance with Recall@K metrics

## Problem Definition

Cross-modal retrieval finds relevant items across modalities:

- **Image → Text**: Given an image, find matching captions
- **Text → Image**: Given a text query, find matching images

Both tasks require a shared embedding space where semantically related images and texts are close.

## Retrieval with Dual Encoders

Models like CLIP and ALIGN produce embeddings that enable efficient retrieval:

```python
import torch
import torch.nn.functional as F

def retrieve(query_embedding, gallery_embeddings, top_k=10):
    """
    Retrieve top-k most similar items from gallery.
    
    Args:
        query_embedding: (D,) normalized query vector
        gallery_embeddings: (N, D) normalized gallery matrix
        top_k: Number of results to return
    """
    similarities = gallery_embeddings @ query_embedding  # (N,)
    top_indices = similarities.topk(top_k).indices
    return top_indices, similarities[top_indices]
```

## Evaluation Metrics

### Recall@K

$$\text{R@K} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\text{correct match in top-}K\text{ results}]$$

Standard evaluation reports R@1, R@5, R@10 for both image→text and text→image directions.

### Median Rank

The median position of the first correct match across all queries. Lower is better.

## Key Benchmarks

| Dataset | Images | Captions/Image | Test Split |
|---------|--------|----------------|-----------|
| Flickr30k | 31K | 5 | 1K |
| MSCOCO | 123K | 5 | 5K |

## References

1. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML.
2. Faghri, F., et al. (2018). VSE++: Improving Visual-Semantic Embeddings with Hard Negatives. BMVC.
