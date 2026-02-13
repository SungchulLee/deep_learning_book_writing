# Vector Databases

## Learning Objectives

- Understand the role of vector databases in RAG systems
- Compare key vector database technologies
- Implement efficient similarity search with FAISS

## Why Vector Databases?

Dense retrieval requires searching over millions of document embeddings. Exact nearest neighbor search is $O(n)$—too slow for production. Vector databases use **Approximate Nearest Neighbor (ANN)** algorithms for sub-linear search.

## Key Technologies

| Database | Type | ANN Algorithm | Key Feature |
|----------|------|--------------|-------------|
| FAISS | Library | IVF, HNSW, PQ | Facebook, production-proven |
| Pinecone | Managed service | Proprietary | Fully managed, easy API |
| Weaviate | Open source | HNSW | Hybrid search, GraphQL |
| Chroma | Open source | HNSW | Lightweight, good for prototyping |
| Milvus | Open source | IVF, HNSW, DiskANN | Scalable, GPU support |
| Qdrant | Open source | HNSW | Rust-based, filtering |

## ANN Algorithms

### HNSW (Hierarchical Navigable Small World)

Graph-based algorithm with logarithmic search complexity. Builds a multi-layer graph where higher layers contain fewer nodes for coarse navigation.

- **Search complexity**: $O(\log n)$
- **Build complexity**: $O(n \log n)$
- **Memory**: High (stores graph structure)
- **Accuracy**: Very high (>95% recall typical)

### IVF (Inverted File Index)

Partitions the embedding space into $k$ clusters using k-means. At search time, only searches the $n_{\text{probe}}$ nearest clusters.

- **Search complexity**: $O(n_{\text{probe}} \cdot n/k)$
- **Tunable**: Trade recall for speed via $n_{\text{probe}}$

### Product Quantization (PQ)

Compresses embeddings by splitting into subvectors and quantizing each independently. Reduces memory by 10-100x with moderate accuracy loss.

## FAISS Example

```python
import faiss
import numpy as np

# Create index
dimension = 1024
n_documents = 1_000_000

# IVF + PQ for large-scale search
n_clusters = 1024
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFPQ(quantizer, dimension, n_clusters, 64, 8)

# Train on sample data
training_data = np.random.randn(50000, dimension).astype('float32')
index.train(training_data)

# Add document embeddings
doc_embeddings = np.random.randn(n_documents, dimension).astype('float32')
index.add(doc_embeddings)

# Search
index.nprobe = 32  # Search 32 nearest clusters
query = np.random.randn(1, dimension).astype('float32')
distances, indices = index.search(query, k=10)
```

## References

1. Johnson, J., et al. (2019). "Billion-scale Similarity Search with GPUs." *IEEE TBD*.
2. Malkov, Y. & Yashunin, D. (2020). "Efficient and Robust ANN using HNSW Graphs." *TPAMI*.
