# 29.6.3 Node Embedding

## Overview
Node embedding learns low-dimensional vector representations preserving graph structure. Methods include unsupervised (DeepWalk, Node2Vec, LINE) and GNN-based approaches.

## DeepWalk
Random walks on graph → treat as sentences → Word2Vec (Skip-gram). Captures structural similarity via co-occurrence in random walks.

## Node2Vec
Extends DeepWalk with biased random walks controlled by parameters $p$ (return) and $q$ (in-out):
- Low $q$: BFS-like (captures local structure)
- High $q$: DFS-like (captures global roles)

## LINE
Preserves first-order (direct connections) and second-order (shared neighborhoods) proximity.

## GNN-based Embeddings
Unsupervised GNN training using contrastive losses (DGI, GraphCL) produces powerful node embeddings for downstream tasks.

## Financial Applications
- Embedding assets in latent space for similarity-based portfolio construction
- Embedding accounts for anomaly/fraud detection
