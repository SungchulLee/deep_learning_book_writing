# 29.7.5 Knowledge Graphs

## Overview
Knowledge graphs store factual knowledge as (head, relation, tail) triples. GNNs learn entity and relation embeddings for link prediction (knowledge graph completion), entity classification, and question answering.

## Methods
- **R-GCN**: Relational GCN with per-relation weight matrices
- **CompGCN**: Composition-based GNN jointly learning entity and relation embeddings
- **KBGAT**: Knowledge base GAT with relation-aware attention

## Scoring Functions
- **TransE**: $\|\mathbf{h} + \mathbf{r} - \mathbf{t}\|$
- **DistMult**: $\mathbf{h}^T \text{diag}(\mathbf{r}) \mathbf{t}$
- **RotatE**: $\|\mathbf{h} \circ \mathbf{r} - \mathbf{t}\|$ (rotation in complex space)

## Financial Applications
- **Company knowledge graphs**: Industry, supplier, competitor relationships
- **Regulatory knowledge**: Rules, requirements, entity obligations
- **Event prediction**: Predict corporate events from KG structure
