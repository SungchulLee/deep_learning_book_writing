# Multi-Document Summarization

## Overview

Multi-document summarization (MDS) produces a unified summary from multiple related documents, requiring cross-document alignment and redundancy handling.

## Challenges

1. **Redundancy**: Same information repeated across documents
2. **Contradiction**: Conflicting information between sources
3. **Coherence**: Organizing information from multiple sources into coherent output
4. **Coverage**: Capturing diverse perspectives

## Approaches

### Cluster-then-Summarize

1. Cluster similar sentences across documents
2. Select representative sentences from each cluster
3. Order and generate final summary

### Hierarchical

```
Documents → Per-document encoding → Cross-document attention → Summary generation
```

### Graph-Based

Build a cross-document graph connecting related entities, events, and facts, then perform graph-based selection or generation.

## Datasets

| Dataset | Domain | Documents/Cluster |
|---------|--------|------------------|
| Multi-News | News | 2-10 |
| DUC 2004 | News | 10 |
| WCEP | News events | 100+ |

## Financial Applications

Multi-document summarization for:
- Aggregating analyst reports on the same company
- Summarizing news coverage of a corporate event
- Combining quarterly earnings across periods for trend analysis

## Summary

1. MDS must handle redundancy, contradiction, and cross-document coherence
2. Cluster-then-summarize is a simple but effective baseline
3. Financial MDS enables comprehensive analysis from multiple sources
