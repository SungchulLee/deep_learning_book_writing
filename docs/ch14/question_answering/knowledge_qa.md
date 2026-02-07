# Knowledge-Based QA

## Overview

Knowledge-Based QA (KBQA) answers questions by querying structured knowledge bases like Wikidata, Freebase, or domain-specific KBs.

## Approaches

### Semantic Parsing

Convert natural language questions to structured queries:

"Who founded Apple?" → `SELECT ?x WHERE { ?x founded Apple_Inc }`

### Embedding-Based

Embed questions and KB elements in shared vector space:

$$\text{score}(q, (s, r, o)) = f(\mathbf{q}, \mathbf{s}, \mathbf{r}, \mathbf{o})$$

### Hybrid

Combine KB lookup with text-based QA for questions requiring both structured and unstructured knowledge.

## Financial KBQA

Query financial knowledge graphs:
- "What is Apple's market cap?" → KB lookup
- "Who are Tesla's board members?" → Entity + relation query
- "Which companies did Berkshire acquire in 2023?" → Temporal + relation query

## Summary

1. KBQA leverages structured knowledge for precise factual answers
2. Semantic parsing translates questions to executable queries
3. Financial KBs enable automated analysis of company relationships and metrics
