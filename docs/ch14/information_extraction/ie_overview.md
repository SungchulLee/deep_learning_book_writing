# Information Extraction Overview

## Learning Objectives

- Understand the IE pipeline and how components interact
- Distinguish between closed and open information extraction
- Identify financial applications of structured knowledge extraction

## What Is Information Extraction?

Information Extraction (IE) transforms unstructured text into structured representations — tables, graphs, and databases that machines can query and reason over. While NLP tasks like classification assign labels to entire documents, IE operates at a finer granularity, identifying specific entities, relationships, and events within text.

### The IE Pipeline

A typical IE system chains several components:

```
Raw Text
  → Named Entity Recognition (NER)
    → Coreference Resolution
      → Relation Extraction
        → Event Extraction
          → Knowledge Graph Construction
```

Each stage builds on the previous:

1. **NER** identifies entity mentions: "Apple", "Tim Cook", "$3 billion"
2. **Coreference Resolution** links mentions to the same entity: "Apple" = "the company" = "it"
3. **Relation Extraction** identifies semantic relationships: (Tim Cook, CEO-of, Apple)
4. **Event Extraction** identifies structured events: Acquisition(buyer=Apple, target=Beats, price=$3B)
5. **Knowledge Graph Construction** aggregates triples into a queryable graph

### Closed vs. Open IE

| Aspect | Closed IE | Open IE |
|--------|-----------|---------|
| Schema | Predefined relation types | No fixed schema |
| Training | Supervised on labeled data | Self-supervised patterns |
| Precision | Higher | Lower |
| Coverage | Limited to known relations | Discovers novel relations |
| Example | "headquartered-in" relation | Any verb phrase as relation |

### Joint vs. Pipeline Models

Pipeline approaches suffer from **error propagation** — NER mistakes cascade into relation extraction errors. Joint models address this by simultaneously predicting entities and relations:

$$\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{NER}} + \lambda \mathcal{L}_{\text{RE}}$$

## Financial Applications

IE is critical in quantitative finance for:

- **Earnings call analysis**: Extract revenue figures, guidance changes, management sentiment
- **News-driven trading signals**: Identify M&A announcements, executive changes, regulatory actions
- **Supply chain mapping**: Build company relationship graphs from SEC filings
- **Risk monitoring**: Track litigation events, sanctions, credit downgrades

### Example: Financial News IE

Input: *"Goldman Sachs announced Thursday it will acquire GreenSky for approximately $2.24 billion in an all-stock deal."*

Extracted structure:

| Component | Value |
|-----------|-------|
| Event Type | Acquisition |
| Buyer | Goldman Sachs |
| Target | GreenSky |
| Price | $2.24 billion |
| Deal Type | All-stock |
| Date | Thursday |

## Evaluation Paradigms

IE evaluation varies by subtask but generally uses:

- **Exact match F1**: Strict boundary and type matching
- **Partial match**: Credit for overlapping spans
- **Slot filling**: Precision/recall on extracted attribute values

## References

1. Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed.). Chapters 17, 21.
2. Sarawagi, S. (2008). Information Extraction. *Foundations and Trends in Databases*, 1(3).
3. Li, J., et al. (2020). A Survey on Deep Learning for Named Entity Recognition. *IEEE TKDE*.
