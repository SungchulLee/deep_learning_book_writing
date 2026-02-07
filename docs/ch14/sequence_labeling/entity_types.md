# Entity Types and Taxonomies

## Learning Objectives

- Understand standard NER entity taxonomies across domains
- Compare CoNLL, OntoNotes, and domain-specific type systems
- Design entity taxonomies for new domains including finance
- Handle hierarchical and fine-grained entity typing

---

## Standard Taxonomies

### CoNLL-2003 (4 Types)

The foundational benchmark taxonomy:

| Type | Code | Examples |
|------|------|----------|
| Person | PER | "Barack Obama", "Marie Curie" |
| Organization | ORG | "Apple Inc.", "United Nations" |
| Location | LOC | "Paris", "Mount Everest" |
| Miscellaneous | MISC | "World Cup", "Nobel Prize" |

### OntoNotes 5.0 (18 Types)

A richer taxonomy separating named, numerical, and temporal entities:

**Named Entities**: PERSON, NORP (nationalities/religious/political groups), FAC (facilities), ORG, GPE (geo-political entities), LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE

**Numerical Entities**: DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL

### ACE (Automatic Content Extraction)

Seven entity types with subtypes: PER, ORG, GPE, LOC, FAC, WEA (weapon), VEH (vehicle). Each type has further subtypes (e.g., PER.Individual, PER.Group).

---

## Domain-Specific Taxonomies

### Biomedical NER

| Type | Examples | Datasets |
|------|----------|----------|
| Gene/Protein | BRCA1, p53, insulin | JNLPBA, BioCreative |
| Disease | diabetes, COVID-19 | NCBI Disease |
| Drug/Chemical | aspirin, metformin | BC5CDR |
| Species | *E. coli*, human | LINNAEUS |
| Cell Type | T-cell, neuron | JNLPBA |

### Financial NER

| Type | Examples | Use Case |
|------|----------|----------|
| Company | "Goldman Sachs", "AAPL" | Entity linking to tickers |
| Financial Instrument | "10-year Treasury", "S&P 500" | Portfolio analysis |
| Monetary Amount | "$2.3 billion", "€50M" | Earnings extraction |
| Economic Indicator | "GDP", "CPI", "unemployment rate" | Macro analysis |
| Date/Period | "Q3 2024", "fiscal year" | Temporal alignment |
| Regulatory Body | "SEC", "Fed", "ECB" | Compliance monitoring |

### Legal NER

Case names, courts, statutes, jurisdictions, parties, judges, and legal citations.

---

## Hierarchical Entity Typing

Fine-grained entity typing assigns entities to nodes in a type hierarchy:

```
Entity
├── Person
│   ├── Politician
│   ├── Athlete
│   ├── Scientist
│   └── Artist
├── Organization
│   ├── Company
│   │   ├── Tech Company
│   │   └── Financial Institution
│   ├── Government Agency
│   └── Educational Institution
└── Location
    ├── Country
    ├── City
    └── Natural Feature
```

### Mathematical Formulation

Given entity mention $m$ with context $c$, predict a set of types $\mathcal{T}_m \subseteq \mathcal{T}$:

$$P(\mathcal{T}_m | m, c) = \prod_{t \in \mathcal{T}} P(t \in \mathcal{T}_m | m, c)$$

Subject to hierarchical consistency: if $t \in \mathcal{T}_m$ and $t'$ is an ancestor of $t$, then $t' \in \mathcal{T}_m$.

---

## Designing Custom Taxonomies

### Guidelines

1. **Mutual exclusivity**: Minimize type overlap at the same hierarchy level
2. **Coverage**: Ensure all entities of interest map to at least one type
3. **Granularity balance**: Too fine-grained increases annotation cost; too coarse reduces utility
4. **Downstream alignment**: Design types that serve the target application
5. **Annotation feasibility**: Types must be distinguishable by human annotators

### Inter-Annotator Agreement

Measure taxonomy quality via Cohen's $\kappa$ or Fleiss' $\kappa$ on pilot annotations:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where $p_o$ is observed agreement and $p_e$ is expected agreement by chance. Target $\kappa > 0.8$ for reliable annotation.

---

## Summary

1. **CoNLL-2003** provides the standard 4-type benchmark for NER
2. **OntoNotes** extends to 18 types including numerical and temporal entities
3. **Domain-specific taxonomies** are essential for biomedical, financial, and legal NLP
4. **Hierarchical typing** enables fine-grained entity classification
5. **Taxonomy design** must balance granularity, coverage, and annotation feasibility

---

## References

1. Tjong Kim Sang, E. F., & De Meulder, F. (2003). CoNLL-2003 Shared Task. *CoNLL*.
2. Weischedel, R., et al. (2013). OntoNotes Release 5.0. LDC.
3. Ling, X., & Weld, D. S. (2012). Fine-Grained Entity Recognition. *AAAI*.
4. Alvarado, J., et al. (2015). Domain-Specific Named Entity Recognition in Financial Texts. *ACL FinNLP Workshop*.
