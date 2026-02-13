# Open Information Extraction

## Learning Objectives

- Understand schema-free triple extraction
- Compare Open IE with closed IE approaches
- Apply Open IE for exploratory text mining

## Motivation

Closed IE requires predefined relation schemas — a bottleneck when processing diverse text. Open IE extracts arbitrary (subject, relation, object) triples without a fixed ontology.

### Example

Input: *"Einstein was born in Ulm and developed the theory of relativity while working at the patent office."*

Open IE output:

- (Einstein, was born in, Ulm)
- (Einstein, developed, the theory of relativity)
- (Einstein, working at, the patent office)

## Key Systems

### ReVerb (Fader et al., 2011)

Uses syntactic constraints on relation phrases. The relation must match patterns like V (verb), V P (verb-preposition), or V W* P (verb-words-preposition). A lexical constraint requires the relation phrase to appear frequently enough to filter noise.

### OLLIE (Mausam et al., 2012)

Extends ReVerb by learning extraction patterns from dependency parses and handling context clauses (attribution, conditionals).

### Stanford OpenIE (Angeli et al., 2015)

Splits complex sentences into short, atomic clauses using natural logic. For instance, *"Born in Ulm, Einstein developed relativity"* becomes two independent statements: *"Einstein was born in Ulm"* and *"Einstein developed relativity"*.

### Neural Open IE

Modern approaches use seq2seq models to generate triples directly from text:

```python
# Using a generative model for Open IE
prompt = (
    "Extract all (subject, relation, object) triples from:\n"
    "Apple acquired Beats for $3B in 2014.\n"
    "Triples:"
)
# Expected output:
# (Apple, acquired, Beats)
# (Apple, acquired Beats for, $3B)
# (acquisition, occurred in, 2014)
```

## Comparison with Closed IE

| Aspect | Closed IE | Open IE |
|--------|-----------|---------|
| Relations | Fixed schema | Any expressed relation |
| Training | Labeled examples | Syntactic patterns / self-supervised |
| Precision | Higher | Lower (noisy extractions) |
| Recall | Limited to schema | Broader coverage |
| Use case | Structured DB population | Exploratory analysis |

## Challenges

1. **Uninformative extractions**: (He, is, good) — too vague to be useful
2. **Overly specific relations**: Long verb phrases reduce generalizability
3. **Implicit relations**: Not all relations have explicit verbal expression
4. **Nested extractions**: Complex sentences may require recursive extraction

## Applications in Finance

Open IE enables exploratory mining of financial text where relation schemas are incomplete: discovering novel supply chain relationships from news, extracting management commentary patterns from earnings calls, and mining regulatory filings for undisclosed business relationships.

## References

1. Fader, A., Soderland, S., & Etzioni, O. (2011). Identifying Relations for Open Information Extraction. *EMNLP*.
2. Mausam, et al. (2012). Open Language Learning for Information Extraction. *EMNLP-CoNLL*.
3. Angeli, G., Premkumar, M. J., & Manning, C. D. (2015). Leveraging Linguistic Structure for Open Domain IE. *ACL*.
