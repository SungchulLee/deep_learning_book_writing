# Rule-Based NER

## Learning Objectives

- Implement pattern-based entity recognition using regular expressions and heuristics
- Understand the role of rule-based systems as baselines and hybrid components
- Design rule cascades for specific entity types

---

## Overview

Rule-based NER uses hand-crafted patterns, regular expressions, and heuristics rather than statistical learning. While modern neural approaches dominate benchmarks, rule-based methods remain valuable as baselines, for low-resource domains, and as components in hybrid systems.

---

## Regular Expression Patterns

### Common Entity Patterns

```python
import re
from typing import List, Tuple

# Pattern definitions for common entity types
PATTERNS = {
    'DATE': [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
    ],
    'MONEY': [
        r'\$[\d,]+\.?\d*\s*(?:million|billion|trillion|M|B|T)?\b',
        r'\b\d+\.?\d*\s*(?:USD|EUR|GBP|JPY)\b',
    ],
    'PERCENT': [
        r'\b\d+\.?\d*\s*%',
        r'\b\d+\.?\d*\s*(?:percent|basis points|bps)\b',
    ],
    'EMAIL': [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    ],
}

def rule_based_ner(text: str) -> List[Tuple[str, str, int, int]]:
    """Extract entities using regex patterns."""
    entities = []
    for entity_type, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append((
                    match.group(), entity_type,
                    match.start(), match.end()
                ))
    return sorted(entities, key=lambda x: x[2])
```

---

## Contextual Rules

Beyond pattern matching, contextual rules use surrounding words:

```python
PERSON_TITLES = {"Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "CEO", "President"}
ORG_SUFFIXES = {"Inc.", "Corp.", "Ltd.", "LLC", "Co.", "Group", "Bank"}

def apply_context_rules(tokens, pos_tags=None):
    """Apply contextual rules for entity recognition."""
    entities = []
    for i, token in enumerate(tokens):
        # Title + capitalized word(s) → PERSON
        if token in PERSON_TITLES and i + 1 < len(tokens):
            j = i + 1
            while j < len(tokens) and tokens[j][0].isupper():
                j += 1
            if j > i + 1:
                entities.append((' '.join(tokens[i+1:j]), 'PER', i+1, j))

        # Capitalized words + org suffix → ORG
        if token in ORG_SUFFIXES and i > 0:
            j = i - 1
            while j >= 0 and tokens[j][0].isupper():
                j -= 1
            if j < i - 1:
                entities.append((' '.join(tokens[j+1:i+1]), 'ORG', j+1, i+1))

    return entities
```

---

## Advantages and Limitations

| Aspect | Rule-Based | Neural |
|--------|------------|--------|
| Precision | High (for covered patterns) | High |
| Recall | Low (limited coverage) | High |
| Interpretability | Full | Limited |
| Development cost | High (manual engineering) | Data collection |
| Adaptation | Manual rule updates | Retraining |
| Best for | Structured entities (dates, money) | Free-form entities |

---

## Hybrid Approaches

Combine rule-based and neural methods:

1. **Pre-annotation**: Use rules to generate silver-standard training data
2. **Post-processing**: Apply rules to fix common neural model errors
3. **Ensemble**: Merge rule-based and neural predictions with conflict resolution
4. **Feature augmentation**: Feed rule match signals as features to neural models

---

## Summary

1. Rule-based NER excels at structured entity types (dates, monetary amounts, emails)
2. Regular expressions and contextual heuristics form the core toolkit
3. Hybrid systems combining rules with neural models often outperform either alone
4. Rule-based systems provide interpretable baselines for any NER project

---

## References

1. Chiticariu, L., Li, Y., & Reiss, F. R. (2013). Rule-Based Information Extraction is Dead! *EMNLP*.
2. Nadeau, D., & Sekine, S. (2007). A Survey of Named Entity Recognition and Classification. *Lingvisticae Investigationes*.
