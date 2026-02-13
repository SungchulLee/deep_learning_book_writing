# Dictionary and Gazetteer Methods

## Learning Objectives

- Implement dictionary-based entity recognition with efficient lookup
- Build and maintain gazetteers for domain-specific NER
- Use fuzzy matching and edit distance for robust dictionary lookup
- Integrate gazetteers as features in neural NER models

---

## Overview

Dictionary-based (gazetteer) NER uses curated lists of known entities to identify mentions in text. This approach is particularly effective when comprehensive entity lists exist (e.g., gene databases, financial instrument lists, geographic databases).

---

## Exact Match Lookup

### Trie-Based Dictionary

```python
from typing import Dict, List, Optional, Set, Tuple

class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.entity_types: Set[str] = set()
        self.is_end: bool = False

class GazetteerNER:
    """Dictionary-based NER using trie for efficient multi-word lookup."""

    def __init__(self):
        self.root = TrieNode()

    def add_entry(self, phrase: str, entity_type: str) -> None:
        """Add a phrase to the gazetteer."""
        tokens = phrase.lower().split()
        node = self.root
        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end = True
        node.entity_types.add(entity_type)

    def find_entities(self, tokens: List[str]) -> List[Tuple[int, int, str, str]]:
        """Find all dictionary matches in token sequence (longest match)."""
        entities = []
        i = 0
        while i < len(tokens):
            node = self.root
            best_match = None
            j = i
            while j < len(tokens) and tokens[j].lower() in node.children:
                node = node.children[tokens[j].lower()]
                if node.is_end:
                    for etype in node.entity_types:
                        best_match = (i, j + 1, etype, ' '.join(tokens[i:j+1]))
                j += 1
            if best_match:
                entities.append(best_match)
                i = best_match[1]  # Skip matched tokens
            else:
                i += 1
        return entities

    def load_from_file(self, filepath: str, entity_type: str) -> int:
        """Load entries from a file (one phrase per line)."""
        count = 0
        with open(filepath) as f:
            for line in f:
                phrase = line.strip()
                if phrase:
                    self.add_entry(phrase, entity_type)
                    count += 1
        return count
```

---

## Fuzzy Matching

Handle misspellings and variations using edit distance:

```python
def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]
```

---

## Gazetteer as Neural Features

Integrate dictionary matches as binary features in neural NER:

```python
import torch
import torch.nn as nn

class GazetteerFeatureExtractor:
    """Extract gazetteer match features for each token."""

    def __init__(self, gazetteers: Dict[str, GazetteerNER]):
        self.gazetteers = gazetteers
        self.feature_dim = len(gazetteers)

    def extract(self, tokens: List[str]) -> torch.Tensor:
        features = torch.zeros(len(tokens), self.feature_dim)
        for gaz_idx, (gaz_name, gaz) in enumerate(self.gazetteers.items()):
            matches = gaz.find_entities(tokens)
            for start, end, _, _ in matches:
                for pos in range(start, end):
                    features[pos, gaz_idx] = 1.0
        return features
```

---

## Financial Gazetteers

Key resources for financial NER:

| Resource | Entity Type | Coverage |
|----------|------------|----------|
| SEC EDGAR | Company names, CIK codes | ~800K entities |
| Bloomberg Open Symbology | Tickers, ISINs | Global instruments |
| FRED | Economic indicators | ~800K time series |
| GeoNames | Locations | ~12M entries |
| Wikidata | Multi-type | ~100M items |

---

## Summary

1. Gazetteers provide high-precision NER for known entities
2. Trie structures enable efficient multi-word entity lookup
3. Fuzzy matching handles spelling variations and noise
4. Dictionary features significantly boost neural NER performance
5. Domain-specific gazetteers are essential for financial, biomedical, and legal NER

---

## References

1. Cohen, W. W., & Sarawagi, S. (2004). Exploiting Dictionaries in Named Entity Extraction. *KDD*.
2. Ratinov, L., & Roth, D. (2009). Design Challenges and Misconceptions in NER. *CoNLL*.
