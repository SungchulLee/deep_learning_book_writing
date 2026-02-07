# NER Evaluation Metrics

## Learning Objectives

- Distinguish between token-level and entity-level evaluation
- Implement exact match, partial match evaluation
- Calculate micro, macro, and weighted F1 scores
- Use the seqeval library for standard evaluation

## Entity-Level Evaluation (Standard)

NER systems are evaluated at the **entity level**: both boundaries AND type must match exactly.

### Core Metrics

For predicted entities $\hat{E}$ and gold entities $E^*$:

$$
\text{Precision} = \frac{|\hat{E} \cap E^*|}{|\hat{E}|}, \quad
\text{Recall} = \frac{|\hat{E} \cap E^*|}{|E^*|}, \quad
F_1 = \frac{2 \cdot P \cdot R}{P + R}
$$

## PyTorch Implementation

```python
from typing import List, Dict, Set
from collections import defaultdict
from dataclasses import dataclass

@dataclass(frozen=True)
class Entity:
    entity_type: str
    start: int
    end: int

def extract_entities(tags: List[str]) -> Set[Entity]:
    """Extract entities from IOB2 tags."""
    entities = set()
    current = None
    
    for i, tag in enumerate(tags):
        if tag == 'O':
            if current:
                entities.add(Entity(current[0], current[1], i))
                current = None
        elif tag.startswith('B-'):
            if current:
                entities.add(Entity(current[0], current[1], i))
            current = (tag[2:], i)
        elif tag.startswith('I-') and current and current[0] == tag[2:]:
            pass  # Continue entity
        else:
            if current:
                entities.add(Entity(current[0], current[1], i))
            current = None
    
    if current:
        entities.add(Entity(current[0], current[1], len(tags)))
    return entities

def compute_ner_metrics(
    pred_tags: List[List[str]],
    gold_tags: List[List[str]],
    average: str = 'micro'
) -> Dict[str, float]:
    """Compute NER F1 metrics."""
    tp_per_type = defaultdict(int)
    fp_per_type = defaultdict(int)
    fn_per_type = defaultdict(int)
    
    for pred, gold in zip(pred_tags, gold_tags):
        pred_ents = extract_entities(pred)
        gold_ents = extract_entities(gold)
        
        for e in pred_ents:
            if e in gold_ents:
                tp_per_type[e.entity_type] += 1
            else:
                fp_per_type[e.entity_type] += 1
        
        for e in gold_ents:
            if e not in pred_ents:
                fn_per_type[e.entity_type] += 1
    
    if average == 'micro':
        tp = sum(tp_per_type.values())
        fp = sum(fp_per_type.values())
        fn = sum(fn_per_type.values())
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        return {'precision': p, 'recall': r, 'f1': f1}
    
    # Macro average
    all_types = set(tp_per_type) | set(fp_per_type) | set(fn_per_type)
    f1s = []
    for t in all_types:
        tp, fp, fn = tp_per_type[t], fp_per_type[t], fn_per_type[t]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
    
    return {'f1': sum(f1s) / len(f1s) if f1s else 0}
```

## Using seqeval Library

```python
from seqeval.metrics import classification_report, f1_score

# Standard evaluation
y_true = [['B-PER', 'I-PER', 'O', 'B-LOC']]
y_pred = [['B-PER', 'I-PER', 'O', 'B-LOC']]

print(classification_report(y_true, y_pred))
print(f"F1: {f1_score(y_true, y_pred):.4f}")
```

## Match Types Summary

| Type | Boundaries | Entity Type | Use Case |
|------|------------|-------------|----------|
| Exact (Strict) | Must match | Must match | Standard benchmark |
| Partial | Must overlap | Must match | Lenient evaluation |
| Type Only | Any | Must match | Type analysis |

## Key Points

1. **Entity-level exact match** is the standard for NER evaluation
2. **Micro F1** weights by entity frequency (standard for CoNLL)
3. **Macro F1** gives equal weight to all entity types
4. Use **seqeval** library for consistent evaluation
