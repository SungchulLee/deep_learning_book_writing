# Named Entity Recognition Fundamentals

## Learning Objectives

By the end of this section, you will be able to:

- Define Named Entity Recognition and understand its role in NLP pipelines
- Identify and categorize standard entity types across different domains
- Formulate NER as a sequence labeling problem with mathematical precision
- Understand the relationship between NER and other NLP tasks
- Analyze the challenges and complexities inherent in entity recognition

## Introduction

Named Entity Recognition (NER) is a fundamental task in Natural Language Processing that identifies and classifies named entities in unstructured text into predefined categories such as person names, organizations, locations, dates, and other domain-specific entities. NER serves as a critical component in information extraction pipelines, enabling downstream applications like knowledge graph construction, question answering, and document understanding.

## Mathematical Formulation

### Sequence Labeling Framework

NER is formally defined as a **sequence labeling** problem. Given an input sequence of tokens:

$$
\mathbf{X} = (x_1, x_2, \ldots, x_n)
$$

where each $x_i$ represents a token (word or subword), the goal is to predict a corresponding sequence of labels:

$$
\mathbf{Y} = (y_1, y_2, \ldots, y_n)
$$

where each $y_i \in \mathcal{L}$ belongs to a predefined label set $\mathcal{L}$.

### Optimal Sequence Prediction

The objective is to find the most likely label sequence given the input:

$$
\mathbf{Y}^* = \arg\max_{\mathbf{Y}} P(\mathbf{Y} | \mathbf{X})
$$

Different modeling approaches decompose this probability differently:

**Independent Classification (Token-level)**:
$$
P(\mathbf{Y} | \mathbf{X}) = \prod_{i=1}^{n} P(y_i | \mathbf{X})
$$

**First-Order Markov (Linear-Chain CRF)**:
$$
P(\mathbf{Y} | \mathbf{X}) = \frac{1}{Z(\mathbf{X})} \prod_{i=1}^{n} \psi(y_{i-1}, y_i, \mathbf{X}, i)
$$

where $Z(\mathbf{X})$ is the partition function ensuring proper normalization.

## Entity Types and Taxonomies

### Standard CoNLL Entity Types

The CoNLL-2003 shared task established the foundational entity taxonomy:

| Entity Type | Code | Description | Examples |
|-------------|------|-------------|----------|
| Person | PER | Names of people | "Barack Obama", "Marie Curie" |
| Organization | ORG | Companies, agencies, institutions | "Apple Inc.", "United Nations" |
| Location | LOC | Countries, cities, geographical features | "Paris", "Mount Everest" |
| Miscellaneous | MISC | Other named entities | "World Cup", "Nobel Prize" |

### Extended OntoNotes Taxonomy

OntoNotes 5.0 provides a more granular classification with 18 entity types:

| Category | Entity Types |
|----------|--------------|
| Named Entities | PERSON, NORP, FAC, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE |
| Numerical Entities | DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL |

### Domain-Specific Entity Types

Different domains require specialized entity taxonomies:

**Biomedical NER**:
- Gene/Protein names (e.g., "BRCA1", "insulin")
- Disease names (e.g., "diabetes", "COVID-19")
- Drug names (e.g., "aspirin", "metformin")
- Chemical compounds (e.g., "H2O", "glucose")

**Financial NER**:
- Company names and tickers (e.g., "AAPL", "Goldman Sachs")
- Financial instruments (e.g., "10-year Treasury")
- Monetary amounts and currencies
- Economic indicators (e.g., "GDP", "CPI")

**Legal NER**:
- Case names and citations
- Legal entities (parties, courts)
- Statutes and regulations
- Dates and jurisdictions

## Entity Representation

### Character-Level Spans

Entities are represented as spans with character offsets:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Entity:
    """Represents a named entity with character-level positioning."""
    text: str           # The entity surface form
    entity_type: str    # Entity category (PER, ORG, LOC, etc.)
    start: int          # Start character offset (inclusive)
    end: int            # End character offset (exclusive)
    confidence: Optional[float] = None  # Model confidence score
    
    def __post_init__(self):
        """Validate entity representation."""
        assert self.end > self.start, "End must be greater than start"
        assert len(self.text) == self.end - self.start, "Text length must match span"
    
    def overlaps(self, other: 'Entity') -> bool:
        """Check if this entity overlaps with another."""
        return not (self.end <= other.start or other.start >= self.end)
    
    def contains(self, other: 'Entity') -> bool:
        """Check if this entity fully contains another."""
        return self.start <= other.start and self.end >= other.end
```

### Token-Level Alignment

When working with tokenized text, entities must be aligned to token boundaries:

```python
from typing import List, Tuple

def align_entities_to_tokens(
    text: str,
    entities: List[Entity],
    token_spans: List[Tuple[int, int]]
) -> List[List[str]]:
    """
    Align character-level entities to token-level labels.
    
    Args:
        text: Original text string
        entities: List of Entity objects with character offsets
        token_spans: List of (start, end) character positions for each token
        
    Returns:
        List of entity labels for each token (IOB2 format)
    """
    labels = ['O'] * len(token_spans)
    
    for entity in sorted(entities, key=lambda e: e.start):
        entity_tokens = []
        
        for idx, (tok_start, tok_end) in enumerate(token_spans):
            # Check if token overlaps with entity
            if tok_start < entity.end and tok_end > entity.start:
                entity_tokens.append(idx)
        
        # Assign IOB2 labels
        for i, tok_idx in enumerate(entity_tokens):
            prefix = 'B' if i == 0 else 'I'
            labels[tok_idx] = f"{prefix}-{entity.entity_type}"
    
    return labels
```

## NER in the NLP Pipeline

### Preprocessing Dependencies

NER systems typically depend on several preprocessing steps:

```
Raw Text → Tokenization → (Optional: POS Tagging) → NER → Downstream Tasks
```

**Tokenization Impact**: The choice of tokenizer significantly affects NER performance:
- Word-level tokenizers preserve entity boundaries but struggle with OOV words
- Subword tokenizers (BPE, WordPiece) handle OOV but may split entities

### Downstream Applications

NER enables numerous downstream applications:

1. **Information Extraction**: Extracting structured data from unstructured text
2. **Question Answering**: Identifying answer candidates in passages
3. **Knowledge Graph Construction**: Populating entity nodes in graphs
4. **Document Classification**: Using entity distributions as features
5. **Coreference Resolution**: Linking pronouns to named entities
6. **Relation Extraction**: Finding relationships between identified entities

## Challenges in NER

### Ambiguity and Context Dependence

The same surface form can represent different entity types:

| Text | Context | Entity Type |
|------|---------|-------------|
| "Washington" | "Washington crossed the Delaware" | PERSON |
| "Washington" | "I visited Washington D.C." | LOCATION |
| "Washington" | "The Washington Post reported..." | ORGANIZATION |

### Entity Boundary Detection

Determining precise entity boundaries presents challenges:

```
"University of California, Berkeley" → Single ORG entity
"New York City Department of Education" → Single ORG entity  
"Dr. Martin Luther King Jr." → Single PER entity with title
```

### Nested and Overlapping Entities

Some text contains nested entity structures:

```
"Bank of [America]_LOC"  → Contains nested LOC within ORG
"[Bank of America]_ORG"  → Full organization name

"[New York]_LOC University" → LOC within broader context
"[New York University]_ORG" → Full organization name
```

### Rare and Emerging Entities

NER systems must handle:
- **Zero-shot entities**: New companies, products, or people not in training data
- **Domain shift**: Entities from specialized domains (e.g., biomedical)
- **Temporal drift**: New entities emerging after model training

## PyTorch Implementation: Entity Data Structures

```python
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class NERExample:
    """A single NER training/inference example."""
    tokens: List[str]
    labels: Optional[List[str]] = None
    entities: List[Entity] = field(default_factory=list)
    
    def to_tensor(
        self, 
        token_to_idx: Dict[str, int],
        label_to_idx: Dict[str, int],
        max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """Convert example to tensors for model input."""
        # Pad or truncate tokens
        token_ids = [token_to_idx.get(t, token_to_idx['<UNK>']) 
                     for t in self.tokens[:max_length]]
        padding_length = max_length - len(token_ids)
        token_ids += [token_to_idx['<PAD>']] * padding_length
        
        # Create attention mask
        attention_mask = [1] * min(len(self.tokens), max_length)
        attention_mask += [0] * padding_length
        
        result = {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }
        
        # Add labels if available (training mode)
        if self.labels is not None:
            label_ids = [label_to_idx.get(l, label_to_idx['O']) 
                        for l in self.labels[:max_length]]
            label_ids += [label_to_idx['O']] * padding_length
            result['labels'] = torch.tensor(label_ids, dtype=torch.long)
        
        return result


class NERDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for NER."""
    
    def __init__(
        self,
        examples: List[NERExample],
        token_to_idx: Dict[str, int],
        label_to_idx: Dict[str, int],
        max_length: int = 512
    ):
        self.examples = examples
        self.token_to_idx = token_to_idx
        self.label_to_idx = label_to_idx
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx].to_tensor(
            self.token_to_idx,
            self.label_to_idx,
            self.max_length
        )
    
    @staticmethod
    def build_vocabularies(
        examples: List[NERExample]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Build token and label vocabularies from examples."""
        tokens = set(['<PAD>', '<UNK>'])
        labels = set(['O'])
        
        for ex in examples:
            tokens.update(ex.tokens)
            if ex.labels:
                labels.update(ex.labels)
        
        token_to_idx = {t: i for i, t in enumerate(sorted(tokens))}
        label_to_idx = {l: i for i, l in enumerate(sorted(labels))}
        
        return token_to_idx, label_to_idx
```

## Evaluation Preview

NER evaluation involves several complementary metrics:

### Token-Level Metrics
- Precision, Recall, F1 for each token prediction
- Useful for debugging but can be misleading

### Entity-Level Metrics (Standard)
- **Exact Match**: Entity boundaries and type must match exactly
- **Partial Match**: Overlapping spans with correct type
- **Type Match**: Correct type regardless of boundaries

The standard evaluation uses **entity-level exact match F1**:

$$
\text{Precision} = \frac{|\text{Predicted} \cap \text{Gold}|}{|\text{Predicted}|}
$$

$$
\text{Recall} = \frac{|\text{Predicted} \cap \text{Gold}|}{|\text{Gold}|}
$$

$$
\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

## Summary

Named Entity Recognition is a foundational NLP task that:

1. **Formulates** entity identification as sequence labeling
2. **Categorizes** entities into domain-specific taxonomies
3. **Represents** entities as character or token spans
4. **Enables** downstream information extraction tasks
5. **Faces** challenges including ambiguity, nested entities, and domain adaptation

The following sections will explore tagging schemes, model architectures, and training procedures in detail.

## References

1. Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition. *CoNLL*.

2. Weischedel, R., et al. (2013). OntoNotes Release 5.0. Linguistic Data Consortium.

3. Lample, G., et al. (2016). Neural Architectures for Named Entity Recognition. *NAACL-HLT*.

4. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT*.
