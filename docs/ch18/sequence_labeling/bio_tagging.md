# BIO Tagging Schemes for Sequence Labeling

## Learning Objectives

By the end of this section, you will be able to:

- Understand and implement IOB, IOB2, and BIOES tagging schemes
- Convert between different tagging formats programmatically
- Validate tag sequences for consistency and correctness
- Analyze the trade-offs between tagging schemes for different models
- Extract entities from tagged sequences accurately

## Introduction

Tagging schemes provide a systematic way to encode entity boundaries within sequence labels. The choice of tagging scheme affects both the label space complexity and the model's ability to learn entity boundaries. This section provides a comprehensive treatment of the most common schemes: IOB, IOB2, and BIOES.

## The BIO Family of Tagging Schemes

### IOB (Inside-Outside-Beginning) - Original

The original IOB scheme, introduced by Ramshaw and Marcus (1995), uses three tag prefixes:

| Prefix | Meaning | Usage |
|--------|---------|-------|
| B- | Beginning | First token of an entity **only if** it immediately follows another entity of the same type |
| I- | Inside | All other tokens inside an entity |
| O | Outside | Tokens not part of any entity |

**Example**:
```
Tokens:  Steve  Jobs   founded  Apple  Inc   in  California
IOB:     I-PER  I-PER  O        I-ORG  I-ORG O   I-LOC
```

Note: In original IOB, the B- prefix is only used to separate consecutive entities of the same type.

### IOB2 (Inside-Outside-Beginning, Version 2)

IOB2, now the de facto standard, modifies the B- prefix usage:

| Prefix | Meaning | Usage |
|--------|---------|-------|
| B- | Beginning | **Always** marks the first token of any entity |
| I- | Inside | Continuation tokens within an entity |
| O | Outside | Tokens not part of any entity |

**Example**:
```
Tokens:  Steve  Jobs   founded  Apple  Inc   in  California
IOB2:    B-PER  I-PER  O        B-ORG  I-ORG O   B-LOC
```

### BIOES (Beginning-Inside-Outside-End-Single)

BIOES (also called BILOU) provides explicit boundary markers:

| Prefix | Meaning | Usage |
|--------|---------|-------|
| B- | Beginning | First token of multi-token entity |
| I- | Inside | Middle tokens of entity (3+ tokens) |
| O | Outside | Non-entity tokens |
| E- | End | Last token of multi-token entity |
| S- | Single | Single-token entities |

**Example**:
```
Tokens:  Steve  Jobs   founded  Apple  Inc   in  California
BIOES:   B-PER  E-PER  O        B-ORG  E-ORG O   S-LOC
```

## Mathematical Analysis of Label Space

### Label Space Complexity

For $k$ entity types, the label space sizes are:

| Scheme | Number of Labels | Formula |
|--------|------------------|---------|
| IOB/IOB2 | $2k + 1$ | B-type, I-type for each type, plus O |
| BIOES | $4k + 1$ | B, I, E, S for each type, plus O |

**Example with 4 entity types (PER, ORG, LOC, MISC)**:
- IOB2: $2 \times 4 + 1 = 9$ labels
- BIOES: $4 \times 4 + 1 = 17$ labels

### Transition Constraints

Valid tag sequences follow specific transition rules. Let $y_{i-1}$ and $y_i$ be consecutive tags:

**IOB2 Valid Transitions**:

$$
\text{Valid}(y_{i-1}, y_i) = \begin{cases}
\text{True} & \text{if } y_i = \text{O} \\
\text{True} & \text{if } y_i = \text{B-}t \text{ for any type } t \\
\text{True} & \text{if } y_i = \text{I-}t \text{ and } y_{i-1} \in \{\text{B-}t, \text{I-}t\} \\
\text{False} & \text{otherwise}
\end{cases}
$$

**BIOES Valid Transitions**:

The transition matrix for BIOES is more constrained:

| From \ To | O | B-t | I-t | E-t | S-t | B-t' | S-t' |
|-----------|---|-----|-----|-----|-----|------|------|
| O | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| B-t | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| I-t | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| E-t | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| S-t | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |

Where $t$ and $t'$ represent the same and different entity types respectively.

## PyTorch Implementation

### Tag Scheme Classes

```python
import torch
from typing import List, Tuple, Dict, Set, Optional
from enum import Enum
from dataclasses import dataclass

class TagScheme(Enum):
    """Supported tagging schemes."""
    IOB = "IOB"
    IOB2 = "IOB2"
    BIOES = "BIOES"


@dataclass
class TagInfo:
    """Parsed information from a tag."""
    prefix: str
    entity_type: Optional[str]
    
    @classmethod
    def parse(cls, tag: str) -> 'TagInfo':
        """Parse a tag string into prefix and entity type."""
        if tag == 'O':
            return cls(prefix='O', entity_type=None)
        
        parts = tag.split('-', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid tag format: {tag}")
        
        return cls(prefix=parts[0], entity_type=parts[1])
    
    def __str__(self) -> str:
        if self.entity_type is None:
            return 'O'
        return f"{self.prefix}-{self.entity_type}"


class TagValidator:
    """Validates tag sequences according to tagging scheme rules."""
    
    @staticmethod
    def validate_iob2(tags: List[str]) -> Tuple[bool, str]:
        """
        Validate IOB2 tag sequence.
        
        Rules:
        1. I-TYPE must follow B-TYPE or I-TYPE of same type
        2. B-TYPE can appear after any tag
        3. O can appear after any tag
        
        Args:
            tags: List of IOB2 tags
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        for i, tag in enumerate(tags):
            if tag == 'O':
                continue
            
            try:
                info = TagInfo.parse(tag)
            except ValueError as e:
                return False, f"Position {i}: {e}"
            
            if info.prefix not in ('B', 'I'):
                return False, f"Position {i}: Invalid prefix '{info.prefix}' for IOB2"
            
            # I- tags must follow B- or I- of same type
            if info.prefix == 'I':
                if i == 0:
                    return False, f"Position {i}: I-{info.entity_type} cannot start sequence"
                
                prev_tag = tags[i - 1]
                if prev_tag == 'O':
                    return False, f"Position {i}: I-{info.entity_type} cannot follow O"
                
                prev_info = TagInfo.parse(prev_tag)
                if prev_info.entity_type != info.entity_type:
                    return False, (f"Position {i}: I-{info.entity_type} cannot follow "
                                   f"{prev_info.prefix}-{prev_info.entity_type}")
        
        return True, "Valid IOB2 sequence"
    
    @staticmethod
    def validate_bioes(tags: List[str]) -> Tuple[bool, str]:
        """
        Validate BIOES tag sequence.
        
        Rules:
        1. B- must be followed by I- or E- of same type
        2. I- must follow B- or I- of same type, must be followed by I- or E-
        3. E- must follow B- or I- of same type
        4. S- represents complete single-token entity
        
        Args:
            tags: List of BIOES tags
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        for i, tag in enumerate(tags):
            if tag == 'O':
                continue
            
            try:
                info = TagInfo.parse(tag)
            except ValueError as e:
                return False, f"Position {i}: {e}"
            
            if info.prefix not in ('B', 'I', 'O', 'E', 'S'):
                return False, f"Position {i}: Invalid prefix '{info.prefix}' for BIOES"
            
            # Check predecessor constraints
            if info.prefix in ('I', 'E'):
                if i == 0:
                    return False, f"Position {i}: {tag} cannot start sequence"
                
                prev_info = TagInfo.parse(tags[i - 1])
                valid_prev = prev_info.prefix in ('B', 'I') and \
                             prev_info.entity_type == info.entity_type
                if not valid_prev:
                    return False, f"Position {i}: {tag} cannot follow {tags[i-1]}"
            
            # Check successor constraints
            if info.prefix in ('B', 'I'):
                if i == len(tags) - 1:
                    return False, f"Position {i}: {tag} cannot end sequence (needs E-)"
                
                next_info = TagInfo.parse(tags[i + 1])
                valid_next = next_info.prefix in ('I', 'E') and \
                             next_info.entity_type == info.entity_type
                if not valid_next:
                    return False, f"Position {i}: {tag} cannot be followed by {tags[i+1]}"
        
        return True, "Valid BIOES sequence"
    
    @classmethod
    def validate(cls, tags: List[str], scheme: TagScheme) -> Tuple[bool, str]:
        """Validate tags according to specified scheme."""
        if scheme == TagScheme.IOB2:
            return cls.validate_iob2(tags)
        elif scheme == TagScheme.BIOES:
            return cls.validate_bioes(tags)
        else:
            raise ValueError(f"Unsupported scheme: {scheme}")
```

### Tag Conversion

```python
class TagConverter:
    """Convert between different tagging schemes."""
    
    @staticmethod
    def iob2_to_bioes(tags: List[str]) -> List[str]:
        """
        Convert IOB2 tags to BIOES tags.
        
        Conversion rules:
        - Single-token entity: B-TYPE → S-TYPE
        - Multi-token start: B-TYPE → B-TYPE (if followed by I-)
        - Multi-token middle: I-TYPE → I-TYPE (if followed by I-)
        - Multi-token end: I-TYPE → E-TYPE (if followed by O or B- or end)
        
        Args:
            tags: List of IOB2 tags
            
        Returns:
            List of BIOES tags
        """
        bioes_tags = []
        n = len(tags)
        
        for i, tag in enumerate(tags):
            if tag == 'O':
                bioes_tags.append('O')
                continue
            
            info = TagInfo.parse(tag)
            
            # Check if this is the last token of the entity
            is_last = (i == n - 1) or \
                      (tags[i + 1] == 'O') or \
                      (TagInfo.parse(tags[i + 1]).prefix == 'B')
            
            if info.prefix == 'B':
                if is_last:
                    # Single-token entity
                    bioes_tags.append(f'S-{info.entity_type}')
                else:
                    # Start of multi-token entity
                    bioes_tags.append(f'B-{info.entity_type}')
            else:  # I- prefix
                if is_last:
                    # End of multi-token entity
                    bioes_tags.append(f'E-{info.entity_type}')
                else:
                    # Middle of multi-token entity
                    bioes_tags.append(f'I-{info.entity_type}')
        
        return bioes_tags
    
    @staticmethod
    def bioes_to_iob2(tags: List[str]) -> List[str]:
        """
        Convert BIOES tags to IOB2 tags.
        
        Conversion rules:
        - S-TYPE → B-TYPE
        - E-TYPE → I-TYPE
        - B-TYPE, I-TYPE, O remain unchanged
        
        Args:
            tags: List of BIOES tags
            
        Returns:
            List of IOB2 tags
        """
        iob2_tags = []
        
        for tag in tags:
            if tag == 'O':
                iob2_tags.append('O')
                continue
            
            info = TagInfo.parse(tag)
            
            if info.prefix == 'S':
                iob2_tags.append(f'B-{info.entity_type}')
            elif info.prefix == 'E':
                iob2_tags.append(f'I-{info.entity_type}')
            else:  # B or I
                iob2_tags.append(tag)
        
        return iob2_tags
    
    @staticmethod
    def tags_to_entities(
        tokens: List[str],
        tags: List[str],
        scheme: TagScheme = TagScheme.IOB2
    ) -> List[Tuple[str, str, int, int]]:
        """
        Extract entities from tagged sequence.
        
        Args:
            tokens: List of tokens
            tags: List of corresponding tags
            scheme: Tagging scheme used
            
        Returns:
            List of (entity_text, entity_type, start_idx, end_idx) tuples
        """
        assert len(tokens) == len(tags), "Tokens and tags must have same length"
        
        # Convert to IOB2 for uniform processing
        if scheme == TagScheme.BIOES:
            tags = TagConverter.bioes_to_iob2(tags)
        
        entities = []
        current_entity = None  # (type, start_idx, tokens)
        
        for i, (token, tag) in enumerate(zip(tokens, tags)):
            if tag == 'O':
                # Close current entity if exists
                if current_entity is not None:
                    ent_type, start_idx, ent_tokens = current_entity
                    entities.append((
                        ' '.join(ent_tokens),
                        ent_type,
                        start_idx,
                        i
                    ))
                    current_entity = None
            
            elif tag.startswith('B-'):
                # Close previous entity and start new one
                if current_entity is not None:
                    ent_type, start_idx, ent_tokens = current_entity
                    entities.append((
                        ' '.join(ent_tokens),
                        ent_type,
                        start_idx,
                        i
                    ))
                
                entity_type = TagInfo.parse(tag).entity_type
                current_entity = (entity_type, i, [token])
            
            elif tag.startswith('I-'):
                # Continue current entity
                if current_entity is not None:
                    current_entity[2].append(token)
        
        # Don't forget the last entity
        if current_entity is not None:
            ent_type, start_idx, ent_tokens = current_entity
            entities.append((
                ' '.join(ent_tokens),
                ent_type,
                start_idx,
                len(tokens)
            ))
        
        return entities
```

### Building Transition Matrices for CRF

```python
def build_transition_mask(
    label_to_idx: Dict[str, int],
    scheme: TagScheme = TagScheme.IOB2
) -> torch.Tensor:
    """
    Build a transition mask for CRF layer.
    
    The mask has value 0 for valid transitions and -inf for invalid ones.
    
    Args:
        label_to_idx: Mapping from label strings to indices
        scheme: Tagging scheme
        
    Returns:
        Tensor of shape (num_labels, num_labels) with transition scores
    """
    num_labels = len(label_to_idx)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    # Initialize with all transitions valid
    mask = torch.zeros(num_labels, num_labels)
    
    # Extract entity types
    entity_types = set()
    for label in label_to_idx:
        if label != 'O':
            info = TagInfo.parse(label)
            entity_types.add(info.entity_type)
    
    for i in range(num_labels):
        for j in range(num_labels):
            from_label = idx_to_label[i]
            to_label = idx_to_label[j]
            
            if not _is_valid_transition(from_label, to_label, scheme):
                mask[i, j] = float('-inf')
    
    return mask


def _is_valid_transition(
    from_tag: str, 
    to_tag: str, 
    scheme: TagScheme
) -> bool:
    """Check if transition from from_tag to to_tag is valid."""
    # O can transition to anything
    if from_tag == 'O':
        if to_tag == 'O':
            return True
        to_info = TagInfo.parse(to_tag)
        if scheme == TagScheme.IOB2:
            return to_info.prefix == 'B'
        else:  # BIOES
            return to_info.prefix in ('B', 'S')
    
    from_info = TagInfo.parse(from_tag)
    
    # Anything can transition to O
    if to_tag == 'O':
        if scheme == TagScheme.IOB2:
            return True
        else:  # BIOES
            return from_info.prefix in ('E', 'S')
    
    to_info = TagInfo.parse(to_tag)
    
    if scheme == TagScheme.IOB2:
        # I-X can only follow B-X or I-X
        if to_info.prefix == 'I':
            return from_info.entity_type == to_info.entity_type
        # B-X can follow anything
        return True
    
    else:  # BIOES
        # I or E must follow B or I of same type
        if to_info.prefix in ('I', 'E'):
            return (from_info.prefix in ('B', 'I') and 
                    from_info.entity_type == to_info.entity_type)
        
        # B or S can only follow O, E, or S
        if to_info.prefix in ('B', 'S'):
            return from_info.prefix in ('E', 'S')
        
        return False
```

## Scheme Selection Guidelines

### When to Use IOB2

**Advantages**:
- Simpler label space ($2k + 1$ vs $4k + 1$)
- More training examples per label
- Widely supported by existing tools and datasets
- Sufficient for most sequence labeling tasks

**Best for**:
- Limited training data
- Simple entity structures
- Compatibility with existing datasets (CoNLL format)

### When to Use BIOES

**Advantages**:
- Explicit boundary markers improve learning
- S- tag helps identify single-token entities
- Better performance on some benchmarks (1-2% F1 improvement)
- More informative for downstream CRF layers

**Best for**:
- Sufficient training data
- Many single-token entities in dataset
- When using CRF or structured prediction layers
- Research benchmarking

### Empirical Comparison

Studies have shown BIOES can improve performance:

| Model | Dataset | IOB2 F1 | BIOES F1 | Δ |
|-------|---------|---------|----------|---|
| BiLSTM-CRF | CoNLL-2003 | 90.94 | 91.21 | +0.27 |
| BERT-base | CoNLL-2003 | 92.4 | 92.8 | +0.4 |

The improvement is more pronounced with CRF layers that can explicitly model transition constraints.

## Working with Subword Tokenizers

Modern transformers use subword tokenization, which complicates tag assignment:

```python
def align_labels_to_subwords(
    word_labels: List[str],
    word_to_subword_map: List[List[int]],
    label_first_subword_only: bool = True
) -> List[str]:
    """
    Align word-level labels to subword tokens.
    
    Args:
        word_labels: Labels for each word
        word_to_subword_map: For each word, list of subword indices
        label_first_subword_only: If True, only first subword gets label
        
    Returns:
        Labels for each subword token
    """
    subword_labels = []
    
    for word_idx, label in enumerate(word_labels):
        subword_indices = word_to_subword_map[word_idx]
        
        for i, subword_idx in enumerate(subword_indices):
            if i == 0:
                # First subword gets the original label
                subword_labels.append(label)
            else:
                if label_first_subword_only:
                    # Non-first subwords get special label (ignored in loss)
                    subword_labels.append('[IGNORE]')
                else:
                    # Propagate I- label to continuation subwords
                    if label.startswith('B-'):
                        subword_labels.append('I-' + label[2:])
                    else:
                        subword_labels.append(label)
    
    return subword_labels
```

## Visualization and Debugging

```python
def visualize_tags(
    tokens: List[str],
    tags: List[str],
    scheme: TagScheme = TagScheme.IOB2
) -> str:
    """
    Create a visual representation of tagged sequence.
    
    Args:
        tokens: List of tokens
        tags: List of tags
        scheme: Tagging scheme
        
    Returns:
        Formatted string with aligned tokens and tags
    """
    # Find max width for alignment
    max_token_len = max(len(t) for t in tokens)
    max_tag_len = max(len(t) for t in tags)
    
    lines = []
    lines.append("Tokens: " + " ".join(f"{t:<{max_token_len}}" for t in tokens))
    lines.append("Tags:   " + " ".join(f"{t:<{max_token_len}}" for t in tags))
    
    # Add entity extraction
    entities = TagConverter.tags_to_entities(tokens, tags, scheme)
    if entities:
        lines.append("\nExtracted Entities:")
        for text, etype, start, end in entities:
            lines.append(f"  [{start}:{end}] {etype}: '{text}'")
    
    return "\n".join(lines)


# Example usage
tokens = ["Barack", "Obama", "visited", "New", "York", "City"]
tags_iob2 = ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "I-LOC"]
tags_bioes = TagConverter.iob2_to_bioes(tags_iob2)

print("IOB2 Format:")
print(visualize_tags(tokens, tags_iob2, TagScheme.IOB2))
print("\nBIOES Format:")
print(visualize_tags(tokens, tags_bioes, TagScheme.BIOES))
```

## Summary

BIO tagging schemes provide a principled way to encode entity boundaries:

1. **IOB2** is the standard scheme with B- always marking entity starts
2. **BIOES** adds explicit end and single markers for better boundary learning
3. **Transition constraints** can be enforced through masks in CRF layers
4. **Subword alignment** requires careful handling of label propagation
5. **Scheme choice** depends on data size, entity characteristics, and model architecture

## References

1. Ramshaw, L. A., & Marcus, M. P. (1995). Text Chunking using Transformation-Based Learning. *ACL Workshop on Very Large Corpora*.

2. Ratinov, L., & Roth, D. (2009). Design Challenges and Misconceptions in Named Entity Recognition. *CoNLL*.

3. Sang, E. F. T. K., & Veenstra, J. (1999). Representing Text Chunks. *EACL*.
