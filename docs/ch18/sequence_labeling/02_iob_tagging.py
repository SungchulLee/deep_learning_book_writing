"""
IOB and BIOES Tagging Schemes for NER
======================================

This module provides detailed explanation and implementation of various tagging
schemes used in Named Entity Recognition:
- IOB (Inside-Outside-Beginning)
- IOB2 (more strict version)
- BIOES (Beginning-Inside-Outside-End-Single)

These schemes are crucial for sequence labeling in NER.

Learning Objectives:
- Understand different tagging schemes
- Convert between tagging schemes
- Implement tag validation
- Handle edge cases

Author: Educational purposes
Date: 2025
"""

from typing import List, Tuple, Dict
from enum import Enum


class TagScheme(Enum):
    """
    Enumeration of supported tagging schemes.
    """
    IOB = "IOB"      # Inside-Outside-Beginning (original IOB)
    IOB2 = "IOB2"    # IOB version 2 (more commonly used)
    BIOES = "BIOES"  # Beginning-Inside-Outside-End-Single


class TagValidator:
    """
    Validates tag sequences according to different tagging schemes.
    
    Each tagging scheme has specific rules about valid tag transitions.
    This class checks if a sequence of tags follows the rules.
    """
    
    @staticmethod
    def validate_iob2(tags: List[str]) -> Tuple[bool, str]:
        """
        Validate IOB2 tag sequence.
        
        IOB2 Rules:
        1. First tag of an entity is always B-TYPE
        2. Continuation tags are I-TYPE
        3. B-TYPE can only be followed by I-TYPE (same type) or O or B-*
        4. I-TYPE can only be followed by I-TYPE (same type) or O or B-*
        5. Cannot have I-TYPE without preceding B-TYPE of same type
        
        Args:
            tags: List of tags (e.g., ["B-PER", "I-PER", "O", "B-LOC"])
            
        Returns:
            Tuple of (is_valid, error_message)
            
        Example:
            >>> validator = TagValidator()
            >>> tags = ["B-PER", "I-PER", "O", "B-LOC"]
            >>> is_valid, msg = validator.validate_iob2(tags)
            >>> print(is_valid)  # True
        """
        for i in range(len(tags)):
            current_tag = tags[i]
            
            # Skip O tags
            if current_tag == "O":
                continue
            
            # Parse tag
            try:
                prefix, entity_type = current_tag.split("-")
            except ValueError:
                return False, f"Invalid tag format at position {i}: {current_tag}"
            
            # Check valid prefix
            if prefix not in ["B", "I"]:
                return False, f"Invalid prefix at position {i}: {prefix}"
            
            # Check I-TYPE must follow B-TYPE or I-TYPE of same type
            if prefix == "I":
                if i == 0:
                    return False, f"I-{entity_type} cannot be first tag"
                
                prev_tag = tags[i-1]
                if prev_tag == "O":
                    return False, f"I-{entity_type} at position {i} follows O"
                
                try:
                    prev_prefix, prev_type = prev_tag.split("-")
                    if prev_type != entity_type:
                        return False, (f"I-{entity_type} at position {i} "
                                     f"follows {prev_prefix}-{prev_type}")
                except ValueError:
                    return False, f"Invalid previous tag at position {i-1}: {prev_tag}"
        
        return True, "Valid IOB2 sequence"
    
    @staticmethod
    def validate_bioes(tags: List[str]) -> Tuple[bool, str]:
        """
        Validate BIOES tag sequence.
        
        BIOES Rules:
        1. S-TYPE is used for single-token entities
        2. B-TYPE starts multi-token entities
        3. I-TYPE continues entities (must follow B or I of same type)
        4. E-TYPE ends entities (must follow B or I of same type)
        5. O is outside any entity
        
        Args:
            tags: List of tags (e.g., ["B-PER", "I-PER", "E-PER", "O", "S-LOC"])
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        for i in range(len(tags)):
            current_tag = tags[i]
            
            # Skip O tags
            if current_tag == "O":
                continue
            
            # Parse tag
            try:
                prefix, entity_type = current_tag.split("-")
            except ValueError:
                return False, f"Invalid tag format at position {i}: {current_tag}"
            
            # Check valid prefix
            if prefix not in ["B", "I", "E", "S"]:
                return False, f"Invalid prefix at position {i}: {prefix}"
            
            # Single entity should not be followed by continuation
            if prefix == "S":
                if i < len(tags) - 1:
                    next_tag = tags[i+1]
                    if next_tag != "O" and not next_tag.startswith("B-") and not next_tag.startswith("S-"):
                        return False, f"S-{entity_type} at position {i} followed by {next_tag}"
            
            # I or E must follow B or I of same type
            if prefix in ["I", "E"]:
                if i == 0:
                    return False, f"{prefix}-{entity_type} cannot be first tag"
                
                prev_tag = tags[i-1]
                if prev_tag == "O":
                    return False, f"{prefix}-{entity_type} at position {i} follows O"
                
                try:
                    prev_prefix, prev_type = prev_tag.split("-")
                    if prev_type != entity_type:
                        return False, (f"{prefix}-{entity_type} at position {i} "
                                     f"follows {prev_prefix}-{prev_type}")
                    if prev_prefix not in ["B", "I"]:
                        return False, (f"{prefix}-{entity_type} at position {i} "
                                     f"follows invalid prefix {prev_prefix}")
                except ValueError:
                    return False, f"Invalid previous tag at position {i-1}: {prev_tag}"
            
            # B or I must be followed by I or E of same type (or end of sequence)
            if prefix in ["B", "I"]:
                if i < len(tags) - 1:
                    next_tag = tags[i+1]
                    if next_tag == "O":
                        return False, f"{prefix}-{entity_type} at position {i} followed by O (should end with E)"
                    if next_tag not in ["O"] and not next_tag.startswith("I-") and not next_tag.startswith("E-"):
                        return False, f"{prefix}-{entity_type} at position {i} not followed by I or E"
                else:
                    # Last tag in B or I state should be E
                    return False, f"{prefix}-{entity_type} at position {i} is last tag (should be E or S)"
        
        return True, "Valid BIOES sequence"


class TagConverter:
    """
    Convert between different tagging schemes.
    
    This is useful when:
    1. Training data is in one format but model expects another
    2. Comparing results across different systems
    3. Preprocessing data for specific models
    """
    
    @staticmethod
    def iob2_to_bioes(tags: List[str]) -> List[str]:
        """
        Convert IOB2 tags to BIOES tags.
        
        Conversion rules:
        - Single-token entity: B-TYPE → S-TYPE
        - Multi-token entity: B-TYPE → B-TYPE, I-TYPE → I-TYPE, last I-TYPE → E-TYPE
        - O remains O
        
        Args:
            tags: List of IOB2 tags
            
        Returns:
            List of BIOES tags
            
        Example:
            >>> converter = TagConverter()
            >>> iob2_tags = ["B-PER", "I-PER", "O", "B-LOC"]
            >>> bioes_tags = converter.iob2_to_bioes(iob2_tags)
            >>> print(bioes_tags)
            ['B-PER', 'E-PER', 'O', 'S-LOC']
        """
        bioes_tags = []
        
        for i in range(len(tags)):
            current_tag = tags[i]
            
            # O tags remain the same
            if current_tag == "O":
                bioes_tags.append("O")
                continue
            
            # Parse current tag
            prefix, entity_type = current_tag.split("-")
            
            # Look ahead to next tag
            is_last = (i == len(tags) - 1)
            if not is_last:
                next_tag = tags[i+1]
                next_continues = (next_tag != "O" and 
                                next_tag.startswith("I-") and 
                                next_tag.split("-")[1] == entity_type)
            else:
                next_continues = False
            
            # Determine BIOES tag
            if prefix == "B":
                if next_continues:
                    # Start of multi-token entity
                    bioes_tags.append(f"B-{entity_type}")
                else:
                    # Single-token entity
                    bioes_tags.append(f"S-{entity_type}")
            
            elif prefix == "I":
                if next_continues:
                    # Middle of entity
                    bioes_tags.append(f"I-{entity_type}")
                else:
                    # End of entity
                    bioes_tags.append(f"E-{entity_type}")
        
        return bioes_tags
    
    @staticmethod
    def bioes_to_iob2(tags: List[str]) -> List[str]:
        """
        Convert BIOES tags to IOB2 tags.
        
        Conversion rules:
        - S-TYPE → B-TYPE
        - B-TYPE → B-TYPE
        - I-TYPE → I-TYPE
        - E-TYPE → I-TYPE
        - O → O
        
        Args:
            tags: List of BIOES tags
            
        Returns:
            List of IOB2 tags
        """
        iob2_tags = []
        
        for tag in tags:
            if tag == "O":
                iob2_tags.append("O")
            else:
                prefix, entity_type = tag.split("-")
                
                if prefix == "S":
                    iob2_tags.append(f"B-{entity_type}")
                elif prefix == "B":
                    iob2_tags.append(f"B-{entity_type}")
                elif prefix in ["I", "E"]:
                    iob2_tags.append(f"I-{entity_type}")
        
        return iob2_tags
    
    @staticmethod
    def tags_to_entities(tokens: List[str], tags: List[str], 
                        scheme: TagScheme = TagScheme.IOB2) -> List[Tuple[str, str, int, int]]:
        """
        Convert token-tag pairs to entity spans.
        
        This extracts actual entities from tagged sequences.
        
        Args:
            tokens: List of token strings
            tags: List of corresponding tags
            scheme: Tagging scheme used (IOB2 or BIOES)
            
        Returns:
            List of tuples: (entity_text, entity_type, start_idx, end_idx)
            
        Example:
            >>> tokens = ["Steve", "Jobs", "founded", "Apple"]
            >>> tags = ["B-PER", "I-PER", "O", "B-ORG"]
            >>> entities = TagConverter.tags_to_entities(tokens, tags)
            >>> print(entities)
            [('Steve Jobs', 'PER', 0, 2), ('Apple', 'ORG', 3, 4)]
        """
        entities = []
        current_entity = None
        current_tokens = []
        current_start = None
        
        for i, (token, tag) in enumerate(zip(tokens, tags)):
            if tag == "O":
                # End current entity if exists
                if current_entity:
                    entity_text = " ".join(current_tokens)
                    entities.append((entity_text, current_entity, current_start, i))
                    current_entity = None
                    current_tokens = []
                    current_start = None
            
            else:
                prefix, entity_type = tag.split("-")
                
                if scheme == TagScheme.IOB2:
                    if prefix == "B":
                        # Save previous entity if exists
                        if current_entity:
                            entity_text = " ".join(current_tokens)
                            entities.append((entity_text, current_entity, current_start, i))
                        
                        # Start new entity
                        current_entity = entity_type
                        current_tokens = [token]
                        current_start = i
                    
                    elif prefix == "I":
                        if current_entity == entity_type:
                            # Continue current entity
                            current_tokens.append(token)
                        else:
                            # This shouldn't happen in valid IOB2
                            # but handle it by starting new entity
                            if current_entity:
                                entity_text = " ".join(current_tokens)
                                entities.append((entity_text, current_entity, current_start, i))
                            
                            current_entity = entity_type
                            current_tokens = [token]
                            current_start = i
                
                elif scheme == TagScheme.BIOES:
                    if prefix == "S":
                        # Single-token entity
                        if current_entity:
                            entity_text = " ".join(current_tokens)
                            entities.append((entity_text, current_entity, current_start, i))
                        
                        entities.append((token, entity_type, i, i+1))
                        current_entity = None
                        current_tokens = []
                        current_start = None
                    
                    elif prefix == "B":
                        # Start new entity
                        if current_entity:
                            entity_text = " ".join(current_tokens)
                            entities.append((entity_text, current_entity, current_start, i))
                        
                        current_entity = entity_type
                        current_tokens = [token]
                        current_start = i
                    
                    elif prefix in ["I", "E"]:
                        if current_entity == entity_type:
                            current_tokens.append(token)
                            
                            # End entity if E tag
                            if prefix == "E":
                                entity_text = " ".join(current_tokens)
                                entities.append((entity_text, current_entity, current_start, i+1))
                                current_entity = None
                                current_tokens = []
                                current_start = None
        
        # Handle final entity if sequence ends mid-entity
        if current_entity:
            entity_text = " ".join(current_tokens)
            entities.append((entity_text, current_entity, current_start, len(tokens)))
        
        return entities


def demonstrate_tagging_schemes():
    """
    Comprehensive demonstration of different tagging schemes.
    """
    print("="*70)
    print("IOB and BIOES Tagging Schemes Demonstration")
    print("="*70)
    
    # Example 1: IOB2 tagging
    print("\n1. IOB2 Tagging Scheme")
    print("-" * 70)
    
    text = "Steve Jobs founded Apple Inc. in Cupertino"
    tokens = ["Steve", "Jobs", "founded", "Apple", "Inc", ".", "in", "Cupertino"]
    iob2_tags = ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "O", "O", "B-LOC"]
    
    print(f"Text: {text}")
    print(f"\nTokens and IOB2 tags:")
    print(f"{'Token':<15} {'IOB2 Tag':<10}")
    print("-" * 30)
    for token, tag in zip(tokens, iob2_tags):
        print(f"{token:<15} {tag:<10}")
    
    # Validate IOB2 sequence
    is_valid, message = TagValidator.validate_iob2(iob2_tags)
    print(f"\nValidation: {message}")
    
    # Example 2: BIOES tagging
    print("\n\n2. BIOES Tagging Scheme")
    print("-" * 70)
    
    # Convert to BIOES
    bioes_tags = TagConverter.iob2_to_bioes(iob2_tags)
    
    print(f"{'Token':<15} {'IOB2':<10} {'BIOES':<10}")
    print("-" * 40)
    for token, iob2, bioes in zip(tokens, iob2_tags, bioes_tags):
        print(f"{token:<15} {iob2:<10} {bioes:<10}")
    
    # Validate BIOES sequence
    is_valid, message = TagValidator.validate_bioes(bioes_tags)
    print(f"\nValidation: {message}")
    
    # Example 3: Tag scheme comparison
    print("\n\n3. Comparing IOB2 and BIOES")
    print("-" * 70)
    
    examples = [
        {
            'text': "IBM",
            'tokens': ["IBM"],
            'iob2': ["B-ORG"],
            'bioes': ["S-ORG"],
            'note': "Single-token entity: B-ORG vs S-ORG"
        },
        {
            'text': "New York",
            'tokens': ["New", "York"],
            'iob2': ["B-LOC", "I-LOC"],
            'bioes': ["B-LOC", "E-LOC"],
            'note': "Two-token entity: B-I vs B-E"
        },
        {
            'text': "University of California",
            'tokens': ["University", "of", "California"],
            'iob2': ["B-ORG", "I-ORG", "I-ORG"],
            'bioes': ["B-ORG", "I-ORG", "E-ORG"],
            'note': "Multi-token entity: last I vs E"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}: {example['text']}")
        print(f"{'Token':<20} {'IOB2':<10} {'BIOES':<10}")
        print("-" * 40)
        for token, iob2, bioes in zip(example['tokens'], example['iob2'], example['bioes']):
            print(f"{token:<20} {iob2:<10} {bioes:<10}")
        print(f"Note: {example['note']}")
    
    # Example 4: Entity extraction from tags
    print("\n\n4. Extracting Entities from Tags")
    print("-" * 70)
    
    tokens_complex = ["Barack", "Obama", "visited", "New", "York", "City", "and", "Microsoft"]
    iob2_tags_complex = ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "I-LOC", "O", "B-ORG"]
    
    print(f"Tokens: {' '.join(tokens_complex)}")
    print(f"\nTags: {' '.join(iob2_tags_complex)}")
    
    entities = TagConverter.tags_to_entities(tokens_complex, iob2_tags_complex)
    
    print(f"\nExtracted entities:")
    for entity_text, entity_type, start, end in entities:
        print(f"  - '{entity_text}' ({entity_type}) at tokens [{start}:{end}]")
    
    # Example 5: Invalid tag sequences
    print("\n\n5. Detecting Invalid Tag Sequences")
    print("-" * 70)
    
    invalid_examples = [
        {
            'tags': ["I-PER", "I-PER", "O"],
            'issue': "Starts with I- instead of B-"
        },
        {
            'tags': ["B-PER", "I-LOC", "O"],
            'issue': "Entity type mismatch (PER→LOC)"
        },
        {
            'tags': ["B-PER", "O", "I-PER"],
            'issue': "I-PER after O (should be B-PER)"
        }
    ]
    
    for i, example in enumerate(invalid_examples, 1):
        print(f"\nInvalid Example {i}:")
        print(f"Tags: {example['tags']}")
        is_valid, message = TagValidator.validate_iob2(example['tags'])
        print(f"Valid: {is_valid}")
        print(f"Issue: {example['issue']}")
        print(f"Error: {message}")
    
    # Example 6: Benefits of each scheme
    print("\n\n6. When to Use Each Scheme")
    print("-" * 70)
    
    print("\nIOB2 (Inside-Outside-Beginning):")
    print("  Advantages:")
    print("    - Simpler: Only 3 tag types (B, I, O)")
    print("    - More compact representation")
    print("    - Widely used in research")
    print("  Disadvantages:")
    print("    - Cannot explicitly mark entity boundaries")
    print("    - Harder for model to learn where entities end")
    
    print("\nBIOES (Beginning-Inside-Outside-End-Single):")
    print("  Advantages:")
    print("    - Explicit entity boundaries (E tag)")
    print("    - Distinguishes single-token entities (S tag)")
    print("    - Better for models: clearer structure")
    print("    - Potentially better performance")
    print("  Disadvantages:")
    print("    - More complex: 5 tag types")
    print("    - Larger label space")
    print("    - More training data needed")
    
    # Example 7: Practical conversion example
    print("\n\n7. Practical Conversion Example")
    print("-" * 70)
    
    # Sentence with multiple entities
    sentence = "Microsoft CEO Satya Nadella spoke at Stanford"
    tokens_ex = ["Microsoft", "CEO", "Satya", "Nadella", "spoke", "at", "Stanford"]
    iob2_ex = ["B-ORG", "O", "B-PER", "I-PER", "O", "O", "B-ORG"]
    
    print(f"Sentence: {sentence}")
    print(f"\nOriginal IOB2 tags:")
    for token, tag in zip(tokens_ex, iob2_ex):
        print(f"  {token:<15} {tag}")
    
    # Convert to BIOES
    bioes_ex = TagConverter.iob2_to_bioes(iob2_ex)
    print(f"\nConverted to BIOES:")
    for token, tag in zip(tokens_ex, bioes_ex):
        print(f"  {token:<15} {tag}")
    
    # Extract entities
    entities_iob2 = TagConverter.tags_to_entities(tokens_ex, iob2_ex, TagScheme.IOB2)
    entities_bioes = TagConverter.tags_to_entities(tokens_ex, bioes_ex, TagScheme.BIOES)
    
    print(f"\nExtracted entities (both schemes give same result):")
    for entity_text, entity_type, start, end in entities_iob2:
        print(f"  - '{entity_text}' ({entity_type})")


def interactive_tag_converter():
    """
    Interactive tool for practicing tag conversion.
    """
    print("\n\n" + "="*70)
    print("Interactive Tag Converter")
    print("="*70)
    
    # Example for practice
    print("\nPractice Example:")
    print("Tokens: ['Barack', 'Obama', 'visited', 'Google']")
    print("IOB2 tags: ['B-PER', 'I-PER', 'O', 'B-ORG']")
    
    tokens = ['Barack', 'Obama', 'visited', 'Google']
    iob2_tags = ['B-PER', 'I-PER', 'O', 'B-ORG']
    
    # Show conversion
    bioes_tags = TagConverter.iob2_to_bioes(iob2_tags)
    
    print("\nConverted to BIOES:")
    for token, iob2, bioes in zip(tokens, iob2_tags, bioes_tags):
        print(f"  {token:<15} {iob2:<10} → {bioes:<10}")
    
    # Validate both
    print("\nValidation:")
    valid_iob2, msg_iob2 = TagValidator.validate_iob2(iob2_tags)
    valid_bioes, msg_bioes = TagValidator.validate_bioes(bioes_tags)
    print(f"  IOB2: {msg_iob2}")
    print(f"  BIOES: {msg_bioes}")
    
    # Extract entities
    entities = TagConverter.tags_to_entities(tokens, iob2_tags)
    print("\nExtracted entities:")
    for text, etype, start, end in entities:
        print(f"  - '{text}' ({etype})")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_tagging_schemes()
    
    # Interactive converter
    interactive_tag_converter()
    
    print("\n" + "="*70)
    print("Demonstration complete!")
    print("="*70)
    print("\nKey takeaways:")
    print("1. IOB2 uses B- and I- prefixes, simpler but less explicit")
    print("2. BIOES adds E- and S- for explicit boundaries")
    print("3. Both schemes can represent the same entities")
    print("4. BIOES often performs better in deep learning models")
    print("5. Always validate tag sequences for consistency")
