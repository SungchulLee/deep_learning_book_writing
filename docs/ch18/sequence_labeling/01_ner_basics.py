"""
Named Entity Recognition - Basics
=================================

This module introduces the fundamental concepts of Named Entity Recognition (NER),
including what entities are, common entity types, and basic text preprocessing for NER.

Learning Objectives:
- Understand what named entities are
- Learn common entity types and tagging schemes
- Explore basic text preprocessing for NER
- Understand the challenges in NER

Author: Educational purposes
Date: 2025
"""

import re
from typing import List, Tuple, Dict
from collections import defaultdict


class EntityType:
    """
    Enumeration of common entity types used in NER.
    
    Standard CoNLL entity types:
    - PER: Person names
    - ORG: Organization names
    - LOC: Location names
    - MISC: Miscellaneous entities
    
    Extended entity types:
    - DATE: Date expressions
    - TIME: Time expressions
    - MONEY: Monetary values
    - PERCENT: Percentages
    """
    
    # Standard entity types
    PERSON = "PER"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    MISCELLANEOUS = "MISC"
    
    # Extended entity types
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    
    @staticmethod
    def all_types():
        """Return all defined entity types."""
        return [
            EntityType.PERSON,
            EntityType.ORGANIZATION,
            EntityType.LOCATION,
            EntityType.MISCELLANEOUS,
            EntityType.DATE,
            EntityType.TIME,
            EntityType.MONEY,
            EntityType.PERCENT,
            EntityType.PRODUCT,
            EntityType.EVENT
        ]


class Entity:
    """
    Represents a named entity in text.
    
    An entity consists of:
    - text: The actual text span
    - entity_type: The type of entity (PER, ORG, LOC, etc.)
    - start: Character start position in the original text
    - end: Character end position in the original text
    - confidence: Optional confidence score (for ML models)
    """
    
    def __init__(self, text: str, entity_type: str, start: int, end: int, confidence: float = 1.0):
        """
        Initialize an Entity.
        
        Args:
            text: The entity text span
            entity_type: Type of entity (PER, ORG, LOC, etc.)
            start: Start character position
            end: End character position
            confidence: Confidence score (0-1), default 1.0
        """
        self.text = text
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.confidence = confidence
    
    def __repr__(self):
        """String representation of the entity."""
        return f"Entity(text='{self.text}', type='{self.entity_type}', span=({self.start}, {self.end}))"
    
    def __eq__(self, other):
        """
        Check equality of two entities.
        Two entities are equal if they have the same span and type.
        """
        if not isinstance(other, Entity):
            return False
        return (self.start == other.start and 
                self.end == other.end and 
                self.entity_type == other.entity_type)
    
    def overlaps(self, other: 'Entity') -> bool:
        """
        Check if this entity overlaps with another entity.
        
        Args:
            other: Another Entity object
            
        Returns:
            True if entities overlap, False otherwise
        """
        return not (self.end <= other.start or other.end <= self.start)
    
    def to_dict(self) -> Dict:
        """Convert entity to dictionary format."""
        return {
            'text': self.text,
            'type': self.entity_type,
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence
        }


class Token:
    """
    Represents a token in tokenized text.
    
    In NER, we need to track not just the token text, but also:
    - Its position in the original text
    - Its features (capitalization, punctuation, etc.)
    - Its entity label (for training data)
    """
    
    def __init__(self, text: str, start: int, end: int):
        """
        Initialize a Token.
        
        Args:
            text: Token text
            start: Start character position in original text
            end: End character position in original text
        """
        self.text = text
        self.start = start
        self.end = end
        self.label = "O"  # Default label is "O" (outside entity)
        
    def __repr__(self):
        """String representation of token."""
        return f"Token('{self.text}', label='{self.label}')"
    
    def get_features(self) -> Dict[str, bool]:
        """
        Extract linguistic features from the token.
        
        These features are useful for traditional ML-based NER:
        - is_capitalized: First letter is uppercase
        - is_all_caps: All letters are uppercase
        - is_title: First letter caps, rest lowercase
        - contains_digit: Token contains numbers
        - contains_punctuation: Token contains punctuation
        - is_alpha: Token contains only letters
        
        Returns:
            Dictionary of boolean features
        """
        features = {
            'is_capitalized': self.text[0].isupper() if self.text else False,
            'is_all_caps': self.text.isupper(),
            'is_title': self.text.istitle(),
            'contains_digit': any(c.isdigit() for c in self.text),
            'contains_punctuation': any(not c.isalnum() for c in self.text),
            'is_alpha': self.text.isalpha(),
            'length': len(self.text),
            'is_short': len(self.text) <= 3,
            'is_long': len(self.text) >= 10
        }
        return features


class SimpleTokenizer:
    """
    Simple whitespace and punctuation-based tokenizer for NER.
    
    This tokenizer:
    1. Splits on whitespace
    2. Separates punctuation from words
    3. Preserves character offsets (important for entity spans)
    
    Note: For production use, consider using spaCy or NLTK tokenizers.
    """
    
    def __init__(self):
        """Initialize the tokenizer."""
        # Pattern to match word characters, numbers, or individual punctuation
        self.pattern = re.compile(r'\w+|[^\w\s]')
    
    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenize text into Token objects.
        
        Args:
            text: Input text string
            
        Returns:
            List of Token objects with text and character positions
            
        Example:
            >>> tokenizer = SimpleTokenizer()
            >>> tokens = tokenizer.tokenize("Apple Inc. is great!")
            >>> for token in tokens:
            ...     print(f"{token.text} ({token.start}:{token.end})")
            Apple (0:5)
            Inc (6:9)
            . (9:10)
            is (11:13)
            great (14:19)
            ! (19:20)
        """
        tokens = []
        
        # Find all matches with their positions
        for match in self.pattern.finditer(text):
            token_text = match.group()
            start = match.start()
            end = match.end()
            
            # Create Token object
            token = Token(token_text, start, end)
            tokens.append(token)
        
        return tokens
    
    def tokenize_with_labels(self, text: str, entities: List[Entity]) -> List[Token]:
        """
        Tokenize text and assign entity labels to tokens.
        
        This is crucial for preparing training data. Each token gets a label
        indicating whether it's part of an entity and what kind.
        
        Args:
            text: Input text string
            entities: List of Entity objects in the text
            
        Returns:
            List of Token objects with assigned labels
            
        Example:
            >>> tokenizer = SimpleTokenizer()
            >>> text = "Steve Jobs founded Apple"
            >>> entities = [
            ...     Entity("Steve Jobs", "PER", 0, 11, 1.0),
            ...     Entity("Apple", "ORG", 20, 25, 1.0)
            ... ]
            >>> tokens = tokenizer.tokenize_with_labels(text, entities)
        """
        # First tokenize normally
        tokens = self.tokenize(text)
        
        # For each token, determine its label based on entities
        for token in tokens:
            token.label = self._get_token_label(token, entities)
        
        return tokens
    
    def _get_token_label(self, token: Token, entities: List[Entity]) -> str:
        """
        Determine the entity label for a token.
        
        Args:
            token: Token to label
            entities: List of entities in the text
            
        Returns:
            Label string (e.g., "B-PER", "I-ORG", "O")
        """
        # Check if token overlaps with any entity
        for entity in entities:
            # Token is inside entity span
            if token.start >= entity.start and token.end <= entity.end:
                # Check if it's the first token of the entity
                is_first = token.start == entity.start
                
                # Return appropriate label (B- for beginning, I- for inside)
                prefix = "B" if is_first else "I"
                return f"{prefix}-{entity.entity_type}"
        
        # Token is not part of any entity
        return "O"


class NERDataset:
    """
    Container for NER dataset with texts and their entity annotations.
    
    This class helps organize and manipulate NER data for training and evaluation.
    """
    
    def __init__(self):
        """Initialize empty dataset."""
        self.samples = []  # List of (text, entities) tuples
    
    def add_sample(self, text: str, entities: List[Entity]):
        """
        Add a sample to the dataset.
        
        Args:
            text: Text string
            entities: List of Entity objects in the text
        """
        self.samples.append((text, entities))
    
    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[str, List[Entity]]:
        """Get a sample by index."""
        return self.samples[idx]
    
    def get_statistics(self) -> Dict:
        """
        Compute dataset statistics.
        
        Returns:
            Dictionary with dataset statistics:
            - num_samples: Total number of samples
            - num_entities: Total number of entities
            - entity_type_counts: Count of each entity type
            - avg_entities_per_sample: Average entities per text
        """
        total_entities = 0
        entity_type_counts = defaultdict(int)
        
        for text, entities in self.samples:
            total_entities += len(entities)
            for entity in entities:
                entity_type_counts[entity.entity_type] += 1
        
        return {
            'num_samples': len(self.samples),
            'num_entities': total_entities,
            'entity_type_counts': dict(entity_type_counts),
            'avg_entities_per_sample': total_entities / len(self.samples) if self.samples else 0
        }
    
    def display_sample(self, idx: int):
        """
        Display a sample with its entities in a readable format.
        
        Args:
            idx: Sample index
        """
        text, entities = self.samples[idx]
        
        print(f"\n{'='*60}")
        print(f"Sample {idx + 1}:")
        print(f"{'='*60}")
        print(f"Text: {text}")
        print(f"\nEntities found: {len(entities)}")
        print("-" * 60)
        
        for i, entity in enumerate(entities, 1):
            print(f"{i}. '{entity.text}' -> {entity.entity_type} (pos: {entity.start}-{entity.end})")
        
        print("="*60)


def demonstrate_ner_basics():
    """
    Demonstration of basic NER concepts.
    
    This function shows:
    1. Creating entities
    2. Tokenizing text
    3. Assigning labels to tokens
    4. Building a simple dataset
    """
    print("="*70)
    print("Named Entity Recognition - Basic Concepts Demonstration")
    print("="*70)
    
    # Example 1: Creating entities
    print("\n1. Creating Named Entities")
    print("-" * 70)
    
    text1 = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    
    # Define entities in the text
    entities1 = [
        Entity("Apple Inc.", EntityType.ORGANIZATION, 0, 10, 1.0),
        Entity("Steve Jobs", EntityType.PERSON, 26, 37, 1.0),
        Entity("Cupertino", EntityType.LOCATION, 41, 50, 1.0),
        Entity("California", EntityType.LOCATION, 52, 62, 1.0)
    ]
    
    print(f"Text: {text1}")
    print(f"\nEntities:")
    for entity in entities1:
        print(f"  - {entity}")
    
    # Example 2: Tokenization
    print("\n\n2. Tokenization for NER")
    print("-" * 70)
    
    tokenizer = SimpleTokenizer()
    tokens = tokenizer.tokenize(text1)
    
    print(f"Tokens extracted: {len(tokens)}")
    for i, token in enumerate(tokens[:10], 1):  # Show first 10 tokens
        features = token.get_features()
        print(f"  {i}. '{token.text}' at ({token.start}:{token.end})")
        print(f"     Features: capitalized={features['is_capitalized']}, "
              f"all_caps={features['is_all_caps']}")
    
    # Example 3: Token labeling
    print("\n\n3. Token Labeling (IOB Scheme)")
    print("-" * 70)
    
    labeled_tokens = tokenizer.tokenize_with_labels(text1, entities1)
    
    print(f"{'Token':<15} {'Label':<10} {'Position'}")
    print("-" * 70)
    for token in labeled_tokens[:20]:  # Show first 20 tokens
        print(f"{token.text:<15} {token.label:<10} ({token.start}:{token.end})")
    
    # Example 4: Entity overlaps
    print("\n\n4. Checking Entity Overlaps")
    print("-" * 70)
    
    entity_a = Entity("New York", EntityType.LOCATION, 0, 8, 1.0)
    entity_b = Entity("York University", EntityType.ORGANIZATION, 4, 19, 1.0)
    
    print(f"Entity A: {entity_a}")
    print(f"Entity B: {entity_b}")
    print(f"Do they overlap? {entity_a.overlaps(entity_b)}")
    
    # Example 5: Building a dataset
    print("\n\n5. Building a NER Dataset")
    print("-" * 70)
    
    dataset = NERDataset()
    
    # Add several samples
    samples = [
        ("Apple Inc. was founded by Steve Jobs in Cupertino.",
         [Entity("Apple Inc.", "ORG", 0, 10, 1.0),
          Entity("Steve Jobs", "PER", 27, 37, 1.0),
          Entity("Cupertino", "LOC", 41, 50, 1.0)]),
        
        ("Google announced a new product in Mountain View.",
         [Entity("Google", "ORG", 0, 6, 1.0),
          Entity("Mountain View", "LOC", 34, 47, 1.0)]),
        
        ("Barack Obama visited Paris in 2015.",
         [Entity("Barack Obama", "PER", 0, 12, 1.0),
          Entity("Paris", "LOC", 21, 26, 1.0)]),
    ]
    
    for text, entities in samples:
        dataset.add_sample(text, entities)
    
    # Display statistics
    stats = dataset.get_statistics()
    print(f"Dataset size: {stats['num_samples']} samples")
    print(f"Total entities: {stats['num_entities']}")
    print(f"Average entities per sample: {stats['avg_entities_per_sample']:.2f}")
    print(f"\nEntity type distribution:")
    for entity_type, count in stats['entity_type_counts'].items():
        print(f"  - {entity_type}: {count}")
    
    # Display a sample
    dataset.display_sample(0)
    
    # Example 6: Common challenges
    print("\n\n6. Common Challenges in NER")
    print("-" * 70)
    
    challenges = [
        ("Ambiguity", 
         "Washington (person or location?)",
         "Context is crucial for disambiguation"),
        
        ("Nested entities",
         "Bank of America in New York",
         "'Bank of America' (ORG) contains 'America' (LOC)"),
        
        ("Entity boundaries",
         "New York City vs. New York",
         "Determining exact entity span"),
        
        ("Rare entities",
         "Newly founded companies or products",
         "Not in training data"),
        
        ("Multi-word entities",
         "University of California, Berkeley",
         "Long entity spans are challenging")
    ]
    
    for challenge_type, example, explanation in challenges:
        print(f"\n{challenge_type}:")
        print(f"  Example: {example}")
        print(f"  Issue: {explanation}")


if __name__ == "__main__":
    # Run the demonstration
    demonstrate_ner_basics()
    
    print("\n" + "="*70)
    print("Demonstration complete!")
    print("="*70)
    
    # Additional interactive example
    print("\n\nTry it yourself!")
    print("-" * 70)
    
    # User can modify this example
    custom_text = "Microsoft CEO Satya Nadella announced new AI products in Seattle."
    custom_entities = [
        Entity("Microsoft", EntityType.ORGANIZATION, 0, 9, 1.0),
        Entity("Satya Nadella", EntityType.PERSON, 14, 27, 1.0),
        Entity("Seattle", EntityType.LOCATION, 58, 65, 1.0)
    ]
    
    print(f"\nCustom text: {custom_text}")
    print(f"\nEntities:")
    for entity in custom_entities:
        print(f"  - {entity}")
    
    # Tokenize and label
    tokenizer = SimpleTokenizer()
    labeled_tokens = tokenizer.tokenize_with_labels(custom_text, custom_entities)
    
    print(f"\nLabeled tokens:")
    print(f"{'Token':<20} {'Label':<10}")
    print("-" * 30)
    for token in labeled_tokens:
        print(f"{token.text:<20} {token.label:<10}")
