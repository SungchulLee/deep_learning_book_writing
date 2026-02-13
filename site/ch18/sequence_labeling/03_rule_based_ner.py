"""
Rule-Based Named Entity Recognition
====================================

This module implements rule-based NER using pattern matching and linguistic rules.
Rule-based systems are fast, interpretable, and work well for domains with clear patterns.

Learning Objectives:
- Implement pattern-based entity extraction
- Use regular expressions for NER
- Understand rule-based system strengths and limitations
- Combine multiple rule types

Key Concepts:
- Pattern matching with regex
- Capitalization patterns
- Context-based rules
- Rule prioritization

Author: Educational purposes
Date: 2025
"""

import re
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass


@dataclass
class Rule:
    """
    Represents a single extraction rule.
    
    Attributes:
        name: Rule identifier
        pattern: Regex pattern to match
        entity_type: Type of entity this rule extracts
        priority: Rule priority (higher = applied first)
        context_required: Optional context words that must be present
    """
    name: str
    pattern: str
    entity_type: str
    priority: int = 1
    context_required: List[str] = None
    
    def __post_init__(self):
        """Compile the regex pattern after initialization."""
        self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE)


class RuleBasedNER:
    """
    Rule-based Named Entity Recognition system.
    
    This system uses hand-crafted rules including:
    1. Regular expression patterns
    2. Capitalization patterns
    3. Context-based rules
    4. Dictionary lookups (integrated with rule matching)
    
    Advantages:
    - Fast and efficient
    - No training data required
    - Highly interpretable
    - Domain-specific rules possible
    - High precision for well-defined patterns
    
    Disadvantages:
    - Low recall (misses variations)
    - Requires manual rule creation
    - Hard to maintain for large rule sets
    - Cannot generalize to unseen patterns
    """
    
    def __init__(self):
        """Initialize the rule-based NER system."""
        self.rules: List[Rule] = []
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """
        Initialize default rule set for common entity types.
        
        These rules cover:
        - Person names (capitalization patterns)
        - Organizations (corporate suffixes)
        - Locations (place indicators)
        - Dates and times
        - Money and percentages
        - Emails and URLs
        """
        
        # Person name patterns
        # Pattern: Capitalized word followed by capitalized word
        # Example: "John Smith", "Mary Johnson"
        self.add_rule(Rule(
            name="person_full_name",
            pattern=r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b',
            entity_type="PER",
            priority=3
        ))
        
        # Organization patterns
        # Pattern: Capitalized words with corporate suffixes
        # Example: "Apple Inc.", "Microsoft Corporation", "Google LLC"
        self.add_rule(Rule(
            name="organization_suffix",
            pattern=r'\b([A-Z][A-Za-z\s&]+)\s+(Inc\.|Corp\.|Corporation|LLC|Ltd\.|Limited|Company|Co\.)\b',
            entity_type="ORG",
            priority=5
        ))
        
        # Pattern: "The" followed by capitalized words
        # Example: "The New York Times", "The Washington Post"
        self.add_rule(Rule(
            name="organization_the",
            pattern=r'\bThe\s+([A-Z][A-Za-z\s]+(?:Inc\.|Corp\.|Times|Post|Bank|University)?)\b',
            entity_type="ORG",
            priority=4
        ))
        
        # Location patterns
        # Pattern: Place-indicating keywords
        # Example: "New York City", "San Francisco", "Mount Everest"
        self.add_rule(Rule(
            name="location_indicators",
            pattern=r'\b((?:New|San|Los|Las)\s+[A-Z][a-z]+(?:\s+(?:City|Beach|Angeles|Vegas))?|'
                   r'Mount\s+[A-Z][a-z]+|Lake\s+[A-Z][a-z]+|'
                   r'[A-Z][a-z]+\s+(?:River|Ocean|Sea|Mountain))\b',
            entity_type="LOC",
            priority=4
        ))
        
        # Pattern: States and countries
        # Simplified list - in practice, use comprehensive gazetteer
        self.add_rule(Rule(
            name="location_places",
            pattern=r'\b(California|Texas|New\s+York|Florida|Illinois|'
                   r'London|Paris|Tokyo|Beijing|Sydney|Berlin|'
                   r'United\s+States|USA|UK|China|Japan|Germany)\b',
            entity_type="LOC",
            priority=3
        ))
        
        # Date patterns
        # Pattern: Various date formats
        # Example: "January 1, 2020", "01/01/2020", "2020-01-01"
        self.add_rule(Rule(
            name="date_full",
            pattern=r'\b((?:January|February|March|April|May|June|July|August|'
                   r'September|October|November|December)\s+\d{1,2},?\s+\d{4}|'
                   r'\d{1,2}/\d{1,2}/\d{2,4}|'
                   r'\d{4}-\d{2}-\d{2})\b',
            entity_type="DATE",
            priority=5
        ))
        
        # Time patterns
        # Example: "3:30 PM", "14:00", "noon"
        self.add_rule(Rule(
            name="time",
            pattern=r'\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?|noon|midnight)\b',
            entity_type="TIME",
            priority=5
        ))
        
        # Money patterns
        # Example: "$100", "$1,000.00", "€50"
        self.add_rule(Rule(
            name="money",
            pattern=r'\b([$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?|'
                   r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|euros|pounds|yen))\b',
            entity_type="MONEY",
            priority=5
        ))
        
        # Percentage patterns
        # Example: "25%", "3.14%"
        self.add_rule(Rule(
            name="percentage",
            pattern=r'\b(\d+(?:\.\d+)?%|'
                   r'\d+(?:\.\d+)?\s*percent)\b',
            entity_type="PERCENT",
            priority=5
        ))
        
        # Email patterns
        # Example: "user@example.com"
        self.add_rule(Rule(
            name="email",
            pattern=r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
            entity_type="CONTACT",
            priority=6
        ))
        
        # URL patterns
        # Example: "https://www.example.com", "www.example.com"
        self.add_rule(Rule(
            name="url",
            pattern=r'\b((?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)\b',
            entity_type="URL",
            priority=6
        ))
    
    def add_rule(self, rule: Rule):
        """
        Add a new rule to the system.
        
        Rules are sorted by priority (highest first) to ensure
        more specific rules are applied before general ones.
        
        Args:
            rule: Rule object to add
        """
        self.rules.append(rule)
        # Sort rules by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities from text using all rules.
        
        Process:
        1. Apply rules in priority order
        2. For each rule match, create entity
        3. Handle overlapping entities (keep higher priority)
        4. Return sorted list of entities
        
        Args:
            text: Input text string
            
        Returns:
            List of entity dictionaries with fields:
            - text: Entity text
            - type: Entity type
            - start: Start position
            - end: End position
            - rule: Rule name that matched
            - confidence: Confidence score
            
        Example:
            >>> ner = RuleBasedNER()
            >>> text = "Apple Inc. was founded by Steve Jobs."
            >>> entities = ner.extract_entities(text)
            >>> for entity in entities:
            ...     print(f"{entity['text']}: {entity['type']}")
            Apple Inc.: ORG
            Steve Jobs: PER
        """
        entities = []
        used_spans = set()  # Track used character spans to handle overlaps
        
        # Apply rules in priority order
        for rule in self.rules:
            # Find all matches for this rule
            for match in rule.compiled_pattern.finditer(text):
                # Get matched text and position
                matched_text = match.group(0)
                start = match.start()
                end = match.end()
                
                # Check if this span overlaps with already extracted entity
                span_range = range(start, end)
                if any(pos in used_spans for pos in span_range):
                    # Skip overlapping entity (lower priority rule)
                    continue
                
                # Check context requirements if specified
                if rule.context_required:
                    # Extract context window around match
                    context_start = max(0, start - 50)
                    context_end = min(len(text), end + 50)
                    context = text[context_start:context_end].lower()
                    
                    # Check if required context words are present
                    if not any(word.lower() in context for word in rule.context_required):
                        continue
                
                # Create entity
                entity = {
                    'text': matched_text,
                    'type': rule.entity_type,
                    'start': start,
                    'end': end,
                    'rule': rule.name,
                    'confidence': 1.0 / (10 - rule.priority)  # Higher priority = higher confidence
                }
                
                entities.append(entity)
                
                # Mark span as used
                used_spans.update(span_range)
        
        # Sort entities by position
        entities.sort(key=lambda e: e['start'])
        
        return entities
    
    def extract_by_pattern(self, text: str, pattern: str, entity_type: str) -> List[Dict]:
        """
        Extract entities using a custom pattern.
        
        This allows quick pattern testing without adding permanent rules.
        
        Args:
            text: Input text
            pattern: Regex pattern string
            entity_type: Type of entity to assign
            
        Returns:
            List of extracted entities
            
        Example:
            >>> ner = RuleBasedNER()
            >>> text = "Call me at 555-1234 or 555-5678"
            >>> entities = ner.extract_by_pattern(
            ...     text, r'\d{3}-\d{4}', 'PHONE'
            ... )
        """
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
        entities = []
        
        for match in compiled_pattern.finditer(text):
            entity = {
                'text': match.group(0),
                'type': entity_type,
                'start': match.start(),
                'end': match.end(),
                'rule': 'custom_pattern',
                'confidence': 0.8
            }
            entities.append(entity)
        
        return entities
    
    def get_entity_context(self, text: str, entity: Dict, window_size: int = 5) -> Dict:
        """
        Get context words around an entity.
        
        Context is useful for:
        - Validating entity type
        - Disambiguation
        - Feature extraction for ML models
        
        Args:
            text: Original text
            entity: Entity dictionary
            window_size: Number of words to include on each side
            
        Returns:
            Dictionary with left_context, entity_text, right_context
        """
        # Simple word-based tokenization
        words = text.split()
        entity_text = entity['text']
        
        # Find entity in word list
        entity_words = entity_text.split()
        entity_start_word = None
        
        for i in range(len(words) - len(entity_words) + 1):
            if words[i:i+len(entity_words)] == entity_words:
                entity_start_word = i
                break
        
        if entity_start_word is None:
            return {'left_context': [], 'entity': entity_text, 'right_context': []}
        
        # Extract context
        left_start = max(0, entity_start_word - window_size)
        left_context = words[left_start:entity_start_word]
        
        right_end = min(len(words), entity_start_word + len(entity_words) + window_size)
        right_context = words[entity_start_word + len(entity_words):right_end]
        
        return {
            'left_context': left_context,
            'entity': entity_text,
            'right_context': right_context
        }
    
    def visualize_entities(self, text: str, entities: List[Dict]):
        """
        Print text with entity annotations.
        
        This creates a visual representation of entities in text.
        
        Args:
            text: Original text
            entities: List of entity dictionaries
        """
        # Create annotated version
        result = []
        last_pos = 0
        
        for entity in sorted(entities, key=lambda e: e['start']):
            # Add text before entity
            result.append(text[last_pos:entity['start']])
            
            # Add annotated entity
            result.append(f"[{entity['text']}]_{entity['type']}")
            
            last_pos = entity['end']
        
        # Add remaining text
        result.append(text[last_pos:])
        
        print(''.join(result))


def demonstrate_rule_based_ner():
    """
    Comprehensive demonstration of rule-based NER.
    """
    print("="*70)
    print("Rule-Based Named Entity Recognition Demonstration")
    print("="*70)
    
    # Initialize NER system
    ner = RuleBasedNER()
    
    # Example 1: Basic entity extraction
    print("\n1. Basic Entity Extraction")
    print("-" * 70)
    
    text1 = ("Apple Inc. was founded by Steve Jobs in Cupertino, California "
             "on January 1, 1976. The company is valued at $2.5 trillion.")
    
    print(f"Text: {text1}\n")
    
    entities1 = ner.extract_entities(text1)
    
    print(f"Found {len(entities1)} entities:")
    print(f"{'Entity':<25} {'Type':<10} {'Position':<12} {'Rule'}")
    print("-" * 70)
    for entity in entities1:
        pos = f"({entity['start']}:{entity['end']})"
        print(f"{entity['text']:<25} {entity['type']:<10} {pos:<12} {entity['rule']}")
    
    # Visualize
    print("\nVisualized:")
    ner.visualize_entities(text1, entities1)
    
    # Example 2: Different entity types
    print("\n\n2. Various Entity Types")
    print("-" * 70)
    
    text2 = ("The meeting is scheduled for January 15, 2025 at 2:30 PM. "
             "Please contact us at support@company.com or visit www.company.com. "
             "The discount is 25% off the $199.99 price.")
    
    print(f"Text: {text2}\n")
    
    entities2 = ner.extract_entities(text2)
    
    # Group by type
    by_type = {}
    for entity in entities2:
        if entity['type'] not in by_type:
            by_type[entity['type']] = []
        by_type[entity['type']].append(entity['text'])
    
    print("Entities grouped by type:")
    for entity_type, texts in by_type.items():
        print(f"\n{entity_type}:")
        for text in texts:
            print(f"  - {text}")
    
    # Example 3: Custom pattern
    print("\n\n3. Custom Pattern Matching")
    print("-" * 70)
    
    text3 = "My phone numbers are 555-1234 and 555-5678. Call me!"
    
    print(f"Text: {text3}\n")
    
    # Extract phone numbers with custom pattern
    phone_pattern = r'\d{3}-\d{4}'
    phone_entities = ner.extract_by_pattern(text3, phone_pattern, 'PHONE')
    
    print(f"Found {len(phone_entities)} phone numbers:")
    for entity in phone_entities:
        print(f"  - {entity['text']}")
    
    # Example 4: Context extraction
    print("\n\n4. Entity Context")
    print("-" * 70)
    
    text4 = "Microsoft CEO Satya Nadella announced new products yesterday in Seattle."
    entities4 = ner.extract_entities(text4)
    
    print(f"Text: {text4}\n")
    
    for entity in entities4:
        context = ner.get_entity_context(text4, entity, window_size=3)
        print(f"\nEntity: {entity['text']} ({entity['type']})")
        print(f"Left context: {' '.join(context['left_context'])}")
        print(f"Right context: {' '.join(context['right_context'])}")
    
    # Example 5: Adding custom rules
    print("\n\n5. Adding Custom Rules")
    print("-" * 70)
    
    # Add custom rule for product names
    product_rule = Rule(
            name="product_names",
            pattern=r'\b(iPhone|iPad|MacBook|Windows|Android|Tesla Model [A-Z])\b',
            entity_type="PRODUCT",
            priority=6
        )
    
    ner.add_rule(product_rule)
    
    text5 = "I bought an iPhone 15 and a MacBook Pro. Also considering a Tesla Model 3."
    
    print(f"Text: {text5}\n")
    
    entities5 = ner.extract_entities(text5)
    products = [e for e in entities5 if e['type'] == 'PRODUCT']
    
    print(f"Found {len(products)} products:")
    for entity in products:
        print(f"  - {entity['text']}")
    
    # Example 6: Handling ambiguity
    print("\n\n6. Handling Ambiguous Cases")
    print("-" * 70)
    
    ambiguous_texts = [
        "Washington visited Washington.",  # Person vs. Location
        "I love Python programming.",     # Programming language vs. snake
        "Apple released new Apple products."  # Company vs. fruit
    ]
    
    print("Ambiguous cases (simple rules may incorrectly tag these):")
    for text in ambiguous_texts:
        print(f"\nText: {text}")
        entities = ner.extract_entities(text)
        if entities:
            for entity in entities:
                print(f"  Tagged: '{entity['text']}' as {entity['type']}")
        else:
            print("  No entities found")
        print("  Note: Context-aware rules or ML models needed for disambiguation")
    
    # Example 7: Performance characteristics
    print("\n\n7. Rule-Based NER Characteristics")
    print("-" * 70)
    
    print("\nStrengths:")
    print("  ✓ Very fast (no model inference)")
    print("  ✓ No training data needed")
    print("  ✓ Highly interpretable (can see exactly why entity was extracted)")
    print("  ✓ Easy to add domain-specific rules")
    print("  ✓ High precision for well-defined patterns")
    print("  ✓ Deterministic (same input always gives same output)")
    
    print("\nWeaknesses:")
    print("  ✗ Low recall (misses variations not covered by rules)")
    print("  ✗ Cannot generalize to unseen patterns")
    print("  ✗ Requires manual rule creation and maintenance")
    print("  ✗ Difficult to handle ambiguity")
    print("  ✗ Rule conflicts can be hard to debug")
    print("  ✗ Does not learn from data")
    
    print("\nBest used for:")
    print("  • Well-defined patterns (dates, emails, URLs)")
    print("  • Domain-specific entities with clear indicators")
    print("  • Quick prototyping before building ML models")
    print("  • Combining with ML models (rule-based + ML hybrid)")
    print("  • High-precision extraction where recall is less critical")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_rule_based_ner()
    
    print("\n" + "="*70)
    print("Demonstration complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Try modifying existing rules")
    print("2. Add custom rules for your domain")
    print("3. Combine with dictionary-based NER (next module)")
    print("4. Experiment with different priority levels")
