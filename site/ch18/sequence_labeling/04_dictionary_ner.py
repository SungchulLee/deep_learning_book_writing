"""
Dictionary-Based Named Entity Recognition
==========================================

This module implements dictionary/gazetteer-based NER using entity dictionaries
and lookup tables. This approach is simple, fast, and effective for well-defined
entity lists.

Learning Objectives:
- Build and use entity dictionaries (gazetteers)
- Implement efficient lookup algorithms
- Handle multi-word entities
- Combine with fuzzy matching
- Manage dictionary updates

Author: Educational purposes
Date: 2025
"""

import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from difflib import SequenceMatcher


class EntityDictionary:
    """
    Entity dictionary (gazetteer) for looking up known entities.
    
    A gazetteer is a list of known entities of specific types.
    For example:
    - Person names: ["Barack Obama", "Steve Jobs", ...]
    - Companies: ["Apple Inc.", "Microsoft", ...]
    - Locations: ["New York", "Paris", ...]
    """
    
    def __init__(self, entity_type: str):
        """
        Initialize entity dictionary.
        
        Args:
            entity_type: Type of entities in this dictionary (PER, ORG, LOC, etc.)
        """
        self.entity_type = entity_type
        self.entities: Set[str] = set()
        self.entities_lower: Dict[str, str] = {}  # lowercase -> original
        self.multi_word_entities: Set[str] = set()
        
    def add_entity(self, entity: str):
        """Add entity to dictionary."""
        self.entities.add(entity)
        self.entities_lower[entity.lower()] = entity
        
        # Track multi-word entities separately for optimization
        if len(entity.split()) > 1:
            self.multi_word_entities.add(entity)
    
    def add_entities(self, entities: List[str]):
        """Add multiple entities at once."""
        for entity in entities:
            self.add_entity(entity)
    
    def contains(self, text: str, case_sensitive: bool = False) -> bool:
        """Check if text is in dictionary."""
        if case_sensitive:
            return text in self.entities
        else:
            return text.lower() in self.entities_lower
    
    def __len__(self):
        """Return number of entities in dictionary."""
        return len(self.entities)


class DictionaryNER:
    """
    Dictionary-based NER system.
    
    Uses entity dictionaries (gazetteers) to identify entities through lookup.
    
    Process:
    1. Tokenize text
    2. Check each span against dictionaries
    3. Return matches as entities
    
    Advantages:
    - Very fast (O(1) lookup)
    - Perfect precision for dictionary entries
    - Easy to update with new entities
    - No training needed
    
    Disadvantages:
    - Zero recall for entities not in dictionary
    - Requires comprehensive dictionaries
    - Struggles with variations and misspellings
    - Cannot handle entity disambiguation
    """
    
    def __init__(self):
        """Initialize dictionary-based NER."""
        self.dictionaries: Dict[str, EntityDictionary] = {}
        self._initialize_default_dictionaries()
    
    def _initialize_default_dictionaries(self):
        """Initialize with sample dictionaries."""
        
        # Person names dictionary
        person_dict = EntityDictionary("PER")
        person_dict.add_entities([
            "Steve Jobs", "Bill Gates", "Elon Musk",
            "Barack Obama", "Donald Trump", "Joe Biden",
            "Mark Zuckerberg", "Jeff Bezos", "Tim Cook",
            "Satya Nadella", "Sundar Pichai"
        ])
        self.dictionaries["PER"] = person_dict
        
        # Organization dictionary
        org_dict = EntityDictionary("ORG")
        org_dict.add_entities([
            "Apple", "Microsoft", "Google", "Amazon", "Facebook", "Meta",
            "Tesla", "SpaceX", "IBM", "Intel", "Nvidia",
            "Harvard University", "Stanford University", "MIT"
        ])
        self.dictionaries["ORG"] = org_dict
        
        # Location dictionary
        loc_dict = EntityDictionary("LOC")
        loc_dict.add_entities([
            "New York", "Los Angeles", "Chicago", "San Francisco",
            "London", "Paris", "Tokyo", "Beijing", "Sydney",
            "California", "Texas", "Florida",
            "United States", "China", "Japan", "Germany", "France"
        ])
        self.dictionaries["LOC"] = loc_dict
    
    def add_dictionary(self, entity_type: str, entities: List[str]):
        """Add or update dictionary for entity type."""
        if entity_type not in self.dictionaries:
            self.dictionaries[entity_type] = EntityDictionary(entity_type)
        self.dictionaries[entity_type].add_entities(entities)
    
    def extract_entities(self, text: str, case_sensitive: bool = False) -> List[Dict]:
        """
        Extract entities using dictionary lookup.
        
        Args:
            text: Input text
            case_sensitive: Whether to use case-sensitive matching
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        words = text.split()
        
        # Check all possible n-grams (up to 5 words)
        for n in range(5, 0, -1):
            for i in range(len(words) - n + 1):
                span = " ".join(words[i:i+n])
                
                # Check against all dictionaries
                for entity_type, dictionary in self.dictionaries.items():
                    if dictionary.contains(span, case_sensitive):
                        # Find position in original text
                        start = text.find(span)
                        if start != -1:
                            entity = {
                                "text": span,
                                "type": entity_type,
                                "start": start,
                                "end": start + len(span),
                                "confidence": 1.0
                            }
                            entities.append(entity)
        
        # Remove duplicates and overlaps
        entities = self._remove_overlaps(entities)
        return entities
    
    def _remove_overlaps(self, entities: List[Dict]) -> List[Dict]:
        """Remove overlapping entities, keeping longer ones."""
        if not entities:
            return []
        
        # Sort by start position, then by length (longer first)
        entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
        
        filtered = []
        for entity in entities:
            # Check if overlaps with any already added entity
            overlaps = False
            for added in filtered:
                if not (entity["end"] <= added["start"] or entity["start"] >= added["end"]):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered


def demonstrate_dictionary_ner():
    """Demonstration of dictionary-based NER."""
    print("="*70)
    print("Dictionary-Based NER Demonstration")
    print("="*70)
    
    ner = DictionaryNER()
    
    text = "Steve Jobs founded Apple in California. Bill Gates started Microsoft."
    print(f"\nText: {text}")
    
    entities = ner.extract_entities(text)
    print(f"\nFound {len(entities)} entities:")
    for entity in entities:
        print(f"  - {entity['text']} ({entity['type']})")


if __name__ == "__main__":
    demonstrate_dictionary_ner()
