"""
NER Dataset Creation and Formatting
====================================

Tools for creating and formatting NER datasets.

Supports:
- IOB tagging
- Data splitting
- Format conversion

Author: Educational purposes
Date: 2025
"""

from typing import List, Tuple, Dict
import random


class NERDatasetBuilder:
    """Build NER datasets with proper formatting."""
    
    def __init__(self):
        """Initialize dataset builder."""
        self.samples = []
    
    def add_sample(self, text: str, entities: List[Dict]):
        """
        Add a sample to the dataset.
        
        Args:
            text: Text string
            entities: List of entity dicts with 'text', 'type', 'start', 'end'
        """
        self.samples.append({
            'text': text,
            'entities': entities
        })
    
    def to_iob_format(self) -> List[Tuple[List[str], List[str]]]:
        """
        Convert dataset to IOB format.
        
        Returns:
            List of (tokens, labels) tuples
        """
        iob_data = []
        
        for sample in self.samples:
            text = sample['text']
            entities = sample['entities']
            
            # Simple tokenization
            tokens = text.split()
            labels = ['O'] * len(tokens)
            
            # Assign labels (simplified - assumes tokens match)
            for entity in entities:
                entity_tokens = entity['text'].split()
                # Find entity position in token list
                for i in range(len(tokens) - len(entity_tokens) + 1):
                    if tokens[i:i+len(entity_tokens)] == entity_tokens:
                        labels[i] = f"B-{entity['type']}"
                        for j in range(1, len(entity_tokens)):
                            labels[i+j] = f"I-{entity['type']}"
                        break
            
            iob_data.append((tokens, labels))
        
        return iob_data
    
    def train_test_split(self, test_size: float = 0.2, 
                        random_seed: int = 42) -> Tuple[List, List]:
        """
        Split dataset into train and test sets.
        
        Args:
            test_size: Fraction of data for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            (train_samples, test_samples) tuple
        """
        random.seed(random_seed)
        samples_copy = self.samples.copy()
        random.shuffle(samples_copy)
        
        split_idx = int(len(samples_copy) * (1 - test_size))
        train = samples_copy[:split_idx]
        test = samples_copy[split_idx:]
        
        return train, test


if __name__ == "__main__":
    # Example
    builder = NERDatasetBuilder()
    
    builder.add_sample(
        "Steve Jobs founded Apple Inc.",
        [
            {'text': 'Steve Jobs', 'type': 'PER', 'start': 0, 'end': 10},
            {'text': 'Apple Inc.', 'type': 'ORG', 'start': 19, 'end': 29}
        ]
    )
    
    iob_data = builder.to_iob_format()
    print(f"IOB format: {iob_data}")
