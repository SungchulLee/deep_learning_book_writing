"""
Feature Extraction for Traditional NER
=======================================

Feature engineering for traditional ML-based NER (CRF, MaxEnt, etc.).

Features extracted:
- Word-level features
- Character-level features
- Orthographic features
- Context features
- Gazetteer features

Author: Educational purposes
Date: 2025
"""

from typing import List, Dict, Set
import re


class FeatureExtractor:
    """Extract features for traditional ML-based NER."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.gazetteers = {}
    
    def extract_token_features(self, tokens: List[str], index: int, 
                              window_size: int = 2) -> Dict:
        """
        Extract comprehensive features for a token.
        
        Args:
            tokens: List of tokens in sentence
            index: Index of current token
            window_size: Context window size
            
        Returns:
            Dictionary of features
        """
        token = tokens[index]
        features = {}
        
        # Basic word features
        features['word'] = token.lower()
        features['word_length'] = len(token)
        
        # Orthographic features
        features['is_capitalized'] = token[0].isupper() if token else False
        features['is_all_caps'] = token.isupper()
        features['is_all_lower'] = token.islower()
        features['is_title'] = token.istitle()
        features['is_alphanumeric'] = token.isalnum()
        features['is_alpha'] = token.isalpha()
        features['is_digit'] = token.isdigit()
        
        # Word shape features
        features['word_shape'] = self.get_word_shape(token)
        features['short_word_shape'] = self.get_word_shape(token, short=True)
        
        # Prefix and suffix features
        for n in range(1, min(5, len(token) + 1)):
            features[f'prefix_{n}'] = token[:n].lower()
            features[f'suffix_{n}'] = token[-n:].lower()
        
        # Character-level features
        features['contains_hyphen'] = '-' in token
        features['contains_digit'] = any(c.isdigit() for c in token)
        features['contains_upper'] = any(c.isupper() for c in token)
        
        # Context features (previous tokens)
        for i in range(1, window_size + 1):
            if index - i >= 0:
                prev_token = tokens[index - i]
                features[f'prev_{i}_word'] = prev_token.lower()
                features[f'prev_{i}_is_cap'] = prev_token[0].isupper() if prev_token else False
        
        # Context features (next tokens)
        for i in range(1, window_size + 1):
            if index + i < len(tokens):
                next_token = tokens[index + i]
                features[f'next_{i}_word'] = next_token.lower()
                features[f'next_{i}_is_cap'] = next_token[0].isupper() if next_token else False
        
        # Position features
        features['is_first'] = (index == 0)
        features['is_last'] = (index == len(tokens) - 1)
        
        return features
    
    @staticmethod
    def get_word_shape(word: str, short: bool = False) -> str:
        """
        Get word shape representation.
        
        Maps characters to shape codes:
        - Uppercase: 'X'
        - Lowercase: 'x'
        - Digit: 'd'
        - Other: 'c'
        
        If short=True, consecutive same chars are collapsed.
        
        Example:
            "iPhone5" -> "xXxxxxd" (long) or "xXxd" (short)
        """
        shape = []
        for char in word:
            if char.isupper():
                shape.append('X')
            elif char.islower():
                shape.append('x')
            elif char.isdigit():
                shape.append('d')
            else:
                shape.append('c')
        
        shape_str = ''.join(shape)
        
        if short:
            # Collapse consecutive same characters
            if not shape_str:
                return shape_str
            compressed = [shape_str[0]]
            for char in shape_str[1:]:
                if char != compressed[-1]:
                    compressed.append(char)
            return ''.join(compressed)
        
        return shape_str


if __name__ == "__main__":
    # Example
    extractor = FeatureExtractor()
    tokens = ["Steve", "Jobs", "founded", "Apple", "Inc", "."]
    
    for i, token in enumerate(tokens):
        features = extractor.extract_token_features(tokens, i)
        print(f"\nToken: {token}")
        print(f"Features: {len(features)} features extracted")
        print(f"Word shape: {features['word_shape']}")
        print(f"Is capitalized: {features['is_capitalized']}")
