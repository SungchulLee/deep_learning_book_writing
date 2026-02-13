"""
CRF-based Named Entity Recognition
===================================

Conditional Random Fields for sequence labeling with feature engineering.

This implementation uses sklearn-crfsuite for training CRF models.

Key concepts:
- Feature extraction for tokens
- Sequence labeling with CRF
- Training and evaluation

Author: Educational purposes  
Date: 2025
"""

import sklearn_crfsuite
from sklearn_crfsuite import metrics
from typing import List, Dict, Tuple


class CRF_NER:
    """CRF-based NER with feature engineering."""
    
    def __init__(self):
        """Initialize CRF model."""
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
    
    def word_features(self, sentence: List[str], i: int) -> Dict:
        """
        Extract features for token at position i.
        
        Features:
        - Word itself (lowercased)
        - Word capitalization patterns
        - Word shape
        - Prefixes and suffixes
        - Context words
        """
        word = sentence[i]
        
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }
        
        # Context features
        if i > 0:
            word_prev = sentence[i-1]
            features.update({
                '-1:word.lower()': word_prev.lower(),
                '-1:word.istitle()': word_prev.istitle(),
                '-1:word.isupper()': word_prev.isupper(),
            })
        else:
            features['BOS'] = True  # Beginning of sentence
        
        if i < len(sentence) - 1:
            word_next = sentence[i+1]
            features.update({
                '+1:word.lower()': word_next.lower(),
                '+1:word.istitle()': word_next.istitle(),
                '+1:word.isupper()': word_next.isupper(),
            })
        else:
            features['EOS'] = True  # End of sentence
        
        return features
    
    def sentence_features(self, sentence: List[str]) -> List[Dict]:
        """Extract features for all tokens in sentence."""
        return [self.word_features(sentence, i) for i in range(len(sentence))]
    
    def train(self, X_train: List[List[str]], y_train: List[List[str]]):
        """
        Train CRF model.
        
        Args:
            X_train: List of sentences (each sentence is list of tokens)
            y_train: List of label sequences (each is list of labels)
        """
        X_train_features = [self.sentence_features(s) for s in X_train]
        self.model.fit(X_train_features, y_train)
    
    def predict(self, X_test: List[List[str]]) -> List[List[str]]:
        """Predict labels for test sentences."""
        X_test_features = [self.sentence_features(s) for s in X_test]
        return self.model.predict(X_test_features)


if __name__ == "__main__":
    # Example usage
    ner = CRF_NER()
    
    # Sample data
    X_train = [["Steve", "Jobs", "founded", "Apple"]]
    y_train = [["B-PER", "I-PER", "O", "B-ORG"]]
    
    ner.train(X_train, y_train)
    
    X_test = [["Bill", "Gates", "started", "Microsoft"]]
    predictions = ner.predict(X_test)
    
    print("Predictions:", predictions)
