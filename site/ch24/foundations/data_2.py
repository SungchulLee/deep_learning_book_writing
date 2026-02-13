"""
Data Utilities for Character-Level Language Model

This module handles text processing for training autoregressive language models
that generate text one character at a time.
"""

import torch
import numpy as np
from typing import Tuple, List


class CharacterDataset:
    """
    Dataset for character-level language modeling.
    
    This class:
    1. Converts text to numerical indices (tokenization)
    2. Creates input-output pairs for training
    3. Provides utilities for encoding/decoding
    """
    
    def __init__(self, text: str, sequence_length: int = 50):
        """
        Initialize the character dataset.
        
        Args:
            text: Input text string to learn from
            sequence_length: Length of character sequences for training
        """
        self.text = text
        self.sequence_length = sequence_length
        
        # Get unique characters (vocabulary)
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Create mappings: character <-> integer
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        print(f"Text length: {len(text)} characters")
        print(f"Vocabulary size: {self.vocab_size} unique characters")
        print(f"Characters: {self.chars}")
        
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of integers.
        
        Args:
            text: String to encode
            
        Returns:
            List of integer indices
        """
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices: List[int]) -> str:
        """
        Convert list of integers back to text.
        
        Args:
            indices: List of integer indices
            
        Returns:
            Decoded string
        """
        return ''.join([self.idx_to_char[idx] for idx in indices])
    
    def create_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create training sequences from text.
        
        For autoregressive language modeling:
        - Input: sequence of characters [c1, c2, ..., cn]
        - Target: next character for each position [c2, c3, ..., cn+1]
        
        Returns:
            Tuple of (inputs, targets) as tensors
        """
        # Encode entire text
        encoded_text = self.encode(self.text)
        
        sequences = []
        targets = []
        
        # Slide window across text
        for i in range(len(encoded_text) - self.sequence_length):
            # Input: characters from i to i+sequence_length
            seq = encoded_text[i:i + self.sequence_length]
            
            # Target: the next character after the sequence
            target = encoded_text[i + self.sequence_length]
            
            sequences.append(seq)
            targets.append(target)
        
        # Convert to tensors
        X = torch.LongTensor(sequences)
        y = torch.LongTensor(targets)
        
        return X, y


def load_sample_text() -> str:
    """
    Load a sample text for demonstration.
    
    Returns:
        Sample text string
    """
    # Sample text: Shakespeare-like passage
    sample = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die—to sleep,
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to: 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep, perchance to dream—ay, there's the rub:
    For in that sleep of death what dreams may come,
    When we have shuffled off this mortal coil,
    Must give us pause—there's the respect
    That makes calamity of so long life.
    """
    
    return sample.strip()


def load_text_file(filepath: str) -> str:
    """
    Load text from a file.
    
    Args:
        filepath: Path to text file
        
    Returns:
        Text content as string
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def train_test_split(X: torch.Tensor, 
                     y: torch.Tensor, 
                     train_ratio: float = 0.9) -> Tuple[torch.Tensor, ...]:
    """
    Split data into train and test sets.
    
    For text, we typically use a higher train ratio (e.g., 90/10)
    since we want as much training data as possible.
    
    Args:
        X: Input sequences
        y: Target characters
        train_ratio: Fraction for training
        
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    n = len(X)
    split_idx = int(n * train_ratio)
    
    # For text, we don't shuffle to maintain some coherence
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    """
    Demo: Process sample text
    """
    
    # Load sample text
    text = load_sample_text()
    print("Sample text:")
    print(text[:200] + "...")
    print()
    
    # Create dataset
    dataset = CharacterDataset(text, sequence_length=20)
    
    # Test encoding/decoding
    sample_text = "Hello, World!"
    encoded = dataset.encode(sample_text)
    decoded = dataset.decode(encoded)
    
    print(f"\nOriginal: {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Match: {sample_text == decoded}")
    
    # Create sequences
    X, y = dataset.create_sequences()
    print(f"\nCreated {len(X)} training sequences")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Show example
    print(f"\nExample sequence:")
    print(f"Input (encoded): {X[0]}")
    print(f"Input (decoded): '{dataset.decode(X[0].tolist())}'")
    print(f"Target (encoded): {y[0]}")
    print(f"Target (decoded): '{dataset.decode([y[0].item()])}'")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_ratio=0.9)
    print(f"\nData split:")
    print(f"Train: {len(X_train)} sequences")
    print(f"Test: {len(X_test)} sequences")
