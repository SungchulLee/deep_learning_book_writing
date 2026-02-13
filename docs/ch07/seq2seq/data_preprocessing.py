"""
Data Preprocessing Utilities for Seq2Seq Models
Includes tokenization, vocabulary building, and data loading
"""

import re
from collections import Counter
import pickle
from pathlib import Path
import unicodedata


class Tokenizer:
    """
    Simple tokenizer for text preprocessing
    """
    
    def __init__(self, lower=True, remove_punct=False):
        self.lower = lower
        self.remove_punct = remove_punct
    
    def tokenize(self, text):
        """Tokenize text into tokens"""
        if self.lower:
            text = text.lower()
        
        if self.remove_punct:
            # Remove punctuation
            text = re.sub(r'[^\w\s]', '', text)
        else:
            # Separate punctuation with spaces
            text = re.sub(r'([.,!?;:])', r' \1 ', text)
        
        # Split on whitespace and filter empty strings
        tokens = text.split()
        tokens = [t for t in tokens if t]
        
        return tokens
    
    def detokenize(self, tokens):
        """Convert tokens back to text"""
        text = ' '.join(tokens)
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text


class Vocabulary:
    """
    Vocabulary class for managing token-to-index mappings
    
    Args:
        max_size: Maximum vocabulary size (None for unlimited)
        min_freq: Minimum frequency for a token to be included
        special_tokens: List of special tokens
    """
    
    def __init__(self, max_size=None, min_freq=1, 
                 special_tokens=['<pad>', '<sos>', '<eos>', '<unk>']):
        self.max_size = max_size
        self.min_freq = min_freq
        self.special_tokens = special_tokens
        
        # Token to index mapping
        self.token2idx = {}
        self.idx2token = {}
        
        # Add special tokens
        for idx, token in enumerate(special_tokens):
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        
        self.pad_idx = self.token2idx.get('<pad>', 0)
        self.sos_idx = self.token2idx.get('<sos>', 1)
        self.eos_idx = self.token2idx.get('<eos>', 2)
        self.unk_idx = self.token2idx.get('<unk>', 3)
    
    def build_vocab(self, texts, tokenizer=None):
        """
        Build vocabulary from texts
        
        Args:
            texts: List of text strings or list of token lists
            tokenizer: Tokenizer function (optional)
        """
        # Count token frequencies
        counter = Counter()
        
        for text in texts:
            if tokenizer is not None:
                tokens = tokenizer(text)
            elif isinstance(text, str):
                tokens = text.split()
            else:
                tokens = text
            
            counter.update(tokens)
        
        # Filter by frequency
        tokens = [token for token, freq in counter.items() if freq >= self.min_freq]
        
        # Sort by frequency (most common first)
        tokens = sorted(tokens, key=lambda t: counter[t], reverse=True)
        
        # Limit vocabulary size
        if self.max_size is not None:
            tokens = tokens[:self.max_size - len(self.special_tokens)]
        
        # Add tokens to vocabulary
        for token in tokens:
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
    
    def encode(self, tokens):
        """Convert tokens to indices"""
        if isinstance(tokens, str):
            tokens = tokens.split()
        return [self.token2idx.get(token, self.unk_idx) for token in tokens]
    
    def decode(self, indices, skip_special=True):
        """Convert indices to tokens"""
        tokens = []
        for idx in indices:
            if skip_special and idx in [self.pad_idx, self.sos_idx, self.eos_idx]:
                if idx == self.eos_idx:
                    break
                continue
            tokens.append(self.idx2token.get(idx, '<unk>'))
        return tokens
    
    def __len__(self):
        return len(self.token2idx)
    
    def save(self, path):
        """Save vocabulary to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'token2idx': self.token2idx,
                'idx2token': self.idx2token,
                'max_size': self.max_size,
                'min_freq': self.min_freq,
                'special_tokens': self.special_tokens
            }, f)
    
    @classmethod
    def load(cls, path):
        """Load vocabulary from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls(
            max_size=data['max_size'],
            min_freq=data['min_freq'],
            special_tokens=data['special_tokens']
        )
        vocab.token2idx = data['token2idx']
        vocab.idx2token = data['idx2token']
        
        return vocab


class ParallelDataset:
    """
    Parallel dataset for sequence-to-sequence tasks
    
    Args:
        src_texts: List of source texts
        trg_texts: List of target texts
        src_vocab: Source vocabulary
        trg_vocab: Target vocabulary
        src_tokenizer: Source tokenizer
        trg_tokenizer: Target tokenizer
        max_len: Maximum sequence length
    """
    
    def __init__(self, src_texts, trg_texts, src_vocab, trg_vocab,
                 src_tokenizer=None, trg_tokenizer=None, max_len=None):
        self.src_texts = src_texts
        self.trg_texts = trg_texts
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_tokenizer = src_tokenizer or Tokenizer()
        self.trg_tokenizer = trg_tokenizer or Tokenizer()
        self.max_len = max_len
    
    def process_pair(self, src_text, trg_text):
        """Process a source-target pair"""
        # Tokenize
        src_tokens = self.src_tokenizer.tokenize(src_text)
        trg_tokens = self.trg_tokenizer.tokenize(trg_text)
        
        # Truncate if necessary
        if self.max_len is not None:
            src_tokens = src_tokens[:self.max_len]
            trg_tokens = trg_tokens[:self.max_len]
        
        # Encode
        src_indices = self.src_vocab.encode(src_tokens)
        trg_indices = [self.trg_vocab.sos_idx] + self.trg_vocab.encode(trg_tokens) + [self.trg_vocab.eos_idx]
        
        return src_indices, trg_indices
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        return self.process_pair(self.src_texts[idx], self.trg_texts[idx])


def load_parallel_data(src_path, trg_path, max_samples=None):
    """
    Load parallel data from files
    
    Args:
        src_path: Path to source file
        trg_path: Path to target file
        max_samples: Maximum number of samples to load
        
    Returns:
        src_texts: List of source texts
        trg_texts: List of target texts
    """
    with open(src_path, 'r', encoding='utf-8') as f:
        src_texts = [line.strip() for line in f]
    
    with open(trg_path, 'r', encoding='utf-8') as f:
        trg_texts = [line.strip() for line in f]
    
    # Ensure same length
    assert len(src_texts) == len(trg_texts), "Source and target files must have same length"
    
    # Limit samples if specified
    if max_samples is not None:
        src_texts = src_texts[:max_samples]
        trg_texts = trg_texts[:max_samples]
    
    return src_texts, trg_texts


def normalize_text(text):
    """
    Normalize text (unicode normalization, etc.)
    
    Args:
        text: Input text
        
    Returns:
        normalized_text: Normalized text
    """
    # Unicode normalization
    text = unicodedata.normalize('NFD', text)
    
    # Remove accents
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    return text


def split_data(data, train_ratio=0.8, val_ratio=0.1):
    """
    Split data into train, validation, and test sets
    
    Args:
        data: List or tuple of data
        train_ratio: Proportion of training data
        val_ratio: Proportion of validation data
        
    Returns:
        train_data, val_data, test_data: Split datasets
    """
    if isinstance(data, tuple):
        # Multiple datasets (e.g., source and target)
        total_len = len(data[0])
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)
        
        train_data = tuple(d[:train_len] for d in data)
        val_data = tuple(d[train_len:train_len + val_len] for d in data)
        test_data = tuple(d[train_len + val_len:] for d in data)
    else:
        # Single dataset
        total_len = len(data)
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)
        
        train_data = data[:train_len]
        val_data = data[train_len:train_len + val_len]
        test_data = data[train_len + val_len:]
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Example")
    print("-" * 50)
    
    # Sample data
    src_texts = [
        "Hello, how are you?",
        "I am doing well, thank you.",
        "What is your name?",
        "My name is Claude.",
        "Nice to meet you!"
    ]
    
    trg_texts = [
        "Bonjour, comment allez-vous?",
        "Je vais bien, merci.",
        "Quel est votre nom?",
        "Je m'appelle Claude.",
        "Enchanté de vous rencontrer!"
    ]
    
    # Create tokenizer
    tokenizer = Tokenizer(lower=True, remove_punct=False)
    
    # Build vocabularies
    print("\nBuilding vocabularies...")
    src_vocab = Vocabulary(max_size=1000, min_freq=1)
    src_vocab.build_vocab(src_texts, tokenizer.tokenize)
    
    trg_vocab = Vocabulary(max_size=1000, min_freq=1)
    trg_vocab.build_vocab(trg_texts, tokenizer.tokenize)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(trg_vocab)}")
    
    # Test encoding/decoding
    print("\nTesting encoding/decoding...")
    test_text = "Hello, how are you?"
    tokens = tokenizer.tokenize(test_text)
    indices = src_vocab.encode(tokens)
    decoded = src_vocab.decode(indices, skip_special=False)
    
    print(f"Original: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Indices: {indices}")
    print(f"Decoded: {decoded}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = ParallelDataset(
        src_texts, trg_texts, src_vocab, trg_vocab,
        src_tokenizer=tokenizer, trg_tokenizer=tokenizer
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test dataset
    src_indices, trg_indices = dataset[0]
    print(f"\nSample data:")
    print(f"Source: {src_texts[0]}")
    print(f"Source indices: {src_indices}")
    print(f"Target: {trg_texts[0]}")
    print(f"Target indices: {trg_indices}")
    
    # Save vocabularies
    print("\nSaving vocabularies...")
    src_vocab.save('src_vocab.pkl')
    trg_vocab.save('trg_vocab.pkl')
    print("Vocabularies saved!")
    
    # Load vocabularies
    print("\nLoading vocabularies...")
    loaded_src_vocab = Vocabulary.load('src_vocab.pkl')
    loaded_trg_vocab = Vocabulary.load('trg_vocab.pkl')
    print(f"Loaded source vocabulary size: {len(loaded_src_vocab)}")
    print(f"Loaded target vocabulary size: {len(loaded_trg_vocab)}")
