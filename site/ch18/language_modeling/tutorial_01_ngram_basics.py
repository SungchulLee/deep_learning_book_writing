"""
Tutorial 01: N-gram Language Models - Basics
==============================================

This tutorial introduces the fundamental concepts of n-gram language models,
the simplest and most interpretable approach to language modeling.

Learning Objectives:
--------------------
1. Understand what n-grams are and why they're useful
2. Build unigram, bigram, and trigram models
3. Calculate probabilities using Maximum Likelihood Estimation (MLE)
4. Understand the limitations of n-gram models

Mathematical Background:
------------------------
An n-gram is a contiguous sequence of n words from a text.

Unigram Model (n=1):
- P(w) = count(w) / total_words
- Assumes words are independent

Bigram Model (n=2):
- P(w_i | w_{i-1}) = count(w_{i-1}, w_i) / count(w_{i-1})
- Uses one word of context

Trigram Model (n=3):
- P(w_i | w_{i-2}, w_{i-1}) = count(w_{i-2}, w_{i-1}, w_i) / count(w_{i-2}, w_{i-1})
- Uses two words of context

Sequence Probability:
- P(w_1, w_2, ..., w_n) = P(w_1) * P(w_2|w_1) * P(w_3|w_1,w_2) * ... * P(w_n|w_{n-k},...,w_{n-1})
"""

import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import math


class UnigramModel:
    """
    Unigram language model that assumes words are independent.
    
    The probability of a word is simply its frequency in the training corpus:
    P(w) = count(w) / total_words
    
    Attributes:
        word_counts (Counter): Count of each word in vocabulary
        total_words (int): Total number of words in training corpus
        vocab (set): Set of all unique words seen during training
    """
    
    def __init__(self):
        """Initialize empty unigram model."""
        self.word_counts = Counter()
        self.total_words = 0
        self.vocab = set()
    
    def train(self, corpus: List[str]) -> None:
        """
        Train unigram model on a corpus of text.
        
        Args:
            corpus: List of sentences (strings) to train on
            
        Example:
            >>> model = UnigramModel()
            >>> model.train(["the cat sat", "the dog ran"])
        """
        # Tokenize and count all words in the corpus
        for sentence in corpus:
            # Convert to lowercase and split into words
            words = sentence.lower().split()
            
            # Update word counts
            self.word_counts.update(words)
            
            # Update total word count
            self.total_words += len(words)
            
            # Update vocabulary
            self.vocab.update(words)
        
        print(f"Trained unigram model on {self.total_words} words")
        print(f"Vocabulary size: {len(self.vocab)} unique words")
    
    def probability(self, word: str) -> float:
        """
        Calculate probability of a word using Maximum Likelihood Estimation.
        
        P(word) = count(word) / total_words
        
        Args:
            word: Word to calculate probability for
            
        Returns:
            Probability of the word (0.0 if unseen)
            
        Example:
            >>> prob = model.probability("cat")
        """
        word = word.lower()
        
        # If word not in vocabulary, return 0 (will be improved with smoothing)
        if word not in self.vocab:
            return 0.0
        
        # Calculate MLE probability
        return self.word_counts[word] / self.total_words
    
    def log_probability(self, word: str) -> float:
        """
        Calculate log probability (base 2) of a word.
        
        Log probabilities are more numerically stable for long sequences.
        
        Args:
            word: Word to calculate log probability for
            
        Returns:
            Log probability (returns -inf for unseen words)
        """
        prob = self.probability(word)
        
        # Return negative infinity for zero probability
        if prob == 0.0:
            return float('-inf')
        
        # Return log base 2 probability
        return math.log2(prob)
    
    def sentence_probability(self, sentence: str) -> float:
        """
        Calculate probability of a sentence under independence assumption.
        
        P(sentence) = P(w_1) * P(w_2) * ... * P(w_n)
        
        Args:
            sentence: Input sentence
            
        Returns:
            Probability of the sentence
        """
        words = sentence.lower().split()
        
        # Start with probability 1.0
        prob = 1.0
        
        # Multiply probabilities of each word
        for word in words:
            prob *= self.probability(word)
            
            # Early stopping if probability becomes 0
            if prob == 0.0:
                return 0.0
        
        return prob
    
    def sentence_log_probability(self, sentence: str) -> float:
        """
        Calculate log probability of a sentence.
        
        log P(sentence) = log P(w_1) + log P(w_2) + ... + log P(w_n)
        
        This is more numerically stable than multiplying probabilities.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Log probability of the sentence
        """
        words = sentence.lower().split()
        
        # Start with log probability 0.0
        log_prob = 0.0
        
        # Sum log probabilities of each word
        for word in words:
            log_prob += self.log_probability(word)
            
            # Early stopping if we hit negative infinity
            if log_prob == float('-inf'):
                return float('-inf')
        
        return log_prob


class BigramModel:
    """
    Bigram language model that conditions each word on the previous word.
    
    The probability of a word given previous context:
    P(w_i | w_{i-1}) = count(w_{i-1}, w_i) / count(w_{i-1})
    
    Attributes:
        bigram_counts (defaultdict): Count of each bigram (w_{i-1}, w_i)
        unigram_counts (Counter): Count of each unigram (for normalization)
        vocab (set): Set of all unique words seen during training
    """
    
    def __init__(self):
        """Initialize empty bigram model."""
        # Use nested defaultdict for bigram counts
        # bigram_counts[w1][w2] = count of (w1, w2)
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()
        
        # Special tokens for sentence boundaries
        self.START_TOKEN = "<s>"
        self.END_TOKEN = "</s>"
    
    def train(self, corpus: List[str]) -> None:
        """
        Train bigram model on a corpus of text.
        
        Args:
            corpus: List of sentences (strings) to train on
            
        Example:
            >>> model = BigramModel()
            >>> model.train(["the cat sat", "the dog ran"])
        """
        total_bigrams = 0
        
        for sentence in corpus:
            # Convert to lowercase and split into words
            words = sentence.lower().split()
            
            # Add sentence boundary tokens
            words = [self.START_TOKEN] + words + [self.END_TOKEN]
            
            # Extract bigrams and update counts
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                
                # Update bigram count
                self.bigram_counts[w1][w2] += 1
                
                # Update unigram count (for first word in bigram)
                self.unigram_counts[w1] += 1
                
                # Update vocabulary (excluding boundary tokens)
                if w1 != self.START_TOKEN:
                    self.vocab.add(w1)
                if w2 != self.END_TOKEN:
                    self.vocab.add(w2)
                
                total_bigrams += 1
        
        print(f"Trained bigram model on {total_bigrams} bigrams")
        print(f"Vocabulary size: {len(self.vocab)} unique words")
    
    def probability(self, word: str, context: str) -> float:
        """
        Calculate conditional probability P(word | context).
        
        P(w_i | w_{i-1}) = count(w_{i-1}, w_i) / count(w_{i-1})
        
        Args:
            word: Current word
            context: Previous word (context)
            
        Returns:
            Conditional probability
            
        Example:
            >>> prob = model.probability("sat", "cat")
        """
        word = word.lower()
        context = context.lower()
        
        # If context hasn't been seen, return 0
        if context not in self.unigram_counts:
            return 0.0
        
        # Get count of bigram (context, word)
        bigram_count = self.bigram_counts[context][word]
        
        # Get count of context word
        context_count = self.unigram_counts[context]
        
        # Calculate conditional probability
        return bigram_count / context_count if context_count > 0 else 0.0
    
    def log_probability(self, word: str, context: str) -> float:
        """
        Calculate log conditional probability.
        
        Args:
            word: Current word
            context: Previous word
            
        Returns:
            Log conditional probability
        """
        prob = self.probability(word, context)
        
        if prob == 0.0:
            return float('-inf')
        
        return math.log2(prob)
    
    def sentence_probability(self, sentence: str) -> float:
        """
        Calculate probability of a sentence.
        
        P(sentence) = P(w_1 | <s>) * P(w_2 | w_1) * ... * P(</s> | w_n)
        
        Args:
            sentence: Input sentence
            
        Returns:
            Probability of the sentence
        """
        words = sentence.lower().split()
        words = [self.START_TOKEN] + words + [self.END_TOKEN]
        
        prob = 1.0
        
        # Calculate product of conditional probabilities
        for i in range(len(words) - 1):
            context, word = words[i], words[i + 1]
            prob *= self.probability(word, context)
            
            if prob == 0.0:
                return 0.0
        
        return prob
    
    def sentence_log_probability(self, sentence: str) -> float:
        """
        Calculate log probability of a sentence.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Log probability of the sentence
        """
        words = sentence.lower().split()
        words = [self.START_TOKEN] + words + [self.END_TOKEN]
        
        log_prob = 0.0
        
        # Calculate sum of log conditional probabilities
        for i in range(len(words) - 1):
            context, word = words[i], words[i + 1]
            log_prob += self.log_probability(word, context)
            
            if log_prob == float('-inf'):
                return float('-inf')
        
        return log_prob


class TrigramModel:
    """
    Trigram language model that conditions each word on two previous words.
    
    The probability of a word given previous context:
    P(w_i | w_{i-2}, w_{i-1}) = count(w_{i-2}, w_{i-1}, w_i) / count(w_{i-2}, w_{i-1})
    
    Attributes:
        trigram_counts (nested defaultdict): Count of each trigram
        bigram_counts (defaultdict): Count of each bigram (for normalization)
        vocab (set): Set of all unique words seen during training
    """
    
    def __init__(self):
        """Initialize empty trigram model."""
        # trigram_counts[w1][w2][w3] = count of (w1, w2, w3)
        self.trigram_counts = defaultdict(lambda: defaultdict(Counter))
        # bigram_counts[w1][w2] = count of (w1, w2)
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()
        
        self.START_TOKEN = "<s>"
        self.END_TOKEN = "</s>"
    
    def train(self, corpus: List[str]) -> None:
        """
        Train trigram model on a corpus of text.
        
        Args:
            corpus: List of sentences (strings) to train on
        """
        total_trigrams = 0
        
        for sentence in corpus:
            words = sentence.lower().split()
            # Add two start tokens for trigram context
            words = [self.START_TOKEN, self.START_TOKEN] + words + [self.END_TOKEN]
            
            # Extract trigrams and update counts
            for i in range(len(words) - 2):
                w1, w2, w3 = words[i], words[i + 1], words[i + 2]
                
                # Update trigram count
                self.trigram_counts[w1][w2][w3] += 1
                
                # Update bigram count (first two words)
                self.bigram_counts[w1][w2] += 1
                
                # Update vocabulary
                for w in [w1, w2, w3]:
                    if w not in [self.START_TOKEN, self.END_TOKEN]:
                        self.vocab.add(w)
                
                total_trigrams += 1
        
        print(f"Trained trigram model on {total_trigrams} trigrams")
        print(f"Vocabulary size: {len(self.vocab)} unique words")
    
    def probability(self, word: str, context1: str, context2: str) -> float:
        """
        Calculate P(word | context1, context2).
        
        P(w_i | w_{i-2}, w_{i-1}) = count(w_{i-2}, w_{i-1}, w_i) / count(w_{i-2}, w_{i-1})
        
        Args:
            word: Current word
            context1: Word two positions before (w_{i-2})
            context2: Word one position before (w_{i-1})
            
        Returns:
            Conditional probability
        """
        word = word.lower()
        context1 = context1.lower()
        context2 = context2.lower()
        
        # Get count of trigram
        trigram_count = self.trigram_counts[context1][context2][word]
        
        # Get count of bigram context
        bigram_count = self.bigram_counts[context1][context2]
        
        # Calculate conditional probability
        return trigram_count / bigram_count if bigram_count > 0 else 0.0
    
    def sentence_probability(self, sentence: str) -> float:
        """
        Calculate probability of a sentence using trigram model.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Probability of the sentence
        """
        words = sentence.lower().split()
        words = [self.START_TOKEN, self.START_TOKEN] + words + [self.END_TOKEN]
        
        prob = 1.0
        
        for i in range(2, len(words)):
            context1, context2, word = words[i-2], words[i-1], words[i]
            prob *= self.probability(word, context1, context2)
            
            if prob == 0.0:
                return 0.0
        
        return prob


def demonstrate_ngram_models():
    """
    Demonstrate unigram, bigram, and trigram models on sample text.
    """
    print("=" * 70)
    print("N-gram Language Models Demonstration")
    print("=" * 70)
    
    # Sample training corpus
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog played",
        "cats and dogs are friends",
        "the quick brown fox jumps"
    ]
    
    print("\nTraining Corpus:")
    for i, sent in enumerate(corpus, 1):
        print(f"  {i}. {sent}")
    
    # Test sentences
    test_sentences = [
        "the cat sat",           # Seen sequence
        "the dog played",        # Partially seen
        "the elephant danced"    # Unseen words
    ]
    
    print("\n" + "=" * 70)
    print("1. UNIGRAM MODEL")
    print("=" * 70)
    
    unigram = UnigramModel()
    unigram.train(corpus)
    
    print("\nWord Probabilities:")
    for word in ["the", "cat", "dog", "sat", "elephant"]:
        prob = unigram.probability(word)
        log_prob = unigram.log_probability(word) if prob > 0 else float('-inf')
        print(f"  P({word}) = {prob:.4f}, log P({word}) = {log_prob:.2f}")
    
    print("\nSentence Probabilities:")
    for sent in test_sentences:
        prob = unigram.sentence_probability(sent)
        log_prob = unigram.sentence_log_probability(sent)
        print(f"  '{sent}'")
        print(f"    P = {prob:.10f}, log P = {log_prob:.2f}")
    
    print("\n" + "=" * 70)
    print("2. BIGRAM MODEL")
    print("=" * 70)
    
    bigram = BigramModel()
    bigram.train(corpus)
    
    print("\nConditional Probabilities:")
    test_bigrams = [
        ("the", "cat"),
        ("cat", "sat"),
        ("dog", "sat"),
        ("the", "elephant")
    ]
    
    for context, word in test_bigrams:
        prob = bigram.probability(word, context)
        log_prob = bigram.log_probability(word, context) if prob > 0 else float('-inf')
        print(f"  P({word} | {context}) = {prob:.4f}, log P = {log_prob:.2f}")
    
    print("\nSentence Probabilities:")
    for sent in test_sentences:
        prob = bigram.sentence_probability(sent)
        log_prob = bigram.sentence_log_probability(sent)
        print(f"  '{sent}'")
        print(f"    P = {prob:.10f}, log P = {log_prob:.2f}")
    
    print("\n" + "=" * 70)
    print("3. TRIGRAM MODEL")
    print("=" * 70)
    
    trigram = TrigramModel()
    trigram.train(corpus)
    
    print("\nConditional Probabilities:")
    test_trigrams = [
        ("<s>", "the", "cat"),
        ("the", "cat", "sat"),
        ("cat", "sat", "on")
    ]
    
    for c1, c2, word in test_trigrams:
        prob = trigram.probability(word, c1, c2)
        print(f"  P({word} | {c1}, {c2}) = {prob:.4f}")
    
    print("\nSentence Probabilities:")
    for sent in test_sentences:
        prob = trigram.sentence_probability(sent)
        print(f"  '{sent}': P = {prob:.10f}")
    
    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS")
    print("=" * 70)
    print("""
1. Unigram Model:
   - Ignores word order and context
   - Highest probability for frequent words like 'the'
   - Cannot capture phrase structures or collocations
   
2. Bigram Model:
   - Captures immediate context (one previous word)
   - Can model simple dependencies like "the cat"
   - Better than unigram for sentence probability
   
3. Trigram Model:
   - Captures longer context (two previous words)
   - More data sparse than bigram (many zero probabilities)
   - Can model longer dependencies
   
4. Data Sparsity Problem:
   - Higher-order n-grams have more zero probabilities
   - Need smoothing techniques to handle unseen n-grams
   - Trade-off between context size and data sparsity
    """)


if __name__ == "__main__":
    demonstrate_ngram_models()
    
    print("\n" + "=" * 70)
    print("EXERCISES")
    print("=" * 70)
    print("""
1. Implement a method to find the most probable next word given a context
2. Calculate perplexity for each model on a test set
3. Visualize the distribution of bigram probabilities
4. Compare model performance as corpus size increases
5. Implement a 4-gram or 5-gram model and observe data sparsity
    """)
