"""
Tutorial 02: N-gram Smoothing Techniques
=========================================

This tutorial covers smoothing techniques that address the zero-probability
problem in n-gram models. Smoothing is essential for handling unseen n-grams.

Learning Objectives:
--------------------
1. Understand the data sparsity problem in n-gram models
2. Implement Laplace (Add-one) smoothing
3. Implement Add-k smoothing
4. Implement linear interpolation
5. Compare different smoothing techniques

Mathematical Background:
------------------------

Problem: Unseen N-grams
-----------------------
In MLE: P(w_i | w_{i-1}) = count(w_{i-1}, w_i) / count(w_{i-1})
If we never saw bigram (w_{i-1}, w_i), then P = 0

This causes:
1. Zero probability for valid but unseen sequences
2. Inability to generalize to new data
3. Numerical underflow in probability calculations

Laplace Smoothing (Add-one):
----------------------------
P_laplace(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + 1) / (count(w_{i-1}) + V)

where V = vocabulary size

Effect:
- Adds 1 to all bigram counts
- Redistributes probability mass to unseen events
- Can over-smooth with large vocabulary

Add-k Smoothing:
----------------
P_add-k(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + k) / (count(w_{i-1}) + k*V)

where 0 < k < 1 (typically k=0.1 to 0.5)

Effect:
- Generalization of Laplace smoothing
- Less aggressive redistribution than Add-one
- k is a hyperparameter to tune

Linear Interpolation:
---------------------
P_interp(w_i | w_{i-1}) = λ₂ * P_ML(w_i | w_{i-1}) + λ₁ * P_ML(w_i)

where λ₂ + λ₁ = 1, and λ values are weights

For trigrams:
P_interp(w_i | w_{i-2}, w_{i-1}) = λ₃ * P_ML(w_i | w_{i-2}, w_{i-1}) 
                                   + λ₂ * P_ML(w_i | w_{i-1})
                                   + λ₁ * P_ML(w_i)

Effect:
- Combines evidence from different n-gram orders
- Backs off to lower-order models for rare contexts
- λ values can be learned from held-out data
"""

import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict


class LaplaceBigramModel:
    """
    Bigram language model with Laplace (add-one) smoothing.
    
    Smoothed probability:
    P(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + 1) / (count(w_{i-1}) + V)
    
    This ensures all bigrams have non-zero probability, even if unseen.
    
    Attributes:
        bigram_counts (defaultdict): Count of each bigram
        unigram_counts (Counter): Count of each unigram
        vocab (set): Vocabulary of all unique words
        vocab_size (int): Size of vocabulary
    """
    
    def __init__(self):
        """Initialize Laplace smoothed bigram model."""
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()
        self.vocab_size = 0
        
        self.START_TOKEN = "<s>"
        self.END_TOKEN = "</s>"
        self.UNK_TOKEN = "<unk>"  # Token for unknown words
    
    def train(self, corpus: List[str], min_freq: int = 1) -> None:
        """
        Train bigram model with Laplace smoothing.
        
        Args:
            corpus: List of sentences to train on
            min_freq: Minimum frequency for a word to be in vocabulary
                     (words below this are replaced with <unk>)
        """
        # First pass: count all words
        word_counts = Counter()
        for sentence in corpus:
            words = sentence.lower().split()
            word_counts.update(words)
        
        # Build vocabulary (words with frequency >= min_freq)
        self.vocab = {word for word, count in word_counts.items() 
                     if count >= min_freq}
        self.vocab.add(self.UNK_TOKEN)  # Always include unknown token
        self.vocab_size = len(self.vocab)
        
        print(f"Vocabulary size: {self.vocab_size} (min_freq={min_freq})")
        
        # Second pass: count bigrams (replacing rare words with <unk>)
        for sentence in corpus:
            words = sentence.lower().split()
            
            # Replace out-of-vocabulary words with <unk>
            words = [word if word in self.vocab else self.UNK_TOKEN 
                    for word in words]
            
            # Add boundary tokens
            words = [self.START_TOKEN] + words + [self.END_TOKEN]
            
            # Count bigrams and unigrams
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                self.bigram_counts[w1][w2] += 1
                self.unigram_counts[w1] += 1
    
    def probability(self, word: str, context: str) -> float:
        """
        Calculate smoothed conditional probability P(word | context).
        
        Laplace smoothing formula:
        P(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + 1) / (count(w_{i-1}) + V)
        
        Args:
            word: Current word
            context: Previous word
            
        Returns:
            Smoothed conditional probability (always > 0)
        """
        word = word.lower()
        context = context.lower()
        
        # Replace OOV words with <unk>
        if word not in self.vocab:
            word = self.UNK_TOKEN
        if context not in self.vocab and context != self.START_TOKEN:
            context = self.UNK_TOKEN
        
        # Get counts with Laplace smoothing: add 1 to numerator and V to denominator
        bigram_count = self.bigram_counts[context][word]
        context_count = self.unigram_counts[context]
        
        # Apply Laplace smoothing
        numerator = bigram_count + 1
        denominator = context_count + self.vocab_size
        
        return numerator / denominator
    
    def sentence_probability(self, sentence: str) -> float:
        """Calculate probability of sentence with smoothing."""
        words = sentence.lower().split()
        
        # Replace OOV words
        words = [word if word in self.vocab else self.UNK_TOKEN 
                for word in words]
        words = [self.START_TOKEN] + words + [self.END_TOKEN]
        
        prob = 1.0
        for i in range(len(words) - 1):
            prob *= self.probability(words[i + 1], words[i])
        
        return prob
    
    def perplexity(self, test_corpus: List[str]) -> float:
        """
        Calculate perplexity on test corpus.
        
        Perplexity = 2^(cross-entropy)
        Lower perplexity = better model
        
        Args:
            test_corpus: List of test sentences
            
        Returns:
            Perplexity score
        """
        total_log_prob = 0.0
        total_words = 0
        
        for sentence in test_corpus:
            words = sentence.lower().split()
            words = [word if word in self.vocab else self.UNK_TOKEN 
                    for word in words]
            words = [self.START_TOKEN] + words + [self.END_TOKEN]
            
            # Sum log probabilities
            for i in range(len(words) - 1):
                prob = self.probability(words[i + 1], words[i])
                total_log_prob += math.log2(prob)
                total_words += 1
        
        # Perplexity = 2^(-average log probability)
        cross_entropy = -total_log_prob / total_words
        perplexity = 2 ** cross_entropy
        
        return perplexity


class AddKBigramModel:
    """
    Bigram language model with Add-k smoothing (generalized Laplace).
    
    Smoothed probability:
    P(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + k) / (count(w_{i-1}) + k*V)
    
    where k is a smoothing parameter (0 < k < 1, typically k=0.5)
    
    Attributes:
        k (float): Smoothing parameter
        bigram_counts (defaultdict): Count of each bigram
        unigram_counts (Counter): Count of each unigram
        vocab (set): Vocabulary of unique words
    """
    
    def __init__(self, k: float = 0.5):
        """
        Initialize Add-k smoothed bigram model.
        
        Args:
            k: Smoothing parameter (default 0.5)
               k=1.0 is equivalent to Laplace smoothing
               k<1.0 is less aggressive smoothing
        """
        self.k = k
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()
        self.vocab_size = 0
        
        self.START_TOKEN = "<s>"
        self.END_TOKEN = "</s>"
        self.UNK_TOKEN = "<unk>"
    
    def train(self, corpus: List[str], min_freq: int = 1) -> None:
        """Train model (same as Laplace model)."""
        # Count words
        word_counts = Counter()
        for sentence in corpus:
            words = sentence.lower().split()
            word_counts.update(words)
        
        # Build vocabulary
        self.vocab = {word for word, count in word_counts.items() 
                     if count >= min_freq}
        self.vocab.add(self.UNK_TOKEN)
        self.vocab_size = len(self.vocab)
        
        print(f"Vocabulary size: {self.vocab_size} (k={self.k})")
        
        # Count bigrams
        for sentence in corpus:
            words = sentence.lower().split()
            words = [word if word in self.vocab else self.UNK_TOKEN 
                    for word in words]
            words = [self.START_TOKEN] + words + [self.END_TOKEN]
            
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                self.bigram_counts[w1][w2] += 1
                self.unigram_counts[w1] += 1
    
    def probability(self, word: str, context: str) -> float:
        """
        Calculate smoothed probability with Add-k smoothing.
        
        P(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + k) / (count(w_{i-1}) + k*V)
        
        Args:
            word: Current word
            context: Previous word
            
        Returns:
            Smoothed conditional probability
        """
        word = word.lower()
        context = context.lower()
        
        # Handle OOV words
        if word not in self.vocab:
            word = self.UNK_TOKEN
        if context not in self.vocab and context != self.START_TOKEN:
            context = self.UNK_TOKEN
        
        # Get counts with Add-k smoothing
        bigram_count = self.bigram_counts[context][word]
        context_count = self.unigram_counts[context]
        
        # Apply Add-k smoothing
        numerator = bigram_count + self.k
        denominator = context_count + (self.k * self.vocab_size)
        
        return numerator / denominator
    
    def sentence_probability(self, sentence: str) -> float:
        """Calculate sentence probability."""
        words = sentence.lower().split()
        words = [word if word in self.vocab else self.UNK_TOKEN 
                for word in words]
        words = [self.START_TOKEN] + words + [self.END_TOKEN]
        
        prob = 1.0
        for i in range(len(words) - 1):
            prob *= self.probability(words[i + 1], words[i])
        
        return prob
    
    def perplexity(self, test_corpus: List[str]) -> float:
        """Calculate perplexity on test corpus."""
        total_log_prob = 0.0
        total_words = 0
        
        for sentence in test_corpus:
            words = sentence.lower().split()
            words = [word if word in self.vocab else self.UNK_TOKEN 
                    for word in words]
            words = [self.START_TOKEN] + words + [self.END_TOKEN]
            
            for i in range(len(words) - 1):
                prob = self.probability(words[i + 1], words[i])
                total_log_prob += math.log2(prob)
                total_words += 1
        
        cross_entropy = -total_log_prob / total_words
        return 2 ** cross_entropy


class InterpolatedBigramModel:
    """
    Bigram model with linear interpolation smoothing.
    
    Combines bigram and unigram probabilities:
    P(w_i | w_{i-1}) = λ₂ * P_bigram(w_i | w_{i-1}) + λ₁ * P_unigram(w_i)
    
    where λ₂ + λ₁ = 1
    
    Attributes:
        lambda2 (float): Weight for bigram probability
        lambda1 (float): Weight for unigram probability
        bigram_counts (defaultdict): Bigram counts
        unigram_counts (Counter): Unigram counts
        total_words (int): Total word count (for unigram probabilities)
    """
    
    def __init__(self, lambda2: float = 0.7, lambda1: float = 0.3):
        """
        Initialize interpolated bigram model.
        
        Args:
            lambda2: Weight for bigram model (default 0.7)
            lambda1: Weight for unigram model (default 0.3)
                    Must satisfy: lambda2 + lambda1 = 1.0
        """
        assert abs(lambda2 + lambda1 - 1.0) < 1e-6, "Lambdas must sum to 1"
        
        self.lambda2 = lambda2
        self.lambda1 = lambda1
        
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.total_words = 0
        self.vocab = set()
        
        self.START_TOKEN = "<s>"
        self.END_TOKEN = "</s>"
        self.UNK_TOKEN = "<unk>"
    
    def train(self, corpus: List[str], min_freq: int = 1) -> None:
        """Train interpolated model."""
        # Count words
        word_counts = Counter()
        for sentence in corpus:
            words = sentence.lower().split()
            word_counts.update(words)
        
        # Build vocabulary
        self.vocab = {word for word, count in word_counts.items() 
                     if count >= min_freq}
        self.vocab.add(self.UNK_TOKEN)
        
        print(f"Vocabulary size: {len(self.vocab)} " +
              f"(λ₂={self.lambda2}, λ₁={self.lambda1})")
        
        # Count bigrams and unigrams
        for sentence in corpus:
            words = sentence.lower().split()
            words = [word if word in self.vocab else self.UNK_TOKEN 
                    for word in words]
            words = [self.START_TOKEN] + words + [self.END_TOKEN]
            
            # Count bigrams
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                self.bigram_counts[w1][w2] += 1
                
                # Count unigrams (excluding start token)
                if w1 != self.START_TOKEN:
                    self.unigram_counts[w1] += 1
                    self.total_words += 1
            
            # Count last word
            if words[-1] != self.END_TOKEN:
                self.unigram_counts[words[-1]] += 1
                self.total_words += 1
    
    def unigram_probability(self, word: str) -> float:
        """
        Calculate unigram probability P(word).
        
        P(word) = count(word) / total_words
        
        Uses add-one smoothing to handle unseen words.
        """
        word = word.lower()
        if word not in self.vocab:
            word = self.UNK_TOKEN
        
        # Add-one smoothing for unigram
        count = self.unigram_counts[word]
        return (count + 1) / (self.total_words + len(self.vocab))
    
    def bigram_probability_ml(self, word: str, context: str) -> float:
        """
        Calculate maximum likelihood bigram probability (no smoothing).
        
        P(word | context) = count(context, word) / count(context)
        """
        word = word.lower()
        context = context.lower()
        
        if word not in self.vocab:
            word = self.UNK_TOKEN
        if context not in self.vocab and context != self.START_TOKEN:
            context = self.UNK_TOKEN
        
        bigram_count = self.bigram_counts[context][word]
        context_count = sum(self.bigram_counts[context].values())
        
        if context_count == 0:
            return 0.0
        
        return bigram_count / context_count
    
    def probability(self, word: str, context: str) -> float:
        """
        Calculate interpolated probability.
        
        P(word | context) = λ₂ * P_bigram(word | context) + λ₁ * P_unigram(word)
        
        Args:
            word: Current word
            context: Previous word
            
        Returns:
            Interpolated probability
        """
        p_bigram = self.bigram_probability_ml(word, context)
        p_unigram = self.unigram_probability(word)
        
        # Linear interpolation
        p_interpolated = self.lambda2 * p_bigram + self.lambda1 * p_unigram
        
        return p_interpolated
    
    def sentence_probability(self, sentence: str) -> float:
        """Calculate sentence probability with interpolation."""
        words = sentence.lower().split()
        words = [word if word in self.vocab else self.UNK_TOKEN 
                for word in words]
        words = [self.START_TOKEN] + words + [self.END_TOKEN]
        
        prob = 1.0
        for i in range(len(words) - 1):
            prob *= self.probability(words[i + 1], words[i])
        
        return prob
    
    def perplexity(self, test_corpus: List[str]) -> float:
        """Calculate perplexity on test corpus."""
        total_log_prob = 0.0
        total_words = 0
        
        for sentence in test_corpus:
            words = sentence.lower().split()
            words = [word if word in self.vocab else self.UNK_TOKEN 
                    for word in words]
            words = [self.START_TOKEN] + words + [self.END_TOKEN]
            
            for i in range(len(words) - 1):
                prob = self.probability(words[i + 1], words[i])
                # Avoid log(0)
                if prob > 0:
                    total_log_prob += math.log2(prob)
                else:
                    total_log_prob += -100  # Large penalty for zero prob
                total_words += 1
        
        cross_entropy = -total_log_prob / total_words
        return 2 ** cross_entropy


def compare_smoothing_techniques():
    """
    Compare different smoothing techniques on the same corpus.
    """
    print("=" * 70)
    print("Comparing Smoothing Techniques")
    print("=" * 70)
    
    # Training corpus
    train_corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog are friends",
        "cats and dogs play together",
        "the quick brown fox jumps",
        "a lazy dog sleeps all day",
        "the cat catches the mouse"
    ]
    
    # Test corpus (includes some unseen bigrams)
    test_corpus = [
        "the cat plays",
        "a dog jumps",
        "the mouse runs"
    ]
    
    print("\nTraining corpus size:", len(train_corpus), "sentences")
    print("Test corpus size:", len(test_corpus), "sentences\n")
    
    # Train different models
    print("Training models...")
    print("-" * 70)
    
    # Laplace smoothing
    laplace_model = LaplaceBigramModel()
    laplace_model.train(train_corpus, min_freq=1)
    
    # Add-k smoothing with different k values
    addk_models = {}
    for k in [0.1, 0.5, 1.0]:
        model = AddKBigramModel(k=k)
        model.train(train_corpus, min_freq=1)
        addk_models[k] = model
    
    # Interpolation with different lambda values
    interp_models = {}
    for lambda2 in [0.5, 0.7, 0.9]:
        lambda1 = 1.0 - lambda2
        model = InterpolatedBigramModel(lambda2=lambda2, lambda1=lambda1)
        model.train(train_corpus, min_freq=1)
        interp_models[lambda2] = model
    
    # Calculate perplexities
    print("\n" + "=" * 70)
    print("Perplexity Comparison on Test Set")
    print("=" * 70)
    
    print(f"\nLaplace (Add-1):")
    laplace_ppl = laplace_model.perplexity(test_corpus)
    print(f"  Perplexity: {laplace_ppl:.2f}")
    
    print(f"\nAdd-k Smoothing:")
    for k, model in addk_models.items():
        ppl = model.perplexity(test_corpus)
        print(f"  k={k}: Perplexity = {ppl:.2f}")
    
    print(f"\nLinear Interpolation:")
    for lambda2, model in interp_models.items():
        ppl = model.perplexity(test_corpus)
        print(f"  λ₂={lambda2}, λ₁={1-lambda2}: Perplexity = {ppl:.2f}")
    
    # Test on specific examples
    print("\n" + "=" * 70)
    print("Probability Comparison for Specific Bigrams")
    print("=" * 70)
    
    test_bigrams = [
        ("the", "cat"),      # Seen bigram
        ("cat", "plays"),    # Unseen bigram
        ("mouse", "runs")    # Both seen separately, bigram unseen
    ]
    
    for context, word in test_bigrams:
        print(f"\nP({word} | {context}):")
        print(f"  Laplace:       {laplace_model.probability(word, context):.6f}")
        print(f"  Add-0.5:       {addk_models[0.5].probability(word, context):.6f}")
        print(f"  Interp (0.7):  {interp_models[0.7].probability(word, context):.6f}")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. Laplace (Add-1) Smoothing:
   - Simple and straightforward
   - Can over-smooth with large vocabulary
   - Not usually best for perplexity
   
2. Add-k Smoothing (k < 1):
   - More flexible than Laplace
   - k=0.5 often works better than k=1.0
   - Requires tuning k on held-out data
   
3. Linear Interpolation:
   - Combines strengths of different n-gram orders
   - More robust to data sparsity
   - λ values can be optimized
   - Generally achieves lower perplexity
   
4. Trade-offs:
   - Simpler smoothing (Laplace) is easier to implement
   - Better smoothing (interpolation) requires more computation
   - Choice depends on data size and application
    """)


if __name__ == "__main__":
    compare_smoothing_techniques()
    
    print("\n" + "=" * 70)
    print("EXERCISES")
    print("=" * 70)
    print("""
1. Implement Good-Turing smoothing
2. Find optimal k value using grid search on validation set
3. Implement deleted interpolation to learn λ values automatically
4. Compare smoothing techniques on different corpus sizes
5. Implement Kneser-Ney smoothing (advanced)
6. Visualize how perplexity changes with different smoothing parameters
    """)
