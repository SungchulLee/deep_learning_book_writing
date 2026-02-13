"""
Tutorial 03: Text Generation with N-grams
==========================================

This tutorial demonstrates how to generate text using n-gram language models.
We explore different generation strategies and their characteristics.

Learning Objectives:
--------------------
1. Implement text generation using n-gram models
2. Understand sampling vs. greedy decoding
3. Control generation with temperature
4. Handle different termination conditions
5. Evaluate generated text quality

Mathematical Background:
------------------------

Text Generation Process:
-----------------------
Given a trained n-gram model, we can generate text by:
1. Start with initial context (e.g., <s> token)
2. Sample next word from P(w | context)
3. Update context with generated word
4. Repeat until stopping condition met

Sampling Strategies:
-------------------

1. Greedy Decoding:
   - Always select most probable word
   - Deterministic output
   - w_next = argmax P(w | context)

2. Random Sampling:
   - Sample from full probability distribution
   - Stochastic output
   - More diverse but potentially less coherent

3. Temperature Sampling:
   - Modify probabilities: P'(w) = P(w)^(1/T) / Z
   - T > 1: More random (flatten distribution)
   - T < 1: More deterministic (sharpen distribution)
   - T = 1: Regular sampling

4. Top-k Sampling:
   - Sample from k most probable words
   - Balances diversity and quality
   - Truncate low-probability options

Stopping Conditions:
-------------------
- Fixed length (generate N words)
- End token (</s>) generated
- Maximum length reached
- Probability threshold
"""

import random
import math
from collections import defaultdict, Counter
from typing import List, Tuple, Optional
import numpy as np


class TextGeneratorBigram:
    """
    Text generator using bigram language model.
    
    Supports multiple generation strategies:
    - Greedy decoding
    - Random sampling
    - Temperature-based sampling
    - Top-k sampling
    
    Attributes:
        bigram_counts (defaultdict): Bigram frequency counts
        unigram_counts (Counter): Unigram frequency counts
        vocab (set): Vocabulary
        START_TOKEN (str): Sentence start token
        END_TOKEN (str): Sentence end token
    """
    
    def __init__(self):
        """Initialize text generator."""
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()
        
        self.START_TOKEN = "<s>"
        self.END_TOKEN = "</s>"
    
    def train(self, corpus: List[str]) -> None:
        """
        Train bigram model for text generation.
        
        Args:
            corpus: List of training sentences
        """
        for sentence in corpus:
            words = sentence.lower().split()
            words = [self.START_TOKEN] + words + [self.END_TOKEN]
            
            # Count bigrams
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                self.bigram_counts[w1][w2] += 1
                self.unigram_counts[w1] += 1
                self.vocab.add(w1)
                self.vocab.add(w2)
        
        print(f"Trained on {len(corpus)} sentences")
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def get_next_word_distribution(self, context: str) -> List[Tuple[str, float]]:
        """
        Get probability distribution over next words given context.
        
        Returns list of (word, probability) tuples sorted by probability.
        
        Args:
            context: Previous word
            
        Returns:
            List of (word, probability) tuples
        """
        context = context.lower()
        
        # Get all possible next words and their counts
        next_word_counts = self.bigram_counts[context]
        
        if not next_word_counts:
            # If context not seen, return uniform over vocabulary
            prob = 1.0 / len(self.vocab)
            return [(word, prob) for word in self.vocab]
        
        # Calculate probabilities
        total_count = sum(next_word_counts.values())
        distribution = [(word, count / total_count) 
                       for word, count in next_word_counts.items()]
        
        # Sort by probability (descending)
        distribution.sort(key=lambda x: x[1], reverse=True)
        
        return distribution
    
    def generate_greedy(self, max_length: int = 20, 
                       start_context: str = None) -> str:
        """
        Generate text using greedy decoding (always pick most probable word).
        
        This produces deterministic output but can be repetitive.
        
        Args:
            max_length: Maximum number of words to generate
            start_context: Starting word (default: <s>)
            
        Returns:
            Generated text
        """
        if start_context is None:
            current_word = self.START_TOKEN
        else:
            current_word = start_context.lower()
        
        generated = []
        
        for _ in range(max_length):
            # Get distribution over next words
            distribution = self.get_next_word_distribution(current_word)
            
            if not distribution:
                break
            
            # Pick most probable word (greedy)
            next_word = distribution[0][0]  # First element has highest prob
            
            # Stop if we hit end token
            if next_word == self.END_TOKEN:
                break
            
            # Skip start token in output
            if next_word != self.START_TOKEN:
                generated.append(next_word)
            
            current_word = next_word
        
        return ' '.join(generated)
    
    def generate_random(self, max_length: int = 20,
                       start_context: str = None) -> str:
        """
        Generate text using random sampling from probability distribution.
        
        This produces stochastic, diverse output.
        
        Args:
            max_length: Maximum number of words to generate
            start_context: Starting word (default: <s>)
            
        Returns:
            Generated text
        """
        if start_context is None:
            current_word = self.START_TOKEN
        else:
            current_word = start_context.lower()
        
        generated = []
        
        for _ in range(max_length):
            # Get distribution over next words
            distribution = self.get_next_word_distribution(current_word)
            
            if not distribution:
                break
            
            # Sample from distribution
            words, probs = zip(*distribution)
            next_word = random.choices(words, weights=probs, k=1)[0]
            
            # Stop if we hit end token
            if next_word == self.END_TOKEN:
                break
            
            if next_word != self.START_TOKEN:
                generated.append(next_word)
            
            current_word = next_word
        
        return ' '.join(generated)
    
    def generate_temperature(self, temperature: float = 1.0,
                            max_length: int = 20,
                            start_context: str = None) -> str:
        """
        Generate text using temperature-based sampling.
        
        Temperature controls randomness:
        - T > 1: More random (flatten distribution)
        - T < 1: More deterministic (sharpen distribution)
        - T = 1: Regular sampling
        
        Formula: P'(w) = P(w)^(1/T) / Z
        where Z is normalization constant
        
        Args:
            temperature: Temperature parameter (default 1.0)
            max_length: Maximum number of words
            start_context: Starting word
            
        Returns:
            Generated text
        """
        if start_context is None:
            current_word = self.START_TOKEN
        else:
            current_word = start_context.lower()
        
        generated = []
        
        for _ in range(max_length):
            # Get distribution
            distribution = self.get_next_word_distribution(current_word)
            
            if not distribution:
                break
            
            # Apply temperature
            words, probs = zip(*distribution)
            
            # Modify probabilities with temperature
            # P'(w) = P(w)^(1/T)
            modified_probs = [p ** (1.0 / temperature) for p in probs]
            
            # Renormalize
            total = sum(modified_probs)
            modified_probs = [p / total for p in modified_probs]
            
            # Sample from modified distribution
            next_word = random.choices(words, weights=modified_probs, k=1)[0]
            
            if next_word == self.END_TOKEN:
                break
            
            if next_word != self.START_TOKEN:
                generated.append(next_word)
            
            current_word = next_word
        
        return ' '.join(generated)
    
    def generate_top_k(self, k: int = 5, max_length: int = 20,
                      start_context: str = None) -> str:
        """
        Generate text using top-k sampling.
        
        Only sample from the k most probable words at each step.
        This balances diversity and quality.
        
        Args:
            k: Number of top words to consider
            max_length: Maximum number of words
            start_context: Starting word
            
        Returns:
            Generated text
        """
        if start_context is None:
            current_word = self.START_TOKEN
        else:
            current_word = start_context.lower()
        
        generated = []
        
        for _ in range(max_length):
            # Get distribution
            distribution = self.get_next_word_distribution(current_word)
            
            if not distribution:
                break
            
            # Keep only top k words
            top_k_distribution = distribution[:min(k, len(distribution))]
            
            # Renormalize
            words, probs = zip(*top_k_distribution)
            total = sum(probs)
            probs = [p / total for p in probs]
            
            # Sample from top k
            next_word = random.choices(words, weights=probs, k=1)[0]
            
            if next_word == self.END_TOKEN:
                break
            
            if next_word != self.START_TOKEN:
                generated.append(next_word)
            
            current_word = next_word
        
        return ' '.join(generated)


class TextGeneratorTrigram:
    """
    Text generator using trigram language model.
    
    Similar to bigram generator but uses two words of context.
    
    Attributes:
        trigram_counts: Trigram frequency counts
        bigram_counts: Bigram frequency counts (for context)
    """
    
    def __init__(self):
        """Initialize trigram text generator."""
        self.trigram_counts = defaultdict(lambda: defaultdict(Counter))
        self.bigram_counts = defaultdict(Counter)
        
        self.START_TOKEN = "<s>"
        self.END_TOKEN = "</s>"
    
    def train(self, corpus: List[str]) -> None:
        """Train trigram model."""
        for sentence in corpus:
            words = sentence.lower().split()
            # Need two start tokens for trigram
            words = [self.START_TOKEN, self.START_TOKEN] + words + [self.END_TOKEN]
            
            # Count trigrams
            for i in range(len(words) - 2):
                w1, w2, w3 = words[i], words[i + 1], words[i + 2]
                self.trigram_counts[w1][w2][w3] += 1
                self.bigram_counts[w1][w2] += 1
        
        print(f"Trained trigram model on {len(corpus)} sentences")
    
    def generate_random(self, max_length: int = 20) -> str:
        """
        Generate text using trigram model.
        
        Args:
            max_length: Maximum words to generate
            
        Returns:
            Generated text
        """
        # Start with two start tokens
        context = (self.START_TOKEN, self.START_TOKEN)
        generated = []
        
        for _ in range(max_length):
            w1, w2 = context
            
            # Get possible next words
            next_word_counts = self.trigram_counts[w1][w2]
            
            if not next_word_counts:
                break
            
            # Sample next word
            words = list(next_word_counts.keys())
            counts = list(next_word_counts.values())
            next_word = random.choices(words, weights=counts, k=1)[0]
            
            if next_word == self.END_TOKEN:
                break
            
            if next_word != self.START_TOKEN:
                generated.append(next_word)
            
            # Update context (shift window)
            context = (w2, next_word)
        
        return ' '.join(generated)


def demonstrate_text_generation():
    """
    Demonstrate different text generation strategies.
    """
    print("=" * 70)
    print("Text Generation with N-gram Models")
    print("=" * 70)
    
    # Training corpus - simple sentences about animals
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog are friends",
        "cats and dogs play together in the park",
        "the quick brown fox jumps over the lazy dog",
        "a cat catches a mouse in the house",
        "dogs like to run and play",
        "the cat sleeps all day long",
        "small dogs and big cats live together",
        "the brown dog runs in the park"
    ]
    
    print("\nTraining corpus size:", len(corpus), "sentences")
    
    # Train bigram generator
    bigram_gen = TextGeneratorBigram()
    bigram_gen.train(corpus)
    
    # Train trigram generator
    trigram_gen = TextGeneratorTrigram()
    trigram_gen.train(corpus)
    
    print("\n" + "=" * 70)
    print("1. GREEDY DECODING (Deterministic)")
    print("=" * 70)
    print("\nAlways picks most probable word.")
    print("Run multiple times to see it's deterministic:\n")
    
    for i in range(3):
        text = bigram_gen.generate_greedy(max_length=15)
        print(f"  Generation {i+1}: {text}")
    
    print("\n" + "=" * 70)
    print("2. RANDOM SAMPLING (Stochastic)")
    print("=" * 70)
    print("\nSamples from full probability distribution.")
    print("Run multiple times to see diversity:\n")
    
    for i in range(5):
        text = bigram_gen.generate_random(max_length=15)
        print(f"  Generation {i+1}: {text}")
    
    print("\n" + "=" * 70)
    print("3. TEMPERATURE SAMPLING")
    print("=" * 70)
    print("\nControls randomness via temperature parameter:")
    
    temperatures = [0.5, 1.0, 1.5, 2.0]
    for temp in temperatures:
        print(f"\nTemperature = {temp}:")
        for i in range(3):
            text = bigram_gen.generate_temperature(
                temperature=temp,
                max_length=15
            )
            print(f"  {text}")
    
    print("\n" + "=" * 70)
    print("4. TOP-K SAMPLING")
    print("=" * 70)
    print("\nOnly samples from k most probable words:")
    
    k_values = [3, 5, 10]
    for k in k_values:
        print(f"\nTop-{k} sampling:")
        for i in range(3):
            text = bigram_gen.generate_top_k(k=k, max_length=15)
            print(f"  {text}")
    
    print("\n" + "=" * 70)
    print("5. BIGRAM vs TRIGRAM")
    print("=" * 70)
    print("\nCompare bigram and trigram generation:\n")
    
    print("Bigram generations:")
    for i in range(3):
        text = bigram_gen.generate_random(max_length=15)
        print(f"  {text}")
    
    print("\nTrigram generations:")
    for i in range(3):
        text = trigram_gen.generate_random(max_length=15)
        print(f"  {text}")
    
    print("\n" + "=" * 70)
    print("6. CONDITIONAL GENERATION (Prompted)")
    print("=" * 70)
    print("\nGenerate text starting with specific words:\n")
    
    prompts = ["the cat", "dogs and", "in the"]
    for prompt in prompts:
        # Get last word as context
        last_word = prompt.split()[-1]
        text = bigram_gen.generate_random(
            max_length=10,
            start_context=last_word
        )
        print(f"  Prompt: '{prompt}' → {prompt} {text}")
    
    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS")
    print("=" * 70)
    print("""
1. Greedy Decoding:
   - Produces same output every time (deterministic)
   - Often repetitive (can get stuck in loops)
   - Selects locally optimal choices
   
2. Random Sampling:
   - Produces different output each time (stochastic)
   - More diverse but sometimes incoherent
   - Can generate unlikely sequences
   
3. Temperature Sampling:
   - T < 1: More conservative, closer to greedy
   - T = 1: Standard random sampling
   - T > 1: More random, flatter distribution
   - Good for controlling creativity
   
4. Top-K Sampling:
   - Balances diversity and quality
   - Prevents very unlikely words
   - k=3 to 10 often works well
   
5. Bigram vs Trigram:
   - Trigrams produce more coherent local text
   - But trigrams have more data sparsity
   - Trade-off between context and coverage
   
6. Limitations:
   - No long-range coherence
   - Cannot capture complex dependencies
   - Topic drift over long generations
   - Grammatical errors possible
    """)


def analyze_generation_quality():
    """
    Analyze quality metrics for generated text.
    """
    print("\n" + "=" * 70)
    print("Analyzing Generation Quality")
    print("=" * 70)
    
    # Simple corpus
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "cats and dogs are friends"
    ] * 5  # Repeat to have more data
    
    # Train model
    gen = TextGeneratorBigram()
    gen.train(corpus)
    
    # Generate multiple samples with different strategies
    num_samples = 20
    
    strategies = {
        'greedy': lambda: gen.generate_greedy(max_length=10),
        'random': lambda: gen.generate_random(max_length=10),
        'temp_0.5': lambda: gen.generate_temperature(0.5, max_length=10),
        'temp_1.5': lambda: gen.generate_temperature(1.5, max_length=10),
        'top_k_5': lambda: gen.generate_top_k(k=5, max_length=10)
    }
    
    print("\nGenerating", num_samples, "samples per strategy...\n")
    
    for strategy_name, generate_func in strategies.items():
        samples = [generate_func() for _ in range(num_samples)]
        
        # Calculate metrics
        unique_samples = len(set(samples))
        avg_length = sum(len(s.split()) for s in samples) / len(samples)
        
        # Diversity: ratio of unique samples
        diversity = unique_samples / num_samples
        
        print(f"{strategy_name}:")
        print(f"  Unique samples: {unique_samples}/{num_samples} (diversity={diversity:.2f})")
        print(f"  Average length: {avg_length:.1f} words")
        print(f"  Example: '{samples[0]}'")
        print()


if __name__ == "__main__":
    demonstrate_text_generation()
    analyze_generation_quality()
    
    print("\n" + "=" * 70)
    print("EXERCISES")
    print("=" * 70)
    print("""
1. Implement beam search for text generation
2. Add nucleus (top-p) sampling
3. Implement length normalization for generation
4. Create an interactive text completion system
5. Generate text with specific sentiment or style
6. Implement repetition penalty to avoid loops
7. Compare n-gram generation with neural models (next tutorial)
    """)
