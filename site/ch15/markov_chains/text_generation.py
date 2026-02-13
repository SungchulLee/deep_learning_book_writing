"""
text_generation.py (Module 09)

Text Generation Using Markov Chains
====================================

Location: 06_markov_chain/03_applications/
Difficulty: ⭐⭐ Elementary
Estimated Time: 2 hours

Learning Objectives:
- Build Markov models from text data
- Generate new text using n-gram models
- Understand order-k Markov chains
- Implement text analysis applications

Mathematical Foundation:
An order-k Markov chain for text:
- State = sequence of k words (k-gram)
- Transitions = next word probabilities
- P(w_{n+1} | w_n, w_{n-1}, ..., w_1) = P(w_{n+1} | w_n, ..., w_{n-k+1})
"""

import numpy as np
from collections import defaultdict, Counter
import random


class MarkovTextGenerator:
    """Generate text using Markov chains."""
    
    def __init__(self, order=1):
        """
        Initialize text generator.
        
        Parameters:
            order (int): Order of Markov chain (number of previous words to consider)
        """
        self.order = order
        self.model = defaultdict(Counter)
        self.start_states = []
    
    def train(self, text):
        """Train model on text."""
        words = text.split()
        
        # Store possible starting states
        for i in range(len(words) - self.order):
            state = tuple(words[i:i+self.order])
            if i == 0 or words[i-1] in '.!?':
                self.start_states.append(state)
            
            # Count transitions
            next_word = words[i+self.order]
            self.model[state][next_word] += 1
    
    def generate(self, length=50, seed=None):
        """Generate text of given length."""
        if seed:
            random.seed(seed)
        
        # Start with random starting state
        current_state = random.choice(self.start_states if self.start_states else list(self.model.keys()))
        result = list(current_state)
        
        for _ in range(length - self.order):
            if current_state not in self.model:
                break
            
            # Get next word probabilities
            next_words = self.model[current_state]
            total = sum(next_words.values())
            
            # Sample next word
            choices = list(next_words.keys())
            weights = [next_words[w]/total for w in choices]
            next_word = random.choices(choices, weights=weights)[0]
            
            result.append(next_word)
            current_state = tuple(result[-self.order:])
        
        return ' '.join(result)


# Example usage
if __name__ == "__main__":
    sample_text = """
    The quick brown fox jumps over the lazy dog. The dog was sleeping under a tree.
    A tree grows in Brooklyn. Brooklyn is a borough of New York. New York never sleeps.
    The quick cat runs through the park. The park is beautiful in spring.
    Spring brings new life to the garden. The garden has many flowers.
    """
    
    print("MARKOV CHAIN TEXT GENERATION")
    print("=" * 70)
    
    for order in [1, 2]:
        print(f"\\nOrder {order} Markov Chain:")
        gen = MarkovTextGenerator(order=order)
        gen.train(sample_text)
        
        for i in range(3):
            print(f"  Generated {i+1}: {gen.generate(length=20)}")
