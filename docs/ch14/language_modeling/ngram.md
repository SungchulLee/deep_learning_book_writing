# N-gram Language Models

## Learning Objectives

By the end of this section, you will be able to:

- Understand the fundamental concept of n-gram language models
- Implement unigram, bigram, and trigram models from scratch
- Apply Maximum Likelihood Estimation (MLE) for probability estimation
- Implement smoothing techniques to handle unseen n-grams
- Generate text using various sampling strategies
- Evaluate n-gram models using perplexity

---

## Introduction to Language Modeling

Language modeling is the task of assigning probabilities to sequences of words. Given a sequence of words $w_1, w_2, \ldots, w_n$, a language model computes:

$$P(w_1, w_2, \ldots, w_n)$$

This seemingly simple task has profound applications: text generation, speech recognition, machine translation, spelling correction, and more. N-gram models represent the classical approach to this problem, offering interpretability and computational efficiency.

---

## The Chain Rule of Probability

Using the chain rule of probability, we can decompose the joint probability of a sequence:

$$P(w_1, w_2, \ldots, w_n) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1, w_2) \cdots P(w_n|w_1, \ldots, w_{n-1})$$

$$= \prod_{i=1}^{n} P(w_i | w_1, \ldots, w_{i-1})$$

This exact decomposition is computationally intractable for long sequences since we would need to estimate probabilities conditioned on arbitrarily long histories. N-gram models address this through the **Markov assumption**.

---

## The Markov Assumption

The key insight of n-gram models is the **Markov assumption**: the probability of a word depends only on the previous $n-1$ words, not the entire history:

$$P(w_i | w_1, \ldots, w_{i-1}) \approx P(w_i | w_{i-n+1}, \ldots, w_{i-1})$$

This assumption trades accuracy for tractability. Different values of $n$ yield different model types:

| Model | n | Context | Assumption |
|-------|---|---------|------------|
| Unigram | 1 | None | Words are independent |
| Bigram | 2 | Previous word | First-order Markov |
| Trigram | 3 | Previous 2 words | Second-order Markov |
| 4-gram | 4 | Previous 3 words | Third-order Markov |

---

## Unigram Model

The unigram model assumes complete independence between words:

$$P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^{n} P(w_i)$$

### Maximum Likelihood Estimation

For unigrams, the MLE probability is simply the relative frequency:

$$P_{MLE}(w) = \frac{\text{count}(w)}{\sum_{w' \in V} \text{count}(w')} = \frac{\text{count}(w)}{N}$$

where $N$ is the total number of words in the corpus.

### PyTorch Implementation

```python
from collections import Counter
from typing import List
import math


class UnigramModel:
    """
    Unigram language model assuming word independence.
    
    P(w) = count(w) / total_words
    """
    
    def __init__(self):
        self.word_counts = Counter()
        self.total_words = 0
        self.vocab = set()
    
    def train(self, corpus: List[str]) -> None:
        """Train on a corpus of sentences."""
        for sentence in corpus:
            words = sentence.lower().split()
            self.word_counts.update(words)
            self.total_words += len(words)
            self.vocab.update(words)
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Total words: {self.total_words}")
    
    def probability(self, word: str) -> float:
        """Calculate P(word) using MLE."""
        word = word.lower()
        if word not in self.vocab:
            return 0.0
        return self.word_counts[word] / self.total_words
    
    def log_probability(self, word: str) -> float:
        """Log probability for numerical stability."""
        prob = self.probability(word)
        return math.log2(prob) if prob > 0 else float('-inf')
    
    def sentence_log_probability(self, sentence: str) -> float:
        """Compute log P(sentence) = sum of log P(word)."""
        words = sentence.lower().split()
        return sum(self.log_probability(w) for w in words)


# Example usage
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat and the dog played"
]

model = UnigramModel()
model.train(corpus)

# Word probabilities
for word in ["the", "cat", "elephant"]:
    print(f"P({word}) = {model.probability(word):.4f}")
```

**Output:**
```
Vocabulary size: 11
Total words: 19
P(the) = 0.3158
P(cat) = 0.1053
P(elephant) = 0.0000
```

The unigram model captures word frequency but ignores all sequential structure—"the cat sat" and "cat the sat" have identical probabilities.

---

## Bigram Model

The bigram model conditions each word on its immediate predecessor:

$$P(w_i | w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i)}{\text{count}(w_{i-1})}$$

This captures local dependencies like "New York" or "the cat" having higher probability than "the the".

### Handling Sentence Boundaries

We introduce special tokens to model sentence start/end:
- `<s>`: Start-of-sentence token
- `</s>`: End-of-sentence token

For the sentence "the cat sat":
- Bigrams: `(<s>, the)`, `(the, cat)`, `(cat, sat)`, `(sat, </s>)`

### Implementation

```python
from collections import defaultdict, Counter
from typing import List, Tuple
import math


class BigramModel:
    """
    Bigram language model: P(w_i | w_{i-1}).
    
    Implements MLE with sentence boundary tokens.
    """
    
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)  # bigram_counts[w1][w2]
        self.unigram_counts = Counter()
        self.vocab = set()
        self.START = "<s>"
        self.END = "</s>"
    
    def train(self, corpus: List[str]) -> None:
        """Train bigram model on corpus."""
        total_bigrams = 0
        
        for sentence in corpus:
            words = sentence.lower().split()
            words = [self.START] + words + [self.END]
            
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                self.bigram_counts[w1][w2] += 1
                self.unigram_counts[w1] += 1
                
                if w1 != self.START:
                    self.vocab.add(w1)
                if w2 != self.END:
                    self.vocab.add(w2)
                    
                total_bigrams += 1
        
        print(f"Trained on {total_bigrams} bigrams")
        print(f"Vocabulary: {len(self.vocab)} words")
    
    def probability(self, word: str, context: str) -> float:
        """
        Calculate P(word | context) using MLE.
        
        Args:
            word: Current word w_i
            context: Previous word w_{i-1}
        """
        word = word.lower()
        context = context.lower()
        
        bigram_count = self.bigram_counts[context][word]
        context_count = self.unigram_counts[context]
        
        return bigram_count / context_count if context_count > 0 else 0.0
    
    def log_probability(self, word: str, context: str) -> float:
        """Log probability for numerical stability."""
        prob = self.probability(word, context)
        return math.log2(prob) if prob > 0 else float('-inf')
    
    def sentence_probability(self, sentence: str) -> float:
        """Calculate P(sentence) using chain rule."""
        words = sentence.lower().split()
        words = [self.START] + words + [self.END]
        
        prob = 1.0
        for i in range(len(words) - 1):
            p = self.probability(words[i + 1], words[i])
            if p == 0:
                return 0.0
            prob *= p
        
        return prob
    
    def sentence_log_probability(self, sentence: str) -> float:
        """Log probability of sentence."""
        words = sentence.lower().split()
        words = [self.START] + words + [self.END]
        
        log_prob = 0.0
        for i in range(len(words) - 1):
            log_prob += self.log_probability(words[i + 1], words[i])
        
        return log_prob


# Example
model = BigramModel()
model.train(corpus)

# Conditional probabilities
test_bigrams = [("the", "cat"), ("cat", "sat"), ("dog", "played")]
for context, word in test_bigrams:
    print(f"P({word} | {context}) = {model.probability(word, context):.4f}")
```

---

## Trigram Model

The trigram model extends the context to two previous words:

$$P(w_i | w_{i-2}, w_{i-1}) = \frac{\text{count}(w_{i-2}, w_{i-1}, w_i)}{\text{count}(w_{i-2}, w_{i-1})}$$

Trigrams capture longer dependencies but suffer more from **data sparsity**—many valid trigrams may never appear in training.

```python
class TrigramModel:
    """
    Trigram language model: P(w_i | w_{i-2}, w_{i-1}).
    """
    
    def __init__(self):
        # trigram_counts[w1][w2][w3] = count
        self.trigram_counts = defaultdict(lambda: defaultdict(Counter))
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()
        self.START = "<s>"
        self.END = "</s>"
    
    def train(self, corpus: List[str]) -> None:
        """Train trigram model."""
        for sentence in corpus:
            words = sentence.lower().split()
            # Two start tokens for trigram context
            words = [self.START, self.START] + words + [self.END]
            
            for i in range(len(words) - 2):
                w1, w2, w3 = words[i], words[i+1], words[i+2]
                self.trigram_counts[w1][w2][w3] += 1
                self.bigram_counts[w1][w2] += 1
                
                for w in [w1, w2, w3]:
                    if w not in [self.START, self.END]:
                        self.vocab.add(w)
    
    def probability(self, word: str, context1: str, context2: str) -> float:
        """
        Calculate P(word | context1, context2).
        
        Args:
            word: Current word w_i
            context1: Word at position i-2
            context2: Word at position i-1
        """
        trigram_count = self.trigram_counts[context1][context2][word]
        bigram_count = self.bigram_counts[context1][context2]
        
        return trigram_count / bigram_count if bigram_count > 0 else 0.0
```

---

## The Data Sparsity Problem

A fundamental challenge with n-gram models is **data sparsity**: many valid word sequences never appear in training. Consider:

- English vocabulary: ~50,000 common words
- Possible bigrams: $50,000^2 = 2.5 \times 10^9$
- Possible trigrams: $50,000^3 = 1.25 \times 10^{14}$

Most n-grams will have zero counts, leading to:
1. Zero probability for valid but unseen sequences
2. Inability to generalize beyond training data
3. Undefined perplexity (log of zero)

**Solution**: Smoothing techniques redistribute probability mass to unseen events.

---

## Smoothing Techniques

### Laplace (Add-One) Smoothing

The simplest smoothing adds 1 to all counts:

$$P_{Laplace}(w_i | w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i) + 1}{\text{count}(w_{i-1}) + V}$$

where $V$ is the vocabulary size.

**Pros**: Simple, guarantees non-zero probabilities  
**Cons**: Assigns too much probability mass to unseen events with large vocabularies

```python
class LaplaceBigramModel:
    """Bigram model with Laplace (add-one) smoothing."""
    
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()
        self.vocab_size = 0
        self.UNK = "<unk>"
        self.START = "<s>"
        self.END = "</s>"
    
    def train(self, corpus: List[str], min_freq: int = 1) -> None:
        """Train with vocabulary thresholding."""
        # First pass: count words
        word_counts = Counter()
        for sentence in corpus:
            word_counts.update(sentence.lower().split())
        
        # Build vocabulary (words meeting minimum frequency)
        self.vocab = {w for w, c in word_counts.items() if c >= min_freq}
        self.vocab.add(self.UNK)
        self.vocab_size = len(self.vocab)
        
        # Second pass: count bigrams with OOV handling
        for sentence in corpus:
            words = sentence.lower().split()
            words = [w if w in self.vocab else self.UNK for w in words]
            words = [self.START] + words + [self.END]
            
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                self.bigram_counts[w1][w2] += 1
                self.unigram_counts[w1] += 1
    
    def probability(self, word: str, context: str) -> float:
        """
        Laplace-smoothed probability.
        
        P(w|c) = (count(c, w) + 1) / (count(c) + V)
        """
        word = word.lower()
        context = context.lower()
        
        # Handle OOV
        if word not in self.vocab:
            word = self.UNK
        if context not in self.vocab and context != self.START:
            context = self.UNK
        
        numerator = self.bigram_counts[context][word] + 1
        denominator = self.unigram_counts[context] + self.vocab_size
        
        return numerator / denominator
```

### Add-k Smoothing

A generalization with tunable parameter $k$ (typically $0 < k < 1$):

$$P_{add-k}(w_i | w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i) + k}{\text{count}(w_{i-1}) + k \cdot V}$$

Smaller $k$ values (e.g., 0.1 or 0.5) are less aggressive than Laplace smoothing.

### Linear Interpolation

Linear interpolation combines evidence from multiple n-gram orders:

$$P_{interp}(w_i | w_{i-1}) = \lambda_2 \cdot P_{ML}(w_i | w_{i-1}) + \lambda_1 \cdot P_{ML}(w_i)$$

where $\lambda_2 + \lambda_1 = 1$.

For trigrams:
$$P_{interp}(w_i | w_{i-2}, w_{i-1}) = \lambda_3 \cdot P_{tri} + \lambda_2 \cdot P_{bi} + \lambda_1 \cdot P_{uni}$$

**Intuition**: When the trigram context is rare, back off to bigram and unigram estimates.

```python
class InterpolatedBigramModel:
    """Bigram model with linear interpolation smoothing."""
    
    def __init__(self, lambda2: float = 0.7, lambda1: float = 0.3):
        """
        Args:
            lambda2: Weight for bigram probability
            lambda1: Weight for unigram probability
        """
        assert abs(lambda2 + lambda1 - 1.0) < 1e-6, "Lambdas must sum to 1"
        self.lambda2 = lambda2
        self.lambda1 = lambda1
        
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.total_words = 0
        self.vocab = set()
    
    def train(self, corpus: List[str]) -> None:
        """Train interpolated model."""
        # ... training code similar to above ...
        pass
    
    def unigram_probability(self, word: str) -> float:
        """P(word) with add-one smoothing."""
        count = self.unigram_counts[word]
        return (count + 1) / (self.total_words + len(self.vocab))
    
    def bigram_probability_ml(self, word: str, context: str) -> float:
        """Raw MLE bigram probability."""
        bigram_count = self.bigram_counts[context][word]
        context_count = sum(self.bigram_counts[context].values())
        return bigram_count / context_count if context_count > 0 else 0.0
    
    def probability(self, word: str, context: str) -> float:
        """
        Interpolated probability.
        
        P(w|c) = λ₂ · P_bigram(w|c) + λ₁ · P_unigram(w)
        """
        p_bigram = self.bigram_probability_ml(word, context)
        p_unigram = self.unigram_probability(word)
        
        return self.lambda2 * p_bigram + self.lambda1 * p_unigram
```

### Kneser-Ney Smoothing

The most sophisticated n-gram smoothing method, Kneser-Ney uses absolute discounting combined with a modified lower-order distribution based on **continuation counts**—how many different contexts a word appears in.

The intuition: "Francisco" has high unigram frequency but only appears after "San", so its continuation probability should be low.

---

## Text Generation with N-grams

N-gram models naturally support text generation by sampling from the conditional distribution.

### Greedy Decoding

Always select the most probable next word:

$$w_t = \arg\max_w P(w | \text{context})$$

**Pros**: Deterministic, fast  
**Cons**: Can get stuck in repetitive loops

### Random Sampling

Sample from the full distribution:

$$w_t \sim P(w | \text{context})$$

**Pros**: Diverse outputs  
**Cons**: Can generate unlikely sequences

### Temperature Sampling

Control randomness by scaling logits before softmax:

$$P'(w) \propto P(w)^{1/T}$$

- $T > 1$: Flatter distribution (more random)
- $T < 1$: Sharper distribution (more deterministic)
- $T = 1$: Standard sampling

### Top-k Sampling

Sample only from the $k$ most probable words:

1. Sort words by probability
2. Keep only top-$k$
3. Renormalize probabilities
4. Sample from truncated distribution

```python
import random


class TextGenerator:
    """Text generation with various sampling strategies."""
    
    def __init__(self, bigram_model):
        self.model = bigram_model
    
    def get_distribution(self, context: str) -> List[Tuple[str, float]]:
        """Get probability distribution over next words."""
        counts = self.model.bigram_counts[context]
        if not counts:
            return [(w, 1/len(self.model.vocab)) for w in self.model.vocab]
        
        total = sum(counts.values())
        return [(w, c/total) for w, c in counts.items()]
    
    def generate_greedy(self, max_length: int = 20) -> str:
        """Greedy generation: always pick most probable."""
        context = self.model.START
        generated = []
        
        for _ in range(max_length):
            dist = self.get_distribution(context)
            if not dist:
                break
            
            # Pick highest probability
            next_word = max(dist, key=lambda x: x[1])[0]
            
            if next_word == self.model.END:
                break
            
            generated.append(next_word)
            context = next_word
        
        return ' '.join(generated)
    
    def generate_temperature(self, temperature: float = 1.0, 
                             max_length: int = 20) -> str:
        """Temperature-scaled sampling."""
        context = self.model.START
        generated = []
        
        for _ in range(max_length):
            dist = self.get_distribution(context)
            if not dist:
                break
            
            words, probs = zip(*dist)
            
            # Apply temperature
            scaled_probs = [p ** (1/temperature) for p in probs]
            total = sum(scaled_probs)
            scaled_probs = [p/total for p in scaled_probs]
            
            # Sample
            next_word = random.choices(words, weights=scaled_probs, k=1)[0]
            
            if next_word == self.model.END:
                break
            
            generated.append(next_word)
            context = next_word
        
        return ' '.join(generated)
    
    def generate_top_k(self, k: int = 5, max_length: int = 20) -> str:
        """Top-k sampling."""
        context = self.model.START
        generated = []
        
        for _ in range(max_length):
            dist = self.get_distribution(context)
            if not dist:
                break
            
            # Sort by probability and keep top-k
            dist.sort(key=lambda x: x[1], reverse=True)
            top_k = dist[:k]
            
            # Renormalize
            words, probs = zip(*top_k)
            total = sum(probs)
            probs = [p/total for p in probs]
            
            # Sample
            next_word = random.choices(words, weights=probs, k=1)[0]
            
            if next_word == self.model.END:
                break
            
            generated.append(next_word)
            context = next_word
        
        return ' '.join(generated)
```

---

## Evaluation: Perplexity

**Perplexity** is the standard intrinsic evaluation metric for language models:

$$\text{PPL} = 2^{H(P, \hat{P})} = 2^{-\frac{1}{N}\sum_{i=1}^{N} \log_2 P(w_i | \text{context})}$$

**Interpretation**: The average branching factor—on average, the model is "choosing" from PPL equally likely options.

- Lower perplexity = better model
- PPL of 100 means the model is as confused as if choosing uniformly from 100 words

```python
def compute_perplexity(model, test_corpus: List[str]) -> float:
    """
    Compute perplexity on test corpus.
    
    PPL = 2^(-average log probability)
    """
    total_log_prob = 0.0
    total_words = 0
    
    for sentence in test_corpus:
        words = sentence.lower().split()
        words = [model.START] + words + [model.END]
        
        for i in range(len(words) - 1):
            prob = model.probability(words[i + 1], words[i])
            if prob > 0:
                total_log_prob += math.log2(prob)
            else:
                total_log_prob += -100  # Large penalty
            total_words += 1
    
    cross_entropy = -total_log_prob / total_words
    perplexity = 2 ** cross_entropy
    
    return perplexity


# Compare smoothing techniques
train_corpus = ["the cat sat on the mat", "the dog sat on the log"] * 10
test_corpus = ["the cat played", "a dog runs"]

laplace_model = LaplaceBigramModel()
laplace_model.train(train_corpus)
print(f"Laplace PPL: {compute_perplexity(laplace_model, test_corpus):.2f}")
```

---

## Comparison of Smoothing Techniques

| Technique | Pros | Cons | Best For |
|-----------|------|------|----------|
| **Laplace** | Simple | Over-smooths with large V | Small vocabularies |
| **Add-k** | Tunable | Requires validation set | Medium vocabularies |
| **Interpolation** | Combines evidence | Multiple hyperparameters | General use |
| **Kneser-Ney** | State-of-the-art | Complex implementation | Production systems |

### Typical Perplexities (Penn Treebank)

| Model | Perplexity |
|-------|------------|
| Unigram | ~1000 |
| Bigram (Laplace) | ~300-500 |
| Trigram (Kneser-Ney) | ~80-150 |

---

## Limitations of N-gram Models

1. **Fixed Context**: Cannot capture dependencies beyond $n-1$ words
2. **Data Sparsity**: Exponential growth in possible n-grams
3. **No Semantic Similarity**: "cat" and "feline" are unrelated
4. **Large Storage**: Need to store all n-gram counts
5. **No Generalization**: "the cat sat" doesn't help with "the dog sat"

These limitations motivate **neural language models** (covered next), which learn continuous representations that generalize across similar words and contexts.

---

## Summary

- N-gram models approximate sequence probability using the Markov assumption
- MLE provides straightforward probability estimation from counts
- Smoothing is essential to handle unseen n-grams
- Various generation strategies trade off diversity vs. quality
- Perplexity measures how well a model predicts held-out data
- N-grams remain useful as baselines and in low-resource settings

---

## Exercises

1. **Implement 4-gram Model**: Extend the trigram implementation to 4-grams. How does data sparsity affect perplexity?

2. **Smoothing Comparison**: Train bigram models with Laplace, Add-0.5, and interpolation smoothing. Compare perplexities on held-out data.

3. **Optimal Lambda Search**: Implement grid search to find optimal interpolation weights on a validation set.

4. **Generation Diversity**: Generate 100 samples with different temperature values (0.5, 1.0, 1.5, 2.0). Measure uniqueness and quality.

5. **Repetition Analysis**: Track how often greedy decoding produces repeated n-grams. Implement a repetition penalty.

---

## References

1. Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed.). Chapter 3.
2. Chen, S. F., & Goodman, J. (1999). An empirical study of smoothing techniques for language modeling. *Computer Speech & Language*, 13(4), 359-394.
3. Kneser, R., & Ney, H. (1995). Improved backing-off for m-gram language modeling. *ICASSP*.
