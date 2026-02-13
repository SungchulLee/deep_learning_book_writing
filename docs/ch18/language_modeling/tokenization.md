# Tokenization: BPE, WordPiece, and SentencePiece

## Learning Objectives

By the end of this section, you will be able to:

- Understand the role of tokenization in language models
- Implement Byte Pair Encoding (BPE) from scratch
- Explain WordPiece and SentencePiece algorithms
- Select appropriate tokenization for different use cases

---

## Why Tokenization Matters

Tokenization converts raw text into discrete units for the model. The choice profoundly affects:

1. **Vocabulary size**: Memory and computation requirements
2. **OOV handling**: Robustness to unseen words
3. **Sequence length**: Context window utilization
4. **Multilingual support**: Character coverage

### Evolution of Tokenization

| Era | Approach | Example |
|-----|----------|---------|
| Classical | Word-level | "playing" → `playing` |
| Character | Char-level | "playing" → `p l a y i n g` |
| Modern | Subword | "playing" → `play ##ing` |

---

## Word-Level Tokenization

### Simple Approach

```python
def word_tokenize(text: str) -> list:
    """Basic whitespace tokenization."""
    return text.lower().split()

# Example
text = "The cat sat on the mat."
tokens = word_tokenize(text)
# ['the', 'cat', 'sat', 'on', 'the', 'mat.']
```

### Problems

1. **Large vocabulary**: 100K+ words for good coverage
2. **OOV words**: "COVID-19", "cryptocurrency" unknown
3. **Morphology ignored**: "play", "plays", "playing" are separate
4. **Multilingual failure**: Agglutinative languages explode vocabulary

---

## Character-Level Tokenization

### Approach

```python
def char_tokenize(text: str) -> list:
    """Character-level tokenization."""
    return list(text)

# Example
tokens = char_tokenize("Hello")
# ['H', 'e', 'l', 'l', 'o']
```

### Trade-offs

| Pros | Cons |
|------|------|
| Small vocabulary (~300) | Very long sequences |
| No OOV words | Hard to learn semantics |
| Language agnostic | Computationally expensive |

---

## Byte Pair Encoding (BPE)

### Algorithm Overview

BPE iteratively merges the most frequent pairs of tokens:

1. Start with character vocabulary
2. Count all adjacent pairs
3. Merge most frequent pair
4. Repeat until vocabulary size reached

### Implementation

```python
from collections import defaultdict, Counter
import re


class BPE:
    """Byte Pair Encoding tokenizer."""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges = {}  # (a, b) -> ab
        self.vocab = {}
    
    def _get_stats(self, vocab: dict) -> Counter:
        """Count frequency of adjacent pairs."""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def _merge_vocab(self, pair: tuple, vocab: dict) -> dict:
        """Merge all occurrences of pair."""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        
        return new_vocab
    
    def train(self, corpus: list):
        """Learn BPE merges from corpus."""
        # Initialize: word frequencies with space-separated characters
        word_freqs = Counter()
        for text in corpus:
            for word in text.lower().split():
                # Add end-of-word marker
                word_freqs[' '.join(list(word)) + ' </w>'] += 1
        
        vocab = dict(word_freqs)
        
        # Initial vocabulary: all characters
        self.vocab = set()
        for word in vocab:
            for char in word.split():
                self.vocab.add(char)
        
        # Iteratively merge pairs
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            
            # Most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge
            vocab = self._merge_vocab(best_pair, vocab)
            
            # Record merge
            self.merges[best_pair] = ''.join(best_pair)
            self.vocab.add(''.join(best_pair))
            
            if (i + 1) % 100 == 0:
                print(f"Merge {i+1}: {best_pair} -> {''.join(best_pair)}")
        
        print(f"Final vocabulary size: {len(self.vocab)}")
    
    def tokenize(self, text: str) -> list:
        """Tokenize text using learned merges."""
        tokens = []
        
        for word in text.lower().split():
            word = ' '.join(list(word)) + ' </w>'
            
            # Apply merges in order
            while True:
                pairs = [(word.split()[i], word.split()[i+1]) 
                        for i in range(len(word.split()) - 1)]
                
                # Find first applicable merge
                merge_found = False
                for pair in pairs:
                    if pair in self.merges:
                        bigram = ' '.join(pair)
                        replacement = self.merges[pair]
                        word = word.replace(bigram, replacement, 1)
                        merge_found = True
                        break
                
                if not merge_found:
                    break
            
            tokens.extend(word.split())
        
        return tokens


# Example usage
corpus = [
    "the cat sat on the mat",
    "the cat and the dog",
    "the dog sat on the log"
] * 100

bpe = BPE(vocab_size=100)
bpe.train(corpus)

text = "the cat sat"
tokens = bpe.tokenize(text)
print(f"Tokens: {tokens}")
```

### BPE in Practice

GPT-2/3 use byte-level BPE:
- Base vocabulary: 256 bytes
- 50,000 merges
- Handles any UTF-8 text

---

## WordPiece

### Key Differences from BPE

WordPiece (used by BERT) selects merges based on **likelihood increase**, not frequency:

$$\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \times \text{freq}(b)}$$

This favors merges where the combination is more common than expected.

### Special Tokens

- `##` prefix indicates continuation: "playing" → `play ##ing`
- `[UNK]` for unknown tokens
- `[CLS]`, `[SEP]` for sentence structure

### Implementation Sketch

```python
class WordPiece:
    """Simplified WordPiece tokenizer."""
    
    def __init__(self, vocab: set, unk_token='[UNK]'):
        self.vocab = vocab
        self.unk_token = unk_token
    
    def tokenize(self, text: str) -> list:
        """Greedy longest-match tokenization."""
        tokens = []
        
        for word in text.lower().split():
            chars = list(word)
            
            if len(chars) > 200:
                tokens.append(self.unk_token)
                continue
            
            is_bad = False
            start = 0
            sub_tokens = []
            
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                
                if cur_substr is None:
                    is_bad = True
                    break
                
                sub_tokens.append(cur_substr)
                start = end
            
            if is_bad:
                tokens.append(self.unk_token)
            else:
                tokens.extend(sub_tokens)
        
        return tokens
```

---

## SentencePiece

### Unigram Language Model

SentencePiece uses a unigram LM approach:

1. Start with large vocabulary
2. Iteratively remove tokens that least affect likelihood
3. Keep vocabulary that maximizes corpus likelihood

$$P(X) = \prod_{i=1}^{n} P(x_i)$$

### Key Features

- **Language-agnostic**: Works on raw bytes
- **Reversible**: Can reconstruct original text
- **Whitespace handling**: `▁` marks word boundaries

### Using SentencePiece

```python
import sentencepiece as spm

# Train
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    model_type='unigram'  # or 'bpe'
)

# Load and use
sp = spm.SentencePieceProcessor()
sp.load('tokenizer.model')

text = "Hello, world!"
tokens = sp.encode_as_pieces(text)
# ['▁Hello', ',', '▁world', '!']

ids = sp.encode_as_ids(text)
# [1234, 5, 678, 9]

# Decode
sp.decode_pieces(tokens)  # "Hello, world!"
```

---

## Comparison

| Feature | BPE | WordPiece | Unigram |
|---------|-----|-----------|---------|
| Merge criterion | Frequency | Likelihood | Likelihood |
| Direction | Bottom-up | Bottom-up | Top-down |
| Deterministic | Yes | Yes | Sampling option |
| Used by | GPT, RoBERTa | BERT | T5, LLaMA |

### Vocabulary Sizes in Practice

| Model | Tokenizer | Vocab Size |
|-------|-----------|------------|
| GPT-2 | BPE | 50,257 |
| BERT | WordPiece | 30,522 |
| T5 | SentencePiece | 32,000 |
| LLaMA | SentencePiece | 32,000 |

---

## Using HuggingFace Tokenizers

```python
from transformers import AutoTokenizer

# Load pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

text = "Hello, how are you doing today?"
encoded = tokenizer(text)

print(f"Tokens: {tokenizer.tokenize(text)}")
print(f"IDs: {encoded['input_ids']}")
print(f"Decoded: {tokenizer.decode(encoded['input_ids'])}")

# Train custom tokenizer
from tokenizers import Tokenizer, models, trainers

tokenizer = Tokenizer(models.BPE())
trainer = trainers.BpeTrainer(vocab_size=5000, special_tokens=["[PAD]", "[UNK]"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)
```

---

## Best Practices

1. **Match the original**: Use the same tokenizer as the pretrained model
2. **Vocabulary size**: 32K-50K balances efficiency and coverage
3. **Special tokens**: Define consistently (`[CLS]`, `[SEP]`, `[MASK]`)
4. **Multilingual**: Larger vocabulary (100K+) for coverage

---

## Summary

| Method | Best For |
|--------|----------|
| Word-level | Simple, domain-specific |
| BPE | General purpose, generation |
| WordPiece | BERT-family models |
| SentencePiece | Multilingual, production |

Modern practice: Use pretrained tokenizers matching your base model.

---

## Exercises

1. Implement BPE from scratch and visualize merge progression
2. Compare token counts: BPE vs WordPiece on same text
3. Train SentencePiece on a multilingual corpus
4. Analyze tokenization of rare/technical terms

---

## References

1. Sennrich, R., et al. (2016). Neural machine translation with BPE. *ACL*.
2. Wu, Y., et al. (2016). Google's neural machine translation system.
3. Kudo, T. (2018). SentencePiece. *EMNLP*.
4. Kudo, T., & Richardson, J. (2018). SentencePiece: Subword regularization.
