# Subword Embeddings

## Overview

Subword embeddings decompose words into smaller units — typically character n-grams — and compose word-level representations from these pieces. This approach, pioneered by FastText, solves the fundamental limitation of word-level embeddings: their inability to represent unseen or rare words. By sharing parameters across morphologically related words, subword methods achieve better generalization, especially for morphologically rich languages.

## Learning Objectives

By the end of this section, you will:

- Extract character n-grams with boundary markers
- Implement n-gram hashing for vocabulary management
- Compose word embeddings from subword representations
- Handle out-of-vocabulary words using n-gram composition
- Compare pooling strategies (sum, mean, attention) for subword aggregation

## Character N-grams

### Definition

A character n-gram is a contiguous sequence of $n$ characters. FastText-style models use n-grams with $n$ typically ranging from 3 to 6, with special boundary markers `<` and `>` to encode word-initial and word-final positions.

### Extracting N-grams

For the word "where", boundary markers produce `<where>`, yielding:

| $n$ | N-grams |
|-----|---------|
| 3 | `<wh`, `whe`, `her`, `ere`, `re>` |
| 4 | `<whe`, `wher`, `here`, `ere>` |
| 5 | `<wher`, `where`, `here>` |
| 6 | `<where`, `where>` |

Plus the special full-word token `<where>`.

```python
def extract_ngrams(word, min_n=3, max_n=6):
    """
    Extract character n-grams from a word with boundary markers.
    
    Args:
        word: Input word string
        min_n: Minimum n-gram length (default: 3)
        max_n: Maximum n-gram length (default: 6)
    
    Returns:
        Set of n-gram strings including the full word token
    """
    # Add boundary markers
    word_with_markers = f"<{word}>"
    ngrams = set()
    
    # Extract n-grams of each length
    for n in range(min_n, max_n + 1):
        for i in range(len(word_with_markers) - n + 1):
            ngram = word_with_markers[i:i+n]
            ngrams.add(ngram)
    
    # Add the full word as a special token
    ngrams.add(word_with_markers)
    
    return ngrams


# Example
word = "where"
ngrams = extract_ngrams(word, min_n=3, max_n=6)
print(f"N-grams for '{word}':")
for n in range(3, 7):
    n_grams_of_length_n = sorted([ng for ng in ngrams if len(ng) == n])
    if n_grams_of_length_n:
        print(f"  {n}-grams: {n_grams_of_length_n}")
```

**Output:**
```
N-grams for 'where':
  3-grams: ['<wh', 'ere', 'her', 're>', 'whe']
  4-grams: ['<whe', 'ere>', 'here', 'wher']
  5-grams: ['<wher', 'here>', 'where']
  6-grams: ['<where', 'where>']
```

### Morphological Sharing via N-grams

Related words share n-grams, enabling parameter sharing that captures morphological relationships:

| Word 1 | Word 2 | Shared N-grams | Relationship |
|--------|--------|----------------|--------------|
| "happy" | "unhappy" | `happ`, `appy`, `ppy>` | Negation prefix |
| "run" | "running" | `run`, `<run` | Inflectional form |
| "nation" | "national" | `natio`, `ation`, `tion` | Derivational suffix |
| "compute" | "computer" | `compu`, `omput`, `mpute` | Agent noun |

## N-gram Hashing

### The Hash Table Approach

The number of distinct character n-grams can be enormous. Rather than maintaining an explicit n-gram vocabulary, FastText maps n-grams to a fixed-size hash table:

$$h: \text{n-gram} \rightarrow \{0, 1, \ldots, B-1\}$$

where $B$ is the number of hash buckets (typically $B = 2{,}000{,}000$).

```python
def hash_ngram(ngram, num_buckets=2000000):
    """
    Hash an n-gram string to a bucket index using FNV-1a hash.
    
    Different n-grams may collide (map to the same bucket),
    sharing an embedding vector. With B=2M, collision rate is low.
    """
    FNV_PRIME = 0x01000193
    FNV_OFFSET = 0x811c9dc5
    
    h = FNV_OFFSET
    for char in ngram.encode('utf-8'):
        h ^= char
        h = (h * FNV_PRIME) & 0xFFFFFFFF
    
    return h % num_buckets
```

## Composing Word Embeddings

### Sum Pooling (Default)

The standard approach sums all n-gram embeddings plus the word-level embedding:

$$\mathbf{v}_w = \mathbf{z}_{\langle w \rangle} + \sum_{g \in \mathcal{G}_w} \mathbf{z}_{h(g)}$$

where $\mathbf{z}_{\langle w \rangle}$ is the whole-word embedding (only for in-vocabulary words) and $\mathbf{z}_{h(g)}$ is the embedding at hash bucket $h(g)$.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def compose_word_embedding(word, word_idx, ngram_embeddings, word_embeddings, 
                           min_n=3, max_n=6, num_buckets=2000000):
    """
    Compose a word embedding from its subword n-grams.
    
    Args:
        word: Word string
        word_idx: Vocabulary index (None for OOV words)
        ngram_embeddings: nn.Embedding for n-gram hash buckets
        word_embeddings: nn.Embedding for whole words
    
    Returns:
        Composed word embedding tensor
    """
    ngrams = extract_ngrams(word, min_n, max_n)
    indices = torch.tensor([hash_ngram(ng, num_buckets) for ng in ngrams], dtype=torch.long)
    
    # Sum n-gram embeddings
    ngram_embs = ngram_embeddings(indices)
    word_emb = ngram_embs.sum(dim=0)
    
    # Add word-level embedding if in vocabulary
    if word_idx is not None:
        word_emb = word_emb + word_embeddings.weight[word_idx]
    
    return word_emb
```

### Mean Pooling

Normalizes by the number of n-grams to prevent longer words from having larger-magnitude embeddings:

$$\mathbf{v}_w = \frac{1}{|\mathcal{G}_w|} \sum_{g \in \mathcal{G}_w} \mathbf{z}_{h(g)}$$

### Attention Pooling

Learns which n-grams are most informative for each word:

$$\mathbf{v}_w = \sum_{g \in \mathcal{G}_w} \alpha_g \, \mathbf{z}_{h(g)}$$

where $\alpha_g = \text{softmax}(a(\mathbf{z}_{h(g)}))$ and $a$ is a learned scoring function.

```python
class AttentionSubwordComposer(nn.Module):
    """Compose word embeddings from subword n-grams using learned attention."""
    
    def __init__(self, num_buckets, embedding_dim):
        super().__init__()
        self.ngram_embeddings = nn.Embedding(num_buckets, embedding_dim)
        self.attention = nn.Linear(embedding_dim, 1)
    
    def forward(self, ngram_indices):
        """
        Args:
            ngram_indices: (num_ngrams,) hashed n-gram indices for one word
        
        Returns:
            Composed word embedding of shape (embedding_dim,)
        """
        ngram_embs = self.ngram_embeddings(ngram_indices)  # (num_ngrams, dim)
        
        attn_scores = self.attention(ngram_embs).squeeze(-1)  # (num_ngrams,)
        attn_weights = F.softmax(attn_scores, dim=0)
        
        word_emb = (attn_weights.unsqueeze(1) * ngram_embs).sum(dim=0)
        return word_emb
```

## Handling OOV Words

### The Key Advantage

FastText-style models can generate embeddings for **any word**, including those never seen during training:

```python
def get_oov_embedding(word, ngram_embeddings, min_n=3, max_n=6, num_buckets=2000000):
    """
    Get embedding for an out-of-vocabulary word.
    
    The embedding is computed purely from character n-grams.
    """
    ngrams = extract_ngrams(word, min_n, max_n)
    indices = torch.tensor([hash_ngram(ng, num_buckets) for ng in ngrams], dtype=torch.long)
    return ngram_embeddings(indices).sum(dim=0)


# Example
ngram_emb_layer = nn.Embedding(2000000, 100)
oov_words = ["unforgettableness", "COVID-19", "blockchain", "deeplearning"]
for word in oov_words:
    emb = get_oov_embedding(word, ngram_emb_layer)
    print(f"'{word}': shape = {emb.shape}, norm = {emb.norm().item():.4f}")
```

### Typo Robustness

Because typos share most n-grams with the correct spelling, their embeddings remain close:

```python
correct = "restaurant"
typos = ["restaruant", "resturant", "restaurnt"]

correct_ngrams = extract_ngrams(correct)
for typo in typos:
    typo_ngrams = extract_ngrams(typo)
    overlap = correct_ngrams & typo_ngrams
    total = correct_ngrams | typo_ngrams
    jaccard = len(overlap) / len(total)
    print(f"'{correct}' vs '{typo}': Jaccard n-gram similarity = {jaccard:.2%}")
```

## N-gram Range Selection

| Language Type | Recommended Range | Rationale |
|---------------|-------------------|-----------|
| English | 3–6 | Captures common prefixes/suffixes |
| German | 3–6 | Handles long compound words via overlap |
| Finnish / Turkish | 4–7 | Longer morphological affixes |
| Chinese / Japanese | 1–3 | Character-level semantics |
| Social media | 3–5 | Short tokens, abbreviations |

## Subword Tokenization Beyond N-grams

While FastText uses overlapping character n-grams, modern transformer models employ data-driven subword tokenization:

| Method | Used By | Approach |
|--------|---------|----------|
| **Character n-grams** | FastText | Overlapping character windows |
| **Byte Pair Encoding (BPE)** | GPT, RoBERTa | Iteratively merge frequent character pairs |
| **WordPiece** | BERT | Likelihood-based subword merging |
| **Unigram LM** | SentencePiece, T5 | Probabilistic subword selection |

These methods share FastText's core insight — representing words via composable subunits — but learn the subword vocabulary from data rather than using fixed-length character windows. BPE and WordPiece produce non-overlapping, variable-length subwords, which are better suited for sequence-to-sequence architectures.

## Key Takeaways

!!! success "Main Concepts"

    1. **Character n-grams** decompose words into overlapping subword units with boundary markers
    2. **Hashing** maps the large n-gram space to a fixed-size embedding table ($B \approx 2M$)
    3. **Word composition** sums (or pools) n-gram embeddings to produce word-level representations
    4. **OOV handling** is automatic — any word can be embedded via its n-gram components
    5. **Morphological sharing** enables related words to have similar embeddings through shared n-grams

!!! tip "Best Practices"

    - Use n-gram range 3–6 for most European languages
    - Adjust range based on the morphological complexity of the target language
    - Sum pooling is the default; attention pooling can improve quality at the cost of added parameters
    - Pre-trained FastText models are available for 157 languages — use them when possible

## References

- Bojanowski, P., et al. (2017). "Enriching Word Vectors with Subword Information"
- Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine Translation of Rare Words with Subword Units" *(BPE)*
- Schuster, M., & Nakajima, K. (2012). "Japanese and Korean Voice Search" *(WordPiece)*
- Kudo, T. (2018). "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates" *(Unigram LM)*
