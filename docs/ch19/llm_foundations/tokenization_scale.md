# Tokenization and Scale

## Learning Objectives

- Understand why subword tokenization is essential for LLMs
- Implement Byte Pair Encoding (BPE) from scratch
- Compare tokenization approaches: BPE, WordPiece, SentencePiece, Unigram
- Analyze vocabulary size trade-offs and their impact on model performance
- Identify challenges in tokenizing financial text

## Why Subword Tokenization?

Word-level tokenization faces two fundamental problems:

1. **Vocabulary explosion**: Natural language has effectively unbounded vocabulary
2. **Out-of-vocabulary (OOV)**: Any word not in the fixed vocabulary cannot be processed

Character-level tokenization solves OOV but creates very long sequences and loses word-level semantics.

**Subword tokenization** provides the optimal balance: common words remain single tokens, while rare words are decomposed into meaningful subword units.

## Byte Pair Encoding (BPE)

BPE (Sennrich et al., 2016) is the most widely used tokenization algorithm in LLMs (used by GPT-2, GPT-3, GPT-4, LLaMA).

### Algorithm

1. Initialize vocabulary with all individual characters (or bytes)
2. Count all adjacent token pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat steps 2-3 for a predetermined number of merges

```python
from collections import Counter, defaultdict


def learn_bpe(corpus: list, num_merges: int) -> list:
    # Initialize: split each word into characters + end-of-word marker
    word_freqs = Counter(corpus)
    splits = {word: list(word) + ['</w>'] for word in word_freqs}

    merges = []
    for i in range(num_merges):
        # Count all adjacent pairs
        pair_counts = defaultdict(int)
        for word, freq in word_freqs.items():
            tokens = splits[word]
            for j in range(len(tokens) - 1):
                pair_counts[(tokens[j], tokens[j + 1])] += freq

        if not pair_counts:
            break

        # Find most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)
        merges.append(best_pair)

        # Apply merge to all words
        a, b = best_pair
        merged = a + b
        for word in splits:
            tokens = splits[word]
            new_tokens = []
            idx = 0
            while idx < len(tokens):
                if idx < len(tokens) - 1 and tokens[idx] == a and tokens[idx + 1] == b:
                    new_tokens.append(merged)
                    idx += 2
                else:
                    new_tokens.append(tokens[idx])
                    idx += 1
            splits[word] = new_tokens

    return merges
```

### Vocabulary Size Trade-offs

| Vocabulary Size | Tokens per Word (avg) | Sequence Length | Model Size Impact |
|----------------|----------------------|----------------|-------------------|
| 256 (byte-level) | ~4.5 | Very long | Minimal vocab embedding |
| 32,000 (GPT-2) | ~1.3 | Moderate | 50M params for embeddings |
| 50,257 (GPT-3) | ~1.2 | Moderate | 77M params for embeddings |
| 128,000 (GPT-4) | ~1.0 | Shorter | 197M params for embeddings |

## Byte-Level BPE

GPT-2 introduced **byte-level BPE**: instead of starting from Unicode characters, start from raw bytes (256 base tokens). This guarantees no OOV tokens, language agnosticism, and robustness to code, math notation, and special characters.

## SentencePiece

SentencePiece (Kudo & Richardson, 2018) treats the input as a raw byte stream, eliminating the need for language-specific pre-tokenization. Used by LLaMA, T5, and many multilingual models. Supports both BPE and Unigram Language Model algorithms.

## Financial Text Tokenization Challenges

Financial text presents unique tokenization difficulties:

```python
# How different tokenizers handle financial text
examples = [
    "AAPL Q3 2024 10-K filing shows revenue up 12%",
    "The 3.25% coupon UST 10Y yield hit 4.73%",
    "VIX spiked 42.3% on 2024-08-05 amid JPY carry unwind",
    "EBITDA/EV multiple compressed to 8.2x from 12.1x",
]
# Issues:
# 1. Ticker symbols (AAPL, VIX) may be split unpredictably
# 2. Numbers with decimals (4.73%) tokenized inconsistently
# 3. Date formats (2024-08-05) may fragment
# 4. Financial ratios (EBITDA/EV, 8.2x) are domain-specific
```

For finance applications, consider domain-specific BPE training on financial corpora and special token registration for common tickers and abbreviations.

## References

1. Sennrich, R., et al. (2016). "Neural Machine Translation of Rare Words with Subword Units." *ACL*.
2. Kudo, T. & Richardson, J. (2018). "SentencePiece: A Simple and Language Independent Subword Tokenizer." *EMNLP*.
3. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners."
