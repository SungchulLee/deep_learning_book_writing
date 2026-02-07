# Subword Segmentation for Machine Translation

## Learning Objectives

- Understand the open vocabulary problem in MT
- Apply BPE and SentencePiece for subword tokenization
- Design shared vocabularies for multilingual MT

## The Open Vocabulary Problem

MT models must handle words never seen during training — morphological variants, proper nouns, technical terms, and compound words. A fixed word-level vocabulary maps all unknown words to a single `<UNK>` token, losing critical information.

## Byte Pair Encoding (BPE)

BPE (Sennrich et al., 2016) iteratively merges the most frequent character pair:

1. Initialize vocabulary with individual characters + end-of-word marker
2. Count all adjacent symbol pairs in the corpus
3. Merge the most frequent pair into a new symbol
4. Repeat for $k$ merge operations

### Example

Corpus frequencies: `low (5), lower (2), newest (6), widest (3)`

Initial: `l o w _, l o w e r _, n e w e s t _, w i d e s t _`

After merges: `e s` -> `es`, `es t` -> `est`, `l o` -> `lo`, `lo w` -> `low`, ...

Result: Common words remain whole; rare words split into meaningful subwords.

## SentencePiece

Language-agnostic tokenization that treats the input as a raw byte stream — no language-specific preprocessing required:

```python
import sentencepiece as spm

# Train tokenizer
spm.SentencePieceTrainer.train(
    input='parallel_corpus.txt',
    model_prefix='mt_tokenizer',
    vocab_size=32000,
    model_type='bpe',  # or 'unigram'
)

# Use tokenizer
sp = spm.SentencePieceProcessor(model_file='mt_tokenizer.model')
tokens = sp.encode("Machine translation is fascinating", out_type=str)
# ['_Machine', '_trans', 'lation', '_is', '_fasci', 'nating']
```

## Shared vs. Separate Vocabularies

| Strategy | Advantages | Disadvantages |
|----------|-----------|---------------|
| Separate | Language-specific optimization | No cross-lingual sharing |
| Joint/Shared | Cross-lingual subword overlap | May favor one language |

For related languages (e.g., Spanish-Portuguese), shared BPE captures cognates as shared subwords, improving translation quality. For distant language pairs (e.g., English-Chinese), separate vocabularies often perform better.

## Impact on MT Quality

Subword segmentation was a crucial innovation that largely solved the OOV problem. Combined with the Transformer architecture, it enabled MT systems to handle arbitrary input text without an explicit vocabulary limitation.

## References

1. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. *ACL*.
2. Kudo, T., & Richardson, J. (2018). SentencePiece: A Simple and Language Independent Subword Tokenizer. *EMNLP*.
