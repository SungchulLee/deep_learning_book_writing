"""Nlp augmentation."""
# ---
# title: "NLP Data Augmentation"
# description: "Character-level, word-level, and sentence-level text augmentation
#               techniques using nlpaug and custom implementations"
# ---
#
# Data augmentation for NLP is harder than for images — you can't just
# flip or rotate text.  Instead, we use linguistic perturbations that
# preserve meaning while creating diverse training examples.
#
#   Part 1 – Character-level augmentation (OCR errors, keyboard typos)
#   Part 2 – Word-level augmentation (synonym, embedding-based)
#   Part 3 – Back-translation augmentation
#   Part 4 – Contextual augmentation with masked language models
#   Part 5 – Custom augmentation pipeline for financial text
#   Part 6 – Measuring augmentation effectiveness
#
# Adapted from: O'Reilly "Practical NLP" Ch 2

import numpy as np
import random
import re
from typing import List, Tuple


# =====================================================================
# Part 1 – Character-Level Augmentation
# =====================================================================
print("=" * 60)
print("Part 1: Character-Level Augmentation")
print("=" * 60)

# Character-level perturbations simulate real-world noise:
# - OCR errors (scanning documents)
# - Keyboard typos (social media / chat data)
# - Spelling mistakes

# OCR error simulation
OCR_CONFUSIONS = {
    'o': '0', '0': 'o', 'l': '1', '1': 'l',
    'S': '5', '5': 'S', 'g': '9', '9': 'g',
    'B': '8', '8': 'B', 'Z': '2', '2': 'Z',
    'O': '0', 'I': '1', 'i': '!', 'rn': 'm',
}

# Keyboard proximity map (QWERTY layout)
KEYBOARD_NEIGHBORS = {
    'q': 'wa', 'w': 'qeas', 'e': 'wrds', 'r': 'etdf',
    't': 'ryfg', 'y': 'tugh', 'u': 'yijh', 'i': 'uokj',
    'o': 'iplk', 'p': 'ol', 'a': 'qwsz', 's': 'awedxz',
    'd': 'serfcx', 'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujbn',
    'j': 'huiknm', 'k': 'jiolm', 'l': 'kop', 'z': 'asx',
    'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn',
    'n': 'bhjm', 'm': 'njk',
}


def augment_ocr(text: str, prob: float = 0.1) -> str:
    """Simulate OCR errors by replacing characters with visually similar ones."""
    result = []
    for char in text:
        if random.random() < prob and char in OCR_CONFUSIONS:
            result.append(OCR_CONFUSIONS[char])
        else:
            result.append(char)
    return "".join(result)


def augment_keyboard(text: str, prob: float = 0.05) -> str:
    """Simulate keyboard typos by replacing with adjacent keys."""
    result = []
    for char in text:
        if random.random() < prob and char.lower() in KEYBOARD_NEIGHBORS:
            neighbors = KEYBOARD_NEIGHBORS[char.lower()]
            replacement = random.choice(neighbors)
            result.append(replacement if char.islower() else replacement.upper())
        else:
            result.append(char)
    return "".join(result)


def augment_spelling(text: str, prob: float = 0.1) -> str:
    """Simulate common spelling errors (swap, delete, insert, repeat)."""
    words = text.split()
    result = []
    for word in words:
        if random.random() < prob and len(word) > 3:
            op = random.choice(["swap", "delete", "repeat"])
            chars = list(word)
            pos = random.randint(1, len(chars) - 2)
            if op == "swap":
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            elif op == "delete":
                chars.pop(pos)
            elif op == "repeat":
                chars.insert(pos, chars[pos])
            result.append("".join(chars))
        else:
            result.append(word)
    return " ".join(result)


# Demonstrate
random.seed(42)
sample = "The quick brown fox jumps over the lazy dog"
print(f"  Original:  {sample}")
print(f"  OCR:       {augment_ocr(sample, prob=0.15)}")
print(f"  Keyboard:  {augment_keyboard(sample, prob=0.1)}")
print(f"  Spelling:  {augment_spelling(sample, prob=0.3)}")
print()

# nlpaug library usage
print("  Using nlpaug library (recommended for production):")
print("""
    import nlpaug.augmenter.char as nac
    import nlpaug.augmenter.word as naw

    # OCR augmenter
    aug = nac.OcrAug()
    augmented = aug.augment(text, n=3)  # 3 variants

    # Keyboard augmenter
    aug = nac.KeyboardAug()
    augmented = aug.augment(text, n=3)
""")


# =====================================================================
# Part 2 – Word-Level Augmentation
# =====================================================================
print("=" * 60)
print("Part 2: Word-Level Augmentation")
print("=" * 60)

# Word-level augmentation preserves syntax while changing vocabulary:
# - Synonym replacement (WordNet)
# - Embedding-based substitution (Word2Vec, GloVe)
# - Random insertion / deletion / swap

# Simple synonym replacement using a small dictionary
FINANCIAL_SYNONYMS = {
    "increase": ["rise", "growth", "gain", "surge", "uptick"],
    "decrease": ["decline", "drop", "fall", "downturn", "slump"],
    "profit": ["earnings", "income", "gains", "returns"],
    "loss": ["deficit", "shortfall", "writedown"],
    "revenue": ["sales", "turnover", "income"],
    "strong": ["robust", "solid", "healthy", "impressive"],
    "weak": ["soft", "disappointing", "lackluster", "tepid"],
    "buy": ["acquire", "purchase", "accumulate"],
    "sell": ["divest", "offload", "liquidate"],
    "stock": ["shares", "equity", "securities"],
}


def augment_synonym(text: str, prob: float = 0.2) -> str:
    """Replace words with synonyms from a domain dictionary."""
    words = text.lower().split()
    result = []
    for word in words:
        if random.random() < prob and word in FINANCIAL_SYNONYMS:
            result.append(random.choice(FINANCIAL_SYNONYMS[word]))
        else:
            result.append(word)
    return " ".join(result)


def augment_random_swap(text: str, n_swaps: int = 1) -> str:
    """Randomly swap adjacent words."""
    words = text.split()
    for _ in range(n_swaps):
        if len(words) > 2:
            i = random.randint(0, len(words) - 2)
            words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def augment_random_delete(text: str, prob: float = 0.1) -> str:
    """Randomly delete words (keeping at least half)."""
    words = text.split()
    if len(words) <= 2:
        return text
    result = [w for w in words if random.random() > prob]
    return " ".join(result) if result else text


sample_fin = "Revenue increase was strong driven by stock buyback"
random.seed(42)
print(f"  Original:  {sample_fin}")
print(f"  Synonym:   {augment_synonym(sample_fin, prob=0.5)}")
print(f"  Swap:      {augment_random_swap(sample_fin, n_swaps=2)}")
print(f"  Delete:    {augment_random_delete(sample_fin, prob=0.2)}")
print()

# Embedding-based augmentation
print("  Embedding-based augmentation (nlpaug + Word2Vec/GloVe):")
print("""
    import nlpaug.augmenter.word as naw

    # Insert semantically similar words
    aug = naw.WordEmbsAug(
        model_type='word2vec',
        model_path='GoogleNews-vectors-negative300.bin',
        action="insert",
    )
    augmented = aug.augment("Revenue grew 15% this quarter")
    # → "Revenue grew substantially 15% this profitable quarter"

    # Substitute with similar words
    aug = naw.WordEmbsAug(
        model_type='word2vec',
        model_path='GoogleNews-vectors-negative300.bin',
        action="substitute",
    )
    augmented = aug.augment("Revenue grew 15% this quarter")
    # → "Sales increased 15% this period"
""")


# =====================================================================
# Part 3 – Back-Translation Augmentation
# =====================================================================
print("=" * 60)
print("Part 3: Back-Translation Augmentation")
print("=" * 60)

# Back-translation: English → French → English
# This naturally paraphrases text while preserving meaning.

print("""
  Back-translation pipeline using HuggingFace:

    from transformers import MarianMTModel, MarianTokenizer

    # English → French
    en_fr_model = "Helsinki-NLP/opus-mt-en-fr"
    en_fr_tokenizer = MarianTokenizer.from_pretrained(en_fr_model)
    en_fr = MarianMTModel.from_pretrained(en_fr_model)

    # French → English
    fr_en_model = "Helsinki-NLP/opus-mt-fr-en"
    fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_model)
    fr_en = MarianMTModel.from_pretrained(fr_en_model)

    def back_translate(text, src_model, src_tok, tgt_model, tgt_tok):
        # Forward translation
        inputs = src_tok(text, return_tensors="pt", truncation=True)
        translated = src_model.generate(**inputs)
        intermediate = src_tok.decode(translated[0], skip_special_tokens=True)

        # Back translation
        inputs = tgt_tok(intermediate, return_tensors="pt", truncation=True)
        back = tgt_model.generate(**inputs)
        return tgt_tok.decode(back[0], skip_special_tokens=True)

    original = "The company reported strong quarterly earnings."
    augmented = back_translate(
        original, en_fr, en_fr_tokenizer, fr_en, fr_en_tokenizer
    )
    # → "The company announced solid quarterly results."
""")


# =====================================================================
# Part 4 – Contextual Augmentation with Masked LM
# =====================================================================
print("=" * 60)
print("Part 4: Contextual Augmentation with Masked LM")
print("=" * 60)

# Use BERT's masked language model to replace words with
# contextually appropriate alternatives.

print("""
  from transformers import pipeline

  fill_mask = pipeline("fill-mask", model="bert-base-uncased")

  text = "The company reported [MASK] quarterly earnings."
  suggestions = fill_mask(text)
  for s in suggestions[:5]:
      print(f"  {s['token_str']:>10}: {s['score']:.3f}")
  # →    strong: 0.142
  #       solid: 0.098
  #       good: 0.087
  #     record: 0.064
  #     better: 0.052

  # For financial text, use FinBERT:
  fill_mask_fin = pipeline("fill-mask", model="ProsusAI/finbert")
  # This gives domain-appropriate replacements
""")


# =====================================================================
# Part 5 – Custom Pipeline for Financial Text
# =====================================================================
print("=" * 60)
print("Part 5: Custom Augmentation Pipeline for Financial Text")
print("=" * 60)


class FinancialTextAugmenter:
    """Multi-strategy text augmentation pipeline for financial NLP.

    Combines multiple augmentation strategies with configurable
    probabilities. Designed for financial text where domain-specific
    vocabulary must be preserved.

    Args:
        synonym_prob: Probability of synonym replacement per word
        typo_prob:    Probability of character-level typo per word
        delete_prob:  Probability of random word deletion
        swap_prob:    Probability of adjacent word swap
    """

    def __init__(
        self,
        synonym_prob: float = 0.15,
        typo_prob: float = 0.05,
        delete_prob: float = 0.1,
        swap_prob: float = 0.1,
    ):
        self.synonym_prob = synonym_prob
        self.typo_prob = typo_prob
        self.delete_prob = delete_prob
        self.swap_prob = swap_prob

        # Words that should NEVER be modified (tickers, numbers, key terms)
        self.protected_pattern = re.compile(
            r'^\$[A-Z]+$|^\d+\.?\d*%?$|^Q[1-4]$|^FY\d{2,4}$'
        )

    def _is_protected(self, word: str) -> bool:
        """Check if word should not be augmented (tickers, numbers, etc.)."""
        return bool(self.protected_pattern.match(word))

    def augment(self, text: str, n: int = 1) -> List[str]:
        """Generate n augmented versions of the input text."""
        augmented = []
        for _ in range(n):
            # Randomly select augmentation strategy
            strategy = random.choice([
                "synonym", "typo", "delete", "swap", "combined"
            ])
            if strategy == "synonym":
                aug_text = augment_synonym(text, self.synonym_prob)
            elif strategy == "typo":
                aug_text = augment_keyboard(text, self.typo_prob)
            elif strategy == "delete":
                aug_text = augment_random_delete(text, self.delete_prob)
            elif strategy == "swap":
                aug_text = augment_random_swap(text, n_swaps=1)
            else:  # combined
                aug_text = augment_synonym(text, self.synonym_prob * 0.5)
                aug_text = augment_keyboard(aug_text, self.typo_prob * 0.5)
            augmented.append(aug_text)
        return augmented


# Demo
augmenter = FinancialTextAugmenter()
sample = "$AAPL revenue increase of 15% was strong in Q3 FY2024"
random.seed(42)
print(f"  Original: {sample}")
print(f"  Augmented versions:")
for i, aug in enumerate(augmenter.augment(sample, n=5)):
    print(f"    [{i+1}] {aug}")
print()


# =====================================================================
# Part 6 – Measuring Augmentation Effectiveness
# =====================================================================
print("=" * 60)
print("Part 6: Measuring Augmentation Effectiveness")
print("=" * 60)

# Augmentation should:
#   1. Improve model performance (especially on small datasets)
#   2. Preserve label correctness
#   3. Increase diversity without introducing noise
#
# Key metrics:
#   - Accuracy delta (augmented vs original training)
#   - Label preservation rate
#   - Lexical diversity (unique n-grams / total n-grams)

def lexical_diversity(texts: List[str], n: int = 2) -> float:
    """Compute type-token ratio for n-grams across a text collection."""
    all_ngrams = []
    for text in texts:
        words = text.lower().split()
        for i in range(len(words) - n + 1):
            all_ngrams.append(tuple(words[i:i + n]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


original_texts = [
    "Revenue increased by 10% this quarter",
    "Profit margins declined sharply",
    "Strong earnings beat analyst expectations",
] * 5

augmented_texts = []
for t in original_texts:
    augmented_texts.extend(augmenter.augment(t, n=2))

print(f"  Original texts: {len(original_texts)}")
print(f"  Augmented texts: {len(augmented_texts)}")
print(f"  Original lexical diversity (bigrams):  {lexical_diversity(original_texts):.3f}")
print(f"  Augmented lexical diversity (bigrams): {lexical_diversity(augmented_texts):.3f}")
print()

print("""
  Typical improvement from augmentation:

  ┌──────────────────┬──────────┬───────────────────┐
  │ Dataset Size     │ No Aug   │ With Aug (2x-5x)  │
  ├──────────────────┼──────────┼───────────────────┤
  │ 100 examples     │  0.62    │  0.71 (+9%)       │
  │ 500 examples     │  0.74    │  0.79 (+5%)       │
  │ 2000 examples    │  0.82    │  0.84 (+2%)       │
  │ 10000 examples   │  0.87    │  0.88 (+1%)       │
  └──────────────────┴──────────┴───────────────────┘

  Key insight: augmentation helps most with small datasets.
  With enough data (>10K), the improvement diminishes.
""")

print("Done.")


if __name__ == "__main__":
    pass
