"""
BLEU Score Evaluation for Sequence-to-Sequence Models

This module implements BLEU (Bilingual Evaluation Understudy) score, a standard
metric for evaluating machine translation and other sequence generation tasks.

BLEU Score Components:
1. N-gram Precision: Fraction of n-grams in prediction that appear in reference
2. Brevity Penalty: Penalizes predictions shorter than references
3. Corpus-Level BLEU: Accumulated scores across multiple sentence pairs

The score ranges from 0 (no overlap) to 1 (perfect match), typically multiplied
by 100 for reporting. BLEU-4 (averaging BLEU-1 through BLEU-4) is standard.

Educational purpose: Chapter 7 - Sequence-to-Sequence Models & Evaluation
Reference: Papineni et al. (2002) "BLEU: a Method for Automatic Evaluation of
Machine Translation"
"""

import math
from typing import List, Tuple, Dict
from collections import Counter
import numpy as np


# ============================================================================
# N-gram Extraction
# ============================================================================

def get_ngrams(tokens: List[str], n: int) -> Counter:
    """
    Extract n-grams from a sequence of tokens.

    Args:
        tokens: List of string tokens
        n: N-gram order (1 for unigrams, 2 for bigrams, etc.)

    Returns:
        Counter object with n-gram frequencies
    """
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams[ngram] += 1
    return ngrams


# ============================================================================
# Sentence-Level BLEU
# ============================================================================

def sentence_bleu(
    prediction: List[str],
    reference: List[str],
    k: int = 4,
    weights: List[float] = None
) -> float:
    """
    Compute BLEU score for a single (prediction, reference) sentence pair.

    The BLEU score is the geometric mean of n-gram precisions, with a brevity penalty.

    BLEU = BP * exp(sum(w_n * log(p_n)))
    where:
    - BP = brevity penalty
    - p_n = n-gram precision (n=1 to k)
    - w_n = weight for each n-gram (default: uniform)

    Args:
        prediction: Predicted token sequence
        reference: Reference/ground-truth token sequence
        k: Maximum n-gram order (default: 4 for BLEU-4)
        weights: Optional weights for each n-gram level. If None, uses uniform weights.

    Returns:
        BLEU score (0 to 1)
    """
    # Default: uniform weights for each n-gram
    if weights is None:
        weights = [1.0 / k] * k

    # Handle empty predictions
    if len(prediction) == 0:
        return 0.0

    # Compute brevity penalty
    # Penalizes if prediction is shorter than reference
    if len(prediction) > len(reference):
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - len(reference) / max(len(prediction), 1))

    # Compute n-gram precisions
    log_precisions = []
    for n in range(1, k + 1):
        pred_ngrams = get_ngrams(prediction, n)
        ref_ngrams = get_ngrams(reference, n)

        # Count matches: minimum of prediction count and reference count
        matches = 0
        for ngram, count in pred_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))

        # Precision: matches / total predicted n-grams
        total_pred = max(1, sum(pred_ngrams.values()))
        precision = matches / total_pred

        # Avoid log(0) by adding small epsilon
        if precision == 0:
            log_precisions.append(0)
        else:
            log_precisions.append(weights[n - 1] * math.log(precision))

    # Geometric mean: exp(sum of weighted log precisions)
    geo_mean = math.exp(sum(log_precisions))

    return brevity_penalty * geo_mean


# ============================================================================
# Corpus-Level BLEU
# ============================================================================

def corpus_bleu(
    predictions: List[List[str]],
    references: List[List[str]],
    k: int = 4,
    weights: List[float] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute corpus-level BLEU score across multiple sentence pairs.

    Corpus-level BLEU accumulates n-gram matches and counts across all sentences
    before computing precision, avoiding issues where some sentences might have
    zero n-gram matches.

    Args:
        predictions: List of predicted token sequences
        references: List of reference token sequences
        k: Maximum n-gram order
        weights: Weights for each n-gram level

    Returns:
        Tuple of (corpus_bleu_score, dict of individual precisions and metrics)
    """
    if weights is None:
        weights = [1.0 / k] * k

    # Accumulate n-gram statistics across corpus
    total_matches = {n: 0 for n in range(1, k + 1)}
    total_pred_ngrams = {n: 0 for n in range(1, k + 1)}
    total_ref_length = 0
    total_pred_length = 0

    for pred, ref in zip(predictions, references):
        total_ref_length += len(ref)
        total_pred_length += len(pred)

        for n in range(1, k + 1):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)

            # Count matches
            matches = 0
            for ngram, count in pred_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))

            total_matches[n] += matches
            total_pred_ngrams[n] += sum(pred_ngrams.values())

    # Compute brevity penalty
    if total_pred_length > total_ref_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - total_ref_length / max(total_pred_length, 1))

    # Compute n-gram precisions and BLEU
    log_precisions = []
    precisions = {}

    for n in range(1, k + 1):
        precision = total_matches[n] / max(1, total_pred_ngrams[n])
        precisions[f'BLEU-{n}'] = precision * 100  # As percentage

        if precision == 0:
            log_precisions.append(0)
        else:
            log_precisions.append(weights[n - 1] * math.log(precision))

    geo_mean = math.exp(sum(log_precisions))
    corpus_score = brevity_penalty * geo_mean

    # Additional metrics
    metrics = {
        'BLEU': corpus_score * 100,
        **precisions,
        'brevity_penalty': brevity_penalty,
        'pred_length': total_pred_length,
        'ref_length': total_ref_length,
    }

    return corpus_score, metrics


# ============================================================================
# Tokenization Helper
# ============================================================================

def simple_tokenize(sentence: str) -> List[str]:
    """
    Simple tokenization: split on whitespace and punctuation.
    (In practice, specialized tokenizers like NLTK are preferred)

    Args:
        sentence: Input sentence

    Returns:
        List of tokens
    """
    # Very basic: split on whitespace
    return sentence.lower().split()


# ============================================================================
# Demo and Evaluation
# ============================================================================

def main():
    """
    Demo: Compare BLEU scores for good and poor translations.
    Shows how BLEU captures translation quality.
    """
    print("BLEU Score Evaluation for Sequence-to-Sequence Models")
    print("=" * 70)

    # Example 1: Good translation
    reference_1 = simple_tokenize("the quick brown fox jumps over the lazy dog")
    prediction_good_1 = simple_tokenize("the quick brown fox jumps over the lazy dog")
    prediction_poor_1 = simple_tokenize("quick brown fox jumps lazy dog")

    print("\nExample 1: Simple Case")
    print(f"Reference:    {' '.join(reference_1)}")
    print(f"Good Pred:    {' '.join(prediction_good_1)}")
    print(f"Poor Pred:    {' '.join(prediction_poor_1)}")

    good_bleu_1 = sentence_bleu(prediction_good_1, reference_1)
    poor_bleu_1 = sentence_bleu(prediction_poor_1, reference_1)

    print(f"\nGood prediction BLEU: {good_bleu_1 * 100:.2f}")
    print(f"Poor prediction BLEU: {poor_bleu_1 * 100:.2f}")

    # Example 2: Machine translation task
    print("\n" + "=" * 70)
    print("\nExample 2: Multiple Sentence Pairs (Corpus BLEU)")

    references = [
        simple_tokenize("the cat sat on the mat"),
        simple_tokenize("hello world from python"),
        simple_tokenize("machine translation is important"),
    ]

    predictions_good = [
        simple_tokenize("the cat sat on the mat"),
        simple_tokenize("hello world from python"),
        simple_tokenize("machine translation is important"),
    ]

    predictions_partial = [
        simple_tokenize("the cat on mat"),
        simple_tokenize("hello world python"),
        simple_tokenize("machine translation important"),
    ]

    predictions_poor = [
        simple_tokenize("cat sat mat"),
        simple_tokenize("hello python"),
        simple_tokenize("translation is important"),
    ]

    # Compute corpus BLEU
    good_score, good_metrics = corpus_bleu(predictions_good, references)
    partial_score, partial_metrics = corpus_bleu(predictions_partial, references)
    poor_score, poor_metrics = corpus_bleu(predictions_poor, references)

    print("\nGood predictions (perfect match):")
    print(f"  BLEU: {good_metrics['BLEU']:.2f}")
    print(f"  BLEU-1: {good_metrics['BLEU-1']:.2f}, BLEU-2: {good_metrics['BLEU-2']:.2f}")
    print(f"  BLEU-3: {good_metrics['BLEU-3']:.2f}, BLEU-4: {good_metrics['BLEU-4']:.2f}")

    print("\nPartial predictions (missing some words):")
    print(f"  BLEU: {partial_metrics['BLEU']:.2f}")
    print(f"  BLEU-1: {partial_metrics['BLEU-1']:.2f}, BLEU-2: {partial_metrics['BLEU-2']:.2f}")
    print(f"  BLEU-3: {partial_metrics['BLEU-3']:.2f}, BLEU-4: {partial_metrics['BLEU-4']:.2f}")

    print("\nPoor predictions (low n-gram overlap):")
    print(f"  BLEU: {poor_metrics['BLEU']:.2f}")
    print(f"  BLEU-1: {poor_metrics['BLEU-1']:.2f}, BLEU-2: {poor_metrics['BLEU-2']:.2f}")
    print(f"  BLEU-3: {poor_metrics['BLEU-3']:.2f}, BLEU-4: {poor_metrics['BLEU-4']:.2f}")

    # Example 3: Brevity penalty effect
    print("\n" + "=" * 70)
    print("\nExample 3: Brevity Penalty Effect")

    reference_3 = simple_tokenize("the quick brown fox jumps over the lazy dog")
    prediction_short = simple_tokenize("quick brown fox")
    prediction_long = simple_tokenize("the quick brown fox jumps over the lazy dog and beyond")

    short_score, short_metrics = corpus_bleu(
        [prediction_short], [reference_3]
    )
    long_score, long_metrics = corpus_bleu(
        [prediction_long], [reference_3]
    )

    print(f"Reference length: {len(reference_3)} tokens")
    print(f"\nShort prediction ({len(prediction_short)} tokens):")
    print(f"  BLEU: {short_metrics['BLEU']:.2f}")
    print(f"  Brevity penalty: {short_metrics['brevity_penalty']:.4f}")

    print(f"\nLong prediction ({len(prediction_long)} tokens):")
    print(f"  BLEU: {long_metrics['BLEU']:.2f}")
    print(f"  Brevity penalty: {long_metrics['brevity_penalty']:.4f}")

    print("\n" + "=" * 70)
    print("\nKey Insights:")
    print("  1. Perfect match achieves BLEU = 100.0")
    print("  2. Missing words reduce n-gram precision")
    print("  3. Shorter predictions are penalized (brevity penalty)")
    print("  4. BLEU-4 emphasizes longer n-gram matches (sequences)")
    print("  5. Corpus BLEU is more robust than sentence BLEU")


if __name__ == "__main__":
    main()
