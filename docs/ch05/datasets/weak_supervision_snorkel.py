"""Weak supervision snorkel."""
# ---
# title: "Weak Supervision with Snorkel"
# description: "Programmatic data labeling using labeling functions, label
#               aggregation, and noise-aware training — no manual annotation"
# ---
#
# Manual annotation is expensive.  Weak supervision replaces it with
# programmatic labeling functions (LFs) that encode heuristics, patterns,
# and domain knowledge.  Snorkel aggregates these noisy labels into
# probabilistic training labels.
#
#   Part 1 – The labeling function abstraction
#   Part 2 – Types of labeling functions (keyword, regex, NLP, model-based)
#   Part 3 – Label aggregation: majority vote vs label model
#   Part 4 – LF analysis: coverage, overlap, conflict
#   Part 5 – End-to-end pipeline: LFs → label model → classifier
#   Part 6 – Financial weak supervision use cases
#
# Adapted from: O'Reilly "Practical NLP" Ch 2

import numpy as np
import re
from typing import List, Callable, Dict, Optional
from collections import Counter


# =====================================================================
# Part 1 – The Labeling Function Abstraction
# =====================================================================
print("=" * 60)
print("Part 1: Labeling Functions — The Core Abstraction")
print("=" * 60)

# A labeling function (LF) is a Python function that:
#   - Takes a single data point as input
#   - Returns one of: a label, or ABSTAIN (-1)
#
# Key insight: each LF doesn't need to be perfect or cover all data.
# We combine many imperfect LFs to create high-quality labels.
#
# ABSTAIN = -1  (the LF doesn't know)
# NEGATIVE = 0
# POSITIVE = 1

ABSTAIN = -1
NEGATIVE = 0
POSITIVE = 1

# Example: YouTube spam detection
# Instead of labeling 2000 comments manually, write 10 heuristics.

print("""
  # Snorkel API:
  from snorkel.labeling import labeling_function

  @labeling_function()
  def lf_contains_link(x):
      return SPAM if "http" in x.text.lower() else ABSTAIN

  @labeling_function()
  def lf_subscribe(x):
      return SPAM if "subscribe" in x.text.lower() else ABSTAIN

  @labeling_function()
  def lf_short_comment(x):
      return HAM if len(x.text.split()) < 5 else ABSTAIN
""")


# Pure Python implementation (no Snorkel dependency)
class DataPoint:
    """Simple data container mimicking a pandas row."""
    def __init__(self, text: str, **kwargs):
        self.text = text
        for k, v in kwargs.items():
            setattr(self, k, v)


# Define labeling functions as regular Python functions
def lf_contains_link(x: DataPoint) -> int:
    """Spam comments often contain links."""
    return POSITIVE if "http" in x.text.lower() else ABSTAIN


def lf_subscribe(x: DataPoint) -> int:
    """Spam comments ask users to subscribe."""
    return POSITIVE if "subscribe" in x.text.lower() else ABSTAIN


def lf_check_out(x: DataPoint) -> int:
    """Spam comments say 'check out my channel'."""
    return POSITIVE if re.search(r"check.*out", x.text, re.I) else ABSTAIN


def lf_my_channel(x: DataPoint) -> int:
    """Spam comments promote their own channel."""
    return POSITIVE if "my channel" in x.text.lower() else ABSTAIN


def lf_please(x: DataPoint) -> int:
    """Spam comments often beg ('please', 'plz')."""
    text_lower = x.text.lower()
    return POSITIVE if "please" in text_lower or "plz" in text_lower else ABSTAIN


def lf_short_comment(x: DataPoint) -> int:
    """Genuine comments tend to be short reactions."""
    return NEGATIVE if len(x.text.split()) < 5 else ABSTAIN


def lf_song_mention(x: DataPoint) -> int:
    """Comments about the song are likely genuine."""
    return NEGATIVE if "song" in x.text.lower() else ABSTAIN


def lf_sentiment_positive(x: DataPoint) -> int:
    """Highly positive sentiment suggests genuine engagement."""
    positive_words = {"love", "amazing", "awesome", "great", "best", "beautiful"}
    words = set(x.text.lower().split())
    if len(words & positive_words) >= 2:
        return NEGATIVE  # genuine fan comment
    return ABSTAIN


# Demonstrate
sample_comments = [
    DataPoint(text="Check out my channel for more videos http://youtube.com/spam"),
    DataPoint(text="This song is amazing! Love it!"),
    DataPoint(text="Please subscribe to my channel plz"),
    DataPoint(text="lol"),
    DataPoint(text="Great performance, one of the best live shows"),
    DataPoint(text="Subscribe for daily content http://bit.ly/xyz"),
]

lfs = [lf_contains_link, lf_subscribe, lf_check_out, lf_my_channel,
       lf_please, lf_short_comment, lf_song_mention, lf_sentiment_positive]

print("  Labeling function outputs (ABSTAIN=-1, NEG=0, POS=1):")
print(f"  {'Comment':<55} ", end="")
for lf in lfs:
    print(f"{lf.__name__[3:]:>10}", end="")
print()
print("  " + "-" * 135)

for dp in sample_comments:
    text_short = dp.text[:53] + ".." if len(dp.text) > 55 else dp.text
    print(f"  {text_short:<55} ", end="")
    for lf in lfs:
        label = lf(dp)
        symbol = "·" if label == ABSTAIN else str(label)
        print(f"{symbol:>10}", end="")
    print()
print()


# =====================================================================
# Part 2 – Types of Labeling Functions
# =====================================================================
print("=" * 60)
print("Part 2: Types of Labeling Functions")
print("=" * 60)

# 1. Keyword-based: check for specific words
# 2. Pattern-based: regex matching
# 3. Heuristic: length, structure, metadata
# 4. NLP-based: NER, POS tags, sentiment
# 5. External model: use an existing (weak) classifier
# 6. Knowledge base: lookup in dictionary / ontology

print("""
  Type 1: Keyword LF (simplest)
  ──────────────────────────────
    @labeling_function()
    def lf_keyword(x):
        return SPAM if "subscribe" in x.text.lower() else ABSTAIN

  Type 2: Pattern LF (regex)
  ──────────────────────────────
    @labeling_function()
    def lf_regex(x):
        return SPAM if re.search(r"check.*out", x.text, re.I) else ABSTAIN

  Type 3: Heuristic LF (metadata / structure)
  ──────────────────────────────────────────────
    @labeling_function()
    def lf_short(x):
        return HAM if len(x.text.split()) < 5 else ABSTAIN

  Type 4: NLP-based LF (requires preprocessing)
  ────────────────────────────────────────────────
    from snorkel.preprocess.nlp import SpacyPreprocessor
    spacy = SpacyPreprocessor(text_field="text", doc_field="doc")

    @labeling_function(pre=[spacy])
    def lf_has_person(x):
        if any(ent.label_ == "PERSON" for ent in x.doc.ents):
            return HAM
        return ABSTAIN

  Type 5: External model LF
  ──────────────────────────────
    from textblob import TextBlob

    @labeling_function()
    def lf_textblob_positive(x):
        polarity = TextBlob(x.text).sentiment.polarity
        return HAM if polarity > 0.9 else ABSTAIN

  Type 6: Programmatic LF factory (for scaling up)
  ─────────────────────────────────────────────────
    def make_keyword_lf(keywords, label):
        def lf(x):
            if any(w in x.text.lower() for w in keywords):
                return label
            return ABSTAIN
        lf.__name__ = f"keyword_{'_'.join(keywords[:2])}"
        return lf

    lf_money = make_keyword_lf(["earn", "money", "income"], SPAM)
    lf_music = make_keyword_lf(["song", "music", "lyrics"], HAM)
""")


# Programmatic LF factory implementation
def make_keyword_lf(keywords: List[str], label: int) -> Callable:
    """Create a labeling function from a keyword list."""
    def lf(x):
        text_lower = x.text.lower()
        if any(kw in text_lower for kw in keywords):
            return label
        return ABSTAIN
    lf.__name__ = f"keyword_{'_'.join(keywords[:2])}"
    return lf


# =====================================================================
# Part 3 – Label Aggregation
# =====================================================================
print("=" * 60)
print("Part 3: Label Aggregation — Majority Vote vs Label Model")
print("=" * 60)

# Given a label matrix L (n_samples × n_lfs), we need to aggregate
# the noisy, overlapping, potentially conflicting labels into
# a single probabilistic label per sample.


def apply_lfs(data_points: List[DataPoint], lfs: List[Callable]) -> np.ndarray:
    """Apply all labeling functions to all data points.

    Returns:
        Label matrix L of shape (n_samples, n_lfs)
        where L[i,j] is the label from LF j on sample i.
    """
    n = len(data_points)
    m = len(lfs)
    L = np.full((n, m), ABSTAIN, dtype=int)
    for i, dp in enumerate(data_points):
        for j, lf in enumerate(lfs):
            L[i, j] = lf(dp)
    return L


def majority_vote(L: np.ndarray, tie_break: int = ABSTAIN) -> np.ndarray:
    """Aggregate labels by majority vote.

    For each sample, count votes (excluding ABSTAIN) and pick
    the most common label.  Ties go to tie_break.
    """
    n = L.shape[0]
    preds = np.full(n, tie_break, dtype=int)
    for i in range(n):
        votes = L[i][L[i] != ABSTAIN]
        if len(votes) == 0:
            continue
        counts = Counter(votes)
        most_common = counts.most_common()
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            preds[i] = most_common[0][0]
    return preds


# Snorkel's Label Model (probabilistic)
print("""
  Majority Vote: simple but ignores LF accuracy differences.

  Label Model (Snorkel): learns the accuracy and correlation of each
  LF and produces probabilistic labels.

    from snorkel.labeling.model import LabelModel

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, seed=42)

    # Get probabilistic labels
    probs = label_model.predict_proba(L=L_train)
    # → shape: (n_samples, n_classes) with values in [0, 1]

    # Get hard labels
    preds = label_model.predict(L=L_train)
""")

# Demo with our sample data
L = apply_lfs(sample_comments, lfs)
preds = majority_vote(L, tie_break=NEGATIVE)

print("  Majority vote predictions:")
for dp, pred in zip(sample_comments, preds):
    label = "SPAM" if pred == POSITIVE else "HAM" if pred == NEGATIVE else "ABSTAIN"
    n_votes = (L[sample_comments.index(dp)] != ABSTAIN).sum()
    print(f"    {label:>7} ({n_votes} LFs voted): {dp.text[:60]}")
print()


# =====================================================================
# Part 4 – LF Analysis: Coverage, Overlap, Conflict
# =====================================================================
print("=" * 60)
print("Part 4: LF Analysis Metrics")
print("=" * 60)


def analyze_lfs(L: np.ndarray, lf_names: List[str]) -> None:
    """Compute and display LF quality metrics.

    Coverage:  fraction of samples where LF does not abstain
    Overlap:   fraction of samples where LF agrees with ≥1 other LF
    Conflict:  fraction of samples where LF disagrees with another LF
    """
    n, m = L.shape

    print(f"  {'LF Name':<25} {'Coverage':>10} {'Overlaps':>10} {'Conflicts':>10}")
    print("  " + "-" * 55)

    for j in range(m):
        # Coverage
        labeled = L[:, j] != ABSTAIN
        coverage = labeled.mean()

        # Overlap: this LF and at least one other both label the sample
        overlaps = 0
        conflicts = 0
        for i in range(n):
            if L[i, j] == ABSTAIN:
                continue
            other_labels = L[i, np.arange(m) != j]
            other_active = other_labels[other_labels != ABSTAIN]
            if len(other_active) > 0:
                overlaps += 1
                if any(other_active != L[i, j]):
                    conflicts += 1

        overlap_rate = overlaps / n
        conflict_rate = conflicts / n

        print(f"  {lf_names[j]:<25} {coverage:>10.1%} {overlap_rate:>10.1%} {conflict_rate:>10.1%}")

    # Overall coverage
    any_label = (L != ABSTAIN).any(axis=1).mean()
    print(f"\n  Overall coverage: {any_label:.1%} of samples have ≥1 LF label")


lf_names = [lf.__name__ for lf in lfs]
analyze_lfs(L, lf_names)
print()

# Snorkel LFAnalysis
print("""
  Snorkel's built-in analysis:
    from snorkel.labeling import LFAnalysis

    analysis = LFAnalysis(L=L_train, lfs=lfs)
    df = analysis.lf_summary()
    # Returns DataFrame with columns:
    #   j, Polarity, Coverage, Overlaps, Conflicts, Correct, Incorrect, Emp. Acc.
""")


# =====================================================================
# Part 5 – End-to-End Pipeline
# =====================================================================
print("=" * 60)
print("Part 5: End-to-End Weak Supervision Pipeline")
print("=" * 60)

# The full pipeline:
#   1. Write labeling functions (heuristics, patterns, models)
#   2. Apply LFs to unlabeled data → label matrix L
#   3. Analyze LFs (coverage, conflicts, accuracy on dev set)
#   4. Train label model to aggregate → probabilistic labels
#   5. Filter out low-confidence samples
#   6. Train end model (e.g., BERT) on probabilistic labels

print("""
  Full Snorkel pipeline:

  # Step 1: Define labeling functions
  lfs = [lf_contains_link, lf_subscribe, lf_check_out,
         lf_my_channel, lf_please, lf_short_comment,
         lf_song_mention, lf_textblob_positive, lf_has_person]

  # Step 2: Apply to unlabeled data
  from snorkel.labeling import PandasLFApplier
  applier = PandasLFApplier(lfs=lfs)
  L_train = applier.apply(df=df_train)  # shape: (n_train, n_lfs)
  L_test = applier.apply(df=df_test)

  # Step 3: Analyze
  from snorkel.labeling import LFAnalysis
  LFAnalysis(L=L_train, lfs=lfs).lf_summary()

  # Step 4: Train label model
  from snorkel.labeling.model import LabelModel
  label_model = LabelModel(cardinality=2, verbose=True)
  label_model.fit(L_train=L_train, n_epochs=500, seed=42)

  # Evaluate label model
  label_model_acc = label_model.score(
      L=L_test, Y=Y_test, tie_break_policy="random"
  )["accuracy"]
  print(f"Label Model Accuracy: {label_model_acc:.1%}")

  # Step 5: Get probabilistic labels and filter
  from snorkel.labeling import filter_unlabeled_dataframe
  from snorkel.utils import probs_to_preds

  probs_train = label_model.predict_proba(L=L_train)
  df_filtered, probs_filtered = filter_unlabeled_dataframe(
      X=df_train, y=probs_train, L=L_train
  )
  preds_filtered = probs_to_preds(probs=probs_filtered)

  # Step 6: Train end classifier
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.linear_model import LogisticRegression

  vectorizer = CountVectorizer(ngram_range=(1, 5))
  X_train = vectorizer.fit_transform(df_filtered.text.tolist())
  X_test = vectorizer.transform(df_test.text.tolist())

  clf = LogisticRegression(C=1e3, solver="liblinear")
  clf.fit(X=X_train, y=preds_filtered)
  print(f"End Model Accuracy: {clf.score(X=X_test, y=Y_test):.1%}")

  # Results from the original notebook (YouTube Spam):
  #   Majority Vote:    89.6%
  #   Label Model:      92.4%
  #   End Model (LR):   93.5%
  # All WITHOUT a single hand-labeled training example!
""")


# Simplified demo
print("  Simplified demo with synthetic data:")
np.random.seed(42)

# Generate synthetic labeled data (pretend we don't have these labels)
n_samples = 500
true_labels = np.random.randint(0, 2, n_samples)
texts = []
for label in true_labels:
    if label == POSITIVE:  # spam
        templates = [
            "Check out my channel http://link.com",
            "Subscribe to my page please",
            "Visit http://spam.com for great deals",
            "Check out my new video plz subscribe",
        ]
    else:  # ham
        templates = [
            "Great song love it",
            "This is amazing",
            "lol",
            "Nice performance really enjoyed this",
        ]
    texts.append(np.random.choice(templates))

data = [DataPoint(text=t) for t in texts]

# Apply LFs
L_demo = apply_lfs(data, lfs)

# Coverage stats
coverage = (L_demo != ABSTAIN).any(axis=1).mean()
avg_lfs = (L_demo != ABSTAIN).sum(axis=1).mean()
print(f"  Coverage: {coverage:.1%} of samples have ≥1 label")
print(f"  Avg LFs per sample: {avg_lfs:.1f}")

# Majority vote accuracy
preds_mv = majority_vote(L_demo, tie_break=NEGATIVE)
# Only evaluate where we have a prediction
labeled_mask = preds_mv != ABSTAIN
mv_acc = (preds_mv[labeled_mask] == true_labels[labeled_mask]).mean()
print(f"  Majority vote accuracy (on labeled): {mv_acc:.3f}")
print(f"  Labeled: {labeled_mask.sum()}/{n_samples} samples")
print()


# =====================================================================
# Part 6 – Financial Weak Supervision Use Cases
# =====================================================================
print("=" * 60)
print("Part 6: Financial Weak Supervision Use Cases")
print("=" * 60)

print("""
  Weak supervision is especially valuable in finance where:
  - Labeled data is scarce (proprietary, expensive to annotate)
  - Domain experts can express heuristics but can't label 10K examples
  - Patterns change over time (need rapid re-labeling)

  Example: Earnings sentiment classification
  ───────────────────────────────────────────
  @labeling_function()
  def lf_beat_estimates(x):
      return BULLISH if re.search(r"beat.*estimates?", x.text, re.I) else ABSTAIN

  @labeling_function()
  def lf_missed_estimates(x):
      return BEARISH if re.search(r"miss(ed)?.*estimates?", x.text, re.I) else ABSTAIN

  @labeling_function()
  def lf_raised_guidance(x):
      return BULLISH if re.search(r"rais(ed|ing).*guidance", x.text, re.I) else ABSTAIN

  @labeling_function()
  def lf_lowered_guidance(x):
      return BEARISH if re.search(r"lower(ed|ing).*guidance", x.text, re.I) else ABSTAIN

  @labeling_function()
  def lf_revenue_growth(x):
      match = re.search(r"revenue.*(grew|increased|up)\s+(\d+)%", x.text, re.I)
      if match and int(match.group(2)) > 10:
          return BULLISH
      return ABSTAIN

  @labeling_function()
  def lf_analyst_upgrade(x):
      return BULLISH if re.search(r"upgrade[sd]?.*to.*buy", x.text, re.I) else ABSTAIN

  @labeling_function()
  def lf_finbert_positive(x):
      # Use pre-trained FinBERT as a weak labeler
      score = finbert_pipeline(x.text)[0]
      if score["label"] == "positive" and score["score"] > 0.8:
          return BULLISH
      return ABSTAIN

  Other financial applications:
  ─────────────────────────────
  1. SEC filing risk classification
     LFs: keyword lists for each risk category, regex for regulatory
     citations, NER for organization mentions

  2. News event detection
     LFs: verb patterns ("acquired", "merged"), entity co-occurrence,
     date proximity, source credibility

  3. Credit risk assessment
     LFs: financial ratio thresholds, industry benchmarks,
     management change signals, debt covenant keywords

  4. ESG classification
     LFs: environmental keywords, governance patterns,
     regulatory compliance mentions, sustainability metrics

  Benefits over manual labeling:
  ┌──────────────────────┬─────────────┬───────────────────┐
  │ Metric               │ Manual      │ Weak Supervision  │
  ├──────────────────────┼─────────────┼───────────────────┤
  │ Labeling time        │ Days-weeks  │ Hours             │
  │ Cost per label       │ $0.50-$5.00 │ ~$0 (code only)   │
  │ Scalability          │ Linear      │ O(1) once written │
  │ Adaptability         │ Re-label    │ Update LFs        │
  │ Consistency          │ Variable    │ Deterministic     │
  │ Domain knowledge     │ Lost        │ Encoded in code   │
  └──────────────────────┴─────────────┴───────────────────┘
""")

print("Done.")


if __name__ == "__main__":
    pass
