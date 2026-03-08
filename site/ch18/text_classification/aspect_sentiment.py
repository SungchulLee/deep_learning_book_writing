"""Aspect sentiment."""
# ---
# title: "Aspect-Based Sentiment Analysis"
# description: "Fine-grained sentiment at entity/aspect level using VADER,
#               PyTorch attention-based models, and transformer approaches"
# ---
#
# Standard sentiment analysis gives one label per document.
# Aspect-Based Sentiment Analysis (ABSA) extracts sentiment for
# SPECIFIC aspects or entities within a text.
#
# Example: "The phone has an amazing camera but terrible battery life."
#   → camera: positive
#   → battery life: negative
#
#   Part 1 – Rule-based ABSA with VADER
#   Part 2 – Aspect extraction (noun phrase chunking)
#   Part 3 – Attention-based ABSA model (PyTorch)
#   Part 4 – Transformer ABSA with auxiliary sentence
#   Part 5 – Financial ABSA use cases
#
# Adapted from: O'Reilly "Practical NLP" Ch 9

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
import re


# =====================================================================
# Part 1 – Rule-Based ABSA with VADER
# =====================================================================
print("=" * 60)
print("Part 1: Rule-Based ABSA with VADER")
print("=" * 60)

# VADER (Valence Aware Dictionary and sEntiment Reasoner) is a
# lexicon-based sentiment tool tuned for social media text.
# It handles: capitalization, punctuation emphasis, degree modifiers,
# negation, and conjunctions.
#
# For ABSA, we:
#   1. Split text into clauses (by punctuation / conjunctions)
#   2. Identify aspect terms in each clause
#   3. Score each clause with VADER
#   4. Assign the clause's sentiment to its aspect

# Simplified VADER-like scoring (real VADER uses ~7500 lexicon entries)
SENTIMENT_LEXICON = {
    # Positive words
    "amazing": 3.0, "excellent": 3.0, "great": 2.5, "good": 2.0,
    "strong": 2.0, "impressive": 2.5, "solid": 1.5, "beat": 2.0,
    "exceeded": 2.5, "growth": 1.5, "profit": 1.5, "surged": 2.5,
    "bullish": 2.0, "outperform": 2.0, "upgrade": 2.0, "record": 2.0,
    # Negative words
    "terrible": -3.0, "poor": -2.0, "bad": -2.0, "weak": -2.0,
    "disappointing": -2.5, "declined": -2.0, "missed": -2.5,
    "bearish": -2.0, "downgrade": -2.5, "loss": -2.0, "risk": -1.5,
    "concern": -1.5, "slowed": -1.5, "underperform": -2.0, "cut": -1.5,
    # Modifiers
    "very": 1.5, "extremely": 2.0, "slightly": 0.5, "somewhat": 0.7,
}

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "hardly", "barely"}
CONJUNCTION_SPLIT = re.compile(r'\b(but|however|although|though|yet|while)\b', re.IGNORECASE)


def vader_score_clause(clause: str) -> float:
    """Score a text clause using a simplified VADER approach."""
    words = clause.lower().split()
    score = 0.0
    negate = False
    modifier = 1.0

    for word in words:
        # Check for negation
        if word in NEGATION_WORDS:
            negate = True
            continue

        # Check for degree modifier
        if word in SENTIMENT_LEXICON and abs(SENTIMENT_LEXICON[word]) < 2.0 and word in {"very", "extremely", "slightly", "somewhat"}:
            modifier = SENTIMENT_LEXICON[word]
            continue

        # Score sentiment word
        if word in SENTIMENT_LEXICON:
            word_score = SENTIMENT_LEXICON[word] * modifier
            if negate:
                word_score *= -0.75  # Negation dampens but doesn't fully flip
                negate = False
            score += word_score
            modifier = 1.0  # Reset modifier after use

    # Normalize to [-1, 1]
    return np.tanh(score / 3.0)


def extract_aspects_and_sentiment(text: str) -> List[Dict]:
    """Extract aspects and their sentiment from text.

    Strategy:
    1. Split on conjunctions (but, however, etc.)
    2. Within each clause, find capitalized/quoted terms as aspects
    3. Score each clause
    """
    # Split on conjunctions
    clauses = CONJUNCTION_SPLIT.split(text)
    clauses = [c.strip() for c in clauses if c.strip() and c.lower() not in
               {"but", "however", "although", "though", "yet", "while"}]

    results = []
    for clause in clauses:
        score = vader_score_clause(clause)
        # Extract aspect candidates (simple heuristic: look for nouns/entities)
        # In practice, use NLP tools for noun phrase extraction
        words = clause.split()
        aspect_candidates = []
        for w in words:
            if w[0] == '$' or (w[0].isupper() and w.lower() not in SENTIMENT_LEXICON
                               and w.lower() not in NEGATION_WORDS
                               and len(w) > 2):
                aspect_candidates.append(w)

        label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
        results.append({
            "clause": clause,
            "aspects": aspect_candidates,
            "score": score,
            "label": label,
        })

    return results


# Demo
texts = [
    "$AAPL reported amazing revenue growth but iPhone margins were disappointing",
    "Tesla has impressive delivery numbers however valuation risk remains a concern",
    "Strong cloud revenue from AWS but advertising growth slowed significantly",
]

for text in texts:
    print(f"\n  Text: {text}")
    aspects = extract_aspects_and_sentiment(text)
    for a in aspects:
        asp_str = ", ".join(a["aspects"]) if a["aspects"] else "(general)"
        print(f"    {a['label']:>8} ({a['score']:+.2f}): [{asp_str}] — {a['clause'][:50]}")

print()

# Full VADER usage
print("  Using the vaderSentiment library:")
print("""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()

    text = "The product is absolutely amazing but the price is too high"
    scores = analyzer.polarity_scores(text)
    # → {'neg': 0.15, 'neu': 0.52, 'pos': 0.33, 'compound': 0.34}

    # Per-aspect analysis: split on "but" and score each part
    parts = text.split(" but ")
    for part in parts:
        s = analyzer.polarity_scores(part)
        print(f"  '{part}' → compound: {s['compound']:.3f}")
    # → 'The product is absolutely amazing' → compound: 0.62
    # → 'the price is too high' → compound: -0.19
""")


# =====================================================================
# Part 2 – Aspect Term Extraction
# =====================================================================
print("=" * 60)
print("Part 2: Aspect Term Extraction")
print("=" * 60)

# Aspect extraction can be done via:
# 1. Dependency parsing (nsubj, dobj relations)
# 2. Frequency-based (most common nouns in domain)
# 3. Attention weights from trained models
# 4. Sequence labeling (BIO tags for aspects)

print("""
  Approach 1: spaCy dependency parsing
  ─────────────────────────────────────
    import spacy
    nlp = spacy.load("en_core_web_sm")

    text = "The camera quality is excellent but battery drains quickly"
    doc = nlp(text)

    aspects = []
    for token in doc:
        # Nouns that are subjects or objects → likely aspects
        if token.dep_ in ("nsubj", "dobj", "attr") and token.pos_ == "NOUN":
            aspects.append(token.text)
    # → ["quality", "battery"]

  Approach 2: Sequence labeling (BIO tags)
  ─────────────────────────────────────────
    # Train a token classifier on annotated ABSA data
    # Labels: B-ASP, I-ASP, O
    # "The [camera quality]_ASP is excellent"
    # → O  B-ASP  I-ASP  O  O

    from transformers import AutoModelForTokenClassification
    model = AutoModelForTokenClassification.from_pretrained(
        "yangheng/deberta-v3-base-absa-v1.1"
    )
""")


# =====================================================================
# Part 3 – Attention-Based ABSA (PyTorch)
# =====================================================================
print("=" * 60)
print("Part 3: Attention-Based ABSA Model (PyTorch)")
print("=" * 60)

# The key idea: given a sentence and a target aspect, use attention
# to focus on words relevant to that aspect.
#
# Architecture:
#   [word embeddings] → BiLSTM → attention(aspect_emb) → classify


class AspectAttentionClassifier(nn.Module):
    """Attention-based ABSA model.

    Uses aspect embedding as query to attend over context words,
    then classifies the attended representation.

    Args:
        vocab_size:  Vocabulary size
        embed_dim:   Embedding dimension
        hidden_dim:  LSTM hidden dimension
        num_classes: Sentiment classes (positive/negative/neutral)
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        # Attention: project aspect and context to same space
        self.attn_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, context_ids, aspect_ids, context_mask=None):
        """
        Args:
            context_ids: Full sentence token IDs [B, L]
            aspect_ids:  Aspect term token IDs [B, A]
            context_mask: Padding mask [B, L]

        Returns:
            logits: Sentiment logits [B, num_classes]
        """
        # Encode context
        ctx_emb = self.embedding(context_ids)       # [B, L, E]
        ctx_out, _ = self.lstm(ctx_emb)             # [B, L, 2H]

        # Encode aspect (average embedding)
        asp_emb = self.embedding(aspect_ids)         # [B, A, E]
        asp_repr = asp_emb.mean(dim=1, keepdim=True) # [B, 1, E]

        # Aspect-aware attention
        # Project context and compute attention scores
        ctx_proj = self.attn_proj(ctx_out)           # [B, L, 2H]
        # Use aspect as query via dot-product with projected context
        # Expand aspect to hidden dim via embedding → use as is
        attn_scores = torch.bmm(ctx_proj, asp_repr.expand(-1, -1, ctx_proj.size(-1)).transpose(1, 2))
        attn_scores = attn_scores.squeeze(-1)        # [B, L]

        if context_mask is not None:
            attn_scores = attn_scores.masked_fill(context_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, L]

        # Weighted sum of context
        weighted = torch.bmm(attn_weights.unsqueeze(1), ctx_out).squeeze(1)  # [B, 2H]

        return self.classifier(weighted)


# Demo with synthetic data
torch.manual_seed(42)
VOCAB = 200
model = AspectAttentionClassifier(VOCAB, embed_dim=32, hidden_dim=32, num_classes=3)

# Synthetic input
ctx = torch.randint(1, VOCAB, (4, 20))
asp = torch.randint(1, VOCAB, (4, 3))
mask = torch.ones(4, 20)

logits = model(ctx, asp, mask)
print(f"  Model output shape: {logits.shape}")
print(f"  Predictions: {F.softmax(logits, dim=-1).detach().numpy().round(3)}")
print(f"  Labels: negative=0, neutral=1, positive=2")
print()


# =====================================================================
# Part 4 – Transformer ABSA with Auxiliary Sentence
# =====================================================================
print("=" * 60)
print("Part 4: Transformer ABSA (Auxiliary Sentence Approach)")
print("=" * 60)

# Modern approach: convert ABSA to sentence-pair classification.
# Input: "[CLS] sentence [SEP] aspect term [SEP]"
# This lets BERT attend to the aspect through cross-attention.

print("""
  from transformers import AutoModelForSequenceClassification, AutoTokenizer

  model_name = "yangheng/deberta-v3-base-absa-v1.1"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name)

  # Construct input as sentence pair
  sentence = "The restaurant has great food but terrible service"
  aspect = "food"

  inputs = tokenizer(
      sentence, aspect,        # sentence pair!
      return_tensors="pt",
      truncation=True,
      max_length=128,
  )

  with torch.no_grad():
      logits = model(**inputs).logits
      # Labels: 0=negative, 1=neutral, 2=positive
      pred = logits.argmax(-1).item()
      labels = ["negative", "neutral", "positive"]
      print(f"Aspect '{aspect}': {labels[pred]}")
  # → Aspect 'food': positive

  # Now check "service"
  inputs = tokenizer(sentence, "service", return_tensors="pt")
  with torch.no_grad():
      pred = model(**inputs).logits.argmax(-1).item()
      print(f"Aspect 'service': {labels[pred]}")
  # → Aspect 'service': negative
""")


# =====================================================================
# Part 5 – Financial ABSA Use Cases
# =====================================================================
print("=" * 60)
print("Part 5: Financial ABSA Use Cases")
print("=" * 60)

print("""
  Financial texts often discuss multiple aspects in one sentence:

  "Revenue grew 15% driven by cloud services,
   but hardware margins compressed due to supply constraints."
  → cloud services: positive
  → hardware margins: negative

  "Management is bullish on AI initiatives but cautious about
   the regulatory environment in Europe."
  → AI initiatives: positive
  → regulatory environment: negative

  Key financial aspects to track:
  ┌──────────────────┬────────────────────────────┐
  │ Category         │ Example Aspects            │
  ├──────────────────┼────────────────────────────┤
  │ Revenue          │ top-line, sales, bookings  │
  │ Profitability    │ margins, EPS, EBITDA       │
  │ Growth           │ YoY growth, guidance, TAM  │
  │ Balance sheet    │ debt, cash, leverage       │
  │ Operations       │ efficiency, headcount      │
  │ Market position  │ market share, competition  │
  │ Risk factors     │ regulation, litigation     │
  │ Management       │ leadership, strategy       │
  └──────────────────┴────────────────────────────┘

  Trading signal from ABSA:
    For each earnings call / news article:
    1. Extract aspect terms (NER + noun phrases)
    2. Score sentiment per aspect
    3. Weight by aspect importance for the stock
    4. Aggregate: signal = Σ (aspect_weight × aspect_sentiment)

    Aspect weights vary by sector:
    - Tech: growth (0.3), margins (0.2), cloud (0.2), AI (0.2)
    - Banks: NII (0.3), credit quality (0.3), fees (0.2)
    - Retail: same-store sales (0.3), margins (0.2), inventory (0.2)
""")

print("Done.")


if __name__ == "__main__":
    pass
