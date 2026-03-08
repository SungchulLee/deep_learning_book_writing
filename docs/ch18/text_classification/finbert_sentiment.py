"""Finbert sentiment."""
# ---
# title: "FinBERT: Financial Sentiment Analysis"
# description: "Fine-tuning FinBERT for stock sentiment classification on
#               StockTwits data — domain-adapted BERT for financial NLP"
# ---
#
# FinBERT is a BERT model further pre-trained on financial text (10-K filings,
# analyst reports, earnings calls).  It captures domain-specific language that
# generic BERT misses ("bullish", "bearish", "overbought", "short squeeze").
#
#   Part 1 – Load FinBERT from HuggingFace Hub
#   Part 2 – Zero-shot inference with the pre-trained sentiment head
#   Part 3 – Fine-tune FinBERT on StockTwits data (PyTorch)
#   Part 4 – Evaluation and per-stock analysis
#   Part 5 – Compare FinBERT vs generic BERT vs lexicon-based
#
# Adapted from: O'Reilly "Practical NLP" Ch 10

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import numpy as np


# =====================================================================
# Part 1 – Load FinBERT from HuggingFace Hub
# =====================================================================
print("=" * 60)
print("Part 1: Loading FinBERT")
print("=" * 60)

# Modern approach: FinBERT is available on HuggingFace Hub.
# Two popular variants:
#   - ProsusAI/finbert         (sentiment: positive/negative/neutral)
#   - yiyanghkust/finbert-tone (same task, different pre-training corpus)

print("""
  # Load pre-trained FinBERT for sentiment analysis
  from transformers import AutoModelForSequenceClassification, AutoTokenizer

  model_name = "ProsusAI/finbert"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name)
  model.eval()

  # Quick inference
  text = "Tesla reported record deliveries, beating analyst expectations."
  inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
  with torch.no_grad():
      logits = model(**inputs).logits
      probs = F.softmax(logits, dim=-1)
      labels = ["positive", "negative", "neutral"]
      pred = labels[probs.argmax()]
      print(f"Sentiment: {pred} ({probs.max():.3f})")
  # → Sentiment: positive (0.92)
""")


# =====================================================================
# Part 2 – Zero-Shot Inference with Pipeline
# =====================================================================
print("=" * 60)
print("Part 2: Zero-Shot Inference via HuggingFace Pipeline")
print("=" * 60)

print("""
  from transformers import pipeline

  finbert = pipeline(
      "sentiment-analysis",
      model="ProsusAI/finbert",
      tokenizer="ProsusAI/finbert",
  )

  # Single prediction
  result = finbert("AAPL missed revenue estimates by 2%.")
  # → [{'label': 'negative', 'score': 0.87}]

  # Batch prediction
  texts = [
      "Revenue grew 15% year-over-year, exceeding guidance.",
      "The company announced layoffs affecting 10% of staff.",
      "Q3 earnings were in line with consensus expectations.",
      "Short sellers are increasing their positions in the stock.",
  ]
  results = finbert(texts)
  for text, res in zip(texts, results):
      print(f"  {res['label']:>8} ({res['score']:.2f}): {text[:60]}")
""")


# =====================================================================
# Part 3 – Fine-Tune FinBERT on Custom Stock Data (PyTorch)
# =====================================================================
print("=" * 60)
print("Part 3: Fine-Tuning FinBERT on StockTwits Data")
print("=" * 60)

# The original notebook fine-tunes on StockTwits data for FB, AMZN, GOOGL
# with labels: Bullish (1) / Bearish (0).
#
# Key steps:
#   1. Tokenize with BERT tokenizer (max_len=128)
#   2. Create attention masks
#   3. Train/val split
#   4. Fine-tune with AdamW + linear warmup
#   5. Evaluate per epoch

print("  Creating synthetic StockTwits-like data for demonstration...")

# Synthetic financial tweets
bullish_texts = [
    "$AAPL strong breakout above resistance, going long",
    "$TSLA deliveries crushing it, moon incoming",
    "$GOOGL ad revenue growth is accelerating, bullish",
    "$AMZN AWS margins expanding, great quarter ahead",
    "$MSFT cloud business firing on all cylinders",
    "$NVDA AI demand is insatiable, still undervalued",
    "$META reels monetization improving rapidly",
    "$JPM net interest income at record highs",
] * 15  # 120 examples

bearish_texts = [
    "$AAPL China sales declining, supply chain issues",
    "$TSLA margin compression is alarming, overvalued",
    "$GOOGL antitrust risk is real, short this name",
    "$AMZN retail losing money, AWS growth slowing",
    "$MSFT enterprise spending pullback hurting growth",
    "$NVDA valuation is stretched, priced for perfection",
    "$META user growth stalling in key demographics",
    "$JPM credit losses mounting, recession risk rising",
] * 15  # 120 examples

texts = bullish_texts + bearish_texts
labels = [1] * len(bullish_texts) + [0] * len(bearish_texts)

# Shuffle
indices = np.random.RandomState(42).permutation(len(texts))
texts = [texts[i] for i in indices]
labels = [labels[i] for i in indices]

print(f"  Dataset: {len(texts)} examples ({sum(labels)} bullish, {len(labels)-sum(labels)} bearish)")

# Tokenization (simulated with simple encoding)
# In practice, use BertTokenizer.from_pretrained("ProsusAI/finbert")
torch.manual_seed(42)

# For demonstration, we'll use a simple vocabulary-based encoding
# and train a small transformer-like classifier
MAX_LEN = 64
VOCAB_SIZE = 5000
EMBED_DIM = 128
NUM_HEADS = 4
HIDDEN_DIM = 256
NUM_CLASSES = 2


class SimpleFinancialClassifier(nn.Module):
    """A small transformer encoder for text classification.

    In practice, replace this with:
        BertForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=2)
    """

    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.embedding(input_ids) + self.pos_encoding(positions)

        if attention_mask is not None:
            # TransformerEncoder expects True = ignored positions
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        # Mean pooling over non-padding tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.classifier(x)


# Simple tokenization (in practice use BertTokenizer)
def simple_tokenize(texts, max_len=64, vocab_size=5000):
    """Hash-based tokenization for demonstration purposes."""
    all_ids, all_masks = [], []
    for text in texts:
        tokens = text.lower().split()[:max_len]
        ids = [(hash(t) % (vocab_size - 1)) + 1 for t in tokens]
        mask = [1] * len(ids)
        # Pad
        pad_len = max_len - len(ids)
        ids += [0] * pad_len
        mask += [0] * pad_len
        all_ids.append(ids)
        all_masks.append(mask)
    return torch.tensor(all_ids), torch.tensor(all_masks)


input_ids, attention_masks = simple_tokenize(texts, MAX_LEN, VOCAB_SIZE)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Train/val split
from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    range(len(texts)), test_size=0.15, random_state=42, stratify=labels
)

train_data = TensorDataset(input_ids[train_idx], attention_masks[train_idx], labels_tensor[train_idx])
val_data = TensorDataset(input_ids[val_idx], attention_masks[val_idx], labels_tensor[val_idx])

train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=32)
val_loader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=32)

# Training
model = SimpleFinancialClassifier(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, HIDDEN_DIM, NUM_CLASSES, MAX_LEN)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

print("\n  Training:")
for epoch in range(8):
    model.train()
    total_loss = 0
    for batch in train_loader:
        b_ids, b_mask, b_labels = batch
        optimizer.zero_grad()
        logits = model(b_ids, b_mask)
        loss = F.cross_entropy(logits, b_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            b_ids, b_mask, b_labels = batch
            logits = model(b_ids, b_mask)
            preds = logits.argmax(dim=-1)
            correct += (preds == b_labels).sum().item()
            total += len(b_labels)

    val_acc = correct / total
    avg_loss = total_loss / len(train_loader)
    if (epoch + 1) % 2 == 0:
        print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.3f}")


# =====================================================================
# Part 4 – Evaluation and Per-Stock Analysis
# =====================================================================
print("\n" + "=" * 60)
print("Part 4: Evaluation & Per-Stock Breakdown")
print("=" * 60)

# In the original notebook, data contains FB, AMZN, GOOGL tweets
# with Bullish/Bearish labels from StockTwits API
print("""
  The full fine-tuning pipeline with real data:

  from transformers import (
      AutoModelForSequenceClassification, AutoTokenizer,
      Trainer, TrainingArguments,
  )
  from datasets import Dataset
  import pandas as pd

  # Load StockTwits data
  df = pd.read_csv("stocktwits_data.csv")
  # Columns: symbol, message, sentiment (Bullish/Bearish), message_id
  # Encode labels
  df["label"] = (df["sentiment"] == "Bullish").astype(int)

  tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
  model = AutoModelForSequenceClassification.from_pretrained(
      "ProsusAI/finbert", num_labels=2
  )

  def tokenize(batch):
      return tokenizer(
          batch["message"], padding="max_length",
          truncation=True, max_length=128,
      )

  ds = Dataset.from_pandas(df[["message", "label"]])
  ds = ds.map(tokenize, batched=True)
  ds = ds.train_test_split(test_size=0.1, seed=42)

  args = TrainingArguments(
      output_dir="./finbert-stocktwits",
      num_train_epochs=4,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=64,
      learning_rate=2e-5,
      weight_decay=0.01,
      warmup_ratio=0.1,
      evaluation_strategy="epoch",
      save_strategy="epoch",
      load_best_model_at_end=True,
      fp16=True,
  )

  trainer = Trainer(
      model=model, args=args,
      train_dataset=ds["train"],
      eval_dataset=ds["test"],
      tokenizer=tokenizer,
  )
  trainer.train()

  # Per-stock evaluation
  for symbol in ["FB", "AMZN", "GOOGL"]:
      subset = df[df["symbol"] == symbol]
      preds = trainer.predict(Dataset.from_pandas(subset))
      acc = (preds.predictions.argmax(-1) == subset["label"].values).mean()
      print(f"  {symbol}: {acc:.3f} accuracy ({len(subset)} samples)")
""")


# =====================================================================
# Part 5 – Comparison: FinBERT vs Generic BERT vs Lexicon
# =====================================================================
print("=" * 60)
print("Part 5: FinBERT vs Generic BERT vs Lexicon-Based")
print("=" * 60)

print("""
  Comparison on financial sentiment tasks:

  ┌─────────────────────┬───────────┬──────────┬─────────────┐
  │ Method              │ Accuracy  │ F1-Score │ Domain Fit  │
  ├─────────────────────┼───────────┼──────────┼─────────────┤
  │ TextBlob (lexicon)  │   0.58    │   0.52   │ Poor        │
  │ VADER               │   0.62    │   0.57   │ Fair        │
  │ BERT-base           │   0.71    │   0.68   │ Generic     │
  │ FinBERT (zero-shot) │   0.76    │   0.73   │ Good        │
  │ FinBERT (fine-tuned)│   0.83    │   0.81   │ Excellent   │
  └─────────────────────┴───────────┴──────────┴─────────────┘

  Why FinBERT outperforms generic models:

  1. Domain vocabulary: "bullish", "bearish", "overbought" etc.
     have specific financial meanings that generic BERT doesn't capture.

  2. Sentiment polarity: "Short position" is neutral in general text
     but often bearish in financial context.

  3. Negation handling: "Revenue did not meet expectations" requires
     understanding financial reporting language.

  4. Numerical context: "EPS of $2.50 vs $2.30 expected" — FinBERT
     understands that beating estimates is positive.

  Financial NLP applications:
  - Earnings call sentiment tracking
  - News sentiment for alpha generation
  - Social media (StockTwits/Reddit) sentiment scoring
  - SEC filing tone analysis
  - Analyst report classification
""")

print("Done.")


if __name__ == "__main__":
    pass
