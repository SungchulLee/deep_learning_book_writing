"""Lime shap interpretability."""
# ---
# title: "LIME & SHAP for NLP Model Interpretability"
# description: "Local explanations with LIME and global/local explanations
#               with SHAP for text classification models"
# ---
#
# Interpretability is critical for financial NLP: regulators and risk
# managers need to understand WHY a model flags a document as relevant
# or classifies sentiment as bearish.
#
#   Part 1 – LIME for text classification (model-agnostic)
#   Part 2 – SHAP with LinearExplainer (for logistic regression / SVM)
#   Part 3 – SHAP DeepExplainer for PyTorch models
#   Part 4 – Comparison: LIME vs SHAP
#   Part 5 – Financial NLP interpretability use cases
#
# Adapted from: O'Reilly "Practical NLP" Ch 4

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =====================================================================
# Part 1 – LIME for Text Classification
# =====================================================================
print("=" * 60)
print("Part 1: LIME (Local Interpretable Model-Agnostic Explanations)")
print("=" * 60)

# LIME explains individual predictions by perturbing the input
# and observing how the prediction changes.
#
# For text: LIME removes random words, creates many perturbed versions,
# gets predictions for all of them, then fits a local linear model
# to approximate the classifier's behaviour near that instance.
#
# Key insight: the linear model's coefficients show which words
# push the prediction towards each class.

# Create a sample dataset
texts = [
    "Revenue grew 15% beating analyst expectations for the quarter",
    "The company reported strong earnings and raised full year guidance",
    "Profit margins expanded as cost cutting measures took effect",
    "Stock buyback program signals management confidence in growth",
    "Dividend increase reflects strong cash flow generation",
    "Revenue declined sharply missing consensus estimates by wide margin",
    "The company warned of deteriorating demand and lowered guidance",
    "Profit margins compressed due to rising input costs and competition",
    "Management announced restructuring and significant layoffs planned",
    "Debt levels are concerning as interest coverage ratio declined",
] * 20
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 20  # 1=positive, 0=negative

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Train a simple TF-IDF + LogisticRegression pipeline
vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

clf = LogisticRegression(class_weight="balanced", random_state=42)
clf.fit(X_train_tfidf, y_train)
print(f"  Classifier accuracy: {accuracy_score(y_test, clf.predict(X_test_tfidf)):.3f}")

# LIME explanation
print("""
  # Using LIME for text explanation:
  from lime.lime_text import LimeTextExplainer
  from sklearn.pipeline import make_pipeline

  # Create a pipeline (vectorizer + classifier)
  pipe = make_pipeline(vectorizer, clf)

  # Initialize LIME explainer
  class_names = ["negative", "positive"]
  explainer = LimeTextExplainer(class_names=class_names)

  # Explain a single prediction
  text = "Revenue declined sharply missing consensus estimates"
  exp = explainer.explain_instance(
      text,
      pipe.predict_proba,
      num_features=6,       # top 6 most important words
      num_samples=1000,     # number of perturbations
  )

  # View explanation
  print(exp.as_list())
  # → [('declined', -0.42),    # pushes toward negative
  #    ('missing', -0.31),     # pushes toward negative
  #    ('revenue', 0.08),      # slightly positive (ambiguous)
  #    ('consensus', -0.12),   # pushes toward negative
  #    ('sharply', -0.19),     # pushes toward negative
  #    ('estimates', 0.05)]    # neutral

  # Visualize (in notebook)
  exp.as_pyplot_figure()  # horizontal bar chart
  exp.show_in_notebook()  # interactive HTML
""")

# Demonstrate the concept without LIME library
print("  Manual LIME-style explanation (without library):")
print("  " + "-" * 50)

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(vectorizer, clf)

sample_text = "Revenue declined sharply missing consensus estimates"
words = sample_text.split()
base_prob = pipe.predict_proba([sample_text])[0]

print(f"  Original prediction: neg={base_prob[0]:.3f}, pos={base_prob[1]:.3f}")
print(f"  Word importance (leave-one-out):")

importances = []
for i, word in enumerate(words):
    # Remove one word at a time
    perturbed = " ".join(w for j, w in enumerate(words) if j != i)
    perturbed_prob = pipe.predict_proba([perturbed])[0]
    # How much does removing this word change the positive probability?
    delta = base_prob[1] - perturbed_prob[1]
    importances.append((word, delta))
    print(f"    {word:>15}: {delta:+.4f} (removing it changes P(pos) by this much)")

print()


# =====================================================================
# Part 2 – SHAP with LinearExplainer
# =====================================================================
print("=" * 60)
print("Part 2: SHAP — LinearExplainer for Sklearn Models")
print("=" * 60)

# SHAP (SHapley Additive exPlanations) computes Shapley values from
# cooperative game theory. Each feature gets a fair attribution of
# how much it contributes to pushing the prediction away from the
# average prediction.
#
# Advantages over LIME:
#   - Theoretically grounded (Shapley values are unique)
#   - Global AND local explanations
#   - Consistent: if a feature's contribution increases, its SHAP value
#     never decreases

print("""
  import shap

  # For linear models, SHAP has an exact solver
  explainer = shap.LinearExplainer(
      clf,
      X_train_tfidf,
      feature_perturbation="interventional",
  )
  shap_values = explainer.shap_values(X_test_tfidf)

  # Global summary: which features matter most across all predictions?
  shap.summary_plot(
      shap_values,
      X_test_tfidf.toarray(),
      feature_names=vectorizer.get_feature_names_out(),
  )
  # → Shows each feature's distribution of SHAP values across test set
  # Red dots = high feature value, Blue dots = low feature value
  # x-axis = SHAP value (impact on model output)

  # Local explanation: why was this particular instance classified this way?
  shap.force_plot(
      explainer.expected_value,
      shap_values[0, :],
      X_test_tfidf[0, :].toarray(),
      feature_names=vectorizer.get_feature_names_out(),
  )
""")

# Manual SHAP-like analysis using model coefficients
print("  Top features by logistic regression coefficient (SHAP-adjacent):")
feature_names = vectorizer.get_feature_names_out()
coefs = clf.coef_[0]
top_positive = np.argsort(coefs)[-5:][::-1]
top_negative = np.argsort(coefs)[:5]

print("  Most positive (bullish indicators):")
for idx in top_positive:
    print(f"    {feature_names[idx]:>20}: {coefs[idx]:+.4f}")

print("  Most negative (bearish indicators):")
for idx in top_negative:
    print(f"    {feature_names[idx]:>20}: {coefs[idx]:+.4f}")
print()


# =====================================================================
# Part 3 – SHAP DeepExplainer for PyTorch Models
# =====================================================================
print("=" * 60)
print("Part 3: SHAP DeepExplainer for PyTorch Models")
print("=" * 60)

# For deep learning models, SHAP uses DeepExplainer (based on DeepLIFT)
# or GradientExplainer (based on integrated gradients).


class TextLSTM(nn.Module):
    """Simple LSTM for text classification."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        return self.fc(h_n.squeeze(0))


# Create a simple vocabulary and encode texts
all_words = set()
for t in texts:
    all_words.update(t.lower().split())
word2idx = {w: i + 1 for i, w in enumerate(sorted(all_words))}

MAX_LEN = 15


def encode_texts(text_list, max_len=MAX_LEN):
    encoded = []
    for t in text_list:
        ids = [word2idx.get(w.lower(), 0) for w in t.split()[:max_len]]
        ids += [0] * (max_len - len(ids))
        encoded.append(ids)
    return torch.tensor(encoded, dtype=torch.long)


X_encoded = encode_texts(texts)
y_tensor = torch.tensor(labels, dtype=torch.long)

# Train LSTM
model = TextLSTM(len(word2idx) + 1, embed_dim=32, hidden_dim=32, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(20):
    logits = model(X_encoded)
    loss = F.cross_entropy(logits, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    preds = model(X_encoded).argmax(dim=-1)
    acc = (preds == y_tensor).float().mean()
print(f"  LSTM accuracy: {acc:.3f}")

print("""
  # SHAP DeepExplainer with PyTorch LSTM:
  import shap

  # Use a subset of training data as background
  background = X_encoded[:20]

  explainer = shap.DeepExplainer(model, background)

  # Explain test predictions
  test_samples = X_encoded[180:185]
  shap_values = explainer.shap_values(test_samples)

  # shap_values is a list (one per class)
  # shap_values[1] = attributions toward the positive class
  # Shape: (num_samples, seq_len)

  # Map back to words for visualization
  idx2word = {v: k for k, v in word2idx.items()}
  sample_words = [
      [idx2word.get(idx.item(), "PAD") for idx in sample]
      for sample in test_samples
  ]

  shap.initjs()
  shap.force_plot(
      explainer.expected_value[1],
      shap_values[1][0],
      sample_words[0],
  )
""")
print()


# =====================================================================
# Part 4 – Comparison: LIME vs SHAP
# =====================================================================
print("=" * 60)
print("Part 4: LIME vs SHAP Comparison")
print("=" * 60)

print("""
  ┌─────────────────┬──────────────────────┬──────────────────────┐
  │ Aspect          │ LIME                 │ SHAP                 │
  ├─────────────────┼──────────────────────┼──────────────────────┤
  │ Theory          │ Local linear approx  │ Shapley values       │
  │ Scope           │ Local only           │ Local + Global       │
  │ Consistency     │ No guarantees        │ Mathematically proven│
  │ Speed           │ Faster (sampling)    │ Slower (exact)       │
  │ Model support   │ Any black-box        │ Optimized explainers │
  │ Stability       │ Varies (random)      │ Deterministic        │
  │ Additivity      │ No                   │ Yes (values sum up)  │
  │ Best for        │ Quick local insights │ Rigorous attribution │
  └─────────────────┴──────────────────────┴──────────────────────┘

  When to use which:
  - LIME: Quick debugging, model-agnostic, any classifier
  - SHAP (Linear): Exact values for linear models (fast)
  - SHAP (Deep):   Neural network attribution
  - SHAP (Kernel): Any model, most expensive but flexible
""")


# =====================================================================
# Part 5 – Financial NLP Interpretability Use Cases
# =====================================================================
print("=" * 60)
print("Part 5: Financial NLP Interpretability Use Cases")
print("=" * 60)

print("""
  1. Regulatory compliance:
     - Show which words triggered a "high risk" classification
     - Audit trail for automated document screening

  2. Sentiment-driven trading:
     - Understand WHY a news article was classified as bearish
     - Distinguish genuine sentiment from noise words

  3. Credit risk assessment:
     - Which phrases in an annual report raised red flags?
     - Feature attribution for creditworthiness scoring

  4. Anomaly detection:
     - Explain why a filing was flagged as unusual
     - Identify key phrases driving anomaly scores

  5. Model debugging:
     - Find spurious correlations (e.g., company name → sentiment)
     - Detect data leakage through feature importance

  Example: Explaining a FinBERT prediction

    text = "Tesla missed delivery targets amid supply chain disruptions"
    # FinBERT predicts: negative (0.89)
    # LIME explanation:
    #   "missed"       → -0.35 (strong negative signal)
    #   "disruptions"  → -0.22 (negative context)
    #   "delivery"     → -0.08 (slightly negative in this context)
    #   "targets"      → -0.05 (associated with "missing")
    #   "Tesla"        →  0.02 (neutral — good! no brand bias)
""")

print("Done.")


if __name__ == "__main__":
    pass
