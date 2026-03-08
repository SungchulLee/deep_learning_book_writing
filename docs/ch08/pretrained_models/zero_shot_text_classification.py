"""Zero shot text classification."""
# ---
# title: "Zero-Shot & Few-Shot Text Classification"
# description: "NLI-based zero-shot classification, embedding lookup with FAISS,
#               data augmentation for low-resource NLP, and domain-adapted fine-tuning"
# ---
#
# When labeled data is scarce or unavailable, several techniques can still yield
# strong text classifiers.  This script demonstrates the progression:
#
#   Part 1 – Zero-shot classification via NLI (Natural Language Inference)
#   Part 2 – Masked LM prompting for zero-shot label scoring
#   Part 3 – Embedding-based few-shot with nearest neighbour lookup
#   Part 4 – Data augmentation for low-resource classification
#   Part 5 – Domain-adapted MLM → fine-tuning pipeline
#   Part 6 – Comparison of all approaches
#
# Adapted from: O'Reilly "NLP with Transformers" Ch 9

import numpy as np
import torch
from typing import List, Dict


# =====================================================================
# Part 1 – Zero-Shot Classification with NLI
# =====================================================================
print("=" * 60)
print("Part 1: Zero-Shot Classification via NLI")
print("=" * 60)

# The key insight: a model trained on Natural Language Inference (NLI)
# can classify text without any task-specific training.  NLI models
# predict whether a "hypothesis" is entailed by a "premise".
#
# For zero-shot classification:
#   premise    = the text to classify
#   hypothesis = "This text is about {label}."
# The entailment score becomes the classification score.

try:
    from transformers import pipeline

    # HuggingFace provides this as a built-in pipeline
    zero_shot_clf = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",   # NLI-trained model
    )

    # Single-label example
    text = "The stock market rallied today after the Federal Reserve announced \
    it would hold interest rates steady for the remainder of the quarter."

    candidate_labels = [
        "finance", "politics", "sports", "technology", "science"
    ]

    result = zero_shot_clf(text, candidate_labels)
    print("  Single-label zero-shot:")
    print(f"  Text: {text[:80]}...")
    for label, score in zip(result["labels"], result["scores"]):
        print(f"    {label:12s}: {score:.3f}")
    print()

    # Multi-label example — a single text can belong to multiple categories
    result_ml = zero_shot_clf(text, candidate_labels, multi_label=True)
    print("  Multi-label zero-shot:")
    for label, score in zip(result_ml["labels"], result_ml["scores"]):
        print(f"    {label:12s}: {score:.3f}")
    print()

except (ImportError, OSError) as e:
    print(f"  Skipping pipeline demo: {e}")
    print()


# =====================================================================
# How zero-shot NLI works under the hood
# =====================================================================
print("  --- Under the hood ---")
print("  For each candidate label, the model evaluates:")
print('    premise:    "<your text>"')
print('    hypothesis: "This example is about <label>."')
print("  The entailment probability becomes the label score.")
print("  Multi-label: each label scored independently (sigmoid)")
print("  Single-label: labels compete via softmax")
print()


# =====================================================================
# Part 2 – Masked LM Prompting
# =====================================================================
print("=" * 60)
print("Part 2: Masked LM Prompting for Zero-Shot")
print("=" * 60)

# An alternative zero-shot approach: use a masked language model
# to score how well candidate words fill a prompt template.

try:
    fill_mask = pipeline("fill-mask", model="bert-base-uncased")

    text = "The researchers discovered a new species of butterfly in the Amazon."
    prompt = f"{text} This text is about [MASK]."

    # Score specific candidate labels
    candidates = ["science", "sports", "finance", "nature"]
    output = fill_mask(prompt, targets=candidates)
    print("  MLM prompting scores:")
    for item in output:
        print(f"    {item['token_str']:12s}: {item['score']:.4f}")
    print()

    # The model assigns higher probability to semantically fitting labels
    print("  Advantage: Works with any MLM, no NLI training needed")
    print("  Limitation: Constrained to single-token labels")
    print()

except (ImportError, OSError) as e:
    print(f"  Skipping MLM demo: {e}")
    print()


# =====================================================================
# Part 3 – Embedding-Based Few-Shot with Nearest Neighbours
# =====================================================================
print("=" * 60)
print("Part 3: Embedding-Based Few-Shot Classification")
print("=" * 60)

# With a handful of labeled examples, we can classify new texts by:
# 1. Encoding all texts using a pretrained encoder
# 2. Finding nearest neighbours among labeled examples
# 3. Majority-voting on labels

try:
    from transformers import AutoTokenizer, AutoModel

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def mean_pooling(model_output, attention_mask):
        """Pool token embeddings to get sentence embedding."""
        token_embs = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
        sum_embs = torch.sum(token_embs * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embs / sum_mask

    def encode_texts(texts: List[str]) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        inputs = tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = mean_pooling(outputs, inputs["attention_mask"])
        return embeddings.cpu().numpy()

    # Simulate a few-shot setup: 2 examples per class
    few_shot_examples = [
        ("Apple reported record quarterly earnings of $83B.", "finance"),
        ("The Fed raised interest rates by 25 basis points.", "finance"),
        ("NVIDIA announced a new GPU architecture for AI.", "technology"),
        ("Python 3.12 brings significant performance improvements.", "technology"),
        ("The Lakers won the NBA championship in overtime.", "sports"),
        ("Serena Williams announced her retirement from tennis.", "sports"),
    ]

    texts = [t for t, _ in few_shot_examples]
    labels = [l for _, l in few_shot_examples]
    embeddings = encode_texts(texts)

    # Classify a new text via nearest neighbour
    query_texts = [
        "Tesla stock surged 15% after strong delivery numbers.",
        "Google released a new version of TensorFlow.",
        "Manchester United signed a new striker for $80M.",
    ]
    query_embeddings = encode_texts(query_texts)

    # Cosine similarity
    from numpy.linalg import norm

    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    print("  Few-shot nearest neighbour classification:")
    for i, q_text in enumerate(query_texts):
        sims = [cosine_sim(query_embeddings[i], e) for e in embeddings]
        best_idx = np.argmax(sims)
        print(f"    Query: {q_text[:60]}...")
        print(f"    → Predicted: {labels[best_idx]} (sim={sims[best_idx]:.3f})")
        print()

    # K-NN with majority voting
    def knn_classify(query_emb, support_embs, support_labels, k=3):
        """K-nearest neighbour classification."""
        sims = [cosine_sim(query_emb, e) for e in support_embs]
        top_k_idx = np.argsort(sims)[-k:]
        votes = [support_labels[i] for i in top_k_idx]
        from collections import Counter
        return Counter(votes).most_common(1)[0][0]

    print("  K-NN (k=3) classification:")
    for i, q_text in enumerate(query_texts):
        pred = knn_classify(query_embeddings[i], embeddings, labels, k=3)
        print(f"    {q_text[:60]}... → {pred}")
    print()

except (ImportError, OSError) as e:
    print(f"  Skipping embedding demo: {e}")
    print()


# =====================================================================
# Part 4 – Data Augmentation for Low-Resource NLP
# =====================================================================
print("=" * 60)
print("Part 4: Data Augmentation Techniques")
print("=" * 60)

# When you have only a handful of labeled examples, augmentation
# can 2-3x your effective training set.

import random
import re

def synonym_replace(text: str, n: int = 1) -> str:
    """Simple synonym replacement using a small dictionary."""
    synonyms = {
        "good": ["great", "excellent", "fine"],
        "bad": ["poor", "terrible", "awful"],
        "big": ["large", "huge", "massive"],
        "small": ["tiny", "little", "miniature"],
        "fast": ["quick", "rapid", "swift"],
    }
    words = text.split()
    for _ in range(n):
        for i, w in enumerate(words):
            if w.lower() in synonyms:
                words[i] = random.choice(synonyms[w.lower()])
                break
    return " ".join(words)


def random_deletion(text: str, p: float = 0.1) -> str:
    """Randomly delete words with probability p."""
    words = text.split()
    if len(words) <= 1:
        return text
    remaining = [w for w in words if random.random() > p]
    return " ".join(remaining) if remaining else random.choice(words)


def random_swap(text: str, n: int = 1) -> str:
    """Randomly swap n pairs of words."""
    words = text.split()
    for _ in range(n):
        if len(words) >= 2:
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
    return " ".join(words)


# Demo augmentation
original = "The stock market performed very good today with big gains"
print(f"  Original:        {original}")
print(f"  Synonym replace: {synonym_replace(original)}")
print(f"  Random deletion: {random_deletion(original, p=0.2)}")
print(f"  Random swap:     {random_swap(original, n=2)}")
print()

# In practice, contextual augmentation with a pretrained MLM is stronger:
# aug = naw.ContextualWordEmbsAug(model_path="distilbert-base-uncased",
#                                  action="substitute")
# augmented = aug.augment(text)
print("  For production, use contextual augmentation with a pretrained MLM")
print("  (e.g., nlpaug library with DistilBERT for word substitution)")
print()


# =====================================================================
# Part 5 – Domain-Adapted MLM → Fine-Tuning Pipeline
# =====================================================================
print("=" * 60)
print("Part 5: Domain Adaptation Pipeline")
print("=" * 60)

# The most effective low-resource strategy combines:
# 1. Pre-train MLM on domain-specific UNLABELED data
# 2. Fine-tune the adapted model on the small labeled set
#
# This works because:
# - Step 1 teaches the model domain vocabulary and patterns
# - Step 2 needs fewer labeled examples to converge

print("""
  Pipeline for low-resource classification:

  Step 1: Domain-adapt with MLM (unsupervised)
  ─────────────────────────────────────────────
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)
    trainer = Trainer(model=model, data_collator=collator,
                      train_dataset=unlabeled_domain_data)
    trainer.train()
    model.save_pretrained("domain-adapted-bert")

  Step 2: Fine-tune classifier (supervised)
  ──────────────────────────────────────────
    config = AutoConfig.from_pretrained("domain-adapted-bert")
    config.num_labels = num_classes
    config.problem_type = "multi_label_classification"  # or single_label
    model = AutoModelForSequenceClassification.from_pretrained(
        "domain-adapted-bert", config=config)
    trainer = Trainer(model=model, train_dataset=small_labeled_set,
                      eval_dataset=validation_set)
    trainer.train()

  Key settings:
    - MLM probability: 0.15 (standard)
    - MLM epochs: 10-20 on domain data
    - Fine-tuning epochs: 10-20 (more epochs for smaller datasets)
    - Learning rate: 3e-5 (fine-tuning), 5e-5 (MLM adaptation)
""")


# =====================================================================
# Part 6 – Comparison Summary
# =====================================================================
print("=" * 60)
print("Part 6: Method Comparison")
print("=" * 60)

print("""
  Method                 | Labels needed | Typical F1 | Best for
  ─────────────────────────────────────────────────────────────────
  Zero-shot (NLI)        | 0             | 0.30-0.50  | Quick baseline, no data
  MLM prompting          | 0             | 0.20-0.40  | Single-token labels
  Embedding + KNN        | 5-50          | 0.40-0.60  | Fast prototyping
  Naive Bayes + Aug      | 10-100        | 0.35-0.55  | Simple, interpretable
  Fine-tune (vanilla)    | 50-500        | 0.50-0.70  | Standard approach
  Domain-adapt + tune    | 50-500        | 0.55-0.75  | Best low-resource
  Full fine-tune         | 1000+         | 0.70-0.90  | Production quality

  Decision tree:
    No labeled data?
      → Zero-shot NLI pipeline
    < 50 labeled examples?
      → Embedding lookup / KNN
    < 500 labeled examples + unlabeled domain data?
      → Domain-adapt MLM then fine-tune
    Plenty of labeled data?
      → Standard fine-tuning
""")

print("Done.")


if __name__ == "__main__":
    pass
