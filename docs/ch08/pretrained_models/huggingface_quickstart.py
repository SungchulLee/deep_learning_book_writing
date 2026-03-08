"""Huggingface quickstart."""
# ---
# title: "HuggingFace Transformers Quickstart"
# description: "Using pretrained transformer models for inference, fine-tuning, and generation"
# ---
#
# While ch08 builds BERT/GPT from scratch in PyTorch, in practice most
# people use the HuggingFace `transformers` library.  This script covers:
#
#   1. Pipeline API — one-line inference for common tasks
#   2. AutoTokenizer + AutoModel — manual loading
#   3. Tokenization details — input IDs, attention masks, special tokens
#   4. Fine-tuning a pretrained model on a small dataset
#   5. Text generation with GPT-2
#
# Adapted from: O'Reilly Hands-On ML 3rd Edition, Chapter 16 (TF → PyTorch)
#
# Install:  pip install transformers datasets torch

import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# =====================================================================
# 1. Pipeline API — zero-code inference
# =====================================================================
print("=" * 60)
print("1. HuggingFace Pipeline API")
print("=" * 60)

from transformers import pipeline

# Sentiment analysis (downloads model on first run)
classifier = pipeline("sentiment-analysis")
result = classifier("The actors were very convincing.")
print(f"Sentiment: {result}")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Batch inference
results = classifier([
    "I loved every minute of this film.",
    "The plot was confusing and the ending was disappointing.",
])
for r in results:
    print(f"  {r['label']:>8s}  (confidence: {r['score']:.3f})")

print()


# =====================================================================
# 2. Natural Language Inference (NLI) with a specific model
# =====================================================================
print("=" * 60)
print("2. NLI with a specific model")
print("=" * 60)

model_name = "cross-encoder/nli-distilroberta-base"
nli_pipe = pipeline("text-classification", model=model_name)

pairs = [
    "A man is eating pizza. [SEP] A man is eating food.",       # entailment
    "A woman is playing guitar. [SEP] Nobody is playing music.", # contradiction
    "It is raining outside. [SEP] The grass is wet.",           # neutral/entailment
]
for text in pairs:
    result = nli_pipe(text)
    print(f"  {result[0]['label']:>14s} ({result[0]['score']:.3f})  ← {text[:50]}...")

print()


# =====================================================================
# 3. Manual loading: AutoTokenizer + AutoModel
# =====================================================================
print("=" * 60)
print("3. Manual tokenizer + model loading")
print("=" * 60)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Tokenize
text = "HuggingFace makes NLP incredibly easy!"
inputs = tokenizer(text, return_tensors="pt")

print(f"Input text: '{text}'")
print(f"Token IDs:  {inputs['input_ids'].tolist()}")
print(f"Decoded:    {tokenizer.decode(inputs['input_ids'][0])}")
print(f"Tokens:     {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

labels = model.config.id2label
for i, (label_id, label_name) in enumerate(labels.items()):
    print(f"  {label_name}: {probs[0, i]:.4f}")

print()


# =====================================================================
# 4. Tokenization details — batches, padding, truncation
# =====================================================================
print("=" * 60)
print("4. Tokenization details")
print("=" * 60)

sentences = [
    "Short text.",
    "This is a much longer sentence that demonstrates padding behaviour.",
]
encoded = tokenizer(
    sentences,
    padding=True,           # pad shorter sequences to longest
    truncation=True,        # truncate to max_length
    max_length=32,
    return_tensors="pt",
)

print(f"input_ids shape:      {encoded['input_ids'].shape}")
print(f"attention_mask shape:  {encoded['attention_mask'].shape}")
print(f"\nSentence 1 tokens:  {tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])}")
print(f"Sentence 1 mask:    {encoded['attention_mask'][0].tolist()}")
print(f"\nSentence 2 tokens:  {tokenizer.convert_ids_to_tokens(encoded['input_ids'][1])}")
print(f"Sentence 2 mask:    {encoded['attention_mask'][1].tolist()}")

# Sentence pairs (for NLI, QA, etc.)
pair_encoded = tokenizer(
    [("I like soccer.", "We all love soccer!")],
    padding=True,
    return_tensors="pt",
)
print(f"\nSentence pair tokens: {tokenizer.convert_ids_to_tokens(pair_encoded['input_ids'][0])}")
print(f"  Note [SEP] separating the two sentences.")

print()


# =====================================================================
# 5. Fine-tuning a pretrained model
# =====================================================================
print("=" * 60)
print("5. Fine-tuning on a small dataset")
print("=" * 60)

from transformers import AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader, TensorDataset

# Small demo dataset (in practice, use HuggingFace `datasets` library)
texts = [
    "The stock market rallied today on strong earnings.",       # positive
    "GDP growth exceeded expectations this quarter.",            # positive
    "The company filed for bankruptcy after years of losses.",   # negative
    "Inflation continues to erode consumer purchasing power.",   # negative
    "New regulations aim to stabilize the financial sector.",    # positive
    "Unemployment claims hit a record high this month.",         # negative
]
labels = [1, 1, 0, 0, 1, 0]

# Tokenize
encoded = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
dataset = TensorDataset(
    encoded["input_ids"],
    encoded["attention_mask"],
    torch.tensor(labels),
)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Load a fresh model (2 labels: negative=0, positive=1)
ft_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
optimizer = torch.optim.AdamW(ft_model.parameters(), lr=2e-5, weight_decay=0.01)

# Fine-tune for 3 epochs
ft_model.train()
for epoch in range(3):
    total_loss = 0
    for batch in loader:
        input_ids, attention_mask, labs = batch
        outputs = ft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labs,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"  Epoch {epoch + 1}: loss = {total_loss / len(loader):.4f}")

# Test inference
ft_model.eval()
test_text = "Markets surged as investor confidence grew."
test_inputs = tokenizer(test_text, return_tensors="pt")
with torch.no_grad():
    logits = ft_model(**test_inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
print(f"\n  Test: '{test_text}'")
print(f"  Predicted: {'positive' if pred == 1 else 'negative'}")

print()


# =====================================================================
# 6. Text generation with GPT-2
# =====================================================================
print("=" * 60)
print("6. Text generation with GPT-2")
print("=" * 60)

generator = pipeline("text-generation", model="gpt2")
prompt = "Deep learning has revolutionized"
results = generator(
    prompt,
    max_new_tokens=50,
    num_return_sequences=2,
    temperature=0.8,
    do_sample=True,
)
for i, r in enumerate(results):
    print(f"\n  --- Generation {i + 1} ---")
    print(f"  {r['generated_text'][:200]}")

print()


# =====================================================================
# 7. Available tasks cheat sheet
# =====================================================================
print("=" * 60)
print("7. Available pipeline tasks")
print("=" * 60)
tasks = [
    ("sentiment-analysis", "Classify text sentiment"),
    ("text-classification", "General text classification"),
    ("ner", "Named entity recognition"),
    ("question-answering", "Extractive QA (context + question)"),
    ("summarization", "Abstractive text summarization"),
    ("translation_en_to_fr", "Machine translation"),
    ("text-generation", "Autoregressive text generation"),
    ("fill-mask", "Masked language model predictions"),
    ("zero-shot-classification", "Classify without training data"),
    ("feature-extraction", "Get hidden state embeddings"),
]
for task, desc in tasks:
    print(f"  {task:<30s}  {desc}")

print("\nDone.")


if __name__ == "__main__":
    pass
