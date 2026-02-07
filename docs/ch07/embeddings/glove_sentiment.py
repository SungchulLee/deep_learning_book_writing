"""
GloVe Sentiment Classifier — IMDB
===================================

Sentiment classification on IMDB using pretrained GloVe embeddings.

Architecture:
    tokens -> GloVe Embedding (frozen or fine-tuned)
            -> mean pooling over sequence
            -> Linear(hidden) -> ReLU -> Linear(1)
            -> BCEWithLogitsLoss

Requirements:
    pip install torchtext spacy torch
    python -m spacy download en_core_web_sm

Note:
    Uses torchtext legacy API (<=0.6).  For modern torchtext (>=0.12),
    see torchtext.datasets.IMDB with DataPipes.

Source: Adapted from colab Section 2 — GloVe Embedding / SentimentClassifier
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data, datasets
from torchtext.vocab import GloVe

# Try to load spaCy tokenizer; fall back to simple split
try:
    import spacy

    _spacy_en = spacy.load("en_core_web_sm")

    def tokenizer(text: str) -> list[str]:
        return [tok.text for tok in _spacy_en.tokenizer(text)]

except (ImportError, OSError):
    print("spaCy not found — falling back to str.split() tokenizer")

    def tokenizer(text: str) -> list[str]:
        return text.split()


# =============================================================================
# Configuration
# =============================================================================
EMBEDDING_DIM = 100
HIDDEN_DIM = 64
OUTPUT_DIM = 1
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Data
# =============================================================================
def load_data():
    TEXT = data.Field(
        tokenize=tokenizer,
        lower=True,
        include_lengths=True,
        batch_first=True,
    )
    LABEL = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(train_data, vectors=GloVe(name="6B", dim=EMBEDDING_DIM))
    LABEL.build_vocab(train_data)

    train_iter, test_iter = data.BucketIterator.splits(
        (train_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        device=DEVICE,
    )
    return TEXT, LABEL, train_iter, test_iter


# =============================================================================
# Model
# =============================================================================
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, text_and_lengths):
        """
        Parameters
        ----------
        text_and_lengths : tuple of (text_tensor, lengths_tensor)
            text_tensor: (batch, seq_len)  — token indices
            lengths_tensor: (batch,)       — actual lengths
        """
        text, lengths = text_and_lengths
        embedded = self.embedding(text)              # (B, L, D)
        pooled = torch.mean(embedded, dim=1)         # (B, D)  — mean over sequence
        hidden = self.relu(self.fc(pooled))          # (B, H)
        return self.out(hidden)                      # (B, 1)

    def load_pretrained_embeddings(self, vectors: torch.Tensor):
        """Copy pretrained GloVe vectors into the embedding layer."""
        self.embedding.weight.data.copy_(vectors)


# =============================================================================
# Training & Evaluation
# =============================================================================
def train_epoch(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
    for batch in iterator:
        text, labels = batch.text, batch.label

        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        rounded = torch.round(torch.sigmoid(predictions))
        epoch_correct += (rounded == labels).sum().item()
        epoch_total += labels.size(0)

    return epoch_loss / len(iterator), epoch_correct / epoch_total


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch.text, batch.label
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels)

            epoch_loss += loss.item()
            rounded = torch.round(torch.sigmoid(predictions))
            epoch_correct += (rounded == labels).sum().item()
            epoch_total += labels.size(0)

    return epoch_loss / len(iterator), epoch_correct / epoch_total


# =============================================================================
# Main
# =============================================================================
def main():
    TEXT, LABEL, train_iter, test_iter = load_data()

    vocab_size = len(TEXT.vocab)
    model = SentimentClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model.load_pretrained_embeddings(TEXT.vocab.vectors)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Vocabulary size : {vocab_size}")
    print(f"Device          : {DEVICE}")
    print(f"Epochs          : {EPOCHS}")
    print()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_iter, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_iter, criterion)
        print(
            f"Epoch {epoch}/{EPOCHS}  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Test  Loss: {test_loss:.4f}  Acc: {test_acc:.4f}"
        )


if __name__ == "__main__":
    main()
