"""
IMDB Feedforward Classifier
=============================

Binary sentiment classification on IMDB using a feedforward network
with an embedding layer.

Architecture:
    token indices -> Embedding(vocab+2, 80)
                  -> flatten
                  -> Linear(80*80, 32) -> ReLU
                  -> Linear(32, 1) -> Sigmoid
                  -> BCELoss

Pipeline:
    1. Download IMDB dataset
    2. Tokenize with torchtext Field (fix_length=80)
    3. Build vocabulary (top 1000 words)
    4. Train with BucketIterator

Requirements:
    pip install torchtext==0.6 torch matplotlib

Note:
    Uses torchtext legacy API (<=0.6). For modern torchtext (>=0.12),
    the dataset/field APIs have changed significantly.

Source: Adapted from
    https://github.com/EmreOzkose/pytorch-Deep-Learning/blob/master/12-regularization.ipynb
"""

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchtext.data import Field, LabelField, BucketIterator
from torchtext import datasets

# Optional: download helper (torchtext 0.6)
try:
    from torchtext.utils import download_from_url, extract_archive
except ImportError:
    download_from_url = None

# =============================================================================
# Configuration
# =============================================================================
DATA_FOLDER = "./imdb"
MODEL_FOLDER = "./model"
os.makedirs(MODEL_FOLDER, exist_ok=True)

MAX_LEN = 80
BATCH_SIZE = 64
NUM_WORDS = 1000
INPUT_DIM = NUM_WORDS + 2  # +2 for <unk> and <pad>
EMBEDDING_DIM = MAX_LEN
HIDDEN_DIM = 32
OUTPUT_DIM = 1
NUM_EPOCHS = 10
LR = 1e-3
SEED = 1337
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torchtext Fields
TEXT = Field(
    sequential=True,
    fix_length=MAX_LEN,
    batch_first=True,
    lower=True,
    dtype=torch.long,
)
LABEL = LabelField(sequential=False, dtype=torch.float)


# =============================================================================
# Data loading
# =============================================================================
def download_imdb():
    """Download IMDB if not present (requires torchtext<=0.6)."""
    if not os.path.exists(DATA_FOLDER) and download_from_url is not None:
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        download_from_url(url, root="./")
        extract_archive("./aclImdb_v1.tar.gz", DATA_FOLDER)
        if os.path.exists("./aclImdb_v1.tar.gz"):
            os.remove("./aclImdb_v1.tar.gz")


class FixBatchGenerator:
    """Wraps BucketIterator to yield (X, y) tuples."""

    def __init__(self, dl, x_field: str, y_field: str):
        self.dl = dl
        self.x_field = x_field
        self.y_field = y_field

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield X, y


def load_data():
    import random

    download_imdb()

    ds_train, ds_test = datasets.IMDB.splits(TEXT, LABEL, path="./imdb/aclImdb/")
    ds_train, ds_valid = ds_train.split(random_state=random.seed(SEED))

    TEXT.build_vocab(ds_train, max_size=NUM_WORDS)
    LABEL.build_vocab(ds_train)

    train_loader, valid_loader, test_loader = BucketIterator.splits(
        (ds_train, ds_valid, ds_test),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        repeat=False,
    )

    train_loader = FixBatchGenerator(train_loader, "text", "label")
    valid_loader = FixBatchGenerator(valid_loader, "text", "label")
    test_loader = FixBatchGenerator(test_loader, "text", "label")

    return train_loader, valid_loader, test_loader


# =============================================================================
# Model
# =============================================================================
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = OUTPUT_DIM,
        max_len: int = MAX_LEN,
    ):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear(max_len * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)                           # (B, L, D)
        embedded = embedded.view(-1, self.max_len * embedded.size(-1))  # (B, L*D)
        out = torch.relu(self.fc1(embedded))                   # (B, H)
        out = torch.sigmoid(self.fc2(out))                     # (B, 1)
        return out


# =============================================================================
# Training
# =============================================================================
def train(model, optimizer, criterion, train_loader, valid_loader):
    step = 0
    best_accuracy = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        for samples, labels in train_loader:
            model.train()
            samples = samples.view(-1, MAX_LEN).to(DEVICE)
            labels = labels.view(-1, 1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(samples)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            step += 1

            if step % 100 == 0:
                val_loss, val_acc = evaluate(model, criterion, valid_loader)
                history["train_loss"].append(loss.item())
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                print(
                    f"Step {step:5d} | "
                    f"Train Loss: {loss.item():.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.2f}%"
                )

                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    torch.save(
                        model.state_dict(),
                        os.path.join(MODEL_FOLDER, "best_model.pth"),
                    )

    print(f"\nBest validation accuracy: {best_accuracy:.2f}%")
    return history


def evaluate(model, criterion, loader):
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for samples, labels in loader:
            samples = samples.view(-1, MAX_LEN).to(DEVICE)
            labels_flat = labels.view(-1).to(DEVICE)

            outputs = model(samples)
            total_loss += criterion(outputs.view(-1, 1), labels_flat.view(-1, 1)).item()
            n_batches += 1

            predicted = outputs.ge(0.5).view(-1).float()
            correct += (predicted.cpu() == labels.float()).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


# =============================================================================
# Plotting
# =============================================================================
def plot_history(history):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3))

    ax0.plot(history["train_loss"], label="Train Loss")
    ax0.plot(history["val_loss"], label="Validation Loss")
    ax0.set_xlabel("Checkpoint (every 100 steps)")
    ax0.set_ylabel("Loss")
    ax0.legend()

    ax1.plot(history["val_acc"], label="Validation Accuracy")
    ax1.set_xlabel("Checkpoint (every 100 steps)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# Main
# =============================================================================
def main():
    train_loader, valid_loader, test_loader = load_data()

    model = FeedforwardNeuralNetModel().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Device: {DEVICE}")
    print(f"Vocab size: {len(TEXT.vocab)} (max {NUM_WORDS} + specials)")
    print()

    history = train(model, optimizer, criterion, train_loader, valid_loader)
    plot_history(history)

    # Final test evaluation
    test_loss, test_acc = evaluate(model, criterion, test_loader)
    print(f"\nTest Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
