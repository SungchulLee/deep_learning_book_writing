"""
Bag-of-Words (BoW) Text Classifier
====================================

Classifies sentences as SPANISH or ENGLISH using a simple linear model
over bag-of-words representations.

Architecture:
    sentence -> BoW vector (word counts, size=vocab)
             -> Linear(vocab, 2)
             -> CrossEntropyLoss

The BoW vector is a sparse representation where each dimension counts
how many times the corresponding word appears in the sentence.

Source: https://github.com/pytorch/tutorials/blob/main/beginner_source/nlp/deep_learning_tutorial.py
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

# =============================================================================
# Configuration
# =============================================================================
EPOCHS = 100
LR = 0.1

# =============================================================================
# Data
# =============================================================================
DATA = [
    ("me gusta comer en la cafeteria".split(), "SPANISH"),
    ("Give it to me".split(), "ENGLISH"),
    ("No creo que sea una buena idea".split(), "SPANISH"),
    ("No it is not a good idea to get lost at sea".split(), "ENGLISH"),
]

TEST_DATA = [
    ("Yo creo que si".split(), "SPANISH"),
    ("it is lost on me".split(), "ENGLISH"),
]

LABEL_TO_IX = {"SPANISH": 0, "ENGLISH": 1}

# Build vocabulary from all data
WORD_TO_IX = {}
for words, _ in DATA + TEST_DATA:
    for word in words:
        if word not in WORD_TO_IX:
            WORD_TO_IX[word] = len(WORD_TO_IX)

VOCAB_SIZE = len(WORD_TO_IX)
NUM_LABELS = len(LABEL_TO_IX)


# =============================================================================
# Helpers
# =============================================================================
def make_bow_vector(sentence: list[str]) -> torch.Tensor:
    """Convert a sentence to a bag-of-words count vector."""
    vec = torch.zeros(VOCAB_SIZE)
    for word in sentence:
        vec[WORD_TO_IX[word]] += 1
    return vec.view(1, -1)


def make_target(label: str) -> torch.Tensor:
    return torch.tensor([LABEL_TO_IX[label]], dtype=torch.long)


# =============================================================================
# Model
# =============================================================================
class BoWClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec: torch.Tensor) -> torch.Tensor:
        return self.linear(bow_vec)


# =============================================================================
# Training & Evaluation
# =============================================================================
def print_probs(model, samples, header: str = ""):
    """Print softmax probabilities for each sample."""
    if header:
        print(header)
    with torch.no_grad():
        for instance, label in samples:
            bow_vec = make_bow_vector(instance)
            scores = model(bow_vec)
            probs = torch.softmax(scores, dim=1)
            pred_label = "SPANISH" if probs[0, 0] > probs[0, 1] else "ENGLISH"
            correct = "✓" if pred_label == label else "✗"
            print(
                f"  [{correct}] gold={label:8s}  pred={pred_label:8s}  "
                f"P(SP)={probs[0,0]:.4f}  P(EN)={probs[0,1]:.4f}"
            )


def train(model, loss_function, optimizer):
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for instance, label in DATA:
            model.zero_grad()
            bow_vec = make_bow_vector(instance)
            target = make_target(label)
            scores = model(bow_vec)
            loss = loss_function(scores, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}  Loss: {total_loss:.4f}")


# =============================================================================
# Main
# =============================================================================
def main():
    model = BoWClassifier(VOCAB_SIZE, NUM_LABELS)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)

    print_probs(model, DATA + TEST_DATA, header="=== Before training ===")

    print("\n=== Training ===")
    train(model, loss_function, optimizer)

    print()
    print_probs(model, DATA + TEST_DATA, header="=== After training (train + test) ===")


if __name__ == "__main__":
    main()
