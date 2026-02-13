"""
LSTM Part-of-Speech Tagger
============================

Trains a simple LSTM-based POS tagger on toy data.

Architecture:
    word indices -> Embedding(vocab, 6)
                 -> LSTM(6, 6)
                 -> Linear(6, tagset_size)
                 -> log_softmax
                 -> NLLLoss

Tags: DET (determiner), NN (noun), V (verb)

Training data:
    "The dog ate the apple"   -> DET NN V DET NN
    "Everybody read that book" -> NN  V  DET NN

Source: https://github.com/pytorch/tutorials/blob/main/beginner_source/nlp/sequence_models_tutorial.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


# =============================================================================
# Data
# =============================================================================
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"]),
]

word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
ix_to_tag = {v: k for k, v in tag_to_ix.items()}

EMBEDDING_DIM = 6
HIDDEN_DIM = 6
VOCAB_SIZE = len(word_to_ix)
TAGSET_SIZE = len(tag_to_ix)
EPOCHS = 300
LR = 0.1


def prepare_sequence(seq: list[str], to_ix: dict[str, int]) -> torch.Tensor:
    return torch.tensor([to_ix[w] for w in seq], dtype=torch.long)


# =============================================================================
# Model
# =============================================================================
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        # LSTM expects (seq_len, batch=1, features)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# =============================================================================
# Training
# =============================================================================
def train(model, optimizer, loss_function):
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for sentence, tags in training_data:
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS}  Loss: {total_loss:.4f}")


# =============================================================================
# Inference
# =============================================================================
def predict(model, sentence: list[str]) -> list[str]:
    with torch.no_grad():
        inputs = prepare_sequence(sentence, word_to_ix)
        tag_scores = model(inputs)
        predicted_ix = tag_scores.argmax(dim=1).tolist()
    return [ix_to_tag[ix] for ix in predicted_ix]


# =============================================================================
# Main
# =============================================================================
def main():
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAGSET_SIZE)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)

    # Before training
    print("=== Before training ===")
    for sent, gold in training_data:
        pred = predict(model, sent)
        print(f"  {' '.join(sent)}")
        print(f"    Gold: {gold}")
        print(f"    Pred: {pred}")

    # Train
    print("\n=== Training ===")
    train(model, optimizer, loss_function)

    # After training
    print("\n=== After training ===")
    for sent, gold in training_data:
        pred = predict(model, sent)
        print(f"  {' '.join(sent)}")
        print(f"    Gold: {gold}")
        print(f"    Pred: {pred}")


if __name__ == "__main__":
    main()
