"""
N-gram Language Model with Word Embeddings
===========================================

Trains a trigram (context_size=2) neural language model on Shakespeare Sonnet 2.
The model learns word embeddings as a byproduct of predicting the next word
from its two preceding context words.

Architecture:
    context words -> Embedding lookup -> concat -> Linear(128) -> ReLU -> Linear(vocab)

Loss options:
    0: nn.CrossEntropyLoss  (default)
    1: F.cross_entropy
    2: nn.NLLLoss (requires log_softmax output)

Source: https://github.com/pytorch/tutorials/blob/main/beginner_source/nlp/word_embeddings_tutorial.py
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =============================================================================
# Configuration
# =============================================================================
parser = argparse.ArgumentParser(description="N-gram Language Model")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
parser.add_argument("--epochs", type=int, default=1000, help="number of epochs (default: 1000)")
parser.add_argument(
    "--loss_function_choice",
    type=int,
    default=0,
    choices=[0, 1, 2],
    help="0: CrossEntropyLoss, 1: F.cross_entropy, 2: NLLLoss",
)
parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

# =============================================================================
# Data — Shakespeare Sonnet 2
# =============================================================================
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}

# Build trigrams: ([w_{i-2}, w_{i-1}], w_i)
ngrams = [
    ([test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)], test_sentence[i])
    for i in range(CONTEXT_SIZE, len(test_sentence))
]

# Select loss function
if ARGS.loss_function_choice == 0:
    loss_function = nn.CrossEntropyLoss()
elif ARGS.loss_function_choice == 1:
    loss_function = F.cross_entropy
else:
    loss_function = nn.NLLLoss()


# =============================================================================
# Model
# =============================================================================
class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, context_size: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(inputs).view(1, -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        if ARGS.loss_function_choice == 2:
            return F.log_softmax(out, dim=1)
        return out


# =============================================================================
# Training
# =============================================================================
def train(model, optimizer):
    losses = []
    for epoch in range(ARGS.epochs):
        total_loss = 0.0
        for context, target in ngrams:
            context_idxs = torch.tensor(
                [word_to_ix[w] for w in context], dtype=torch.long
            )
            model.zero_grad()
            log_probs = model(context_idxs)
            target_tensor = torch.tensor([word_to_ix[target]], dtype=torch.long)
            loss = loss_function(log_probs, target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
    return losses


# =============================================================================
# Main
# =============================================================================
def main():
    vocab_size = len(vocab)
    model = NGramLanguageModeler(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=ARGS.lr)

    losses = train(model, optimizer)

    # Plot training loss
    _, ax = plt.subplots(figsize=(12, 3))
    ax.plot(losses, label="loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Inspect learned embedding for "beauty"
    beauty_embedding = model.embeddings.weight[word_to_ix["beauty"]]
    print(f"\nEmbedding for 'beauty':\n{beauty_embedding}")


if __name__ == "__main__":
    main()
