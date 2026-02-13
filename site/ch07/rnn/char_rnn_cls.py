"""
Character-Level RNN — Name Classification
==========================================

Classifies surnames into their language of origin using a character-level RNN.
Each character is one-hot encoded and fed sequentially; the final hidden state
drives a softmax over 18 nationality classes.

Source: https://github.com/pytorch/tutorials/blob/main/intermediate_source/char_rnn_classification_tutorial.py

Architecture
------------
At every time-step the RNN receives the concatenation of the current
character one-hot vector and the previous hidden state:

    h_t = tanh(W_ih @ [x_t ; h_{t-1}] + b_ih)
    y   = softmax(W_ho @ h_{t_last} + b_ho)

The model is trained with NLLLoss (negative log-likelihood) on the
log-softmax output, using vanilla SGD (manual parameter updates).

Pipeline
--------
1.  **download.py**  — fetch and unzip the names dataset
2.  **generate_global_name_space.py** — build vocabulary, category mappings,
    and hyper-parameter constants
3.  **model.py**  — define the single-layer RNN
4.  **train.py**  — training loop, evaluation, and confusion matrix
5.  **predict.py** — top-k nationality prediction for a given surname

All five modules are merged into this single file for portability.

Run
---
    python char_rnn_classification.py
"""

# =============================================================================
# Imports
# =============================================================================
import glob
import math
import os
import random
import string
import time
import unicodedata
from io import open
from zipfile import ZipFile

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import requests
import torch
import torch.nn as nn


# =============================================================================
# 1. Download helpers
# =============================================================================
def download_and_extract_data(url: str, target_directory: str) -> None:
    """Download a zip file from *url* and extract it into *target_directory*."""
    os.makedirs(target_directory, exist_ok=True)
    response = requests.get(url)
    zip_file_path = os.path.join(target_directory, "data.zip")
    with open(zip_file_path, "wb") as f:
        f.write(response.content)
    with ZipFile(zip_file_path, "r") as zf:
        zf.extractall(target_directory)
    os.remove(zip_file_path)
    print("Data download and extraction completed.")


def find_files(path: str):
    return glob.glob(path)


# =============================================================================
# 2. Global namespace — vocabulary, categories, hyper-parameters
# =============================================================================
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)
# print(N_LETTERS)  # 57


def unicode_to_ascii(s: str) -> str:
    """Strip accents / diacritics and keep only characters in ALL_LETTERS."""
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in ALL_LETTERS
    )


def read_lines(filename: str):
    with open(filename, encoding="utf-8") as f:
        return [unicode_to_ascii(line.strip()) for line in f]


def build_category_lines_and_all_categories(data_dir: str = "data/names"):
    """Return ``(category_lines, all_categories)`` built from text files."""
    category_lines: dict[str, list[str]] = {}
    all_categories: list[str] = []
    for filename in find_files(os.path.join(data_dir, "*.txt")):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        category_lines[category] = read_lines(filename)
    return category_lines, all_categories


# ---------------------------------------------------------------------------
# Hyper-parameters (kept as module-level constants for clarity)
# ---------------------------------------------------------------------------
N_HIDDEN = 128
LEARNING_RATE = 0.005
N_ITERS = 100_000
PRINT_EVERY = 5_000
PLOT_EVERY = 1_000


# =============================================================================
# 3. Model
# =============================================================================
class RNN(nn.Module):
    """Single-layer Elman RNN for sequence classification.

    Parameters
    ----------
    input_size : int
        Dimensionality of the one-hot character encoding (``N_LETTERS``).
    hidden_size : int
        Number of hidden units.
    output_size : int
        Number of nationality classes (``N_CATEGORIES``).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor, shape ``(1, input_size)``
            One-hot encoded character.
        hidden : Tensor, shape ``(1, hidden_size)``
            Previous hidden state.

        Returns
        -------
        output : Tensor, shape ``(1, output_size)``
            Log-probabilities over classes.
        hidden : Tensor, shape ``(1, hidden_size)``
            Updated hidden state.
        """
        combined = torch.cat((x, hidden), dim=1)  # (1, input_size + hidden_size)
        hidden = self.i2h(combined)                # (1, hidden_size)
        output = self.h2o(combined)                # (1, output_size)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(1, self.hidden_size)


# =============================================================================
# 4. Encoding utilities
# =============================================================================
def letter_to_index(letter: str) -> int:
    return ALL_LETTERS.find(letter)


def letter_to_tensor(letter: str) -> torch.Tensor:
    """Return a ``(1, N_LETTERS)`` one-hot tensor for a single character."""
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


def line_to_tensor(line: str) -> torch.Tensor:
    """Return a ``(len(line), 1, N_LETTERS)`` tensor of one-hot vectors."""
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor


# =============================================================================
# 5. Training
# =============================================================================
def category_from_output(output: torch.Tensor, all_categories: list[str]):
    """Return the predicted category name and its index."""
    _, top_i = output.topk(1)
    cat_i = top_i[0].item()
    return all_categories[cat_i], cat_i


def random_choice(lst):
    return lst[random.randint(0, len(lst) - 1)]


def random_training_example(all_categories, category_lines):
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor(
        [all_categories.index(category)], dtype=torch.long
    )
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


def train_step(
    rnn: RNN,
    criterion: nn.Module,
    category_tensor: torch.Tensor,
    line_tensor: torch.Tensor,
):
    """Execute one training step (forward + backward + manual SGD update)."""
    hidden = rnn.init_hidden()
    rnn.zero_grad()

    for i in range(line_tensor.size(0)):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Manual SGD
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-LEARNING_RATE)

    return output, loss.item()


def time_since(since: float) -> str:
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {int(s)}s"


def run_train_loop(rnn, criterion, all_categories, category_lines):
    current_loss = 0.0
    all_losses: list[float] = []
    iters: list[int] = []
    start = time.time()

    for it in range(1, N_ITERS + 1):
        cat, line, cat_tensor, line_tensor = random_training_example(
            all_categories, category_lines
        )
        output, loss = train_step(rnn, criterion, cat_tensor, line_tensor)
        current_loss += loss

        if it % PRINT_EVERY == 0:
            guess, _ = category_from_output(output, all_categories)
            correct = "✓" if guess == cat else f"✗ ({cat})"
            pct = it / N_ITERS * 100
            print(
                f"{it} {pct:.0f}% ({time_since(start)}) "
                f"{loss:.4f} {line} / {guess} {correct}"
            )

        if it % PLOT_EVERY == 0:
            all_losses.append(current_loss / PLOT_EVERY)
            iters.append(it)
            current_loss = 0.0

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(iters, all_losses, label="loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg Loss")
    ax.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# 6. Evaluation & confusion matrix
# =============================================================================
def evaluate(rnn: RNN, line_tensor: torch.Tensor):
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size(0)):
        output, hidden = rnn(line_tensor[i], hidden)
    return output


def compute_confusion_matrix(rnn, all_categories, category_lines, n_samples=10_000):
    n_cat = len(all_categories)
    confusion = torch.zeros(n_cat, n_cat)

    for _ in range(n_samples):
        cat, line, _, line_tensor = random_training_example(
            all_categories, category_lines
        )
        output = evaluate(rnn, line_tensor)
        _, guess_i = category_from_output(output, all_categories)
        cat_i = all_categories.index(cat)
        confusion[cat_i][guess_i] += 1

    # Normalise rows
    for i in range(n_cat):
        confusion[i] = confusion[i] / confusion[i].sum()

    draw_confusion(confusion, all_categories)
    return confusion


def draw_confusion(confusion, all_categories):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    n = len(all_categories)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_categories, rotation=90)
    ax.set_yticklabels(all_categories)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.tight_layout()
    plt.show()


# =============================================================================
# 7. Prediction
# =============================================================================
def predict(rnn, input_line, all_categories, n_predictions=3):
    print(f"\n> {input_line}")
    with torch.no_grad():
        output = evaluate(rnn, line_to_tensor(input_line))
        topv, topi = output.topk(n_predictions, dim=1)
        exp_sum = torch.exp(output).sum()

        for i in range(n_predictions):
            val = topv[0][i].item()
            idx = topi[0][i].item()
            prob = np.exp(val) / exp_sum.item()
            print(f"  ({prob:.4f}) {all_categories[idx]}")


# =============================================================================
# 8. Main
# =============================================================================
def main():
    # --- Download data if missing -------------------------------------------
    if not os.path.isdir("data/names"):
        download_and_extract_data(
            "https://download.pytorch.org/tutorial/data.zip", "./"
        )

    # --- Build global lookups -----------------------------------------------
    category_lines, all_categories = build_category_lines_and_all_categories()
    n_categories = len(all_categories)
    print(f"Categories ({n_categories}): {all_categories}")

    if n_categories == 0:
        raise RuntimeError(
            "No data found.  Download from "
            "https://download.pytorch.org/tutorial/data.zip"
        )

    # --- Instantiate model and train ----------------------------------------
    model = RNN(N_LETTERS, N_HIDDEN, n_categories)
    criterion = nn.NLLLoss()
    print(model)

    run_train_loop(model, criterion, all_categories, category_lines)
    compute_confusion_matrix(model, all_categories, category_lines)

    # --- Save & reload for prediction demo ----------------------------------
    torch.save(model.state_dict(), "classification_weights.pth")

    model2 = RNN(N_LETTERS, N_HIDDEN, n_categories)
    model2.load_state_dict(torch.load("classification_weights.pth"))
    model2.eval()

    predict(model2, "Dovesky", all_categories)
    predict(model2, "Jackson", all_categories)
    predict(model2, "Satoshi", all_categories)


if __name__ == "__main__":
    main()
