"""
Character-Level RNN — Name Generation
======================================

Generates plausible surnames conditioned on a nationality label.  The model is
an RNN that takes a category one-hot, the previous character one-hot, and the
previous hidden state, and outputs a distribution over the next character
(including an EOS token).

Source: https://github.com/pytorch/tutorials/blob/main/intermediate_source/char_rnn_generation_tutorial.py

Architecture
------------
At each time-step *t* the network receives the concatenation
``[category ; x_t ; h_{t-1}]`` and produces:

    h_t = W_ih @ [category ; x_t ; h_{t-1}]       (input → hidden)
    o_t = W_io @ [category ; x_t ; h_{t-1}]       (input → output)
    y_t = W_oo @ [h_t ; o_t]                       (combined → output)
    y_t = dropout(y_t)
    y_t = log_softmax(y_t)

Training uses NLLLoss over character-level targets that are shifted by one
position (the target for character *i* is character *i+1*, with the last
target being the EOS token).

Pipeline
--------
1.  **download.py** — fetch and unzip the names dataset
2.  **generate_global_name_space.py** — vocabulary, category maps, constants
3.  **model.py** — RNN with category conditioning
4.  **train.py** — training loop
5.  **generate_name.py** — autoregressive sampling conditioned on category
    and starting letter

All five modules are merged into this single file for portability.

Run
---
    python char_rnn_generation.py
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
import requests
import torch
import torch.nn as nn


# =============================================================================
# 1. Download helpers
# =============================================================================
def download_and_extract_data(url: str, target_directory: str) -> None:
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
# 2. Global namespace
# =============================================================================
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS) + 1  # +1 for EOS marker


def unicode_to_ascii(s: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in ALL_LETTERS[:-1]
        # ALL_LETTERS already includes the special chars; EOS is virtual
    )


def read_lines(filename: str):
    with open(filename, encoding="utf-8") as f:
        return [unicode_to_ascii(line.strip()) for line in f]


def build_category_lines_and_all_categories(data_dir: str = "data/names"):
    category_lines: dict[str, list[str]] = {}
    all_categories: list[str] = []
    for filename in find_files(os.path.join(data_dir, "*.txt")):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        category_lines[category] = read_lines(filename)
    return category_lines, all_categories


# Hyper-parameters
N_HIDDEN = 128
LEARNING_RATE = 0.0005
N_ITERS = 100_000
PRINT_EVERY = 5_000
PLOT_EVERY = 500
MAX_LENGTH = 20


# =============================================================================
# 3. Model
# =============================================================================
class RNN(nn.Module):
    """Conditional character-level RNN for name generation.

    The forward pass receives a category one-hot, a character one-hot,
    and the previous hidden state and outputs log-probabilities over the
    next character (including EOS).

    Parameters
    ----------
    input_size : int
        Character vocabulary size (``N_LETTERS`` = 58).
    hidden_size : int
        Hidden state dimensionality.
    output_size : int
        Output vocabulary size (same as ``N_LETTERS``).
    n_categories : int
        Number of nationality categories.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_categories: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        combined_size = n_categories + input_size + hidden_size
        self.i2h = nn.Linear(combined_size, hidden_size)
        self.i2o = nn.Linear(combined_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        category: torch.Tensor,
        x: torch.Tensor,
        hidden: torch.Tensor,
    ):
        """
        Parameters
        ----------
        category : Tensor, shape ``(1, n_categories)``
        x : Tensor, shape ``(1, input_size)``
        hidden : Tensor, shape ``(1, hidden_size)``

        Returns
        -------
        output : Tensor, shape ``(1, output_size)``
        hidden : Tensor, shape ``(1, hidden_size)``
        """
        combined = torch.cat((category, x, hidden), dim=1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output_combined = torch.cat((hidden, output), dim=1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(1, self.hidden_size)


# =============================================================================
# 4. Encoding utilities
# =============================================================================
def category_tensor(category: str, all_categories: list[str]) -> torch.Tensor:
    """One-hot vector for a category, shape ``(1, n_categories)``."""
    idx = all_categories.index(category)
    tensor = torch.zeros(1, len(all_categories))
    tensor[0][idx] = 1
    return tensor


def input_tensor(line: str) -> torch.Tensor:
    """One-hot matrix for input characters (excluding EOS), shape
    ``(len(line), 1, N_LETTERS)``."""
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, ch in enumerate(line):
        tensor[i][0][ALL_LETTERS.find(ch)] = 1
    return tensor


def target_tensor(line: str) -> torch.LongTensor:
    """Index vector: characters shifted by one, with EOS appended.
    Shape ``(len(line),)``."""
    indices = [ALL_LETTERS.find(line[i]) for i in range(1, len(line))]
    indices.append(N_LETTERS - 1)  # EOS
    return torch.LongTensor(indices)


def random_choice(lst):
    return lst[random.randint(0, len(lst) - 1)]


def random_training_example(all_categories, category_lines):
    cat = random_choice(all_categories)
    line = random_choice(category_lines[cat])
    cat_t = category_tensor(cat, all_categories)
    inp_t = input_tensor(line)
    tgt_t = target_tensor(line)
    return cat_t, inp_t, tgt_t


# =============================================================================
# 5. Training
# =============================================================================
def train_step(
    rnn: RNN,
    criterion: nn.Module,
    cat_t: torch.Tensor,
    inp_t: torch.Tensor,
    tgt_t: torch.Tensor,
):
    tgt_t.unsqueeze_(-1)
    hidden = rnn.init_hidden()
    rnn.zero_grad()

    loss = torch.tensor(0.0)
    for i in range(inp_t.size(0)):
        output, hidden = rnn(cat_t, inp_t[i], hidden)
        loss += criterion(output, tgt_t[i])

    loss.backward()

    # Manual SGD
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-LEARNING_RATE)

    return output, loss.item() / inp_t.size(0)


def time_since(since: float) -> str:
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {int(s)}s"


def run_train_loop(rnn, criterion, all_categories, category_lines):
    current_loss = 0.0
    all_losses: list[float] = []
    iters: list[int] = []
    start = time.time()

    for it in range(1, N_ITERS + 1):
        cat_t, inp_t, tgt_t = random_training_example(
            all_categories, category_lines
        )
        output, loss = train_step(rnn, criterion, cat_t, inp_t, tgt_t)
        current_loss += loss

        if it % PRINT_EVERY == 0:
            pct = it / N_ITERS * 100
            print(f"{time_since(start)} ({it} {pct:.0f}%) {loss:.4f}")

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
# 6. Name generation (autoregressive sampling)
# =============================================================================
def sample(
    model: RNN,
    category: str,
    all_categories: list[str],
    start_letter: str = "A",
) -> str:
    """Generate a single name conditioned on *category* and *start_letter*."""
    with torch.no_grad():
        cat_t = category_tensor(category, all_categories)
        inp = input_tensor(start_letter)
        hidden = model.init_hidden()

        output_name = start_letter

        for _ in range(MAX_LENGTH):
            output, hidden = model(cat_t, inp[0], hidden)
            _, topi = output.topk(1)
            topi = topi[0][0]
            if topi == N_LETTERS - 1:
                break
            letter = ALL_LETTERS[topi]
            output_name += letter
            inp = input_tensor(letter)

    return output_name


def samples(
    model: RNN,
    category: str,
    all_categories: list[str],
    start_letters: str = "ABC",
):
    """Generate and print names for each starting letter."""
    for ch in start_letters:
        print(f"  {sample(model, category, all_categories, ch)}")


# =============================================================================
# 7. Main
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
    model = RNN(N_LETTERS, N_HIDDEN, N_LETTERS, n_categories)
    criterion = nn.NLLLoss()
    print(model)

    run_train_loop(model, criterion, all_categories, category_lines)

    # --- Save & reload for generation demo ----------------------------------
    torch.save(model.state_dict(), "generation_weights.pth")

    model2 = RNN(N_LETTERS, N_HIDDEN, N_LETTERS, n_categories)
    model2.load_state_dict(torch.load("generation_weights.pth"))
    model2.eval()

    print("\n--- Russian names ---")
    samples(model2, "Russian", all_categories, "RUS")
    print("\n--- German names ---")
    samples(model2, "German", all_categories, "GER")
    print("\n--- Spanish names ---")
    samples(model2, "Spanish", all_categories, "SPA")
    print("\n--- Chinese names ---")
    samples(model2, "Chinese", all_categories, "CHI")


if __name__ == "__main__":
    main()
