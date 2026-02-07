"""
Sequence-to-Sequence with Bahdanau Attention — French→English Translation
==========================================================================

An encoder–decoder architecture with additive (Bahdanau) attention for
translating short French sentences into English.

Source: https://github.com/pytorch/tutorials/blob/main/intermediate_source/seq2seq_translation_tutorial.py

Architecture
------------

**Encoder** (``EncoderRNN``):
    Embedding → GRU  →  all hidden states  +  final hidden state

**Attention** (``BahdanauAttention``):
    score(s_t, h_j) = V_a^T  tanh(W_a s_t  +  U_a h_j)
    α_{t,j}        = softmax_j(score)
    context_t      = Σ_j  α_{t,j}  h_j

**Decoder** (``AttnDecoderRNN``):
    Embedding → concat(embedded, context) → GRU → Linear → LogSoftmax

Teacher forcing is used with probability ``TEACHER_FORCING_RATIO`` during
training.

Pipeline
--------
1.  **download.py** — fetch ``data.zip`` containing ``eng-fra.txt``
2.  **global_name_space.py** — ``Lang`` class, text normalisation,
    pair filtering, constants
3.  **model.py** — Encoder, Decoder (with and without attention),
    Bahdanau Attention
4.  **load_data.py** — sentence→tensor conversion, DataLoader creation
5.  **main.py** — training loop, evaluation, attention visualisation

All modules are merged into this single file for portability.

Run
---
    python seq2seq_attention.py
"""

# =============================================================================
# Imports
# =============================================================================
import math
import os
import random
import re
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
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# 1. Download
# =============================================================================
def download_data() -> None:
    if not os.path.exists("./data"):
        url = "https://download.pytorch.org/tutorial/data.zip"
        response = requests.get(url)
        zip_path = "data.zip"
        with open(zip_path, "wb") as f:
            f.write(response.content)
        with ZipFile(zip_path, "r") as zf:
            zf.extractall("./")
        os.remove(zip_path)
    print("Data ready.")


# =============================================================================
# 2. Global namespace — Lang, normalisation, pair filtering, constants
# =============================================================================
SOS_TOKEN = 0
EOS_TOKEN = 1
MAX_LENGTH = 10

ENG_PREFIXES = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re ",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 128
BATCH_SIZE = 32
TEACHER_FORCING_RATIO = 0.5


class Lang:
    """Vocabulary manager: word ↔ index mappings."""

    def __init__(self, name: str):
        self.name = name
        self.word2index: dict[str, int] = {}
        self.word2count: dict[str, int] = {}
        self.index2word: dict[int, str] = {0: "SOS", 1: "EOS"}
        self.n_words: int = 2  # SOS + EOS

    def add_sentence(self, sentence: str) -> None:
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word: str) -> None:
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def __repr__(self) -> str:
        return f"Lang({self.name})"


# --- Text normalisation -----------------------------------------------------
def unicode_to_ascii(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalize_string(s: str) -> str:
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# --- Read and filter pairs ---------------------------------------------------
def read_langs(lang1: str, lang2: str, reverse: bool = False):
    path = f"data/{lang1}-{lang2}.txt"
    with open(path, encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    pairs = [[normalize_string(s) for s in line.split("\t")] for line in lines]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


def filter_pair(p) -> bool:
    return (
        len(p[0].split(" ")) < MAX_LENGTH
        and len(p[1].split(" ")) < MAX_LENGTH
        and p[1].startswith(ENG_PREFIXES)
    )


def filter_pairs(pairs):
    return [p for p in pairs if filter_pair(p)]


def prepare_data(lang1: str, lang2: str, reverse: bool = True):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    pairs = filter_pairs(pairs)
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print(f"  Pairs: {len(pairs)}")
    print(f"  {input_lang.name} vocab: {input_lang.n_words}")
    print(f"  {output_lang.name} vocab: {output_lang.n_words}")
    return input_lang, output_lang, pairs


# =============================================================================
# 3. Data loading — sentence → tensor, DataLoader
# =============================================================================
def indexes_from_sentence(lang: Lang, sentence: str) -> list[int]:
    return [lang.word2index[w] for w in sentence.split(" ")]


def tensor_from_sentence(lang: Lang, sentence: str) -> torch.Tensor:
    indices = indexes_from_sentence(lang, sentence)
    indices.append(EOS_TOKEN)
    return torch.tensor(indices, dtype=torch.long, device=DEVICE)


def get_dataloader(
    input_lang: Lang,
    output_lang: Lang,
    pairs: list,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for i, (inp, tgt) in enumerate(pairs):
        inp_idx = indexes_from_sentence(input_lang, inp)
        tgt_idx = indexes_from_sentence(output_lang, tgt)
        inp_idx.append(EOS_TOKEN)
        tgt_idx.append(EOS_TOKEN)
        input_ids[i, : len(inp_idx)] = inp_idx
        target_ids[i, : len(tgt_idx)] = tgt_idx

    dataset = TensorDataset(
        torch.LongTensor(input_ids).to(DEVICE),
        torch.LongTensor(target_ids).to(DEVICE),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# =============================================================================
# 4. Models
# =============================================================================
class EncoderRNN(nn.Module):
    """GRU encoder.  Processes an input sequence token-by-token and returns
    all hidden states plus the final hidden state.

    Parameters
    ----------
    input_size : int   — source vocabulary size
    hidden_size : int  — GRU hidden dimensionality
    dropout_p : float  — embedding dropout probability
    """

    def __init__(self, input_size: int, hidden_size: int, dropout_p: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : LongTensor, shape ``(B, T)``

        Returns
        -------
        all_hidden : Tensor, shape ``(B, T, H)``
        last_hidden : Tensor, shape ``(1, B, H)``
        """
        hidden = None
        all_hidden_list = []

        for t in range(x.size(1)):
            embedded = self.dropout(self.embedding(x[:, t].unsqueeze(1)))
            output, hidden = self.gru(embedded, hidden)
            all_hidden_list.append(hidden.squeeze(0).unsqueeze(1))

        all_hidden = torch.cat(all_hidden_list, dim=1)  # (B, T, H)
        return all_hidden, hidden  # (B,T,H), (1,B,H)


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention.

    score(s, h) = V^T tanh(W_a s + U_a h)

    Parameters
    ----------
    hidden_size : int
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        """
        Parameters
        ----------
        query : Tensor, shape ``(B, 1, H)`` — decoder hidden state
        keys  : Tensor, shape ``(B, T, H)`` — all encoder hidden states

        Returns
        -------
        context : Tensor, shape ``(B, 1, H)``
        weights : Tensor, shape ``(B, 1, T)``
        """
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))  # (B,T,1)
        scores = scores.squeeze(2).unsqueeze(1)  # (B,1,T)
        weights = F.softmax(scores, dim=-1)       # (B,1,T)
        context = torch.bmm(weights, keys)        # (B,1,H)
        return context, weights


class AttnDecoderRNN(nn.Module):
    """GRU decoder with Bahdanau attention.

    At each time-step the decoder:
    1. Embeds the previous token.
    2. Computes attention over encoder hidden states.
    3. Concatenates embedding and context → feeds into GRU.
    4. Projects GRU output to vocabulary.

    Parameters
    ----------
    hidden_size : int   — GRU hidden dimensionality
    output_size : int   — target vocabulary size
    dropout_p : float   — embedding dropout probability
    """

    def __init__(self, hidden_size: int, output_size: int, dropout_p: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        all_encoder_hidden: torch.Tensor,
        target_tensor: torch.Tensor | None = None,
    ):
        """
        Parameters
        ----------
        encoder_hidden : Tensor, shape ``(1, B, H)``  — last encoder hidden
        all_encoder_hidden : Tensor, shape ``(B, T, H)`` — all encoder states
        target_tensor : LongTensor | None, shape ``(B, T)`` — for teacher forcing

        Returns
        -------
        decoder_outputs : Tensor, shape ``(B, T, V)`` — log-probs
        decoder_hidden  : Tensor, shape ``(1, B, H)``
        attentions      : Tensor, shape ``(B, T, T_enc)``
        """
        batch_size = all_encoder_hidden.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=DEVICE
        ).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden

        decoder_outputs = []
        attentions = []

        for t in range(MAX_LENGTH):
            dec_out, decoder_hidden, attn_w = self._step(
                decoder_input, decoder_hidden, all_encoder_hidden
            )
            decoder_outputs.append(dec_out)
            attentions.append(attn_w)

            if target_tensor is not None and random.random() < TEACHER_FORCING_RATIO:
                decoder_input = target_tensor[:, t].unsqueeze(1)
            else:
                _, topi = dec_out.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)
        return decoder_outputs, decoder_hidden, attentions

    def _step(self, x, hidden, encoder_hidden):
        embedded = self.dropout(self.embedding(x))            # (B,1,H)
        query = hidden.permute(1, 0, 2)                       # (B,1,H)
        context, attn_w = self.attention(query, encoder_hidden)  # (B,1,H), (B,1,T)
        gru_in = torch.cat((embedded, context), dim=2)        # (B,1,2H)
        output, hidden = self.gru(gru_in, hidden)             # (B,1,H), (1,B,H)
        output = self.out(output)                              # (B,1,V)
        return output, hidden, attn_w


# =============================================================================
# 5. Training
# =============================================================================
def as_minutes(s: float) -> str:
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {int(s)}s"


def time_since(since: float, percent: float) -> str:
    now = time.time()
    elapsed = now - since
    estimated = elapsed / percent
    remaining = estimated - elapsed
    return f"{as_minutes(elapsed)} (- {as_minutes(remaining)})"


def train_epoch(dataloader, encoder, decoder, enc_opt, dec_opt, criterion):
    total_loss = 0.0
    for input_t, target_t in dataloader:
        enc_opt.zero_grad()
        dec_opt.zero_grad()

        all_hidden, last_hidden = encoder(input_t)
        dec_out, _, _ = decoder(last_hidden, all_hidden, target_t)

        loss = criterion(
            dec_out.view(-1, dec_out.size(-1)),
            target_t.view(-1),
        )
        loss.backward()

        enc_opt.step()
        dec_opt.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(
    dataloader: DataLoader,
    encoder: EncoderRNN,
    decoder: AttnDecoderRNN,
    n_epochs: int = 80,
    lr: float = 0.001,
    print_every: int = 5,
    plot_every: int = 5,
):
    start = time.time()
    plot_losses: list[float] = []
    print_loss_total = 0.0
    plot_loss_total = 0.0

    enc_opt = optim.Adam(encoder.parameters(), lr=lr)
    dec_opt = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(dataloader, encoder, decoder, enc_opt, dec_opt, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            avg = print_loss_total / print_every
            print_loss_total = 0.0
            pct = epoch / n_epochs
            print(f"{time_since(start, pct)} (epoch {epoch} {pct*100:.0f}%) loss={avg:.4f}")

        if epoch % plot_every == 0:
            plot_losses.append(plot_loss_total / plot_every)
            plot_loss_total = 0.0

    # --- Plot loss curve -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(plot_losses)
    ax.set_xlabel(f"Every {plot_every} epochs")
    ax.set_ylabel("Avg Loss")
    ax.set_title("Training Loss")
    plt.tight_layout()
    plt.show()


# =============================================================================
# 6. Evaluation & visualisation
# =============================================================================
def evaluate_sentence(
    encoder: EncoderRNN,
    decoder: AttnDecoderRNN,
    sentence: str,
    input_lang: Lang,
    output_lang: Lang,
):
    """Translate a single source sentence and return decoded words + attention."""
    with torch.no_grad():
        inp_t = tensor_from_sentence(input_lang, sentence).unsqueeze(0)
        # Pad to MAX_LENGTH
        padded = torch.zeros(1, MAX_LENGTH, dtype=torch.long, device=DEVICE)
        padded[0, : inp_t.size(1)] = inp_t

        all_hidden, last_hidden = encoder(padded)
        dec_out, _, attentions = decoder(last_hidden, all_hidden)

        decoded_words = []
        for step in range(MAX_LENGTH):
            _, topi = dec_out[:, step, :].topk(1)
            idx = topi[0, 0].item()
            if idx == EOS_TOKEN:
                decoded_words.append("<EOS>")
                break
            decoded_words.append(output_lang.index2word.get(idx, "<UNK>"))

        if "<EOS>" not in decoded_words:
            decoded_words.append("<EOS>")

    return decoded_words, attentions


def evaluate_randomly(encoder, decoder, pairs, input_lang, output_lang, n=10):
    for _ in range(n):
        pair = random.choice(pairs)
        print(f"> {pair[0]}")
        print(f"= {pair[1]}")
        words, _ = evaluate_sentence(encoder, decoder, pair[0], input_lang, output_lang)
        print(f"< {' '.join(words)}\n")


def show_attention(input_sentence, output_words, attentions):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    n_out = len(output_words)
    n_in = len(input_sentence.split(" ")) + 1  # +1 for <EOS>
    cax = ax.matshow(attentions[0, :n_out, :n_in].cpu().numpy(), cmap="bone")
    fig.colorbar(cax)

    ax.set_xticks(range(n_in))
    ax.set_yticks(range(n_out))
    ax.set_xticklabels(input_sentence.split(" ") + ["<EOS>"], rotation=90)
    ax.set_yticklabels(output_words)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.tight_layout()
    plt.show()


def evaluate_and_show_attention(encoder, decoder, sentence, input_lang, output_lang):
    words, attentions = evaluate_sentence(
        encoder, decoder, sentence, input_lang, output_lang
    )
    print(f"input  = {sentence}")
    print(f"output = {' '.join(words)}")
    show_attention(sentence, words, attentions)


# =============================================================================
# 7. Main
# =============================================================================
def main():
    # --- Data ----------------------------------------------------------------
    download_data()
    input_lang, output_lang, pairs = prepare_data("eng", "fra")
    print(f"Example pair: {random.choice(pairs)}")

    dataloader = get_dataloader(input_lang, output_lang, pairs)

    # --- Models --------------------------------------------------------------
    encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(DEVICE)
    decoder = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words).to(DEVICE)
    print(encoder)
    print(decoder)

    # --- Train ---------------------------------------------------------------
    train_model(
        dataloader, encoder, decoder,
        n_epochs=80, lr=0.001, print_every=5, plot_every=5,
    )

    # --- Save ----------------------------------------------------------------
    os.makedirs("model", exist_ok=True)
    torch.save(encoder.state_dict(), "model/encoder_weights.pth")
    torch.save(decoder.state_dict(), "model/decoder_weights.pth")

    # --- Reload & evaluate ---------------------------------------------------
    encoder2 = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(DEVICE)
    decoder2 = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words).to(DEVICE)
    encoder2.load_state_dict(torch.load("model/encoder_weights.pth"))
    decoder2.load_state_dict(torch.load("model/decoder_weights.pth"))
    encoder2.eval()
    decoder2.eval()

    evaluate_randomly(encoder2, decoder2, pairs, input_lang, output_lang)

    test_sentences = [
        "il n est pas aussi grand que son pere",
        "je suis trop fatigue pour conduire",
        "je suis desole si c est une question idiote",
        "je suis reellement fiere de vous",
    ]
    for s in test_sentences:
        evaluate_and_show_attention(encoder2, decoder2, s, input_lang, output_lang)


if __name__ == "__main__":
    main()
