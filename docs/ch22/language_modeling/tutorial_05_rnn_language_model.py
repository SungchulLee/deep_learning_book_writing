"""tutorial_05_rnn_language_model — tutorial_05_rnn_language_model 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch20/language_modeling/tutorial_05_rnn_language_model.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

"""
길잡이 05: 되돌이 그물 말 모델

수학적 바탕:
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
P(w_t | w_1,...,w_{t-1}) = softmax(y_t)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# ========================================================================
# 메인
# ========================================================================


class RNNLanguageModel(nn.Module):
    """맥락 길이가 바뀌는 되돌이 그물 바탕 말 모델."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 hidden_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super(RNNLanguageModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            embedding_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        embeds = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embeds, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden
    
    def init_hidden(self, batch_size: int):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)


if __name__ == "__main__":
    print("RNN Language Model: handles variable-length sequences")
    print("Challenge: vanishing/exploding gradients")


class RNNDataset(Dataset):
    """말 모델용 자료. 글월을 max_seq_len 길이의 토막으로 자른다.

    말 모델이 배우는 것은 "다음 낱말 맞히기"이므로, 들임과 정답이 한 칸씩
    어긋난 같은 토막이다.

        토막      the  cat  sat  on   the  mat
        들임 x    the  cat  sat  on   the
        정답 y         cat  sat  on   the  mat

    Args:
        corpus: 토큰의 열. 낱말 문자열이어도 되고 이미 번호로 바꾼 것이어도 된다.
        vocab:  낱말 -> 번호 사전. 문자열 corpus를 번호로 바꿀 때 쓴다.
        max_seq_len: 토막 하나의 길이.
    """

    def __init__(self, corpus, vocab, max_seq_len: int = 50):
        self.vocab = vocab
        self.max_seq_len = max_seq_len

        if len(corpus) and isinstance(corpus[0], str):
            unk = vocab.get("<unk>", 0) if hasattr(vocab, "get") else 0
            ids = [vocab.get(t, unk) if hasattr(vocab, "get") else vocab[t] for t in corpus]
        else:
            ids = list(corpus)

        # 정답이 한 칸 뒤이므로 토막마다 max_seq_len + 1개가 필요하다
        step = max_seq_len
        self.chunks = [ids[i:i + max_seq_len + 1]
                       for i in range(0, max(len(ids) - max_seq_len, 0), step)]
        self.chunks = [c for c in self.chunks if len(c) >= 2]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def collate_fn(batch, pad_value: int = 0):
    """길이가 다른 토막들을 한 배치로 묶는다. 짧은 것은 뒤를 채운다."""
    xs, ys = zip(*batch)
    return (pad_sequence(xs, batch_first=True, padding_value=pad_value),
            pad_sequence(ys, batch_first=True, padding_value=pad_value))
