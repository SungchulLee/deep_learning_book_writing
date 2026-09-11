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
