"""rnn_baseline — rnn_baseline 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch10/training_and_inference/rnn_baseline.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

import torch.nn as nn


class RNNBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, num_classes=10):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.rnn(x)
        return self.classifier(hidden[-1])


if __name__ == "__main__":
    pass
