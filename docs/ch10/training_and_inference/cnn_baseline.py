"""cnn_baseline — cnn_baseline 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch10/training_and_inference/cnn_baseline.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

import torch.nn as nn


class CNNBaseline(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, L]
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        return self.classifier(x)


if __name__ == "__main__":
    pass
