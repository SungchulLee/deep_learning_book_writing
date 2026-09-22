"""transformer_model — transformer_model 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch10/training_and_inference/transformer_model.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

import torch
import torch.nn as nn


class TransformerForComparison(nn.Module):
    def __init__(self, input_dim, d_model=256, num_heads=8,
                 num_layers=6, num_classes=10):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=d_model * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.mean(dim=0)  # 전역 평균 풀링
        return self.classifier(x)


if __name__ == "__main__":
    pass
