"""cnn_vs_vit — cnn_vs_vit 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch10/transformers_vision/cnn_vs_vit.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """견주기 위한 전통적인 합성곱 신경망 구조."""
    def __init__(self, n_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            # 블록 1: 224 → 112
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 블록 2: 112 → 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 블록 3: 56 → 28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 블록 4: 28 → 14
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class HybridCNNViT(nn.Module):
    """합성곱 밑동 + 트랜스포머 — 둘을 섞은 모형.

    비전 트랜스포머는 그림을 16x16 조각으로 잘라 곧바로 선형 사영한다.
    조각 안쪽을 볼 장치가 없으므로 가장자리나 결 같은 낮은 수준의 무늬를
    처음부터 어텐션으로 배워야 하고, 그래서 자료가 많이 든다.

    섞은 모형은 앞쪽 몇 겹을 합성곱으로 둔다. 합성곱이 낮은 수준의 무늬를
    맡고, 그 위에서 어텐션이 먼 관계를 맡는다. 조각을 자르는 일도
    합성곱의 stride가 대신하므로 따로 자를 필요가 없다.

    224x224 -> (stride 4, 4번) -> 14x14 = 196 토큰, ViT-Base와 같은 수다.
    """

    def __init__(self, n_classes: int = 10, embed_dim: int = 192,
                 depth: int = 4, num_heads: int = 3):
        super().__init__()
        # 합성곱 밑동. 네 번 반으로 줄여 224 -> 14
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, embed_dim, 3, stride=2, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 14 * 14 + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.stem(x)                       # (B, embed_dim, 14, 14)
        x = x.flatten(2).transpose(1, 2)       # (B, 196, embed_dim) — 조각 자르기를 대신한다
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed[:, : x.shape[1] + 1]
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))   # 맨 앞 토큰 하나로 가른다
