"""dcgan — DCGAN의 생성기와 판별기.

DCGAN 논문(Radford et al., 2016)이 정한 규칙을 따른다.

- 풀링을 쓰지 않는다. 생성기는 stride를 준 전치 합성곱으로 키우고,
  판별기는 stride를 준 합성곱으로 줄인다.
- 두 그물 모두 배치 정규화를 쓰되 **생성기의 마지막 층과 판별기의 첫 층에는
  넣지 않는다.** 넣으면 표본이 한곳으로 몰린다.
- 생성기는 ReLU와 마지막에 tanh, 판별기는 LeakyReLU(0.2)를 쓴다.

64x64로 내놓는다. MNIST를 쓸 때는 자료를 64로 키워 넣는다.
"""

import torch
import torch.nn as nn


class DCGANGenerator(nn.Module):
    """숨은 벡터 -> 64x64 그림.

    Args:
        latent_dim: 숨은 벡터의 차원
        image_channels: 내놓을 그림의 채널 수 (MNIST는 1, 색 그림은 3)
        feature_maps: 마지막 층의 결 지도 수. 안쪽은 이 값의 배수로 잡는다.
    """

    def __init__(self, latent_dim: int = 100, image_channels: int = 1,
                 feature_maps: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        f = feature_maps
        self.net = nn.Sequential(
            # (latent_dim, 1, 1) -> (f*8, 4, 4)
            nn.ConvTranspose2d(latent_dim, f * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(f * 8), nn.ReLU(True),
            # -> (f*4, 8, 8)
            nn.ConvTranspose2d(f * 8, f * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 4), nn.ReLU(True),
            # -> (f*2, 16, 16)
            nn.ConvTranspose2d(f * 4, f * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 2), nn.ReLU(True),
            # -> (f, 32, 32)
            nn.ConvTranspose2d(f * 2, f, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f), nn.ReLU(True),
            # -> (image_channels, 64, 64). 마지막에는 배치 정규화가 없다
            nn.ConvTranspose2d(f, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:                       # (B, latent_dim) -> (B, latent_dim, 1, 1)
            z = z[:, :, None, None]
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    """64x64 그림 -> 진짜일 로짓 하나."""

    def __init__(self, image_channels: int = 1, feature_maps: int = 64):
        super().__init__()
        f = feature_maps
        self.net = nn.Sequential(
            # 첫 층에는 배치 정규화가 없다
            nn.Conv2d(image_channels, f, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f, f * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f * 2, f * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f * 4, f * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)            # 로짓. 시그모이드는 손실 쪽에서


def weights_init(m: nn.Module) -> None:
    """DCGAN 논문이 정한 첫자리매김: 평균 0, 표준편차 0.02의 정규분포."""
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
