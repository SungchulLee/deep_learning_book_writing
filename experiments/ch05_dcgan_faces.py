"""64x64 얼굴로 DCGAN 을 돌려 본다 — 논문이 쓴 크기 그대로.

`docs/ch05/generative/02_dcgan.md` 는 28x28 흑백 MNIST 로 DCGAN 을 보인다. 그
쪽 2 절이 「널리 쓰이는 64x64 DCGAN 과 이 쪽의 28x28 을 나란히」 놓고 눈금이
열 배 다르다고 적었는데, 그러면 **눈금을 맞추면 달라지는가**가 남는다.
이 스크립트가 그 물음을 맡는다.

자료: LFW (Labeled Faces in the Wild). sklearn 이 내려받아 주는 13,233 장을
      얼굴 둘레로 잘라 64x64 컬러로 쓴다. 좌우 뒤집기로 두 배 불린다.
      LFW 는 연구 목적 공개 자료이고, 원본 사진은 이 저장소에 담지 않는다 —
      돌릴 때마다 sklearn 이 ./data/sklearn 에 받아 둔다.

구조: 02_dcgan.md 2 절의 대조표에서 「널리 쓰이는」 쪽 그대로다.
      생성기 z 100 -> 512x4x4 -> ... -> 3x64x64, 출력만 tanh 이고 거기엔
      배치 정규화를 걸지 않는다. 판별기는 그 거울이고 **입력 레이어에만**
      배치 정규화를 걸지 않는다. 웨이트는 N(0, 0.02) 로 초기화한다.
      곧 그 표의 아홉 줄을 이번에는 **아홉 줄 다** 지킨다.

실행: python experiments/ch05_dcgan_faces.py            (저장소 뿌리에서)
      EPOCHS 환경변수로 에포크를 바꿀 수 있다.
남기는 것: lfw_dcgan_g.pt (생성기), lfw_64.pt (잘라 둔 자료 캐시)
"""

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SEED, BATCH, ZDIM = 42, 128, 100
# 책에 실린 그림은 **20 에포크**에서 나왔다. 120 을 돌릴 작정이었는데 10 에포크
# 까지는 95초/에포크이던 것이 기계가 스왑을 쓰기 시작하면서 19분/에포크까지
# 늘어져 20 에서 끊었다. 메모리가 넉넉하면 더 돌리는 편이 낫다.
EPOCHS = int(os.environ.get("EPOCHS", 20))
LR, BETAS = 2e-4, (0.5, 0.999)
CACHE = Path("lfw_64.pt")
OUT = Path("lfw_dcgan_g.pt")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_faces():
    """LFW 를 64x64 컬러로. 한 번 받아 두면 캐시에서 읽는다."""
    if CACHE.exists():
        return torch.load(CACHE, weights_only=True)
    from sklearn.datasets import fetch_lfw_people
    d = fetch_lfw_people(color=True, resize=0.5,
                         slice_=(slice(61, 189), slice(61, 189)),
                         data_home="./data/sklearn", download_if_missing=True)
    x = torch.from_numpy(d.images).permute(0, 3, 1, 2).contiguous()   # (N,3,64,64)
    x = (x * 255).round().clamp(0, 255).to(torch.uint8)
    torch.save(x, CACHE)
    return x


# === 구조 — 논문의 64x64 그대로 ==============================================
class G64(nn.Module):
    def __init__(self):
        super().__init__()
        def up(i, o, bn=True):
            layers = [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False)]
            if bn:
                layers += [nn.BatchNorm2d(o), nn.ReLU(True)]
            return layers
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ZDIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),            # 4x4
            *up(512, 256), *up(256, 128), *up(128, 64),     # 8, 16, 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),  # 64 — BN 없음
            nn.Tanh())

    def forward(self, z):
        return self.net(z.reshape(-1, ZDIM, 1, 1))


class D64(nn.Module):
    def __init__(self):
        super().__init__()
        def down(i, o, bn=True):
            layers = [nn.Conv2d(i, o, 4, 2, 1, bias=False)]
            if bn:
                layers.append(nn.BatchNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.net = nn.Sequential(
            *down(3, 64, bn=False),                         # 입력 레이어엔 BN 없음
            *down(64, 128), *down(128, 256), *down(256, 512),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False))

    def forward(self, x):
        return self.net(x).reshape(-1, 1)


def weights_init(m):
    """논문 4 절. 이름이 아니라 타입으로 가린다."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    X = load_faces()
    X = torch.cat([X, X.flip(-1)])                     # 좌우 뒤집기로 두 배
    print(f"  자료 {len(X):,}장 x 3 x 64 x 64   장치 {device}", flush=True)

    torch.manual_seed(SEED)
    G, D = G64().to(device), D64().to(device)
    G.apply(weights_init); D.apply(weights_init)
    print(f"  생성기 {sum(p.numel() for p in G.parameters()):,}개  "
          f"판별기 {sum(p.numel() for p in D.parameters()):,}개", flush=True)

    oG = optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    oD = optim.Adam(D.parameters(), lr=LR, betas=BETAS)
    bce = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(X), batch_size=BATCH, shuffle=True,
                        generator=g, drop_last=True)

    t0 = time.time()
    for ep in range(EPOCHS):
        G.train(); D.train(); dl = gl = nb = 0.0
        for (xb,) in loader:
            xb = (xb.to(device).float() / 127.5) - 1.0     # [-1, 1]
            n = xb.size(0)
            ones = torch.ones(n, 1, device=device)
            zeros = torch.zeros(n, 1, device=device)

            fake = G(torch.randn(n, ZDIM, device=device))
            lossD = bce(D(xb), ones) + bce(D(fake.detach()), zeros)
            oD.zero_grad(); lossD.backward(); oD.step()

            lossG = bce(D(fake), ones)
            oG.zero_grad(); lossG.backward(); oG.step()
            dl += lossD.item(); gl += lossG.item(); nb += 1

        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"    {ep+1:3d}/{EPOCHS}  D {dl/nb:.4f}  G {gl/nb:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            torch.save(G.state_dict(), OUT)
    torch.save(G.state_dict(), OUT)
    print(f"  wrote {OUT}  ({time.time()-t0:.0f}s)")
