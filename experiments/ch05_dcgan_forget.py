"""DCGAN 논문의 「그리는 법을 잊게 만들기」를 이 책의 생성기로 해 본다.

그 글(arXiv:1511.06434) 6.3.1절은 침실 그림에서 **창문**을 지웠다. 둘째로 높은
컨볼루션 레이어의 활성이 창문 상자 안에 있는지를 맞히는 로지스틱 회귀를 맞추고,
웨이트가 양수인 피처 맵 200개를 모든 자리에서 꺼 버린 뒤 다시 뽑았더니
「네트워크가 창문 그리는 것을 거의 잊었다」고 적는다.

여기서는 지울 것이 창문이 아니라 **숫자 0**이다. 생성기의 둘째로 높은 레이어가
내놓는 피처 맵 64개를 공간 평균 내어, 「평가 분류기가 0이라 읽는가」를 맞히는
로지스틱 회귀를 맞춘다. 그리고 웨이트가 큰 쪽부터 K개를 끈다.

**대조군이 핵심이다.** 맵을 끄면 그림이 그냥 망가지면서 어떤 숫자든 줄어들 수
있으므로, 같은 수의 맵을 **무작위로** 끈 것과 견주지 않으면 아무 말도 할 수 없다.

읽는 것: mnist_dcgan_g.pt, mnist_judge.pt  (둘 다 저장소 뿌리)
실행: python experiments/ch05_dcgan_forget.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED, ZDIM, N = 42, 64, 5000
JM, JS = 0.1307, 0.3081
TARGET = 0


class GConv(nn.Module):
    """02_dcgan.md 의 생성기 그대로."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ZDIM, 128 * 7 * 7),
                                nn.BatchNorm1d(128 * 7 * 7), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(),                  # <- 여기가 그 레이어
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        return self.net(self.fc(z).reshape(-1, 128, 7, 7))


class JudgeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2); self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128); self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return self.fc2(self.dropout(torch.relu(self.fc1(x.flatten(1)))))


G = GConv()
G.load_state_dict(torch.load("mnist_dcgan_g.pt", map_location="cpu",
                             weights_only=True))
G.eval()
J = JudgeCNN()
J.load_state_dict(torch.load("mnist_judge.pt", map_location="cpu",
                             weights_only=True))
J.eval()


@torch.no_grad()
def read(x_pm1):
    xj = (((x_pm1.reshape(-1, 1, 28, 28) + 1) / 2) - JM) / JS
    return torch.softmax(J(xj), 1).max(1)


@torch.no_grad()
def decode(act):
    return torch.cat([G.net[3:](act[i:i + 1000]) for i in range(0, len(act), 1000)])


def fit_logreg(x, y, steps=600):
    """공간 평균 낸 피처 맵으로 「이 그림이 TARGET 인가」를 맞춘다."""
    xs = (x - x.mean(0)) / x.std(0)
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(xs @ w + b, y).backward()
        opt.step()
    return w.detach()


if __name__ == "__main__":
    z = torch.randn(N, ZDIM, generator=torch.Generator().manual_seed(SEED))
    with torch.no_grad():
        act = torch.cat([G.net[:3](G.fc(z[i:i + 1000]).reshape(-1, 128, 7, 7))
                         for i in range(0, N, 1000)])
    img = decode(act)
    conf0, cls0 = read(img)
    base = 100 * (cls0 == TARGET).float().mean()
    print(f"  피처 맵 {act.shape[1]}개 x {act.shape[2]}x{act.shape[3]}")
    print(f"  끄기 전 {TARGET} 의 몫 {base:.1f}%   확신도 {conf0.mean():.3f}\n")

    w = fit_logreg(act.mean((2, 3)), (cls0 == TARGET).float())
    npos = int((w > 0).sum())

    def kill(idx):
        a = act.clone(); a[:, idx] = 0
        im = decode(a)
        c, l = read(im)
        return 100 * (l == TARGET).float().mean(), c.mean()

    print(f"  {'끈 맵':>12}{'고른 것':>12}{'무작위':>12}{'확신도':>12}")
    rows = [("상위 4", 4), ("상위 8", 8), ("상위 16", 16),
            (f"양수 전부({npos})", npos)]
    for name, k in rows:
        picked, cp = kill(w.argsort(descending=True)[:k])
        g = torch.Generator().manual_seed(1)
        rand, _ = kill(torch.randperm(act.shape[1], generator=g)[:k])
        print(f"  {name:>12}{picked:>11.1f}%{rand:>11.1f}%{cp:>12.3f}")

    print("\n  고른 쪽과 무작위 쪽이 갈리지 않으면 「잊게 만들었다」고 할 수 없다.")
