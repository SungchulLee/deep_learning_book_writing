# DCGAN — 만들기에도 이웃 관계가 있다

[GAN](01_gan.md)은 손실을 바꾸어 선명함을 얻었다. 구조는 여전히 촘촘한 층이었다. DCGAN은 그 구조를 **전치 합성곱**으로 바꾼다.

```python
"""DCGAN. 손실은 그대로 두고 구조만 합성곱으로 바꾼다.

mnist_judge.pt는 5.1절에서 학습해 둔 것을 읽어 쓴다.
GAN은 [-1, 1]로 두고 tanh로 내놓으므로, 심판에 넣기 전에 정규화를 되돌린다.
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

SEED, BATCH, ZDIM, EPOCHS = 42, 100, 64, 30
N_SAMPLE = 5000
JM, JS = 0.1307, 0.3081                 # 심판이 기대하는 정규화
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 이 절만 [-1, 1]을 쓴다 — 생성기가 tanh로 내놓기 때문이다
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5,), (0.5,))])
tr = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
te = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)
Xtr = torch.cat([x for x, _ in DataLoader(tr, batch_size=2000, shuffle=False)])
Xte = torch.cat([x for x, _ in DataLoader(te, batch_size=2000, shuffle=False)])
yte = torch.cat([y for _, y in DataLoader(te, batch_size=2000, shuffle=False)])


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


judge = JudgeCNN().to(device)
judge.load_state_dict(torch.load("mnist_judge.pt", weights_only=True))
judge.eval()


@torch.no_grad()
def predict(x_pm1):
    """[-1,1] -> [0,1] -> 심판 정규화 -> 심판."""
    xj = (((x_pm1 + 1) / 2) - JM) / JS
    p = torch.cat([torch.softmax(judge(xj[i:i + 1000].to(device)), 1).cpu()
                   for i in range(0, len(xj), 1000)])
    conf, pred = p.max(1)
    return conf, pred


def within_class_var(imgs01, group, min_n=20):
    """같은 클래스로 묶인 것들의 화소 분산을, 무리 크기로 가중해 평균낸다."""
    vs, ws = [], []
    for d in range(10):
        g = imgs01[group == d]
        if len(g) < min_n:
            continue
        vs.append(g.flatten(1).var(0).mean().item()); ws.append(len(g))
    return sum(v * w for v, w in zip(vs, ws)) / sum(ws) if vs else 0.0


# 진짜 자료의 기준선 — 여기서는 심판이 아니라 진짜 라벨로 묶는다
REAL_VAR = within_class_var((Xte + 1) / 2, yte)


def evaluate(samples_pm1, tag):
    conf, pred = predict(samples_pm1)
    share = torch.bincount(pred, minlength=10).float()
    share = share / share.sum()
    nz = share[share > 0]
    bal = (-(nz * nz.log()).sum() / torch.tensor(10.0).log()).item()
    wv = within_class_var((samples_pm1 + 1) / 2, pred)
    print(f"  {tag:14s} 확신도 {conf.mean():.3f}  고름 {bal:.3f}  "
          f"최대몫 {share.max():.3f}  클래스안 다양함 {wv / REAL_VAR:.3f}", flush=True)


def train_gan(Gc, Dc, tag):
    torch.manual_seed(SEED)
    G, D = Gc().to(device), Dc().to(device)
    # GAN 표준값. 이 장의 Adam 1e-3으로는 학습되지 않는다
    oG = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    oD = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    gg = torch.Generator().manual_seed(SEED)
    ld = DataLoader(TensorDataset(Xtr), batch_size=BATCH, shuffle=True, generator=gg)
    bce = nn.BCEWithLogitsLoss()
    t0 = time.time()
    for ep in range(EPOCHS):
        G.train(); D.train(); dl = gl = nb = 0.0
        for (xb,) in ld:
            xb = xb.to(device); n = xb.size(0)
            ones = torch.ones(n, 1, device=device)
            zeros = torch.zeros(n, 1, device=device)

            # 판별기: 진짜는 1, 가짜는 0
            fake = G(torch.randn(n, ZDIM, device=device))
            lossD = bce(D(xb), ones) + bce(D(fake.detach()), zeros)
            oD.zero_grad(); lossD.backward(); oD.step()

            # 생성기: 판별기가 진짜라고 하게 만든다
            lossG = bce(D(fake), ones)
            oG.zero_grad(); lossG.backward(); oG.step()

            dl += lossD.item(); gl += lossG.item(); nb += 1
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"    [{tag}] {ep+1:2d}/{EPOCHS}  D {dl/nb:.4f}  G {gl/nb:.4f}", flush=True)
    print(f"  {tag} 완료 ({time.time()-t0:.0f}s)", flush=True)
    return G.eval()


def sample(G, n=N_SAMPLE):
    with torch.no_grad():
        gg = torch.Generator().manual_seed(SEED)
        z = torch.randn(n, ZDIM, generator=gg)
        return torch.cat([G(z[i:i + 1000].to(device)).cpu()
                          for i in range(0, n, 1000)])


# === 촘촘한 층을 합성곱으로 바꾼다 ==========================================
class GConv(nn.Module):
    """작은 특징 맵에서 시작해 전치 합성곱으로 두 번 키운다."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ZDIM, 128 * 7 * 7),
                                nn.BatchNorm1d(128 * 7 * 7), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(),                      # 7 -> 14
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())      # 14 -> 28

    def forward(self, z):
        return self.net(self.fc(z).reshape(-1, 128, 7, 7))


class DConv(nn.Module):
    """3장의 CNN과 하는 일이 거의 같다. 그림을 받아 수 하나를 내놓는다."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),                        # 28 -> 14
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2), # 14 -> 7
            nn.Flatten(), nn.Linear(128 * 7 * 7, 1))

    def forward(self, x):
        return self.net(x)


G = train_gan(GConv, DConv, "dcgan")
evaluate(sample(G), "dcgan")
```

**출력:**

```
    [dcgan]  1/30  D 0.3718  G 2.5620
    [dcgan] 10/30  D 0.3873  G 2.7085
    [dcgan] 20/30  D 0.2937  G 3.1807
    [dcgan] 30/30  D 0.3053  G 3.3621
  dcgan 완료 (948s)
  dcgan          확신도 0.909  고름 0.970  최대몫 0.162  클래스안 다양함 1.085
```

작은 특징 맵에서 시작해 두 번 키워 $28 \times 28$로 만든다. [3장의 3걸음 → 4걸음](../../ch03/mnist/04_cnn.md)과 같은 수법이며, 다만 이번에는 **읽는 쪽이 아니라 그리는 쪽**이다.

판별기도 합성곱으로 바꾼다. 이쪽은 3장의 CNN과 하는 일이 거의 같다. 그림을 받아 하나의 수를 내놓는다.

---

## 1. 결과

| | 확신도 | 고름 | 클래스 안 다양함 | 학습 시간 |
|---|---|---|---|---|
| [GAN (MLP)](01_gan.md) | 0.908 | 0.945 | 0.953 | 202초 |
| **DCGAN** | 0.909 | **0.970** | **1.085** | 948초 |

확신도는 같고 고름과 다양함이 오른다. 확신도가 이미 0.908로 천장 가까워 더 오를 자리가 없었다. 그림에서 보이는 점 잡음이 사라진 것은 **깨끗함**이지 확신도로 잡히는 성질이 아니다.

학습 시간이 4.7배라는 점도 적어 둔다. 전치 합성곱이 비싸며, [5.1절의 conv AE](../dimreduction/04_ae_cnn.md)가 637초를 쓴 것과 같은 까닭이다.

---

## 2. 두 걸음이 서로 다른 것을 바꾸었다

| 걸음 | 바꾼 것 | 얻은 것 |
|---|---|---|
| VAE → [GAN](01_gan.md) | **손실** (제곱오차 → 적대적) | 클래스 안 다양함 0.628 → 0.953 |
| GAN → DCGAN | **구조** (촘촘 → 합성곱) | 0.953 → 1.085, 고름 0.945 → 0.970 |

**나누어 재었기에 할 수 있는 말이다.** DCGAN만 돌렸다면 0.628에서 1.085까지 오른 것을 보고도 손실 덕인지 구조 덕인지 가릴 수 없었다. 중간에 MLP GAN을 둔 까닭이 이것이며, [4.6절](../../ch04/06_distillation.md)이 홑모델 교사를 대조군으로 둔 것과 같은 생각이다.

그리고 **큰 몫은 손실이 가져갔다.** 0.628 → 0.953이 손실이고, 0.953 → 1.085가 구조다.

---

## 3. 손실 곡선이 더 나빠 보이는데 표본은 더 좋다

| 에포크 | [GAN](01_gan.md) $D$ | $G$ | DCGAN $D$ | $G$ |
|---|---|---|---|---|
| 1 | 0.811 | 1.687 | 0.372 | 2.562 |
| 10 | 0.589 | 2.607 | 0.387 | 2.709 |
| 20 | 1.026 | 1.360 | 0.294 | 3.181 |
| 30 | **1.037** | **1.310** | **0.305** | **3.362** |

MLP 쪽은 균형점($D \approx 1.386$)으로 되돌아갔다. DCGAN은 처음부터 끝까지 판별기가 앞섰고 격차가 **벌어지기까지** 했다.

그런데 표본은 DCGAN 쪽이 더 좋다. 고름도 다양함도 높다.

이것이 [앞 쪽](01_gan.md)이 말한 "손실 곡선을 믿지 마라"의 가장 뚜렷한 증거다. 두 모델 가운데 **손실만 보고 고르면 틀린 쪽을 고른다.**

---

## 4. 사다리의 끝

[5.1절](../dimreduction/index.md)부터 여덟 모델을 걸어 왔다.

| | 모델 | 새로 할 수 있게 된 일 |
|---|---|---|
| 5.1 | PCA → AE_CNN | 줄였다 되살린다 |
| 5.2 | VAE → cVAE | 표본을 만든다, 원하는 숫자를 만든다 |
| 5.3 | GAN → DCGAN | **선명한 표본을 만든다** |

걸음마다 새로 할 수 있게 되는 일이 있었고, 걸음마다 값을 치렀다. VAE는 되살리기를 내주고 뽑기를 샀고, GAN은 인코더를 내주고 선명함을 샀다.

**무엇이 가장 좋은 모델인가에 답이 없는 까닭이 그것이다.** 이 장이 [5.1절](../dimreduction/index.md)부터 되풀이해 온 이야기이며, 자를 바꾸면 순위도 바뀐다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff normal" title="중간"></span>
DCGAN의 클래스 안 다양함이 1.085로 1을 넘었다. "진짜보다 다양하다"가 아니라면 무엇일 수 있는가? 가려내는 방법을 적어라.

</div>

??? success "연습문제 1 풀이"
    이 자는 **심판이 같은 클래스로 묶은 표본들** 사이의 분산이므로, 묶기가 틀리면 값이 부풀려진다. 심판이 7로 읽은 그림 가운데 사람이 보기에 7이 아닌 것이 섞이면 그 무리가 실제보다 다양해 보인다.

    **가려내는 방법 하나.** 확신도가 높은 표본만 골라 다시 재는 것이다.

    ```python
    keep = conf > 0.9                    # 심판이 확신하는 것만
    within = within_class_var(samples[keep], pred[keep])
    ```

    묶기가 원인이라면 이 값이 1 아래로 내려올 것이다. 그대로 1을 넘으면 다른 까닭을 찾아야 한다.

    **방법 둘.** 진짜 자료를 같은 방식으로 다시 재어 본다. 곧 **진짜 라벨 대신 심판의 예측으로** 묶어 분산을 구한다. 진짜 자료조차 이렇게 재면 1을 넘는다면, 1이라는 기준선 자체가 두 가지 다른 방식으로 계산된 값을 견준 것이라 공정하지 않았다는 뜻이다.

    둘째 방법이 더 근본적이다. 본문의 분모는 **진짜 라벨**로 묶어 쟀고 분자는 **심판의 예측**으로 묶어 쟀다. 같은 방식으로 맞추지 않은 것이 1을 넘긴 원인일 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절은 생성기만 합성곱으로 바꾼 것이 아니라 판별기도 함께 바꾸었다. 둘 가운데 어느 쪽이 이득을 가져왔는지 가리려면 어떻게 해야 하는가?

</div>

??? success "연습문제 2 풀이"
    **네 가지를 다 돌려야 한다.** 생성기와 판별기를 각각 촘촘/합성곱으로 두면 $2 \times 2$가 된다.

    | | 판별기 촘촘 | 판별기 합성곱 |
    |---|---|---|
    | 생성기 촘촘 | [GAN](01_gan.md) | ? |
    | 생성기 합성곱 | ? | DCGAN |

    본문은 대각선 두 칸만 재었으므로 둘의 몫을 나눌 수 없다.

    **어느 쪽이 더 중요할지 짐작해 보면** 판별기 쪽일 수 있다. 생성기는 결국 판별기가 주는 신호를 따라가므로, 판별기가 이웃 관계를 볼 줄 알아야 "이 획은 이어져 있지 않다" 같은 것을 지적할 수 있다. 촘촘한 판별기는 화소를 따로따로 보므로 그런 지적을 할 수 없고, 그러면 생성기가 아무리 좋은 구조여도 배울 것이 없다.

    이 짐작을 확인하려면 위 표의 빈 칸 두 개를 채우면 된다. [4.6절](../../ch04/06_distillation.md)이 대조군으로 이득을 둘로 나눈 것과 같은 방식이며, 본문이 "두 걸음이 서로 다른 것을 바꾸었다"고만 적고 판별기 몫을 따로 말하지 않은 까닭도 이 실험을 하지 않았기 때문이다.
