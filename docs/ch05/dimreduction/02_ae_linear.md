# AE_Linear — PCA를 다시 찾아낼 뿐이다

2걸음은 [1걸음](01_pca.md)과 **같은 일을 경사 하강법으로** 한다. 활성화 함수 없이 층 두 개면 된다.

```python
"""2걸음: 선형 오토인코더. PCA와 같은 곳에 닿는지 확인한다.

mnist_judge.pt는 5.1절 첫 쪽에서 학습해 둔 것을 읽어 쓴다.
규약은 이 장 전체와 같다. Adam 1e-3, 배치 100, 씨앗 42, 인코더 100 에포크.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

SEED, BATCH, LR, AE_EPOCHS = 42, 100, 1e-3, 100
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))])
tr_ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
te_ds = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)


def materialize(ds):
    xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=2000, shuffle=False):
        xs.append(x); ys.append(y)
    return torch.cat(xs), torch.cat(ys)


Xtr_img, _ = materialize(tr_ds)
Xte_img, yte = materialize(te_ds)
Xtr, Xte = Xtr_img.flatten(1), Xte_img.flatten(1)


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
def identity(flat):
    pred = torch.cat([judge(flat[i:i + 1000].reshape(-1, 1, 28, 28).to(device))
                      .argmax(1).cpu() for i in range(0, len(flat), 1000)])
    return 100.0 * (pred == yte).float().mean().item()


def train_ae(model, X):
    """되살리기만 배운다. 라벨은 한 번도 쓰지 않는다."""
    torch.manual_seed(SEED)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(X), batch_size=BATCH, shuffle=True, generator=g)
    for _ in range(AE_EPOCHS):
        model.train()
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            F.mse_loss(model(xb), xb).backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def reconstruct(model, X):
    return torch.cat([model(X[i:i + 1000].to(device)).cpu()
                      for i in range(0, len(X), 1000)])


# === 2걸음: 활성화가 없는 오토인코더 =======================================
class LinearAE(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.enc = nn.Linear(784, k)            # 활성화가 없다 = 선형
        self.dec = nn.Linear(k, 784)

    def forward(self, x):
        return self.dec(self.enc(x))


for k in (64, 2):
    m = train_ae(LinearAE(k), Xtr)
    rec = reconstruct(m, Xte)
    enc_p = sum(p.numel() for p in m.enc.parameters())
    print(f"  ae_linear{k:<2d}  복원 MSE {((rec - Xte) ** 2).mean():.5f}  "
          f"인코더 {enc_p:,}  같은 숫자로 {identity(rec):.2f}%")
```

**출력:**

```
  ae_linear64  복원 MSE 0.09587  인코더 50,240  같은 숫자로 97.80%
  ae_linear2   복원 MSE 0.58666  인코더 1,570  같은 숫자로 35.54%
```

100 에포크를 돌린 결과가 이렇다.

| 부호 크기 | | 복원 MSE | 같은 숫자로 |
|---|---|---|---|
| 64 | PCA | **0.09530** | 97.81% |
| 64 | AE_Linear | 0.09587 | 97.80% |
| 2 | PCA | **0.58645** | 35.66% |
| 2 | AE_Linear | 0.58666 | 35.54% |

**두 자 모두에서 같은 값이 나온다.** MSE는 0.6%와 0.04% 차이이고, 숫자 정체는 0.01%포인트와 0.12%포인트 차이다.

---

## 1. 당연한 일이다

활성화가 없는 오토인코더가 표현할 수 있는 것은 **선형 사영뿐**이다. 그리고 제곱오차를 가장 작게 만드는 $k$차원 선형 사영은 PCA가 주는 것이다. 그러므로 둘이 같은 곳에 닿는 것이 이치에 맞는다.

경사 하강법은 `eigh`가 곧장 계산하는 답을 **270초에 걸쳐 더듬어 찾은** 셈이다.

MSE가 PCA보다 **조금 나쁜** 것도 이치에 맞는다. PCA는 정확한 최적해이고 Adam은 그 근처에서 멈춘다. 부호가 2일 때 차이가 더 작은 것(0.04% 대 0.6%)도 그렇다. 2차원을 찾는 일이 64차원을 찾는 일보다 쉽다.

---

## 2. 두 사다리의 2걸음이 갈라진다

여기가 이 절에서 가장 눈여겨볼 자리다.

| 사다리 | 1걸음 → 2걸음 | 얻은 것 |
|---|---|---|
| 분류 ([4.1절](../../ch04/01_two_ladders.md)) | 템플릿 → 선형, 학습 | **+9.46%p** |
| 표현 (이 절) | PCA → AE_Linear | **−0.01%p** (부호 2에서는 −0.12%p) |

**똑같은 수법인데 한쪽은 9.46%포인트를 벌고 다른 쪽은 벌기는커녕 조금 잃는다.** 잃는 것까지 이치에 맞는다. 방금 본 대로 PCA가 정확한 최적해이고 학습은 그 언저리에서 멈추니, 잘해야 비기고 대개는 아주 조금 못 미친다.

까닭은 **닫힌 꼴 해가 그 목표에 최적이었는가**이다.

| | 1걸음의 닫힌 꼴 해 | 그 목표에 최적인가 | 학습이 채울 자리 |
|---|---|---|---|
| 분류 사다리 | 클래스 평균 템플릿 | **아니다** | 넓다 |
| 표현 사다리 | PCA | **그렇다** | 없다 |

클래스 평균은 "3은 이렇게 생겼다"만 담을 뿐 "여기가 켜져 있으면 3이 아니다"를 담지 못한다. 분류에 최적이 아니었으니 학습이 채울 자리가 넓었다. PCA는 제곱오차라는 제 목표에 **이미 최적**이다. 채울 자리가 없다.

곧 **학습이 값어치가 있는 것은 닫힌 꼴 해가 없거나 그것이 목표와 어긋날 때뿐이다.** 어느 사다리에서든 "학습하면 좋아진다"고 말할 수 있는 것이 아니다.

---

## 3. 이 걸음을 굳이 밟는 까닭

결과가 PCA와 같다면 왜 학습시키는가. 두 가지 쓸모가 있다.

**첫째, 위의 대비를 얻는다.** 분류 사다리와 나란히 놓아야 "학습이 언제 값어치가 있는가"를 말할 수 있다.

**둘째, 점검 도구가 된다.** 새 오토인코더를 짰을 때 활성화를 빼고 돌려 PCA와 맞는지 보면, 인코더·디코더·손실·자료 전처리가 모두 제대로 붙었는지를 **한 번에** 확인할 수 있다.

!!! note "이 확인이 네 번 어긋나지 않았다"
    [4.5절](../../ch04/05_dimensionality_reduction.md)이 CIFAR-10에서 한 번, 이 절이 부호 크기 두 가지와 **서로 독립인 자 두 가지**로 네 번. 모두 맞는다.

    이론이 주는 값은 이렇게 공짜 점검 도구로 쓸 수 있다.

[다음 걸음](03_ae_mlp.md)이 활성화를 넣으면 이 벽이 무너진다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
AE_Linear의 복원 MSE가 PCA보다 **낮게** 나왔다면 무엇을 의심해야 하는가?

</div>

??? success "연습문제 1 풀이"
    **구현이 틀렸다고 의심해야 한다.**

    부호 크기가 $k$일 때 제곱오차를 가장 작게 만드는 선형 사영은 상위 $k$개 주성분이 이루는 부분공간이며, 이는 증명된 사실이다. 활성화가 없는 오토인코더가 나타낼 수 있는 것은 선형 사영뿐이므로 그보다 잘할 수 없다.

    흔한 원인은 이렇다. 어딘가에 활성화가 남아 있거나(정말로 선형인지), MSE를 서로 다른 자료에서 쟀거나(한쪽은 학습, 한쪽은 시험), 정규화·중심화가 서로 달랐거나.

    본문에서 부호 2와 64, 자 두 가지로 네 번 확인한 까닭이 이것이다. 이론값은 **공짜로 얻는 점검 도구**다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
AE_Linear가 PCA와 같은 **부분공간**을 찾는다고 해서 같은 **축**을 찾는 것은 아니다. 둘이 어떻게 다른지 말하고, 확인하는 방법을 적어라.

</div>

??? success "연습문제 2 풀이"
    PCA의 축은 **분산이 큰 순서로 줄 세워진 직교 축**이다. 1번 주성분이 가장 많은 분산을 담고, 축끼리는 서로 직각이다.

    AE_Linear에는 그런 제약이 전혀 없다. 손실은 "되살리기만 잘하면 된다"이므로, 같은 부분공간을 **아무렇게나 비스듬히 놓인 축**으로 표현해도 손실이 똑같다. 축의 순서도 크기도 정해지지 않는다.

    확인하는 방법은 이렇다. 인코더 가중치 $W \in \mathbb{R}^{k \times 784}$의 행들이 이루는 공간을 PCA의 상위 $k$개 주성분이 이루는 공간과 견준다.

    ```python
    # 두 부분공간이 같은지: 주성분으로 사영해도 길이가 보존되는가
    W = ae.enc.weight.detach()            # (k, 784)
    Q, _ = torch.linalg.qr(W.T)           # 인코더 공간의 정규직교 기저
    proj = V.T @ Q                        # V는 PCA 상위 k개
    print(torch.linalg.svdvals(proj))     # 모두 1에 가까우면 같은 공간
    ```

    특이값이 모두 1 언저리면 같은 부분공간이다. 그런데 $W$의 행 하나를 PCA의 주성분 하나와 직접 견주면 닮지 않았을 것이다.

    **부분공간은 정해져 있고 그 안의 좌표계는 정해져 있지 않다**는 것이 요점이다. 오토인코더 계열이 대체로 그러하며, [5.2절](../latent_generative/index.md)의 VAE는 손실에 항을 더해 그 자유를 일부 묶는다.
