# PCA — 닫힌 꼴이라 씨앗이 없다

사다리의 1걸음이다. 주성분 분석은 학습하지 않는다. 공분산 행렬을 고유분해하여 분산이 큰 방향 $k$개를 고르면 끝이며, 경사 하강법도 에포크도 씨앗도 없다.

**그래서 이 사다리의 1걸음에도 오차 막대가 없다.** [3장 템플릿 학습](../../ch03/mnist/01_template_learning.md), [4.5절 PCA](../../ch04/05_dimensionality_reduction.md)와 같다. 두 사다리가 똑같이 완전히 재현되는 바닥에서 출발한다.

---

## 1. 코드

```python
"""1걸음: PCA. 학습이 없으므로 씨앗도 없고 에포크도 없다.

앞서 만들어 둔 mnist_judge.pt를 읽어, 되살린 그림이 아직 같은 숫자로
읽히는지를 함께 잰다.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))])
tr_ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
te_ds = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)


def materialize(ds):
    xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=2000, shuffle=False):
        xs.append(x.flatten(1)); ys.append(y)
    return torch.cat(xs), torch.cat(ys)


Xtr, ytr = materialize(tr_ds)                  # (60000, 784)
Xte, yte = materialize(te_ds)


# === 심판 (앞 쪽에서 학습해 둔 것을 읽는다) =================================
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
    """되살린 (N, 784)를 심판에 넣어 원래 라벨과 같게 읽는 비율."""
    pred = torch.cat([judge(flat[i:i + 1000].reshape(-1, 1, 28, 28).to(device))
                      .argmax(1).cpu() for i in range(0, len(flat), 1000)])
    return 100.0 * (pred == yte).float().mean().item()


# === PCA ====================================================================
mu = Xtr.mean(0, keepdim=True)
Xc = Xtr - mu
cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)          # (784, 784)

# torch.linalg.eigh는 이 크기에서 실패하는 빌드가 있다(LAPACK 작업공간 문제).
# numpy의 경로를 float64로 쓰면 안정적이며, 어차피 한 번만 계산한다
ev, evec = np.linalg.eigh(cov.double().numpy())
evals = torch.from_numpy(np.ascontiguousarray(ev[::-1])).float()
evecs = torch.from_numpy(np.ascontiguousarray(evec[:, ::-1])).float()
torch.save((mu, evecs, evals), "mnist_pca.pt")

for k in (2, 16, 32, 64):
    V = evecs[:, :k]                            # 위에서 k개
    Z = (Xte - mu) @ V                          # 부호로 줄이기
    rec = Z @ V.T + mu                          # 되살리기
    mse = ((Xte - rec) ** 2).mean().item()
    var = (evals[:k].sum() / evals.sum()).item()
    print(f"  PCA-{k:3d}  설명분산 {100 * var:5.1f}%  "
          f"복원 MSE {mse:.5f}  같은 숫자로 {identity(rec):.2f}%")
```

**출력:**

```
  PCA-  2  설명분산  16.8%  복원 MSE 0.58645  같은 숫자로 35.66%
  PCA- 16  설명분산  59.4%  복원 MSE 0.28296  같은 숫자로 83.69%
  PCA- 32  설명분산  74.4%  복원 MSE 0.17728  같은 숫자로 94.39%
  PCA- 64  설명분산  86.2%  복원 MSE 0.09530  같은 숫자로 97.81%
```

---

## 2. 주성분을 몇 개 쓸 것인가

![MNIST 숫자 열 개와 이를 주성분 2, 16, 32, 64개로 줄였다 되살린 모습. 2개로는 숫자가 다른 숫자로 바뀌고 64개로는 원본과 거의 같다](../figures/pca_reconstructions.svg)

| | 설명 분산 | 복원 MSE | 같은 숫자로 읽히는 비율 |
|---|---|---|---|
| 원본 (784차원) | 100% | 0 | **98.83%** |
| PCA-2 | 16.8% | 0.5865 | **35.66%** |
| PCA-16 | 59.4% | 0.2830 | 83.69% |
| PCA-32 | 74.4% | 0.1773 | 94.39% |
| PCA-64 | 86.2% | 0.0953 | 97.81% |

64개를 쓰면 97.81%로 거의 돌아온다. 784개를 64개로, 곧 **12분의 1로 줄이고도 숫자의 정체는 거의 그대로**다.

---

## 3. PCA-2는 흐리게 만드는 것이 아니라 다른 숫자로 만든다

그림의 둘째 줄을 보라. 4는 9가 되고, 5도 9가 되며, 2는 3처럼 보인다.

수로도 그렇다. **35.66%**다. 우연이 10%이니 아주 못 맞히는 것은 아니지만, 원본의 98.83%에서 3분의 2가 날아갔다.

**2차원 선형 부분공간은 숫자를 서로 겹쳐 놓는다.** 주성분 2개가 쥐고 있는 것은 분산이 가장 큰 두 방향뿐이고, 그 두 방향에서 값이 비슷한 숫자들은 같은 자리로 모인다.

이 실패가 나머지 세 걸음이 무엇을 고쳐야 하는지를 정한다. [다음 걸음](02_ae_linear.md)은 같은 2차원을 경사 하강법으로 찾아보고, 그다음 걸음이 **굽은** 2차원을 찾는다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
PCA-2는 4를 9로, 5도 9로 되살린다. 어떤 숫자 짝이 서로 잘 섞이는지 알아보려면 무엇을 재면 되는가?

</div>

??? success "연습문제 1 풀이"
    **혼동 행렬을 만들면 된다.** 복원본을 심판에 넣되 맞은 비율만 세지 말고, 원래 라벨 $i$가 어떤 라벨 $j$로 읽혔는지를 $10 \times 10$ 표에 쌓는다.

    ```python
    conf = torch.zeros(10, 10)
    for true, pred in zip(yte, judge(rec).argmax(1)):
        conf[true, pred] += 1
    ```

    이 표가 알려 주는 것은 PCA의 성질이다. 주성분 2개가 쥐고 있는 것은 분산이 가장 큰 두 방향이고, 그 방향에서 값이 비슷한 숫자들은 같은 자리로 모인다. 4와 9가 섞이는 까닭은 둘 다 위쪽이 닫히거나 열린 고리에 세로획이 붙은 모양이라, 전체 밝기 분포가 닮았기 때문이다.

    같은 표를 [AE_MLP-2](03_ae_mlp.md)에도 만들어 견주면, 비선형 사상이 **어떤 짝을 떼어 놓는 데 성공했는지**가 드러난다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
설명 분산과 복원 MSE는 같은 것을 다르게 적은 값이다. 둘의 관계를 식으로 적어라.

</div>

??? success "연습문제 2 풀이"
    자료를 중심화한 뒤 상위 $k$개 주성분으로 사영했다 되살리면, 잃는 것은 **버린 방향의 분산**이다. 고윳값 $\lambda_1 \ge \lambda_2 \ge \cdots$을 분산이 큰 순서라 하면

    $$
    \text{복원 MSE} = \frac{1}{D}\sum_{i > k} \lambda_i,
    \qquad
    \text{설명 분산} = \frac{\sum_{i \le k} \lambda_i}{\sum_i \lambda_i}
    $$

    이다($D$는 화소 수 784). 곧 **설명 분산이 오르는 만큼 복원 MSE가 내린다.** 본문 표의 두 열이 나란히 움직이는 까닭이 이것이며, 둘은 사실상 같은 수를 다르게 적은 것이다.

    그런데 세 번째 열인 **숫자 정체는 이 관계 밖에 있다.** PCA-16은 분산의 59%를 쥐고도 정체는 83.69%를 지키고, PCA-2는 분산 16.8%에 정체 35.66%다. 분산과 정체가 비례하지 않는다. 분산이 큰 방향이 숫자를 가르는 방향과 같지 않기 때문이며, 같은 이야기를 [4.5절](../../ch04/05_dimensionality_reduction.md)이 CIFAR-10에서 더 크게 겪었다.
