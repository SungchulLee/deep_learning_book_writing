# 차원축소 — 네 걸음 전체

[3장](../../ch03/index.md)과 [4장](../../ch04/index.md)은 **가르는** 법을 배웠다. 이 장은 **만드는** 법을 배운다.

그런데 첫걸음은 만들기가 아니라 **되살리기**다. 그림을 몇 개의 수로 줄였다가 그 수만으로 다시 그려 낼 수 있다면, 그 수들은 그림에 있던 것을 쥐고 있다는 뜻이다. 없던 수를 넣어 없던 그림을 만드는 일은 그다음이다.

---

## 1. 왜 MNIST로 돌아오는가

[4장](../../ch04/index.md)은 MNIST가 너무 쉽다며 CIFAR-10으로 옮겨 갔다. 그 판단은 **정확도라는 자**에서 옳았다. 99.02%에 이르면 남은 오차가 1%포인트도 되지 않아 다음 걸음을 잴 수 없었다.

이 장은 자가 다르다. 그리고 그 자에서는 MNIST가 전혀 포화하지 않았다.

| | 4장이 CIFAR-10을 고른 까닭 | 이 장이 MNIST로 돌아온 까닭 |
|---|---|---|
| 재는 것 | 시험 정확도 | 복원 품질, 그리고 나중에는 표본 품질 |
| MNIST의 형편 | 99.02%로 포화 | **눈으로 판정할 수 있는 유일한 자료** |

되살린 그림이 7인지 아닌지는 보면 안다. CIFAR-10에서 32×32짜리 복원은 잘된 것도 갈색 얼룩이고 잘못된 것도 갈색 얼룩이라, 가장 중요한 판정 도구를 잃는다. 게다가 잠재 공간을 2차원으로 두고 그림으로 그리는 일이 MNIST에서만 뜻을 갖는다.

**어느 데이터가 알맞은가는 무엇을 재려느냐가 정한다.** 4장과 이 장이 서로 다른 답을 낸 것이 그 보기다.

---

## 2. 재는 자가 둘이다

**복원 MSE** — 되살린 그림이 원본과 화소 단위로 얼마나 가까운가. 이 사다리가 실제로 최소화하는 값이다.

**같은 숫자로 읽히는 비율** — 되살린 그림을 [3장 4걸음](../../ch03/mnist/04_cnn.md)의 합성곱 신경망에 넣어, 원래 라벨과 같게 읽는 비율을 센다. 이 절에서 쓰는 심판은 같은 구조를 같은 규약으로 학습해 **98.83%**를 얻은 것이다.

자를 둘 두는 까닭이 있다. MSE는 "화소가 가깝다"만 말할 뿐 **"3이 아직 3인가"**는 말해 주지 않는다. 뒤에 보겠지만 두 자는 자주 서로 다른 답을 낸다.

!!! note "심판의 98.83%에 대하여"
    [4.1절](../../ch04/01_two_ladders.md)이 같은 구조를 씨앗 다섯 개로 재었을 때 98.97~99.30이었다. 여기의 98.83%는 섞는 차례를 따로 고정한 별개의 실행이라 그 범위보다 조금 아래로 나왔다. 3장이 보고한 99.22%를 재현한 값이 아니라 **또 한 번의 뽑기**다.

    심판으로 쓰기에는 넉넉하다. 이 절이 견주는 것은 심판의 절대 성능이 아니라 복원본 사이의 차이이며, 모든 줄이 같은 심판을 쓴다.

### 심판을 만들어 두는 코드

이 절의 모든 쪽이 이 심판을 쓴다. 한 번 학습해 파일로 남겨 두고 뒤에서는 읽어 쓴다.

```python
"""심판 CNN을 학습해 mnist_judge.pt로 남긴다.

3장 4걸음과 같은 구조·같은 규약이다. 이 장의 모든 절이 이 파일을 읽어,
되살린 그림과 만들어 낸 그림이 아직 숫자로 읽히는지를 잰다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

SEED, BATCH, LR, EPOCHS = 42, 100, 1e-3, 5
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === 자료 ===================================================================
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))])
tr_ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
te_ds = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)


def materialize(ds):
    """DataLoader를 한 번 돌려 텐서로 펼쳐 둔다. 뒤에서 되풀이해 쓰기 편하다."""
    xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=2000, shuffle=False):
        xs.append(x); ys.append(y)
    return torch.cat(xs), torch.cat(ys)


Xtr_img, ytr = materialize(tr_ds)          # (60000, 1, 28, 28)
Xte_img, yte = materialize(te_ds)
print(f"자료 {tuple(Xtr_img.flatten(1).shape)}")


# === 심판 ===================================================================
class JudgeCNN(nn.Module):
    """3장 4걸음의 구조 그대로."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return self.fc2(self.dropout(torch.relu(self.fc1(x.flatten(1)))))


if __name__ == "__main__":
    torch.manual_seed(SEED)
    judge = JudgeCNN().to(device)
    opt = optim.Adam(judge.parameters(), lr=LR)
    # 섞는 차례를 전역 난수와 떼어 놓는다 (4.1절 「이 표를 믿어도 되는가」 참고)
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(Xtr_img, ytr), batch_size=BATCH,
                        shuffle=True, generator=g)

    for _ in range(EPOCHS):
        judge.train()
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(judge(xb.to(device)), yb.to(device)).backward()
            opt.step()

    judge.eval()
    with torch.no_grad():
        pred = torch.cat([judge(Xte_img[i:i + 1000].to(device)).argmax(1).cpu()
                          for i in range(0, len(Xte_img), 1000)])
    acc = 100.0 * (pred == yte).float().mean()
    torch.save(judge.state_dict(), "mnist_judge.pt")
    print(f"심판 CNN {acc:.2f}%  -> mnist_judge.pt")
```

**출력:**

```
자료 (60000, 784)
심판 CNN 98.83%  (35s)  -> mnist_judge.pt
```

---

## 3. 사다리는 3장·4장과 같은 네 걸음이다

| | 분류 사다리 (3·4장) | 이 절의 표현 사다리 |
|---|---|---|
| 1 | 템플릿 — 닫힌 꼴 | [**PCA**](01_pca.md) — 닫힌 꼴 |
| 2 | 선형 + 소프트맥스 | [**AE_Linear**](02_ae_linear.md) |
| 3 | 다층 퍼셉트론 | [**AE_MLP**](03_ae_mlp.md) |
| 4 | 합성곱 신경망 | [**AE_CNN**](04_ae_cnn.md) |

같은 네 생각을 다른 목표에 건다. 라벨은 한 번도 쓰지 않는다.

규약도 그대로다. Adam $10^{-3}$, 배치 100, 씨앗 42이며, 인코더는 100 에포크를 돈다([CIFAR-10 쪽](05_cifar_classification.md)에서 20 에포크로는 모자랐던 것을 보았다).

---

## 4. 네 걸음의 결과

부호를 64로 두었을 때다.

| 부호 64 | 복원 MSE | 같은 숫자로 | 인코더 매개변수 |
|---|---|---|---|
| 원본 (784차원) | 0 | **98.83%** | — |
| [PCA](01_pca.md) | 0.0953 | 97.81% | 50,176 |
| [AE_Linear](02_ae_linear.md) | 0.0959 | 97.80% | 50,240 |
| [AE_MLP](03_ae_mlp.md) | 0.0546 | 98.51% | 217,408 |
| [AE_CNN](04_ae_cnn.md) | **0.0243** | 98.52% | 105,216 |

부호를 2로 두면 차이가 훨씬 크게 벌어진다.

| 부호 2 | 복원 MSE | 같은 숫자로 |
|---|---|---|
| [PCA](01_pca.md) | 0.5865 | **35.66%** |
| [AE_Linear](02_ae_linear.md) | 0.5867 | 35.54% |
| [AE_MLP](03_ae_mlp.md) | **0.4223** | **70.13%** |
| [AE_CNN](04_ae_cnn.md) | 0.4742 | 61.78% |

미리 짚어 둘 것이 셋이다.

**첫째, 2걸음이 아무것도 벌지 못한다.** [분류 사다리](../../ch04/01_two_ladders.md)에서는 같은 수법이 10.32%포인트를 벌었는데 여기서는 −0.01%포인트, 곧 오히려 아주 조금 잃는다. 까닭은 [AE_Linear](02_ae_linear.md)가 다룬다.

**둘째, 부호 크기에 따라 순위가 뒤집힌다.** 부호 64에서는 AE_CNN이 이기고 부호 2에서는 AE_MLP가 이긴다. [AE_CNN](04_ae_cnn.md)이 그 까닭을 본다.

**셋째, 두 자가 자주 어긋난다.** 부호 64에서 MSE는 네 배 차이인데 숫자 정체는 0.7%포인트 차이다. 이것도 [AE_CNN](04_ae_cnn.md)에서 정리한다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
위의 상자는 심판의 98.83%가 낮아도 괜찮다고 말한다 — "이 절이 견주는 것은 심판의 절대 성능이 아니라 복원본 사이의 차이이며, 모든 줄이 같은 심판을 쓴다."

이 논리가 **막아 주는 것과 막아 주지 못하는 것**을 가려라. 부호 64의 네 줄이 어디에 놓이는지 셈해 보고, [AE_MLP](03_ae_mlp.md)의 98.51%와 [AE_CNN](04_ae_cnn.md)의 98.52%를 두고 "둘이 같다"고 말할 수 있는지 따져라.

</div>

??? success "연습문제 1 풀이"
    ```python
    judge = 98.83
    rows = [("PCA-64", 97.81), ("AE_Linear-64", 97.80),
            ("AE_MLP-64", 98.51), ("AE_CNN-64", 98.52)]
    for n, v in rows:
        print(f"{n:14} {v:6.2f}%   심판까지 {judge - v:.2f}%p")
    print(f"네 줄이 놓인 폭 {98.52 - 97.80:.2f}%p")
    ```

    ```
    PCA-64          97.81%   심판까지 1.02%p
    AE_Linear-64    97.80%   심판까지 1.03%p
    AE_MLP-64       98.51%   심판까지 0.32%p
    AE_CNN-64       98.52%   심판까지 0.31%p
    네 줄이 놓인 폭 0.72%p
    ```

    **같은 심판을 쓰는 것이 막아 주는 것은 치우침이다.** 심판을 더 좋은 것으로 바꾸면 모든 줄이 함께 오르내릴 테니, 줄 사이의 순서와 간격은 심판을 고른 탓이 아니다. 그만큼은 상자의 말이 옳다.

    **막아 주지 못하는 것은 분해능이다.** 네 줄이 겨우 0.72%포인트 안에 몰려 있고, 위의 둘은 심판 자신의 정확도에서 0.3%포인트 안에 들어와 있다. 이 자리에서 98.51과 98.52의 차이, 곧 **0.01%포인트**를 두고 무엇을 말하려면 심판이 그만큼 가늘게 재야 하는데, 시험 자료 10,000장에서 0.01%포인트는 **한 장**이다. 한 장으로는 아무것도 말할 수 없다.

    그러므로 "둘이 같다"는 말은 **"이 자로는 가를 수 없다"**로 적어야 맞다. 실제로 다른 자는 둘을 또렷이 가른다 — 복원 MSE는 0.0546과 0.0243으로 56% 차이다. 같은 두 모델을 한 자는 같다 하고 다른 자는 두 배 차이라 한다.

    !!! warning "98.83%는 천장이 아니다"
        "복원본이 원본보다 잘 읽힐 수는 없으니 98.83%가 천장"이라고 생각하기 쉽다. **그렇지 않다.** 이 장의 [cVAE](../latent_generative/03_cvae.md)는 부호 2에서 정체 **99.96%**를 받는다. 심판 자신의 98.83%보다 높다.

        까닭은 복원본이 원본보다 **읽기 쉬울 수 있다**는 데 있다. 되살리기와 만들기는 전형적인 모양으로 매끄럽게 미는 쪽이라, 사람이 알아보기 어려운 별난 손글씨가 반듯한 숫자로 바뀐다. 심판이 원본에서 틀리던 1.17%의 일부가 복원본에서는 맞는다.

        곧 98.83%는 천장이 아니라 **눈금의 한 자리**일 뿐이다. 그 언저리에서 분해능이 나빠지는 것은 여전히 사실이지만, 넘을 수 없는 선으로 읽으면 틀린다.

    고치는 길은 자를 바꾸는 것이다. 맞혔는지(0 또는 1)가 아니라 **심판의 확신도**를 재면 98% 언저리에서도 눈금이 남는다. [AE_CNN 쪽](04_ae_cnn.md)의 연습문제가 그것을 제안하고, [5.2절](../latent_generative/index.md)과 [5.3절](../generative/index.md)이 표본을 잴 때 실제로 그 자를 쓴다.

