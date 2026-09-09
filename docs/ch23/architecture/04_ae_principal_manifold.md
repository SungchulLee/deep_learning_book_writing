# 자기 부호기의 주다양체

1차원 병목을 가진 비선형 자기 부호기는 자료를 꿰는 굽은 주다양체를 배워, 주성분 분석의 선형 주축을 비선형 짜임까지 잡도록 넓힌다. 이 보기는 2차원 S자 곡선 자료 묶음에 작은 자기 부호기를 익혀, 풀개가 최소 제곱 뜻에서 자료에 가장 잘 맞는 매끄러운 1차원 곡선, 곧 첫 주성분의 비선형 짝을 그려 내는 것을 보인다.

## 1. 코드

```python
"""자기 부호기의 주다양체."""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

def make_s_curve(n=400, noise=0.07, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(-2.5, 2.5, size=n)
    x = t
    y = np.sin(1.5 * t)
    X = np.stack([x, y], axis=1)
    X += rng.normal(scale=noise, size=X.shape)
    return X.astype(np.float32)

X = make_s_curve(n=600, noise=0.06, seed=42)
X_mean, X_std = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)
Xn = (X - X_mean) / X_std
Xt = torch.from_numpy(Xn)

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 16), nn.Tanh(),
            nn.Linear(16, 32), nn.Tanh(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

ae = AE()
opt = optim.Adam(ae.parameters(), lr=5e-3)
loss_fn = nn.MSELoss()

for ep in range(1, 60001):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 128):
        xb = Xt[perm[i:i+128]]
        xhat, _ = ae(xb)
        loss = loss_fn(xhat, xb)
        opt.zero_grad()
        loss.backward()
        opt.step()

if __name__ == "__main__":
    pass
```

## 2. 논의

주성분 분석이 자료에 선을 맞출 때는 직교 쏘기 어긋남을 가장 작게 하는 선형 아래 공간을 찾는다. 1차원 병목과 비선형 깨어남(쌍곡탄젠트)을 가진 자기 부호기는 이를 넓혀 굽은 1차원 다양체를 찾는다. 풀개를 숨은 값의 범위에 걸쳐 훑으면 자료 구름을 꿰는 매끄러운 2차원 곡선이 그려지는데 이것이 "주다양체" 또는 "주곡선"이다.

(정류 선형 대신) 쌍곡탄젠트 깨어남을 고른 것은 일부러이다. 쌍곡탄젠트는 매끄럽고 미분할 수 있는 내놓기를 내어 꺾임 없는 이어진 곡선을 만든다. 정류 선형은 조각마다 선형인 다양체를 만드는데 그것도 옳지만 매끄러운 곡선을 어림하려면 숨은 단위가 더 필요할 수 있다. 그물의 깊이(부호기와 풀개에 각각 세 층)가 S자 곡선의 굽음을 나타낼 담이를 넉넉히 준다.

익히기는 들임 점과 병목을 지나 다시 세운 것 사이 평균 제곱 어긋남 손실을 쓴다. 이는 자료 점마다 배운 다양체까지의 제곱 거리를 가장 작게 하는 것과 같다. 담이와 익히기가 넉넉하면 자기 부호기는 자료가 가장 빽빽한 자리를 지나는 다양체로 모이며, 이는 주성분 분석의 주축이 자료 구름의 가운데를 지나는 것과 비슷하다.

## 연습문제

**연습문제 1.**
S자 곡선을 (2차원으로 쏜 `sklearn.datasets.make_swiss_roll`의) 나선 자료 묶음으로 갈음하고 자기 부호기를 다시 익혀라. 배운 다양체가 나선을 충실히 그려 내는가?

??? success "연습문제 1 풀이"
    ```python
    from sklearn.datasets import make_swiss_roll
    X_3d, t = make_swiss_roll(600, noise=0.3, random_state=42)
    X_spiral = X_3d[:, [0, 2]]  # 2차원으로 쏜다
    ```
    1차원 병목을 가진 자기 부호기는 나선을 그려 낼 수 있지만 나선이 제 몸 가까이 되돌아와 스스로 엇갈리는 곳에서는 힘겨울 수 있다. 그물을 깊게 하거나 넓히면 더 빡빡한 굽이를 잡는 데 도움이 된다.

---

**연습문제 2.**
병목을 2차원으로 늘려 같은 S자 곡선에 익혀라. 배운 나타냄에 무슨 일이 생기며, 여기서 1차원 병목이 왜 더 뜻이 있는가?

??? success "연습문제 2 풀이"
    2차원 자료에 2차원 병목을 쓰면 자기 부호기가 다시 세우기 어긋남이 아주 낮은 거의 항등인 대응을 배울 수 있다. 그러나 더는 뜻 있는 차원 줄이기를 하지 않는다. 숨은 공간이 들임 공간을 그대로 비칠 뿐이다. 1차원 병목은 그물이 흔들림의 으뜸 결(점이 늘어선 곡선)을 찾게 하며 이것이 다양체 배움의 목적이다.

---

**연습문제 3.**
배운 다양체를 (R이나 파이썬 감개로 쓸 수 있는) `princurve` 알고리즘으로 맞춘 주곡선과 견주어라. 두 곡선을 자료 위에 겹쳐 그리고 차이를 논하라.

??? success "연습문제 3 풀이"
    주곡선 알고리즘은 점을 매끄러운 곡선에 되풀이해 쏘고 곡선이 조건부 평균을 지나도록 새로 고친다. 두 방식 모두 비슷한 풀이로 모이지만 자기 부호기가 더 융통성 있고(더 복잡한 위상을 나타낼 수 있다) 차원이 높은 자료로 규모를 키우기 쉽다. 주곡선 알고리즘은 스스로 한결같음 잣대의 국소 최적으로 모인다는 이론 보장이 더 세다.

## 정리하며

**다룬 것** — 자기 부호기의 주다양체

주성분 분석이 자료에 선을 맞출 때는 직교 쏘기 어긋남을 가장 작게 하는 선형 아래 공간을 찾는다.

고갱이 갈래는 `AE`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
