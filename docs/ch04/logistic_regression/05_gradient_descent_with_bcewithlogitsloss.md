# BCEWithLogitsLoss를 쓰는 경사 하강법

BCEWithLogitsLoss를 쓰는 경사 하강법.

이 튜토리얼은 PyTorch에서 로지스틱 회귀에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 1. 코드

```python
"""BCEWithLogitsLoss를 쓰는 경사 하강법."""
# [Code Source](https://github.com/patrickloeber/pytorchTutorial)

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ========================================================================
# 메인
# ========================================================================

# ---------------------------------------------------------------------------
# 0) 데이터 준비 (Breast Cancer Wisconsin 데이터셋: 이진 분류)
#    X: (n_samples, n_features)         e.g., (569, 30)  float64
#    y: (n_samples,)                    e.g., (569,)     int (0/1) → will cast to float
# ---------------------------------------------------------------------------
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target                   # X: (569, 30), y: (569,)

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# 특징을 표준화한다 (학습 데이터로 적합시켜 둘 다에 적용)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# torch.float32로
X_train = torch.from_numpy(X_train.astype(np.float32))  # (N_train, n_features)
X_test  = torch.from_numpy(X_test.astype(np.float32))   # (N_test,  n_features)
y_train = torch.from_numpy(y_train.astype(np.float32))  # (N_train,)
y_test  = torch.from_numpy(y_test.astype(np.float32))   # (N_test,)

# BCEWithLogits용으로 목표의 모양을 바꾼다: (N, 1)
y_train = y_train.view(-1, 1)   # (N_train, 1)
y_test  = y_test.view(-1, 1)    # (N_test,  1)

# ---------------------------------------------------------------------------
# 1) 모델 (로짓 버전)
#    - 선형 층에서 **로짓**(날것의 점수)을 반환한다.
#    - 여기서 시그모이드를 적용하지 말 것. 안정성을 위해 BCEWithLogitsLoss를 쓴다.
#      (시그모이드와 BCE를 안정한 커널 하나로 합친 것이다.)
#    모양:
#      input x: (batch_size, n_features)
#      logits : (batch_size, 1)
# ---------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1)  # weight: (1, n_features), bias: (1,)

    def forward(self, x):
        logits = self.linear(x)   # (batch_size, 1), raw scores
        return logits

model = Model(n_features)

# ---------------------------------------------------------------------------
# 2) 손실과 최적화기
#    - nn.BCEWithLogitsLoss를 쓴다 (확률이 아니라 로짓을 받는다).
#    - 클래스가 불균형하면 pos_weight=tensor([w])를 넘길 수 있다.
# ---------------------------------------------------------------------------
num_epochs = 100
learning_rate = 0.01
criterion = nn.BCEWithLogitsLoss()                 # ← stable sigmoid + BCE
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ---------------------------------------------------------------------------
# 3) 학습 루프
#    반복마다의 모양:
#      logits = model(X_train)                → (N_train, 1)
#      loss   = criterion(logits, y_train)    → () scalar
# ---------------------------------------------------------------------------
for epoch in range(num_epochs):
    logits = model(X_train)                   # (N_train, 1), raw scores
    loss = criterion(logits, y_train)         # () scalar tensor

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        # loss.item()은 파이썬 float을 반환한다 (GPU에 있으면 동기화된다)
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# ---------------------------------------------------------------------------
# 4) 평가
#    - 확률을 얻으려면 평가 시점에만 시그모이드를 적용한다.
#    - 클래스를 얻으려면 로짓을 0으로 문턱값 처리한다 (sigmoid(0)=0.5이므로).
#      y_cls = (logits >= 0).float()
# ---------------------------------------------------------------------------
with torch.no_grad():
    logits_test = model(X_test)                       # (N_test, 1)
    probs_test  = torch.sigmoid(logits_test)          # (N_test, 1) in [0,1] (optional)
    y_predicted_cls = (logits_test >= 0).float()      # (N_test, 1) {0,1} without calling sigmoid

    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')


if __name__ == "__main__":
    pass
```

## 2. 논의

`Model` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 분류 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `Model`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
층이나 블록의 개수를 설정할 수 있도록 `Model`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = Model(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — BCEWithLogitsLoss를 쓰는 경사 하강법

`Model` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다.

핵심 클래스는 `Model`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
