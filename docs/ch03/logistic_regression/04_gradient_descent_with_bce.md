# BCE를 쓰는 경사 하강법

BCE를 쓰는 경사 하강법.

이 튜토리얼은 PyTorch에서 로지스틱 회귀에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python
"""BCE를 쓰는 경사 하강법."""
# [Code Source](https://github.com/patrickloeber/pytorchTutorial)

import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ========================================================================
# 메인
# ========================================================================

# ---------------------------------------------------------------------------
# 0) 데이터 준비 (Breast Cancer Wisconsin 데이터셋: 이진 분류)
#    X: (n_samples, n_features)         e.g., (569, 30)  float64 (from sklearn)
#    y: (n_samples,)                     e.g., (569,)     int (0/1) → will cast to float
#    팁: 완전한 재현성을 원하면 numpy/torch의 씨앗을 설정하고
#         데이터 분할에 random_state=...를 넘긴다 (아래에 이미 되어 있다).
# ---------------------------------------------------------------------------
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
# X: (569, 30) float64
# y: (569,) int64
print(f"Number of Positive Cases in {y.shape[0]} Patients of Original Data :  {y.sum()}")  # e.g., 357/569 positives

n_samples, n_features = X.shape             # n_samples=569, n_features=30

# 분할:
# X_train: (N_train, n_features)            (455, 30) float64
# X_test : (N_test,  n_features)            (114, 30) float64
# y_train: (N_train,)                        (455,)    int64
# y_test : (N_test,)                         (114,)    int64
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)
print(f"Number of Positive Cases in {y_train.shape[0]} Patients of Train Data :  {y_train.sum()}")  # e.g., 288
print(f"Number of Positive Cases in {y_test.shape[0]} Patients of Test Data :  {y_test.sum()}")    # e.g., 69

# 특징 스케일링 (학습 데이터로 적합시켜 둘 다에 적용):
# 변환 후:
# X_train: (N_train, n_features)            float64
# X_test : (N_test,  n_features)            float64
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# torch 텐서로 바꾼다 (층과 연산의 PyTorch 기본값에 맞춰 float32로):
# X_train: (N_train, n_features)            torch.float32
# X_test : (N_test,  n_features)            torch.float32
# y_train: (N_train,)                        torch.float32
# y_test : (N_test,)                         torch.float32
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test  = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test  = torch.from_numpy(y_test.astype(np.float32))

# BCE용으로 목표의 모양을 바꾼다:
# y_train: (N_train, 1)
# y_test : (N_test,  1)
y_train = y_train.view(y_train.shape[0], 1)
y_test  = y_test.view(y_test.shape[0],  1)

# ---------------------------------------------------------------------------
# 1) 모델
#    선형 층:
#      self.linear.weight: (1, n_features)
#      self.linear.bias  : (1,)
#    순전파:
#      input  x: (batch_size, n_features)
#      출력 y: (batch_size, 1)  시그모이드 후 → [0,1] 범위의 확률
#    참고 (안정성 모범 사례):
#      실전에서는 (forward에 시그모이드를 두지 않고) 로짓을 반환하고
#      nn.BCEWithLogitsLoss(내부에서 안정한 시그모이드와 BCE를 적용한다)를 쓰는 편이 낫다.
#      여기서는 설명을 분명히 하려고 sigmoid+BCELoss를 그대로 쓴다.
# ---------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1)  # (n_features → 1)
        # print(f"{self.linear.weight.shape = }") # (1,30)
        # print(f"{self.linear.bias.shape = }")   # (1,)

    def forward(self, x):  # x: (batch_size, n_features) torch.float32
        x = self.linear(x)  # x: (batch_size, 1) torch.float32  ← raw score (logit)
        y_pred = torch.sigmoid(x)  # y_pred: (batch_size, 1) torch.float32 (probabilities)
        return y_pred

model = Model(n_features)

# ---------------------------------------------------------------------------
# 2) 손실과 최적화기
#    기준(BCELoss)이 기대하는 것:
#      input : (batch_size, 1) probabilities in [0,1]
#      target: (batch_size, 1) in {0,1} (float OK: 0.0/1.0)
#    loss: 모양이 ()인 스칼라 텐서 (0차원)
#    팁(다시 한 번): 안정성을 위해 모델의 로짓과 nn.BCEWithLogitsLoss를 함께 쓰는 편이 낫다.
#    참고: 여기서 learning_rate=10.은 설명을 위해 일부러 크게 잡은 값이다. 실무에서는
#          0.1이나 0.01 정도로 시작하거나 Adam 같은 최적화기를 쓴다.
# ---------------------------------------------------------------------------
num_epochs = 100
learning_rate = 10.
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ---------------------------------------------------------------------------
# 3) 학습 루프
#    반복마다의 모양:
#      y_pred = model(X_train)              → (N_train, 1)
#      loss = criterion(y_pred, y_train)    → () scalar
#    모범 사례: 매 단계 역전파 **전에** optimizer.zero_grad()를 호출한다
#    PyTorch에서는 경사가 기본적으로 누적되기 때문이다.
# ---------------------------------------------------------------------------
for epoch in range(num_epochs):
    # 순전파와 손실
    y_pred = model(X_train)              # (N_train, 1)
    loss = criterion(y_pred, y_train)    # () scalar tensor

    # 새 단계 전에 경사를 0으로 만든다 (정석적인 위치는 역전파 전이다)
    optimizer.zero_grad()

    # 역전파와 갱신
    loss.backward()          # populates gradients matching parameter shapes:
                             #   linear.weight.grad: (1, n_features)
                             #   linear.bias.grad  : (1,)
    optimizer.step()         # in-place param update (SGD step)

    if (epoch + 1) % 10 == 0:
        # loss.item() → 파이썬 float(호스트). GPU에 있으면 호스트로 동기화된다.
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# ---------------------------------------------------------------------------
# 4) 평가
#    학습/평가 동작이 다른 층(Dropout/BN)을 쓴다면 평가 모드로 바꾼다:
#      model.eval()
#    y_predicted: (N_test, 1) probabilities in [0,1]
#    y_predicted_cls: (N_test, 1) in {0,1} after thresholding via .round()
#                     - .round()는 0.5를 문턱값으로 삼는 것과 같다:
#                       p>=0.5 → 1.0, p<0.5 → 0.0 (데이터형은 float32로 유지)
#                     - 정수형을 명시적으로 얻으려면:
#                       y_predicted_cls = y_predicted.round().to(torch.int64)
#    y_test: (N_test, 1) is float32 here; equality on 0.0/1.0 floats is exact.
#    eq 연산:
#      - y_predicted_cls.eq(y_test) → torch.bool 마스크
#      - 불리언을 더하면 True가 1로 세어진다
#      - N으로 나누면 [0,1] 범위의 정확도가 된다
#    대안 (확률 대신 로짓을 쓰는 경우):
#      - logits = model(X_test); y_cls = (logits >= 0).to(y_test.dtype)
# ---------------------------------------------------------------------------
with torch.no_grad():
    y_predicted = model(X_test)              # (N_test, 1), probabilities
    y_predicted_cls = y_predicted.round()    # (N_test, 1) 0.0/1.0 as float32
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])  # accuracy in [0,1]
    # 정수로 명시적으로 비교하고 싶다면:
    # y_predicted_cls = y_predicted.round().to(torch.int64)
    # y_test_int = y_test.to(torch.int64)
    # acc = y_predicted_cls.eq(y_test_int).sum() / y_test_int.shape[0]
    print(f'accuracy: {acc.item():.4f}')


if __name__ == "__main__":
    pass```

## 논의

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
