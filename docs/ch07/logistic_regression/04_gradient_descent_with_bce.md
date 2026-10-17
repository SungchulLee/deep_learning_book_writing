# BCE를 쓰는 경사 하강법

표준화한 피처로 로지스틱 회귀를 학습하고 확률 출력에 BCE를 적용한다. 다음 페이지의 로짓 로스와 비교할 기준을 만든다.

## 1. 코드

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

# 피처 스케일링 (학습 데이터로 적합시켜 둘 다에 적용):
# 변환 후:
# X_train: (N_train, n_features)            float64
# X_test : (N_test,  n_features)            float64
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# torch 텐서로 바꾼다 (레이어와 연산의 PyTorch 기본값에 맞춰 float32로):
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
#    선형 레이어:
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
# 2) 로스와 최적화기
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
    # 순전파와 로스
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
#    학습/평가 동작이 다른 레이어(Dropout/BN)을 쓴다면 평가 모드로 바꾼다:
#      model.eval()
#    y_predicted: (N_test, 1) probabilities in [0,1]
#    y_predicted_cls: (N_test, 1) in {0,1} after thresholding via .round()
#                     - .round()는 0.5를 임계값으로 삼는 것과 같다:
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
    pass
```

**출력:**

```
Number of Positive Cases in 569 Patients of Original Data :  357
Number of Positive Cases in 455 Patients of Train Data :  288
Number of Positive Cases in 114 Patients of Test Data :  69
epoch: 10, loss = 0.2590
epoch: 20, loss = 0.0510
epoch: 30, loss = 0.0431
epoch: 40, loss = 0.0389
epoch: 50, loss = 0.0357
epoch: 60, loss = 0.0330
epoch: 70, loss = 0.0308
epoch: 80, loss = 0.0289
epoch: 90, loss = 0.0274
epoch: 100, loss = 0.0260
accuracy: 0.9561
```

## 2. 논의

### BCE의 입력 조건

모델 내부에서 선형 점수에 시그모이드를 적용하므로 `BCELoss`의 입력은 확률이다. 출력과 정답은 같은 모양이어야 하며, 이 예제의 정답은 0·1 실수형 텐서다. 로짓을 그대로 BCE에 넣거나 시그모이드를 두 번 적용하지 않는다.

### 전처리와 파라미터

피처 30개와 바이어스로 총 31개 파라미터를 학습한다. StandardScaler는 학습 데이터에서만 적합하고 테스트셋에는 같은 변환을 적용한다. 결과 확률은 데이터의 클래스 1 인코딩에 대응하며, 이 데이터에서는 양성 종양 라벨이다.

### 수치 안정성과 학습 상태

확률을 만든 뒤 로그 로스를 계산하면 큰 로짓에서 포화와 정밀도 문제가 생길 수 있다. 다음 절의 `BCEWithLogitsLoss`는 이를 합친 안정적인 계산을 제공한다. 로스와 정확도를 함께 관찰하되 테스트셋을 반복 튜닝에 사용하면 별도 최종 평가가 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
입력 피처 수가 $d$인 단일 선형 이진 분류기의 파라미터 수와 출력 모양을 설명하라.

</div>

??? success "풀이"
    웨이트 $d$개와 바이어스 1개로 총 $d+1$개다. 배치 입력 $(B,d)$에 대한 출력은 $(B,1)$이다. 시그모이드에는 학습 파라미터가 없다. 유방암 데이터의 피처 30개를 쓰는 경우에는 31개다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
옵티마이저를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 옵티마이저와 학습 수렴을 비교하라. 두 로스 곡선을 같은 그래프에 그려라.

</div>

??? success "풀이"
    옵티마이저를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 경사와 경사 제곱의 이동 평균으로 성분별 업데이트를 조정한다. 어느 옵티마이저가 더 빨리 수렴하는지는 문제와 러닝레이트에 달려 있다. 같은 초기 웨이트와 데이터 순서, 평가 기준을 사용하고 각 러닝레이트를 조정해 비교한다. 여러 시드의 변동과 검증 성능도 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
시그모이드를 적용한 모델에 `BCEWithLogitsLoss`를 그대로 바꾸어 사용하면 같은가?

</div>

??? success "풀이"
    같지 않다. 로짓 로스에는 시그모이드 이전 점수를 전달해야 한다. 모델의 확률 출력을 그대로 전달하면 이를 로짓으로 해석해 다른 목적과 경사를 계산한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 일반적인 계층 확장 코드가 이 페이지의 원래 클래스에 그대로 적용되는지 검토하라. 모델과 데이터셋의 역할, 생성자 인수, 입력·출력 모양과 비선형성에서 필요한 수정 사항을 설명하라.

</div>

??? success "풀이"
    아래 코드는 일반적인 ModuleList 구성 패턴의 일부이며 이 페이지 클래스의 완성된 대체 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    적용하려면 생성자와 `forward`를 함께 정의하고 마지막 출력 모양과 로스의 입력 조건을 유지해야 한다. 선형 계층만 여러 개 합성하면 여전히 하나의 선형 사상이므로 표현력을 늘리려면 적절한 비선형성이 필요하다. Dataset은 표본 접근 인터페이스이므로 계층을 등록할 대상이 아니며 별도 모델 클래스에 이 구성을 적용한다.

## 정리하며

모델 출력과 로스의 입력 조건을 일치시키고 데이터 분할, 전처리와 지표의 의미를 함께 확인한다. 기록된 실행 결과는 해당 설정의 예시이며 모든 환경의 수렴이나 일반화를 보장하지 않는다.
