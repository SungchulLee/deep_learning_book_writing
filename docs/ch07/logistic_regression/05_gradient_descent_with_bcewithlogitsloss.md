# BCEWithLogitsLoss를 쓰는 경사 하강법

시그모이드를 모델에서 분리하고 로짓을 `BCEWithLogitsLoss`에 직접 전달한다. 확률 변환과 학습 손실을 나누는 이유를 이해한다.

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

**출력:**

```
epoch: 10, loss = 0.6835
epoch: 20, loss = 0.5401
epoch: 30, loss = 0.4533
epoch: 40, loss = 0.3959
epoch: 50, loss = 0.3551
epoch: 60, loss = 0.3245
epoch: 70, loss = 0.3006
epoch: 80, loss = 0.2814
epoch: 90, loss = 0.2654
epoch: 100, loss = 0.2519
accuracy: 0.9035
```

## 2. 논의

### 모델은 로짓을 반환한다

선형 계층의 출력은 제한 없는 실수 점수다. `BCEWithLogitsLoss`는 이 점수에서 시그모이드와 로그 손실을 안정적으로 결합한다. 모델이나 손실 앞에 시그모이드를 다시 추가하면 같은 목적을 계산하지 않는다. 정답은 로짓과 같은 모양의 0·1 실수형 텐서다.

### 예측과 확률

추론에서 확률이 필요하면 로짓에 시그모이드를 한 번 적용한다. 확률 0.5 기준은 로짓 0 기준과 같으며 이 기준이 모든 오류 비용과 클래스 비율에서 최적이라는 뜻은 아니다. 로짓의 부호가 같아도 크기가 다르면 손실과 확률 보정은 달라질 수 있다.

### 비교 조건

앞 절과 같은 초기 파라미터, 데이터 분할, 전처리와 업데이트를 맞추면 두 손실의 수학적 관계를 점검할 수 있다. 다만 극단적인 점수에서는 수치 경로가 달라질 수 있다. 유한한 출력이 나온다는 사실과 안정적인 학습률로 수렴했다는 사실을 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
입력 특징 수가 $d$인 단일 선형 이진 분류기의 파라미터 수와 출력 모양을 설명하라.

</div>

??? success "연습문제 1 풀이"
    가중치 $d$개와 편향 1개로 총 $d+1$개다. 배치 입력 $(B,d)$에 대한 출력은 $(B,1)$이다. 시그모이드에는 학습 파라미터가 없다. 유방암 데이터의 특징 30개를 쓰는 경우에는 31개다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
옵티마이저를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 옵티마이저와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

</div>

??? success "연습문제 2 풀이"
    옵티마이저를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 경사와 경사 제곱의 이동 평균으로 성분별 업데이트를 조정한다. 어느 옵티마이저가 더 빨리 수렴하는지는 문제와 학습률에 달려 있다. 같은 초기 가중치와 데이터 순서, 평가 기준을 사용하고 각 학습률을 조정해 비교한다. 여러 시드의 변동과 검증 성능도 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
모델의 확률이 아니라 로짓을 반환하면 추론에서 어떤 변환이 필요한가?

</div>

??? success "연습문제 3 풀이"
    확률을 보고 싶을 때 시그모이드를 한 번 적용한다. 확률 0.5 임곗값의 클래스 결정은 로짓을 0과 비교해도 같다. 학습 손실에는 로짓을 직접 전달한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 일반적인 계층 확장 코드가 이 페이지의 원래 클래스에 그대로 적용되는지 검토하라. 모델과 데이터셋의 역할, 생성자 인수, 입력·출력 모양과 비선형성에서 필요한 수정 사항을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 일반적인 ModuleList 구성 패턴의 일부이며 이 페이지 클래스의 완성된 대체 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    적용하려면 생성자와 `forward`를 함께 정의하고 마지막 출력 모양과 손실의 입력 조건을 유지해야 한다. 선형 계층만 여러 개 합성하면 여전히 하나의 선형 사상이므로 표현력을 늘리려면 적절한 비선형성이 필요하다. Dataset은 표본 접근 인터페이스이므로 계층을 등록할 대상이 아니며 별도 모델 클래스에 이 구성을 적용한다.

## 정리하며

모델 출력과 손실의 입력 조건을 일치시키고 데이터 분할, 전처리와 지표의 의미를 함께 확인한다. 기록된 실행 결과는 해당 설정의 예시이며 모든 환경의 수렴이나 일반화를 보장하지 않는다.
