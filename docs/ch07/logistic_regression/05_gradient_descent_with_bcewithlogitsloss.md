# BCEWithLogitsLoss를 쓰는 경사 하강법

시그모이드를 모델에서 분리하고 로짓(시그모이드에 넣기 전의 선형 점수 $w^\top x+b$)을 `BCEWithLogitsLoss`에 직접 전달한다. 확률 변환과 트레인 로스를 나누는 이유를 이해한다.

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

# 이 쪽이 싣는 수가 돌릴 때마다 달라지지 않게 시드를 고정한다.
torch.manual_seed(0)
np.random.seed(0)


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

# 피처를 표준화한다 (트레인 데이터로 적합시켜 둘 다에 적용)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# torch.float32로
X_train = torch.from_numpy(X_train.astype(np.float32))  # (N_train, n_features)
X_test  = torch.from_numpy(X_test.astype(np.float32))   # (N_test,  n_features)
y_train = torch.from_numpy(y_train.astype(np.float32))  # (N_train,)
y_test  = torch.from_numpy(y_test.astype(np.float32))   # (N_test,)

# BCEWithLogits용으로 목표의 형태를 바꾼다: (N, 1)
y_train = y_train.view(-1, 1)   # (N_train, 1)
y_test  = y_test.view(-1, 1)    # (N_test,  1)

# ---------------------------------------------------------------------------
# 1) 모델 (로짓 버전)
#    - 선형 레이어에서 **로짓**(날것의 점수)을 반환한다.
#    - 여기서 시그모이드를 적용하지 말 것. 안정성을 위해 BCEWithLogitsLoss를 쓴다.
#      (시그모이드와 BCE를 수치적으로 안정한 연산 하나로 합친 것이다.)
#    형태:
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
# 2) 로스와 옵티마이저
#    - nn.BCEWithLogitsLoss를 쓴다 (확률이 아니라 로짓을 받는다).
#    - 클래스가 불균형하면 pos_weight=tensor([w])를 넘길 수 있다.
# ---------------------------------------------------------------------------
num_epochs = 100
learning_rate = 0.01
criterion = nn.BCEWithLogitsLoss()                 # ← stable sigmoid + BCE
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ---------------------------------------------------------------------------
# 3) 트레인 루프
#    반복마다의 형태:
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
#    - 클래스를 얻으려면 로짓을 0으로 임계값 처리한다 (sigmoid(0)=0.5이므로).
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
epoch: 10, loss = 0.5531
epoch: 20, loss = 0.4631
epoch: 30, loss = 0.4033
epoch: 40, loss = 0.3608
epoch: 50, loss = 0.3288
epoch: 60, loss = 0.3039
epoch: 70, loss = 0.2838
epoch: 80, loss = 0.2673
epoch: 90, loss = 0.2534
epoch: 100, loss = 0.2415
accuracy: 0.8947
```

## 2. 논의

### 모델은 로짓을 반환한다

선형 레이어의 출력은 제한 없는 실수 점수다. `BCEWithLogitsLoss`는 이 점수에 시그모이드와 로그 로스를 하나로 합쳐 안정적으로 계산한다. 모델 끝에 시그모이드를 다시 추가하면 시그모이드가 두 번 걸려 같은 목적 함수를 계산하지 않는다. 정답은 로짓과 같은 형태의 0·1 실수형 텐서다.

### 예측과 확률

추론에서 확률이 필요하면 로짓에 시그모이드를 한 번 적용한다. 확률 0.5 기준은 로짓 0 기준과 같으며 이 기준이 모든 오류 비용과 클래스 비율에서 최적이라는 뜻은 아니다. 로짓의 부호가 같아도 크기가 다르면 로스와 확률 보정은 달라질 수 있다.

### 비교 조건

이 쪽의 시험 정확도 0.8947을 앞 쪽 「BCE를 쓰는 경사 하강법」의 0.9649와 그대로 견주면 안 된다. 두 쪽은 로스만 다른 것이 아니다. 러닝레이트가 앞 쪽은 10, 이 쪽은 0.01이고 초기화 시드도 다르다(1234와 0). 이 쪽의 로스는 100에포크 뒤에도 0.2415로 아직 내려가는 중이다. 그러므로 이 차이를 로스 함수 탓으로 읽을 수 없다. 러닝레이트와 에포크에 따라 정확도가 얼마나 달라지는지는 다음 쪽 「경사 하강법 심화」가 잰다.

앞 쪽과 초기 파라메터, 데이터 분할, 전처리, 러닝레이트를 똑같이 맞추면 두 로스가 수학적으로 같은 값을 내는지 점검할 수 있다. 다만 극단적인 점수에서는 수치 계산 경로가 달라 값이 벌어질 수 있다. 유한한 출력이 나온다는 사실과 안정적인 러닝레이트로 수렴했다는 사실을 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
입력 피처 수가 $d$인 단일 선형 이진 분류기의 파라메터 수와 출력 형태를 설명하라.

</div>

??? success "풀이"
    웨이트 $d$개와 바이어스 1개로 총 $d+1$개다. 배치 입력 $(B,d)$에 대한 출력은 $(B,1)$이다. 시그모이드에는 트레인 파라메터가 없다. 유방암 데이터의 피처 30개를 쓰는 경우에는 31개다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
옵티마이저를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 옵티마이저와 트레인 수렴을 비교하라. 두 로스 곡선을 같은 그래프에 그려라.

</div>

??? success "풀이"
    옵티마이저를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 그래디언트와 그래디언트 제곱의 이동 평균으로 성분별 업데이트를 조정한다. 어느 옵티마이저가 더 빨리 수렴하는지는 문제와 러닝레이트에 달려 있다. 같은 초기 웨이트와 데이터 순서, 평가 기준을 사용하고 각 러닝레이트를 조정해 비교한다. 여러 시드의 변동과 검증 성능도 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
모델이 확률이 아니라 로짓을 반환하면 추론에서 어떤 변환이 필요한가?

</div>

??? success "풀이"
    확률을 보고 싶을 때 시그모이드를 한 번 적용한다. 확률 0.5 임곗값의 클래스 결정은 로짓을 0과 비교해도 같다. 트레인 로스에는 로짓을 직접 전달한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
트레인을 마친 뒤 시험 집합의 `logits_test`와 `y_test`로 `criterion(logits_test, y_test)`와 `nn.BCELoss()(torch.sigmoid(logits_test), y_test)`를 견주어라. 이어서 로짓에 100을 곱한 `100 * logits_test`로 같은 비교를 되풀이하고, 두 값이 갈리는 까닭을 설명하라. 로짓 $z$와 정답 $y$에 대한 안정한 식 $\max(z,0)-zy+\log(1+e^{-|z|})$의 평균도 함께 계산하라.

</div>

??? success "풀이"
    아래 코드는 이 쪽의 프로그램 끝에 이어 붙여 돌린다.
    ```python
    bce = nn.BCELoss()
    with torch.no_grad():
        for scale in [1., 100.]:
            z = scale * logits_test                              # (114, 1)
            with_logits = criterion(z, y_test)
            two_steps = bce(torch.sigmoid(z), y_test)
            stable = (z.clamp(min=0) - z * y_test
                      + torch.log1p(torch.exp(-z.abs()))).mean()
            print(f"scale={scale:5.0f}  BCEWithLogitsLoss={with_logits.item():.4f}  "
                  f"sigmoid+BCELoss={two_steps.item():.4f}  직접 계산={stable.item():.4f}")
    ```
    ```
    scale=    1  BCEWithLogitsLoss=0.2770  sigmoid+BCELoss=0.2770  직접 계산=0.2770
    scale=  100  BCEWithLogitsLoss=5.9059  sigmoid+BCELoss=8.8255  직접 계산=5.9059
    ```
    로짓이 작을 때는 세 값이 0.2770으로 같다. 두 로스는 수학적으로 같은 함수이기 때문이다. 로짓에 100을 곱하면 절댓값이 최대 1195쯤까지 커지고, 시험 표본 114개 가운데 108개에서 `torch.sigmoid`가 float32로 정확히 0이나 1이 된다. 그 가운데 틀린 쪽으로 포화한 표본이 9개다. 이 표본의 참 로스는 $|z|$(수백)인데, `BCELoss`는 $\log 0$을 $-100$으로 잘라 한 표본에 100까지만 매긴다. 그래서 `sigmoid+BCELoss`의 8.8255는 참값 5.9059와 다른 수다. 값만 틀리는 것이 아니라, 잘린 구간에서는 로짓이 더 나빠져도 로스가 변하지 않아 그래디언트도 사라진다.

    `BCEWithLogitsLoss`는 확률을 거치지 않고 $\max(z,0)-zy+\log(1+e^{-|z|})$를 계산한다. 지수의 인수가 늘 0 이하라 넘치지 않고, 로그의 인수는 1과 2 사이라 $\log 0$이 나오지 않는다. 직접 계산한 값이 두 경우 모두 `criterion`과 일치한다.

## 정리하며

모델 출력과 로스의 입력 조건을 일치시키고 데이터 분할, 전처리와 지표의 의미를 함께 확인한다. 기록된 실행 결과는 해당 설정의 예시이며 모든 환경의 수렴이나 일반화를 보장하지 않는다.
