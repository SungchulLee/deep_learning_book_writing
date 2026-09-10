# 선형 모델

선형 모델은 지도 학습의 기초이자 신경망의 기본 구성 요소이다. 심층 신경망의 모든 완전연결 층은 선형 모델 뒤에 비선형 함수가 붙은 것이므로, 선형 회귀와 로지스틱 회귀를 이해하는 것은 딥러닝을 이해하기 위한 선결 조건이다.

---

## 1. 정의

선형 모델은 출력을 입력 특징의 선형 결합으로 예측한다.

$$
\hat{y} = \mathbf{x}^\top \boldsymbol{\beta} + \beta_0
$$

회귀에서는 출력이 연속값이다. 분류에서는 이 선형 결합이 시그모이드(이진)나 소프트맥스(다중 클래스)를 거쳐 확률이 된다. 최소제곱 회귀의 닫힌 형태 해는 정규방정식이다.

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$

---

## 2. 설명

선형 회귀는 예측이라는 일 전체의 뼈대를 가장 적은 부품으로 보여 준다.

$$
\text{데이터} \longrightarrow \text{모델} \longrightarrow \text{예측}
$$

한 변수짜리로 적으면 흩어진 점들 사이를 지나는 직선 하나를 찾는 일이다.

$$
y = \alpha x + \beta
$$

!!! note "통계와 기계학습의 차이"
    이 뼈대는 통계학에서도 똑같다. 데이터를 받아 모델을 세우고 예측한다는 순서가 같으므로, **다른 것은 모델뿐이다.** 통계학이 해석 가능하고 가정이 뚜렷한 모델을 고르는 자리에서, 기계학습은 가정을 줄이고 표현력이 큰 모델을 고른다. 선형 회귀는 두 분야가 같은 모델을 쓰는 몇 안 되는 지점이며, 그래서 둘을 견주기에 가장 좋은 출발점이다.

**정칙화(regularization)** 는 큰 가중치에 벌점을 부과하여 과적합을 막는다.

- **릿지(L2)**: $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \alpha \|\boldsymbol{\beta}\|^2$을 최소화한다. 모든 계수를 0 쪽으로 줄이지만 정확히 0으로 만들지는 않는다. 특징들이 상관되어 있을 때 사용한다.
- **라쏘(L1)**: $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \alpha \|\boldsymbol{\beta}\|_1$을 최소화한다. 일부 계수를 정확히 0으로 만들어 특징 선택을 수행한다.
- **엘라스틱넷**: L1과 L2 벌점을 결합한다. 특징들이 상관되어 있으면서 희소성도 원할 때 사용한다.

**딥러닝과의 연결**: PyTorch의 `nn.Linear` 층 하나가 바로 이 모델을 구현한다. L2 정칙화는 최적화기의 `weight_decay`에 대응한다. L1 정칙화는 손실에 직접 더해 주어야 한다.

---

## 3. 예제

```python
import torch
import torch.nn as nn
import numpy as np

# ── 1. 참 모델을 정해 두고 그 모델에서 데이터를 만든다 ────────────────
# 참값을 알고 있어야 두 방법이 얼마나 잘 복원하는지 견줄 수 있다.
np.random.seed(42)          # 씨앗을 고정해 실행할 때마다 같은 결과가 나오게 한다
n, d = 200, 5               # 표본 200개, 특성 5개

X_np = np.random.randn(n, d)                    # 설계 행렬: 표준정규에서 뽑는다
true_w = np.array([3.0, -1.5, 0.0, 0.0, 2.0])   # 참 계수. 3번째와 4번째는 0이므로
                                                # 그 두 특성은 y와 아무 관계가 없다
y_np = X_np @ true_w + 0.5 * np.random.randn(n) # y = Xw + 잡음(표준편차 0.5)

# ── 2. 방법 A: 정규방정식으로 한 번에 푼다 ──────────────────────────
# 최소제곱해는 닫힌 형태로 존재하므로 반복이 필요 없다.
# lstsq는 (X^T X)^{-1} X^T y 를 수치적으로 안정하게 푼 것이다.
beta_hat = np.linalg.lstsq(X_np, y_np, rcond=None)[0]
print(f"True weights:     {true_w}")
print(f"Estimated weights:{np.round(beta_hat, 2)}")

# ── 3. 방법 B: 같은 문제를 경사 하강법으로 푼다 ─────────────────────
# 닫힌 형태가 있는데도 굳이 이렇게 푸는 까닭은, 이 절차가 신경망으로
# 그대로 이어지기 때문이다. 모델만 바꾸면 나머지 뼈대는 똑같다.
X_t = torch.tensor(X_np, dtype=torch.float32)
y_t = torch.tensor(y_np, dtype=torch.float32)

# bias=False: 데이터를 절편 없이 만들었으므로 편향 항을 두지 않는다
model = nn.Linear(d, 1, bias=False)

# weight_decay가 곧 L2(릿지) 정칙화이다. 손실에 alpha*||w||^2 를 더하는 것과 같다
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
loss_fn = nn.MSELoss()

for epoch in range(500):
    pred = model(X_t).squeeze()   # 순전파: 예측을 만든다
    loss = loss_fn(pred, y_t)     # 예측이 얼마나 틀렸는지 잰다

    optimizer.zero_grad()         # 지난 걸음의 기울기를 지운다(안 지우면 누적된다)
    loss.backward()               # 역전파: 손실을 가중치로 미분한다
    optimizer.step()              # 기울기 반대 방향으로 가중치를 한 걸음 옮긴다

# ── 4. 두 방법의 결과를 견준다 ──────────────────────────────────────
# 정칙화 때문에 경사 하강법 쪽 계수가 0에 아주 조금 더 가깝게 나온다.
learned = model.weight.detach().numpy().flatten()
print(f"PyTorch weights:  {np.round(learned, 2)}")
print(f"Final MSE loss:   {loss.item():.4f}")
```

**출력:**

```
True weights:     [ 3.  -1.5  0.   0.   2. ]
Estimated weights:[ 2.97 -1.47  0.01  0.03  2.03]
PyTorch weights:  [ 2.96 -1.46  0.01  0.03  2.01]
Final MSE loss:   0.2655
```

---

## 연습문제

**연습문제 1.**
MSE 손실 $\ell(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$의 경사를 0으로 두어 정규방정식 $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$을 유도하라.

??? success "연습문제 1 풀이"
    전개하면 $\ell = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{y}^\top\mathbf{y} - 2\boldsymbol{\beta}^\top\mathbf{X}^\top\mathbf{y} + \boldsymbol{\beta}^\top\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta}$이다. 경사를 구하면 $\nabla_{\boldsymbol{\beta}}\ell = -2\mathbf{X}^\top\mathbf{y} + 2\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta}$이다. 0으로 두면 $\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^\top\mathbf{y}$이다. $\mathbf{X}^\top\mathbf{X}$가 가역이면 $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$이다. $\square$

---

**연습문제 2.**
릿지 회귀의 기하학적 해석을 설명하라. $\alpha \to 0$일 때와 $\alpha \to \infty$일 때 해는 어떻게 되는가?

??? success "연습문제 2 풀이"
    릿지 회귀는 $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \alpha\|\boldsymbol{\beta}\|^2$을 최소화한다. 닫힌 형태 해는 $\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$이다. 기하학적으로 릿지는 $\boldsymbol{\beta}$가 $\ell_2$ 공 안에 있도록 제약한다. $\alpha \to 0$이면 릿지는 OLS(정칙화 없음)로 수렴한다. $\alpha \to \infty$이면 벌점이 지배하여 $\hat{\boldsymbol{\beta}} \to \mathbf{0}$이 되고 모델은 $\mathbf{y}$의 평균을 예측한다. $\alpha\mathbf{I}$를 더하면 $\mathbf{X}^\top\mathbf{X}$가 특이행렬이어도 가역성이 보장된다.

---

**연습문제 3.**
선형 회귀에서 L2 정칙화가 SGD의 `weight_decay`와 동등함을 보여라. 구체적으로 감쇠 계수 $\lambda$인 가중치 감쇠의 경사 갱신이 $\alpha = \lambda$인 L2 정칙화 손실의 경사와 일치함을 보여라.

??? success "연습문제 3 풀이"
    L2 정칙화 손실은 $\ell_{\text{reg}} = \ell(\boldsymbol{\beta}) + \frac{\alpha}{2}\|\boldsymbol{\beta}\|^2$이다. 그 경사는 $\nabla \ell_{\text{reg}} = \nabla \ell + \alpha \boldsymbol{\beta}$이다. SGD 갱신은 $\boldsymbol{\beta} \leftarrow \boldsymbol{\beta} - \eta(\nabla \ell + \alpha \boldsymbol{\beta}) = (1 - \eta\alpha)\boldsymbol{\beta} - \eta \nabla \ell$이다. 계수 $\lambda$인 가중치 감쇠는 $\boldsymbol{\beta} \leftarrow (1 - \eta\lambda)\boldsymbol{\beta} - \eta \nabla \ell$을 준다. $\alpha = \lambda$일 때 둘은 동일하다. 참고: 이 동등성은 SGD에서는 정확히 성립하지만 Adam에서는 그렇지 않다. Adam에서는 분리된 가중치 감쇠(AdamW)가 L2 정칙화와 다르다. $\square$

---

**연습문제 4.**
어떤 데이터셋에 표본이 $n = 50$개, 특징이 $d = 100$개 있다. 정규방정식이 실패하는 이유를 설명하고 해결책 두 가지를 제안하라.

??? success "연습문제 4 풀이"
    정규방정식은 $(100 \times 100)$ 행렬인 $\mathbf{X}^\top\mathbf{X}$의 역행렬을 필요로 한다. 표본이 $n = 50$개뿐이므로 $\text{rank}(\mathbf{X}) \leq 50 < 100$이고 따라서 $\mathbf{X}^\top\mathbf{X}$는 특이행렬(비가역)이다. 해결책: (1) **릿지 회귀**: $\alpha\mathbf{I}$를 더하면 임의의 $\alpha > 0$에 대해 $(\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I})$가 가역이 된다. (2) **라쏘(L1 정칙화)**: 많은 계수를 0으로 만들어 사실상 $\leq 50$개의 특징을 선택한다. (3) **PCA/특징 선택**: 적합 전에 $d$를 $n$보다 작게 줄인다.

---

**연습문제 5.**
편향을 포함한 `nn.Linear(784, 10)`의 매개변수 개수를 계산하라. 이를 `nn.Linear(784, 128)` 다음에 `nn.Linear(128, 10)`이 오는 2층 신경망과 비교하라.

??? success "연습문제 5 풀이"
    단일 층: 가중치 $784 \times 10 = 7{,}840$개, 편향 $10$개. 총 $7{,}850$개. 2층: 첫 층 $784 \times 128 + 128 = 100{,}480$개, 둘째 층 $128 \times 10 + 10 = 1{,}290$개. 총 $101{,}770$개. 2층 신경망은 매개변수가 $\approx 13\times$ 많지만 (층 사이에 ReLU를 두면) 비선형 결정 경계를 학습할 수 있는 반면, 단일 층은 선형 경계로 제한된다. 대부분의 실제 문제에서는 이 추가 용량이 매개변수 비용을 치를 만한 가치가 있다.

## 정리하며

이 마당은 정의、설명、예제을 차례로 짚었다.
