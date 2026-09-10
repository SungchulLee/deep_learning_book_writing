# 후버 손실
후버 손실(매끄러운 L1 손실이라고도 한다)은 구간별 정의로 MSE와 MAE의 장점을 결합한다. 작은 오차에서는 이차, 큰 오차에서는 선형이다. 이 혼합 설계는 최적점 근처에서 MSE의 매끄러운 기울기를 주면서 이상점에 대한 MAE의 견고성도 물려받는다. 1964년 피터 후버가 로버스트 통계학의 주춧돌로 제안했으며, 물체 검출의 표준 회귀 손실이자 잡음 섞인 실제 데이터의 실용적인 기본값이 되었다.

---

## 1. 수학적 정의

$$\mathcal{L}_{\text{Huber}}(r) = \begin{cases} 
\frac{1}{2}r^2 & \text{if } |r| \leq \delta \\[4pt]
\delta |r| - \frac{1}{2}\delta^2 & \text{if } |r| > \delta
\end{cases}$$

여기서 $r = y - \hat{y}$은 잔차이고 $\delta > 0$은 **전환 문턱값**이다(PyTorch의 `SmoothL1Loss`에서는 `beta`이라 부른다).

두 가지는 $|r| = \delta$에서 값과 일계도함수가 모두 맞도록 설계되어 손실 함수가 매끄럽게(연속적으로 미분 가능하게) 된다.

- **$|r| = \delta$에서:** 이차 쪽은 $\frac{1}{2}\delta^2$을, 선형 쪽은 $\delta \cdot \delta - \frac{1}{2}\delta^2 = \frac{1}{2}\delta^2$을 준다. ✓
- **$|r| = \delta$에서의 도함수:** 이차 쪽은 $\delta$을, 선형 쪽은 크기가 $\delta$인 $\delta \cdot \text{sign}(r)$을 준다. ✓

---

## 2. 경사 분석

$$\frac{\partial \mathcal{L}_{\text{Huber}}}{\partial \hat{y}} = \begin{cases}
\hat{y} - y & \text{if } |y - \hat{y}| \leq \delta \\[4pt]
\delta \cdot \text{sign}(\hat{y} - y) & \text{if } |y - \hat{y}| > \delta
\end{cases}$$

이 기울기의 모양은 두 세계의 좋은 점을 결합한다.

**최적점 근처** ($|r| \leq \delta$): 기울기가 MSE처럼 오차에 비례한다. 덕분에 정밀하게 다듬고 매끄럽게 수렴할 수 있다. $r \to 0$일 때 기울기가 매끄럽게 사라지므로 MAE의 진동 문제를 피한다.

**최적점에서 멀 때** ($|r| > \delta$): 기울기의 크기가 MAE처럼 $\delta$으로 제한된다. 덕분에 이상점이 학습을 흔드는 폭발적인 기울기를 만들지 못한다.

### 기울기의 비교

| 오차의 영역 | MSE 기울기 | MAE 기울기 | 후버 기울기 |
|-------------|--------------|--------------|----------------|
| 작은 오차 ($|r| \ll \delta$) | $\propto r$ (줄어듦) | $\pm 1/m$ (상수) | $\propto r$ (줄어듦) |
| 문턱값에서 ($|r| = \delta$) | $\propto \delta$ | $\pm 1/m$ (상수) | $\pm \delta$ (연속) |
| 큰 오차 ($|r| \gg \delta$) | $\propto r$ (커짐!) | $\pm 1/m$ (상수) | $\pm \delta$ (유계) |

---

## 3. delta의 구실

문턱값 $\delta$은 이차 구간과 선형 구간 사이의 전환을 조절하므로 손실의 성격을 정한다.

- **$\delta \to \infty$**: 모든 오차가 이차 구간에 들어간다 → 후버 손실 $\approx$ MSE
- **$\delta \to 0$**: 모든 오차가 선형 구간에 들어간다 → 후버 손실 $\approx$ MAE
- **중간 $\delta$**: 혼합 거동. $\delta$보다 작은 오차는 이차로, 큰 오차는 선형으로 다룬다

최적의 $\delta$은 예상되는 잡음의 척도에 달렸다. 좋은 어림 규칙은 $\delta$을 "보통" 오차의 예상 크기로 두어 진짜 이상점만 선형 구간에 들어가게 하는 것이다.

---

## 4. 로버스트 통계학과의 관계

후버의 본디 동기는 정규 잡음 아래에서 평균만큼 효율적이면서도 오염에 훨씬 잘 견디는 **M 추정량**을 정의하는 것이었다. 후버 손실은 관측 하나가 미치는 영향을 제한하여 이를 이룬다.

추정량의 **영향 함수**는 관측 하나가 추정값을 얼마나 바꾸는지를 잰다. (MSE에 최적인) 평균의 영향 함수는 유계가 아니다. 이상점 하나가 추정값을 얼마든지 옮길 수 있다. (MAE에 최적인) 중앙값의 영향 함수는 유계이지만 정규 잡음 아래에서 추정량이 비효율적이다. 후버 손실은 절충을 준다. (중앙값처럼) 영향이 유계이면서 (평균처럼) 정규 잡음 아래에서 최적에 가까운 효율을 낸다.

수치로 보면 후버 추정량은 정규 잡음 아래에서 평균 효율의 약 95%를 내면서 10% 남짓의 오염까지 견딘다.

---

## 5. PyTorch: `nn.SmoothL1Loss`과 `nn.HuberLoss`

PyTorch는 서로 밀접한 두 가지 구현을 제공한다.

### `nn.SmoothL1Loss`

PyTorch의 기본값이며 `beta` 매개변수($\delta$에 해당한다)를 쓴다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

actual = torch.tensor([85.0, 90.0, 88.0, 92.0, 15.0])
predicted = torch.tensor([84.0, 89.0, 87.0, 91.0, 87.0])

# 기본값: beta=1.0
smooth_l1 = nn.SmoothL1Loss()
loss = smooth_l1(predicted, actual)
print(f"Smooth L1 Loss (beta=1.0): {loss.item():.4f}")

# 사용자 지정 beta (전환 문턱값)
smooth_l1_beta5 = nn.SmoothL1Loss(beta=5.0)
loss_beta5 = smooth_l1_beta5(predicted, actual)
print(f"Smooth L1 Loss (beta=5.0): {loss_beta5.item():.4f}")
```

**출력:**

```
Smooth L1 Loss (beta=1.0): 14.7000
Smooth L1 Loss (beta=5.0): 13.9800
```

!!! note "배율 관례"
    `nn.SmoothL1Loss`은 이차 쪽을 `beta`으로 나누므로 식은 다음과 같다.

    $$\text{SmoothL1}(r) = \begin{cases} \frac{r^2}{2\beta} & |r| < \beta \\ |r| - \frac{\beta}{2} & |r| \geq \beta \end{cases}$$

    이는 이차 쪽에서 $1/\delta$배를 빼면 $\delta = \beta$인 표준 후버 손실과 같다.

### `nn.HuberLoss`

PyTorch 1.9에 들어왔으며 `delta` 매개변수를 쓰는 표준 후버 식이다.

```python
huber = nn.HuberLoss(delta=1.0)
loss = huber(predicted, actual)
print(f"Huber Loss (delta=1.0): {loss.item():.4f}")
```

**출력:**

```
Huber Loss (delta=1.0): 14.7000
```

### 둘의 비교

| 특징 | `nn.SmoothL1Loss` | `nn.HuberLoss` |
|---------|-------------------|-----------------|
| 매개변수 이름 | `beta` | `delta` |
| 기본값 | 1.0 | 1.0 |
| 이차 쪽 | $r^2 / (2\beta)$ | $r^2 / 2$ |
| 유래 | 물체 검출 (Fast R-CNN) | 로버스트 통계학 (Huber, 1964) |

`beta=delta=1`이면 둘은 이차 쪽에서 $\delta$배만큼 다르다. 대부분의 쓰임에서는 어느 쪽이든 되므로 분야의 관례에 따라 고르라(물체 검출은 SmoothL1을, 일반 회귀는 Huber를 쓴다).

### 줄이는 방식의 선택

```python
# 평균 (기본값): 모든 원소의 평균
loss_mean = nn.SmoothL1Loss(reduction='mean')(predicted, actual)

# 합: 모든 원소 손실의 합
loss_sum = nn.SmoothL1Loss(reduction='sum')(predicted, actual)

# 없음: 원소별 손실
loss_none = nn.SmoothL1Loss(reduction='none')(predicted, actual)
print(f"Per-element losses: {loss_none}")
```

**출력:**

```
Per-element losses: tensor([ 0.5000,  0.5000,  0.5000,  0.5000, 71.5000])
```

---

## 6. 구간별 분석

```python
# 각 오차가 어느 구간에 들어가는지 보이기 (beta=1.0)
errors = actual - predicted
for i, error in enumerate(errors):
    abs_error = abs(error.item())
    if abs_error < 1.0:
        regime = "quadratic (MSE-like)"
        loss_val = 0.5 * error.item()**2
    else:
        regime = "linear (MAE-like)"
        loss_val = abs_error - 0.5
    print(f"Error {i+1}: {error.item():7.1f} → {regime}, loss={loss_val:.2f}")
```

**출력:**

```
Error 1:     1.0 → linear (MAE-like), loss=0.50
Error 2:     1.0 → linear (MAE-like), loss=0.50
Error 3:     1.0 → linear (MAE-like), loss=0.50
Error 4:     1.0 → linear (MAE-like), loss=0.50
Error 5:   -72.0 → linear (MAE-like), loss=71.50
```

---

## 7. 비교 실험

```python
# MSE, MAE, 후버로 보는 세 가지 상황
scenarios = {
    "Clean data": (
        torch.tensor([85.0, 90.0, 88.0, 92.0, 87.0]),
        torch.tensor([84.0, 89.0, 87.0, 91.0, 86.0])
    ),
    "Moderate errors": (
        torch.tensor([85.0, 90.0, 88.0, 92.0, 87.0]),
        torch.tensor([80.0, 85.0, 83.0, 87.0, 82.0])
    ),
    "With outlier": (
        torch.tensor([85.0, 90.0, 88.0, 92.0, 15.0]),
        torch.tensor([84.0, 89.0, 87.0, 91.0, 87.0])
    )
}

for name, (actual, pred) in scenarios.items():
    mse = F.mse_loss(pred, actual)
    mae = F.l1_loss(pred, actual)
    huber = F.smooth_l1_loss(pred, actual)
    print(f"\n{name}:")
    print(f"  MSE:       {mse.item():10.2f}")
    print(f"  MAE:       {mae.item():10.2f}")
    print(f"  Smooth L1: {huber.item():10.2f}")
```

**출력:**

```

Clean data:
  MSE:             1.00
  MAE:             1.00
  Smooth L1:       0.50

Moderate errors:
  MSE:            25.00
  MAE:             5.00
  Smooth L1:       4.50

With outlier:
  MSE:          1037.60
  MAE:            15.20
  Smooth L1:      14.70
```

---

## 8. delta 고르기

**작은 $\delta$ (예: 0.1~0.5):** 이상점을 더 세게 잘라 낸다. 대부분의 오차에서 MAE처럼 움직인다. 이상점이 잦고 잡음의 바닥이 낮을 때 쓴다.

**중간 $\delta$ (예: 1.0):** 균형 잡힌 거동이며 표준 기본값이다. 대부분의 문제에서 출발점으로 쓴다.

**큰 $\delta$ (예: 5.0~10.0):** 잘라 내기 전에 더 큰 오차를 허용한다. 대부분의 오차에서 MSE처럼 움직인다. 이상점이 드물고 보통 오차에 MSE 같은 정밀함을 원할 때 쓴다.

**데이터에 기반한 방법:** 먼저 MSE로 적합한 뒤 학습 잔차의 중앙절대편차(MAD)로 $\delta$을 정한다. $\delta = 1.4826 \cdot \text{MAD}$이다(1.4826이라는 인수는 정규 잡음 아래에서 MAD를 표준편차와 맞추어 준다).

---

## 9. 응용

**물체 검출.** SmoothL1은 Faster R-CNN, SSD, YOLO 계열에서 경계 상자 회귀의 표준 손실이다. 경계 상자의 목푯값은 좌표 차이가 클 수 있으므로(어려운 검출에서 오는 이상점) 안정적인 학습에 선형 꼬리가 꼭 필요하다.

**강화 학습.** 후버 손실은 심층 Q 학습(DQN)의 시간차 오차에 자주 쓰인다. 목푯값이 현재 신경망에서 부트스트랩되어 잡음이 많고 큰 잔차를 낼 수 있기 때문이다.

**금융 모형화.** 잡음의 분포가 두꺼운 꼬리를 갖는 수익률 예측은 후버의 유계 영향 함수의 덕을 본다.

---

## 10. 핵심 정리

후버 손실은 MSE와 MAE를 원칙 있게 결합한 것으로, 최적점 근처(이차 구간)에서는 매끄러운 기울기를, 큰 오차(선형 구간)에서는 유계인 기울기를 준다. 문턱값 매개변수 $\delta$이 전환을 조절하며 "보통" 오차의 예상 척도를 반영해야 한다. PyTorch에서 (물체 검출에서 온) `nn.SmoothL1Loss`과 (로버스트 통계학에서 온) `nn.HuberLoss`은 같은 착상을 배율만 조금 달리하여 구현한다. 이 손실은 정규 잡음 아래에서 최적에 가까운 통계적 효율을 내면서 오염에도 견디는데, 이는 순수한 MSE나 MAE가 저마다 줄 수 없는 절충이다.

---

## 연습문제

**연습문제 1.**
후버 손실을 유도하고 $\delta$에서 이차에서 선형으로 넘어감을 보여라.

??? success "연습문제 1 풀이"
    후버 손실은 $L_\delta(r) = \begin{cases} \frac{1}{2}r^2 & |r| \leq \delta \\ \delta|r| - \frac{1}{2}\delta^2 & |r| > \delta \end{cases}$이다. 잔차가 작으면($|r| \leq \delta$) MSE처럼 이차이고, 잔차가 크면 MAE처럼 선형이다. 전환은 매끄럽다($|r| = \delta$에서 일계도함수가 연속이다).

---

**연습문제 2.**
후버 손실을 바닥부터 구현하고 `torch.nn.HuberLoss`과 맞는지 확인하라.

??? success "연습문제 2 풀이"
    ```python
    def huber_loss(pred, target, delta=1.0):
        r = pred - target
        return torch.where(r.abs() <= delta,
                          0.5 * r**2,
                          delta * r.abs() - 0.5 * delta**2).mean()
    ```

---

**연습문제 3.**
후버 손실의 확률적 해석을 설명하라.

??? success "연습문제 3 풀이"
    후버 손실은 0 근처에서는 정규분포이고 꼬리에서는 라플라스분포인 분포 아래에서의 최대가능도 추정에 해당한다. 그래서 (잔차가 큰) 이상점에 견디면서도 (잔차가 작은) 얌전한 데이터에서는 효율을 유지한다.

---

**연습문제 4.**
$\delta$의 선택은 편향과 견고성의 절충에 어떤 영향을 주는가?

??? success "연습문제 4 풀이"
    $\delta$이 크면 MSE처럼 움직인다(효율적이지만 이상점에 민감하다). $\delta$이 작으면 MAE처럼 움직인다(견고하지만 정규 잡음에서는 덜 효율적이다). 대표적인 기본값은 $\delta = 1.0$이다. 교차 검증이나 보통 잔차의 척도가 선택을 이끌어야 한다.

## 정리하며

이 마당은 수학적 정의、경사 분석、delta의 구실、로버스트 통계학과의 관계을 차례로 짚었다.
