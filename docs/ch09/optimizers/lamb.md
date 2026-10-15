# LAMB

LAMB는 Adam 계열의 좌표별 갱신에 층 또는 파라미터 텐서별 신뢰 비율을 더하는 옵티마이저이다. 원 논문은 특정 분산 하드웨어와 BERT 학습 설정에서 큰 배치의 효율을 보고했다. 그 시간이나 안정성을 다른 환경의 보장으로 읽지 않는다.

## 1. 알고리즘

LAMB는 [Adam](adam.md) 위에 세워진다. 각 시각 $t$에서 기울기를 $g_t = \nabla_\theta L(\theta_t)$이라 하자. 매개변수 텐서가 $\theta_t^{(l)}$인 각 층 $l$에 대해 LAMB는 다음 단계를 수행한다. 원소별 연산은 모두 층 안에서 이루어진다.

**1단계.** (Adam과 같이) 치우친 일차 모멘트와 이차 모멘트의 추정값을 갱신한다.

$$
m_t = \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t
$$

$$
v_t = \beta_2 \, v_{t-1} + (1 - \beta_2) \, g_t^2
$$

**2단계.** 편향을 보정한 모멘트를 계산한다.

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**3단계.** Adam형 방향에 L2 계수 항(계수 $\lambda$)을 더한 층별 갱신 방향 $r_t^{(l)}$을 계산한다.

$$
r_t^{(l)} = \frac{\hat{m}_t^{(l)}}{\sqrt{\hat{v}_t^{(l)}} + \epsilon} + \lambda \, \theta_t^{(l)}
$$

**4단계.** $\ell_2$ 노름으로 **신뢰 비율** $\phi^{(l)}$을 계산한다.

$$
\phi^{(l)} = \frac{\|\theta_t^{(l)}\|_2}{\|r_t^{(l)}\|_2}
$$

$\|\theta_t^{(l)}\|_2 = 0$이거나 $\|r_t^{(l)}\|_2 = 0$이면 신뢰 비율을 $1$로 둔다.

**5단계.** 층별로 배율을 조정한 갱신을 적용한다.

$$
\theta_{t+1}^{(l)} = \theta_t^{(l)} - \eta \, \phi^{(l)} \, r_t^{(l)}
$$

여기서 $\eta$은 전역 학습률, $\beta_1$과 $\beta_2$은 모멘트의 감쇠율(보통 $0.9$과 $0.999$), $\epsilon$은 수치 안정성을 위한 작은 상수이다.

---

## 2. 신뢰 비율의 직관

신뢰 비율 $\phi^{(l)} = \|\theta_t^{(l)}\|_2 / \|r_t^{(l)}\|_2$은 층마다 갱신의 배율을 조정하여, 날 기울기의 크기와 무관하게 상대적인 변화 $\|\Delta\theta^{(l)}\|_2 / \|\theta^{(l)}\|_2$이 학습률 $\eta$만으로 정해지게 한다. 매개변수가 작은 층은 그에 비례해 작은 갱신을, 큰 층은 큰 갱신을 받으므로 어느 한 층이 최적화 걸음을 좌지우지하지 못한다.

!!! note "LARS와의 관계"
    LAMB는 Adam과 LARS(층별 적응형 학습률 조정)를 결합한 것으로 볼 수 있다. LARS는 같은 신뢰 비율 착상을 모멘텀을 쓰는 SGD에 적용한다. LAMB는 이를 Adam으로 넓혀 Adam의 매개변수별 적응성을 물려받으면서 LARS의 층별 배율 조정을 더한다.

---

## 3. PyTorch 예제

```python
"""대규모 배치 학습을 위한 LAMB 최적화기 사용 예."""

# === LAMB 최적화기 준비 ===

# LAMB는 PyTorch 본체에 없다. 다음으로 설치한다:
#   pip install torch-optimizer

if __name__ == "__main__":
    import torch
    import torch.nn as nn

    try:
        from torch_optimizer import Lamb

        model = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Linear(3072, 768),
        )
        optimizer = Lamb(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # 모의 학습 단계
        x = torch.randn(64, 768)
        y = torch.randn(64, 768)
        loss = nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item():.4f}")

    except ImportError:
        print("Install torch-optimizer: pip install torch-optimizer")
```

**출력:**

```
Install torch-optimizer: pip install torch-optimizer
```

---

## 4. LAMB를 쓸 때

LAMB는 특정한 상황을 위해 설계되었다.

- **대규모 배치 분산 학습**: 여러 GPU/TPU에 걸쳐 배치 크기가 8K~65K이고 표준 Adam이나 [AdamW](adamw.md)가 불안정해질 때.
- **큰 트랜스포머의 사전 학습**: 원 논문은 대규모 BERT 사전 학습에서의 효과를 보였다.
- **처리량과 품질 비교**: 배치 확대의 속도와 학습 품질은 통신·장치·전체 스텝 수까지 포함해 검증한다.

작은 배치에서도 효과가 없는지 여부는 실제 비교의 문제이다. 신뢰 비율 계산 비용과 AdamW 등 기준 모델의 성능을 같은 예산에서 비교한다. 특정 배치 크기만으로 선택을 정하지 않는다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
LAMB의 신뢰 비율을 포함한 갱신을 쓰고 AdamW의 감쇠와 구분하라.

</div>

??? success "연습문제 1 풀이"
    $r=\hat m/(\sqrt{\hat v}+\epsilon)+\lambda\theta$, $\phi=\|\theta\|/\|r\|$, $\theta^+=\theta-\eta\phi r$이다. 본문 식에서는 감쇠 항도 신뢰 비율에 곱해지므로 AdamW의 독립적인 감쇠와 같지 않다. 노름이 0인 경우의 대체 규칙과 제외 계수는 구현에 따라 정한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
신뢰 비율이 양의 유한값이고 클리핑이 없을 때 상대 갱신 노름을 구하라.

</div>

??? success "연습문제 2 풀이"
    $\|\Delta\theta\|=\eta\|\theta\|$이므로 상대 노름은 $\eta$이다. 이는 갱신 길이의 관계이며 손실 감소·일반화·수렴을 보장하지 않는다. 신뢰 비율을 잘라 쓰거나 0 노름 대체 규칙을 쓰면 이 등식도 달라질 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
경사 네 개만으로 LAMB의 신뢰 비율을 구할 수 없는 이유는 무엇인가?

</div>

??? success "연습문제 3 풀이"
    현재 파라미터 텐서와 적률 상태, beta·epsilon·감쇠 강도가 필요하다. 신뢰 비율은 파라미터와 전체 갱신 방향의 노름을 사용하므로 경사 제곱 누적만의 함수가 아니다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
큰 배치에서 LAMB의 성능을 평가할 조건을 제안하라.

</div>

??? success "연습문제 4 풀이"
    같은 데이터·목표 품질에서 전체 표본 처리량, 갱신 횟수와 학습률 스케줄, 통신 시간·메모리를 기록한다. AdamW 등도 같은 탐색 예산으로 조율한다. 배치 확대의 품질·시간 절충을 확인하고 모든 목적에 일반 GD의 안정 학습률 공식을 그대로 쓰지 않는다.

## 정리하며

LAMB의 핵심은 텐서별 상대 갱신 크기를 조절하는 신뢰 비율이다. 감쇠 항의 배율과 0 노름 예외를 확인하고, 큰 배치의 실제 시간과 품질을 비교한다.

**참고 문헌**

- You, Y., Li, J., Reddi, S., Hseu, J., Kumar, S., Bhojanapalli, S., Song, X., Demmel, J., Keutzer, K., & Hsieh, C.-J. (2020). Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes. *ICLR 2020*.
