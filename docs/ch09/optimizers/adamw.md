# AdamW

AdamW는 웨이트 감쇠를 Adam의 적응형 경사 변환과 분리한다. 결합형 L2와 다른 갱신을 제공하며, 감쇠 대상과 러닝레이트·스케줄을 함께 정해야 한다.

## 1. Adam + L2의 문제

모멘텀 없는 SGD에서 $\lambda\|\theta\|^2/2$를 로스에 더하면 다음 감쇠 갱신과 같다.

$$\theta_{t+1} = \theta_t - \eta(g_t + \lambda\theta_t) = (1 - \eta\lambda)\theta_t - \eta g_t$$

Adam에서는 L2 정규화가 적응형 배율 조정 *전에* 그래디언트에 적용된다.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)(g_t + \lambda\theta_t)$$

적응형 분모 $\sqrt{\hat{v}_t}$이 그래디언트와 함께 정규화 항의 배율까지 조정하여 사실상 파라메터마다 *다른* 웨이트 감쇠를 걸게 되고, 고르게 정규화한다는 취지가 무너진다.

---

## 2. AdamW의 갱신 규칙

AdamW는 적응형 갱신 바깥에서 파라메터에 웨이트 감쇠를 곧바로 적용한다.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\theta_{t+1} = (1 - \eta\lambda)\theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

같은 파라미터 그룹 안에서 감쇠 인수는 `(1 - lr * weight_decay)`로 적용된다. 좌표별 적응 분모에는 들어가지 않지만 전역 러닝레이트와 스케줄에는 의존한다. 감쇠에서 제외하는 바이어스·정규화 계수 등 그룹 설정도 확인한다.

---

## 3. PyTorch 구현

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                              betas=(0.9, 0.999), weight_decay=0.01)
```

---

## 4. 대표적인 하이퍼파라미터

- $\eta = 0.001$~$0.0001$ (러닝레이트)
- $\lambda = 0.01$~$0.1$ (웨이트 감쇠)
- $\beta_1 = 0.9$, $\beta_2 = 0.999$

---

## 5. 언제 쓰는가

AdamW는 다음에 권장되는 기본 옵티마이저이다.

- 트랜스포머 모델 (BERT, GPT, 비전 트랜스포머)
- 미리 학습된 모델의 미세 조정
- 적응형 최적화와 웨이트 감쇠를 함께 쓰고 싶은 모든 과제

---

## 6. 핵심 정리

AdamW의 핵심은 감쇠와 적응형 변환의 분리이다. 이것이 항상 더 좋은 성능을 보장하거나 모든 계수에 감쇠를 적용해야 한다는 뜻은 아니다. Adam의 결합형·분리형 옵션도 사용 버전의 API를 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
결합형 L2 Adam과 AdamW가 다른 이유를 설명하라.

</div>

??? success "연습문제 1 풀이"
    결합형은 g+lambda*w를 적률 상태에 누적하여 패널티가 1차·2차 상태 모두에 영향을 준다. AdamW는 데이터 경사로 적률을 갱신하고 별도로 `(1-lr*lambda)w`를 적용한다. 단순히 lambda를 적응 분모로 나누는 정적 관계만으로 전체 차이를 나타낼 수 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
AdamW를 선택할 때 감쇠 대상과 스케줄을 함께 확인해야 하는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"
    감쇠는 각 단계의 lr과 weight_decay의 곱에 의존하고, 어느 파라미터 그룹을 감쇠하는지도 결과를 바꾼다. 결합 L2와 분리 감쇠를 후보로 비교할 수 있으며 이름만으로 최적값이나 우열을 정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
AdamW의 갱신 규칙을 구현하고 Adam과 어디가 다른지 보여라.

</div>

??? success "연습문제 3 풀이"
    아래 조각은 기존 m_hat·v_hat·lr·lambda 정의가 필요한 설명 코드이다. 데이터 갱신 뒤 새 계수에 감쇠를 곱하면 본문의 기존 계수 기준 식과 교차항만큼 달라질 수 있으므로 정확한 갱신 순서도 맞춰야 한다.
    ```python
    # Adam: p.data -= lr * m_hat / (v_hat.sqrt() + eps)  # L2 in grad
    # AdamW: 웨이트 감쇠는 따로 적용된다
    for p in model.parameters():
        p.data *= (1 - lr * weight_decay)  # 분리된 웨이트 감쇠
        p.data -= lr * m_hat / (v_hat.sqrt() + eps)  # Adam의 스텝
    ```

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
결합형 L2의 효과를 lambda/sqrt(v) 하나로 정확히 표현할 수 없는 이유는 무엇인가?

</div>

??? success "연습문제 4 풀이"
    L2 항은 현재 분모뿐 아니라 이후 모멘텀과 제곱 경사의 이력도 바꾼다. v는 분산이 아니라 비중심 2차 적률 추정값이고 평균 경사의 크기도 반영한다. 따라서 드문 계수가 항상 더 강하게 정규화된다는 식으로 단정할 수 없다.

## 정리하며

AdamW는 감쇠를 적응형 적률과 분리하지만 전역 러닝레이트에서 분리하지는 않는다. 파라미터 그룹과 갱신 순서, 감쇠 배율을 명시하고 같은 검증 조건에서 선택한다.
