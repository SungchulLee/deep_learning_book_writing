# 시장 조작 탐지

시장 조작, 곧 자유로운 시장의 작동을 일부러 왜곡하는 일은 머신러닝에서 유난히 적대적인 분야다. 섭동이 수학적 가정에 머무는 일반적인 적대적 강건성 연구와 달리, 시장 조작에는 사람과 알고리즘 참여자를 함께 속이려고 관측되는 신호를 조작하는, 금전적 동기를 가진 실제 행위자가 있다.

---

## 1. 적대적 공격으로 본 시장 조작

### 적대적 머신러닝과의 유사성

| 적대적 머신러닝의 개념 | 시장 조작에서 대응하는 것 |
|------------------------|---------------------------|
| 입력 섭동 | 스푸핑 주문, 자전 거래 |
| 표적 공격 | 특정 알고리즘 거래 신호를 일부러 유발 |
| 회피 공격 | 감시 시스템의 탐지를 회피 |
| 섭동 예산 | 규제 위험, 자본 제약 |
| 블랙박스 공격 | 감시 모델을 모른 채 하는 섭동 |

### 형식화

시장 조작자는 목표를 이루려고 관측되는 시장 신호 $\mathbf{x}_t$(가격, 거래량, 호가창)를 바꾼다.

$$
\mathbf{x}_t^{\text{manip}} = \mathbf{x}_t + \boldsymbol{\delta}_t, \quad \text{subject to } \boldsymbol{\delta}_t \in \mathcal{C}_t
$$

여기서 $\mathcal{C}_t$는 금전적 제약과 규제상의 제약을 담는다.

- 주문이 실제로 체결될 수 있어야 한다(유효한 가격과 수량)
- 조작에 드는 비용을 그 전략의 수익으로 메울 수 있어야 한다
- 패턴이 단순한 규칙 기반 감시에 걸리지 않아야 한다

---

## 2. 시장 조작의 유형

### 스푸핑과 레이어링

체결되기 전에 취소할 생각으로 주문을 넣어 매도·매수 수요에 대한 거짓 인상을 만든다. 여러 가격대에 주문을 겹겹이 쌓으면 레이어링이라 한다.

- **적대적 영향**: 호가창 기반 머신러닝 모델을 속여 가격 움직임을 잘못 예측하게 한다
- **탐지의 어려움**: 정상적인 주문 정정·취소와 구별하기 어렵다
- **강건성 요건**: 탐지 모델이 적대적으로 조작된 주문 피처에 강건해야 한다

### 자전 거래

실제 소유권 이전 없이 스스로 사고팔아 거래량만 부풀린다.

- **적대적 영향**: 거래 알고리즘이 쓰는 거래량 기반 신호를 부풀린다
- **탐지의 어려움**: 거래 하나하나는 정상으로 보인다
- **강건성 요건**: 모델이 여러 거래에 걸친 공모 패턴을 탐지해야 한다

### 펌프 앤 덤프

허위 홍보로 자산 가격을 끌어올린 뒤 팔아 치우는 조직적인 작전이다.

- **적대적 영향**: 감성 신호와 모멘텀 신호를 조작한다
- **탐지의 어려움**: 정상처럼 보이는 시장 움직임과 소셜 미디어 조작을 함께 쓴다
- **강건성 요건**: 가격, 거래량, 텍스트 데이터에 걸친 멀티모달 강건성

---

## 3. 강건한 감시 시스템

### 강건한 탐지 구조

아래 코드는 주문 수준 모델과 시계열 패턴 모델의 점수를 가중 평균해, 한 종류의 신호만 조작해서는 경보를 피하기 어렵게 한다.

```python
import torch
import torch.nn as nn
from typing import Dict, List

class RobustSurveillanceSystem:
    """
    적대적 강건한 저자 흔들기 탐지.
    
    탐지하는 신호 여럿을 맞서며 익히기와 아울러
    맞추어 오는 흔드는 이에게 버틴다.
    """
    
    def __init__(
        self,
        order_model: nn.Module,
        pattern_model: nn.Module,
        ensemble_weights: List[float] = [0.5, 0.5]
    ):
        self.order_model = order_model      # 주문 레이어의 결
        self.pattern_model = pattern_model    # 이음 레이어의 결
        self.weights = ensemble_weights
    
    def detect(
        self,
        order_features: torch.Tensor,
        sequence_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        여러 신호로 흔들기를 알아낸다.
        
        주문 레이어 모형과 결 레이어 모형을 모둠으로 묶어
        신호 하나만 흔드는 일에 강건하게 한다.
        """
        with torch.no_grad():
            score_order = torch.sigmoid(self.order_model(order_features))
            score_pattern = torch.sigmoid(self.pattern_model(sequence_features))
            
            # 짐 준 모둠
            combined = (
                self.weights[0] * score_order +
                self.weights[1] * score_pattern
            )
        
        return {
            'manipulation_score': combined,
            'order_score': score_order,
            'pattern_score': score_pattern,
            'alert': combined > 0.5
        }
```

### 방어 전략

1. **여러 신호로 탐지**: 가격, 거래량, 호가창, 행동 신호를 함께 써서 한 종류의 신호만 조작해서는 통하지 않게 한다
2. **시간적 일관성**: 여러 시간 스케일에 걸쳐 피처가 일관되는지 살핀다
3. **시장 간 분석**: 연관된 시장과 상품에 걸쳐 신호를 평가한다
4. **적대적 트레인**: 시뮬레이션한 조작 전략에 맞서 탐지 모델을 트레인한다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 클래스를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 뉴럴 네트워크의 강건성과 어떻게 이어지는지 설명하여라.

</div>

??? success "풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 뉴럴 네트워크에서는 국소 선형 근사 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(그래디언트의 부호를 쓴다)이 왜 잘 통하는지를 설명해 준다. 차원이 높은 모델이 취약한 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 절에서 다룬 공격이나 방어를 CIFAR-10의 ResNet-18 모델에 구현하여라. $\epsilon = 8/255$의 PGD-20 공격 아래에서 클린 정확도와 강건 정확도를 보고하여라.

</div>

??? success "풀이"
    일반 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 절의 방법을 적용한 결과는 기법에 따라 다르다. 적대적 트레인은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 바운드를 준다. 정확도와 강건성의 트레이드오프는 근본적인 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 떨어진다. 무작위 시드 3개의 평균과 표준편차로 보고하여라. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
섭동 공 안에서 클래스별 데이터의 서포트가 서로 겹친다고 볼 때, 모델이 용량을 키우지 않고서는 어떤 방어도 클린 데이터의 높은 정확도와 $\ell_\infty$ 섭동에 대한 높은 강건성을 함께 이룰 수 없음을 증명하여라.

</div>

??? success "풀이"
    두 클래스의 서포트가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 근본적인 트레이드오프다. 겹치는 서포트의 비중이 피할 수 없는 정확도 손실을 정한다. 자연 이미지 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 관찰되는 10~15%의 정확도 하락을 설명해 준다. 모델 용량을 키우면(더 넓은 네트워크) 복잡한 강건 결정 경계를 더 잘 그려 트레이드오프를 어느 정도 완화할 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 시스템(사기 탐지나 거래 신호 생성 등)에서 적대적 강건성이 어떻게 드러나는지 논하여라. 위협 모델이 비전과 어떻게 다른가?

</div>

??? success "풀이"
    금융에서 적대자는 탐지 시스템에 맞추어 전략을 바꾸는 교묘한 집단(사기꾼, 시장 조작자)이다. 비전과 다른 핵심은 이렇다. (1) 섭동할 수 있는 피처가 경제적으로 그럴듯한 범위로 제약된다(사기꾼이 자기 거래 이력을 통째로 바꿀 수는 없다). (2) 공격이 순차적이고 적응적이다(적대자가 시스템의 응답을 보고 고친다). (3) 거짓 양성과 거짓 음성의 비용이 비대칭이다(정상 거래를 막는 것과 사기를 놓치는 것). (4) $\ell_p$ 노름은 의미가 없고 도메인에 맞는 섭동 모델이 있어야 한다. 방어는 적응형 적대자에게도 강건해야 하므로, 탐지 기준이 알려지면 회피할 수 있는 탐지 기반 방법은 쓸모가 크게 줄어든다. $\square$

## 정리하며

시장 조작 탐지는 실제 적대자가 있는 분야에 적대적 강건성을 적용한 것이다. 핵심 통찰은 조작이 금전적으로 수지가 맞아야 한다는 것이며, 이것이 적대자의 섭동 공간을 제약해 더 효과적인 탐지의 실마리가 된다.

**살펴볼 거리**

1. Cao, Y., et al. (2021). "Adversarial Attacks on Machine Learning-Based Market Surveillance." Journal of Financial Data Science.
2. Golmohammadi, K., & Zaiane, O. R. (2015). "Time Series Contextual Anomaly Detection for Detecting Market Manipulation in Stock Market." IEEE DSAA.
