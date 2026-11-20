# 위협 모델

**위협 모델**은 적대자가 무엇을 알고, 무엇을 할 수 있으며, 무엇을 이루려 하는지에 대한 가정을 형식화한다. 알맞은 위협 모델을 고르는 일이 중요하다. 너무 약하면 평가가 의미가 없고, 너무 세면 방어가 쓸모없다. 위협 모델이 다르면 공격과 방어도 근본부터 달라진다.

---

## 1. 적대자의 앎

### 화이트박스 공격

**화이트박스** 설정에서 적대자는 다음을 모두 안다.

- 모델 구조 $f_\theta$
- 모델 파라메터 $\theta$
- 학습 데이터의 분포 $\mathcal{D}$
- 그래디언트 계산 $\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y)$

이는 **가장 센 공격자**이며 방어하는 쪽에는 최악의 경우다. 화이트박스 공격은 모델이 얼마나 취약한지의 상한을 잡아 준다.

**뜻하는 바:**

- 공격자는 최적화 기반의 공격에 쓸 정확한 그래디언트를 계산할 수 있다
- 공격자는 구조에 특유한 취약점을 찾아내 악용할 수 있다
- 어떤 방어든 임의의 그래디언트 기반 섭동에도 강건해야 한다
- 보안을 평가할 때는 늘 화이트박스 평가를 넣어야 한다

### 블랙박스 공격

**블랙박스** 설정에서 적대자는 다음만 할 수 있다.

- 모델에 입력을 질의하고 출력을 관찰하기
- 예측한 클래스 레이블 $\hat{y} = f_\theta(\mathbf{x})$ 관찰하기
- 경우에 따라 확신도 점수 $p(\hat{y}|\mathbf{x})$ 관찰하기

블랙박스 공격은 몇 가지 전략을 쓴다.

1. **스코어 기반 공격**: 확신도 점수에 유한 차분을 써서 그래디언트를 근사한다
2. **결정 기반 공격**: 최종 예측 레이블만으로 결정 경계를 탐색한다
3. **전이 공격**: 대리 모델을 공격해 얻은 섭동을 타깃 모델에 옮긴다

### 그레이박스 공격

**그레이박스**는 그 사이의 자리를 아우른다.

- 구조는 알고 웨이트는 모름
- 학습 절차는 알고 정확한 모델은 모름
- 비슷한 모델에 접근할 수 있음(같은 과제, 다른 학습 실행)

실제로는 이것이 가장 현실적인 설정인 경우가 많다. 특히 배포된 금융 시스템에서는 공개 문헌이나 역공학으로 모델 구조를 짐작할 수 있다.

---

## 2. 적대자의 목표

### 비표적 공격

목표는 **어떻게든 오분류하게** 하는 것이다.

$$
\text{Find } \boldsymbol{\delta} \text{ such that } f_\theta(\mathbf{x} + \boldsymbol{\delta}) \neq y, \quad \|\boldsymbol{\delta}\|_p \leq \varepsilon
$$

**최적화 형태:**

$$
\boldsymbol{\delta}^* = \arg\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)
$$

참 레이블에 대한 로스를 가장 크게 하여 예측을 올바른 클래스에서 밀어낸다.

### 표적 공격

목표는 모델이 **지정한 클래스** $y_{\text{target}}$를 예측하게 만드는 것이다.

$$
\text{Find } \boldsymbol{\delta} \text{ such that } f_\theta(\mathbf{x} + \boldsymbol{\delta}) = y_{\text{target}}, \quad \|\boldsymbol{\delta}\|_p \leq \varepsilon
$$

**최적화 형태:**

$$
\boldsymbol{\delta}^* = \arg\min_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y_{\text{target}})
$$

### 확신도 공격

오분류를 넘어, 공격은 모델의 **확신도**를 노릴 수도 있다.

**높은 확신도로 오분류하게 하기:**

$$
\boldsymbol{\delta}^* = \arg\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \left[ \max_{y' \neq y} \log p(y' | \mathbf{x} + \boldsymbol{\delta}) - \log p(y | \mathbf{x} + \boldsymbol{\delta}) \right]
$$

**맞게 분류하되 확신도를 낮추기:**

$$
\boldsymbol{\delta}^* = \arg\min_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} p(y | \mathbf{x} + \boldsymbol{\delta})
$$

### 목표 비교

| 항목 | 비표적 | 표적 | 확신도 |
|--------|-----------|----------|------------|
| 최적화 | 로스를 가장 크게(참 레이블) | 로스를 가장 작게(타깃 레이블) | 마진을 가장 크게 |
| 그래디언트 방향 | 상승 | 하강 | 경우에 따라 |
| 난이도 | 더 쉬움 | 더 어려움 | 중간 |
| 섭동의 크기 | 흔히 더 작음 | 흔히 더 큼 | 경우에 따라 |
| 금융에서의 예 | 회피 공격 | 사칭 공격 | 보정 교란 공격 |

---

## 3. PyTorch로 구현하기

### 위협 모델 정의

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class AttackerKnowledge(Enum):
    WHITE_BOX = "white_box"
    BLACK_BOX = "black_box"
    GRAY_BOX = "gray_box"

class AttackerGoal(Enum):
    UNTARGETED = "untargeted"
    TARGETED = "targeted"
    CONFIDENCE = "confidence"

@dataclass
class ThreatModel:
    """
    적대적 으름 구조를 밝힌 것.
    
    Attributes
    ----------
    knowledge : AttackerKnowledge
        치는 이가 모형에 대해 아는 것
    goal : AttackerGoal
        치는 이가 이루려는 것
    norm : str
        섭동 노름의 옭아맴('linf', 'l2', 'l1')
    epsilon : float
        섭동 예산
    query_budget : int, 골라 씀
        검은 상자 치기의 가장 많은 질의 수
    """
    knowledge: AttackerKnowledge
    goal: AttackerGoal
    norm: str = 'linf'
    epsilon: float = 8/255
    query_budget: Optional[int] = None
    
    def __post_init__(self):
        if self.knowledge == AttackerKnowledge.BLACK_BOX:
            if self.query_budget is None:
                self.query_budget = 10000
    
    def __repr__(self):
        return (
            f"ThreatModel(\n"
            f"  knowledge={self.knowledge.value},\n"
            f"  goal={self.goal.value},\n"
            f"  norm={self.norm},\n"
            f"  epsilon={self.epsilon:.4f}"
            + (f",\n  query_budget={self.query_budget}" 
               if self.query_budget else "") +
            f"\n)"
        )

# 여느 기준 으름 구조
STANDARD_CIFAR10 = ThreatModel(
    knowledge=AttackerKnowledge.WHITE_BOX,
    goal=AttackerGoal.UNTARGETED,
    norm='linf',
    epsilon=8/255
)

REALISTIC_DEPLOYMENT = ThreatModel(
    knowledge=AttackerKnowledge.BLACK_BOX,
    goal=AttackerGoal.TARGETED,
    norm='l2',
    epsilon=0.5,
    query_budget=1000
)

FINANCIAL_API = ThreatModel(
    knowledge=AttackerKnowledge.GRAY_BOX,
    goal=AttackerGoal.UNTARGETED,
    norm='linf',
    epsilon=0.05,  # 결 분야의 섭동 예산
    query_budget=500
)
```

---

## 4. 알맞은 위협 모델 고르기

### 보안 평가에서

- 보수적인 베이스라인으로 화이트박스 공격을 쓴다
- 여러 노름($\ell_\infty$, $\ell_2$, $\ell_1$)을 고려한다
- 표적 공격과 비표적 공격을 모두 시험한다

### 현실적인 평가에서

- 블랙박스나 그레이박스 설정을 쓴다
- 현실적인 질의 예산을 둔다
- 공개 모델에서 전이되는 공격을 고려한다

### 금융 응용에서는

| 쓰임 | 권장 위협 모델 | 까닭 |
|-------------|-------------------------|-----------|
| 실시간 거래 | 블랙박스, 질의 수 제한 | 공격자는 출력만 본다 |
| 배치 추론 | 화이트박스(모델 유출을 가정) | 오프라인 시스템이므로 보수적으로 본다 |
| 고객 대상 API | 그레이박스, 스코어 기반 | 공격자가 엔드포인트를 탐색할 수 있다 |
| 내부 위험 모델 | 화이트박스, 표적 | 내부자 위협 |
| 사기 탐지 | 블랙박스, 결정 기반 | 사기꾼은 승인/거절만 본다 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 클래스를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 뉴럴 네트워크의 강건성과 어떻게 이어지는지 설명하여라.

</div>

??? success "풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 뉴럴 네트워크에서는 국소 선형 근사 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$가 FGSM(그래디언트의 부호를 쓴다)이 왜 잘 통하는지를 설명해 준다. 차원이 높은 모델이 취약한 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 절에서 다룬 공격이나 방어를 CIFAR-10의 ResNet-18 모델에 구현하여라. $\epsilon = 8/255$의 PGD-20 공격 아래에서 클린 정확도와 강건 정확도를 보고하여라.

</div>

??? success "풀이"
    일반 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 절의 방법을 적용한 결과는 기법에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 바운드를 준다. 정확도와 강건성의 트레이드오프는 근본적인 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 떨어진다. 무작위 시드 3개의 평균과 표준편차로 보고하여라. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
섭동 공 안에서 클래스별 데이터의 서포트가 서로 겹친다고 볼 때, 모델이 용량을 키우지 않고서는 어떤 방어도 클린 데이터의 높은 정확도와 $\ell_\infty$ 섭동에 대한 높은 강건성을 함께 이룰 수 없음을 증명하여라.

</div>

??? success "풀이"
    두 클래스의 서포트가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$가 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 근본적인 트레이드오프다. 겹치는 서포트의 비중이 피할 수 없는 정확도 손실을 정한다. 자연 이미지 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 관찰되는 10~15%의 정확도 하락을 설명해 준다. 모델 용량을 키우면(더 넓은 네트워크) 복잡한 강건 결정 경계를 더 잘 그려 트레이드오프를 어느 정도 완화할 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 시스템(사기 탐지나 거래 신호 생성 등)에서 적대적 강건성이 어떻게 드러나는지 논하여라. 위협 모델이 비전과 어떻게 다른가?

</div>

??? success "풀이"
    금융에서 적대자는 탐지 시스템에 맞추어 전략을 바꾸는 교묘한 집단(사기꾼, 시장 조작자)이다. 비전과 다른 핵심은 이렇다. (1) 섭동할 수 있는 피처가 경제적으로 그럴듯한 범위로 제약된다(사기꾼이 자기 거래 이력을 통째로 바꿀 수는 없다). (2) 공격이 순차적이고 적응적이다(적대자가 시스템의 응답을 보고 고친다). (3) 거짓 양성과 거짓 음성의 비용이 비대칭이다(정상 거래를 막는 것과 사기를 놓치는 것). (4) $\ell_p$ 노름은 의미가 없고 도메인에 맞는 섭동 모델이 있어야 한다. 방어는 적응형 적대자에게도 강건해야 하므로, 탐지 기준이 알려지면 회피할 수 있는 탐지 기반 방법은 쓸모가 크게 줄어든다. $\square$

## 정리하며

| 통찰 | 핵심 |
|---------|-----------|
| 화이트박스 | 모델을 모두 앎. 가장 센 공격. 최악의 경우 평가 |
| 블랙박스 | 질의만 할 수 있음. 현실적이나 더 약함 |
| 그레이박스 | 일부만 앎. 배포 환경에서 가장 현실적 |
| 비표적 | 어떻게든 오분류하게 한다 |
| 표적 | 지정한 클래스로 오분류하게 한다 |
| 확신도 | 예측의 확신도를 흔든다 |

위협 모델을 이해하는 일은 뉴럴 네트워크를 공격할 때도 방어할 때도 먼저 있어야 한다. 알맞은 위협 모델은 배포 환경과 보안 요건에 달려 있다.

**살펴볼 거리**

1. Biggio, B., & Roli, F. (2018). "Wild Patterns: Ten Years After the Rise of Adversarial Machine Learning." Pattern Recognition.
2. Carlini, N., et al. (2019). "On Evaluating Adversarial Robustness." arXiv preprint arXiv:1902.06705.
3. Gilmer, J., et al. (2018). "Motivating the Rules of the Game for Adversarial Example Research." arXiv preprint arXiv:1807.06732.
