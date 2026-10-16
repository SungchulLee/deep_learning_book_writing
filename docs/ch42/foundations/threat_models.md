# 위협 모델

**위협 모델**은 겨루는 이가 무엇을 알고, 무엇을 할 수 있으며, 무엇을 이루려 하는지에 대한 가정을 꼴로 적는다. 알맞은 위협 모델을 고르는 일이 중요하다. 너무 여리면 따짐이 뜻이 없고, 너무 세면 방어가 쓸모없다. 위협 모델이 다르면 공격과 방어도 피처부터 달라진다.

---

## 1. 겨루는 이의 앎

### 화이트박스 공격

**화이트박스** 자리에서 겨루는 이는 다음을 온전히 본다.

- 모델 구조 $f_\theta$
- 모델 파라메터 $\theta$
- 학습 데이터의 분포 $\mathcal{D}$
- 그래디언트 계산 $\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y)$

이는 **가장 센 치는 이**이며 막는 쪽에는 가장 나쁜 자리다. 화이트박스 공격은 모델이 얼마나 무른지의 위끝을 잡아 준다.

**뜻하는 바:**

- 치는 이는 최적화 기초의 공격에 쓸 정확한 그래디언트를 셈할 수 있다
- 치는 이는 구조에 매인 무른 데를 짚어내 쓸 수 있다
- 어떤 방어든 아무 그래디언트 기반의 섭동에도 든든해야 한다
- 지킴을 따질 때는 늘 화이트박스 따짐을 넣어야 한다

### 블랙박스 공격

**블랙박스** 자리에서 겨루는 이는 다음만 할 수 있다.

- 입력으로 모델에 묻고 출력을 살피기
- 예측한 클래스 레이블 $\hat{y} = f_\theta(\mathbf{x})$을 살피기
- 때에 따라 자신함 점수 $p(\hat{y}|\mathbf{x})$을 살피기

블랙박스 공격은 몇 가지 꾀를 쓴다.

1. **스코어 기반 공격**: 자신함 점수로 마디 있는 차를 써서 그래디언트를 어림한다
2. **판단 기초 공격**: 마지막 추론만으로 결정 경계를 둘러본다
3. **전이되는 공격**: 대신 쓰는 모델을 쳐서 섭동을 타깃으로 옮긴다

### 잿빛 상자 공격

**잿빛 상자**은 그 사이의 자리를 아우른다.

- 구조는 알고 짐은 모름
- 학습 절차는 알고 정확한 모델은 모름
- 가까운 모델을 볼 수 있음(같은 일, 다른 학습 판)

참으로는 이것이 가장 그럴듯한 자리인 일이 잦다. 특히 내놓은 금융 구조에서는 열린 글이나 거꾸로 뜯어보기로 모델 구조를 짐작할 수 있다.

---

## 2. 겨루는 이의 목표

### 비표적 공격

목표는 **어떻게든 틀리게 가르게** 하는 것이다.

$$
\text{Find } \boldsymbol{\delta} \text{ such that } f_\theta(\mathbf{x} + \boldsymbol{\delta}) \neq y, \quad \|\boldsymbol{\delta}\|_p \leq \varepsilon
$$

**최적화 꼴:**

$$
\boldsymbol{\delta}^* = \arg\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)
$$

참 레이블에 대한 로스를 가장 크게 하여 추론을 옳은 클래스에서 밀어낸다.

### 표적 공격

목표는 **고른 클래스** $y_{\text{target}}$을 억지로 미루어 보게 하는 것이다.

$$
\text{Find } \boldsymbol{\delta} \text{ such that } f_\theta(\mathbf{x} + \boldsymbol{\delta}) = y_{\text{target}}, \quad \|\boldsymbol{\delta}\|_p \leq \varepsilon
$$

**최적화 꼴:**

$$
\boldsymbol{\delta}^* = \arg\min_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y_{\text{target}})
$$

### 자신함 공격

틀리게 가르는 것을 넘어, 공격은 모델의 **자신함**을 노릴 수도 있다.

**크게 자신하며 틀리게 가르기:**

$$
\boldsymbol{\delta}^* = \arg\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \left[ \max_{y' \neq y} \log p(y' | \mathbf{x} + \boldsymbol{\delta}) - \log p(y | \mathbf{x} + \boldsymbol{\delta}) \right]
$$

**머뭇거리며 옳게 가르기:**

$$
\boldsymbol{\delta}^* = \arg\min_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} p(y | \mathbf{x} + \boldsymbol{\delta})
$$

### 목표 비교

| 피처 | 타깃 없음 | 타깃 있음 | 자신함 |
|--------|-----------|----------|------------|
| 최적화 | 로스를 가장 크게(참 레이블) | 로스를 가장 작게(타깃 레이블) | 틈을 가장 크게 |
| 그래디언트 방향 | 오름 | 내림 | 그때마다 |
| 어려움 | 더 쉬움 | 더 어려움 | 가운데 |
| 섭동의 크기 | 흔히 더 작음 | 흔히 더 큼 | 그때마다 |
| 금융에서의 걸림 | 비껴가기 공격 | 남인 척하기 공격 | 눈금 흔들기 공격 |

---

## 3. PyTorch로 짜기

### 위협 모델 차림

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
    맞섬의 으름 구조를 밝힌 것.
    
    Attributes
    ----------
    knowledge : AttackerKnowledge
        치는 이가 모형에 대해 아는 것
    goal : AttackerGoal
        치는 이가 이루려는 것
    norm : str
        흔듦 노름의 옭아맴('linf', 'l2', 'l1')
    epsilon : float
        흔듦 예산
    query_budget : int, 골라 씀
        검은 상자 치기의 가장 많은 물음 수
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

# 여느 잣대 으름 구조
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
    epsilon=0.05,  # 결 밭의 흔듦 예산
    query_budget=500
)
```

---

## 4. 알맞은 위협 모델 고르기

### 지킴을 따질 때

- 조심스러운 밑금으로 화이트박스 공격을 쓴다
- 여러 노름($\ell_\infty$, $\ell_2$, $\ell_1$)을 헤아린다
- 타깃 있는 자리와 없는 자리를 다 해 본다

### 그럴듯하게 따질 때

- 블랙박스나 잿빛 상자 구조를 쓴다
- 그럴듯한 물음 예산을 지운다
- 열린 모델에서 전이되는 공격을 헤아린다

### 금융 쓰임에서는

| 쓰임 | 즐겨 쓸 위협 모델 | 까닭 |
|-------------|-------------------------|-----------|
| 제때 거래 | 블랙박스, 물음 마디 있음 | 치는 이는 출력만 본다 |
| 배치 추론 | 화이트박스(모델이 샜다고 본다) | 떨어져 도는 구조에 조심스럽다 |
| 손님을 마주하는 API | 잿빛 상자, 스코어 기반 | 치는 이가 끝자리를 더듬을 수 있다 |
| 안쪽 위험 모델 | 화이트박스, 타깃 있음 | 안쪽 사람이 으르는 자리 |
| 속임수 알아내기 | 블랙박스, 판단 기초 | 속임수꾼은 받음/물림만 본다 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 클래스를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 뉴럴 넷트웍의 강건성과 어떻게 이어지는지 밝혀라.

</div>

??? success "연습문제 1 풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 뉴럴 넷트웍에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(그래디언트의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차원이 높은 모델이 무른 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 마디에서 다룬 공격나 방어를 CIFAR-10의 ResNet-18 모델에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 공격 아래에서 클린 정확도와 강건 정확도를 알려라.

</div>

??? success "연습문제 2 풀이"
    여느 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 걸음 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 마디의 방법을 걸면 피처와는 재주에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 테두리를 준다. 정확도와 강건성의 트레이드오프는 기초부터 있는 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 든다. 아무렇게나 하는 시드 3개의 평균과 표준편차로 알려라. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
섭동 공 안에서 클래스별 데이터의 밑자리가 서로 겹친다고 볼 때, 모델이 담는 힘을 키우지 않고서는 어떤 방어도 맑은 데이터의 높은 정확도와 $\ell_\infty$ 섭동에 대한 높은 강건성을 함께 이룰 수 없음을 증명하여라.

</div>

??? success "연습문제 3 풀이"
    두 클래스의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 기초 트레이드오프다. 겹치는 밑자리의 몫이 피할 수 없는 정확도 로스를 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 정확도 떨어짐을 밝혀 준다. 모델이 담는 힘을 키우면(더 너른 네트워크) 얽힌 강건한 결정 경계를 더 잘 그려 트레이드오프를 얼마쯤 눅일 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 구조(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 강건성이 어떻게 드러나는지 다루어라. 위협 모델이 보기 다룸과 어떻게 다른가?

</div>

??? success "연습문제 4 풀이"
    금융에서 겨루는 이는 알아내는 구조에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 핵심은 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 공격이 잇따르며 맞추어 간다(겨루는 이가 구조의 되받음을 보고 손본다). (3) 헛 정확도와 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 섭동 모델이 있어야 한다. 방어는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 기초의 길은 많이 걸러진다. $\square$

## 정리하며

| 깨침 | 핵심 |
|---------|-----------|
| 화이트박스 | 모델을 온전히 봄. 가장 센 공격. 가장 나쁜 자리의 따짐 |
| 블랙박스 | 물음만 할 수 있음. 그럴듯하나 더 여림 |
| 잿빛 상자 | 얼마쯤만 앎. 내놓은 자리에 가장 우도 |
| 타깃 없음 | 어떻게든 틀리게 가르게 한다 |
| 타깃 있음 | 고른 대로 틀리게 가르게 한다 |
| 자신함 | 추론의 자신함을 흔든다 |

위협 모델을 아는 일은 뉴럴 넷트웍을 치는 데도 막는 데도 먼저 있어야 한다. 알맞은 구조는 내놓는 자리와 지킴 요건에 매인다.

**살펴볼 거리**

1. Biggio, B., & Roli, F. (2018). "Wild Patterns: Ten Years After the Rise of Adversarial Machine Learning." Pattern Recognition.
2. Carlini, N., et al. (2019). "On Evaluating Adversarial Robustness." arXiv preprint arXiv:1902.06705.
3. Gilmer, J., et al. (2018). "Motivating the Rules of the Game for Adversarial Example Research." arXiv preprint arXiv:1807.06732.
