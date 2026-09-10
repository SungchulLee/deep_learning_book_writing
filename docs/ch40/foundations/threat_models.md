# 으름 얼개

**으름 얼개**은 겨루는 이가 무엇을 알고, 무엇을 할 수 있으며, 무엇을 이루려 하는지에 대한 가정을 꼴로 적는다. 알맞은 으름 얼개를 고르는 일이 종요롭다. 너무 여리면 따짐이 뜻이 없고, 너무 세면 막이가 쓸모없다. 으름 얼개가 다르면 치기와 막이도 결부터 달라진다.

---

## 1. 겨루는 이의 앎

### 흰 상자 치기

**흰 상자** 자리에서 겨루는 이는 다음을 온전히 본다.

- 모형 얼개 $f_\theta$
- 모형 매개변수 $\theta$
- 익힘 자료의 분포 $\mathcal{D}$
- 기울기 셈 $\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y)$

이는 **가장 센 치는 이**이며 막는 쪽에는 가장 나쁜 자리다. 흰 상자 치기는 모형이 얼마나 무른지의 위끝을 잡아 준다.

**뜻하는 바:**

- 치는 이는 가장 좋게 하기 바탕의 치기에 쓸 정확한 기울기를 셈할 수 있다
- 치는 이는 얼개에 매인 무른 데를 짚어내 쓸 수 있다
- 어떤 막이든 아무 기울기 바탕의 흔듦에도 든든해야 한다
- 지킴을 따질 때는 늘 흰 상자 따짐을 넣어야 한다

### 검은 상자 치기

**검은 상자** 자리에서 겨루는 이는 다음만 할 수 있다.

- 들임으로 모형에 묻고 날임을 살피기
- 미루어 본 갈래 이름표 $\hat{y} = f_\theta(\mathbf{x})$을 살피기
- 때에 따라 자신함 점수 $p(\hat{y}|\mathbf{x})$을 살피기

검은 상자 치기는 몇 가지 꾀를 쓴다.

1. **점수 바탕 치기**: 자신함 점수로 마디 있는 차를 써서 기울기를 어림한다
2. **판단 바탕 치기**: 마지막 미루어 봄만으로 판단의 금을 둘러본다
3. **옮아가는 치기**: 대신 쓰는 모형을 쳐서 흔듦을 과녁으로 옮긴다

### 잿빛 상자 치기

**잿빛 상자**은 그 사이의 자리를 아우른다.

- 얼개는 알고 짐은 모름
- 익힘 절차는 알고 정확한 모형은 모름
- 가까운 모형을 볼 수 있음(같은 일, 다른 익힘 판)

참으로는 이것이 가장 그럴듯한 자리인 일이 잦다. 특히 내놓은 금융 얼개에서는 열린 글이나 거꾸로 뜯어보기로 모형 얼개를 짐작할 수 있다.

---

## 2. 겨루는 이의 목표

### 과녁 없는 치기

목표는 **어떻게든 틀리게 가르게** 하는 것이다.

$$
\text{Find } \boldsymbol{\delta} \text{ such that } f_\theta(\mathbf{x} + \boldsymbol{\delta}) \neq y, \quad \|\boldsymbol{\delta}\|_p \leq \varepsilon
$$

**가장 좋게 하기 꼴:**

$$
\boldsymbol{\delta}^* = \arg\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)
$$

참 이름표에 대한 잃음을 가장 크게 하여 미루어 봄을 옳은 갈래에서 밀어낸다.

### 과녁 있는 치기

목표는 **고른 갈래** $y_{\text{target}}$을 억지로 미루어 보게 하는 것이다.

$$
\text{Find } \boldsymbol{\delta} \text{ such that } f_\theta(\mathbf{x} + \boldsymbol{\delta}) = y_{\text{target}}, \quad \|\boldsymbol{\delta}\|_p \leq \varepsilon
$$

**가장 좋게 하기 꼴:**

$$
\boldsymbol{\delta}^* = \arg\min_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y_{\text{target}})
$$

### 자신함 치기

틀리게 가르는 것을 넘어, 치기는 모형의 **자신함**을 노릴 수도 있다.

**크게 자신하며 틀리게 가르기:**

$$
\boldsymbol{\delta}^* = \arg\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \left[ \max_{y' \neq y} \log p(y' | \mathbf{x} + \boldsymbol{\delta}) - \log p(y | \mathbf{x} + \boldsymbol{\delta}) \right]
$$

**머뭇거리며 옳게 가르기:**

$$
\boldsymbol{\delta}^* = \arg\min_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} p(y | \mathbf{x} + \boldsymbol{\delta})
$$

### 목표 견주기

| 결 | 과녁 없음 | 과녁 있음 | 자신함 |
|--------|-----------|----------|------------|
| 가장 좋게 하기 | 잃음을 가장 크게(참 이름표) | 잃음을 가장 작게(과녁 이름표) | 틈을 가장 크게 |
| 기울기 방향 | 오름 | 내림 | 그때마다 |
| 어려움 | 더 쉬움 | 더 어려움 | 가운데 |
| 흔듦의 크기 | 흔히 더 작음 | 흔히 더 큼 | 그때마다 |
| 금융에서의 걸림 | 비껴가기 치기 | 남인 척하기 치기 | 눈금 흔들기 치기 |

---

## 3. PyTorch로 짜기

### 으름 얼개 차림

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
    맞섬의 으름 얼개를 밝힌 것.
    
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

# 여느 잣대 으름 얼개
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

## 4. 알맞은 으름 얼개 고르기

### 지킴을 따질 때

- 조심스러운 밑금으로 흰 상자 치기를 쓴다
- 여러 노름($\ell_\infty$, $\ell_2$, $\ell_1$)을 헤아린다
- 과녁 있는 자리와 없는 자리를 다 해 본다

### 그럴듯하게 따질 때

- 검은 상자나 잿빛 상자 얼개를 쓴다
- 그럴듯한 물음 예산을 지운다
- 열린 모형에서 옮아가는 치기를 헤아린다

### 금융 쓰임에서는

| 쓰임 | 즐겨 쓸 으름 얼개 | 까닭 |
|-------------|-------------------------|-----------|
| 제때 거래 | 검은 상자, 물음 마디 있음 | 치는 이는 날임만 본다 |
| 묶음 미루어 봄 | 흰 상자(모형이 샜다고 본다) | 떨어져 도는 얼개에 조심스럽다 |
| 손님을 마주하는 API | 잿빛 상자, 점수 바탕 | 치는 이가 끝자리를 더듬을 수 있다 |
| 안쪽 무릅씀 모형 | 흰 상자, 과녁 있음 | 안쪽 사람이 으르는 자리 |
| 속임수 알아내기 | 검은 상자, 판단 바탕 | 속임수꾼은 받음/물림만 본다 |

---

## 연습문제

**연습문제 1.**
선형 가름개 $f(x) = w^T x + b$에서 미루어 본 갈래를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 흔듦을 셈하여라. 이것이 신경 그물의 든든함과 어떻게 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    선형 가름개에서 $\ell_\infty$ 노름으로 잰 판단의 금까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 흔듦은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 신경 그물에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(기울기의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차수가 높은 모형이 무른 까닭은 $\|w\|_1$은 차수와 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 든든함의 여유가 줄어들기 때문이다. $\square$

---

**연습문제 2.**
이 마디에서 다룬 치기나 막이를 CIFAR-10의 ResNet-18 모형에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 치기 아래에서 맑은 맞음과 든든한 맞음을 알려라.

??? success "연습문제 2 풀이"
    여느 ResNet-18은 맑은 맞음이 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 걸음 크기 $2/255$) 아래의 든든한 맞음은 $\sim$0%이다. 이 마디의 방법을 걸면 결과는 재주에 따라 다르다. 맞서며 익히기는 맑은 맞음 $\sim$83%에 든든한 맞음 $\sim$50%이고, 밝혀 낸 막이는 더 낮지만 증명할 수 있는 테두리를 준다. 맞음과 든든함의 맞바꿈은 밑바탕부터 있는 것이라, 든든함을 높이면 맑은 맞음이 흔히 5~15% 든다. 아무렇게나 하는 씨앗 3개의 평균과 잣대 어긋남으로 알려라. $\square$

---

**연습문제 3.**
흔듦 공 안에서 갈래별 자료의 밑자리가 서로 겹친다고 볼 때, 모형이 담는 힘을 키우지 않고서는 어떤 막이도 맑은 자료의 높은 맞음과 $\ell_\infty$ 흔듦에 대한 높은 든든함을 함께 이룰 수 없음을 증명하여라.

??? success "연습문제 3 풀이"
    두 갈래의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{갈래 1}, x_2 \in \text{갈래 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 든든한 가름개는 적어도 하나를 틀리게 가를 수밖에 없다(흔듦 공이 겹치기 때문이다). 이것이 맞음과 든든함의 밑바탕 맞바꿈이다. 겹치는 밑자리의 몫이 피할 수 없는 맞음 잃음을 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 맞음 떨어짐을 밝혀 준다. 모형이 담는 힘을 키우면(더 너른 그물) 얽힌 든든한 판단의 금을 더 잘 그려 맞바꿈을 얼마쯤 눅일 수 있다. $\square$

---

**연습문제 4.**
금융 기계 배움 얼개(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 든든함이 어떻게 드러나는지 다루어라. 으름 얼개가 보기 다룸과 어떻게 다른가?

??? success "연습문제 4 풀이"
    금융에서 겨루는 이는 알아내는 얼개에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 고갱이는 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 치기가 잇따르며 맞추어 간다(겨루는 이가 얼개의 되받음을 보고 손본다). (3) 헛 맞음과 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 흔듦 모형이 있어야 한다. 막이는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 바탕의 길은 많이 걸러진다. $\square$

## 정리하며

| 깨침 | 고갱이 |
|---------|-----------|
| 흰 상자 | 모형을 온전히 봄. 가장 센 치기. 가장 나쁜 자리의 따짐 |
| 검은 상자 | 물음만 할 수 있음. 그럴듯하나 더 여림 |
| 잿빛 상자 | 얼마쯤만 앎. 내놓은 자리에 가장 그럴듯함 |
| 과녁 없음 | 어떻게든 틀리게 가르게 한다 |
| 과녁 있음 | 고른 대로 틀리게 가르게 한다 |
| 자신함 | 미루어 봄의 자신함을 흔든다 |

으름 얼개를 아는 일은 신경 그물을 치는 데도 막는 데도 먼저 있어야 한다. 알맞은 얼개는 내놓는 자리와 지킴 요건에 매인다.

**살펴볼 거리**

1. Biggio, B., & Roli, F. (2018). "Wild Patterns: Ten Years After the Rise of Adversarial Machine Learning." Pattern Recognition.
2. Carlini, N., et al. (2019). "On Evaluating Adversarial Robustness." arXiv preprint arXiv:1902.06705.
3. Gilmer, J., et al. (2018). "Motivating the Rules of the Game for Adversarial Example Research." arXiv preprint arXiv:1807.06732.
