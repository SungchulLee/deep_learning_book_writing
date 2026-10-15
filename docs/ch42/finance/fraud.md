# 사기 탐지 모델의 강건성

속임수 알아내기 구조는 본디 겨루는 자리다. 속임수꾼은 속이려는 뜻은 지킨 채 알아내기를 비껴가는 거래를 부지런히 지어낸다. 그래서 적대적 강건성은 이론의 걱정이 아니라 굴러가는 데 꼭 있어야 할 것이 된다. 적대적 샘플이 연구거리인 그림 분류와 달리, 속임수 알아내기는 꾀를 끊임없이 갈아 대는 **참 겨루는 이**와 마주한다.

---

## 1. 속임수 알아내기의 위협 모델

### 겨루는 이의 모습

속임수꾼은 다음 제약 아래 움직인다.

- **앎**: 흔히 블랙박스나 잿빛 상자다. 받음/물림 판단은 보지만 모델을 들여다보는 일은 드물다
- **목표**: 타깃 있는 비껴가기. 속임 거래를 옳은 거래처럼 보이게 한다
- **제약**: 속임의 돈 목표를 지켜야 한다(돈이 참으로 옮겨져야 하고, 훔친 물건을 받아야 한다)
- **물음 예산**: 속임을 한 번 해 볼 때의 값과 들킬 위험에 마디 지어진다

### 꼴로 적기

$f(\mathbf{x}) = 1$이 속임을 뜻하는 속임수 알아내개 $f_\theta: \mathbb{R}^d \to \{0, 1\}$을 두자. 겨루는 이는 다음을 찾는다.

$$
\mathbf{x}_{\text{evasion}} = \arg\min_{\mathbf{x}' \in \mathcal{C}} f_\theta(\mathbf{x}')
$$

이때 제약 모임 $\mathcal{C}$은 속임의 뜻을 지킨다(그 거래가 여전히 겨루는 이의 돈 목표를 이뤄야 한다).

### 특징 공간의 섭동

$\ell_p$ 노름을 쓰는 그림 공격과 달리, 속임 공격은 밭에 맞는 제약을 지닌 **특징 공간**에서 움직인다.

| 특징 갈래 | 흔들 수 있나? | 제약 |
|-------------|-------------|------------|
| 거래 값 | 얼마쯤 | 돈 목표를 이뤄야 한다 |
| 가게 갈래 | 그렇다 | 옳은 갈래에서 고른다 |
| 하루의 때 | 그렇다 | 문 여는 동안에 |
| 장치 손자국 | 그렇다 | 속이거나 새 장치를 쓴다 |
| IP 자리 | 그렇다 | VPN/대리를 쓴다 |
| 거래 잦기 | 얼마쯤 | 거래를 마쳐야 한다 |
| 카드 있음 표시 | 붙박이 | 몸으로 옭매인다 |

---

## 2. 속임수 알아내기의 적대적 학습

### 맞춰 고친 적대적 학습 틀

여느 적대적 학습은 표로 된 금융 데이터에 맞게 고쳐야 한다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class FraudRobustTrainer:
    """
    속임수 알아내기에 맞춘 맞서며 익히기.
    
    그림 맞서며 익히기와 다른 핵심:
    - 결마다 다른 흔듦 예산
    - 옭아맴을 아는 흔듦(갈래 결, 옳은 자리)
    - 어긋난 잃음(놓침이 헛 맞음보다 값이 크다)
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_budgets: torch.Tensor,
        categorical_mask: torch.Tensor,
        num_iter: int = 10,
        alpha_scale: float = 2.0,
        fn_weight: float = 10.0,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.feature_budgets = feature_budgets  # 결마다의 엡실론
        self.categorical_mask = categorical_mask  # 갈래 결이면 1
        self.num_iter = num_iter
        self.alpha_scale = alpha_scale
        self.fn_weight = fn_weight  # 놓침에 주는 짐
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
    
    def _constrained_pgd(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        결마다 옭아맴을 지닌 PGD.
        
        이어지는 결: 결마다의 엡실론 안에서 흔든다
        갈래 결: 붙박아 둔다(또는 옳은 값으로만 흔든다)
        """
        eps = self.feature_budgets.to(self.device)
        cat_mask = self.categorical_mask.to(self.device)
        alpha = self.alpha_scale * eps / self.num_iter
        
        # 첫자리를 잡는다
        delta = torch.zeros_like(x)
        cont_mask = 1 - cat_mask
        
        # 이어지는 결만 아무 첫자리
        delta = delta + cont_mask * torch.empty_like(x).uniform_(-1, 1) * eps
        
        for _ in range(self.num_iter):
            x_adv = (x + delta).requires_grad_(True)
            logits = self.model(x_adv)
            
            # 짐 준 잃음: 비껴감을 벌한다(속임을 옳은 것으로 가름)
            loss = F.cross_entropy(logits, y, reduction='none')
            # 속임 보기에 짐을 더 준다(겨루는 이가 비껴가려 한다)
            weights = torch.where(y == 1, self.fn_weight, 1.0)
            loss = (weights * loss).mean()
            
            self.model.zero_grad()
            loss.backward()
            grad = x_adv.grad.data
            
            with torch.no_grad():
                # 이어지는 결만 고친다
                delta = delta + cont_mask * alpha * grad.sign()
                delta = torch.clamp(delta, -eps, eps) * cont_mask
        
        return torch.clamp(x + delta, 0, 1).detach()
    
    def train_epoch(self, train_loader, optimizer):
        """속임을 아는 맞서며 익히기로 한 판 익힌다."""
        self.model.train()
        total_loss = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            # 맞서는 보기를 만든다
            x_adv = self._constrained_pgd(x, y)
            
            # 맞서는 보기로 익힌다
            optimizer.zero_grad()
            logits = self.model(x_adv)
            
            # 어긋난 잃음
            weights = torch.where(y == 1, self.fn_weight, 1.0)
            loss = (weights * F.cross_entropy(
                logits, y, reduction='none'
            )).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            total += len(y)
        
        return {'loss': total_loss / total}
```

---

## 3. 강건한 속임수 알아내기를 따지는 자

속임수 알아내기에 표준 정확도만으로는 모자란다. 맞서는 자리에서 걸리는 자는 이렇다.

| 자 | 정의 | 겨냥 |
|--------|-----------|--------|
| 강건한 참 정확도 비율 | 맞서서 비껴갈 때의 참 정확도 비율 | 가장 크게 |
| 문턱에서의 헛 정확도 비율 | 굴리는 문턱에서의 헛 정확도 비율 | 가장 작게 |
| 강건한 AUPRC | 공격 아래 촘촘함-되불러옴 굽이 아래 넓이 | 가장 크게 |
| 비껴간 비율 | 알아내기를 비껴간 속임의 몫 | 가장 작게 |

---

## 4. 참으로 즐겨 쓸 길

1. **특징마다 예산을 달리한다**: 특징마다 흔들 수 있는 정도가 다르다
2. **갈래 제약을 지킨다**: 겨루는 이가 띄엄한 특징을 마음대로 바꿀 수는 없다
3. **어긋나게 학습된이다**: 놓친 속임에 헛 정확도보다 훨씬 큰 짐을 준다
4. **비껴가는 특징을 지켜본다**: 겨루는 이가 어느 특징을 가장 많이 흔드는지 좇는다
5. **방어를 앙상블로**: 든든하도록 규칙 기초와 머신러닝 기초의 알아내기를 아우른다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 갈래를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 신경망의 강건성과 어떻게 이어지는지 밝혀라.

</div>

??? success "연습문제 1 풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 신경망에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(기울기의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차원이 높은 모델이 무른 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 마디에서 다룬 공격나 방어를 CIFAR-10의 ResNet-18 모델에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 공격 아래에서 클린 정확도와 강건 정확도를 알려라.

</div>

??? success "연습문제 2 풀이"
    여느 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 걸음 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 마디의 방법을 걸면 특징과는 재주에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 테두리를 준다. 정확도와 강건성의 트레이드오프는 기초부터 있는 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 든다. 아무렇게나 하는 시드 3개의 평균과 표준편차로 알려라. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
섭동 공 안에서 갈래별 데이터의 밑자리가 서로 겹친다고 볼 때, 모델이 담는 힘을 키우지 않고서는 어떤 방어도 맑은 데이터의 높은 정확도와 $\ell_\infty$ 섭동에 대한 높은 강건성을 함께 이룰 수 없음을 증명하여라.

</div>

??? success "연습문제 3 풀이"
    두 갈래의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{갈래 1}, x_2 \in \text{갈래 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 기초 트레이드오프다. 겹치는 밑자리의 몫이 피할 수 없는 정확도 손실을 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 정확도 떨어짐을 밝혀 준다. 모델이 담는 힘을 키우면(더 너른 네트워크) 얽힌 강건한 결정 경계를 더 잘 그려 트레이드오프를 얼마쯤 눅일 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 구조(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 강건성이 어떻게 드러나는지 다루어라. 위협 모델이 보기 다룸과 어떻게 다른가?

</div>

??? success "연습문제 4 풀이"
    금융에서 겨루는 이는 알아내는 구조에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 핵심은 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 공격이 잇따르며 맞추어 간다(겨루는 이가 구조의 되받음을 보고 손본다). (3) 헛 정확도와 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 섭동 모델이 있어야 한다. 방어는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 기초의 길은 많이 걸러진다. $\square$

## 정리하며

이 절에서는 속임수 알아내기의 위협 모델, 속임수 알아내기의 적대적 학습, 강건한 속임수 알아내기를 따지는 자, 참으로 즐겨 쓸 길을 살펴보았다.

**살펴볼 거리**

1. Cartella, F., et al. (2021). "Adversarial Attacks on Fraud Detection Systems." Future Generation Computer Systems.
2. Chen, H., et al. (2020). "Robustness of Machine Learning Based Fraud Detection." ACM SIGKDD Workshop.
