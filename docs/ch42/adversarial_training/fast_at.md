# Fast Adversarial Training

**빠른 적대적 학습**(웡 등, 2020)은 첫자리를 아무렇게나 잡은 **한 걸음 FGSM**으로 든든하게 익히며, 계산 비용의 몇 분의 일로 PGD 적대적 학습에 가까운 강건성에 다다른다. 핵심 알아냄은, 모자란다고 여겨졌던 FGSM 기초 학습이 제대로 된 아무 첫자리와 조심스러운 하이퍼파라미터 고르기를 곁들이면 잘 듣는다는 것이다.

---

## 1. 밑그림

굿펠로 등(2015)의 이른 일은 FGSM 적대적 샘플로 익히자고 내놓았으나, 이 길은 FGSM에만 든든하고 더 센 여러 걸음 공격에는 무른 모델을 낳는 것으로 드러났다. 매드리 등(2018)은 참으로 든든하려면 PGD 기초 학습이 있어야 하나 계산 비용이 10배 든다고 밝혔다.

웡 등(2020)은 한 걸음 학습을 다시 살펴, 어그러짐이 한 걸음 공격의 타고난 한계가 아니라 **무너지듯 지나친 맞춤** 때문임을 밝혔다.

---

## 2. 무너지듯 지나친 맞춤

### 문제

FGSM으로 맞서며 익히는 동안 강건 정확도가 한 판 만에 약 45%에서 약 0%으로 갑자기 무너질 수 있다. 모델이 **FGSM 전문가**가 되어 FGSM에는 든든하나 PGD에는 아주 무르다.

### 까닭

모델이 FGSM만 겨냥해 그래디언트를 가리는 법을 배워 든든하다는 거짓 느낌을 만든다. 그래디언트 한 걸음인 FGSM 공격은 그래디언트 가리기에 쉽게 속는다.

### 풀이

FGSM 걸음에 앞서 **첫자리를 아무렇게나 잡으면** 무너지듯 지나친 맞춤을 막는다.

$$
\begin{aligned}
\boldsymbol{\delta}_0 &\sim \text{Uniform}[-\varepsilon, \varepsilon]^d \\
\boldsymbol{\delta}_{\text{FGSM}} &= \Pi_\varepsilon\left(\boldsymbol{\delta}_0 + \alpha \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}_0), y))\right)
\end{aligned}
$$

아무 비롯 자리는 익히는 동안 적대적 샘플을 다양하게 하여, 모델이 붙박인 공격 방향에만 길드는 것을 막는다.

---

## 3. 알고리즘

**알고리즘: 빠른 적대적 학습**

```
판마다:
    잔 배치 (x, y)마다:
        1. 아무 첫자리: δ ~ Uniform[-ε, ε]
        2. 그래디언트를 셈한다: g = ∇_x L(f_θ(x + δ), y)
        3. FGSM 걸음: δ ← Π_ε(δ + α · sign(g))
        4. 잃음을 셈한다: L = CE(f_θ(clip(x + δ, 0, 1)), y)
        5. 고친다: θ ← θ - η · ∇_θ L
```

온 값: **배치마다 앞으로-되돌아 걸음 2번**(하나는 그래디언트, 하나는 파라메터 고침).

---

## 4. PyTorch로 짜기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict
from tqdm import tqdm

class FastAdversarialTrainer:
    """
    FGSM과 아무 첫자리를 쓰는 빠른 맞서며 익히기.
    
    여느 익힘의 약 2배 값으로 PGD에 가까운 든든함을 이룬다.
    
    Parameters
    ----------
    model : nn.Module
        익힐 모형
    epsilon : float
        흔듦 예산
    alpha : float
        FGSM 걸음 크기(흔히 엡실론의 1.25배)
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        alpha: float = 10/255,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha  # 엡실론보다 조금 크다
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
    
    def _fgsm_with_random_start(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """고르게 아무렇게나 첫자리를 잡는 FGSM 치기."""
        # 엡실론 공 안에서 아무 첫자리
        delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad_(True)
        
        # 흔든 들임으로 앞으로 걸음
        logits = self.model(torch.clamp(x + delta, 0, 1))
        loss = F.cross_entropy(logits, y)
        
        # 되돌아 걸음
        self.model.zero_grad()
        loss.backward()
        
        # FGSM 걸음
        with torch.no_grad():
            delta = delta + self.alpha * delta.grad.sign()
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        
        return torch.clamp(x + delta, 0, 1).detach()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """한 판 익힌다."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Fast AT')
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            # FGSM과 아무 첫자리로 맞서는 보기를 만든다
            x_adv = self._fgsm_with_random_start(x, y)
            
            # 맞서는 보기로 모형을 고친다
            optimizer.zero_grad()
            logits = self.model(x_adv)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
            
            pbar.set_postfix({
                'loss': f'{total_loss/total:.4f}',
                'acc': f'{correct/total:.2%}'
            })
        
        return {
            'loss': total_loss / total,
            'accuracy': correct / total
        }
```

---

## 5. 무너지듯 지나친 맞춤 막기

### 강건 정확도를 보고 일찍 멈추기

익히는 동안 PGD 강건 정확도를 지켜보고 갑자기 떨어지면 멈춘다.

```python
def check_catastrophic_overfitting(
    model, test_loader, epsilon, device, prev_robust_acc
):
    """무너지듯 지나친 맞춤이 일어났는지 짚어낸다."""
    model.eval()
    # 일부에 대해 빠른 PGD-10 따짐
    correct = 0
    total = 0
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        if total > 1000:
            break
        # PGD-10 치기
        x_adv = pgd_attack(model, x, y, epsilon, steps=10)
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
    
    robust_acc = correct / total
    
    # 든든한 맞음이 10% 넘게 떨어지면 표시한다
    if prev_robust_acc - robust_acc > 0.1:
        print(f"알림: 무너지듯 지나친 맞춤을 짚어냈다! "
              f"{prev_robust_acc:.2%} -> {robust_acc:.2%}")
        return True, robust_acc
    
    return False, robust_acc
```

---

## 6. 잘 드는 적대적 학습 방법 비교

| 방법 | 배치마다 걸음 | 견준 값 | 클린 정확도 | 강건 정확도 |
|--------|-------------|---------------|-----------|------------|
| PGD 적대적 학습(10걸음) | 11 | 10배 | 85% | 48% |
| 값싼 적대적 학습($m=8$) | 8 | 약 1.2배 | 83% | 43% |
| **빠른 적대적 학습** | **2** | **약 2배** | **83%** | **43%** |
| FGSM 적대적 학습(아무 첫자리 없음) | 2 | 2배 | 90% | 0%* |

*아무 첫자리가 없으면 무너지듯 지나친 맞춤이 일어난다.

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

빠른 적대적 학습은 아무 첫자리를 곁들이면 한 걸음 적대적 학습도 쓸 만함을 보여 준다. 여느 학습의 2배 값이면 되므로, 큰 금융 모델 학습을 아우른 계산이 넉넉지 않은 자리에서 강건성을 얻는 손에 잡히는 길이 된다.

**살펴볼 거리**

1. Wong, E., Rice, L., & Kolter, J. Z. (2020). "Fast is Better than Free: Revisiting Adversarial Training." ICLR.
2. Andriushchenko, M., & Flammarion, N. (2020). "Understanding and Improving Fast Adversarial Training." NeurIPS.
