# MART: 잘못 분류를 아는 강건한 학습

**MART**(잘못 분류를 아는 맞섬 든든 학습. 왕 등, 2020)은 옳게 가르기가 얼마나 어려운지에 따라 학습 보기마다 다른 짐을 주어 적대적 학습을 낫게 한다. 핵심 깨침은 보기가 강건성에 이바지하는 몫이 저마다 다르다는 것이다. 본디 가르기 어려운 보기가 든든하게 익히는 동안 더 어텐션을 받아야 한다.

---

## 1. 왜 하는가

여느 적대적 학습과 TRADES은 보기를 다 똑같이 다룬다. 그런데

- **쉬운 보기**(맑은 자신함이 높음, $p_y(\mathbf{x}) \approx 1$): 이미 잘 갈린다. 모델의 여유가 커서 잘 속지 않는다
- **어려운 보기**(맑은 자신함이 낮음, $p_y(\mathbf{x}) \approx 0$): 결정 경계 가까이 있어 적대적 섭동에 쉽게 틀리게 갈린다

MART은 방어의 힘을 가장 있어야 할 어려운 보기에 모은다.

---

## 2. 수학 꼴

### MART 손실

$$
\mathcal{L}_{\text{MART}} = \text{BCE}(f_\theta(\mathbf{x}_{\text{adv}}), y) + \lambda \cdot (1 - p_y(\mathbf{x})) \cdot \text{KL}(f_\theta(\mathbf{x}) \| f_\theta(\mathbf{x}_{\text{adv}}))
$$

여기서

- $\text{BCE}$은 적대적 샘플에 대한 두 값 엇갈린 엔트로피(북돋운 엇갈린 엔트로피)
- $p_y(\mathbf{x}) = \text{softmax}(f_\theta(\mathbf{x}))_y$은 맑은 입력에서 참 갈래의 낌새
- $(1 - p_y(\mathbf{x}))$은 **잘못 분류를 아는 짐**
- $\lambda$은 다독임의 계산

### 짐의 풀이

짐 $(1 - p_y(\mathbf{x}))$은 어텐션을 주는 구조 노릇을 한다.

| $p_y(\mathbf{x})$ | 짐 | 풀이 |
|-------------------|--------|----------------|
| 약 1.0(자신하며 정확도) | 약 0 | 다독임이 적다. 여기서는 모델이 든든하다 |
| 약 0.5(불확실성) | 약 0.5 | 다독임이 가운데다 |
| 약 0(틀리게 갈림) | 약 1.0 | 다독임이 크다. 가장 무르다 |

---

## 3. PyTorch로 짜기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict
from tqdm import tqdm

class MARTTrainer:
    """
    MART: 잘못 가름을 아는 든든한 익히기.
    
    한결같음 잃음에 (1 - p_y(x)) 짐을 주어
    맞서며 익히기의 힘을 어려운 보기에 모은다.
    
    Parameters
    ----------
    model : nn.Module
        익힐 모형
    lam : float
        KL 항의 다독임 셈
    epsilon : float
        흔듦 예산
    alpha : float
        PGD 걸음 크기
    num_iter : int
        PGD 되돌이
    """
    
    def __init__(
        self,
        model: nn.Module,
        lam: float = 6.0,
        epsilon: float = 8/255,
        alpha: float = 2/255,
        num_iter: int = 10,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.lam = lam
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
    
    def _pgd_attack(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """PGD 맞서는 보기를 만든다."""
        x_adv = x + torch.empty_like(x).uniform_(
            -self.epsilon, self.epsilon
        )
        x_adv = torch.clamp(x_adv, 0, 1)
        
        for _ in range(self.num_iter):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(self.model(x_adv), y)
            self.model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv.detach()
    
    def _mart_loss(
        self,
        logits_adv: torch.Tensor,
        logits_clean: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        MART 잃음을 셈한다.
        
        BCE(맞섬, y) + λ · (1 - p_y(x)) · KL(맑음 || 맞섬)
        """
        # 맞서는 보기의 북돋운 엇갈린 엔트로피
        adv_probs = F.softmax(logits_adv, dim=1)
        # BCE 꼴: 잣대 잡은 -log(p_y(x_adv))
        tmp = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(
            tmp[:, -1] == y, tmp[:, -2], tmp[:, -1]
        )
        loss_adv = F.cross_entropy(logits_adv, y) + \
                   F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
        
        # 잘못 가름을 아는 짐
        with torch.no_grad():
            p_clean = F.softmax(logits_clean, dim=1)
            p_y = p_clean.gather(1, y.view(-1, 1)).squeeze()
            weight = 1.0 - p_y  # 틀리게 갈린 것에 큰 짐
        
        # 짐을 준 KL 갈림
        kl = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_clean.detach(), dim=1),
            reduction='none'
        ).sum(dim=1)
        
        loss_kl = (weight * kl).mean()
        
        return loss_adv + self.lam * loss_kl
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """MART 잃음으로 한 판 익힌다."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='MART Training')
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            # 맑은 앞으로 걸음
            logits_clean = self.model(x)
            
            # 맞서는 보기를 만든다
            x_adv = self._pgd_attack(x, y)
            
            # 맞서는 앞으로 걸음
            logits_adv = self.model(x_adv)
            
            # MART 잃음
            loss = self._mart_loss(logits_adv, logits_clean, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            correct += (logits_clean.argmax(1) == y).sum().item()
            total += len(y)
            
            pbar.set_postfix({'loss': f'{total_loss/total:.4f}'})
        
        return {
            'loss': total_loss / total,
            'clean_accuracy': correct / total
        }
```

---

## 4. 적대적 학습 방법 비교

| 방법 | 핵심 깨침 | 클린 정확도 | 강건 정확도 | 하이퍼파라미터 |
|--------|----------|-----------|------------|----------------|
| 여느 적대적 학습 | 적대적 샘플의 가장 큰 손실 | 85% | 48% | $\varepsilon$ |
| TRADES | 드러내 놓은 트레이드오프 | 87% | 46% | $\beta$ |
| MART | 어려운 보기에 힘을 모음 | 86% | 49% | $\lambda$ |

CIFAR-10, $\varepsilon = 8/255$에서의 어림 특징과다.

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

MART의 잘못 분류를 아는 짐 주기는 가장 걸리는 곳, 곧 모델이 어려워하는 보기에서 강건성을 콕 집어 올린다. 그래서 갈래가 치우쳤거나 어려움이 들쭉날쭉한 데이터 꾸러미에 잘 듣는데, 이는 금융 쓰임에서 흔하다.

**살펴볼 거리**

1. Wang, Y., Zou, D., Yi, J., Bailey, J., Ma, X., & Gu, Q. (2020). "Improving Adversarial Robustness Requires Revisiting Misclassified Examples." ICLR.
