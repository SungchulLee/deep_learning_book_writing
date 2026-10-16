# 쿼리 기반 공격

**물음 기초 블랙박스 공격**은 타깃 모델에 거듭 묻고 그 되받음으로 섭동 찾기를 이끌어 적대적 샘플을 만든다. 물음이 없는 전이되는 공격과 달리 타깃 모델과 곧바로 마주하지만 그 속은 알 필요가 없다. 마디 지어진 **물음 예산** 안에서 공격이 잘 먹히게 하는 것이 어려운 대목이다.

---

## 1. 공격의 클래스

물음 기초 공격은 물음마다 얻는 소식에 따라 갈린다.

| 클래스 | 물음마다의 소식 | 물음이 잘 듦 | 공격의 세기 |
|------|----------------------|------------------|-----------------|
| **스코어 기반** | 온 확률 벡터 $p(y|\mathbf{x})$ | 높음 | 계산 |
| **판단 기초** | 굳은 레이블만 $\hat{y} = \arg\max_y p(y|\mathbf{x})$ | 낮음 | 가운데 |
| **위 $k$개 기초** | 위 $k$개 클래스와 점수 | 가운데 | 계산 |

---

## 2. 마디 있는 차로 그래디언트 어림하기

스코어 기반 공격의 핵심은 함수 값만으로 **그래디언트를 어림하는** 것이다. 로스 $\mathcal{L}(f(\mathbf{x}), y)$을 볼 수 있으면 쪽미분을 어림할 수 있다.

**앞으로 차:**

$$
\frac{\partial \mathcal{L}}{\partial x_i} \approx \frac{\mathcal{L}(f(\mathbf{x} + h\mathbf{e}_i), y) - \mathcal{L}(f(\mathbf{x}), y)}{h}
$$

**가운데 차(더 맞다):**

$$
\frac{\partial \mathcal{L}}{\partial x_i} \approx \frac{\mathcal{L}(f(\mathbf{x} + h\mathbf{e}_i), y) - \mathcal{L}(f(\mathbf{x} - h\mathbf{e}_i), y)}{2h}
$$

**값:** 차원이 $d$인 입력에서 그래디언트 어림마다 $O(d)$번 물어야 한다. $d \approx 3 \times 32 \times 32 = 3{,}072$인 그림(CIFAR-10)에서는 비싸지만 할 만하다.

### 아무 방향으로 어림하기(NES)

타고난 진화 꾀(NES)은 아무 방향을 써서 그래디언트를 더 잘 어림한다.

$$
\nabla_\mathbf{x} \mathcal{L} \approx \frac{1}{n\sigma} \sum_{i=1}^n \mathcal{L}(f(\mathbf{x} + \sigma \mathbf{u}_i), y) \cdot \mathbf{u}_i
$$

여기서 $\mathbf{u}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$은 아무 가우스 방향이다. 입력 차원과 상관없이 물음이 $n$번(흔히 $n = 50\text{-}200$)이면 된다.

---

## 3. PyTorch로 짜기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class NESAttack:
    """
    타고난 진화 꾀(NES)에 기댄 검은 상자 치기.
    
    아무 방향을 뽑아 그래디언트를 어림한 뒤
    PGD 결로 고친다.
    
    Parameters
    ----------
    model : nn.Module
        타깃 모형(검은 상자 물음 판수로 다룬다)
    epsilon : float
        흔듦 예산(Linf)
    num_samples : int
        그래디언트 어림마다의 아무 방향 수
    sigma : float
        NES에서 뽑을 때의 표준편차
    step_size : float
        치기의 스텝 크기
    max_queries : int
        받아 주는 온 물음의 가장 많은 수
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        num_samples: int = 100,
        sigma: float = 0.001,
        step_size: float = 0.01,
        max_queries: int = 10000,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.epsilon = epsilon
        self.num_samples = num_samples
        self.sigma = sigma
        self.step_size = step_size
        self.max_queries = max_queries
        self.device = device or next(model.parameters()).device
        
        self.model.eval()
        self.model.to(self.device)
        self.total_queries = 0
    
    def _query(self, x: torch.Tensor) -> torch.Tensor:
        """모형에 묻고 로짓을 돌려준다(API 부름을 흉내 낸다)."""
        self.total_queries += x.shape[0]
        with torch.no_grad():
            return self.model(x.to(self.device))
    
    def _estimate_gradient(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        맞짝으로 뽑는 NES으로 그래디언트를 어림한다.
        
        흩어짐을 줄이려 (u, -u) 짝을 쓴다.
        """
        batch_size = x.shape[0]
        grad_estimate = torch.zeros_like(x)
        
        for _ in range(self.num_samples // 2):
            # 아무 방향
            u = torch.randn_like(x)
            
            # 앞뒤로 묻는다
            x_plus = x + self.sigma * u
            x_minus = x - self.sigma * u
            
            logits_plus = self._query(x_plus)
            logits_minus = self._query(x_minus)
            
            # 잃음의 차
            loss_plus = F.cross_entropy(logits_plus, y, reduction='none')
            loss_minus = F.cross_entropy(logits_minus, y, reduction='none')
            
            # NES 그래디언트 어림(맞짝)
            diff = (loss_plus - loss_minus).view(-1, 1, 1, 1)
            grad_estimate += diff * u
        
        grad_estimate /= (self.num_samples * self.sigma)
        return grad_estimate
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """NES-PGD으로 맞서는 보기를 만든다."""
        x = x.to(self.device)
        y = y.to(self.device)
        self.total_queries = 0
        
        x_adv = x.clone()
        
        max_iter = self.max_queries // self.num_samples
        
        for t in range(max_iter):
            if self.total_queries >= self.max_queries:
                break
            
            # 아직 쳐야 할 보기를 살핀다
            with torch.no_grad():
                pred = self.model(x_adv).argmax(dim=1)
                still_correct = (pred == y)
            
            if not still_correct.any():
                break
            
            # 그래디언트를 어림한다
            grad = self._estimate_gradient(x_adv, y)
            
            # PGD 결의 고침
            x_adv = x_adv + self.step_size * grad.sign()
            
            # 되비춘다
            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv.detach()
    
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_adv: torch.Tensor
    ) -> Dict[str, float]:
        """물음 수와 함께 치기를 따진다."""
        with torch.no_grad():
            clean_pred = self.model(x.to(self.device)).argmax(1)
            adv_pred = self.model(x_adv.to(self.device)).argmax(1)
            y_dev = y.to(self.device)
        
        return {
            'clean_accuracy': (clean_pred == y_dev).float().mean().item(),
            'robust_accuracy': (adv_pred == y_dev).float().mean().item(),
            'attack_success_rate': (adv_pred != y_dev).float().mean().item(),
            'total_queries': self.total_queries,
            'avg_queries_per_example': self.total_queries / len(x)
        }
```

---

## 4. 물음이 잘 드는 정도 비교

| 방법 | 보기마다의 물음 수 | 먹힌 비율 | 그래디언트 얻기 |
|--------|---------------------|--------------|-----------------|
| 자리마다의 마디 있는 차 | 되돌 때마다 $O(d)$ | 높음 | 온전한 어림 |
| NES(표본 100개) | 되돌 때마다 약 100 | 높음 | 어림 |
| 밴딧(앞선 앎) | 되돌 때마다 약 50 | 높음 | 데이터에 매임 |
| 네모 공격 | 되돌 때마다 약 1 | 가운데~높음 | 없음 |

---

## 5. 참으로 헤아릴 것

### 금융 자리의 물음 예산

금융 API은 흔히 부르는 잦기를 마디 짓는다. 금융 모델의 그럴듯한 물음 예산은 이렇다.

| 자리 | 흔한 예산 | 까닭 |
|---------|---------------|-----------|
| 열린 미쁨 API | 물음 100~1000 | 잦기가 마디 지어진 끝자리 |
| 거래 신호 API | 물음 10~100 | 제때 늦음 요건 |
| 안쪽 모델 살피기 | 물음 10,000 넘음 | 다스려진 따짐 |

### 물음 공격을 막기

- **물음 알아내기**: 비슷한 입력의 수상한 피처를 지켜본다
- **잦기 마디 짓기**: 쓰는 이나 한 판마다 물음 수를 제약된이다
- **출력 흔들기**: 그래디언트 어림을 막으려 모델 출력에 잡음을 더한다
- **상태를 지닌 알아내기**: 물음의 이음을 좇고 튀는 피처에 표시한다

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
    여느 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 마디의 방법을 걸면 피처와는 재주에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 테두리를 준다. 정확도와 강건성의 트레이드오프는 기초부터 있는 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 든다. 아무렇게나 하는 시드 3개의 평균과 표준편차로 알려라. $\square$

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

이 절에서는 공격의 클래스, 마디 있는 차로 그래디언트 어림하기, PyTorch로 짜기, 물음이 잘 드는 정도 비교를 살펴보았다.

**살펴볼 거리**

1. Ilyas, A., et al. (2018). "Black-Box Adversarial Attacks with Limited Queries and Information." ICML.
2. Ilyas, A., Engstrom, L., & Madry, A. (2019). "Prior Convictions: Black-Box Adversarial Attacks with Bandits and Priors." ICLR.
3. Andriushchenko, M., et al. (2020). "Square Attack: A Query-Efficient Black-Box Adversarial Attack via Random Search." ECCV.
