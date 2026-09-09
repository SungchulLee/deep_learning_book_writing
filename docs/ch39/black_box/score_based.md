# 점수 바탕 치기

**점수 바탕 치기**은 모형의 날임 낌새 분포 $p(y|\mathbf{x})$이나 로짓을 볼 수 있음을 써서 기울기 없이 맞서는 보기를 짓는다. 흰 상자(기울기를 온전히 봄)와 판단 바탕(굳은 이름표만) 사이의 가운데 자리이며, 판단 바탕보다 물음이 훨씬 잘 든다.

---

## 1. 수학 틀

### 점수로 기울기 어림하기

모형 점수로 잃음 함수를 잴 수 있으면 여러 뽑기 꾀로 기울기를 어림할 수 있다.

**자리마다 어림하기:**

$$
\hat{g}_i = \frac{\mathcal{L}(f(\mathbf{x} + h\mathbf{e}_i)) - \mathcal{L}(f(\mathbf{x} - h\mathbf{e}_i))}{2h}
$$

차수가 $d$이면 온전한 기울기 어림에 물음이 $2d$번 든다.

**아무 방향으로 어림하기:**

$$
\hat{\nabla} \mathcal{L} \approx \frac{d}{n\sigma} \sum_{i=1}^n \left[\mathcal{L}(f(\mathbf{x} + \sigma \mathbf{u}_i)) - \mathcal{L}(f(\mathbf{x}))\right] \mathbf{u}_i
$$

여기서 $\mathbf{u}_i$은 낱 공에서 뽑는다. 차수와 상관없이 물음이 $n$번이면 된다.

### 앞선 앎을 곁들인 밴딧 다듬기

일리아스 등(2019)은 기울기 어림에 **자료에 매인 앞선 앎**을 넣어 물음이 더 잘 들게 했다. 고갱이 깨침은 맞서는 흔듦에 자리 얼개가 있다는 것이다(가까운 낱그림점끼리 비슷하게 흔들린다).

기울기 어림은 때에 매인 앞선 앎 $\mathbf{p}^{(t)}$을 쓴다.

$$
\hat{\nabla} \mathcal{L} \approx \frac{1}{\sigma}\left[\mathcal{L}(f(\mathbf{x} + \sigma \mathbf{q})) - \mathcal{L}(f(\mathbf{x}))\right] \mathbf{q}
$$

여기서 $\mathbf{q} = \beta \mathbf{p}^{(t)} + (1-\beta)\mathbf{u}$은 앞선 앎과 아무 방향을 섞은 것이고, 앞선 앎은 먹힌 치기의 방향을 보고 고친다.

---

## 2. 네모 치기

**네모 치기**(안드리우셴코 등, 2020)은 기울기 어림을 아예 쓰지 않고, 그 자리에 몰린 네모 꼴 고침으로 **아무 뒤지기** 꾀를 쓰는 점수 바탕 검은 상자 치기다.

### 알고리즘

되돌 때마다

1. 들임에서 네모 자리를 아무렇게나 고른다
2. 그 자리의 흔듦을 뽑는다
3. 잃음이 커질 때만 그 고침을 받는다

네모의 크기는 크게 비롯해 되돌수록 줄어, 거친 데서 고운 데로 뒤지는 꾀가 된다.

### 고갱이 결

- **기울기 어림이 없다**: 뽑기에 기댄 기울기 방법의 덤 값을 비껴간다
- **물음이 아주 잘 든다**: 흔히 물음 1,000~5,000번이면 된다
- **그 자리에 몰린 고침**: 그림의 자리 얼개를 쓴다
- **오토어택의 한 몫**: 검은 상자 몫을 맡는다

```python
import torch
import torch.nn as nn
from typing import Optional, Dict

class SimpleSquareAttack:
    """
    점수 바탕 검은 상자 자리를 위한 줄여 적은 네모 치기.
    
    아무 네모 꼴 흔듦을 쓰고 잃음이 나아지는지에 따라
    받거나 물린다.
    
    Parameters
    ----------
    model : nn.Module
        과녁 모형
    epsilon : float
        Linf 흔듦 예산
    max_queries : int
        가장 많은 물음 수
    p_init : float
        처음에 흔들 그림의 몫
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        max_queries: int = 5000,
        p_init: float = 0.8,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.epsilon = epsilon
        self.max_queries = max_queries
        self.p_init = p_init
        self.device = device or next(model.parameters()).device
        self.model.eval()
    
    def _get_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """틈 잃음을 셈한다(옳은 갈래 로짓 틈의 음수)."""
        with torch.no_grad():
            logits = self.model(x.to(self.device))
            # 틈 잃음: 가장 큰 다른 로짓 - 참 로짓
            true_logit = logits.gather(1, y.view(-1, 1)).squeeze()
            mask = torch.ones_like(logits).scatter_(1, y.view(-1, 1), 0)
            max_other = (logits * mask - (1 - mask) * 1e9).max(dim=1)[0]
            return max_other - true_logit
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """아무 네모 뒤지기로 맞서는 보기를 만든다."""
        x = x.to(self.device)
        y = y.to(self.device)
        N, C, H, W = x.shape
        
        # 아무 Linf 흔듦으로 첫자리를 잡는다
        x_adv = x + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
        
        best_loss = self._get_loss(x_adv, y)
        queries = torch.ones(N, device=self.device)
        
        for i in range(self.max_queries):
            # 맞추어 가는 네모 크기: 되돌수록 줄어든다
            p = self.p_init * (1 - i / self.max_queries)
            s = max(int(round(p * min(H, W))), 1)
            
            # 네모의 아무 자리
            r = torch.randint(0, H - s + 1, (N,))
            c_pos = torch.randint(0, W - s + 1, (N,))
            
            # 고침을 내놓는다: 네모 자리의 아무 값
            x_new = x_adv.clone()
            for b in range(N):
                # 네모의 아무 흔듦
                patch = torch.empty(C, s, s, device=self.device).uniform_(
                    -self.epsilon, self.epsilon
                )
                x_new[b, :, r[b]:r[b]+s, c_pos[b]:c_pos[b]+s] = \
                    x[b, :, r[b]:r[b]+s, c_pos[b]:c_pos[b]+s] + patch
            
            x_new = torch.clamp(x_new, 0, 1)
            # Linf 옭아맴을 지킨다
            delta = torch.clamp(x_new - x, -self.epsilon, self.epsilon)
            x_new = torch.clamp(x + delta, 0, 1)
            
            # 잃음이 나아지면 받는다
            new_loss = self._get_loss(x_new, y)
            improved = new_loss > best_loss
            x_adv[improved] = x_new[improved]
            best_loss[improved] = new_loss[improved]
            queries += 1
        
        return x_adv.detach()
```

---

## 3. 점수 바탕 방법 견주기

| 방법 | 보기마다의 물음 | 낌새가 있어야 함 | 고갱이 나은 점 |
|--------|----------------|----------------------|---------------|
| NES | 5,000~20,000 | 그렇다 | 온전한 기울기 어림 |
| 밴딧 | 2,000~10,000 | 그렇다 | 앞선 앎이 이끄는 잘 듦 |
| 네모 치기 | 1,000~5,000 | 그렇다(또는 로짓) | 기울기 어림이 없음 |
| SimBA | 1,000~10,000 | 그렇다 | 단순한 아무 뒤지기 |

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

점수 바탕 치기는 모형의 날임 낌새를 써서 기울기 없이 맞서는 보기를 짓는다. 물음이 잘 드는 데서는 네모 치기가 가장 앞서고, NES과 밴딧은 더 이치에 닿는 기울기 어림을 준다. API의 잦기 마디로 물음 예산이 좁은 금융 쓰임에서는 네모 치기의 잘 듦 덕에 이를 즐겨 쓴다.

**살펴볼 거리**

1. Andriushchenko, M., et al. (2020). "Square Attack: A Query-Efficient Black-Box Adversarial Attack via Random Search." ECCV.
2. Ilyas, A., Engstrom, L., & Madry, A. (2019). "Prior Convictions: Black-Box Adversarial Attacks with Bandits and Priors." ICLR.
3. Guo, C., et al. (2019). "Simple Black-Box Adversarial Attacks." ICML.
