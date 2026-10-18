# 결정 기반 공격

**판단 기초 공격**은 가장 빡빡한 블랙박스 자리에서 움직인다. 겨루는 이는 자신함 점수나 확률 없이 모델의 **굳은 레이블** 추론 $\hat{y} = \arg\max_y f(\mathbf{x})$만 본다. 소식이 이토록 모자라도 판단 기초 공격은 섭동의 크기가 맞먹는 적대적 샘플을 찾아낸다. 다만 스코어 기반보다 물음이 더 든다.

---

## 1. 수학 틀

### 문제 차림

판단 판수만 주어졌을 때

$$
\mathcal{O}(\mathbf{x}) = \arg\max_y f_\theta(\mathbf{x})_y
$$

치는 이는 $\mathcal{O}(\mathbf{x}_{\text{adv}}) \neq y$이면서 $\|\mathbf{x}_{\text{adv}} - \mathbf{x}\|_p$이 가장 작은 $\mathbf{x}_{\text{adv}}$을 찾아야 한다.

그래디언트 소식도 이어지는 로스 신호도 없다. 어떤 점이 참 레이블과 다르게 갈리는지를 알려 주는 두 값 신호뿐이다.

---

## 2. 금 공격

**금 공격**(브렌델 등, 2018)은 결정 경계를 따라 아무렇게나 걸으며 본디 입력까지의 거리를 차츰 줄인다.

### 알고리즘

1. 맞선다고 알려진 점에서 비롯한다(다른 클래스의 아무 그림 따위)
2. 되돌 때마다:
    - **곧은 스텝**: $\mathbf{x}_{\text{adv}}$과 $\mathbf{x}_0$을 잇는 줄에 곧은 방향으로 흔든다
    - **밑자리 쪽 스텝**: $\mathbf{x}_0$ 쪽으로 조금 옮긴다
    - **받기/물리기**: 새 점이 여전히 맞설 때만 그 고침을 남긴다
3. 받는 비율이 약 50%가 되도록 스텝 크기를 맞춘다

금 공격은 보기마다 $O(10^3\text{-}10^5)$번 물어야 하지만 점수 소식은 있어야 하지 않다.

---

## 3. 홉스킵점프 공격

**홉스킵점프**(첸 등, 2020)은 판단 물음만으로 **금의 곧은 방향**을 어림해 금 공격을 낫게 한다.

### 핵심 깨침

결정 경계 가까운 점에서 금의 곧은 방향을 이렇게 어림한다.

$$
\hat{\nabla} \phi(\mathbf{x}) \approx \frac{1}{B} \sum_{b=1}^B \text{sign}\left(\mathcal{O}(\mathbf{x} + \delta \mathbf{u}_b) \neq y\right) \cdot \mathbf{u}_b
$$

여기서 $\mathbf{u}_b$은 아무 낱 벡터이고 $\delta$은 작은 스텝이다. 금의 곧은 방향을 이렇게 몬테카를로로 어림하면 굳은 레이블만으로도 그래디언트 같은 고침을 할 수 있다.

### 알고리즘의 스텝

1. **두 쪽 갈라 찾기**: $\mathbf{x}_0$과 $\mathbf{x}_{\text{adv}}$ 사이를 두 쪽 갈라 찾아 결정 경계 위의 점을 찾는다
2. **그래디언트 어림**: 아무렇게나 뽑아 금의 곧은 방향을 어림한다
3. **금 스텝**: 어림한 그래디언트 방향으로 옮긴다
4. **두 쪽 갈라 찾기**: 금으로 되비춘다
5. 스텝 크기를 줄여 가며 되풀이한다

---

## 4. PyTorch로 짜기

```python
import torch
import torch.nn as nn
from typing import Optional, Dict

class BoundaryAttack:
    """
    판단 바탕 검은 상자 자리를 위한 줄여 적은 금 치기.
    
    판단의 금을 따라 아무렇게나 걸으며
    본디 들임까지의 거리를 차츰 줄인다.
    
    Parameters
    ----------
    model : nn.Module
        타깃 모형(굳은 이름표만 쓴다)
    max_queries : int
        모형에 물을 가장 많은 수
    init_delta : float
        곧은 스텝의 처음 스텝 크기
    init_epsilon : float  
        밑자리 쪽 스텝의 처음 스텝 크기
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_queries: int = 25000,
        init_delta: float = 0.1,
        init_epsilon: float = 0.1,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.max_queries = max_queries
        self.delta = init_delta
        self.epsilon_step = init_epsilon
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self.queries = 0
    
    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        """굳은 이름표 미루어 봄을 얻는다(판단 판수를 흉내 낸다)."""
        self.queries += x.shape[0]
        with torch.no_grad():
            return self.model(x.to(self.device)).argmax(dim=1)
    
    def _is_adversarial(
        self, x: torch.Tensor, true_label: int
    ) -> bool:
        """x이 맞서는지 살핀다(참 이름표와 다른지)."""
        pred = self._predict(x.unsqueeze(0))
        return pred.item() != true_label
    
    def _binary_search_boundary(
        self, x_orig: torch.Tensor, x_adv: torch.Tensor,
        true_label: int, tol: float = 1e-4
    ) -> torch.Tensor:
        """두 쪽 갈라 찾아 판단의 금 위의 점을 찾는다."""
        low, high = 0.0, 1.0
        
        for _ in range(20):  # 두 쪽 갈라 찾기에 약 20번 묻는다
            mid = (low + high) / 2
            x_mid = (1 - mid) * x_orig + mid * x_adv
            
            if self._is_adversarial(x_mid, true_label):
                high = mid
            else:
                low = mid
        
        return (1 - high) * x_orig + high * x_adv
    
    def _attack_single(
        self, x: torch.Tensor, true_label: int
    ) -> torch.Tensor:
        """보기 하나를 친다."""
        x = x.to(self.device)
        C, H, W = x.shape
        
        # 첫자리: 맞서는 비롯 점을 찾는다
        # (들임 밭의 아무 그림)
        for _ in range(100):
            x_init = torch.rand_like(x)
            if self._is_adversarial(x_init, true_label):
                break
        else:
            return x  # 맞서는 첫자리를 찾지 못했다
        
        # 금까지 두 쪽 갈라 찾는다
        x_adv = self._binary_search_boundary(x, x_init, true_label)
        
        # 거듭 다듬기
        while self.queries < self.max_queries:
            # 곧은 흔듦
            noise = torch.randn_like(x)
            # x_adv - x 방향의 몫을 없앤다
            direction = x_adv - x
            direction_flat = direction.view(-1)
            noise_flat = noise.view(-1)
            noise_flat = noise_flat - (noise_flat @ direction_flat) / \
                        (direction_flat @ direction_flat + 1e-8) * direction_flat
            noise = noise_flat.view(x.shape)
            noise = noise / (noise.norm() + 1e-8)
            
            # 금을 따라 걷는다
            d_norm = (x_adv - x).norm()
            x_candidate = x_adv + self.delta * d_norm * noise
            
            # 밑자리 쪽으로 걷는다
            x_candidate = (1 - self.epsilon_step) * x_candidate + \
                         self.epsilon_step * x
            x_candidate = torch.clamp(x_candidate, 0, 1)
            
            # 여전히 맞서고 더 가까우면 받는다
            if self._is_adversarial(x_candidate, true_label):
                new_dist = (x_candidate - x).norm()
                old_dist = (x_adv - x).norm()
                if new_dist < old_dist:
                    x_adv = x_candidate
        
        return x_adv
    
    def generate(
        self, x: torch.Tensor, y: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """배치 하나에 대해 맞서는 보기를 만든다."""
        x_adv = x.clone()
        self.queries = 0
        
        for i in range(len(x)):
            x_adv[i] = self._attack_single(x[i], y[i].item())
        
        return x_adv.detach()
```

---

## 5. 물음 번거로움 비교

| 방법 | 보기마다의 물음 수 | 쓰는 소식 | 공격의 세기 |
|--------|---------------------|-----------------|-----------------|
| 금 공격 | $10^3 - 10^5$ | 굳은 레이블 | 가운데 |
| 홉스킵점프 | $10^3 - 10^4$ | 굳은 레이블 | 계산 |
| Sign-OPT | $10^3 - 10^4$ | 굳은 레이블 | 계산 |
| GeoDA | $10^2 - 10^3$ | 굳은 레이블 | 가운데 |

---

## 6. 판단 기초 공격이 걸리는 자리

다음과 같을 때 판단 기초 공격이 알맞은 위협 모델이다.

- 타깃 구조가 마지막 판단만 돌려줄 때(받음/물림, 삼/팜)
- 자신함 점수를 쓰는 이에게 보이지 않을 때
- 스코어 기반 공격을 막으려 API의 되받음을 일부러 줄였을 때

### 금융에서의 보기

- **미쁨 판단**: 신청하는 이는 점수가 아니라 받음/물림만 본다
- **속임수 알림**: 거래에 표시가 붙거나 풀릴 뿐 확률은 없다
- **거래 신호**: 구조가 자신함 없이 삼/쥠/팜만 낸다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 클래스를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 뉴럴 넷트웍의 강건성과 어떻게 이어지는지 밝혀라.

</div>

??? success "풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 뉴럴 넷트웍에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(그래디언트의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차원이 높은 모델이 무른 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 절에서 다룬 공격나 방어를 CIFAR-10의 ResNet-18 모델에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 공격 아래에서 클린 정확도와 강건 정확도를 알려라.

</div>

??? success "풀이"
    여느 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 절의 방법을 걸면 피처와는 재주에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 테두리를 준다. 정확도와 강건성의 트레이드오프는 기초부터 있는 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 든다. 아무렇게나 하는 시드 3개의 평균과 표준편차로 알려라. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
섭동 공 안에서 클래스별 데이터의 밑자리가 서로 겹친다고 볼 때, 모델이 담는 힘을 키우지 않고서는 어떤 방어도 맑은 데이터의 높은 정확도와 $\ell_\infty$ 섭동에 대한 높은 강건성을 함께 이룰 수 없음을 증명하여라.

</div>

??? success "풀이"
    두 클래스의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 기초 트레이드오프다. 겹치는 밑자리의 몫이 피할 수 없는 정확도 로스를 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 정확도 떨어짐을 밝혀 준다. 모델이 담는 힘을 키우면(더 너른 네트워크) 얽힌 강건한 결정 경계를 더 잘 그려 트레이드오프를 얼마쯤 눅일 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 구조(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 강건성이 어떻게 드러나는지 다루어라. 위협 모델이 보기 다룸과 어떻게 다른가?

</div>

??? success "풀이"
    금융에서 겨루는 이는 알아내는 구조에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 핵심은 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 공격이 잇따르며 맞추어 간다(겨루는 이가 구조의 되받음을 보고 손본다). (3) 헛 정확도와 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 섭동 모델이 있어야 한다. 방어는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 기초의 길은 많이 걸러진다. $\square$

## 정리하며

| 방법 | 물음 | 있어야 할 레이블 | 잘 맞는 자리 |
|--------|---------|---------------|----------|
| 금 공격 | 많음 | 굳은 것만 | 처음 둘러보기 |
| 홉스킵점프 | 가운데 | 굳은 것만 | 잘 드는 금 공격 |
| Sign-OPT | 가운데 | 굳은 것만 | 표적 공격 |

판단 기초 공격은 소식을 아무리 옥죄어도 적대적 공격을 막지 못하고 물음 값만 올릴 뿐임을 보여 준다. 이는 지킴이 걸린 쓰임의 API을 꾸미는 데 큰 뜻을 지닌다.

**살펴볼 거리**

1. Brendel, W., Rauber, J., & Bethge, M. (2018). "Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models." ICLR.
2. Chen, J., et al. (2020). "HopSkipJumpAttack: A Query-Efficient Decision-Based Attack." IEEE S&P.
3. Cheng, M., et al. (2019). "Sign-OPT: A Query-Efficient Hard-Label Adversarial Attack." ICLR.
