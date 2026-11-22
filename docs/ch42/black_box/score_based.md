# 점수 기반 공격

**스코어 기반 공격**은 모델의 출력 확률 분포 $p(y|\mathbf{x})$나 로짓을 볼 수 있다는 점을 이용해 그래디언트 없이 적대적 샘플을 만든다. 화이트박스(그래디언트를 다 봄)와 결정 기반(하드 레이블만 봄) 사이에 놓이며, 결정 기반 공격보다 질의 효율이 훨씬 높다.

---

## 1. 수학적 틀

### 점수로 그래디언트 근사하기

모델 점수로 로스 함수를 잴 수 있으면 여러 샘플링 전략으로 그래디언트를 근사할 수 있다.

**좌표별 근사:**

$$
\hat{g}_i = \frac{\mathcal{L}(f(\mathbf{x} + h\mathbf{e}_i)) - \mathcal{L}(f(\mathbf{x} - h\mathbf{e}_i))}{2h}
$$

차원이 $d$이면 그래디언트 전체를 한 번 근사하는 데 질의가 $2d$번 든다.

**무작위 방향으로 근사하기:**

$$
\hat{\nabla} \mathcal{L} \approx \frac{d}{n\sigma} \sum_{i=1}^n \left[\mathcal{L}(f(\mathbf{x} + \sigma \mathbf{u}_i)) - \mathcal{L}(f(\mathbf{x}))\right] \mathbf{u}_i
$$

여기서 $\mathbf{u}_i$는 단위 구면에서 균일하게 뽑는다. 차원과 상관없이 근사 한 번에 질의가 $n$번이면 된다.

### 사전 정보를 쓰는 밴딧 최적화

Ilyas 등(2019)은 그래디언트 근사에 **데이터 의존 사전 정보**(prior)를 넣어 질의 효율을 높였다. 핵심 통찰은 그래디언트에 공간 구조가 있다는 것이다(가까운 픽셀끼리 그래디언트가 비슷하다). 시간 축의 사전 정보도 쓴다 — 연속한 반복의 그래디언트는 서로 닮았다.

그래디언트 근사는 반복마다 갱신되는 사전 정보 $\mathbf{p}^{(t)}$를 쓴다.

$$
\hat{\nabla} \mathcal{L} \approx \frac{1}{\sigma}\left[\mathcal{L}(f(\mathbf{x} + \sigma \mathbf{q})) - \mathcal{L}(f(\mathbf{x}))\right] \mathbf{q}
$$

여기서 $\mathbf{q} = \beta \mathbf{p}^{(t)} + (1-\beta)\mathbf{u}$는 사전 정보와 무작위 방향을 섞은 것이고, 사전 정보는 앞서 얻은 그래디언트 근사를 보고 갱신한다.

---

## 2. Square Attack

**Square Attack**(Andriushchenko 등, 2020)은 그래디언트 근사를 아예 쓰지 않고, 정사각형 영역에 국한된 갱신으로 **무작위 탐색**을 하는 스코어 기반 블랙박스 공격이다.

### 알고리즘

반복마다

1. 입력에서 정사각형 영역을 무작위로 고른다
2. 그 영역의 섭동을 새로 뽑는다
3. 로스가 커질 때만 그 갱신을 받아들인다

정사각형의 크기는 크게 시작해 반복할수록 줄어, 거친 탐색에서 세밀한 탐색으로 옮아간다.

### 핵심 특징

- **그래디언트 근사가 없다**: 샘플링 기반 그래디언트 방법의 추가 비용을 피한다
- **질의 효율이 아주 높다**: 흔히 질의 1,000~5,000번이면 된다
- **국소화된 갱신**: 이미지의 공간 구조를 이용한다
- **AutoAttack의 구성 요소**: 그 안에서 블랙박스 공격을 맡는다

```python
import torch
import torch.nn as nn
from typing import Optional, Dict

class SimpleSquareAttack:
    """
    점수 바탕 검은 상자 자리를 위한 줄여 적은 네모 치기.
    
    아무 네모 형태 섭동을 쓰고 로스가 나아지는지에 따라
    받거나 물린다.
    
    Parameters
    ----------
    model : nn.Module
        타깃 모형
    epsilon : float
        Linf 섭동 예산
    max_queries : int
        가장 많은 질의 수
    p_init : float
        처음에 흔들 그림의 기여도
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
        """틈 로스를 셈한다(옳은 클래스 로짓 틈의 음수)."""
        with torch.no_grad():
            logits = self.model(x.to(self.device))
            # 틈 로스: 가장 큰 다른 로짓 - 참 로짓
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
        
        # 아무 Linf 섭동으로 첫자리를 잡는다
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
                # 네모의 아무 섭동
                patch = torch.empty(C, s, s, device=self.device).uniform_(
                    -self.epsilon, self.epsilon
                )
                x_new[b, :, r[b]:r[b]+s, c_pos[b]:c_pos[b]+s] = \
                    x[b, :, r[b]:r[b]+s, c_pos[b]:c_pos[b]+s] + patch
            
            x_new = torch.clamp(x_new, 0, 1)
            # Linf 옭아맴을 지킨다
            delta = torch.clamp(x_new - x, -self.epsilon, self.epsilon)
            x_new = torch.clamp(x + delta, 0, 1)
            
            # 로스가 나아지면 받는다
            new_loss = self._get_loss(x_new, y)
            improved = new_loss > best_loss
            x_adv[improved] = x_new[improved]
            best_loss[improved] = new_loss[improved]
            queries += 1
        
        return x_adv.detach()
```

---

## 3. 스코어 기반 방법 비교

| 방법 | 샘플마다의 질의 | 확률이 필요함 | 핵심 장점 |
|--------|----------------|----------------------|---------------|
| NES | 5,000~20,000 | 그렇다 | 원리적인 그래디언트 근사 |
| 밴딧 | 2,000~10,000 | 그렇다 | 사전 정보가 주는 효율 |
| Square Attack | 1,000~5,000 | 그렇다(또는 로짓) | 그래디언트 근사가 없음 |
| SimBA | 1,000~10,000 | 그렇다 | 단순한 무작위 탐색 |

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
    일반 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 절의 방법을 적용한 결과는 기법에 따라 다르다. 적대적 트레인은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 바운드를 준다. 정확도와 강건성의 트레이드오프는 근본적인 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 떨어진다. 무작위 시드 3개의 평균과 표준편차로 보고하여라. $\square$

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

스코어 기반 공격은 모델의 출력 확률을 써서 그래디언트 없이 적대적 샘플을 만든다. 질의 효율이 중요한 곳에서는 Square Attack이 가장 앞서고, NES와 밴딧은 더 원리적인 그래디언트 근사를 준다. API의 호출 빈도 제한으로 질의 예산이 좁은 금융 응용에서는 그 효율 덕에 Square Attack을 즐겨 쓴다.

**살펴볼 거리**

1. Andriushchenko, M., et al. (2020). "Square Attack: A Query-Efficient Black-Box Adversarial Attack via Random Search." ECCV.
2. Ilyas, A., Engstrom, L., & Madry, A. (2019). "Prior Convictions: Black-Box Adversarial Attacks with Bandits and Priors." ICLR.
3. Guo, C., et al. (2019). "Simple Black-Box Adversarial Attacks." ICML.
