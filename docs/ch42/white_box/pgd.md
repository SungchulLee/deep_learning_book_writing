# PGD 공격

**투영 경사 하강법(PGD)**은 FGSM을 여러 스텝 반복 공격으로 넓혀 훨씬 세게 만든다. 매드리 등(2018)이 내놓았으며, 적대적 강건성을 평가하는 사실상의 기준으로 여겨지고 강건한 적대적 학습의 기초가 된다.

---

## 1. 수학 기초

### FGSM에서 반복 공격으로

FGSM은 그래디언트 방향으로 큰 스텝을 한 번 밟는다. 이는 다음 까닭에 가장 좋지 않다.

1. $\mathbf{x}$에서 멀어지면 선형 근사가 나빠진다
2. 한 스텝이면 가장 좋은 자리를 지나치거나 못 미칠 수 있다
3. 로스의 터는 볼록하지 않다

PGD은 **작은 스텝을 여러 번** 밟고 반복마다 그래디언트를 다시 셈해 이를 푼다.

### PGD 형태

PGD은 제약된 최적화 문제

$$
\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)
$$

를 투영 경사 상승법으로 푼다. 반복마다의 고침은

$$
\mathbf{x}^{(t+1)} = \Pi_{\mathbf{x} + \mathcal{S}}\left( \mathbf{x}^{(t)} + \alpha \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}^{(t)}), y)) \right)
$$

여기서

- $\Pi_{\mathbf{x} + \mathcal{S}}$은 $\mathbf{x}$ 둘레 $\varepsilon$ 공으로 투영한다
- $\alpha$은 스텝 크기
- $\mathcal{S} = \{\boldsymbol{\delta} : \|\boldsymbol{\delta}\|_p \leq \varepsilon\}$은 받아 주는 섭동의 모임

### 알고리즘

**알고리즘: PGD 공격**

**입력:** 클린 샘플 $\mathbf{x}$, 레이블 $y$, 모델 $f_\theta$, 엡실론 $\varepsilon$, 스텝 크기 $\alpha$, 회귀 $T$

**출력:** 적대적 샘플 $\mathbf{x}_{\text{adv}}$

1. **초기화:** $\mathbf{u} \sim \text{Uniform}[-\varepsilon, \varepsilon]^d$일 때 $\mathbf{x}^{(0)} = \mathbf{x} + \mathbf{u}$
2. $t = 0, 1, \ldots, T-1$**마다**:
   - 그래디언트를 계산한다: $\mathbf{g} = \nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}^{(t)}), y)$
   - 고친다: $\tilde{\mathbf{x}}^{(t+1)} = \mathbf{x}^{(t)} + \alpha \cdot \text{sign}(\mathbf{g})$
   - 투영한다: $\mathbf{x}^{(t+1)} = \Pi_{\varepsilon}(\tilde{\mathbf{x}}^{(t+1)}, \mathbf{x})$
   - 옳은 자리로 잘라 낸다: $\mathbf{x}^{(t+1)} = \text{clip}(\mathbf{x}^{(t+1)}, 0, 1)$
3. **돌려준다:** $\mathbf{x}_{\text{adv}} = \mathbf{x}^{(T)}$

### 투영 연산

**$\ell_\infty$ 투영:**

$$
\Pi_\varepsilon^{\infty}(\tilde{\mathbf{x}}, \mathbf{x})_i = \text{clip}(\tilde{x}_i, x_i - \varepsilon, x_i + \varepsilon)
$$

**$\ell_2$ 투영:**

$$
\Pi_\varepsilon^{2}(\tilde{\mathbf{x}}, \mathbf{x}) = 
\begin{cases}
\tilde{\mathbf{x}} & \text{if } \|\tilde{\mathbf{x}} - \mathbf{x}\|_2 \leq \varepsilon \\
\mathbf{x} + \varepsilon \cdot \frac{\tilde{\mathbf{x}} - \mathbf{x}}{\|\tilde{\mathbf{x}} - \mathbf{x}\|_2} & \text{otherwise}
\end{cases}
$$

### 스텝 크기 고르기

| 전략 | 식 | 붙임말 |
|----------|---------|-------|
| **곧게** | $\alpha = \varepsilon / T$ | 조심스럽다 |
| **기준 잡은 곧게** | $\alpha = 2\varepsilon / T$ | 여느 값(매드리 등) |
| **세게** | $\alpha = 2.5\varepsilon / T$ | 더 빨리 모인다 |

---

## 2. PyTorch로 구현하기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Dict

class PGD:
    """
    투영 경사 하강법(PGD) 치기.
    
    Parameters
    ----------
    model : nn.Module
        칠 뉴럴 네트워크
    epsilon : float
        가장 큰 섭동의 크기
    alpha : float
        되돌 때마다의 스텝 크기
    num_iter : int
        PGD 회귀 횟수
    norm : str
        노름 옭아맴('linf' 또는 'l2')
    random_init : bool
        첫자리를 아무렇게나 잡을지
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        alpha: Optional[float] = None,
        num_iter: int = 10,
        norm: Literal['linf', 'l2'] = 'linf',
        random_init: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha if alpha else 2 * epsilon / num_iter
        self.num_iter = num_iter
        self.norm = norm
        self.random_init = random_init
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device or next(model.parameters()).device
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.model.eval()
        self.model.to(self.device)
    
    def _project(self, x_adv: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """엡실론 공으로 되비춘다."""
        delta = x_adv - x
        
        if self.norm == 'linf':
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        elif self.norm == 'l2':
            delta_flat = delta.view(delta.shape[0], -1)
            norm = delta_flat.norm(p=2, dim=1, keepdim=True)
            factor = torch.clamp(norm / self.epsilon, min=1.0)
            delta = (delta_flat / factor).view(delta.shape)
        
        return torch.clamp(x + delta, self.clip_min, self.clip_max)
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """PGD으로 맞서는 보기를 만든다."""
        x = x.to(self.device)
        y = y.to(self.device)
        
        # 첫자리를 잡는다
        if self.random_init:
            if self.norm == 'linf':
                delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            else:
                delta = torch.randn_like(x)
                delta = delta / delta.view(len(x), -1).norm(p=2, dim=1, keepdim=True).view(-1,1,1,1)
                delta = delta * self.epsilon * torch.rand(len(x), 1, 1, 1, device=self.device)
            x_adv = torch.clamp(x + delta, self.clip_min, self.clip_max)
        else:
            x_adv = x.clone()
        
        # PGD 회귀
        for _ in range(self.num_iter):
            x_adv.requires_grad_(True)
            logits = self.model(x_adv)
            
            if targeted:
                loss = -self.loss_fn(logits, target_labels.to(self.device))
            else:
                loss = self.loss_fn(logits, y)
            
            self.model.zero_grad()
            loss.backward()
            grad = x_adv.grad.data
            
            # 그래디언트 한 스텝
            if self.norm == 'linf':
                x_adv = x_adv.detach() + self.alpha * torch.sign(grad)
            else:
                grad_norm = grad.view(len(x), -1).norm(p=2, dim=1, keepdim=True).view(-1,1,1,1)
                x_adv = x_adv.detach() + self.alpha * grad / (grad_norm + 1e-8)
            
            x_adv = self._project(x_adv, x)
        
        return x_adv.detach()
    
    def evaluate(self, x: torch.Tensor, y: torch.Tensor, x_adv: torch.Tensor) -> Dict[str, float]:
        """치기가 얼마나 잘 먹히는지 따진다."""
        with torch.no_grad():
            clean_acc = (self.model(x.to(self.device)).argmax(1) == y.to(self.device)).float().mean()
            robust_acc = (self.model(x_adv.to(self.device)).argmax(1) == y.to(self.device)).float().mean()
            delta = (x_adv - x).view(len(x), -1)
            
        return {
            'clean_accuracy': clean_acc.item(),
            'robust_accuracy': robust_acc.item(),
            'attack_success_rate': 1 - robust_acc.item(),
            'avg_linf': delta.abs().max(dim=1)[0].mean().item(),
            'avg_l2': delta.norm(p=2, dim=1).mean().item()
        }
```

---

## 3. 비교: FGSM과 PGD

| 피처 | FGSM | PGD |
|--------|------|-----|
| 스텝 | 1 | $T$(10~100) |
| 초기화 | 없음 | 무작위로 |
| 세기 | 여림 | 계산 |
| 빠르기 | 빠름 | $T$배 느림 |

**흔한 피처와(CIFAR-10, ε=8/255):**

| 방법 | 먹힌 비율 |
|--------|--------------|
| FGSM | 약 65% |
| PGD-10 | 약 85% |
| PGD-40 | 약 92% |

---

## 4. 클래스

### 모멘텀 반복 FGSM(MI-FGSM)

$$
\mathbf{g}^{(t)} = \mu \cdot \mathbf{g}^{(t-1)} + \frac{\nabla_\mathbf{x} \mathcal{L}}{\|\nabla_\mathbf{x} \mathcal{L}\|_1}
$$

그래디언트의 모멘텀을 쌓아 전이를 낫게 한다.

### 오토-PGD(APGD)

로스가 나아지는 정도에 따라 스텝 크기를 맞춘다. AutoAttack의 핵심 기여도다.

---

## 5. 적대적 학습와의 이어짐

PGD은 강건하게 학습하는 데 쓰는 여느 공격이다.

$$
\min_\theta \mathbb{E}_{(\mathbf{x},y)}\left[\max_{\|\boldsymbol{\delta}\|_\infty \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)\right]
$$

안쪽의 최대화를 PGD으로 근사한다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 클래스를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 뉴럴 네트워크의 강건성과 어떻게 이어지는지 설명하여라.

</div>

??? success "풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 뉴럴 네트워크에서는 국소 선형 근사 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(그래디언트의 부호를 쓴다)이 왜 잘 통하는지를 설명해 준다. 차원이 높은 모델이 취약한 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

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
    두 클래스의 서포트가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 기초 트레이드오프다. 겹치는 서포트의 비중이 피할 수 없는 정확도 손실을 정한다. 자연 이미지 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 관찰되는 10~15%의 정확도 하락을 설명해 준다. 모델 용량을 키우면(더 넓은 네트워크) 복잡한 강건 결정 경계를 더 잘 그려 트레이드오프를 어느 정도 완화할 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 시스템(사기 탐지나 거래 신호 생성 등)에서 적대적 강건성이 어떻게 드러나는지 논하여라. 위협 모델이 비전과 어떻게 다른가?

</div>

??? success "풀이"
    금융에서 적대자는 탐지 시스템에 맞추어 전략을 바꾸는 교묘한 집단(사기꾼, 시장 조작자)이다. 비전과 다른 핵심은 이렇다. (1) 섭동할 수 있는 피처가 경제적으로 그럴듯한 범위로 제약된다(사기꾼이 자기 거래 이력을 통째로 바꿀 수는 없다). (2) 공격이 순차적이고 적응적이다(적대자가 시스템의 응답을 보고 고친다). (3) 거짓 양성과 거짓 음성의 비용이 비대칭이다(정상 거래를 막는 것과 사기를 놓치는 것). (4) $\ell_p$ 노름은 의미가 없고 도메인에 맞는 섭동 모델이 있어야 한다. 방어는 적응형 적대자에게도 강건해야 하므로, 탐지 기준이 알려지면 회피할 수 있는 탐지 기반 방법은 쓸모가 크게 줄어든다. $\square$

## 정리하며

이 절에서는 수학 기초, PyTorch로 구현하기, 비교: FGSM과 PGD, 클래스를 살펴보았다.

**살펴볼 거리**

1. Madry, A., et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR.
2. Dong, Y., et al. (2018). "Boosting Adversarial Attacks with Momentum." CVPR.
