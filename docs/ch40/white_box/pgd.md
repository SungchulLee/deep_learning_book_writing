# 되비춘 기울기 내림(PGD) 치기

**되비춘 기울기 내림(PGD)**은 FGSM을 여러 걸음 되돌이 치기로 넓혀 훨씬 세게 만든다. 매드리 등(2018)이 내놓았으며, 맞섬에 든든함을 따지는 사실상의 잣대로 여겨지고 든든한 맞서며 익히기의 밑바탕이 된다.

---

## 1. 수학 밑바탕

### FGSM에서 되돌이 치기로

FGSM은 기울기 방향으로 큰 걸음을 한 번 밟는다. 이는 다음 까닭에 가장 좋지 않다.

1. $\mathbf{x}$에서 멀어지면 선형 어림이 나빠진다
2. 한 걸음이면 가장 좋은 자리를 지나치거나 못 미칠 수 있다
3. 잃음의 터는 볼록하지 않다

PGD은 **작은 걸음을 여러 번** 밟고 되돌 때마다 기울기를 다시 셈해 이를 푼다.

### PGD 꼴

PGD은 옭아맨 가장 좋게 하기 문제

$$
\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)
$$

를 되비춘 기울기 오름으로 푼다. 되돌 때마다의 고침은

$$
\mathbf{x}^{(t+1)} = \Pi_{\mathbf{x} + \mathcal{S}}\left( \mathbf{x}^{(t)} + \alpha \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}^{(t)}), y)) \right)
$$

여기서

- $\Pi_{\mathbf{x} + \mathcal{S}}$은 $\mathbf{x}$ 둘레 $\varepsilon$ 공으로 되비춘다
- $\alpha$은 걸음 크기
- $\mathcal{S} = \{\boldsymbol{\delta} : \|\boldsymbol{\delta}\|_p \leq \varepsilon\}$은 받아 주는 흔듦의 모임

### 알고리즘

**알고리즘: PGD 치기**

**들임:** 맑은 보기 $\mathbf{x}$, 이름표 $y$, 모형 $f_\theta$, 엡실론 $\varepsilon$, 걸음 크기 $\alpha$, 되돌이 $T$

**날임:** 맞서는 보기 $\mathbf{x}_{\text{adv}}$

1. **첫자리:** $\mathbf{u} \sim \text{Uniform}[-\varepsilon, \varepsilon]^d$일 때 $\mathbf{x}^{(0)} = \mathbf{x} + \mathbf{u}$
2. $t = 0, 1, \ldots, T-1$**마다**:
   - 기울기를 셈한다: $\mathbf{g} = \nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}^{(t)}), y)$
   - 고친다: $\tilde{\mathbf{x}}^{(t+1)} = \mathbf{x}^{(t)} + \alpha \cdot \text{sign}(\mathbf{g})$
   - 되비춘다: $\mathbf{x}^{(t+1)} = \Pi_{\varepsilon}(\tilde{\mathbf{x}}^{(t+1)}, \mathbf{x})$
   - 옳은 자리로 잘라 낸다: $\mathbf{x}^{(t+1)} = \text{clip}(\mathbf{x}^{(t+1)}, 0, 1)$
3. **돌려준다:** $\mathbf{x}_{\text{adv}} = \mathbf{x}^{(T)}$

### 되비추는 셈

**$\ell_\infty$ 되비추기:**

$$
\Pi_\varepsilon^{\infty}(\tilde{\mathbf{x}}, \mathbf{x})_i = \text{clip}(\tilde{x}_i, x_i - \varepsilon, x_i + \varepsilon)
$$

**$\ell_2$ 되비추기:**

$$
\Pi_\varepsilon^{2}(\tilde{\mathbf{x}}, \mathbf{x}) = 
\begin{cases}
\tilde{\mathbf{x}} & \text{if } \|\tilde{\mathbf{x}} - \mathbf{x}\|_2 \leq \varepsilon \\
\mathbf{x} + \varepsilon \cdot \frac{\tilde{\mathbf{x}} - \mathbf{x}}{\|\tilde{\mathbf{x}} - \mathbf{x}\|_2} & \text{otherwise}
\end{cases}
$$

### 걸음 크기 고르기

| 꾀 | 식 | 붙임말 |
|----------|---------|-------|
| **곧게** | $\alpha = \varepsilon / T$ | 조심스럽다 |
| **잣대 잡은 곧게** | $\alpha = 2\varepsilon / T$ | 여느 값(매드리 등) |
| **세게** | $\alpha = 2.5\varepsilon / T$ | 더 빨리 모인다 |

---

## 2. PyTorch로 짜기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Dict

class PGD:
    """
    되비춘 기울기 내림(PGD) 치기.
    
    Parameters
    ----------
    model : nn.Module
        칠 신경 그물
    epsilon : float
        가장 큰 흔듦의 크기
    alpha : float
        되돌 때마다의 걸음 크기
    num_iter : int
        PGD 되돌이 횟수
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
        
        # PGD 되돌이
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
            
            # 기울기 한 걸음
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

## 3. 견주기: FGSM과 PGD

| 결 | FGSM | PGD |
|--------|------|-----|
| 걸음 | 1 | $T$(10~100) |
| 첫자리 | 없음 | 아무렇게나 |
| 세기 | 여림 | 셈 |
| 빠르기 | 빠름 | $T$배 느림 |

**흔한 결과(CIFAR-10, ε=8/255):**

| 방법 | 먹힌 비율 |
|--------|--------------|
| FGSM | 약 65% |
| PGD-10 | 약 85% |
| PGD-40 | 약 92% |

---

## 4. 갈래

### 밀어 나감 되돌이 FGSM(MI-FGSM)

$$
\mathbf{g}^{(t)} = \mu \cdot \mathbf{g}^{(t-1)} + \frac{\nabla_\mathbf{x} \mathcal{L}}{\|\nabla_\mathbf{x} \mathcal{L}\|_1}
$$

기울기의 밀어 나감을 쌓아 옮아감을 낫게 한다.

### 오토-PGD(APGD)

잃음이 나아지는 정도에 따라 걸음 크기를 맞춘다. 오토어택의 고갱이 몫이다.

---

## 5. 맞서며 익히기와의 이어짐

PGD은 든든하게 익히는 데 쓰는 여느 치기다.

$$
\min_\theta \mathbb{E}_{(\mathbf{x},y)}\left[\max_{\|\boldsymbol{\delta}\|_\infty \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)\right]
$$

안쪽의 가장 크게 하기를 PGD으로 어림한다.

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

이 마당은 수학 밑바탕、PyTorch로 짜기、견주기: FGSM과 PGD、갈래을 차례로 짚었다.

**살펴볼 거리**

1. Madry, A., et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR.
2. Dong, Y., et al. (2018). "Boosting Adversarial Attacks with Momentum." CVPR.
