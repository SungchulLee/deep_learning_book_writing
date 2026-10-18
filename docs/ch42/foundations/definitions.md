# 강건성 정의

강건성을 엄밀히 뜻매김해야 방어 방법을 뜻있게 따지고 견줄 수 있다. 이 절는 핵심 자를 꼴로 적고, 적대적 취약성을 재는 수학 틀을 세우며, 공격을 짜는 데 다시 쓸 수 있는 PyTorch 밑 클래스를 준다.

---

## 1. 핵심 정의

### 점마다의 강건성

분류기 $f$이 레이블 $y$인 입력 $\mathbf{x}$에서 반지름 $\varepsilon$의 $\ell_p$ 노름 아래 다음을 채우면 **점마다 든든하다**고 한다.

$$
\forall \boldsymbol{\delta} \in \mathbb{R}^d, \quad \|\boldsymbol{\delta}\|_p \leq \varepsilon \implies f(\mathbf{x} + \boldsymbol{\delta}) = y
$$

분류기가 $\mathbf{x}$ 둘레 $\varepsilon$ 공의 모든 점을 옳게 가른다는 뜻이다.

### 가장 작은 적대적 섭동

입력 $\mathbf{x}$의 **가장 작은 적대적 섭동**은 틀리게 가르게 만드는 가장 작은 섭동이다.

$$
\varepsilon^*(\mathbf{x}) = \min \left\{ \|\boldsymbol{\delta}\|_p : f(\mathbf{x} + \boldsymbol{\delta}) \neq y \right\}
$$

이는 "결정 경계까지의 거리"를 재며, 보기 하나하나의 강건성을 재는 가장 기초가 되는 자다.

### 강건성 위험

**강건성 위험**(맞섬 위험)은 여느 분류의 위험을 가장 나쁜 자리로 넓힌 것이다.

$$
R_{\text{rob}}(f, \varepsilon) = \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}}\left[\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathbf{1}[f(\mathbf{x} + \boldsymbol{\delta}) \neq y]\right]
$$

여느 위험은 $\varepsilon = 0$인 남다른 자리다.

$$
R_{\text{std}}(f) = \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}}[\mathbf{1}[f(\mathbf{x}) \neq y]]
$$

---

## 2. 공격이 먹힌 정도를 재는 자

### 공격이 먹힌 비율(ASR)

공격이 틀리게 가르게 만든 보기의 몫이다.

$$
\text{ASR} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[f_\theta(\mathbf{x}_i + \boldsymbol{\delta}_i^*) \neq y_i]
$$

클래스 $y_{\text{target}}$을 노린 표적 공격에서는

$$
\text{ASR}_{\text{targeted}} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[f_\theta(\mathbf{x}_i + \boldsymbol{\delta}_i^*) = y_{\text{target}}]
$$

### 강건 정확도

섭동 예산 안에서 있을 수 있는 가장 센 적대적 공격 아래의 정확도다.

$$
\text{Robust Acc}(\varepsilon) = \frac{1}{N} \sum_{i=1}^N \min_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathbf{1}[f_\theta(\mathbf{x}_i + \boldsymbol{\delta}) = y_i]
$$

**ASR과의 사이:** 비표적 공격에서는 $\text{ASR} = 1 - \text{Robust Acc}$이다.

참으로는 정확한 강건 정확도를 다룰 수 없다(안쪽의 어려운 최대화를 풀어야 한다). 그래서 PGD이나 오토어택 같은 센 공격로 어림한다.

### 섭동의 크기를 재는 자

**먹힌 공격의 평균 $\ell_p$ 거리:**

$$
\bar{d}_p = \frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \|\boldsymbol{\delta}_i^*\|_p
$$

여기서 $\mathcal{S} = \{i : f_\theta(\mathbf{x}_i + \boldsymbol{\delta}_i^*) \neq y_i\}$은 먹힌 공격의 모임이다.

**보기마다 있어야 할 가장 작은 섭동:**

$$
\varepsilon^*_i = \min \{\varepsilon : \exists \boldsymbol{\delta}, \|\boldsymbol{\delta}\|_p \leq \varepsilon, f_\theta(\mathbf{x}_i + \boldsymbol{\delta}) \neq y_i\}
$$

### 인증 정확도

증명할 수 있는 다짐을 지닌 방어에서 반지름 $r$의 **인증 정확도**는 이렇다.

$$
\text{Certified Acc}(r) = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[f(\mathbf{x}_i) = y_i \text{ and } R_i \geq r]
$$

여기서 $R_i$은 보기 $i$에서 강건성 인증의 반지름이다. 이는 참 강건 정확도의 **아래끝**이다.

---

## 3. PyTorch로 짜기: 공격 밑 클래스

```python
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Literal, Dict

class AdversarialAttack(ABC):
    """
    맞서는 치기의 뼈대 클래스.
    
    치기는 모두 다음을 나눠 지닌다:
    - 타깃 모형
    - 흔듦 예산(엡실론)
    - 노름 옭아맴
    - 맞서는 보기를 만들고 따지는 방법
    
    Parameters
    ----------
    model : nn.Module
        칠 뉴럴 넷트웍
    epsilon : float
        흔듦 예산
    norm : str
        노름 옭아맴('linf', 'l2', 'l1')
    clip_min : float
        옳은 들임의 가장 작은 값
    clip_max : float
        옳은 들임의 가장 큰 값
    device : torch.device, 골라 씀
        셈할 장치
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        norm: Literal['linf', 'l2', 'l1'] = 'linf',
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.epsilon = epsilon
        self.norm = norm
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device or next(model.parameters()).device
        
        self.model.eval()
        self.model.to(self.device)
    
    @abstractmethod
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        맞서는 보기를 만든다.
        
        Parameters
        ----------
        x : torch.Tensor
            맑은 들임, 꼴 (N, C, H, W) 또는 (N, D)
        y : torch.Tensor
            참 이름표, 꼴 (N,)
        targeted : bool
            타깃 있는 치기를 할지
        target_labels : torch.Tensor, 골라 씀
            타깃 있는 치기의 타깃 이름표
            
        Returns
        -------
        x_adv : torch.Tensor
            맞서는 보기
        """
        pass
    
    def project(
        self,
        delta: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        흔듦을 엡실론 공으로 되비추고 옳은 자리로 잘라 낸다.
        """
        if self.norm == 'linf':
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        elif self.norm == 'l2':
            batch_size = delta.shape[0]
            delta_flat = delta.view(batch_size, -1)
            norms = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
            factor = torch.clamp(norms / self.epsilon, min=1.0)
            delta_flat = delta_flat / factor
            delta = delta_flat.view(delta.shape)
        elif self.norm == 'l1':
            batch_size = delta.shape[0]
            delta_flat = delta.view(batch_size, -1)
            for i in range(batch_size):
                delta_flat[i] = self._project_l1(delta_flat[i], self.epsilon)
            delta = delta_flat.view(delta.shape)
        
        # x + delta이 옳은 자리에 있도록 잘라 낸다
        delta = torch.clamp(x + delta, self.clip_min, self.clip_max) - x
        return delta
    
    @staticmethod
    def _project_l1(v: torch.Tensor, radius: float) -> torch.Tensor:
        """벡터 v를 주어진 반지름의 L1 공으로 되비춘다."""
        if torch.norm(v, p=1) <= radius:
            return v
        u = torch.abs(v)
        sorted_u, _ = torch.sort(u, descending=True)
        cumsum = torch.cumsum(sorted_u, dim=0)
        rho = torch.where(
            sorted_u > (cumsum - radius) / (torch.arange(len(u), device=v.device) + 1),
            torch.arange(len(u), device=v.device),
            torch.zeros_like(torch.arange(len(u), device=v.device))
        ).max()
        theta = (cumsum[rho] - radius) / (rho + 1)
        return torch.sign(v) * torch.clamp(torch.abs(v) - theta, min=0)
    
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_adv: torch.Tensor
    ) -> Dict[str, float]:
        """
        치기가 얼마나 잘 먹히는지 따진다.
        
        Returns
        -------
        metrics : dict
            - clean_accuracy: 맑은 보기의 맞음
            - robust_accuracy: 맞서는 보기의 맞음
            - attack_success_rate: 먹힌 치기의 몫
            - avg_perturbation: 흔듦의 평균 크기
        """
        with torch.no_grad():
            clean_pred = self.model(x.to(self.device)).argmax(dim=1)
            clean_correct = (clean_pred == y.to(self.device)).sum().item()
            
            adv_pred = self.model(x_adv.to(self.device)).argmax(dim=1)
            adv_correct = (adv_pred == y.to(self.device)).sum().item()
            
            delta = (x_adv - x).view(len(x), -1)
            
            if self.norm == 'linf':
                avg_pert = delta.abs().max(dim=1)[0].mean().item()
            elif self.norm == 'l2':
                avg_pert = torch.norm(delta, p=2, dim=1).mean().item()
            elif self.norm == 'l1':
                avg_pert = torch.norm(delta, p=1, dim=1).mean().item()
        
        n = len(y)
        return {
            'clean_accuracy': clean_correct / n,
            'robust_accuracy': adv_correct / n,
            'attack_success_rate': 1 - adv_correct / n,
            'avg_perturbation': avg_pert
        }
```

---

## 4. 강건성 굽이

**강건성 굽이**은 강건 정확도를 섭동 예산 $\varepsilon$의 함수로 그린다.

$$
\varepsilon \mapsto \text{Robust Acc}(\varepsilon)
$$

이 굽이는 늘 오르지 않으며(예산이 클수록 치는 이가 세진다) $\varepsilon \to \infty$이면 0에 다가간다. 강건성 굽이 아래 넓이는 모든 섭동 켜에 걸친 강건성을 값 하나로 간추려 준다.

```python
def compute_robustness_curve(
    attack_class,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilons: list,
    **attack_kwargs
) -> dict:
    """
    든든함 굽이를 셈한다: 엡실론에 대한 든든한 맞음.
    
    Parameters
    ----------
    attack_class : type
        치기 클래스(PGD, FGSM 따위)
    model : nn.Module
        따질 모형
    x, y : torch.Tensor
        시험 들임과 이름표
    epsilons : list[float]
        따져 볼 흔듦 예산의 목록
    
    Returns
    -------
    results : dict
        엡실론을 든든한 맞음에 맞춘 사전
    """
    results = {}
    for eps in epsilons:
        attack = attack_class(model, epsilon=eps, **attack_kwargs)
        x_adv = attack.generate(x, y)
        metrics = attack.evaluate(x, y, x_adv)
        results[eps] = metrics['robust_accuracy']
    return results
```

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

| 자 | 식 | 재는 것 |
|--------|---------|----------|
| 공격이 먹힌 비율 | $\frac{1}{N}\sum \mathbf{1}[f(\mathbf{x}_i + \boldsymbol{\delta}_i) \neq y_i]$ | 공격이 잘 먹히는 정도 |
| 강건 정확도 | $\frac{1}{N}\sum \min_\delta \mathbf{1}[f(\mathbf{x}_i + \boldsymbol{\delta}_i) = y_i]$ | 모델이 버티는 정도 |
| 인증 정확도 | 증명할 수 있는 다짐을 지닌 강건 정확도 | 가장 나쁜 자리에서 버티는 정도 |
| 가장 작은 섭동 | $\min\{\varepsilon : \exists \boldsymbol{\delta}, f(\mathbf{x}+\boldsymbol{\delta}) \neq y\}$ | 결정 경계까지의 거리 |

이 정의들이 이어지는 마디의 모든 공격과 방어 따짐의 기초가 된다.

**살펴볼 거리**

1. Madry, A., et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR.
2. Croce, F., & Hein, M. (2020). "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks." ICML.
3. Carlini, N., et al. (2019). "On Evaluating Adversarial Robustness." arXiv preprint arXiv:1902.06705.
