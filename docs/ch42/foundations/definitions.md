# 강건성 정의

강건성을 엄밀히 정의해야 방어 방법을 의미 있게 평가하고 비교할 수 있다. 이 절은 핵심 지표를 형식화하고, 적대적 취약성을 재는 수학적 틀을 세우며, 공격을 구현하는 데 다시 쓸 수 있는 PyTorch 기반 클래스를 제시한다.

---

## 1. 핵심 정의

### 점별 강건성

분류기 $f$가 레이블 $y$인 입력 $\mathbf{x}$에서 반지름 $\varepsilon$의 $\ell_p$ 노름 아래 다음을 만족하면 **점별로 강건하다**(pointwise robust)고 한다.

$$
\forall \boldsymbol{\delta} \in \mathbb{R}^d, \quad \|\boldsymbol{\delta}\|_p \leq \varepsilon \implies f(\mathbf{x} + \boldsymbol{\delta}) = y
$$

분류기가 $\mathbf{x}$ 둘레 $\varepsilon$ 공의 모든 점을 올바르게 분류한다는 뜻이다.

### 가장 작은 적대적 섭동

입력 $\mathbf{x}$의 **가장 작은 적대적 섭동**은 오분류하게 만드는 가장 작은 섭동이다.

$$
\varepsilon^*(\mathbf{x}) = \min \left\{ \|\boldsymbol{\delta}\|_p : f(\mathbf{x} + \boldsymbol{\delta}) \neq y \right\}
$$

이는 "결정 경계까지의 거리"를 재며, 샘플 하나하나의 강건성을 재는 가장 기본적인 척도다.

### 강건성 위험

**강건성 위험**(적대적 위험)은 표준 분류 위험을 최악의 경우로 확장한 것이다.

$$
R_{\text{rob}}(f, \varepsilon) = \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}}\left[\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathbf{1}[f(\mathbf{x} + \boldsymbol{\delta}) \neq y]\right]
$$

표준 위험은 $\varepsilon = 0$인 특수한 경우다.

$$
R_{\text{std}}(f) = \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}}[\mathbf{1}[f(\mathbf{x}) \neq y]]
$$

---

## 2. 공격 성공을 재는 척도

### 공격 성공률(ASR)

공격이 오분류하게 만든 샘플의 비율이다.

$$
\text{ASR} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[f_\theta(\mathbf{x}_i + \boldsymbol{\delta}_i^*) \neq y_i]
$$

클래스 $y_{\text{target}}$을 노린 표적 공격에서는

$$
\text{ASR}_{\text{targeted}} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[f_\theta(\mathbf{x}_i + \boldsymbol{\delta}_i^*) = y_{\text{target}}]
$$

### 강건 정확도

섭동 예산 안에서 가능한 가장 센 적대적 공격 아래의 정확도다.

$$
\text{Robust Acc}(\varepsilon) = \frac{1}{N} \sum_{i=1}^N \min_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathbf{1}[f_\theta(\mathbf{x}_i + \boldsymbol{\delta}) = y_i]
$$

**ASR과의 관계:** 비표적 공격에서 모든 샘플을 분모로 두면 $\text{ASR} = 1 - \text{Robust Acc}$이다.

실제로는 정확한 강건 정확도를 계산할 수 없다(안쪽의 어려운 최대화 문제를 풀어야 한다). 그래서 PGD나 AutoAttack 같은 센 공격으로 근사한다.

### 섭동의 크기를 재는 척도

**성공한 공격의 평균 $\ell_p$ 거리:**

$$
\bar{d}_p = \frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \|\boldsymbol{\delta}_i^*\|_p
$$

여기서 $\mathcal{S} = \{i : f_\theta(\mathbf{x}_i + \boldsymbol{\delta}_i^*) \neq y_i\}$은 성공한 공격의 집합이다.

**샘플마다 필요한 가장 작은 섭동:**

$$
\varepsilon^*_i = \min \{\varepsilon : \exists \boldsymbol{\delta}, \|\boldsymbol{\delta}\|_p \leq \varepsilon, f_\theta(\mathbf{x}_i + \boldsymbol{\delta}) \neq y_i\}
$$

### 인증 정확도

증명할 수 있는 보장을 지닌 방어에서 반지름 $r$의 **인증 정확도**는 이렇다.

$$
\text{Certified Acc}(r) = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[f(\mathbf{x}_i) = y_i \text{ and } R_i \geq r]
$$

여기서 $R_i$는 샘플 $i$에서 인증된 강건 반지름이다. 이는 참 강건 정확도의 **하한**이다.

---

## 3. PyTorch로 구현하기: 공격 기반 클래스

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
    - 섭동 예산(엡실론)
    - 노름 옭아맴
    - 맞서는 보기를 만들고 평가하는 방법
    
    Parameters
    ----------
    model : nn.Module
        칠 뉴럴 네트워크
    epsilon : float
        섭동 예산
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
            맑은 들임, 형태 (N, C, H, W) 또는 (N, D)
        y : torch.Tensor
            참 이름표, 형태 (N,)
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
        섭동을 엡실론 공으로 되비추고 옳은 자리로 잘라 낸다.
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
            - clean_accuracy: 맑은 보기의 정확도
            - robust_accuracy: 맞서는 보기의 정확도
            - attack_success_rate: 먹힌 치기의 기여도
            - avg_perturbation: 섭동의 평균 크기
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

## 4. 강건성 곡선

**강건성 곡선**은 강건 정확도를 섭동 예산 $\varepsilon$의 함수로 그린다.

$$
\varepsilon \mapsto \text{Robust Acc}(\varepsilon)
$$

이 곡선은 단조 비증가하며(예산이 클수록 공격자가 세진다) $\varepsilon \to \infty$이면 0에 다가간다. 강건성 곡선 아래 넓이는 모든 섭동 크기에 걸친 강건성을 값 하나로 요약해 준다.

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
    강건성 굽이를 셈한다: 엡실론에 대한 강건한 정확도.
    
    Parameters
    ----------
    attack_class : type
        치기 클래스(PGD, FGSM 따위)
    model : nn.Module
        따질 모형
    x, y : torch.Tensor
        시험 들임과 이름표
    epsilons : list[float]
        따져 볼 섭동 예산의 목록
    
    Returns
    -------
    results : dict
        엡실론을 강건한 정확도에 맞춘 사전
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
    일반 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 절의 방법을 적용한 결과는 기법에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 바운드를 준다. 정확도와 강건성의 트레이드오프는 근본적인 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 떨어진다. 무작위 시드 3개의 평균과 표준편차로 보고하여라. $\square$

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

| 척도 | 식 | 재는 것 |
|--------|---------|----------|
| 공격 성공률 | $\frac{1}{N}\sum \mathbf{1}[f(\mathbf{x}_i + \boldsymbol{\delta}_i) \neq y_i]$ | 공격이 성공하는 정도 |
| 강건 정확도 | $\frac{1}{N}\sum \min_\delta \mathbf{1}[f(\mathbf{x}_i + \boldsymbol{\delta}_i) = y_i]$ | 모델이 버티는 정도 |
| 인증 정확도 | 증명할 수 있는 보장을 지닌 강건 정확도 | 최악의 경우에도 버티는 정도 |
| 가장 작은 섭동 | $\min\{\varepsilon : \exists \boldsymbol{\delta}, f(\mathbf{x}+\boldsymbol{\delta}) \neq y\}$ | 결정 경계까지의 거리 |

이 정의들이 이어지는 절의 모든 공격과 방어 평가의 기초가 된다.

**살펴볼 거리**

1. Madry, A., et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR.
2. Croce, F., & Hein, M. (2020). "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks." ICML.
3. Carlini, N., et al. (2019). "On Evaluating Adversarial Robustness." arXiv preprint arXiv:1902.06705.
