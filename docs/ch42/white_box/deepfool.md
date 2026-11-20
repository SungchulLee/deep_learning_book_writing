# DeepFool

**DeepFool**(Moosavi-Dezfooli 등, 2016)은 입력을 가장 가까운 결정 경계 너머로 옮기는 **가장 작은 섭동**을 찾는 기하 기반의 적대적 공격이다. 미리 정한 $\varepsilon$을 쓰는 예산 고정 공격(FGSM, PGD)과 달리, DeepFool은 오분류하게 만드는 가장 작은 섭동을 반복해서 계산하므로 입력 근처에서 결정 경계가 어떻게 생겼는지를 들여다보게 해 준다.

---

## 1. 수학 기초

### 이진 분류기일 때

이진 아핀 분류기 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$를 보자. 결정 경계는 초평면 $\{\mathbf{x} : f(\mathbf{x}) = 0\}$이다. 이 경계에 닿는 가장 작은 섭동은 $\mathbf{x}$를 그 초평면에 수직으로 투영하는 것이다.

$$
\boldsymbol{\delta}^* = -\frac{f(\mathbf{x})}{\|\mathbf{w}\|_2^2} \mathbf{w}
$$

그 크기는

$$
\|\boldsymbol{\delta}^*\|_2 = \frac{|f(\mathbf{x})|}{\|\mathbf{w}\|_2}
$$

이는 곧 $\mathbf{x}$에서 결정 초평면까지의 거리다.

### 뉴럴 네트워크로 넓히기

비선형 분류기에서 DeepFool은 현재 점에서 결정 경계를 선형화하고 그 경계로 투영하기를 되풀이한다. 반복 $t$마다

1. $\mathbf{x}^{(t)}$ 둘레에서 분류기를 국소 아핀 함수로 근사한다
2. 근사한 경계를 넘는 가장 작은 섭동을 계산한다
3. 갱신한다: $\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} + \boldsymbol{\delta}^{(t)}$
4. 오분류될 때까지 되풀이한다

### 여러 클래스로 넓히기

로짓이 $Z_k(\mathbf{x})$인 다중 클래스 분류기에서 클래스 $y$(참)와 클래스 $k$ 사이의 결정 경계는

$$
\{\mathbf{x} : Z_y(\mathbf{x}) = Z_k(\mathbf{x})\}
$$

반복마다 DeepFool은 다음을 한다.

1. 클래스마다 그 경계까지의 거리를 선형 근사로 계산한다.

$$
d_k = \frac{|Z_k(\mathbf{x}^{(t)}) - Z_y(\mathbf{x}^{(t)})|}{\|\nabla_\mathbf{x} Z_k(\mathbf{x}^{(t)}) - \nabla_\mathbf{x} Z_y(\mathbf{x}^{(t)})\|_2}
$$

2. 가장 가까운 경계를 고른다: $\hat{k} = \arg\min_{k \neq y} d_k$

3. 그 경계 쪽으로의 섭동을 계산한다.

$$
\boldsymbol{\delta}^{(t)} = \frac{\left|Z_{\hat{k}}(\mathbf{x}^{(t)}) - Z_y(\mathbf{x}^{(t)})\right|}{\|\mathbf{w}_{\hat{k}} - \mathbf{w}_y\|_2^2} (\mathbf{w}_{\hat{k}} - \mathbf{w}_y)
$$

여기서 $\mathbf{w}_k = \nabla_\mathbf{x} Z_k(\mathbf{x}^{(t)})$이다.

---

## 2. 알고리즘

**알고리즘: DeepFool**

**입력:** 입력 $\mathbf{x}$, 분류기 $f$, 최대 반복 $T$, 오버슈트 $\eta$

**출력:** 가장 작은 적대적 섭동 $\hat{\boldsymbol{\delta}}$

1. 초기화: $\mathbf{x}^{(0)} = \mathbf{x}$, $\hat{\boldsymbol{\delta}} = \mathbf{0}$
2. $f(\mathbf{x}^{(t)}) = f(\mathbf{x})$이고 $t < T$인 동안:
    - 클래스 $k \neq y$마다 $\mathbf{w}_k' = \nabla_\mathbf{x} Z_k - \nabla_\mathbf{x} Z_y$와 $f_k' = Z_k - Z_y$를 계산한다
    - 가장 가까운 경계를 찾는다: $\hat{k} = \arg\min_{k \neq y} \frac{|f_k'|}{\|\mathbf{w}_k'\|_2}$
    - 스텝을 계산한다: $\boldsymbol{\delta}^{(t)} = \frac{|f_{\hat{k}}'|}{\|\mathbf{w}_{\hat{k}}'\|_2^2} \mathbf{w}_{\hat{k}}'$
    - 갱신한다: $\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} + (1 + \eta) \boldsymbol{\delta}^{(t)}$
    - 누적한다: $\hat{\boldsymbol{\delta}} \leftarrow \hat{\boldsymbol{\delta}} + (1 + \eta) \boldsymbol{\delta}^{(t)}$
3. $\hat{\boldsymbol{\delta}}$를 돌려준다

오버슈트(overshoot) 파라메터 $\eta > 0$(흔히 0.02)은 선형 근사로 구한 스텝을 조금 더 늘려, 섭동이 경계 위에 딱 걸치지 않고 넘어가도록 한다.

---

## 3. PyTorch로 구현하기

```python
import torch
import torch.nn as nn
from typing import Optional, Dict

class DeepFool:
    """
    딥풀 치기: 틀리게 가르게 하는 가장 작은 L2 섭동을 찾는다.
    
    판단의 금을 거듭 곧게 펴고 가장 가까운 클래스의 금으로
    되비춘다.
    
    Parameters
    ----------
    model : nn.Module
        칠 뉴럴 네트워크
    num_classes : int
        날임 클래스의 수
    max_iter : int
        가장 많은 회귀
    overshoot : float
        금을 반드시 넘게 하는 지나침 파라메터
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 10,
        max_iter: int = 50,
        overshoot: float = 0.02,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.overshoot = overshoot
        self.device = device or next(model.parameters()).device
        
        self.model.eval()
        self.model.to(self.device)
    
    def _attack_single(self, x: torch.Tensor) -> tuple:
        """
        들임 하나를 친다.
        
        (섭동, 회귀 수, 본디 이름표, 마지막 이름표)을 돌려준다
        """
        x = x.unsqueeze(0).to(self.device).clone().detach()
        x_orig = x.clone()
        
        # 본디 미루어 봄을 얻는다
        with torch.no_grad():
            logits = self.model(x)
            orig_label = logits.argmax(dim=1).item()
        
        x_pert = x.clone()
        total_pert = torch.zeros_like(x)
        
        for iteration in range(self.max_iter):
            x_pert.requires_grad_(True)
            logits = self.model(x_pert)
            pred = logits.argmax(dim=1).item()
            
            if pred != orig_label:
                break
            
            # 클래스마다 그래디언트를 셈한다
            grads = []
            for k in range(self.num_classes):
                if k == orig_label:
                    grads.append(None)
                    continue
                
                self.model.zero_grad()
                if x_pert.grad is not None:
                    x_pert.grad.zero_()
                
                logits[0, k].backward(retain_graph=True)
                grad_k = x_pert.grad.data.clone()
                
                # 본디 클래스의 그래디언트도 있어야 한다
                self.model.zero_grad()
                x_pert.grad.zero_()
                logits[0, orig_label].backward(retain_graph=True)
                grad_orig = x_pert.grad.data.clone()
                
                grads.append(grad_k - grad_orig)
            
            # 가장 가까운 금을 찾는다
            min_dist = float('inf')
            best_delta = None
            
            for k in range(self.num_classes):
                if k == orig_label:
                    continue
                
                w_k = grads[k]
                f_k = (logits[0, k] - logits[0, orig_label]).item()
                
                w_norm = w_k.view(-1).norm(p=2).item()
                if w_norm < 1e-8:
                    continue
                
                dist = abs(f_k) / w_norm
                
                if dist < min_dist:
                    min_dist = dist
                    best_delta = (abs(f_k) / (w_norm ** 2)) * w_k
            
            if best_delta is None:
                break
            
            # 지나침을 곁들여 고친다
            step = (1 + self.overshoot) * best_delta
            total_pert += step
            x_pert = (x_orig + total_pert).detach()
            x_pert = torch.clamp(x_pert, 0, 1)
        
        final_pert = (x_pert - x_orig).squeeze(0)
        with torch.no_grad():
            final_label = self.model(x_pert).argmax(dim=1).item()
        
        return final_pert, iteration + 1, orig_label, final_label
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        배치 하나에 대해 맞서는 보기를 만든다.
        
        Parameters
        ----------
        x : torch.Tensor
            맑은 들임, 형태 (N, C, H, W)
        y : torch.Tensor
            참 이름표(따질 때만 쓴다)
            
        Returns
        -------
        x_adv : torch.Tensor
            맞서는 보기
        """
        x_adv = x.clone()
        
        for i in range(len(x)):
            pert, iters, orig, final = self._attack_single(x[i])
            x_adv[i] = torch.clamp(x[i].to(self.device) + pert, 0, 1)
        
        return x_adv.detach()
    
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_adv: torch.Tensor
    ) -> Dict[str, float]:
        """섭동의 자와 함께 치기를 따진다."""
        with torch.no_grad():
            x_dev = x.to(self.device)
            y_dev = y.to(self.device)
            x_adv_dev = x_adv.to(self.device)
            
            clean_pred = self.model(x_dev).argmax(1)
            adv_pred = self.model(x_adv_dev).argmax(1)
            
            clean_acc = (clean_pred == y_dev).float().mean().item()
            robust_acc = (adv_pred == y_dev).float().mean().item()
            
            delta = (x_adv_dev - x_dev).view(len(x), -1)
            l2_norms = delta.norm(p=2, dim=1)
        
        return {
            'clean_accuracy': clean_acc,
            'robust_accuracy': robust_acc,
            'attack_success_rate': 1 - robust_acc,
            'avg_l2': l2_norms.mean().item(),
            'median_l2': l2_norms.median().item(),
            'min_l2': l2_norms.min().item(),
            'max_l2': l2_norms.max().item()
        }
```

---

## 4. 특징과 비교

### 장점

- **가장 작은 섭동을 찾는다**: $\varepsilon$이 고정된 공격과 달리 결정 경계까지의 실제 거리를 드러낸다
- **기하학적 통찰**: 결정 경계가 얼마나 가까운지를 곧바로 잰다
- **맞출 하이퍼파라미터가 거의 없다**: 고를 $\varepsilon$이 없다(오버슈트와 최대 반복만 빼면)
- **이론적 근거**: 아핀 분류기에서는 정확히 최적이다

### 한계

- **느리다**: 반복마다 클래스마다의 그래디언트를 계산해야 한다
- **L2 위주**: 원래 형태가 $\ell_2$ 노름에 맞다. $\ell_\infty$ 변형도 있으나 덜 깔끔하다
- **가장 센 공격은 아니다**: 예산이 고정된 평가에는 PGD와 C&W가 낫다
- **배치 처리가 어렵다**: 이 구현은 샘플을 하나씩 따로 다룬다

### 다른 공격과 비교

| 항목 | DeepFool | PGD | C&W |
|--------|----------|-----|-----|
| 가장 작은 $\|\boldsymbol{\delta}\|$을 찾음 | 그렇다 | 아니다 | 그렇다 |
| $\varepsilon$ 고정된 평가 | 아니다 | 그렇다 | 아니다 |
| 속도 | 느림 | 중간 | 아주 느림 |
| 주 노름 | $\ell_2$ | $\ell_\infty$ 또는 $\ell_2$ | $\ell_2$ |
| 용도 | 기하 분석 | 강건성 평가 | 방어 깨기 |

---

## 5. 쓸 자리

DeepFool은 다음에 특히 쓸모 있다.

- **강건성의 여유 재기**: 데이터셋에 걸친 가장 작은 섭동의 평균은 전형적인 입력이 결정 경계에 얼마나 가까운지를 수로 알려 준다
- **구조 비교**: 어느 모델의 여유가 더 넓은지 드러낸다
- **결정 경계의 기하 이해**: 가장 작은 섭동의 방향이 가장 취약한 입력 방향을 알려 준다
- **금융 응용**: 입력을 아주 조금 흔들었을 때 모델의 신뢰 판단이나 거래 신호가 얼마나 바뀌는지 재기

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

이 절에서는 DeepFool의 수학 기초와 알고리즘, PyTorch 구현을 보고, 그 특징을 PGD·C&W와 비교했다.

**살펴볼 거리**

1. Moosavi-Dezfooli, S. M., Fawzi, A., & Frossard, P. (2016). "DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks." CVPR.
2. Moosavi-Dezfooli, S. M., et al. (2017). "Universal Adversarial Perturbations." CVPR.
