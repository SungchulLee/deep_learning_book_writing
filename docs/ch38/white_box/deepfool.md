# 딥풀
## 들머리

**딥풀**(무사비-데즈풀리 등, 2016)은 들임을 가장 가까운 판단의 금 너머로 옮기는 **가장 작은 흔듦**을 찾는 꼴 바탕의 맞서는 치기다. 미리 정한 $\varepsilon$을 쓰는 예산 붙박이 치기(FGSM, PGD)와 달리, 딥풀은 틀리게 가르게 만드는 가장 작은 흔듦을 거듭 셈해 모형의 그 자리 판단 꼴을 들여다보게 해 준다.

## 수학 밑바탕

### 둘 가름개일 때

둘 가름 아핀 가름개 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$을 보자. 판단의 금은 넘평면 $\{\mathbf{x} : f(\mathbf{x}) = 0\}$이다. 이 금을 넘는 가장 작은 흔듦은 그 넘평면으로의 곧은 되비춤이다.

$$
\boldsymbol{\delta}^* = -\frac{f(\mathbf{x})}{\|\mathbf{w}\|_2^2} \mathbf{w}
$$

그 크기는

$$
\|\boldsymbol{\delta}^*\|_2 = \frac{|f(\mathbf{x})|}{\|\mathbf{w}\|_2}
$$

이는 곧 $\mathbf{x}$에서 판단 넘평면까지의 거리다.

### 신경 그물로 넓히기

곧지 않은 가름개에서 딥풀은 이제의 점에서 판단의 금을 곧게 펴고 그 금으로 거듭 되비춘다. 되돌이 $t$마다

1. $\mathbf{x}^{(t)}$ 둘레에서 가름개를 그 자리 아핀으로 어림한다
2. 어림한 금을 넘는 가장 작은 흔듦을 셈한다
3. 고친다: $\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} + \boldsymbol{\delta}^{(t)}$
4. 틀리게 갈릴 때까지 되풀이한다

### 여러 갈래로 넓히기

로짓이 $Z_k(\mathbf{x})$인 여러 갈래 가름개에서 갈래 $y$(참)과 갈래 $k$ 사이의 판단의 금은

$$
\{\mathbf{x} : Z_y(\mathbf{x}) = Z_k(\mathbf{x})\}
$$

되돌 때마다 딥풀은 다음을 한다.

1. 갈래마다의 금까지 곧게 편 거리를 셈한다.

$$
d_k = \frac{|Z_k(\mathbf{x}^{(t)}) - Z_y(\mathbf{x}^{(t)})|}{\|\nabla_\mathbf{x} Z_k(\mathbf{x}^{(t)}) - \nabla_\mathbf{x} Z_y(\mathbf{x}^{(t)})\|_2}
$$

2. 가장 가까운 금을 고른다: $\hat{k} = \arg\min_{k \neq y} d_k$

3. 그 금 쪽으로의 흔듦을 셈한다.

$$
\boldsymbol{\delta}^{(t)} = \frac{Z_{\hat{k}}(\mathbf{x}^{(t)}) - Z_y(\mathbf{x}^{(t)})}{\|\mathbf{w}_{\hat{k}} - \mathbf{w}_y\|_2^2} (\mathbf{w}_{\hat{k}} - \mathbf{w}_y)
$$

여기서 $\mathbf{w}_k = \nabla_\mathbf{x} Z_k(\mathbf{x}^{(t)})$이다.

## 알고리즘

**알고리즘: 딥풀**

**들임:** 들임 $\mathbf{x}$, 가름개 $f$, 가장 많은 되돌이 $T$, 지나침 $\eta$

**날임:** 가장 작은 맞서는 흔듦 $\hat{\boldsymbol{\delta}}$

1. 첫자리: $\mathbf{x}^{(0)} = \mathbf{x}$, $\hat{\boldsymbol{\delta}} = \mathbf{0}$
2. $f(\mathbf{x}^{(t)}) = f(\mathbf{x})$이고 $t < T$인 동안:
    - 갈래 $k \neq y$마다 $\mathbf{w}_k' = \nabla_\mathbf{x} Z_k - \nabla_\mathbf{x} Z_y$과 $f_k' = Z_k - Z_y$을 셈한다
    - 가장 가까운 금을 찾는다: $\hat{k} = \arg\min_{k \neq y} \frac{|f_k'|}{\|\mathbf{w}_k'\|_2}$
    - 걸음을 셈한다: $\boldsymbol{\delta}^{(t)} = \frac{|f_{\hat{k}}'|}{\|\mathbf{w}_{\hat{k}}'\|_2^2} \mathbf{w}_{\hat{k}}'$
    - 고친다: $\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} + (1 + \eta) \boldsymbol{\delta}^{(t)}$
    - 쌓는다: $\hat{\boldsymbol{\delta}} \leftarrow \hat{\boldsymbol{\delta}} + (1 + \eta) \boldsymbol{\delta}^{(t)}$
3. $\hat{\boldsymbol{\delta}}$을 돌려준다

지나침 매개변수 $\eta > 0$(흔히 0.02)은 흔듦이 금 위에 딱 걸치지 않고 넘어가도록 한다.

## PyTorch으로 짜기

```python
import torch
import torch.nn as nn
from typing import Optional, Dict

class DeepFool:
    """
    딥풀 치기: 틀리게 가르게 하는 가장 작은 L2 흔듦을 찾는다.
    
    판단의 금을 거듭 곧게 펴고 가장 가까운 갈래의 금으로
    되비춘다.
    
    Parameters
    ----------
    model : nn.Module
        칠 신경 그물
    num_classes : int
        날임 갈래의 수
    max_iter : int
        가장 많은 되돌이
    overshoot : float
        금을 반드시 넘게 하는 지나침 매개변수
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
        
        (흔듦, 되돌이 수, 본디 이름표, 마지막 이름표)을 돌려준다
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
            
            # 갈래마다 기울기를 셈한다
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
                
                # 본디 갈래의 기울기도 있어야 한다
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
        묶음 하나에 대해 맞서는 보기를 만든다.
        
        Parameters
        ----------
        x : torch.Tensor
            맑은 들임, 꼴 (N, C, H, W)
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
        """흔듦의 자와 함께 치기를 따진다."""
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

## 결과 견주기

### 센 데

- **가장 작은 흔듦을 찾는다**: $\varepsilon$이 붙박인 치기와 달리 판단의 금까지의 참 거리를 드러낸다
- **꼴로 보는 깨침**: 판단의 금이 얼마나 가까운지를 곧바로 잰다
- **하이퍼파라미터를 맞출 것 없음**: 고를 $\varepsilon$이 없다(지나침과 가장 많은 되돌이만 빼면)
- **이론에 뿌리내림**: 아핀 가름개에서는 가장 좋다

### 한계

- **느리다**: 되돌 때마다 갈래마다의 기울기를 셈해야 한다
- **L2만**: 본디 꼴이 $\ell_2$ 노름에 맞다. $\ell_\infty$ 갈래(두루 쓰는 딥풀)도 있으나 덜 깔끔하다
- **가장 센 치기는 아니다**: 예산이 붙박인 따짐에는 PGD과 C&W이 낫다
- **차례로 해야 한다**: 보기 하나하나를 따로 다뤄야 한다

### 다른 치기와 견주기

| 결 | 딥풀 | PGD | C&W |
|--------|----------|-----|-----|
| 가장 작은 $\|\boldsymbol{\delta}\|$을 찾음 | 그렇다 | 아니다 | 그렇다 |
| $\varepsilon$ 붙박인 따짐 | 아니다 | 그렇다 | 아니다 |
| 빠르기 | 느림 | 가운데 | 아주 느림 |
| 으뜸 노름 | $\ell_2$ | $\ell_\infty$ 또는 $\ell_2$ | $\ell_2$ |
| 쓸 자리 | 꼴 살피기 | 든든함 따지기 | 막이 깨기 |

## 쓸 자리

딥풀은 다음에 더욱 값지다.

- **든든함의 여유 재기**: 자료 꾸러미에 걸친 가장 작은 흔듦의 평균은 여느 들임이 판단의 금에 얼마나 가까운지를 수로 알려 준다
- **얼개 견주기**: 어느 모형의 여유가 더 넓은지 드러낸다
- **판단 꼴 알아보기**: 가장 작은 흔듦의 방향이 가장 무른 들임 차수를 알려 준다
- **금융에 쓰기**: 들임을 아주 조금 흔들었을 때 모형의 미쁨 판단이나 거래 신호가 얼마나 바뀌는지 재기

## 살펴볼 거리

1. Moosavi-Dezfooli, S. M., Fawzi, A., & Frossard, P. (2016). "DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks." CVPR.
2. Moosavi-Dezfooli, S. M., et al. (2017). "Universal Adversarial Perturbations." CVPR.

## 익힘 문제

**익힘 1.**
선형 가름개 $f(x) = w^T x + b$에서 미루어 본 갈래를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 흔듦을 셈하여라. 이것이 신경 그물의 든든함과 어떻게 이어지는지 밝혀라.

??? success "익힘 1 풀이"
    선형 가름개에서 $\ell_\infty$ 노름으로 잰 판단의 금까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 흔듦은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 신경 그물에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(기울기의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차수가 높은 모형이 무른 까닭은 $\|w\|_1$은 차수와 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 든든함의 여유가 줄어들기 때문이다. $\square$

---

**익힘 2.**
이 마디에서 다룬 치기나 막이를 CIFAR-10의 ResNet-18 모형에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 치기 아래에서 맑은 맞음과 든든한 맞음을 알려라.

??? success "익힘 2 풀이"
    여느 ResNet-18은 맑은 맞음이 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 걸음 크기 $2/255$) 아래의 든든한 맞음은 $\sim$0%이다. 이 마디의 방법을 걸면 결과는 재주에 따라 다르다. 맞서며 익히기는 맑은 맞음 $\sim$83%에 든든한 맞음 $\sim$50%이고, 밝혀 낸 막이는 더 낮지만 증명할 수 있는 테두리를 준다. 맞음과 든든함의 맞바꿈은 밑바탕부터 있는 것이라, 든든함을 높이면 맑은 맞음이 흔히 5~15% 든다. 아무렇게나 하는 씨앗 3개의 평균과 잣대 어긋남으로 알려라. $\square$

---

**익힘 3.**
흔듦 공 안에서 갈래별 자료의 밑자리가 서로 겹친다고 볼 때, 모형이 담는 힘을 키우지 않고서는 어떤 막이도 맑은 자료의 높은 맞음과 $\ell_\infty$ 흔듦에 대한 높은 든든함을 함께 이룰 수 없음을 증명하여라.

??? success "익힘 3 풀이"
    두 갈래의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{갈래 1}, x_2 \in \text{갈래 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 든든한 가름개는 적어도 하나를 틀리게 가를 수밖에 없다(흔듦 공이 겹치기 때문이다). 이것이 맞음과 든든함의 밑바탕 맞바꿈이다. 겹치는 밑자리의 몫이 피할 수 없는 맞음 잃음을 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 맞음 떨어짐을 밝혀 준다. 모형이 담는 힘을 키우면(더 너른 그물) 얽힌 든든한 판단의 금을 더 잘 그려 맞바꿈을 얼마쯤 눅일 수 있다. $\square$

---

**익힘 4.**
금융 기계 배움 얼개(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 든든함이 어떻게 드러나는지 다루어라. 으름 얼개가 보기 다룸과 어떻게 다른가?

??? success "익힘 4 풀이"
    금융에서 겨루는 이는 알아내는 얼개에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 고갱이는 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 치기가 잇따르며 맞추어 간다(겨루는 이가 얼개의 되받음을 보고 손본다). (3) 헛 맞음과 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 흔듦 모형이 있어야 한다. 막이는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 바탕의 길은 많이 걸러진다. $\square$
