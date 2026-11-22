# MART: 오분류를 고려한 강건 트레인

**MART**(Misclassification Aware adveRsarial Training, 오분류를 고려한 적대적 트레인. 왕 등, 2020)는 올바르게 분류하기가 얼마나 어려운지에 따라 트레인 샘플마다 다른 가중치를 주어 적대적 트레인을 개선한다. 핵심 통찰은 샘플마다 강건성에 기여하는 정도가 다르다는 것이다. 원래 분류하기 어려운 샘플이 강건 트레인 동안 더 많은 주의를 받아야 한다.

---

## 1. 왜 하는가

표준 적대적 트레인과 TRADES는 모든 샘플을 똑같이 다룬다. 그런데

- **쉬운 샘플**(클린 확신도가 높음, $p_y(\mathbf{x}) \approx 1$): 이미 잘 분류된다. 결정 경계까지 여유가 커서 잘 속지 않는다
- **어려운 샘플**(클린 확신도가 낮음, $p_y(\mathbf{x}) \approx 0$): 결정 경계 가까이 있어 적대적 섭동에 쉽게 오분류된다

MART는 방어 역량을 가장 필요한 어려운 샘플에 모은다.

---

## 2. 수학 형태

### MART 로스

$$
\mathcal{L}_{\text{MART}} = \text{BCE}(f_\theta(\mathbf{x}_{\text{adv}}), y) + \lambda \cdot (1 - p_y(\mathbf{x})) \cdot \text{KL}(f_\theta(\mathbf{x}) \| f_\theta(\mathbf{x}_{\text{adv}}))
$$

여기서

- $\text{BCE}$는 적대적 샘플에 대한 부스티드 크로스 엔트로피(boosted cross entropy)다. 크로스 엔트로피에, 정답을 뺀 클래스 가운데 확률이 가장 큰 클래스의 확률을 낮추는 항을 더한 것이다(아래 코드의 `loss_adv`)
- $p_y(\mathbf{x}) = \text{softmax}(f_\theta(\mathbf{x}))_y$는 클린 입력에서 참 클래스의 확률
- $(1 - p_y(\mathbf{x}))$는 **오분류를 고려한 가중치**
- $\lambda$는 정칙화 계수

### 가중치의 해석

가중치 $(1 - p_y(\mathbf{x}))$는 어려운 샘플에 주의를 모으는 장치 노릇을 한다.

| $p_y(\mathbf{x})$ | 가중치 | 해석 |
|-------------------|--------|----------------|
| 약 1.0(확신하는 정답) | 약 0 | 정칙화가 약하다. 이 샘플에서는 모델이 이미 강건하다 |
| 약 0.5(불확실) | 약 0.5 | 정칙화가 중간이다 |
| 약 0(오분류) | 약 1.0 | 정칙화가 강하다. 가장 취약한 샘플이다 |

---

## 3. PyTorch로 구현하기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict
from tqdm import tqdm

class MARTTrainer:
    """
    MART: 잘못 가름을 아는 강건한 익히기.
    
    한결같음 로스에 (1 - p_y(x)) 짐을 주어
    맞서며 익히기의 힘을 어려운 보기에 모은다.
    
    Parameters
    ----------
    model : nn.Module
        익힐 모형
    lam : float
        KL 항의 다독임 셈
    epsilon : float
        섭동 예산
    alpha : float
        PGD 스텝 크기
    num_iter : int
        PGD 회귀
    """
    
    def __init__(
        self,
        model: nn.Module,
        lam: float = 6.0,
        epsilon: float = 8/255,
        alpha: float = 2/255,
        num_iter: int = 10,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.lam = lam
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
    
    def _pgd_attack(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """PGD 맞서는 보기를 만든다."""
        x_adv = x + torch.empty_like(x).uniform_(
            -self.epsilon, self.epsilon
        )
        x_adv = torch.clamp(x_adv, 0, 1)
        
        for _ in range(self.num_iter):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(self.model(x_adv), y)
            self.model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv.detach()
    
    def _mart_loss(
        self,
        logits_adv: torch.Tensor,
        logits_clean: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        MART 로스를 셈한다.
        
        BCE(적대적 샘플, y) + λ · (1 - p_y(x)) · KL(원본 || 적대적 샘플)
        """
        # 맞서는 보기의 북돋운 엇갈린 엔트로피
        adv_probs = F.softmax(logits_adv, dim=1)
        # BCE 형태: 기준 잡은 -log(p_y(x_adv))
        tmp = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(
            tmp[:, -1] == y, tmp[:, -2], tmp[:, -1]
        )
        loss_adv = F.cross_entropy(logits_adv, y) + \
                   F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
        
        # 잘못 가름을 아는 짐
        with torch.no_grad():
            p_clean = F.softmax(logits_clean, dim=1)
            p_y = p_clean.gather(1, y.view(-1, 1)).squeeze()
            weight = 1.0 - p_y  # 틀리게 갈린 것에 큰 짐
        
        # 짐을 준 KL 발산
        kl = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_clean.detach(), dim=1),
            reduction='none'
        ).sum(dim=1)
        
        loss_kl = (weight * kl).mean()
        
        return loss_adv + self.lam * loss_kl
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """MART 로스로 한 판 익힌다."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='MART Training')
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            # 맑은 앞으로 스텝
            logits_clean = self.model(x)
            
            # 맞서는 보기를 만든다
            x_adv = self._pgd_attack(x, y)
            
            # 맞서는 앞으로 스텝
            logits_adv = self.model(x_adv)
            
            # MART 로스
            loss = self._mart_loss(logits_adv, logits_clean, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            correct += (logits_clean.argmax(1) == y).sum().item()
            total += len(y)
            
            pbar.set_postfix({'loss': f'{total_loss/total:.4f}'})
        
        return {
            'loss': total_loss / total,
            'clean_accuracy': correct / total
        }
```

---

## 4. 적대적 트레인 방법 비교

| 방법 | 핵심 통찰 | 클린 정확도 | 강건 정확도 | 하이퍼파라미터 |
|--------|----------|-----------|------------|----------------|
| 표준 적대적 트레인 | 적대적 샘플에서의 최대 로스 | 85% | 48% | $\varepsilon$ |
| TRADES | 명시적인 트레이드오프 | 87% | 46% | $\beta$ |
| MART | 어려운 샘플에 집중 | 86% | 49% | $\lambda$ |

CIFAR-10, $\varepsilon = 8/255$에서의 대략적인 값이다.

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
    일반 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 절의 방법을 적용한 결과는 기법에 따라 다르다. 적대적 트레인은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 바운드를 준다. 정확도와 강건성의 트레이드오프는 근본적인 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 떨어진다. 무작위 시드 3개의 평균과 표준편차로 보고하여라. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
섭동 공 안에서 클래스별 데이터의 서포트가 서로 겹친다고 볼 때, 모델이 용량을 키우지 않고서는 어떤 방어도 클린 데이터의 높은 정확도와 $\ell_\infty$ 섭동에 대한 높은 강건성을 함께 이룰 수 없음을 증명하여라.

</div>

??? success "풀이"
    두 클래스의 서포트가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 근본적인 트레이드오프다. 겹치는 서포트의 비중이 피할 수 없는 정확도 손실을 정한다. 자연 이미지 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 관찰되는 10~15%의 정확도 하락을 설명해 준다. 모델 용량을 키우면(더 넓은 네트워크) 복잡한 강건 결정 경계를 더 잘 그려 트레이드오프를 어느 정도 완화할 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 시스템(사기 탐지나 거래 신호 생성 등)에서 적대적 강건성이 어떻게 드러나는지 논하여라. 위협 모델이 비전과 어떻게 다른가?

</div>

??? success "풀이"
    금융에서 적대자는 탐지 시스템에 맞추어 전략을 바꾸는 교묘한 집단(사기꾼, 시장 조작자)이다. 비전과 다른 핵심은 이렇다. (1) 섭동할 수 있는 피처가 경제적으로 그럴듯한 범위로 제약된다(사기꾼이 자기 거래 이력을 통째로 바꿀 수는 없다). (2) 공격이 순차적이고 적응적이다(적대자가 시스템의 응답을 보고 고친다). (3) 거짓 양성과 거짓 음성의 비용이 비대칭이다(정상 거래를 막는 것과 사기를 놓치는 것). (4) $\ell_p$ 노름은 의미가 없고 도메인에 맞는 섭동 모델이 있어야 한다. 방어는 적응형 적대자에게도 강건해야 하므로, 탐지 기준이 알려지면 회피할 수 있는 탐지 기반 방법은 쓸모가 크게 줄어든다. $\square$

## 정리하며

MART의 오분류를 고려한 가중치는 가장 중요한 곳, 곧 모델이 어려워하는 샘플에서 강건성을 집중적으로 끌어올린다. 그래서 클래스가 불균형하거나 샘플마다 난이도 차이가 큰 데이터셋에 잘 듣는데, 이런 데이터는 금융 응용에서 흔하다.

**살펴볼 거리**

1. Wang, Y., Zou, D., Yi, J., Bailey, J., Ma, X., & Gu, Q. (2020). "Improving Adversarial Robustness Requires Revisiting Misclassified Examples." ICLR.
