# TRADES: 이론에 근거한 트레이드오프

**TRADES**(TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization, 트레이드오프에서 착안해 대리 로스를 최소화하는 적대적 방어. 장 등, 2019)는 강건 최적화의 목표를 명시적으로 분해해, 클린 정확도와 적대적 강건성 사이의 트레이드오프를 다룬다. 둘을 암묵적으로 저울질하는 표준 적대적 학습과 달리, TRADES는 이 트레이드오프를 직접 조절하는 파라메터 $\beta$를 준다.

---

## 1. 수학 기초

### 강건성 위험의 분해

TRADES의 핵심 통찰은 강건성 위험을 두 항으로 분해할 수 있다는 것이다.

$$
R_{\text{rob}}(f) \leq R_{\text{std}}(f) + R_{\text{boundary}}(f)
$$

여기서 $R_{\text{boundary}}$는 섭동이 올바르게 분류된 점을 결정 경계 너머로 밀어낼 확률을 잰다. 따라서 두 항을 따로 최적화하면 된다.

### TRADES 형태

TRADES는 다음을 최소화한다.

$$
\mathcal{L}_{\text{TRADES}} = \underbrace{\mathcal{L}_{\text{CE}}(f_\theta(\mathbf{x}), y)}_{\text{clean accuracy}} + \beta \cdot \underbrace{\text{KL}(f_\theta(\mathbf{x}) \| f_\theta(\mathbf{x}_{\text{adv}}))}_{\text{local smoothness}}
$$

여기서

- 첫째 항은 클린 입력에서 예측이 맞도록 한다(표준 크로스 엔트로피)
- 둘째 항은 KL 발산으로 클린 입력과 섭동된 입력의 예측이 **일관되도록** 이끈다
- $\beta > 0$이 트레이드오프를 조절한다. $\beta$가 클수록 클린 정확도를 희생하고 강건성을 앞세운다

### TRADES에서 적대적 샘플 만들기

적대적 샘플 $\mathbf{x}_{\text{adv}}$는 크로스 엔트로피가 아니라 KL 발산을 **최대화**하도록 만든다.

$$
\mathbf{x}_{\text{adv}} = \arg\max_{\|\boldsymbol{\delta}\| \leq \varepsilon} \text{KL}(f_\theta(\mathbf{x}) \| f_\theta(\mathbf{x} + \boldsymbol{\delta}))
$$

이는 KL 목표에 PGD를 적용해 근사적으로 푼다.

---

## 2. 표준 적대적 학습과 비교

| 항목 | 표준 적대적 학습 | TRADES |
|--------|-------------|--------|
| 학습 로스 | $\mathcal{L}_{\text{CE}}(f(\mathbf{x}_{\text{adv}}), y)$ | $\mathcal{L}_{\text{CE}}(f(\mathbf{x}), y) + \beta \cdot \text{KL}$ |
| 공격의 목표 | 크로스 엔트로피 최대화 | KL 발산 최대화 |
| 트레이드오프 조절 | 암묵적($\varepsilon$으로) | 명시적($\beta$로) |
| 클린 정확도 | 낮음 | 높음 |
| 강건 정확도 | 조금 높음 | 조금 낮음 |
| 이론적 근거 | min-max 최적화 | 대리 로스 분해 |

---

## 3. PyTorch로 구현하기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
from tqdm import tqdm

class TRADESTrainer:
    """
    TRADES: 이론에 닿는 맞바꿈.
    
    로스 = L_CE(f(x), y) + β · KL(f(x) || f(x_adv))
    
    Parameters
    ----------
    model : nn.Module
        익힐 모형
    beta : float
        맞바꿈 파라메터(흔히 1~6. 6이면 더 강건하다)
    epsilon : float
        섭동 예산
    alpha : float
        x_adv을 만들 PGD 스텝 크기
    num_iter : int
        x_adv을 만들 PGD 회귀
    """
    
    def __init__(
        self,
        model: nn.Module,
        beta: float = 6.0,
        epsilon: float = 8/255,
        alpha: float = 2/255,
        num_iter: int = 10,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.beta = beta
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
    
    def _generate_trades_adversary(
        self,
        x: torch.Tensor,
        logits_clean: torch.Tensor
    ) -> torch.Tensor:
        """
        KL 발산을 가장 크게 하여 맞서는 보기를 만든다.
        
        엇갈린 엔트로피를 가장 크게 하는 여느 맞서며 익히기와 달리,
        TRADES은 맑은 미루어 봄과 맞서는 미루어 봄의 KL 발산을 가장 크게 한다.
        """
        x_adv = x.clone().detach()
        x_adv += torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
        
        # 떼어 낸 맑은 소프트맥스(타깃 분포)
        p_clean = F.softmax(logits_clean.detach(), dim=1)
        
        for _ in range(self.num_iter):
            x_adv.requires_grad_(True)
            
            # KL 발산: KL(원본 || 적대적 샘플)
            loss_kl = F.kl_div(
                F.log_softmax(self.model(x_adv), dim=1),
                p_clean,
                reduction='batchmean'
            )
            
            self.model.zero_grad()
            loss_kl.backward()
            
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv.detach()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """TRADES 로스로 한 판 익힌다."""
        self.model.train()
        total_loss = 0
        total_natural = 0
        total_robust = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='TRADES Training')
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            # 맑은 앞으로 스텝
            logits_clean = self.model(x)
            loss_natural = F.cross_entropy(logits_clean, y)
            
            # 맞서는 보기를 만든다(KL을 가장 크게)
            x_adv = self._generate_trades_adversary(x, logits_clean)
            
            # TRADES 로스
            logits_adv = self.model(x_adv)
            loss_robust = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_clean.detach(), dim=1),
                reduction='batchmean'
            )
            
            loss = loss_natural + self.beta * loss_robust
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            total_natural += loss_natural.item() * len(y)
            total_robust += loss_robust.item() * len(y)
            correct += (logits_clean.argmax(1) == y).sum().item()
            total += len(y)
            
            pbar.set_postfix({
                'loss': f'{total_loss/total:.4f}',
                'nat': f'{total_natural/total:.4f}',
                'rob': f'{total_robust/total:.4f}'
            })
        
        return {
            'total_loss': total_loss / total,
            'natural_loss': total_natural / total,
            'robust_loss': total_robust / total,
            'clean_accuracy': correct / total
        }
```

---

## 4. 베타 고르기

파라메터 $\beta$로 정확도와 강건성의 트레이드오프를 직접 조절할 수 있다.

| $\beta$ | 클린 정확도 | 강건 정확도 | 쓰임 |
|---------|---------------|----------------|----------|
| 1.0 | 약 89% | 약 43% | 클린 정확도를 앞세움 |
| 3.0 | 약 88% | 약 45% | 균형 |
| 6.0 | 약 87% | 약 46% | 흔히 쓰는 기본값 |
| 10.0 | 약 85% | 약 47% | 강건성을 앞세움 |

CIFAR-10에서 $\varepsilon = 8/255$일 때의 대표적인 값이다.

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
    두 클래스의 서포트가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 근본적인 트레이드오프다. 겹치는 서포트의 비중이 피할 수 없는 정확도 손실을 정한다. 자연 이미지 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 관찰되는 10~15%의 정확도 하락을 설명해 준다. 모델 용량을 키우면(더 넓은 네트워크) 복잡한 강건 결정 경계를 더 잘 그려 트레이드오프를 어느 정도 완화할 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 시스템(사기 탐지나 거래 신호 생성 등)에서 적대적 강건성이 어떻게 드러나는지 논하여라. 위협 모델이 비전과 어떻게 다른가?

</div>

??? success "풀이"
    금융에서 적대자는 탐지 시스템에 맞추어 전략을 바꾸는 교묘한 집단(사기꾼, 시장 조작자)이다. 비전과 다른 핵심은 이렇다. (1) 섭동할 수 있는 피처가 경제적으로 그럴듯한 범위로 제약된다(사기꾼이 자기 거래 이력을 통째로 바꿀 수는 없다). (2) 공격이 순차적이고 적응적이다(적대자가 시스템의 응답을 보고 고친다). (3) 거짓 양성과 거짓 음성의 비용이 비대칭이다(정상 거래를 막는 것과 사기를 놓치는 것). (4) $\ell_p$ 노름은 의미가 없고 도메인에 맞는 섭동 모델이 있어야 한다. 방어는 적응형 적대자에게도 강건해야 하므로, 탐지 기준이 알려지면 회피할 수 있는 탐지 기반 방법은 쓸모가 크게 줄어든다. $\square$

## 정리하며

TRADES는 클린 정확도와 강건 정확도의 트레이드오프를 명시적이고 원리적으로 다루는 틀을 준다. 강건성 위험을 표준 위험 항과 경계 위험 항으로 분해한 덕에 $\beta$로 세밀하게 조절할 수 있어, 강건성과 함께 클린 정확도도 지켜야 할 때 즐겨 쓴다.

**살펴볼 거리**

1. Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., & Jordan, M. (2019). "Theoretically Principled Trade-off between Robustness and Accuracy." ICML.
